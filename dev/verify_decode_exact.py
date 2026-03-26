"""
Verify sglang decode correctness by comparing against official model's
full-sequence recomputation.

Method:
  1. Official: prefill(N+1 tokens) → take logits at position N (last token)
  2. sglang:   prefill(N tokens) + decode(1 token) → take decode logits
  Both should produce the same output for the last token.

This avoids the KV cache issue entirely — the official model always
gets the full sequence, sglang uses its incremental decode.
"""
import os, sys, types, logging, tempfile

# veomni mock
for sub in ["veomni","veomni.distributed","veomni.distributed.parallel_state",
            "veomni.distributed.sequence_parallel","veomni.utils","veomni.utils.logging",
            "veomni.utils.import_utils","veomni.models","veomni.models.module_utils","veomni.models.loader"]:
    sys.modules[sub] = types.ModuleType(sub)
class _PS: sp_enabled = False
sys.modules["veomni.distributed.parallel_state"].get_parallel_state = lambda: _PS()
sys.modules["veomni.distributed.sequence_parallel"].slice_position_embedding = lambda x, **kw: x
def _gl(n=None):
    l = logging.getLogger(n)
    if not hasattr(l, "info_rank0"): l.info_rank0 = l.info
    if not hasattr(l, "warning_rank0"): l.warning_rank0 = l.warning
    return l
sys.modules["veomni.utils.logging"].get_logger = _gl
sys.modules["veomni.utils.import_utils"].is_liger_kernel_available = lambda: True
sys.modules["veomni.utils.import_utils"].is_torch_npu_available = lambda: False
sys.modules["veomni.utils.import_utils"].is_transformers_version_greater_or_equal_to = lambda v: True
import torch.nn as nn
sys.modules["veomni.models.module_utils"].GradientCheckpointingLayer = nn.Module
class _FR:
    def register(self, n): return lambda f: f
sys.modules["veomni.models.loader"].MODELING_REGISTRY = _FR()
sys.modules["veomni.models.loader"].MODEL_CONFIG_REGISTRY = _FR()
for _n in ["models.SWANGPT", "models.DRT", "models.SWANNSA"]:
    sys.modules[_n] = types.ModuleType(_n)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "InfiniteLongLM"))

import torch
import torch.distributed as dist

# Import official model FIRST
from models.FlashHSA.modeling_hsa_lmk import HSAForCausalLM as OfficialModel, HierarchicalSparseAttention
from models.FlashHSA.configuration_hsa import HSAConfig as OfficialConfig
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

_oi = HierarchicalSparseAttention.__init__
def _pi(s,c,l): _oi(s,c,l); s.num_key_value_groups = c.num_attention_heads // c.num_key_value_heads if not hasattr(s,'num_key_value_groups') else None
HierarchicalSparseAttention.__init__ = _pi

# tilelang stride fix
import ops.topk_group as _tm, ops.hsa_fwd_bwd_group as _hm
_ot = _tm.online_topk_group; _oh = _hm.HSA_block_M_group
def _d(t): return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)
_tm.online_topk_group = lambda q, k, *a, **kw: _ot(_d(q), _d(k), *a, **kw)
_hm.HSA_block_M_group = lambda q, k, v, *a, **kw: _oh(_d(q), _d(k), _d(v), *a, **kw)

# sglang imports
from sglang.srt.layers import dp_attention as dpa
dpa.get_attention_tp_size = lambda: 1
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.configs.flash_hsa import FlashHSAConfig
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch, ForwardMode, compute_position, compute_decode_positions_landmark,
)
from sglang.srt.models.flash_hsa import HSAForCausalLM as SglangModel
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def init_sglang():
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix="vd_", suffix=".t")
        dist.init_process_group(backend="gloo", init_method=f"file://{p}", rank=0, world_size=1)
    if not ps.model_parallel_is_initialized():
        ps._WORLD = ps.init_world_group(ranks=[0], local_rank=0, backend="gloo")
        ps._TP = ps.init_model_parallel_group(group_ranks=[[0]], local_rank=0, backend="gloo",
            use_custom_allreduce=False, use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False, group_name="tp")
        ps._PP = ps.init_model_parallel_group(group_ranks=[[0]], local_rank=0, backend="gloo",
            use_custom_allreduce=False, use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False, group_name="pp")
    if getattr(dpa, "_ATTN_TP_RANK", None) is None:
        dpa._ATTN_TP_RANK = 0; dpa._ATTN_TP_SIZE = 1
        dpa._ATTN_DP_RANK = 0; dpa._ATTN_DP_SIZE = 1
        dpa._LOCAL_ATTN_DP_RANK = 0; dpa._LOCAL_ATTN_DP_SIZE = 1
        dpa._ENABLE_DP_ATTENTION_FLAG = False; dpa._ATTN_TP_GROUP = ps.get_tp_group()
    try:
        sa = ServerArgs(model_path="dummy"); sa.attention_backend = "hsa"; sa.enable_dp_lm_head = False
        set_global_server_args_for_scheduler(sa)
    except: pass


def transfer_weights(src, dst):
    ssd = src.state_dict(); dsd = dst.state_dict(); n = 0
    for name, param in dsd.items():
        if name in ssd and ssd[name].shape == param.shape:
            param.data.copy_(ssd[name]); n += 1
        elif ("embed_tokens" in name or "lm_head" in name) and name in ssd:
            mr = min(ssd[name].shape[0], param.shape[0])
            param.data[:mr].copy_(ssd[name][:mr]); n += 1
        elif "gate_up_proj" in name:
            gn = name.replace("gate_up_proj", "gate_proj"); un = name.replace("gate_up_proj", "up_proj")
            if gn in ssd and un in ssd:
                f = torch.cat([ssd[gn], ssd[un]], dim=0)
                if f.shape == param.shape: param.data.copy_(f); n += 1
    return n, len(dsd)


def official_prefill(model, real_tokens, page_size, vocab_size, device):
    """Official model: full-sequence prefill, return ALL logits."""
    ids_raw = torch.tensor([real_tokens], device=device, dtype=torch.long)
    ids_lmk = insert_special_tokens(ids_raw, vocab_size, page_size)
    pos_ids = create_position_ids_with_landmarks(len(real_tokens), page_size, device)
    mask = torch.ones((1, ids_lmk.shape[1]), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids_lmk, position_ids=pos_ids, attention_mask=mask, use_cache=False)
    return out.logits[:, :, :vocab_size].float()


def sglang_prefill_and_decode(model, cfg, real_tokens_base, decode_token, device, dtype):
    """sglang: prefill N tokens, then decode 1 token. Return decode logits."""
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); VS = int(cfg.vocab_size); mc = 1024

    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(size=2048, page_size=PS, dtype=dtype, head_num=int(cfg.num_key_value_heads),
                             head_dim=int(cfg.head_dim), layer_num=int(cfg.num_hidden_layers),
                             device=device, enable_memory_saver=False, enable_alt_stream=False)
    mr = types.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=model,
        model_config=types.SimpleNamespace(is_encoder_decoder=False, context_len=mc,
                                            num_attention_heads=int(cfg.num_attention_heads),
                                            get_num_kv_heads=lambda tp: int(cfg.num_key_value_heads) // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend="hsa", speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id))
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool; mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)

    # Prefill
    fi = Req._hsa_insert_lmk_prompt(real_tokens_base, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    r2t[0, :pl] = tl[:pl]
    ep = torch.tensor([0], device=device, dtype=torch.int32)
    es = torch.tensor([pl], device=device, dtype=torch.int32)
    pos, esl = compute_position("hsa", ep, es, pl, page_size=PS, enable_landmark_positions=True)
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.tensor(fi, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([pl], device=device, dtype=torch.int32),
        out_cache_loc=tl[:pl].to(torch.int64), seq_lens_sum=pl,
        seq_lens_cpu=torch.tensor([pl], device="cpu", dtype=torch.int32),
        positions=pos, extend_prefix_lens=ep, extend_seq_lens=es,
        extend_start_loc=esl, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[pl],
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool, attn_backend=be)
    be.init_forward_metadata(fb)
    with torch.no_grad():
        model.model(fb.input_ids, fb.positions, fb)

    # Decode
    dsl = pl + 1
    r2t[0, pl] = tl[pl]
    dp = compute_decode_positions_landmark(
        torch.tensor([dsl], device=device, dtype=torch.int32), page_size=PS)
    fbd = ForwardBatch(
        forward_mode=ForwardMode.DECODE, batch_size=1,
        input_ids=torch.tensor([decode_token], device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([dsl], device=device, dtype=torch.int32),
        out_cache_loc=tl[pl:pl+1].to(torch.int64), seq_lens_sum=dsl,
        seq_lens_cpu=torch.tensor([dsl], device="cpu", dtype=torch.int32),
        positions=dp, req_to_token_pool=mr.req_to_token_pool,
        token_to_kv_pool=pool, attn_backend=be)
    be.init_forward_metadata(fbd)
    with torch.no_grad():
        h = model.model(fbd.input_ids, fbd.positions, fbd)
    if isinstance(h, tuple): h = h[0]
    h = model.model.norm(h)
    lm_w = model.lm_head.weight
    return (h @ lm_w.t())[:, :VS].float()


def main():
    device = torch.device("cuda"); dtype = torch.bfloat16
    init_sglang()
    torch.manual_seed(42)

    config_dict = dict(
        vocab_size=256, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4,
        head_dim=64, rms_norm_eps=1e-6, chunk_size=64, hsa_topk=2, hsa_mode="sparse",
        full_attn_interleave=1, hsa_heads=8, hsa_qk_ratio=4,
        use_sliding_window=True, sliding_window=64,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act="silu",
    )
    PS = 64; VS = 256

    # Create official model
    oc = OfficialConfig(**config_dict); oc._attn_implementation = "eager"
    off_model = OfficialModel(oc).to(device=device, dtype=dtype); off_model.eval()
    for mod in off_model.modules():
        if isinstance(mod, HierarchicalSparseAttention):
            _tf = mod.topk_func; _hf = mod.hsa_func
            mod.topk_func = lambda q, k, *a, _f=_tf, **kw: _f(_d(q), _d(k), *a, **kw)
            mod.hsa_func = lambda q, k, v, *a, _f=_hf, **kw: _f(_d(q), _d(k), _d(v), *a, **kw)

    # Create sglang model and transfer weights
    cfg = FlashHSAConfig(
        model_type="flash_hsa_innerx", architectures=["HSAForCausalLM"],
        vocab_size=256, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4,
        head_dim=64, rms_norm_eps=1e-6, chunk_size=64, hsa_topk=2, hsa_mode="sparse",
        full_attn_interleave=1, hsa_heads=8, hsa_qk_ratio=4,
        use_sliding_window_merging=True, sliding_window_merging_size=64,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act="silu")
    sg_model = SglangModel(cfg).to(device=device, dtype=dtype); sg_model.eval()
    loaded, total = transfer_weights(off_model, sg_model)
    print(f"Weights: {loaded}/{total}")
    for m in sg_model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    print("=" * 70)
    print("DECODE VERIFICATION")
    print("Official prefill(N+1)[-1] vs sglang prefill(N) + decode(1)")
    print("=" * 70)

    decode_token = 42
    test_cases = [
        (10, "10 tokens, 0 pages"),
        (61, "61 tokens, 0 pages"),
        (62, "62 tokens, 0 pages"),
        (63, "63 tokens, 1 page"),
        (65, "65 tokens, 1 page + 2"),
        (126, "126 tokens, 2 pages"),
        (130, "130 tokens, 2 pages + 4"),
    ]

    for n_tokens, desc in test_cases:
        real_base = list(range(5, 5 + n_tokens))
        real_full = real_base + [decode_token]

        # Official: prefill full N+1 tokens.
        # Take logits at the position of token `decode_token`, NOT the last position.
        # LMK insertion may add LMK AFTER the decode token, making it non-last.
        off_logits = official_prefill(off_model, real_full, PS, VS, device)
        # Find the engine position of the decode token.
        fi_full = insert_special_tokens(torch.tensor([real_full]), VS, PS).squeeze().tolist()
        # The decode token position in the engine sequence:
        # It's at the same position as in sglang's base prefill.
        fi_base = Req._hsa_insert_lmk_prompt(real_base, page_size=PS, lmk_id=VS)
        decode_engine_pos = len(fi_base)  # decode token goes at this index
        # In the official full sequence, the same token is at the same index
        # (all tokens before it are identical).
        off_last = off_logits[0, decode_engine_pos]  # [V]

        # sglang: prefill N tokens, decode 1
        sg_last = sglang_prefill_and_decode(sg_model, cfg, real_base, decode_token, device, dtype)
        sg_last = sg_last.view(-1)  # [V]

        diff = (off_last - sg_last).abs()
        mx = diff.max().item()
        mn = diff.mean().item()
        cos = torch.nn.functional.cosine_similarity(off_last.unsqueeze(0), sg_last.unsqueeze(0)).item()
        am_off = off_last.argmax().item()
        am_sg = sg_last.argmax().item()
        match = am_off == am_sg
        status = "PASS" if mx < 0.1 else ("CLOSE" if mx < 1.0 else "FAIL")

        print(f"  N={n_tokens:3d} ({desc})")
        print(f"    max_err={mx:.4f} mean_err={mn:.4f} cos_sim={cos:.6f}")
        print(f"    off_argmax={am_off} sg_argmax={am_sg} {'OK' if match else 'MISMATCH'} [{status}]")

        # Also check sglang self-consistency: sg_prefill(N+1)[-1] vs sg_prefill(N)+decode(1)
        # Use sglang_prefill_and_decode with real_full as base and a dummy decode to get all-prefill logits
        # Actually simpler: just call official_prefill with sg_model (same LMK insertion logic)
        # No — official_prefill uses insert_special_tokens which is the official function.
        # But sglang uses Req._hsa_insert_lmk_prompt which should produce the same result.
        # Let's just do sglang prefill of real_full directly:
        sg_full_logits_obj = sglang_prefill_and_decode.__code__  # can't easily reuse, skip for now
        # Instead, use official_prefill with the sglang model directly (they share the same interface for prefill)
        # The official_prefill function uses insert_special_tokens + create_position_ids_with_landmarks
        # and just calls model(input_ids, position_ids, attention_mask) — works for any model.
        # But sg_model.forward expects (input_ids, positions, forward_batch), not HF-style.
        # Skip self-check for now — the official vs sglang comparison is the meaningful test.
        pass


if __name__ == "__main__":
    main()
