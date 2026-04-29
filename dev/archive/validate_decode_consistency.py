"""
Validate sglang HSA decode by self-consistency:
  prefill(N+1 tokens)[-1] should equal prefill(N tokens) + decode(1 token)

This verifies KV cache is correctly populated during prefill and read during decode,
independent of the official model.
"""
import os, sys, types, logging, tempfile

# veomni mock (must be before any sglang import)
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

# Import official model BEFORE sglang (to avoid utils module shadowing)
from models.FlashHSA.modeling_hsa_lmk import HSAForCausalLM as OfficialModel, HierarchicalSparseAttention as _HSA_Attn
from models.FlashHSA.configuration_hsa import HSAConfig as OfficialConfig
_oi = _HSA_Attn.__init__
def _pi_global(s,c,l): _oi(s,c,l); s.num_key_value_groups = c.num_attention_heads // c.num_key_value_heads if not hasattr(s,'num_key_value_groups') else None
_HSA_Attn.__init__ = _pi_global

from sglang.srt.layers import dp_attention as dpa
dpa.get_attention_tp_size = lambda: 1
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.configs.flash_hsa import FlashHSAConfig
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch, ForwardMode, compute_position, compute_decode_positions_landmark,
)
from sglang.srt.models.flash_hsa import HSAForCausalLM
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def init_infra():
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix="val_", suffix=".t")
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


def make_env(model, cfg, device, dtype):
    """Create fresh backend + pool for one test run."""
    PS = cfg.chunk_size
    lmk_id = int(cfg.vocab_size)
    mc = 1024
    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(size=2048, page_size=PS, dtype=dtype, head_num=int(cfg.num_key_value_heads),
                             head_dim=int(cfg.head_dim), layer_num=int(cfg.num_hidden_layers),
                             device=device, enable_memory_saver=False, enable_alt_stream=False)
    mr = types.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=model,
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, context_len=mc,
            num_attention_heads=int(cfg.num_attention_heads),
            get_num_kv_heads=lambda tp: int(cfg.num_key_value_heads) // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend="hsa", speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id))
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool
    mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)
    return be, mr, pool, r2t


def run_prefill(model, cfg, real_tokens, device, dtype):
    """Prefill and return last-token logits."""
    be, mr, pool, r2t = make_env(model, cfg, device, dtype)
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); VS = int(cfg.vocab_size); mc = 1024
    fi = Req._hsa_insert_lmk_prompt(real_tokens, page_size=PS, lmk_id=lmk_id)
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

    # Hook RadixAttention to check K/V norms entering the backend
    from sglang.srt.layers.radix_attention import RadixAttention as _RA
    _kv_norms = {}
    def _hook_ra(mod, args, kwargs, out):
        _kv_norms["k_norm"] = args[1].float().norm().item()
        _kv_norms["v_norm"] = args[2].float().norm().item()
    _handles = []
    for m in model.modules():
        if isinstance(m, _RA):
            _handles.append(m.register_forward_hook(_hook_ra, with_kwargs=True))

    with torch.no_grad():
        h = model.model(fb.input_ids, fb.positions, fb)

    for _h in _handles: _h.remove()
    if _kv_norms:
        k_in = _kv_norms.get("k_norm", -1)
        v_in = _kv_norms.get("v_norm", -1)
        kb_out = pool.get_key_buffer(0)[:pl].float().norm().item()
        if k_in > 0.1 and kb_out < 0.01:
            print(f"    WARNING: K entering attn has norm={k_in:.2f} but KV cache is {kb_out:.4f}")
            print(f"    This means set_kv_buffer is not writing to the pool correctly")

    if isinstance(h, tuple): h = h[0]
    h = model.model.norm(h)
    logits = (h @ model.lm_head.weight.t())[:, :VS].float()
    return logits, be, mr, pool


def run_decode(model, cfg, be, mr, pool, prefill_real_tokens, decode_token_id, device, dtype):
    """Run one decode step after prefill, return logits."""
    PS = cfg.chunk_size; lmk_id = int(cfg.vocab_size); VS = int(cfg.vocab_size); mc = 1024
    fi = Req._hsa_insert_lmk_prompt(prefill_real_tokens, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
    dsl = pl + 1
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    mr.req_to_token_pool.req_to_token[0, pl] = tl[pl]
    dp = compute_decode_positions_landmark(
        torch.tensor([dsl], device=device, dtype=torch.int32), page_size=PS)
    fbd = ForwardBatch(
        forward_mode=ForwardMode.DECODE, batch_size=1,
        input_ids=torch.tensor([decode_token_id], device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([dsl], device=device, dtype=torch.int32),
        out_cache_loc=tl[pl:pl+1].to(torch.int64), seq_lens_sum=dsl,
        seq_lens_cpu=torch.tensor([dsl], device="cpu", dtype=torch.int32),
        positions=dp, req_to_token_pool=mr.req_to_token_pool,
        token_to_kv_pool=pool, attn_backend=be)
    be.init_forward_metadata(fbd)
    # Verify ForwardBatch has correct pool
    _fb_pool = getattr(fbd, "token_to_kv_pool", "MISSING")
    if _fb_pool != "MISSING" and _fb_pool is not None:
        _fb_k = _fb_pool.get_key_buffer(0)
        _fb_k_norm = _fb_k[:pl].float().norm().item()
        if _fb_k_norm < 0.01:
            print(f"    BUG: ForwardBatch pool has empty KV (norm={_fb_k_norm:.4f})")
    else:
        print(f"    BUG: ForwardBatch has no token_to_kv_pool!")
    with torch.no_grad():
        h = model.model(fbd.input_ids, fbd.positions, fbd)
    if isinstance(h, tuple): h = h[0]
    h = model.model.norm(h)
    return (h @ model.lm_head.weight.t())[:, :VS].float()


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    init_infra()

    torch.manual_seed(42)
    cfg = FlashHSAConfig(
        model_type="flash_hsa_innerx", architectures=["HSAForCausalLM"],
        vocab_size=256, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4,
        head_dim=64, rms_norm_eps=1e-6, chunk_size=64, hsa_topk=2, hsa_mode="sparse",
        full_attn_interleave=1, hsa_heads=8, hsa_qk_ratio=4,
        use_sliding_window_merging=True, sliding_window_merging_size=64,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act="silu")
    # Create official model first (proper weight init), then transfer to sglang model
    oc = OfficialConfig(**{k: v for k, v in cfg.to_dict().items() if k not in ('_name_or_path','transformers_version','architectures','model_type','sliding_window_merging_size','use_sliding_window_merging','sliding_window_attention_size','use_sliding_window_attention','hsa_selection_strategy')})
    oc._attn_implementation = "eager"
    official_model = OfficialModel(oc).to(device=device, dtype=dtype)
    official_model.eval()

    model = HSAForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()
    # Transfer weights from official model (proper init) to sglang model
    off_sd = official_model.state_dict()
    sg_sd = model.state_dict()
    loaded = 0
    for name, param in sg_sd.items():
        if name in off_sd and off_sd[name].shape == param.shape:
            param.data.copy_(off_sd[name]); loaded += 1
        elif ("embed_tokens" in name or "lm_head" in name) and name in off_sd:
            mr2 = min(off_sd[name].shape[0], param.shape[0])
            param.data[:mr2].copy_(off_sd[name][:mr2]); loaded += 1
        elif "gate_up_proj" in name:
            gn = name.replace("gate_up_proj","gate_proj"); un = name.replace("gate_up_proj","up_proj")
            if gn in off_sd and un in off_sd:
                f = torch.cat([off_sd[gn], off_sd[un]], dim=0)
                if f.shape == param.shape: param.data.copy_(f); loaded += 1
    print(f"Transferred {loaded}/{len(sg_sd)} weights from official model")
    del official_model  # free memory

    for m in model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    PS = cfg.chunk_size
    decode_token = 42

    print()
    print("=" * 70)
    print("SGLANG DECODE SELF-CONSISTENCY TEST")
    print("prefill(N+1)[-1] vs prefill(N) + decode(1)")
    print("=" * 70)
    print(f"Config: hidden=1024, heads=16, kv_heads=4, chunk_size={PS}, topk=2")
    print()

    test_cases = [
        (5,  "5 tokens, 0 pages"),
        (10, "10 tokens, 0 pages"),
        (30, "30 tokens, 0 pages"),
        (62, "62 tokens, 0 pages (near page boundary)"),
        (63, "63 tokens, 1 page (exactly at boundary)"),
        (65, "65 tokens, 1 page + 2 extra"),
        (100,"100 tokens, 1 page + partial"),
        (126,"126 tokens, 2 pages"),
        (130,"130 tokens, 2 pages + partial"),
    ]

    results = []
    for n_tokens, desc in test_cases:
        real_base = list(range(5, 5 + n_tokens))
        real_full = real_base + [decode_token]

        # Method A: prefill all N+1 tokens, take last logits
        logits_full, _, _, _ = run_prefill(model, cfg, real_full, device, dtype)
        last_prefill = logits_full[0, -1]  # [V]
        pass

        # Method B: prefill N tokens, then decode 1
        _, be, mr, pool = run_prefill(model, cfg, real_base, device, dtype)

        # Quick sanity: is KV populated?
        kb = pool.get_key_buffer(0)
        fi = Req._hsa_insert_lmk_prompt(real_base, page_size=PS, lmk_id=int(cfg.vocab_size))
        kv_norm = kb[:len(fi)].float().norm().item()

        last_decode = run_decode(model, cfg, be, mr, pool, real_base, decode_token, device, dtype)
        last_decode = last_decode.squeeze(0)  # [V]

        lp = last_prefill.view(-1)
        ld = last_decode.view(-1)
        diff = (lp - ld).abs()
        mx = diff.max().item()
        mn = diff.mean().item()
        # Use cosine similarity (robust to scale)
        cos_sim = torch.nn.functional.cosine_similarity(lp.unsqueeze(0), ld.unsqueeze(0)).item()
        # Relative error
        max_mag = max(lp.abs().max().item(), ld.abs().max().item(), 1e-8)
        rel_err = mx / max_mag
        am_pf = lp.argmax().item()
        am_dc = ld.argmax().item()
        match = am_pf == am_dc
        # Top-5 agreement
        k_top = min(5, lp.shape[0])
        top5_pf = set(lp.topk(k_top).indices.tolist())
        top5_dc = set(ld.topk(k_top).indices.tolist())
        top5_overlap = len(top5_pf & top5_dc)

        status = "PASS" if cos_sim > 0.999 else ("CLOSE" if cos_sim > 0.99 else "FAIL")
        results.append((n_tokens, desc, cos_sim, rel_err, am_pf, am_dc, match, top5_overlap, kv_norm, status))
        print(f"  N={n_tokens:3d} ({desc})")
        print(f"    KV norm: {kv_norm:.2f}  cos_sim={cos_sim:.6f}  rel_err={rel_err:.6f}")
        print(f"    argmax: prefill={am_pf} decode={am_dc} {'OK' if match else 'MISMATCH'}  top5_overlap={top5_overlap}/5  [{status}]")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'N':>5} {'Pages':>5} {'Cos Sim':>10} {'Rel Err':>10} {'Argmax':>8} {'Top5':>5} {'Status':>6}")
    print("-" * 55)
    for n, desc, cs, re, ap, ad, match, t5, kvn, st in results:
        pages = n // (PS - 1)
        am = "OK" if match else "MISS"
        print(f"{n:>5} {pages:>5} {cs:>10.6f} {re:>10.6f} {am:>8} {t5:>5} {st:>6}")


if __name__ == "__main__":
    main()
