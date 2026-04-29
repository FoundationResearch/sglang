"""
Debug: isolate where the decode divergence happens.

Runs prefill + 1 decode step on both models, captures intermediate values
at each stage of the HSA attention layer, and prints where they diverge.
"""

import os
import sys
import types
import logging
import tempfile

# ---- veomni mock (same as compare script) ----
def _install_veomni_mock():
    import torch.nn as nn
    veomni = types.ModuleType("veomni")
    sys.modules["veomni"] = veomni
    for sub in ["distributed", "distributed.parallel_state", "distributed.sequence_parallel",
                "utils", "utils.logging", "utils.import_utils",
                "models", "models.module_utils", "models.loader"]:
        m = types.ModuleType(f"veomni.{sub}")
        sys.modules[f"veomni.{sub}"] = m
    class _FakePS:
        sp_enabled = False
    sys.modules["veomni.distributed.parallel_state"].get_parallel_state = lambda: _FakePS()
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
    sys.modules["veomni.models.module_utils"].GradientCheckpointingLayer = nn.Module
    class _FR:
        def register(self, n): return lambda f: f
    sys.modules["veomni.models.loader"].MODELING_REGISTRY = _FR()
    sys.modules["veomni.models.loader"].MODEL_CONFIG_REGISTRY = _FR()

_install_veomni_mock()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "InfiniteLongLM"))
for _n in ["models.SWANGPT", "models.DRT", "models.SWANNSA"]:
    sys.modules[_n] = types.ModuleType(_n)

import torch
import torch.distributed as dist
from models.FlashHSA.modeling_hsa_lmk import (
    HSAForCausalLM as OfficialModel,
    HierarchicalSparseAttention,
)
from models.FlashHSA.configuration_hsa import HSAConfig
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

# Patch official model
_orig_init = HierarchicalSparseAttention.__init__
def _patched_init(self, config, layer_idx):
    _orig_init(self, config, layer_idx)
    if not hasattr(self, "num_key_value_groups"):
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
HierarchicalSparseAttention.__init__ = _patched_init

import ops.topk_group as _topk_mod
import ops.hsa_fwd_bwd_group as _hsa_mod
_orig_topk = _topk_mod.online_topk_group
_orig_hsa_fwd = _hsa_mod.HSA_block_M_group
def _densify(t): return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)
_topk_mod.online_topk_group = lambda q, k, *a, **kw: _orig_topk(_densify(q), _densify(k), *a, **kw)
_hsa_mod.HSA_block_M_group = lambda q, k, v, *a, **kw: _orig_hsa_fwd(_densify(q), _densify(k), _densify(v), *a, **kw)

from sglang.srt.layers import dp_attention as dpa
dpa.get_attention_tp_size = lambda: 1


def _cmp(name, a, b):
    """Compare two tensors and print the result."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    diff = (a.float() - b.float()).abs()
    mx = diff.max().item()
    mn = diff.mean().item()
    status = "OK" if mx < 0.05 else ("CLOSE" if mx < 0.5 else "DIVERGED")
    print(f"  {name}: max_err={mx:.6f} mean_err={mn:.6f} [{status}]")


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    config_dict = dict(
        vocab_size=256, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4,
        head_dim=64, rms_norm_eps=1e-6, attention_bias=False,
        chunk_size=64, hsa_topk=2, hsa_mode="sparse",
        full_attn_interleave=1, hsa_heads=8, hsa_qk_ratio=4,
        enable_gate=False, use_sliding_window=True, sliding_window=64,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act="silu",
    )
    page_size = 64
    vocab_size = 256
    real_tokens = list(range(5, 5 + 65))

    # ============================================================
    # Official model: prefill + decode
    # ============================================================
    print("=" * 60)
    print("Setting up official model")
    official_config = HSAConfig(**config_dict)
    official_config._attn_implementation = "eager"
    official_model = OfficialModel(official_config).to(device=device, dtype=dtype)
    official_model.eval()
    for mod in official_model.modules():
        if isinstance(mod, HierarchicalSparseAttention):
            _otf = mod.topk_func; _ohf = mod.hsa_func
            mod.topk_func = lambda q, k, *a, _f=_otf, **kw: _f(_densify(q), _densify(k), *a, **kw)
            mod.hsa_func = lambda q, k, v, *a, _f=_ohf, **kw: _f(_densify(q), _densify(k), _densify(v), *a, **kw)

    # Hook the official HSA attention to capture intermediates during decode
    official_attn = None
    for mod in official_model.modules():
        if isinstance(mod, HierarchicalSparseAttention):
            official_attn = mod
            break

    official_captured = {}
    def _hook_official(mod, args, kwargs, output):
        hs = args[0] if args else kwargs.get("hidden_states")
        official_captured["hidden_in"] = hs.detach().clone()
        if isinstance(output, tuple):
            official_captured["attn_out"] = output[0].detach().clone()
        else:
            official_captured["attn_out"] = output.detach().clone()
    h_off = official_attn.register_forward_hook(_hook_official, with_kwargs=True)

    # Prefill
    input_ids_raw = torch.tensor([real_tokens], device=device, dtype=torch.long)
    input_ids_lmk = insert_special_tokens(input_ids_raw, vocab_size, page_size)
    position_ids = create_position_ids_with_landmarks(len(real_tokens), page_size, device)
    attention_mask = torch.ones((1, input_ids_lmk.shape[1]), dtype=torch.long, device=device)

    with torch.no_grad():
        off_prefill = official_model(input_ids=input_ids_lmk, position_ids=position_ids,
                                      attention_mask=attention_mask, use_cache=True)

    # Decode
    decode_token = 42
    decode_input = torch.tensor([[decode_token]], device=device, dtype=torch.long)
    L_prefill = input_ids_lmk.shape[1]
    decode_pos = L_prefill - (L_prefill // page_size)
    decode_position_ids = torch.tensor([[decode_pos]], device=device, dtype=torch.long)
    decode_mask = torch.ones((1, L_prefill + 1), dtype=torch.long, device=device)

    with torch.no_grad():
        off_decode = official_model(input_ids=decode_input, position_ids=decode_position_ids,
                                     attention_mask=decode_mask,
                                     past_key_values=off_prefill.past_key_values, use_cache=True)
    h_off.remove()

    off_decode_logits = off_decode.logits[:, -1, :vocab_size]
    off_hidden_in = official_captured["hidden_in"]  # [1, 1, hidden]
    off_attn_out = official_captured["attn_out"]    # [1, 1, hidden]

    print(f"Official decode: hidden_in={off_hidden_in.shape}, attn_out={off_attn_out.shape}")
    print(f"Official decode logits argmax: {off_decode_logits.argmax().item()}")

    # ============================================================
    # sglang model: prefill + decode
    # ============================================================
    print("\n" + "=" * 60)
    print("Setting up sglang model")

    from sglang.srt.configs.flash_hsa import FlashHSAConfig
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch, ForwardMode, compute_position, compute_decode_positions_landmark,
    )
    from sglang.srt.models.flash_hsa import (
        HSAForCausalLM as SglangModel,
        FlashHSAInnerXHierarchicalSparseAttention,
    )
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    if not dist.is_initialized():
        _, path = tempfile.mkstemp(prefix="sglang_dbg_", suffix=".tmp")
        dist.init_process_group(backend="gloo", init_method=f"file://{path}", rank=0, world_size=1)
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

    cfg = FlashHSAConfig(
        model_type="flash_hsa_innerx", architectures=["HSAForCausalLM"],
        vocab_size=256, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4,
        head_dim=64, rms_norm_eps=1e-6, attention_bias=False,
        chunk_size=64, hsa_topk=2, hsa_mode="sparse",
        full_attn_interleave=1, hsa_heads=8, hsa_qk_ratio=4, enable_gate=False,
        use_sliding_window_merging=True, sliding_window_merging_size=64,
        use_sliding_window_attention=False, sliding_window_attention_size=None,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act="silu",
    )
    sglang_model = SglangModel(cfg).to(device=device, dtype=dtype)
    sglang_model.eval()

    # Transfer weights
    off_sd = official_model.state_dict()
    sg_sd = sglang_model.state_dict()
    for name, param in sg_sd.items():
        if name in off_sd and off_sd[name].shape == param.shape:
            param.data.copy_(off_sd[name])
        elif "embed_tokens" in name or "lm_head" in name:
            if name in off_sd:
                mr = min(off_sd[name].shape[0], param.shape[0])
                param.data[:mr].copy_(off_sd[name][:mr])
        elif "gate_up_proj" in name:
            gn = name.replace("gate_up_proj", "gate_proj")
            un = name.replace("gate_up_proj", "up_proj")
            if gn in off_sd and un in off_sd:
                fused = torch.cat([off_sd[gn], off_sd[un]], dim=0)
                if fused.shape == param.shape: param.data.copy_(fused)

    for module in sglang_model.modules():
        if hasattr(module, "cos_sin_cache") and module.cos_sin_cache is not None:
            module.cos_sin_cache = module.cos_sin_cache.to(torch.float32)

    # Setup backend
    lmk_id = int(cfg.vocab_size)
    max_ctx = 256; max_tok = 1024
    model_runner = types.SimpleNamespace(
        device=device, page_size=page_size, sliding_window_size=None, model=sglang_model,
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, context_len=max_ctx, num_attention_heads=16,
            get_num_kv_heads=lambda tp: 4 // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend="hsa", speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id),
    )
    r2t = torch.zeros((1, max_ctx), dtype=torch.int32, device=device)
    model_runner.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    model_runner.token_to_kv_pool = MHATokenToKVPool(
        size=max_tok, page_size=page_size, dtype=dtype, head_num=4, head_dim=64,
        layer_num=1, device=device, enable_memory_saver=False, enable_alt_stream=False)
    model_runner.token_to_kv_pool_allocator = object()
    backend = HSAAttnBackend(model_runner)

    # Hook sglang attention
    sg_attn = None
    for mod in sglang_model.modules():
        if isinstance(mod, FlashHSAInnerXHierarchicalSparseAttention):
            sg_attn = mod; break

    sglang_captured = {}
    def _hook_sglang(mod, args, kwargs, output):
        sglang_captured["hidden_in"] = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
        if sglang_captured["hidden_in"] is not None:
            sglang_captured["hidden_in"] = sglang_captured["hidden_in"].detach().clone()
        sglang_captured["attn_out"] = output.detach().clone() if not isinstance(output, tuple) else output[0].detach().clone()
    h_sg = sg_attn.register_forward_hook(_hook_sglang, with_kwargs=True)

    # Prefill
    fill_ids = Req._hsa_insert_lmk_prompt(real_tokens, page_size=page_size, lmk_id=lmk_id)
    prefill_len = len(fill_ids)
    token_locs = torch.arange(0, max_ctx, dtype=torch.int32, device=device)
    r2t[0, :prefill_len] = token_locs[:prefill_len]

    ext_prefix = torch.tensor([0], device=device, dtype=torch.int32)
    ext_seq = torch.tensor([prefill_len], device=device, dtype=torch.int32)
    pos_sg, ext_start = compute_position("hsa", ext_prefix, ext_seq, prefill_len,
                                          page_size=page_size, enable_landmark_positions=True)
    fb_ext = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.tensor(fill_ids, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([prefill_len], device=device, dtype=torch.int32),
        out_cache_loc=token_locs[:prefill_len].to(torch.int64),
        seq_lens_sum=prefill_len, seq_lens_cpu=torch.tensor([prefill_len], device="cpu", dtype=torch.int32),
        positions=pos_sg, extend_prefix_lens=ext_prefix, extend_seq_lens=ext_seq,
        extend_start_loc=ext_start, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[prefill_len],
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool, attn_backend=backend)
    backend.init_forward_metadata(fb_ext)
    with torch.no_grad():
        sglang_model.model(fb_ext.input_ids, fb_ext.positions, fb_ext)

    # Decode
    decode_seq_len = prefill_len + 1
    r2t[0, prefill_len] = token_locs[prefill_len]
    dec_pos = compute_decode_positions_landmark(
        torch.tensor([decode_seq_len], device=device, dtype=torch.int32), page_size=page_size)
    fb_dec = ForwardBatch(
        forward_mode=ForwardMode.DECODE, batch_size=1,
        input_ids=torch.tensor([decode_token], device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([decode_seq_len], device=device, dtype=torch.int32),
        out_cache_loc=token_locs[prefill_len:prefill_len+1].to(torch.int64),
        seq_lens_sum=decode_seq_len,
        seq_lens_cpu=torch.tensor([decode_seq_len], device="cpu", dtype=torch.int32),
        positions=dec_pos,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool, attn_backend=backend)

    backend.init_forward_metadata(fb_dec)
    with torch.no_grad():
        sg_dec = sglang_model.model(fb_dec.input_ids, fb_dec.positions, fb_dec)
    h_sg.remove()

    sg_hidden_in = sglang_captured.get("hidden_in")
    sg_attn_out = sglang_captured.get("attn_out")

    # ============================================================
    # Compare intermediates
    # ============================================================
    print("\n" + "=" * 60)
    print("DECODE INTERMEDIATE COMPARISON")
    print("=" * 60)

    if sg_hidden_in is not None and off_hidden_in is not None:
        # sglang hidden_in is [T] not [B,T,D], reshape
        if sg_hidden_in.dim() == 2:
            sg_hidden_in = sg_hidden_in.unsqueeze(0)
        if off_hidden_in.dim() == 3 and sg_hidden_in.dim() == 3:
            _cmp("hidden_states input to attention", off_hidden_in, sg_hidden_in)
        else:
            print(f"  hidden_in shapes: official={off_hidden_in.shape}, sglang={sg_hidden_in.shape}")

    if sg_attn_out is not None and off_attn_out is not None:
        if sg_attn_out.dim() == 2:
            sg_attn_out = sg_attn_out.unsqueeze(0)
        if off_attn_out.dim() == 3 and sg_attn_out.dim() == 3:
            _cmp("attention output", off_attn_out, sg_attn_out)
        else:
            print(f"  attn_out shapes: official={off_attn_out.shape}, sglang={sg_attn_out.shape}")

    # Also compare the pre-norm hidden states (embedding output)
    # and post-attn logits
    if isinstance(sg_dec, tuple):
        sg_dec_h = sg_dec[0]
    else:
        sg_dec_h = sg_dec
    sg_dec_h = sglang_model.model.norm(sg_dec_h)
    lm_w = sglang_model.lm_head.weight
    sg_logits = (sg_dec_h @ lm_w.t())[:, :vocab_size]

    print(f"\n  Official decode logits argmax: {off_decode_logits.argmax().item()}")
    print(f"  sglang  decode logits argmax: {sg_logits.argmax().item()}")
    _cmp("decode logits", off_decode_logits.unsqueeze(0), sg_logits.unsqueeze(0))


if __name__ == "__main__":
    main()
