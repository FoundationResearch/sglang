"""
Comprehensive verification: official vs sglang HSA at multiple prompt lengths.

Tests prefill and decode alignment across different scenarios:
  - Short prompt (no completed pages, SWA-only)
  - 1 completed page
  - 2 completed pages
  - 3+ completed pages

Also captures per-head-group (SWA vs HSA) outputs to isolate divergence.

Usage:
  CUDA_VISIBLE_DEVICES=4 python dev/verify_alignment_multi_length.py
"""

import os
import sys
import types
import logging
import tempfile
import json

# ---- veomni mock ----
def _install_veomni_mock():
    import torch.nn as nn
    for sub in ["veomni", "veomni.distributed", "veomni.distributed.parallel_state",
                "veomni.distributed.sequence_parallel", "veomni.utils", "veomni.utils.logging",
                "veomni.utils.import_utils", "veomni.models", "veomni.models.module_utils",
                "veomni.models.loader"]:
        sys.modules[sub] = types.ModuleType(sub)
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
    HSAForCausalLM as OfficialModel, HierarchicalSparseAttention,
)
from models.FlashHSA.configuration_hsa import HSAConfig
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

# Patches
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


def setup_sglang_infra():
    """One-time sglang infrastructure setup."""
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix="verify_", suffix=".tmp")
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


def transfer_weights(official_model, sglang_model):
    off_sd = official_model.state_dict()
    sg_sd = sglang_model.state_dict()
    loaded = 0
    for name, param in sg_sd.items():
        if name in off_sd:
            src = off_sd[name]
            if src.shape == param.shape:
                param.data.copy_(src); loaded += 1
            elif "embed_tokens" in name or "lm_head" in name:
                mr = min(src.shape[0], param.shape[0])
                param.data[:mr].copy_(src[:mr]); loaded += 1
        elif "gate_up_proj" in name:
            gn = name.replace("gate_up_proj", "gate_proj")
            un = name.replace("gate_up_proj", "up_proj")
            if gn in off_sd and un in off_sd:
                fused = torch.cat([off_sd[gn], off_sd[un]], dim=0)
                if fused.shape == param.shape: param.data.copy_(fused); loaded += 1
    return loaded, len(sg_sd)


def run_comparison(num_real_tokens, config_dict, official_model, sglang_model, backend,
                   model_runner, results):
    """Run prefill + decode comparison for a given prompt length."""
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch, ForwardMode, compute_position, compute_decode_positions_landmark,
    )
    from sglang.srt.managers.schedule_batch import Req

    device = torch.device("cuda")
    dtype = torch.bfloat16
    page_size = config_dict["chunk_size"]
    lmk_id = config_dict["vocab_size"]
    vocab_size = config_dict["vocab_size"]

    real_tokens = list(range(5, 5 + num_real_tokens))

    # ---- Official prefill ----
    input_ids_raw = torch.tensor([real_tokens], device=device, dtype=torch.long)
    input_ids_lmk = insert_special_tokens(input_ids_raw, vocab_size, page_size)
    position_ids = create_position_ids_with_landmarks(num_real_tokens, page_size, device)
    attention_mask = torch.ones((1, input_ids_lmk.shape[1]), dtype=torch.long, device=device)
    L_lmk = input_ids_lmk.shape[1]

    with torch.no_grad():
        off_out = official_model(input_ids=input_ids_lmk, position_ids=position_ids,
                                  attention_mask=attention_mask, use_cache=True)
    off_logits = off_out.logits[:, :, :vocab_size].float()

    # ---- sglang prefill ----
    fill_ids = Req._hsa_insert_lmk_prompt(real_tokens, page_size=page_size, lmk_id=lmk_id)
    prefill_len = len(fill_ids)
    assert prefill_len == L_lmk

    max_ctx = model_runner.req_to_token_pool.req_to_token.shape[1]
    token_locs = torch.arange(0, max_ctx, dtype=torch.int32, device=device)
    model_runner.req_to_token_pool.req_to_token[0, :] = 0  # reset
    model_runner.req_to_token_pool.req_to_token[0, :prefill_len] = token_locs[:prefill_len]

    ext_prefix = torch.tensor([0], device=device, dtype=torch.int32)
    ext_seq = torch.tensor([prefill_len], device=device, dtype=torch.int32)
    pos_sg, ext_start = compute_position("hsa", ext_prefix, ext_seq, prefill_len,
                                          page_size=page_size, enable_landmark_positions=True)

    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.tensor(fill_ids, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([prefill_len], device=device, dtype=torch.int32),
        out_cache_loc=token_locs[:prefill_len].to(torch.int64),
        seq_lens_sum=prefill_len,
        seq_lens_cpu=torch.tensor([prefill_len], device="cpu", dtype=torch.int32),
        positions=pos_sg, extend_prefix_lens=ext_prefix, extend_seq_lens=ext_seq,
        extend_start_loc=ext_start, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[prefill_len],
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool, attn_backend=backend)
    backend.init_forward_metadata(fb)
    with torch.no_grad():
        sg_h = sglang_model.model(fb.input_ids, fb.positions, fb)
    if isinstance(sg_h, tuple): sg_h = sg_h[0]
    sg_h = sglang_model.model.norm(sg_h)
    lm_w = sglang_model.lm_head.weight
    sg_logits = (sg_h @ lm_w.t()).unsqueeze(0)[:, :, :vocab_size].float()

    # Prefill comparison
    diff = (off_logits - sg_logits).abs()
    n_pages = num_real_tokens // (page_size - 1)  # completed pages
    prefill_result = {
        "num_real_tokens": num_real_tokens,
        "num_engine_tokens": prefill_len,
        "completed_pages": n_pages,
        "prefill_max_err": diff.max().item(),
        "prefill_mean_err": diff.mean().item(),
        "prefill_argmax_match": (off_logits.argmax(-1) == sg_logits.argmax(-1)).float().mean().item(),
    }

    # Per-page breakdown
    page_errors = []
    for p in range(n_pages + 1):
        start = p * page_size
        end = min((p + 1) * page_size, prefill_len)
        if start >= prefill_len:
            break
        page_diff = diff[0, start:end]
        page_errors.append({
            "page": p,
            "pos_range": f"{start}-{end-1}",
            "max_err": page_diff.max().item(),
            "mean_err": page_diff.mean().item(),
            "argmax_match": (off_logits[0, start:end].argmax(-1) == sg_logits[0, start:end].argmax(-1)).float().mean().item(),
        })
    prefill_result["page_errors"] = page_errors

    # ---- Decode ----
    decode_token = 42
    decode_seq_len = prefill_len + 1
    model_runner.req_to_token_pool.req_to_token[0, prefill_len] = token_locs[prefill_len]

    decode_input = torch.tensor([[decode_token]], device=device, dtype=torch.long)
    decode_engine_pos = prefill_len
    decode_pos_val = decode_engine_pos - (decode_engine_pos // page_size)
    decode_position_ids = torch.tensor([[decode_pos_val]], device=device, dtype=torch.long)
    decode_mask = torch.ones((1, prefill_len + 1), dtype=torch.long, device=device)

    with torch.no_grad():
        off_dec = official_model(input_ids=decode_input, position_ids=decode_position_ids,
                                  attention_mask=decode_mask,
                                  past_key_values=off_out.past_key_values, use_cache=True)
    off_dec_logits = off_dec.logits[:, -1:, :vocab_size].float()

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
        sg_dec_h = sglang_model.model(fb_dec.input_ids, fb_dec.positions, fb_dec)
    if isinstance(sg_dec_h, tuple): sg_dec_h = sg_dec_h[0]
    sg_dec_h = sglang_model.model.norm(sg_dec_h)
    sg_dec_logits = (sg_dec_h @ lm_w.t()).unsqueeze(0)[:, :, :vocab_size].float()

    dec_diff = (off_dec_logits - sg_dec_logits).abs()
    decode_result = {
        "decode_max_err": dec_diff.max().item(),
        "decode_mean_err": dec_diff.mean().item(),
        "decode_argmax_match": (off_dec_logits.argmax(-1) == sg_dec_logits.argmax(-1)).all().item(),
        "off_argmax": off_dec_logits.argmax(-1).item(),
        "sg_argmax": sg_dec_logits.argmax(-1).item(),
    }

    result = {**prefill_result, **decode_result}
    results.append(result)
    return result


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

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

    # Test lengths: N real tokens → N // (page_size-1) completed pages
    test_lengths = [
        (10,   "short, 0 pages (SWA-only)"),
        (63,   "exactly 1 page, 1 completed page"),
        (65,   "1 full page + 2 extra"),
        (126,  "exactly 2 pages, 2 completed pages"),
        (130,  "2 full pages + 4 extra"),
        (200,  "3 full pages + partial"),
        (400,  "6+ pages"),
    ]

    print("=" * 70)
    print("MULTI-LENGTH ALIGNMENT VERIFICATION")
    print("=" * 70)
    print(f"Config: hidden={config_dict['hidden_size']}, heads={config_dict['num_attention_heads']}, "
          f"kv_heads={config_dict['num_key_value_heads']}, head_dim={config_dict['head_dim']}")
    print(f"HSA: hsa_heads={config_dict['hsa_heads']}, topk={config_dict['hsa_topk']}, "
          f"chunk_size={page_size}, window={config_dict['sliding_window']}")
    print()

    # Setup official model
    torch.manual_seed(42)
    official_config = HSAConfig(**config_dict)
    official_config._attn_implementation = "eager"
    official_model = OfficialModel(official_config).to(device=device, dtype=dtype)
    official_model.eval()
    for mod in official_model.modules():
        if isinstance(mod, HierarchicalSparseAttention):
            _otf = mod.topk_func; _ohf = mod.hsa_func
            mod.topk_func = lambda q, k, *a, _f=_otf, **kw: _f(_densify(q), _densify(k), *a, **kw)
            mod.hsa_func = lambda q, k, v, *a, _f=_ohf, **kw: _f(_densify(q), _densify(k), _densify(v), *a, **kw)

    # Setup sglang model
    setup_sglang_infra()
    from sglang.srt.configs.flash_hsa import FlashHSAConfig
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.models.flash_hsa import HSAForCausalLM as SglangModel

    cfg = FlashHSAConfig(
        model_type="flash_hsa_innerx", architectures=["HSAForCausalLM"],
        vocab_size=256, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=4,
        head_dim=64, rms_norm_eps=1e-6, chunk_size=64, hsa_topk=2, hsa_mode="sparse",
        full_attn_interleave=1, hsa_heads=8, hsa_qk_ratio=4, enable_gate=False,
        use_sliding_window_merging=True, sliding_window_merging_size=64,
        use_sliding_window_attention=False, sliding_window_attention_size=None,
        tie_word_embeddings=False, rope_theta=10000.0, hidden_act="silu",
    )
    sglang_model = SglangModel(cfg).to(device=device, dtype=dtype)
    sglang_model.eval()
    loaded, total = transfer_weights(official_model, sglang_model)
    print(f"Weights transferred: {loaded}/{total}")
    for module in sglang_model.modules():
        if hasattr(module, "cos_sin_cache") and module.cos_sin_cache is not None:
            module.cos_sin_cache = module.cos_sin_cache.to(torch.float32)

    lmk_id = int(cfg.vocab_size)
    max_ctx = 1024
    r2t = torch.zeros((1, max_ctx), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(size=2048, page_size=page_size, dtype=dtype, head_num=4,
                             head_dim=64, layer_num=1, device=device,
                             enable_memory_saver=False, enable_alt_stream=False)
    mr = types.SimpleNamespace(
        device=device, page_size=page_size, sliding_window_size=None, model=sglang_model,
        model_config=types.SimpleNamespace(is_encoder_decoder=False, context_len=max_ctx,
                                            num_attention_heads=16,
                                            get_num_kv_heads=lambda tp: 4 // tp),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend="hsa", speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id),
    )
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool
    mr.token_to_kv_pool_allocator = object()
    backend = HSAAttnBackend(mr)

    # Run comparisons
    results = []
    for num_tokens, desc in test_lengths:
        print(f"\n--- {num_tokens} real tokens ({desc}) ---")
        try:
            r = run_comparison(num_tokens, config_dict, official_model, sglang_model,
                               backend, mr, results)
            # Print summary
            print(f"  Prefill: max_err={r['prefill_max_err']:.4f}  mean_err={r['prefill_mean_err']:.4f}  "
                  f"argmax_match={r['prefill_argmax_match']*100:.1f}%")
            for pe in r["page_errors"]:
                status = "OK" if pe["max_err"] < 0.05 else ("CLOSE" if pe["max_err"] < 0.5 else "DIVERGED")
                print(f"    Page {pe['page']} (pos {pe['pos_range']}): max_err={pe['max_err']:.4f} "
                      f"argmax={pe['argmax_match']*100:.0f}% [{status}]")
            dmatch = "YES" if r["decode_argmax_match"] else "NO"
            print(f"  Decode:  max_err={r['decode_max_err']:.4f}  mean_err={r['decode_mean_err']:.4f}  "
                  f"argmax_match={dmatch}  (off={r['off_argmax']} sg={r['sg_argmax']})")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({"num_real_tokens": num_tokens, "error": str(e)})

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Tokens':>8} {'Pages':>5} {'Prefill Max':>12} {'Prefill Argmax':>14} {'Decode Max':>11} {'Decode Argmax':>13}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['num_real_tokens']:>8} {'ERROR':>5}")
            continue
        dm = "YES" if r["decode_argmax_match"] else "NO"
        print(f"{r['num_real_tokens']:>8} {r['completed_pages']:>5} "
              f"{r['prefill_max_err']:>12.4f} {r['prefill_argmax_match']*100:>13.1f}% "
              f"{r['decode_max_err']:>11.4f} {dm:>13}")

    # Save results
    out_path = os.path.join(_SCRIPT_DIR, "milestone", "alignment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
