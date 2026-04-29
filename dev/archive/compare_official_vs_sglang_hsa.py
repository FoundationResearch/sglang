"""
Compare official FlashHSA (modeling_hsa_lmk.py) vs sglang HSA (flash_hsa.py)
using identical random weights on the same input.

Runs prefill (extend) on both implementations with the same:
  - Random weights (shared state_dict)
  - Input token IDs (with LMK inserted)
  - Position IDs (landmark-aware)

Then compares output logits to verify numerical equivalence.

Usage:
  python dev/compare_official_vs_sglang_hsa.py

Requirements:
  pip install transformers torch einops tilelang liger-kernel
"""

import os
import sys
import types
import logging
import tempfile

# ============================================================
# 1. Install veomni mock (same as run_official_hsa_inference.py)
# ============================================================

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

    def _get_logger(name=None):
        logger = logging.getLogger(name)
        if not hasattr(logger, "info_rank0"):
            logger.info_rank0 = logger.info
        if not hasattr(logger, "warning_rank0"):
            logger.warning_rank0 = logger.warning
        return logger
    sys.modules["veomni.utils.logging"].get_logger = _get_logger
    sys.modules["veomni.utils.import_utils"].is_liger_kernel_available = lambda: True
    sys.modules["veomni.utils.import_utils"].is_torch_npu_available = lambda: False
    sys.modules["veomni.utils.import_utils"].is_transformers_version_greater_or_equal_to = lambda v: True
    sys.modules["veomni.models.module_utils"].GradientCheckpointingLayer = nn.Module

    class _FR:
        def register(self, n):
            return lambda f: f
    sys.modules["veomni.models.loader"].MODELING_REGISTRY = _FR()
    sys.modules["veomni.models.loader"].MODEL_CONFIG_REGISTRY = _FR()


_install_veomni_mock()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ILL_DIR = os.path.join(_SCRIPT_DIR, "InfiniteLongLM")
if _ILL_DIR not in sys.path:
    sys.path.insert(0, _ILL_DIR)

for _n in ["models.SWANGPT", "models.DRT", "models.SWANNSA"]:
    sys.modules[_n] = types.ModuleType(_n)

# ============================================================
# 2. Imports
# ============================================================

import torch
import torch.distributed as dist

# Official model
from models.FlashHSA.modeling_hsa_lmk import (
    HSAForCausalLM as OfficialHSAForCausalLM,
    HierarchicalSparseAttention,
)
from models.FlashHSA.configuration_hsa import HSAConfig
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

# Patch official model for inference compatibility
_orig_hsa_init = HierarchicalSparseAttention.__init__
def _patched_hsa_init(self, config, layer_idx):
    _orig_hsa_init(self, config, layer_idx)
    if not hasattr(self, "num_key_value_groups"):
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
HierarchicalSparseAttention.__init__ = _patched_hsa_init

# sglang imports
from sglang.srt.layers import dp_attention as dpa
dpa.get_attention_tp_size = lambda: 1


# ============================================================
# 3. Weight transfer: official → sglang
# ============================================================

def transfer_weights(official_model, sglang_model):
    """Copy weights from official model to sglang model.

    Both models use the same weight naming convention for HSA layers:
      model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,hsa_q_proj,...}.weight
      model.layers.{i}.self_attn.{q_norm,k_norm,lmk_q_norm}.weight
      model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
      model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
      model.embed_tokens.weight
      model.norm.weight
      lm_head.weight
    """
    official_sd = official_model.state_dict()
    sglang_sd = sglang_model.state_dict()

    loaded = 0
    skipped = []
    for name, param in sglang_sd.items():
        if name in official_sd:
            src = official_sd[name]
            if src.shape == param.shape:
                param.data.copy_(src)
                loaded += 1
            else:
                # Handle embed_tokens/lm_head vocab padding difference: copy min rows.
                if "embed_tokens" in name or "lm_head" in name:
                    min_rows = min(src.shape[0], param.shape[0])
                    param.data[:min_rows].copy_(src[:min_rows])
                    loaded += 1
                else:
                    skipped.append(f"{name}: shape mismatch {src.shape} vs {param.shape}")
        elif "gate_up_proj" in name:
            # sglang fuses gate_proj + up_proj → gate_up_proj.
            # e.g. "model.layers.0.mlp.gate_up_proj.weight" →
            #      "model.layers.0.mlp.gate_proj.weight" + "model.layers.0.mlp.up_proj.weight"
            gate_name = name.replace("gate_up_proj", "gate_proj")
            up_name = name.replace("gate_up_proj", "up_proj")
            if gate_name in official_sd and up_name in official_sd:
                gate_w = official_sd[gate_name]
                up_w = official_sd[up_name]
                fused = torch.cat([gate_w, up_w], dim=0)
                if fused.shape == param.shape:
                    param.data.copy_(fused)
                    loaded += 1
                else:
                    skipped.append(f"{name}: fused shape mismatch {fused.shape} vs {param.shape}")
            else:
                skipped.append(f"{name}: gate/up not found in official model")
        else:
            skipped.append(f"{name}: not in official model")

    print(f"  Transferred {loaded}/{len(sglang_sd)} parameters")
    if skipped:
        print(f"  Skipped {len(skipped)} parameters:")
        for s in skipped[:10]:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    return loaded


# ============================================================
# 4. sglang model setup (from test infrastructure)
# ============================================================

def setup_sglang_model(config_dict, official_model):
    """Set up sglang model with matching weights and infrastructure."""
    from sglang.srt.configs.flash_hsa import FlashHSAConfig
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch, ForwardMode, compute_position,
    )
    from sglang.srt.models.flash_hsa import HSAForCausalLM as SglangHSAForCausalLM
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Init process group if needed.
    if not dist.is_initialized():
        _, path = tempfile.mkstemp(prefix="sglang_cmp_", suffix=".tmp")
        dist.init_process_group(backend="gloo", init_method=f"file://{path}", rank=0, world_size=1)
    if not ps.model_parallel_is_initialized():
        ps._WORLD = ps.init_world_group(ranks=[0], local_rank=0, backend="gloo")
        ps._TP = ps.init_model_parallel_group(
            group_ranks=[[0]], local_rank=0, backend="gloo",
            use_custom_allreduce=False, use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False, group_name="tp",
        )
        ps._PP = ps.init_model_parallel_group(
            group_ranks=[[0]], local_rank=0, backend="gloo",
            use_custom_allreduce=False, use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False, group_name="pp",
        )
    if getattr(dpa, "_ATTN_TP_RANK", None) is None:
        dpa._ATTN_TP_RANK = 0
        dpa._ATTN_TP_SIZE = 1
        dpa._ATTN_DP_RANK = 0
        dpa._ATTN_DP_SIZE = 1
        dpa._LOCAL_ATTN_DP_RANK = 0
        dpa._LOCAL_ATTN_DP_SIZE = 1
        dpa._ENABLE_DP_ATTENTION_FLAG = False
        dpa._ATTN_TP_GROUP = ps.get_tp_group()

    # Global ServerArgs.
    try:
        sa = ServerArgs(model_path="dummy")
        sa.attention_backend = "hsa"
        sa.enable_dp_lm_head = False
        set_global_server_args_for_scheduler(sa)
    except Exception:
        pass

    # Build sglang config.
    page_size = config_dict["chunk_size"]
    cfg = FlashHSAConfig(
        model_type=config_dict.get("model_type", "flash_hsa_innerx"),
        architectures=["HSAForCausalLM"],
        vocab_size=config_dict["vocab_size"],
        hidden_size=config_dict["hidden_size"],
        intermediate_size=config_dict["intermediate_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        head_dim=config_dict["head_dim"],
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
        attention_bias=config_dict.get("attention_bias", False),
        chunk_size=page_size,
        hsa_topk=config_dict["hsa_topk"],
        hsa_mode=config_dict.get("hsa_mode", "sparse"),
        full_attn_interleave=config_dict["full_attn_interleave"],
        hsa_heads=config_dict["hsa_heads"],
        hsa_qk_ratio=config_dict["hsa_qk_ratio"],
        enable_gate=config_dict.get("enable_gate", False),
        use_sliding_window_merging=config_dict.get("use_sliding_window", True),
        sliding_window_merging_size=config_dict.get("sliding_window", page_size),
        use_sliding_window_attention=False,
        sliding_window_attention_size=None,
        tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
        rope_theta=config_dict.get("rope_theta", 10000.0),
        hidden_act=config_dict.get("hidden_act", "silu"),
    )

    # Create sglang model & transfer weights.
    sglang_model = SglangHSAForCausalLM(cfg).to(device=device, dtype=dtype)
    sglang_model.eval()
    # Debug: show official MLP weight names.
    for n in official_model.state_dict():
        if "mlp" in n:
            print(f"  Official MLP weight: {n} {official_model.state_dict()[n].shape}")
    print("Transferring weights official → sglang...")
    transfer_weights(official_model, sglang_model)

    # Build model_runner stub.
    lmk_id = int(cfg.vocab_size)
    max_context_len = 256
    max_total_num_tokens = 1024

    model_runner = types.SimpleNamespace()
    model_runner.device = device
    model_runner.page_size = page_size
    model_runner.sliding_window_size = None
    model_runner.model = sglang_model
    model_runner.model_config = types.SimpleNamespace(
        is_encoder_decoder=False,
        context_len=max_context_len,
        num_attention_heads=int(cfg.num_attention_heads),
        get_num_kv_heads=lambda tp_size: int(cfg.num_key_value_heads) // int(tp_size),
    )
    model_runner.hybrid_gdn_config = None
    model_runner.kimi_linear_config = None
    model_runner.gpu_id = 0
    model_runner.server_args = types.SimpleNamespace(
        attention_backend="hsa",
        speculative_num_draft_tokens=0,
        speculative_num_steps=0,
        triton_attention_num_kv_splits=8,
        triton_attention_split_tile_size=None,
        enable_deterministic_inference=False,
        hsa_topk=None,
        hsa_selection_strategy=None,
        hsa_layers=None,
        hsa_window_size=None,
        hsa_enable_swa_merging=None,
        hsa_lmk_id=lmk_id,
    )

    req_to_token = torch.zeros((1, max_context_len), dtype=torch.int32, device=device)
    model_runner.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=req_to_token)
    model_runner.token_to_kv_pool = MHATokenToKVPool(
        size=max_total_num_tokens,
        page_size=page_size,
        dtype=dtype,
        head_num=int(cfg.num_key_value_heads),
        head_dim=int(cfg.head_dim),
        layer_num=int(cfg.num_hidden_layers),
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    model_runner.token_to_kv_pool_allocator = object()

    backend = HSAAttnBackend(model_runner)

    # Ensure rotary embedding cos_sin_cache is float32 (required by sgl_kernel).
    for module in sglang_model.modules():
        if hasattr(module, "cos_sin_cache") and module.cos_sin_cache is not None:
            module.cos_sin_cache = module.cos_sin_cache.to(torch.float32)

    return sglang_model, backend, model_runner, cfg, lmk_id, page_size


# ============================================================
# 5. Run comparison
# ============================================================

def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ---- Config (small model for fast testing) ----
    # Config using realistic dimensions that satisfy both:
    # - tilelang kernels (official): chunk_size >= 64, head_dim >= 64
    # - sglang Triton kernels: any size works
    # NOTE: First run may be slow due to Triton JIT compilation (~2-5 min).
    config_dict = dict(
        model_type="flash_hsa_innerx",
        vocab_size=256,
        hidden_size=1024,          # 16 heads * 64 head_dim
        intermediate_size=4096,
        num_hidden_layers=1,       # single layer for fast testing
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        chunk_size=64,
        hsa_topk=2,
        hsa_mode="sparse",
        full_attn_interleave=1,    # every layer is HSA
        hsa_heads=8,
        hsa_qk_ratio=4,
        enable_gate=False,
        use_sliding_window=True,
        sliding_window=64,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        hidden_act="silu",
    )
    page_size = config_dict["chunk_size"]
    lmk_id = config_dict["vocab_size"]

    # ---- Prompt: 65 real tokens → 1 full page + 2 extra tokens ----
    real_tokens = list(range(5, 5 + 65))

    print("=" * 60)
    print("OFFICIAL vs SGLANG HSA COMPARISON")
    print("=" * 60)
    print(f"Config: hidden={config_dict['hidden_size']}, layers={config_dict['num_hidden_layers']}, "
          f"heads={config_dict['num_attention_heads']}, kv_heads={config_dict['num_key_value_heads']}")
    print(f"HSA: hsa_heads={config_dict['hsa_heads']}, topk={config_dict['hsa_topk']}, "
          f"chunk_size={page_size}")
    print(f"Prompt: {len(real_tokens)} real tokens")

    # ---- Official model ----
    print("\n--- Official Model ---")
    torch.manual_seed(42)
    official_config = HSAConfig(**config_dict)
    official_config._attn_implementation = "eager"
    official_model = OfficialHSAForCausalLM(official_config).to(device=device, dtype=dtype)
    official_model.eval()

    # Densify wrapper for tilelang stride issue
    def _densify(t):
        return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)

    for mod in official_model.modules():
        if isinstance(mod, HierarchicalSparseAttention):
            _orig_tf = mod.topk_func
            _orig_hf = mod.hsa_func
            mod.topk_func = lambda q, k, *a, _f=_orig_tf, **kw: _f(_densify(q), _densify(k), *a, **kw)
            mod.hsa_func = lambda q, k, v, *a, _f=_orig_hf, **kw: _f(_densify(q), _densify(k), _densify(v), *a, **kw)

    print(f"  Params: {sum(p.numel() for p in official_model.parameters()) / 1e3:.1f}K")

    # Prepare input with LMK.
    input_ids_raw = torch.tensor([real_tokens], device=device, dtype=torch.long)
    input_ids_lmk = insert_special_tokens(input_ids_raw, lmk_id, page_size)
    position_ids = create_position_ids_with_landmarks(len(real_tokens), page_size, device)
    L_lmk = input_ids_lmk.shape[1]
    attention_mask = torch.ones((1, L_lmk), dtype=torch.long, device=device)

    print(f"  Input (with LMK): {input_ids_lmk.shape[1]} tokens")
    print(f"  Positions: {position_ids.squeeze().tolist()}")

    with torch.no_grad():
        official_out = official_model(
            input_ids=input_ids_lmk,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    official_logits = official_out.logits[:, :, :config_dict["vocab_size"]]
    print(f"  Output logits: {official_logits.shape}")

    # ---- sglang model ----
    print("\n--- sglang Model ---")
    (sglang_model, backend, model_runner, cfg, _, _) = setup_sglang_model(config_dict, official_model)
    print(f"  Params: {sum(p.numel() for p in sglang_model.parameters()) / 1e3:.1f}K")

    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch, ForwardMode, compute_position,
    )
    from sglang.srt.managers.schedule_batch import Req

    # Build ForwardBatch for extend (prefill).
    fill_ids = Req._hsa_insert_lmk_prompt(real_tokens, page_size=page_size, lmk_id=lmk_id)
    prefill_len = len(fill_ids)
    assert prefill_len == L_lmk, f"LMK insertion mismatch: {prefill_len} vs {L_lmk}"

    token_locs = torch.arange(0, prefill_len, dtype=torch.int32, device=device)
    model_runner.req_to_token_pool.req_to_token[0, :prefill_len] = token_locs

    extend_prefix_lens = torch.tensor([0], device=device, dtype=torch.int32)
    extend_seq_lens = torch.tensor([prefill_len], device=device, dtype=torch.int32)
    positions_sg, extend_start_loc = compute_position(
        attn_backend="hsa",
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_sum=prefill_len,
        page_size=page_size,
        enable_landmark_positions=True,
    )

    print(f"  Input (with LMK): {prefill_len} tokens")
    print(f"  Positions: {positions_sg.tolist()}")

    # Compare positions.
    pos_official = position_ids.squeeze().cpu().tolist()
    pos_sglang = positions_sg.cpu().tolist()
    if pos_official == pos_sglang:
        print("  Positions match!")
    else:
        diffs = [(i, pos_official[i], pos_sglang[i])
                 for i in range(min(len(pos_official), len(pos_sglang)))
                 if pos_official[i] != pos_sglang[i]]
        print(f"  POSITION MISMATCH at {len(diffs)} positions:")
        for i, off, sg in diffs[:5]:
            print(f"    token {i}: official={off}, sglang={sg}")
        print("  Official LMK positions get pos = last_real_pos + 1 (next pos)")
        print("  sglang   LMK positions get pos = last_real_pos     (same pos)")
        print("  NOTE: This is a known difference in position encoding semantics.")

    out_cache_loc = token_locs[:prefill_len].to(torch.int64)
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor(fill_ids, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([prefill_len], device=device, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=prefill_len,
        seq_lens_cpu=torch.tensor([prefill_len], device="cpu", dtype=torch.int32),
        positions=positions_sg,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        extend_start_loc=extend_start_loc,
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[prefill_len],
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        attn_backend=backend,
    )
    backend.init_forward_metadata(fb)

    with torch.no_grad():
        sglang_hidden = sglang_model.model(fb.input_ids, fb.positions, fb)
    # sglang model returns (hidden_states, residual) tuple from the last layer.
    if isinstance(sglang_hidden, tuple):
        sglang_hidden = sglang_hidden[0]
    # Apply final norm + lm_head weight directly.
    # (ParallelLMHead.forward() refuses direct calls; use the raw weight.)
    sglang_hidden = sglang_model.model.norm(sglang_hidden)
    lm_weight = sglang_model.lm_head.weight  # [padded_vocab, hidden]
    sglang_logits = (sglang_hidden @ lm_weight.t()).unsqueeze(0)  # [1, T, padded_vocab]
    sglang_logits = sglang_logits[:, :, :config_dict["vocab_size"]]
    print(f"  Output logits: {sglang_logits.shape}")

    # ---- Compare ----
    print("\n--- Comparison ---")
    # Both should be [1, L_lmk, vocab_size]
    off_l = official_logits.float()
    sg_l = sglang_logits.float()

    # Handle potential shape differences (sglang may only return last-token logits).
    if off_l.shape != sg_l.shape:
        print(f"  Shape mismatch: official={off_l.shape}, sglang={sg_l.shape}")
        # Compare last token only.
        off_l = off_l[:, -1:, :]
        sg_l = sg_l[:, -1:, :]
        print(f"  Comparing last token only: {off_l.shape}")

    abs_diff = (off_l - sg_l).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # Token-by-token argmax comparison.
    off_argmax = off_l.argmax(dim=-1)
    sg_argmax = sg_l.argmax(dim=-1)
    argmax_match = (off_argmax == sg_argmax).float().mean().item()

    print(f"  Max absolute error:  {max_abs:.6f}")
    print(f"  Mean absolute error: {mean_abs:.6f}")
    print(f"  Argmax match rate:   {argmax_match * 100:.1f}%")

    # Per-position breakdown.
    if off_l.shape[1] > 1:
        print(f"\n  Per-position max error:")
        for t in range(off_l.shape[1]):
            err = abs_diff[0, t].max().item()
            match = "OK" if off_argmax[0, t] == sg_argmax[0, t] else "MISMATCH"
            print(f"    pos {t:3d}: max_err={err:.6f}  argmax={match}")

    if max_abs < 0.1:
        print(f"\n  PASS: outputs match within tolerance (max_err={max_abs:.6f})")
    else:
        print(f"\n  WARN: large difference detected (max_err={max_abs:.6f})")
        print("  This may indicate weight mapping or computation differences.")

    # ============================================================
    # DECODE COMPARISON
    # ============================================================
    print(f"\n{'='*60}")
    print("DECODE COMPARISON (1 token after prefill)")
    print(f"{'='*60}")

    next_token_id = 42  # arbitrary decode token
    vocab_size = config_dict["vocab_size"]

    # ---- Official decode ----
    print("\n--- Official Decode ---")
    # official_out.past_key_values has the KV cache from prefill.
    # Feed one token, get next logits.
    decode_input = torch.tensor([[next_token_id]], device=device, dtype=torch.long)
    # Position for this token: it's at engine-visible index = prefill_len (0-based).
    # pos = prefill_len - floor(prefill_len / page_size)
    decode_engine_pos = prefill_len
    decode_pos = decode_engine_pos - (decode_engine_pos // page_size)
    decode_position_ids = torch.tensor([[decode_pos]], device=device, dtype=torch.long)
    decode_mask = torch.ones((1, prefill_len + 1), dtype=torch.long, device=device)

    with torch.no_grad():
        official_dec_out = official_model(
            input_ids=decode_input,
            position_ids=decode_position_ids,
            attention_mask=decode_mask,
            past_key_values=official_out.past_key_values,
            use_cache=True,
        )
    official_dec_logits = official_dec_out.logits[:, -1:, :vocab_size]  # [1, 1, V]
    print(f"  Decode position: {decode_pos}")
    print(f"  Output logits: {official_dec_logits.shape}")

    # ---- sglang decode ----
    print("\n--- sglang Decode ---")
    from sglang.srt.model_executor.forward_batch_info import compute_decode_positions_landmark

    # Register the new token slot in req_to_token.
    decode_seq_len = prefill_len + 1
    decode_token_loc = torch.tensor([prefill_len], device=device, dtype=torch.int32)
    model_runner.req_to_token_pool.req_to_token[0, prefill_len] = decode_token_loc[0]

    # Compute decode position.
    dec_pos = compute_decode_positions_landmark(
        torch.tensor([decode_seq_len], device=device, dtype=torch.int32),
        page_size=page_size,
    )
    print(f"  Decode position: {int(dec_pos.item())}")

    # Verify positions match.
    assert int(dec_pos.item()) == decode_pos, (
        f"Decode position mismatch: official={decode_pos}, sglang={int(dec_pos.item())}"
    )
    print("  Decode positions match!")

    out_cache_loc_dec = decode_token_loc[:1].to(torch.int64)
    fb_dec = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=torch.tensor([next_token_id], device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([decode_seq_len], device=device, dtype=torch.int32),
        out_cache_loc=out_cache_loc_dec,
        seq_lens_sum=decode_seq_len,
        seq_lens_cpu=torch.tensor([decode_seq_len], device="cpu", dtype=torch.int32),
        positions=dec_pos,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        attn_backend=backend,
    )

    backend.init_forward_metadata(fb_dec)
    with torch.no_grad():
        sglang_dec_hidden = sglang_model.model(fb_dec.input_ids, fb_dec.positions, fb_dec)
    if isinstance(sglang_dec_hidden, tuple):
        sglang_dec_hidden = sglang_dec_hidden[0]
    sglang_dec_hidden = sglang_model.model.norm(sglang_dec_hidden)
    sglang_dec_logits = (sglang_dec_hidden @ lm_weight.t()).unsqueeze(0)  # [1, 1, padded_V]
    sglang_dec_logits = sglang_dec_logits[:, :, :vocab_size]
    print(f"  Output logits: {sglang_dec_logits.shape}")

    # ---- Compare decode ----
    print("\n--- Decode Comparison ---")
    off_dec = official_dec_logits.float()
    sg_dec = sglang_dec_logits.float()

    dec_abs_diff = (off_dec - sg_dec).abs()
    dec_max = dec_abs_diff.max().item()
    dec_mean = dec_abs_diff.mean().item()
    off_dec_argmax = off_dec.argmax(dim=-1)
    sg_dec_argmax = sg_dec.argmax(dim=-1)
    dec_argmax_match = (off_dec_argmax == sg_dec_argmax).all().item()

    print(f"  Max absolute error:  {dec_max:.6f}")
    print(f"  Mean absolute error: {dec_mean:.6f}")
    print(f"  Official argmax: {int(off_dec_argmax.item())}")
    print(f"  sglang  argmax:  {int(sg_dec_argmax.item())}")
    print(f"  Argmax match: {'YES' if dec_argmax_match else 'NO'}")

    # Top-5 comparison.
    off_top5_vals, off_top5_ids = off_dec.squeeze().topk(5)
    sg_top5_vals, sg_top5_ids = sg_dec.squeeze().topk(5)
    print(f"\n  Official top-5: {list(zip(off_top5_ids.tolist(), [f'{v:.3f}' for v in off_top5_vals.tolist()]))}")
    print(f"  sglang  top-5:  {list(zip(sg_top5_ids.tolist(), [f'{v:.3f}' for v in sg_top5_vals.tolist()]))}")

    if dec_max < 0.1:
        print(f"\n  DECODE PASS: outputs match within tolerance (max_err={dec_max:.6f})")
    elif dec_max < 2.0:
        print(f"\n  DECODE CLOSE: small difference (max_err={dec_max:.6f})")
    else:
        print(f"\n  DECODE WARN: large difference (max_err={dec_max:.6f})")


if __name__ == "__main__":
    main()
