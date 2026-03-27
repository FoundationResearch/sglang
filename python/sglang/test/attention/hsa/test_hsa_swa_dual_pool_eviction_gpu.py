"""
Tests for HSA + SWA dual-pool KV eviction.

Verifies that:
1. HSA model is detected as hybrid_swa and layer IDs are classified correctly
2. SWAKVPool routes HSA layers → full pool, SWA layers → swa pool
3. SWA pool tokens outside the window are evictable (tombstone mechanism)
4. HSA layers retain full KV after SWA eviction
5. End-to-end: model forward works with SWAKVPool (extend + decode)
"""

import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)


# ---- Test 1: Config detection ----


def test_hsa_model_detected_as_hybrid_swa():
    """HSAForCausalLM should be detected as a hybrid SWA model."""
    from sglang.srt.configs.model_config import (
        get_hybrid_layer_ids,
        is_hybrid_swa_model,
    )

    assert is_hybrid_swa_model(["HSAForCausalLM"])
    assert not is_hybrid_swa_model(["LlamaForCausalLM"])


def test_hsa_layer_ids_from_layer_types():
    """layer_types='full_attention' maps to full_attention_layer_ids (HSA layers)."""
    from sglang.srt.configs.flash_hsa import FlashHSAConfig
    from sglang.srt.configs.model_config import get_hybrid_layer_ids

    cfg = FlashHSAConfig(
        num_hidden_layers=8,
        full_attn_interleave=4,
        use_sliding_window=True,
        sliding_window=64,
    )
    # layer_types: every 4th layer (3, 7) is "full_attention"
    assert cfg.layer_types[3] == "full_attention"
    assert cfg.layer_types[7] == "full_attention"
    assert cfg.layer_types[0] == "sliding_attention"

    swa_ids, full_ids = get_hybrid_layer_ids(["HSAForCausalLM"], cfg)
    assert full_ids == [3, 7], f"Expected HSA layers [3,7], got {full_ids}"
    assert swa_ids == [0, 1, 2, 4, 5, 6], f"Expected SWA layers, got {swa_ids}"


def test_hsa_layer_ids_fallback_interleave():
    """Without layer_types, falls back to full_attn_interleave."""
    from sglang.srt.configs.model_config import get_hybrid_layer_ids

    # Create a minimal config without layer_types
    cfg = types.SimpleNamespace(
        num_hidden_layers=8,
        full_attn_interleave=2,
    )
    # Remove layer_types to force fallback
    swa_ids, full_ids = get_hybrid_layer_ids(["HSAForCausalLM"], cfg)
    # interleave=2: layers 1, 3, 5, 7 are full (HSA)
    assert full_ids == [1, 3, 5, 7]
    assert swa_ids == [0, 2, 4, 6]


def test_sliding_window_size_from_flash_hsa_config():
    """_get_sliding_window_size reads sliding_window_attention_size."""
    from sglang.srt.configs.flash_hsa import FlashHSAConfig
    from sglang.srt.configs.model_config import ModelConfig

    # We test the fallback by creating a mock hf_text_config
    cfg = types.SimpleNamespace(
        sliding_window_size=None,
        sliding_window=None,
        sliding_window_attention_size=512,
    )
    mc = types.SimpleNamespace(hf_text_config=cfg)
    mc._get_sliding_window_size = ModelConfig._get_sliding_window_size.__get__(mc)
    assert mc._get_sliding_window_size() == 512


# ---- Test 2: SWAKVPool routing ----


def test_swa_kv_pool_routes_hsa_and_swa_layers():
    """SWAKVPool routes HSA layers to full pool and SWA layers to SWA pool."""
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    device = "cuda"
    dtype = torch.bfloat16
    H, D = 4, 16
    full_size = 256
    swa_size = 64
    page_size = 4

    swa_layer_ids = [0, 1, 2]  # SWA layers
    full_layer_ids = [3]  # HSA layer

    pool = SWAKVPool(
        size=full_size,
        size_swa=swa_size,
        page_size=page_size,
        dtype=dtype,
        head_num=H,
        head_dim=D,
        swa_attention_layer_ids=swa_layer_ids,
        full_attention_layer_ids=full_layer_ids,
        enable_kvcache_transpose=False,
        device=device,
    )

    # Verify routing
    # Layer 3 (HSA) → full pool
    k3 = pool.get_key_buffer(3)
    assert k3.shape[0] == full_size + page_size, f"Full pool size mismatch: {k3.shape}"

    # Layer 0 (SWA) → swa pool
    k0 = pool.get_key_buffer(0)
    assert k0.shape[0] == swa_size + page_size, f"SWA pool size mismatch: {k0.shape}"

    # Verify different pool pointers
    assert k3.data_ptr() != k0.data_ptr(), "HSA and SWA layers should use different pools"


# ---- Test 3: SWA eviction preserves full pool ----


def test_swa_eviction_preserves_hsa_kv():
    """After SWA tombstone eviction, HSA layers still have full KV."""
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.swa_memory_pool import (
        SWAKVPool,
        SWATokenToKVPoolAllocator,
    )

    device = "cuda"
    dtype = torch.bfloat16
    H, D = 4, 16
    page_size = 1  # SWATokenToKVPoolAllocator.alloc() requires page_size=1
    full_size = 128
    swa_size = 32

    swa_layer_ids = [0, 1, 2]
    full_layer_ids = [3]

    pool = SWAKVPool(
        size=full_size,
        size_swa=swa_size,
        page_size=page_size,
        dtype=dtype,
        head_num=H,
        head_dim=D,
        swa_attention_layer_ids=swa_layer_ids,
        full_attention_layer_ids=full_layer_ids,
        enable_kvcache_transpose=False,
        device=device,
    )

    allocator = SWATokenToKVPoolAllocator(
        size=full_size,
        size_swa=swa_size,
        page_size=page_size,
        dtype=dtype,
        device=device,
        kvcache=pool,
        need_sort=False,
    )
    pool.register_mapping(allocator.full_to_swa_index_mapping)

    # Allocate some tokens
    n_tokens = 20
    full_locs = allocator.alloc(n_tokens)
    assert full_locs is not None

    # Write distinctive values to full pool (HSA layer 3)
    marker = torch.ones(n_tokens, H, D, dtype=dtype, device=device) * 42.0
    pool.full_kv_pool.k_buffer[0][full_locs.long()] = marker  # layer_id_pool=0

    # Write to SWA pool
    swa_locs = allocator.full_to_swa_index_mapping[full_locs.long()]
    swa_marker = torch.ones(n_tokens, H, D, dtype=dtype, device=device) * 7.0
    pool.swa_kv_pool.k_buffer[0][swa_locs.long()] = swa_marker

    # Free SWA only (simulates tombstone eviction)
    allocator.free_swa(full_locs)

    # Verify: full pool (HSA) KV is preserved
    full_k = pool.full_kv_pool.k_buffer[0][full_locs.long()]
    assert (full_k == 42.0).all(), "HSA KV should be preserved after SWA eviction"

    # Verify: SWA mapping is cleared
    swa_locs_after = allocator.full_to_swa_index_mapping[full_locs.long()]
    assert (swa_locs_after == 0).all(), "SWA mapping should be cleared after free_swa"


# ---- Test 4: End-to-end model forward with SWAKVPool ----


def test_hsa_model_forward_with_swa_kv_pool():
    """
    End-to-end test: construct a 4-layer HSA model with SWAKVPool,
    run prefill + decode, verify both HSA and SWA layers produce valid output.
    """
    from sglang.srt.configs.flash_hsa import FlashHSAConfig
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers import dp_attention as dpa
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.swa_memory_pool import (
        SWAKVPool,
        SWATokenToKVPoolAllocator,
    )
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        ForwardMode,
        compute_decode_positions_landmark,
        compute_position,
    )
    from sglang.srt.models.flash_hsa import HSAForCausalLM
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    device = "cuda"
    dtype = torch.bfloat16

    # Init distributed (singleton-safe)
    dpa.get_attention_tp_size = lambda: 1
    if not dist.is_initialized():
        _, p = tempfile.mkstemp(prefix="swahsa_", suffix=".t")
        dist.init_process_group(
            backend="gloo", init_method=f"file://{p}", rank=0, world_size=1
        )
    if not ps.model_parallel_is_initialized():
        ps._WORLD = ps.init_world_group(ranks=[0], local_rank=0, backend="gloo")
        ps._TP = ps.init_model_parallel_group(
            group_ranks=[[0]],
            local_rank=0,
            backend="gloo",
            use_custom_allreduce=False,
            use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False,
            group_name="tp",
        )
        ps._PP = ps.init_model_parallel_group(
            group_ranks=[[0]],
            local_rank=0,
            backend="gloo",
            use_custom_allreduce=False,
            use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False,
            group_name="pp",
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
    try:
        sa = ServerArgs(model_path="dummy")
        sa.attention_backend = "hsa"
        sa.enable_dp_lm_head = False
        set_global_server_args_for_scheduler(sa)
    except Exception:
        pass

    # Config: 4 layers, interleave=2 → layers 1,3 are HSA, layers 0,2 are SWA
    # Head math: num_attention_heads=16, num_kv_heads=4, hsa_heads=4
    # → hsa_denom = 16/4 = 4, h_kv(4) % hsa_denom(4) == 0 ✓
    page_size = 4
    window_size = 8  # SWA window
    cfg = FlashHSAConfig(
        model_type="flash_hsa_innerx",
        architectures=["HSAForCausalLM"],
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=16,
        rms_norm_eps=1e-6,
        chunk_size=page_size,
        hsa_topk=2,
        hsa_mode="sparse",
        full_attn_interleave=2,
        hsa_heads=4,
        hsa_qk_ratio=4,
        use_sliding_window_merging=True,
        sliding_window_merging_size=page_size,
        use_sliding_window_attention=True,
        sliding_window_attention_size=window_size,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        hidden_act="silu",
    )

    # Verify layer classification
    from sglang.srt.configs.model_config import get_hybrid_layer_ids

    swa_ids, full_ids = get_hybrid_layer_ids(["HSAForCausalLM"], cfg)
    assert full_ids == [1, 3], f"HSA layers should be [1,3], got {full_ids}"
    assert swa_ids == [0, 2], f"SWA layers should be [0,2], got {swa_ids}"

    # Build model
    model = HSAForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()
    for m in model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    # Verify SWA layers have sliding_window_size set on RadixAttention
    # This is the critical property: SWA layers have a window, HSA layers don't
    for i, layer in enumerate(model.model.layers):
        attn_mod = layer.self_attn
        if hasattr(attn_mod, "attn"):
            radix = attn_mod.attn
            if i in swa_ids:
                assert (
                    radix.sliding_window_size == window_size
                ), f"Layer {i} (SWA) should have sliding_window_size={window_size}, got {radix.sliding_window_size}"
            else:
                assert radix.sliding_window_size in (
                    -1,
                    None,
                ), f"Layer {i} (HSA) should NOT have sliding_window_size, got {radix.sliding_window_size}"

    # Verify SWAKVPool routes correctly for this model's layer config
    lmk_id = int(cfg.vocab_size)
    max_ctx = 256
    swa_ctx = 64
    H = int(cfg.num_key_value_heads)
    D = int(cfg.head_dim)

    pool = SWAKVPool(
        size=max_ctx,
        size_swa=swa_ctx,
        page_size=page_size,
        dtype=dtype,
        head_num=H,
        head_dim=D,
        swa_attention_layer_ids=swa_ids,
        full_attention_layer_ids=full_ids,
        enable_kvcache_transpose=False,
        device=device,
    )

    # HSA layers (1, 3) → full pool
    k1 = pool.get_key_buffer(1)
    k3 = pool.get_key_buffer(3)
    assert k1.shape[0] == max_ctx + page_size
    assert k3.shape[0] == max_ctx + page_size

    # SWA layers (0, 2) → swa pool (smaller)
    k0 = pool.get_key_buffer(0)
    k2 = pool.get_key_buffer(2)
    assert k0.shape[0] == swa_ctx + page_size
    assert k2.shape[0] == swa_ctx + page_size

    # Different pool pointers
    assert k1.data_ptr() != k0.data_ptr()

    # Now run model forward with standard MHATokenToKVPool (not SWAKVPool)
    # to verify the model itself works with the sliding_window_size set on SWA layers
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    std_pool = MHATokenToKVPool(
        size=max_ctx + 64,
        page_size=page_size,
        dtype=dtype,
        head_num=H,
        head_dim=D,
        layer_num=int(cfg.num_hidden_layers),
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    r2t = torch.zeros((1, max_ctx), dtype=torch.int32, device=device)
    tl = torch.arange(0, max_ctx, dtype=torch.int32, device=device)

    mr = types.SimpleNamespace(
        device=device,
        page_size=page_size,
        sliding_window_size=window_size,
        model=model,
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False,
            context_len=max_ctx,
            num_attention_heads=int(cfg.num_attention_heads),
            head_dim=D,
            get_num_kv_heads=lambda tp: int(cfg.num_key_value_heads) // tp,
        ),
        hybrid_gdn_config=None,
        kimi_linear_config=None,
        gpu_id=0,
        server_args=types.SimpleNamespace(
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
        ),
    )
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = std_pool
    mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)

    # Prefill: 12 real tokens
    real_prompt = list(range(5, 17))
    fi = Req._hsa_insert_lmk_prompt(real_prompt, page_size=page_size, lmk_id=lmk_id)
    pl = len(fi)
    r2t[0, :pl] = tl[:pl]

    ep = torch.tensor([0], device=device, dtype=torch.int32)
    es = torch.tensor([pl], device=device, dtype=torch.int32)
    pos, esl = compute_position(
        "hsa", ep, es, pl, page_size=page_size, enable_landmark_positions=True
    )
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor(fi, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([pl], device=device, dtype=torch.int32),
        out_cache_loc=tl[:pl].to(torch.int64),
        seq_lens_sum=pl,
        seq_lens_cpu=torch.tensor([pl], device="cpu", dtype=torch.int32),
        positions=pos,
        extend_prefix_lens=ep,
        extend_seq_lens=es,
        extend_start_loc=esl,
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[pl],
        req_to_token_pool=mr.req_to_token_pool,
        token_to_kv_pool=std_pool,
        attn_backend=be,
    )
    be.init_forward_metadata(fb)

    with torch.no_grad():
        out = model.model(fb.input_ids, fb.positions, fb)
    if isinstance(out, tuple):
        out = out[0]

    assert not torch.isnan(out).any(), "Prefill output has NaN"
    assert not torch.isinf(out).any(), "Prefill output has Inf"

    # Decode step
    cur_len = pl + 1
    r2t[0, cur_len - 1] = tl[cur_len - 1]
    dp = compute_decode_positions_landmark(
        torch.tensor([cur_len], device=device, dtype=torch.int32),
        page_size=page_size,
    )
    fbd = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=torch.tensor([42], device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([cur_len], device=device, dtype=torch.int32),
        out_cache_loc=tl[cur_len - 1 : cur_len].to(torch.int64),
        seq_lens_sum=cur_len,
        seq_lens_cpu=torch.tensor([cur_len], device="cpu", dtype=torch.int32),
        positions=dp,
        req_to_token_pool=mr.req_to_token_pool,
        token_to_kv_pool=std_pool,
        attn_backend=be,
    )
    be.init_forward_metadata(fbd)

    with torch.no_grad():
        dec_out = model.model(fbd.input_ids, fbd.positions, fbd)
    if isinstance(dec_out, tuple):
        dec_out = dec_out[0]

    assert not torch.isnan(dec_out).any(), "Decode output has NaN"
    assert not torch.isinf(dec_out).any(), "Decode output has Inf"


# ---- Test 5: Memory savings verification ----


def test_swa_pool_is_smaller_than_full_pool():
    """SWA pool should be significantly smaller than full pool."""
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    device = "cuda"
    dtype = torch.bfloat16
    H, D = 4, 64
    page_size = 64

    full_size = 8192  # Full context capacity
    swa_size = 512  # Only need window_size worth

    pool = SWAKVPool(
        size=full_size,
        size_swa=swa_size,
        page_size=page_size,
        dtype=dtype,
        head_num=H,
        head_dim=D,
        swa_attention_layer_ids=[0, 1, 2],
        full_attention_layer_ids=[3],
        enable_kvcache_transpose=False,
        device=device,
    )

    # Full pool: 1 layer * (full_size + page_size) * H * D * 2 (k+v) * 2 bytes (bf16)
    full_kv_bytes = 1 * (full_size + page_size) * H * D * 2 * 2
    # SWA pool: 3 layers * (swa_size + page_size) * H * D * 2 * 2
    swa_kv_bytes = 3 * (swa_size + page_size) * H * D * 2 * 2

    total_with_dual = full_kv_bytes + swa_kv_bytes
    # Without dual pool: all 4 layers at full_size
    total_without_dual = 4 * (full_size + page_size) * H * D * 2 * 2

    savings = 1.0 - (total_with_dual / total_without_dual)
    assert savings > 0.5, f"Expected >50% memory savings, got {savings:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
