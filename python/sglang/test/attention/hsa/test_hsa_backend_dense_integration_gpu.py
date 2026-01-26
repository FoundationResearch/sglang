import os
import types

import pytest
import torch

# Patch DP-attention globals before importing backends.
# TritonAttnBackend imports get_attention_tp_size() and expects it to be initialized.
# For unit tests, we force TP size = 1 (single-GPU).
from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test

_HSA_VERBOSE = os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") == "1"


def _vprint(*args):
    if _HSA_VERBOSE:
        print(*args, flush=True)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)


class _MinimalModelConfig:
    def __init__(self, *, context_len: int, num_attention_heads: int, num_kv_heads: int):
        self.context_len = context_len
        self.num_attention_heads = num_attention_heads
        self._num_kv_heads = num_kv_heads
        self.is_encoder_decoder = False

    def get_num_kv_heads(self, _tp_size: int):
        return self._num_kv_heads


class _MinimalServerArgs:
    def __init__(self):
        # Speculative (unused in this test, but required by TritonAttnBackend __init__)
        self.speculative_num_draft_tokens = 0
        self.speculative_num_steps = 0

        # Triton decode config
        self.triton_attention_num_kv_splits = 8
        self.triton_attention_split_tile_size = None
        self.enable_deterministic_inference = False

        # HSA args (read by HSAAttnBackend)
        self.hsa_topk = 64
        self.hsa_selection_strategy = "head"
        self.hsa_layers = None
        self.hsa_window_size = None
        self.hsa_enable_swa_fusion = False


@pytest.mark.skipif(
    torch.cuda.device_count() == 0, reason="CUDA device required for this test"
)
def test_hsa_backend_real_triton_decode_integration_cuda():
    """
    Integration test:
    - Use real TritonAttnBackend (no monkeypatch) under HSAAttnBackend
    - Run init_forward_metadata + forward_decode
    - Verify KV write succeeds and HSA selection (FlashHSA semantics: E_i = K(LMK)) runs without error
      and masks out non-completed pages.
    """
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend

    device = "cuda"
    dtype = torch.float16

    # Keep shapes tiny to reduce compile time.
    batch_size = 1
    num_heads = 2
    head_dim = 8
    page_size = 4

    # We want total_len to be a page boundary so page 0 is "completed" under LMK semantics:
    # completed_pages = floor(seq_len / page_size).
    # Pre-existing tokens: 3, decode adds 1 -> total_len=4 -> completed_pages=1 -> page_id 0 is eligible.
    prefill_len = 3
    decode_len = 1
    total_len = prefill_len + decode_len
    assert total_len % page_size == 0

    max_batch_size = 8
    max_context_len = 64
    max_total_num_tokens = max_batch_size * max_context_len

    # Minimal model_runner that TritonAttnBackend expects.
    model_runner = types.SimpleNamespace()
    model_runner.device = device
    model_runner.dtype = dtype
    model_runner.gpu_id = 0
    model_runner.page_size = page_size
    model_runner.sliding_window_size = None
    model_runner.hybrid_gdn_config = None
    model_runner.kimi_linear_config = None
    model_runner.server_args = _MinimalServerArgs()
    model_runner.model_config = _MinimalModelConfig(
        context_len=max_context_len, num_attention_heads=num_heads, num_kv_heads=num_heads
    )

    # Pools
    req_to_token = torch.zeros(
        (max_batch_size, max_context_len), dtype=torch.int32, device=device
    )
    model_runner.req_to_token_pool = types.SimpleNamespace(
        size=max_batch_size, req_to_token=req_to_token
    )

    model_runner.token_to_kv_pool = MHATokenToKVPool(
        size=max_total_num_tokens,
        page_size=page_size,
        dtype=dtype,
        head_num=num_heads,
        head_dim=head_dim,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )

    # Not used when sliding_window_size is None, but required to exist.
    model_runner.token_to_kv_pool_allocator = object()

    hsa = HSAAttnBackend(model_runner)

    layer = RadixAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        scaling=1.0,
        num_kv_heads=num_heads,
        layer_id=0,
    )

    # Build req_to_token mapping so it spans two pages, but only page 0 is completed.
    # Use locs [1,2,3,4] -> page_ids [0,0,0,1]. LMK for page 0 is loc=3.
    token_locs = torch.arange(1, 1 + total_len, dtype=torch.int32, device=device)  # [1,2,3,4]
    model_runner.req_to_token_pool.req_to_token[0, :total_len] = token_locs

    # Prefill KV cache for first prefill_len tokens.
    prefill_loc = token_locs[:prefill_len].to(torch.int64)
    cache_k = torch.randn(
        (prefill_len, num_heads, head_dim), device=device, dtype=dtype
    )
    cache_v = torch.randn_like(cache_k)
    model_runner.token_to_kv_pool.set_kv_buffer(layer, prefill_loc, cache_k, cache_v)

    # Decode inputs (one token).
    q = torch.randn((batch_size, num_heads * head_dim), device=device, dtype=dtype)
    k_new = torch.full((batch_size, num_heads, head_dim), 3.0, device=device, dtype=dtype)
    v_new = torch.zeros_like(k_new)

    out_cache_loc = token_locs[-1:].to(torch.int64)  # last token loc
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=torch.randint(0, 100, (batch_size, 1), device=device),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([total_len], device=device, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=total_len,
        seq_lens_cpu=torch.tensor([total_len], device="cpu", dtype=torch.int32),
        attn_backend=hsa,
    )
    forward_batch.req_to_token_pool = model_runner.req_to_token_pool
    forward_batch.token_to_kv_pool = model_runner.token_to_kv_pool

    hsa.init_forward_metadata(forward_batch)
    out = hsa.forward_decode(q, k_new, v_new, layer, forward_batch, save_kv_cache=True)
    assert out.is_cuda
    assert out.shape == (batch_size, num_heads * head_dim)

    # Selection should have populated metadata. Under completed-page masking, only page 0 is eligible.
    md = hsa.forward_metadata
    assert md is not None
    assert md.hsa_selected_page_ids is not None
    # [B, H, K] with K=64 by default; first entry should be page 0, and page 1 must not appear.
    selected = md.hsa_selected_page_ids[0, 0].tolist()
    _vprint("### test_hsa_backend_real_triton_decode_integration_cuda")
    _vprint(f"- page_size={page_size} total_len={total_len} completed_pages={total_len // page_size}")
    _vprint(f"- token_locs={token_locs.tolist()}")
    _vprint(f"- selected_page_ids(head0)={selected[:8]} ...")
    _vprint("=> Conclusion: selection ran and excluded non-completed pages under LMK semantics.")
    assert 0 in selected
    assert 1 not in selected


