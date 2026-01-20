import types

import pytest
import torch

# Patch DP-attention globals before importing backends.
# TritonAttnBackend imports get_attention_tp_size() and expects it to be initialized.
# For unit tests, we force TP size = 1 (single-GPU).
from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test


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
    - Verify KV write succeeds and completed-page repr hook writes into chunk_repr_buffer
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

    # We want seq_len to be a page boundary so repr hook triggers.
    # Pre-existing tokens: 3, decode adds 1 -> total_len=4 (completed).
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

    # Build req_to_token mapping so kv_indices includes all tokens up to total_len.
    # Avoid loc=0 (reserved/padded); start at page_size for convenience.
    token_locs = torch.arange(
        page_size, page_size + total_len, dtype=torch.int32, device=device
    )  # [4,5,6,7]
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

    out_cache_loc = token_locs[-1:].to(torch.int64)  # last token loc, [7]
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

    # Completed-page repr hook should have written to page_id = loc // page_size.
    page_id = int(out_cache_loc.item()) // page_size
    repr_saved = model_runner.token_to_kv_pool.chunk_repr_buffer[0][page_id]
    assert torch.allclose(repr_saved, k_new[0], atol=0, rtol=0)


