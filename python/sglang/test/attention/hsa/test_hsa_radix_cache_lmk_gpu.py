import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)


def test_hsa_radix_cache_prefix_match_includes_lmk_and_is_page_aligned_cuda():
    """
    Validate Radix prefix cache behavior under FlashHSA LMK semantics:
    - LMK is inserted into engine-visible token ids (fill_ids)
    - RadixCache stores/matches prefixes using fill_ids (including LMK)
    - For page_size>1, match is truncated to a multiple of page_size
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.sampling.sampling_params import SamplingParams

    device = "cuda"
    page_size = 4
    lmk_id = 1000

    # Minimal pools.
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=64, device=device, enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=1024,
        page_size=page_size,
        dtype=torch.float16,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=1024, dtype=torch.int64, device=device, kvcache=kv_pool, need_sort=False
    )

    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            enable_kv_cache_events=False,
        )
    )

    # Request 1: build engine-visible fill_ids with LMK inserted.
    origin = [1, 2, 3, 4, 5, 6, 7]  # 7 real tokens
    req1 = Req("r1", "", origin, SamplingParams(max_new_tokens=16))
    req1.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)
    req1.req_pool_idx = req_to_token_pool.alloc(1)[0]
    req1.init_next_round_input(cache)  # sets fill_ids and (empty) prefix match state

    assert req1.fill_ids == [1, 2, 3, lmk_id, 4, 5, 6, lmk_id, 7]
    # Allocate token slots and populate req_to_token_pool for all engine-visible tokens.
    kv_slots1 = allocator.alloc(len(req1.fill_ids))
    assert kv_slots1 is not None
    req_to_token_pool.req_to_token[req1.req_pool_idx, : len(req1.fill_ids)] = kv_slots1.to(
        torch.int32
    )

    # Cache the unfinished request (stores page-aligned prefix based on fill_ids).
    cache.cache_unfinished_req(req1)

    # Request 2: same prompt, expect prefix hit (page-aligned) including LMK tokens.
    req2 = Req("r2", "", origin, SamplingParams(max_new_tokens=16))
    req2.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)
    req2.req_pool_idx = req_to_token_pool.alloc(1)[0]
    req2.init_next_round_input(cache)

    # max_prefix_len in init_next_round_input is input_len-1 => 8, already page-aligned.
    expected_tokens = [1, 2, 3, lmk_id, 4, 5, 6, lmk_id]
    assert req2.fill_ids[: len(expected_tokens)] == expected_tokens

    expected_kv = kv_slots1[: len(expected_tokens)].to(torch.int64)
    assert req2.prefix_indices.numel() == expected_kv.numel()
    assert torch.equal(req2.prefix_indices, expected_kv)


