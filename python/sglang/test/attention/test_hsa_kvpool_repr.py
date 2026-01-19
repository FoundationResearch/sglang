import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this test")
def test_hsa_kvpool_chunk_repr_write_read_and_version_guard_cuda():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = MHATokenToKVPool(
        size=256,
        page_size=16,
        dtype=torch.float16,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )

    layer_id = 0
    page_ids = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64)
    versions = pool.bump_page_version(page_ids)

    repr_in = torch.randn(
        (page_ids.numel(), pool.head_num, pool.head_dim), device="cuda", dtype=torch.float16
    )
    pool.save_chunk_repr(layer_id=layer_id, page_ids=page_ids, repr=repr_in, page_version=versions)

    repr_out_ok = pool.get_chunk_repr(layer_id=layer_id, page_ids=page_ids, page_version=versions)
    assert torch.allclose(repr_out_ok, repr_in)

    repr_out_bad = pool.get_chunk_repr(layer_id=layer_id, page_ids=page_ids, page_version=versions + 1)
    assert torch.count_nonzero(repr_out_bad).item() == 0


