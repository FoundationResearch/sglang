import types

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)


class _DummyDenseBackendWritesKV:
    """Dummy backend that mimics TritonAttnBackend's KV write behavior."""

    def __init__(self, model_runner, **_kwargs):
        self.model_runner = model_runner
        self.forward_metadata = types.SimpleNamespace(
            kv_indptr=torch.tensor([0, 0], device=model_runner.device, dtype=torch.int32),
            kv_indices=torch.empty((0,), device=model_runner.device, dtype=torch.int64),
            window_kv_indptr=None,
            window_kv_indices=None,
        )

    def init_forward_metadata(self, forward_batch):
        pass

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        return torch.full_like(q, 7.0)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        return torch.full_like(q, 9.0)


def _make_hsa_backend(monkeypatch, *, page_size: int):
    import sglang.srt.layers.attention.hsa_backend as hsa_backend_mod

    monkeypatch.setattr(hsa_backend_mod, "TritonAttnBackend", _DummyDenseBackendWritesKV)

    server_args = types.SimpleNamespace(
        hsa_topk=64,
        hsa_selection_strategy="head",
        hsa_layers=None,
        hsa_window_size=None,
        hsa_enable_swa_fusion=False,
    )
    model_config = types.SimpleNamespace(is_encoder_decoder=False)
    model_runner = types.SimpleNamespace(
        device=torch.device("cuda"),
        page_size=page_size,
        server_args=server_args,
        model_config=model_config,
    )
    return hsa_backend_mod.HSAAttnBackend(model_runner)


@pytest.mark.parametrize(
    "seq_len,token_loc,should_write",
    [
        (3, 6, False),  # not completed for page_size=4
        (4, 7, True),  # completed for page_size=4
    ],
)
def test_hsa_backend_writes_chunk_repr_on_decode_completion_cuda(
    monkeypatch, seq_len, token_loc, should_write
):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    page_size = 4
    hsa = _make_hsa_backend(monkeypatch, page_size=page_size)

    pool = MHATokenToKVPool(
        size=256,
        page_size=page_size,
        dtype=torch.float16,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )

    layer = types.SimpleNamespace(layer_id=0)
    page_id = token_loc // page_size

    # q is arbitrary; k/v must be shaped for set_kv_buffer.
    q = torch.randn((1, 16), device="cuda", dtype=torch.float16)
    k = torch.full((1, pool.head_num, pool.head_dim), 2.0, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)

    forward_batch = types.SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
        out_cache_loc=torch.tensor([token_loc], device="cuda", dtype=torch.int64),
        token_to_kv_pool=pool,
        extend_seq_lens=None,
        extend_start_loc=None,
        extend_prefix_lens=None,
    )

    out = hsa.forward_decode(q, k, v, layer, forward_batch, save_kv_cache=True)
    assert torch.all(out == 7.0)

    repr_saved = pool.chunk_repr_buffer[0][page_id]
    if should_write:
        assert torch.allclose(repr_saved, k[0])
    else:
        assert torch.count_nonzero(repr_saved).item() == 0


def test_hsa_backend_writes_chunk_repr_on_extend_boundary_cuda(monkeypatch):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    page_size = 4
    hsa = _make_hsa_backend(monkeypatch, page_size=page_size)

    pool = MHATokenToKVPool(
        size=256,
        page_size=page_size,
        dtype=torch.float16,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    layer = types.SimpleNamespace(layer_id=0)

    # One sequence: prefix=3, extend=2. Boundary at t=1 -> prefix+t == 4.
    # We'll set out_cache_loc[0] to a location that maps to page_id=1.
    token_locs = torch.tensor([7, 8], device="cuda", dtype=torch.int64)
    page_id_completed = int(token_locs[0].item()) // page_size

    # Two new tokens to write.
    q = torch.randn((2, 16), device="cuda", dtype=torch.float16)
    k = torch.full((2, pool.head_num, pool.head_dim), 3.0, device="cuda", dtype=torch.float16)
    v = torch.zeros_like(k)

    forward_batch = types.SimpleNamespace(
        batch_size=1,
        seq_lens=torch.tensor([5], device="cuda", dtype=torch.int32),
        out_cache_loc=token_locs,
        token_to_kv_pool=pool,
        extend_prefix_lens=torch.tensor([3], device="cuda", dtype=torch.int32),
        extend_seq_lens=torch.tensor([2], device="cuda", dtype=torch.int32),
        extend_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
    )

    out = hsa.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=True)
    assert torch.all(out == 9.0)

    # Only the first token hits the boundary -> only its page should be written.
    repr_saved = pool.chunk_repr_buffer[0][page_id_completed]
    assert torch.allclose(repr_saved, k[0])


