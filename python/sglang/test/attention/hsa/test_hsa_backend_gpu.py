import os
import types

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)

_HSA_VERBOSE = os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") == "1"


def _vprint(*args):
    if _HSA_VERBOSE:
        print(*args, flush=True)


class _FakeReqToTokenPool:
    def __init__(self, req_to_token: torch.Tensor):
        self.req_to_token = req_to_token


class _DummyDenseBackend:
    """
    CUDA dummy backend to validate HSAAttnBackend wiring.

    It mimics the subset of TritonAttnBackend API that HSAAttnBackend uses, and
    returns deterministic CUDA tensors.
    """

    def __init__(self, model_runner, **_kwargs):
        self.model_runner = model_runner
        self.called_init = False
        self.called_decode = 0
        self.called_extend = 0
        self.forward_metadata = types.SimpleNamespace(
            kv_indptr=torch.tensor([0, 0], device=model_runner.device, dtype=torch.int32),
            kv_indices=torch.empty((0,), device=model_runner.device, dtype=torch.int64),
            window_kv_indptr=None,
            window_kv_indices=None,
        )

    def init_forward_metadata(self, forward_batch):
        # Create some dummy kv_indptr/kv_indices so HSAMetadata carries non-None pointers.
        self.called_init = True
        bs = int(forward_batch.batch_size)
        # pretend each seq has length 1 in this dummy metadata
        self.forward_metadata.kv_indptr = torch.arange(
            0, bs + 1, device=self.model_runner.device, dtype=torch.int32
        )
        self.forward_metadata.kv_indices = torch.zeros(
            (bs,), device=self.model_runner.device, dtype=torch.int64
        )

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        self.called_decode += 1
        # Return a CUDA tensor with a recognizable value
        return torch.full_like(q, 3.0)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        self.called_extend += 1
        return torch.full_like(q, 5.0)

    # CUDA graph plumbing not needed for these smoke tests
    def init_cuda_graph_state(self, *args, **kwargs):
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        raise NotImplementedError()

    def get_verify_buffers_to_fill_after_draft(self):
        raise NotImplementedError()

    def update_verify_buffers_to_fill_after_draft(self, *args, **kwargs):
        raise NotImplementedError()


def _make_minimal_hsa_backend(monkeypatch):
    """
    Construct HSAAttnBackend with TritonAttnBackend monkeypatched to our CUDA dummy.
    """
    import sglang.srt.layers.attention.hsa_backend as hsa_backend_mod

    monkeypatch.setattr(hsa_backend_mod, "TritonAttnBackend", _DummyDenseBackend)

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
        page_size=4,
        server_args=server_args,
        model_config=model_config,
    )
    hsa = hsa_backend_mod.HSAAttnBackend(model_runner)
    return hsa


def test_hsa_backend_init_forward_metadata_cuda(monkeypatch):
    hsa = _make_minimal_hsa_backend(monkeypatch)

    # token_loc table (page_size=1 semantics): shape [B, max_seqlen_k]
    # Use values that imply page_ids [1,5,9] for page_size=4 at positions 0,4,8.
    req_to_token = torch.tensor(
        [
            [4, 5, 6, 7, 20, 21, 22, 23, 36, 37],
            [8, 9, 10, 11, 24, 25, 26, 27, 40, 41],
        ],
        device="cuda",
        dtype=torch.int32,
    )

    forward_batch = types.SimpleNamespace(
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor([10, 7], device="cuda", dtype=torch.int32),
        seq_lens_cpu=None,  # force GPU path in HSAAttnBackend
        req_to_token_pool=_FakeReqToTokenPool(req_to_token),
        spec_info=None,
    )

    hsa.init_forward_metadata(forward_batch)
    assert hsa.forward_metadata is not None
    _vprint("### test_hsa_backend_init_forward_metadata_cuda")
    _vprint(f"- page_table_1.shape={tuple(hsa.forward_metadata.page_table_1.shape)}")
    _vprint(f"- real_page_table[0]={hsa.forward_metadata.real_page_table[0].tolist()}")
    _vprint("=> Conclusion: HSAMetadata tables were constructed and dense metadata pointers exist.")

    # Dense backend init was called
    assert hsa._dense_backend.called_init is True

    # page_table_1 should be sliced to max_seqlen_k computed from GPU seqlens (=10)
    assert hsa.forward_metadata.page_table_1.is_cuda
    assert hsa.forward_metadata.page_table_1.shape == (2, 10)

    # real_page_table should be page_ids at stride positions 0,4,8
    assert hsa.forward_metadata.real_page_table.is_cuda
    assert torch.equal(
        hsa.forward_metadata.real_page_table[0].cpu(),
        torch.tensor([1, 5, 9], dtype=torch.int32),
    )

    # kv_indptr/kv_indices should be non-None from dummy dense backend
    assert hsa.forward_metadata.kv_indptr is not None
    assert hsa.forward_metadata.kv_indices is not None


def test_hsa_backend_forward_delegates_to_dense_cuda(monkeypatch):
    hsa = _make_minimal_hsa_backend(monkeypatch)

    # Minimal forward_batch required by signature (not used by dummy forward)
    forward_batch = types.SimpleNamespace(forward_mode=None)
    layer = types.SimpleNamespace()

    q = torch.randn((4, 128), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = hsa.forward_decode(q, k, v, layer, forward_batch, save_kv_cache=False)
    assert out.is_cuda
    assert torch.all(out == 3.0)
    assert hsa._dense_backend.called_decode == 1

    out2 = hsa.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=False)
    _vprint("### test_hsa_backend_forward_delegates_to_dense_cuda")
    _vprint(f"- decode_out_unique={float(out.flatten()[0].item())} extend_out_unique={float(out2.flatten()[0].item())}")
    _vprint("=> Conclusion: HSAAttnBackend delegates compute to dense backend in Phase-1.")
    assert out2.is_cuda
    assert torch.all(out2 == 5.0)
    assert hsa._dense_backend.called_extend == 1


