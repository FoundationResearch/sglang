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


class _FakeKVPool:
    def __init__(self, device: torch.device, nloc: int, h: int, d: int):
        self.k = torch.zeros((nloc, h, d), device=device, dtype=torch.float16)
        self.v = torch.zeros((nloc, h, d), device=device, dtype=torch.float16)
        self.calls = []

    def set_kv_buffer(self, layer, out_cache_loc, k, v):
        # record call only (wiring test)
        self.calls.append(("set_kv_buffer", int(layer.layer_id)))

    def get_key_buffer(self, layer_id: int):
        return self.k

    def get_value_buffer(self, layer_id: int):
        return self.v


def test_hsa_backend_forward_decode_uses_hsa_kernel(monkeypatch):
    import sglang.srt.layers.attention.hsa_backend as hsa_backend_mod

    # Patch kernel launcher to return a deterministic tensor so we can assert wiring.
    def _fake_hsa_decode_paged_fwd(**kwargs):
        q = kwargs["q"]
        return torch.full_like(q, 7.0)

    monkeypatch.setattr(hsa_backend_mod, "hsa_decode_paged_fwd", _fake_hsa_decode_paged_fwd)

    class _DummyDenseBackend:
        def __init__(self, model_runner, **_kwargs):
            self.forward_metadata = types.SimpleNamespace(
                kv_indptr=None,
                kv_indices=None,
                window_kv_indptr=None,
                window_kv_indices=None,
            )

        def init_forward_metadata(self, forward_batch):
            return

        def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
            return torch.full_like(q, 3.0)

        def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
            return torch.full_like(q, 5.0)

        # cuda-graph plumbing not needed here
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

    monkeypatch.setattr(hsa_backend_mod, "TritonAttnBackend", _DummyDenseBackend)

    server_args = types.SimpleNamespace(
        hsa_topk=2,
        hsa_selection_strategy="head",
        hsa_layers="0",
        hsa_window_size=None,
        hsa_enable_swa_fusion=False,
    )
    model_config = types.SimpleNamespace(is_encoder_decoder=False)
    model_runner = types.SimpleNamespace(
        device=torch.device("cuda"),
        page_size=4,
        server_args=server_args,
        model_config=model_config,
        model=types.SimpleNamespace(config=types.SimpleNamespace(hsa_topk=2)),
    )
    backend = hsa_backend_mod.HSAAttnBackend(model_runner)

    # Minimal forward metadata: page_table_1 and selected pages/scores must exist.
    B, HQ, H, D, K = 2, 4, 2, 8, 2
    backend.forward_metadata = types.SimpleNamespace(
        page_size=4,
        cache_seqlens_int32=torch.tensor([4, 4], device="cuda", dtype=torch.int32),
        page_table_1=torch.zeros((B, 4), device="cuda", dtype=torch.int32),
        hsa_selected_page_ids=torch.tensor(
            [[[0, 1], [0, 1]], [[0, 1], [0, 1]]], device="cuda", dtype=torch.int32
        ),
        hsa_selected_scores=torch.zeros((B, H, K), device="cuda", dtype=torch.float32),
    )

    forward_batch = types.SimpleNamespace(
        batch_size=B,
        token_to_kv_pool=_FakeKVPool(torch.device("cuda"), nloc=64, h=H, d=D),
        out_cache_loc=torch.zeros((B,), device="cuda", dtype=torch.int32),
    )
    layer = types.SimpleNamespace(
        layer_id=0,
        tp_q_head_num=HQ,
        tp_k_head_num=H,
        qk_head_dim=D,
        v_head_dim=D,
        scaling=1.0,
    )
    q = torch.randn((B, HQ * D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H * D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H * D), device="cuda", dtype=torch.float16)

    out = backend.forward_decode(q, k, v, layer, forward_batch, save_kv_cache=True)
    assert out.shape == q.shape
    assert torch.all(out == 7.0)
    assert forward_batch.token_to_kv_pool.calls and forward_batch.token_to_kv_pool.calls[0][0] == "set_kv_buffer"

    _vprint("### test_hsa_backend_forward_decode_uses_hsa_kernel")
    _vprint(f"- out_unique={float(out.flatten()[0].item())}")
    _vprint("=> Conclusion: HSAAttnBackend decode path dispatches to hsa_decode_paged_fwd for HSA layers.")


