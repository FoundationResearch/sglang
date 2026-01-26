import os

import pytest
import torch


def _vprint(msg: str):
    if os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") not in ("", "0", "false", "False"):
        print(msg, flush=True)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)


def test_flashhsa_triton_hsa_decode_kernel_scaffold_cuda():
    # This is a *smoke test* for the Triton kernel scaffold:
    # - compiles on first use
    # - produces finite outputs with correct shape
    from sglang.srt.layers.attention.hsa.kernels.hsa_decode import hsa_decode_fwd

    device = "cuda"
    B = 2
    H = 2
    G = 2
    HQ = H * G
    D = 16
    TOPK = 3
    page_size = 8

    # Create a toy contiguous KV cache where page_id maps to contiguous slots.
    num_pages = 16
    nloc = num_pages * page_size
    q = torch.randn((B, HQ, D), device=device, dtype=torch.float16)
    k = torch.randn((nloc, H, D), device=device, dtype=torch.float16)
    v = torch.randn((nloc, H, D), device=device, dtype=torch.float16)

    selected_page_ids = torch.tensor(
        [
            [[0, 1, -1], [2, 3, -1]],
            [[4, 5, 6], [7, 8, 9]],
        ],
        device=device,
        dtype=torch.int32,
    )
    # weights per q head (already normalized externally in the future)
    w = torch.rand((B, HQ, TOPK), device=device, dtype=torch.float16)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

    out = hsa_decode_fwd(
        q=q,
        k_cache=k,
        v_cache=v,
        selected_page_ids=selected_page_ids,
        hsa_weights=w,
        page_size=page_size,
        mask_last_token=True,
    )
    assert out.shape == (B, HQ, D)
    assert torch.isfinite(out).all()
    _vprint(f"out[0,0,:4]={out[0,0,:4].float().tolist()}")


