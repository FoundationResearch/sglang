import os

import pytest
import torch


def _vprint(*args):
    if os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") not in ("", "0", "false", "False"):
        print(*args, flush=True)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)


def _torch_ref_decode_paged(
    *,
    q: torch.Tensor,  # [B,HQ,D]
    k_cache: torch.Tensor,  # [Nloc,H,D]
    v_cache: torch.Tensor,  # [Nloc,H,D]
    page_table_1: torch.Tensor,  # [B,MAX_T]
    selected_page_ids: torch.Tensor,  # [B,H,K]
    weights: torch.Tensor,  # [B,HQ,K]
    page_size: int,
    sm_scale: float,
    mask_last_token: bool,
) -> torch.Tensor:
    B, HQ, D = q.shape
    _, H, _ = k_cache.shape
    G = HQ // H
    K = selected_page_ids.shape[2]

    out = torch.zeros((B, HQ, D), device=q.device, dtype=torch.float32)
    for b in range(B):
        for hq in range(HQ):
            kv_h = hq // G
            qq = q[b, hq].float()
            for ki in range(K):
                pid = int(selected_page_ids[b, kv_h, ki].item())
                w = float(weights[b, hq, ki].item())
                if pid < 0 or w == 0.0:
                    continue
                token_start = pid * page_size
                token_end = token_start + page_size
                token_locs = page_table_1[b, token_start:token_end].to(torch.int64)
                k = k_cache[token_locs, kv_h].float()  # [S,D]
                v = v_cache[token_locs, kv_h].float()
                if mask_last_token:
                    k = k[:-1]
                    v = v[:-1]
                logits = (k @ qq) * sm_scale  # [S']
                p = torch.softmax(logits, dim=0)
                out[b, hq] += w * (p @ v)
    return out


def test_flashhsa_triton_hsa_decode_paged_matches_torch_cuda():
    from sglang.srt.layers.attention.hsa.kernels.hsa_decode import hsa_decode_paged_fwd

    device = "cuda"
    torch.manual_seed(0)

    B = 2
    H = 2
    G = 2
    HQ = H * G
    D = 16
    TOPK = 3
    page_size = 8
    num_pages = 12
    max_t = num_pages * page_size
    nloc = 4096

    q = torch.randn((B, HQ, D), device=device, dtype=torch.bfloat16)
    k = torch.randn((nloc, H, D), device=device, dtype=torch.bfloat16)
    v = torch.randn((nloc, H, D), device=device, dtype=torch.bfloat16)

    # Non-contiguous token_loc mapping: each token index maps to a random KV slot.
    page_table_1 = torch.randint(0, nloc, (B, max_t), device=device, dtype=torch.int32)

    selected_page_ids = torch.tensor(
        [
            [[0, 1, -1], [2, 3, -1]],
            [[4, 5, 6], [7, 8, 9]],
        ],
        device=device,
        dtype=torch.int32,
    )
    weights = torch.rand((B, HQ, TOPK), device=device, dtype=torch.bfloat16)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

    sm_scale = float(D) ** -0.5
    out_triton = hsa_decode_paged_fwd(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table_1=page_table_1,
        selected_page_ids=selected_page_ids,
        hsa_weights=weights,
        page_size=page_size,
        sm_scale=sm_scale,
        mask_last_token=True,
    )
    out_ref = _torch_ref_decode_paged(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table_1=page_table_1,
        selected_page_ids=selected_page_ids,
        weights=weights,
        page_size=page_size,
        sm_scale=sm_scale,
        mask_last_token=True,
    )
    # Compare in fp32
    torch.testing.assert_close(out_triton.float(), out_ref, rtol=2e-2, atol=2e-2)
    _vprint("max_abs_err=", (out_triton.float() - out_ref).abs().max().item())


def test_flashhsa_triton_hsa_decode_paged_masks_lmk_slot_cuda():
    from sglang.srt.layers.attention.hsa.kernels.hsa_decode import hsa_decode_paged_fwd

    device = "cuda"
    torch.manual_seed(0)

    B = 1
    H = 1
    G = 2
    HQ = H * G
    D = 16
    TOPK = 1
    page_size = 8
    num_pages = 2
    max_t = num_pages * page_size
    nloc = 2048

    q = torch.randn((B, HQ, D), device=device, dtype=torch.bfloat16)
    k = torch.randn((nloc, H, D), device=device, dtype=torch.bfloat16)
    v = torch.randn((nloc, H, D), device=device, dtype=torch.bfloat16)

    # Map token indices to a fixed KV slot sequence for page 0 so we can poison LMK slot.
    base = torch.randint(0, nloc - page_size, (1,), device=device, dtype=torch.int64).item()
    token_locs_page0 = torch.arange(base, base + page_size, device=device, dtype=torch.int64)
    page_table_1 = torch.randint(0, nloc, (B, max_t), device=device, dtype=torch.int32)
    page_table_1[0, 0:page_size] = token_locs_page0.to(torch.int32)

    # Poison LMK slot (last token in page) with huge V so if it's not masked, output will blow up.
    lmk_loc = int(token_locs_page0[-1].item())
    v[lmk_loc, 0, :] = 10000.0

    selected_page_ids = torch.tensor([[[0]]], device=device, dtype=torch.int32)  # [B,H,K]
    weights = torch.ones((B, HQ, TOPK), device=device, dtype=torch.bfloat16)
    sm_scale = float(D) ** -0.5

    out_masked = hsa_decode_paged_fwd(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table_1=page_table_1,
        selected_page_ids=selected_page_ids,
        hsa_weights=weights,
        page_size=page_size,
        sm_scale=sm_scale,
        mask_last_token=True,
    )
    out_unmasked = hsa_decode_paged_fwd(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table_1=page_table_1,
        selected_page_ids=selected_page_ids,
        hsa_weights=weights,
        page_size=page_size,
        sm_scale=sm_scale,
        mask_last_token=False,
    )

    # Masked output should be finite and not dominated by the poisoned LMK slot.
    assert torch.isfinite(out_masked).all()
    # Unmasked output should be very large in magnitude (at least one element).
    assert (out_unmasked.abs().max() > 100.0).item()

