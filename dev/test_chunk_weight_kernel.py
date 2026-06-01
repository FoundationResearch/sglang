"""Correctness check: fused_chunk_weight_per_qhead_decode vs torch reference."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from sglang.srt.layers.attention.hsa.kernels.chunk_weight import (
    fused_chunk_weight_per_qhead_decode,
)


def torch_reference(per_qhead_scores, per_qhead_lse, selected_page_ids, Gh, enable_softmax1, out_dtype):
    B, HQ, TOPK = per_qhead_scores.shape
    H = selected_page_ids.shape[1]
    valid = selected_page_ids >= 0  # [B, H, TOPK]
    valid_hq = valid.unsqueeze(2).expand(B, H, Gh, TOPK).reshape(B, HQ, TOPK)
    scores_hq = per_qhead_scores.masked_fill(~valid_hq, float("-inf"))
    if not enable_softmax1:
        cat_scores = torch.cat([scores_hq, per_qhead_lse.unsqueeze(-1)], dim=-1)
        swa_weight_idx = -1
    else:
        cat_scores = torch.cat(
            [
                scores_hq,
                per_qhead_lse.unsqueeze(-1),
                torch.zeros(B, HQ, 1, device=scores_hq.device, dtype=scores_hq.dtype),
            ],
            dim=-1,
        )
        swa_weight_idx = -2
    merged_w = torch.softmax(cat_scores, dim=-1)
    merged_w = torch.nan_to_num(merged_w, nan=0.0)
    w_q = merged_w[:, :, :TOPK].to(out_dtype).contiguous()
    swa_w_q = merged_w[:, :, swa_weight_idx]
    return w_q, swa_w_q


def run_case(B, H, Gh, TOPK, softmax1, seed):
    torch.manual_seed(seed)
    device = "cuda"
    HQ = H * Gh

    scores = torch.randn(B, HQ, TOPK, device=device, dtype=torch.float32)
    lse = torch.randn(B, HQ, device=device, dtype=torch.bfloat16)
    sel = torch.randint(-1, 100, (B, H, TOPK), device=device, dtype=torch.int32)

    w_q_ref, swa_ref = torch_reference(scores, lse, sel, Gh, softmax1, torch.bfloat16)
    w_q_new, swa_new = fused_chunk_weight_per_qhead_decode(
        per_qhead_scores=scores,
        per_qhead_lse=lse,
        selected_page_ids=sel,
        Gh=Gh,
        enable_softmax1=softmax1,
        out_dtype=torch.bfloat16,
    )

    w_diff = (w_q_new.float() - w_q_ref.float()).abs().max().item()
    swa_diff = (swa_new - swa_ref.float()).abs().max().item()
    print(f"B={B} H={H} Gh={Gh} TOPK={TOPK} softmax1={softmax1} seed={seed} "
          f"w_max_diff={w_diff:.2e}  swa_max_diff={swa_diff:.2e}")
    # bf16 has 2^-7 ≈ 8e-3 ULP at scale-1
    assert w_diff < 1e-2, f"w_q diverged: {w_diff}"
    assert swa_diff < 1e-3, f"swa_w_q diverged: {swa_diff}"


def run_all_invalid_case():
    """Edge: all candidates invalid (-1). Output should be all zeros + swa_w = 1."""
    torch.manual_seed(0)
    device = "cuda"
    B, H, Gh, TOPK = 1, 2, 8, 32
    HQ = H * Gh
    scores = torch.randn(B, HQ, TOPK, device=device, dtype=torch.float32)
    lse = torch.zeros(B, HQ, device=device, dtype=torch.bfloat16)
    sel = torch.full((B, H, TOPK), -1, device=device, dtype=torch.int32)
    w_q_new, swa_new = fused_chunk_weight_per_qhead_decode(
        per_qhead_scores=scores, per_qhead_lse=lse, selected_page_ids=sel,
        Gh=Gh, enable_softmax1=False, out_dtype=torch.bfloat16,
    )
    assert (w_q_new.float() == 0).all(), "All-invalid should give zero weights"
    # swa_w = exp(0 - 0) / (0 + exp(0)) = 1.0
    assert (swa_new - 1.0).abs().max() < 1e-5, f"swa_w should be 1.0, got {swa_new}"
    print("All-invalid edge case OK.")


if __name__ == "__main__":
    print("=== Random parametric cases ===")
    for seed in range(3):
        for softmax1 in (False, True):
            run_case(B=1, H=2, Gh=8, TOPK=32, softmax1=softmax1, seed=seed)
            run_case(B=2, H=4, Gh=4, TOPK=64, softmax1=softmax1, seed=seed)
    print("=== Edge cases ===")
    run_all_invalid_case()
    print("ALL PASS")
