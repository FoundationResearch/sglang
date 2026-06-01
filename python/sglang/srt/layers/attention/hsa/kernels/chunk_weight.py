"""Fused chunk-weight softmax kernels for HSA decode.

Replaces the per-layer sequence

    valid_hq = valid.unsqueeze(2).expand(...).reshape(...)
    scores_hq = per_qhead_scores.masked_fill(~valid_hq, float("-inf"))
    cat_scores = torch.cat([scores_hq, lse_hq.unsqueeze(-1)], dim=-1)         # (+ optional 0)
    merged_w = torch.softmax(cat_scores, dim=-1)
    merged_w = torch.nan_to_num(merged_w, nan=0.0)
    w_q = merged_w[:, :, :TOPK].to(q_hsa.dtype).contiguous()
    swa_w_q = merged_w[:, :, swa_weight_idx]

with a single triton kernel.  Saves ~7 op launches per HSA layer per decode
step, which at 16 layers × hundreds of decode tokens is the dominant remaining
Python/dispatch overhead in `forward_decode` after R9-R12.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_chunk_weight_per_qhead_kernel(
    scores_ptr,           # [B, HQ, TOPK]  float32
    lse_ptr,              # [B, HQ]        bf16 OR float32
    selected_pages_ptr,   # [B, H, TOPK]   int32  (>=0 means valid)
    w_q_ptr,              # [B, HQ, TOPK]  bf16   (output)
    swa_w_q_ptr,          # [B, HQ]        float32 (output)
    HQ: tl.constexpr,
    H: tl.constexpr,
    Gh: tl.constexpr,
    TOPK: tl.constexpr,
    SOFTMAX1: tl.constexpr,
    LSE_IS_BF16: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // HQ
    hq = pid % HQ
    h = hq // Gh

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < TOPK

    # Load per-q-head scores (float32).
    scores = tl.load(
        scores_ptr + (b * HQ + hq) * TOPK + k_offs,
        mask=k_mask,
        other=float("-inf"),
    )

    # Valid mask: derived from selected_page_ids >= 0 (broadcast h_kv -> h_q).
    sel_pages = tl.load(
        selected_pages_ptr + (b * H + h) * TOPK + k_offs,
        mask=k_mask,
        other=-1,
    )
    valid = sel_pages >= 0

    # Mask invalid entries to -inf (softmax kills them).
    scores = tl.where(valid & k_mask, scores, float("-inf"))

    # Load internal-SWA logsumexp (per b, hq) — bf16 or float32.
    if LSE_IS_BF16:
        lse_raw = tl.load(lse_ptr + b * HQ + hq).to(tl.float32)
    else:
        lse_raw = tl.load(lse_ptr + b * HQ + hq)

    # Numerical-stability max across scores, lse, and (if SOFTMAX1) 0.
    scores_max = tl.max(scores, axis=0)
    max_val = tl.maximum(scores_max, lse_raw)
    if SOFTMAX1:
        max_val = tl.maximum(max_val, 0.0)

    exp_scores = tl.where(k_mask, tl.exp(scores - max_val), 0.0)
    exp_lse = tl.exp(lse_raw - max_val)
    total = tl.sum(exp_scores, axis=0) + exp_lse
    if SOFTMAX1:
        total = total + tl.exp(-max_val)

    # Safe divide: total==0 (everything -inf, including lse) -> all weights 0.
    # Equivalent to torch.nan_to_num(softmax, nan=0.0).
    inv_total = tl.where(total > 0.0, 1.0 / total, 0.0)

    w_vals = exp_scores * inv_total
    swa_w_val = exp_lse * inv_total

    tl.store(
        w_q_ptr + (b * HQ + hq) * TOPK + k_offs,
        w_vals.to(tl.bfloat16),
        mask=k_mask,
    )
    tl.store(swa_w_q_ptr + b * HQ + hq, swa_w_val)


def fused_chunk_weight_per_qhead_decode(
    per_qhead_scores: torch.Tensor,
    per_qhead_lse: torch.Tensor,
    selected_page_ids: torch.Tensor,
    Gh: int,
    enable_softmax1: bool,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Fused replacement for the per-q-head chunk-weight softmax block.

    Inputs
    ------
    per_qhead_scores  : [B, HQ_hsa, TOPK] float32  (selector raw per-q-head scores)
    per_qhead_lse     : [B, HQ_hsa]      bf16 OR float32 (internal-SWA LSE per q-head)
    selected_page_ids : [B, H_hsa, TOPK] int32     (-1 marks unused topk slots)
    Gh                : HQ_hsa // H_hsa
    enable_softmax1   : bool                       (adds a +0 logit term)

    Outputs
    -------
    w_q     : [B, HQ_hsa, TOPK] bf16    (chunk weights for hsa_decode_paged_fwd)
    swa_w_q : [B, HQ_hsa]      float32  (SWA blend weight)
    """
    assert per_qhead_scores.dtype == torch.float32, per_qhead_scores.dtype
    assert per_qhead_scores.is_contiguous(), "per_qhead_scores must be contiguous"
    assert selected_page_ids.dtype == torch.int32, selected_page_ids.dtype
    assert selected_page_ids.is_contiguous(), "selected_page_ids must be contiguous"

    B, HQ, TOPK = per_qhead_scores.shape
    H = selected_page_ids.shape[1]
    assert HQ == H * Gh, f"HQ={HQ} != H*Gh={H*Gh}"
    assert selected_page_ids.shape == (B, H, TOPK), (
        f"selected_page_ids {tuple(selected_page_ids.shape)} != (B={B}, H={H}, TOPK={TOPK})"
    )
    assert per_qhead_lse.shape == (B, HQ), (
        f"per_qhead_lse {tuple(per_qhead_lse.shape)} != (B={B}, HQ={HQ})"
    )
    lse_contig = per_qhead_lse.contiguous()

    device = per_qhead_scores.device
    w_q = torch.empty((B, HQ, TOPK), device=device, dtype=out_dtype)
    swa_w_q = torch.empty((B, HQ), device=device, dtype=torch.float32)

    BLOCK_K = max(triton.next_power_of_2(TOPK), 16)

    fused_chunk_weight_per_qhead_kernel[(B * HQ,)](
        per_qhead_scores,
        lse_contig,
        selected_page_ids,
        w_q,
        swa_w_q,
        HQ=HQ,
        H=H,
        Gh=Gh,
        TOPK=TOPK,
        SOFTMAX1=bool(enable_softmax1),
        LSE_IS_BF16=(lse_contig.dtype == torch.bfloat16),
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )

    return w_q, swa_w_q
