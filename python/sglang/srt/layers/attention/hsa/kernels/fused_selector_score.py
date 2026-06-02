"""Fused selector-score kernel: gather K from cache + Q·K + mask.

Replaces the ~7-op chain that precedes torch.topk in the selector's decode
fast path:

    safe_page_ids = cand_page_ids.clamp(min=0).to(torch.int64)
    lmk_token_pos = safe_page_ids * page_size + (page_size - 1)
    lmk_token_pos_safe = lmk_token_pos.clamp(max=...)
    lmk_locs = torch.gather(page_table_1.to(torch.int64), 1, lmk_token_pos_safe)
    flat_lmk_locs = lmk_locs.reshape(-1)
    flat_repr = k_cache[flat_lmk_locs]
    flat_repr = flat_repr[:, kv_head_offset:kv_head_offset+kv_head_count, :]
    cand_repr = flat_repr.view(B, C_max, H_sel, D)
    # GQA reduction (q.sum over G if HQ > H_sel) ...
    scores_d = torch.einsum("bhd,bchd->bhc", q_grouped, cand_repr) * sm_scale
    scores_d = scores_d.masked_fill(~cand_mask.unsqueeze(1), float("-inf"))

with a single streaming triton kernel.  Each program handles one (b, h)
and walks the candidate axis, dereferencing K cache per candidate.

The downstream `torch.topk(scores, K, sorted=False)` (R22) consumes the
output and is left untouched.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_selector_score_kernel(
    q_ptr,                 # [B*H*D]              fp32 (already GQA-summed)
    cand_page_ids_ptr,     # [B*C]                int32 (-1 = invalid)
    cand_mask_ptr,         # [B*C]                bool
    page_table_1_ptr,      # [B*max_seqlen]       int32
    k_cache_ptr,           # [Nloc*H_total*D]     bf16
    scores_out_ptr,        # [B*H*C]              fp32 (output)
    sm_scale,
    H: tl.constexpr,                   # H_sel (kv-head subset for HSA)
    C: tl.constexpr,                   # number of candidate slots (= C_max)
    D: tl.constexpr,
    H_total: tl.constexpr,
    H_offset: tl.constexpr,            # offset into H_total to find HSA kv heads
    max_seqlen: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Grid = (B*H, num_c_chunks).  Each program handles BLOCK_C candidates for
    a single (b, h), so total programs = B*H*num_c_chunks — enough SMs
    parallel for short context."""
    pid_bh = tl.program_id(axis=0)
    c_chunk = tl.program_id(axis=1)
    b = pid_bh // H
    h = pid_bh % H
    h_global = H_offset + h

    d_offs = tl.arange(0, D)
    q = tl.load(q_ptr + (b * H + h) * D + d_offs).to(tl.float32)

    c_offs = c_chunk * BLOCK_C + tl.arange(0, BLOCK_C)
    c_in = c_offs < C

    cand_pid = tl.load(
        cand_page_ids_ptr + b * C + c_offs, mask=c_in, other=-1,
    )
    cand_valid = tl.load(
        cand_mask_ptr + b * C + c_offs, mask=c_in, other=0,
    ).to(tl.int1)
    valid = c_in & cand_valid & (cand_pid >= 0)

    lmk_pos = cand_pid.to(tl.int64) * page_size + (page_size - 1)
    lmk_pos_safe = tl.minimum(tl.maximum(lmk_pos, 0), max_seqlen - 1)

    token_loc = tl.load(
        page_table_1_ptr + b * max_seqlen + lmk_pos_safe,
        mask=valid, other=0,
    ).to(tl.int64)

    kv_off = token_loc[:, None] * H_total * D + h_global * D + d_offs[None, :]
    k_vecs = tl.load(
        k_cache_ptr + kv_off, mask=valid[:, None], other=0.0,
    ).to(tl.float32)

    scores = tl.sum(q[None, :] * k_vecs, axis=1) * sm_scale
    scores = tl.where(valid, scores, float("-inf"))

    tl.store(
        scores_out_ptr + (b * H + h) * C + c_offs,
        scores.to(tl.bfloat16), mask=c_in,
    )


def fused_selector_score_decode(
    *,
    q: torch.Tensor,                # [B, H, D] fp32 OR bf16
    cand_page_ids: torch.Tensor,    # [B, C] int32
    cand_mask: torch.Tensor,        # [B, C] bool
    page_table_1: torch.Tensor,     # [B, max_seqlen] int32
    k_cache_full: torch.Tensor,     # [Nloc, H_total, D] bf16
    H_offset: int,                  # h_global = H_offset + h
    sm_scale: float,
    page_size: int,
):
    """Returns scores [B, H, C] fp32 (with -inf at invalid positions)."""
    assert q.dim() == 3
    B, H, D = q.shape
    C = cand_page_ids.shape[1]
    Nloc, H_total, D2 = k_cache_full.shape
    assert D == D2
    max_seqlen = page_table_1.shape[1]

    q_f32 = q.to(torch.float32).contiguous()
    scores = torch.empty((B, H, C), device=q.device, dtype=torch.bfloat16)

    # Small BLOCK_C → more grid programs → better SM utilisation at short ctx.
    BLOCK_C = min(max(triton.next_power_of_2(C), 16), 32)
    num_chunks = (int(C) + BLOCK_C - 1) // BLOCK_C

    fused_selector_score_kernel[(B * H, num_chunks)](
        q_f32,
        cand_page_ids.contiguous(),
        cand_mask.contiguous(),
        page_table_1.contiguous(),
        k_cache_full.contiguous(),
        scores,
        float(sm_scale),
        H=int(H), C=int(C), D=int(D),
        H_total=int(H_total),
        H_offset=int(H_offset),
        max_seqlen=int(max_seqlen),
        page_size=int(page_size),
        BLOCK_C=int(BLOCK_C),
        num_warps=4,
    )
    return scores
