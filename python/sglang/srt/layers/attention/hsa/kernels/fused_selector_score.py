"""Fused selector-score kernel: gather K from cache + Q·K + mask.

R26 update — folds the per-layer `cand_page_ids` / `cand_mask` materialisation
into the kernel by computing them inline from `effective_cands[b]`.  The
upstream q.sum (GQA reduction) and bf16-output choice are kept identical to
R24 — switching either inside the kernel regressed perf (q.sum unroll
serialised gmem loads; fp32 scores knocked sbtopk off its bf16 specialised
template).

Replaces this PyTorch chain that ran every layer:

    cand_range = torch.arange(C_max)
    cand_page_ids = cand_range.expand(B, C_max)
    cand_mask = cand_page_ids < effective_cands
    cand_page_ids = cand_page_ids.masked_fill(~cand_mask, -1)
    # ... plus the existing gather+matmul+mask the kernel already did ...

The downstream `torch.topk(scores, K, sorted=False)` (R22) consumes the
output and is left untouched.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_selector_score_kernel(
    q_ptr,                 # [B*HQ*D]             bf16 — raw q (R30 sums over G inline)
    cache_seqlens_ptr,     # [B]                  int32 — per-request KV length
    page_table_1_ptr,      # [B*max_seqlen]       int32
    k_cache_ptr,           # [Nloc*H_total*D]     bf16
    scores_out_ptr,        # [B*H_sel*C]          bf16 (output)
    sm_scale,
    HQ: tl.constexpr,                  # query heads per request (H_sel * G)
    H_sel: tl.constexpr,               # kv-head subset for HSA
    G: tl.constexpr,                   # GQA group size (HQ // H_sel)
    C: tl.constexpr,                   # number of candidate slots (= C_max)
    D: tl.constexpr,
    H_total: tl.constexpr,
    H_offset: tl.constexpr,            # offset into H_total to find HSA kv heads
    max_seqlen: tl.constexpr,
    page_size: tl.constexpr,
    hsa_window: tl.constexpr,          # 0 = no window exclusion
    BLOCK_C: tl.constexpr,
):
    """Grid = (B*H_sel, num_c_chunks).  Each program handles BLOCK_C candidates
    for a single (b, h_sel).  Total programs = B*H_sel*num_c_chunks.

    R29: effective_cands is computed inline from cache_seqlens.
    R30: q.sum over GQA group computed inline via 2D load — saves the host-side
    `q.view(B, H_sel, G, D).sum(dim=2)` (~60µs/step in aten::sum at 16K).
    """
    pid_bh = tl.program_id(axis=0)
    c_chunk = tl.program_id(axis=1)
    b = pid_bh // H_sel
    h = pid_bh % H_sel

    # R30: 2D load [G, D] of q for this (b, h_sel) group, then sum over G.
    # Single coalesced load — earlier static_range unrolled-load attempt
    # regressed (likely register pressure / serialised loads).
    g_offs = h * G + tl.arange(0, G)        # [G]
    d_offs = tl.arange(0, D)                # [D]
    q_2d = tl.load(
        q_ptr + b * HQ * D + g_offs[:, None] * D + d_offs[None, :]
    )                                       # [G, D] bf16
    q = tl.sum(q_2d.to(tl.float32), axis=0)  # [D] fp32 (GQA-summed)

    # R26+R29: compute effective_cands inline from cache_seqlens.
    seqlen = tl.load(cache_seqlens_ptr + b)
    completed_pages = seqlen // page_size
    if hsa_window > 0:
        # query_pos = seqlen - 1; limit_chunk = (query_pos - hsa_window + 1) // page_size
        limit_raw = (seqlen - hsa_window) // page_size
        limit_chunk = tl.maximum(limit_raw, 0)
        eff = tl.minimum(completed_pages, limit_chunk)
    else:
        eff = completed_pages
    eff = tl.maximum(eff, 0)

    c_offs = c_chunk * BLOCK_C + tl.arange(0, BLOCK_C)
    c_in = c_offs < C
    valid = c_in & (c_offs < eff)

    cand_pid = c_offs.to(tl.int64)
    lmk_pos = cand_pid * page_size + (page_size - 1)
    lmk_pos_safe = tl.minimum(tl.maximum(lmk_pos, 0), max_seqlen - 1)

    token_loc = tl.load(
        page_table_1_ptr + b * max_seqlen + lmk_pos_safe,
        mask=valid, other=0,
    ).to(tl.int64)

    h_global = H_offset + h
    kv_off = token_loc[:, None] * H_total * D + h_global * D + d_offs[None, :]
    k_vecs = tl.load(
        k_cache_ptr + kv_off, mask=valid[:, None], other=0.0,
    ).to(tl.float32)

    scores = tl.sum(q[None, :] * k_vecs, axis=1) * sm_scale
    scores = tl.where(valid, scores, float("-inf"))

    tl.store(
        scores_out_ptr + (b * H_sel + h) * C + c_offs,
        scores.to(tl.bfloat16), mask=c_in,
    )


def fused_selector_score_decode(
    *,
    q: torch.Tensor,                # [B, HQ, D] bf16 — raw q BEFORE GQA sum (R30)
    H_sel: int,                     # number of HSA kv-heads (= HQ // G)
    cache_seqlens: torch.Tensor,    # [B] int32 — per-request KV length (R29)
    C_max: int,                     # candidate slot count
    page_table_1: torch.Tensor,     # [B, max_seqlen] int32
    k_cache_full: torch.Tensor,     # [Nloc, H_total, D] bf16
    H_offset: int,                  # h_global = H_offset + h
    sm_scale: float,
    page_size: int,
    hsa_window: int = 0,            # R29: window-exclusion computed in kernel
):
    """Returns scores [B, H_sel, C_max] bf16 (with -inf at invalid positions)."""
    if q.dim() == 2:
        B = q.shape[0]
        HQD = int(q.shape[1])
        D_q = HQD // ((HQD // (q.shape[1] // 1)) if False else 1)  # placeholder
    assert q.dim() == 3
    B, HQ, D = q.shape
    assert HQ % H_sel == 0, f"HQ={HQ} not divisible by H_sel={H_sel}"
    G = HQ // H_sel
    Nloc, H_total, D2 = k_cache_full.shape
    assert D == D2
    max_seqlen = page_table_1.shape[1]

    q_in = q.contiguous()
    cache_seqlens_i32 = cache_seqlens.contiguous()
    scores = torch.empty((B, H_sel, C_max), device=q.device, dtype=torch.bfloat16)

    BLOCK_C = min(max(triton.next_power_of_2(C_max), 16), 32)
    num_chunks = (int(C_max) + BLOCK_C - 1) // BLOCK_C

    fused_selector_score_kernel[(B * H_sel, num_chunks)](
        q_in,
        cache_seqlens_i32,
        page_table_1.contiguous(),
        k_cache_full.contiguous(),
        scores,
        float(sm_scale),
        HQ=int(HQ), H_sel=int(H_sel), G=int(G),
        C=int(C_max), D=int(D),
        H_total=int(H_total),
        H_offset=int(H_offset),
        max_seqlen=int(max_seqlen),
        page_size=int(page_size),
        hsa_window=int(hsa_window),
        BLOCK_C=int(BLOCK_C),
        num_warps=4,
    )
    return scores
