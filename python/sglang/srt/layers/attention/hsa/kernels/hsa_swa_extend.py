"""
Dedicated HSA internal-SWA kernel for extend (prefill).

R39 motivation: at L=16K the SWA branch dominated the R38 profile
(_fwd_kernel_unified took 103 ms, 49% of prefill GPU time). The
shared sglang `_fwd_kernel_unified` was written for full-batch dense
attention — grid (batch, q-head, q-block) reloads K once PER q-head,
so the 4 q-heads in HSA's GQA group of an HSA layer reload the same
K tile 4 times.

This kernel mirrors the sparse R38 G-fusion trick: process all G
q-heads of one kv-head in the same program by packing the Q tile
to [BLOCK_M*G, D]. Grid drops 4x to (T/BLOCK_M, H_kv), K bandwidth
roughly halves, and the M=BLOCK_M*G=128 (BLOCK_M=32 G=4) is squarely
in TC's sweet spot.

Mask logic identical to R36 unified-kernel path:
  - causal: q_pos >= k_pos
  - chunk-aligned SW: k_pos >= max(0, ((q_pos - SW + 1) // ps) * ps)
  - LMK exclusion: (k_pos + 1) % page_size != 0

Outputs:
  out  [T, HQ_hsa, D]    bf16
  lse  [T, HQ_hsa]       fp32  (per-q-head LSE for merged softmax)

Single-batch only (matches SGLang continuous-batching guarantee in
_compute_internal_swa_extend_batched).
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception as e:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = e


def _require_triton():  # pragma: no cover
    if triton is None or tl is None:
        raise ImportError(
            "Triton is required for HSA kernels. Import error: "
            + repr(globals().get("_TRITON_IMPORT_ERROR"))
        )


@triton.jit
def hsa_swa_extend_kernel(
    Q_ptr,          # [T, HQ, D]
    K_ptr,          # [Nloc, H, D] pool
    V_ptr,          # [Nloc, H, D] pool
    KV_IDX_ptr,     # [total_kv] cache slot for each engine pos
    OUT_ptr,        # [T, HQ, D]
    LSE_ptr,        # [T, HQ] fp32
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
    stride_lse_b: tl.constexpr, stride_lse_h: tl.constexpr,
    sm_scale: tl.constexpr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOTAL_KV: tl.constexpr,
    PREFIX_LEN: tl.constexpr,        # absolute Q start offset (Q[i] is at engine pos PREFIX_LEN+i)
    SW: tl.constexpr,                # sliding window size (hsa_window)
    PAGE_SIZE: tl.constexpr,         # also LMK period
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)  # kv-head

    G: tl.constexpr = HQ // H
    kv_h = pid_h
    hq_start = kv_h * G

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M] q-token indices (0-indexed in extend)
    mask_m = offs_m < T

    offs_d = tl.arange(0, D)              # [D]
    offs_n = tl.arange(0, BLOCK_N)        # [BLOCK_N]
    offs_r = tl.arange(0, BLOCK_M * G)    # [BLOCK_M*G] packed-row index
    row_to_m = offs_r // G
    row_to_g = offs_r % G

    # Engine positions for each Q row
    q_abs = PREFIX_LEN + offs_m   # [BLOCK_M]
    # Chunk-aligned SW lower bound for each Q row
    # raw_start = q_abs - SW + 1; chunk_start = (raw_start // PAGE_SIZE) * PAGE_SIZE; clamp >= 0
    raw_starts = q_abs - SW + 1
    chunk_starts = (raw_starts // PAGE_SIZE) * PAGE_SIZE
    chunk_starts = tl.maximum(chunk_starts, 0)  # [BLOCK_M]

    # Build Q [BLOCK_M*G, D]: row r is (m=row_to_m[r], g=row_to_g[r])
    # row_tok_flat[r] = offs_m[row_to_m[r]]
    row_eq_m = (row_to_m[:, None] == tl.arange(0, BLOCK_M)[None, :])  # [BLOCK_M*G, BLOCK_M]
    row_eq_m_i = row_eq_m.to(tl.int64)
    row_tok_flat = tl.sum(row_eq_m_i * offs_m[None, :].to(tl.int64), axis=1)  # [BLOCK_M*G]
    row_mask_flat = tl.sum(row_eq_m_i * mask_m[None, :].to(tl.int64), axis=1) > 0
    row_hq_flat = hq_start.to(tl.int64) + row_to_g.to(tl.int64)
    # Per-row chunk_start and q_abs (broadcast m-scalar via the same trick)
    row_chunk_start = tl.sum(row_eq_m_i * chunk_starts[None, :], axis=1)
    row_q_abs = tl.sum(row_eq_m_i * q_abs[None, :], axis=1)

    q_offsets = (
        row_tok_flat[:, None] * stride_qb
        + row_hq_flat[:, None] * stride_qh
        + offs_d[None, :].to(tl.int64) * stride_qd
    )
    q = tl.load(Q_ptr + q_offsets, mask=row_mask_flat[:, None], other=0.0)  # bf16 [BLOCK_M*G, D]

    # Online softmax accumulators (per packed row)
    acc = tl.zeros((BLOCK_M * G, D), dtype=tl.float32)
    e_max = tl.full((BLOCK_M * G,), -float("inf"), dtype=tl.float32)
    deno = tl.zeros((BLOCK_M * G,), dtype=tl.float32)

    # Per-block bounds:
    # - K lower bound: smallest chunk_start across the block (q_abs is monotone in m,
    #   so chunk_starts is non-decreasing; min = chunk_starts at smallest q_abs).
    # - K upper bound: largest q_abs + 1 (causal), capped to TOTAL_KV.
    # Aligning lower bound DOWN to BLOCK_N keeps the loop sees the same range
    # across all programs that share the tile boundary.
    block_q_min = pid_m * BLOCK_M + PREFIX_LEN
    block_q_max = block_q_min + BLOCK_M - 1
    block_k_lo_raw = block_q_min - SW + 1
    block_k_lo = (block_k_lo_raw // PAGE_SIZE) * PAGE_SIZE
    block_k_lo = tl.maximum(block_k_lo, 0)
    block_k_lo = (block_k_lo // BLOCK_N) * BLOCK_N
    block_k_hi = tl.minimum(block_q_max + 1, TOTAL_KV)
    for start_n in range(block_k_lo, block_k_hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_abs = start_n + offs_n  # [BLOCK_N] engine positions
        mask_n = k_abs < TOTAL_KV

        # Build [BLOCK_M*G, BLOCK_N] mask
        causal = k_abs[None, :] <= row_q_abs[:, None]
        sw_lower = k_abs[None, :] >= row_chunk_start[:, None]
        lmk_ok = ((k_abs + 1) % PAGE_SIZE) != 0  # [BLOCK_N]
        final_mask = (
            causal
            & sw_lower
            & lmk_ok[None, :]
            & mask_n[None, :]
            & row_mask_flat[:, None]
        )

        # Skip empty tiles
        SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0
        if not SKIP_TILE:
            # Gather cache slots
            slot = tl.load(KV_IDX_ptr + k_abs, mask=mask_n, other=0).to(tl.int64)
            k_offsets = (
                slot[:, None] * stride_kb
                + kv_h.to(tl.int64) * stride_kh
                + offs_d[None, :].to(tl.int64) * stride_kd
            )
            k = tl.load(K_ptr + k_offsets, mask=mask_n[:, None], other=0.0)  # bf16 [BLOCK_N, D]

            # Q @ K^T via TC
            qk = tl.dot(q, k.trans()) * sm_scale  # [BLOCK_M*G, BLOCK_N]
            qk = tl.where(final_mask, qk, -float("inf"))

            # Online softmax
            row_max = tl.max(qk, axis=1)  # [BLOCK_M*G]
            row_max_fixed = tl.where(row_max == -float("inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            p = tl.where(final_mask, p, 0.0)
            deno = deno * re_scale + tl.sum(p, axis=1)

            v_offsets = (
                slot[:, None] * stride_vb
                + kv_h.to(tl.int64) * stride_vh
                + offs_d[None, :].to(tl.int64) * stride_vd
            )
            v = tl.load(V_ptr + v_offsets, mask=mask_n[:, None], other=0.0)  # bf16 [BLOCK_N, D]
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            e_max = n_e_max

    # Store output
    out_offsets = (
        row_tok_flat[:, None] * stride_ob
        + row_hq_flat[:, None] * stride_oh
        + offs_d[None, :].to(tl.int64) * stride_od
    )
    deno_safe = tl.where(deno > 0, deno, 1.0)
    out = acc / deno_safe[:, None]
    tl.store(OUT_ptr + out_offsets, out.to(tl.bfloat16), mask=row_mask_flat[:, None])

    # Store LSE = e_max + log(deno).  For rows that never saw a valid k, deno=0 ->
    # downstream merged softmax should treat them as -inf attention contribution,
    # which it does via the masked_fill in merged_w computation.
    lse = e_max + tl.log(tl.where(deno > 0, deno, 1e-20))
    lse_offsets = row_tok_flat * stride_lse_b + row_hq_flat * stride_lse_h
    tl.store(LSE_ptr + lse_offsets, lse, mask=row_mask_flat)


def hsa_swa_extend_fwd(
    *,
    q_hsa: torch.Tensor,           # [T, HQ_hsa, D]
    k_cache_hsa: torch.Tensor,     # [Nloc, H_hsa, D] (pre-sliced view)
    v_cache_hsa: torch.Tensor,     # [Nloc, H_hsa, D]
    kv_indices: torch.Tensor,      # [total_kv] int64 cache slot for engine pos i
    prefix_len: int,
    extend_len: int,
    hsa_window: int,
    page_size: int,
    sm_scale: float,
    block_m: int = 16,
    block_n: int = 64,
):
    """Returns (out, lse) where:
       out: [T, HQ_hsa, D] bf16
       lse: [T, HQ_hsa] fp32  (per-q-head logsumexp for merged softmax with selector)
    """
    _require_triton()
    T, HQ, D = q_hsa.shape
    Nloc, H, Dk = k_cache_hsa.shape
    assert Dk == D
    assert HQ % H == 0
    total_kv = prefix_len + extend_len

    out = torch.empty((T, HQ, D), device=q_hsa.device, dtype=torch.bfloat16)
    lse = torch.full((T, HQ), float("-inf"), device=q_hsa.device, dtype=torch.float32)

    q_ = q_hsa if q_hsa.is_contiguous() else q_hsa.contiguous()
    k_ = k_cache_hsa.contiguous()
    v_ = v_cache_hsa.contiguous()
    idx_ = kv_indices.to(torch.int64) if kv_indices.dtype != torch.int64 else kv_indices
    idx_ = idx_.contiguous()

    grid = (triton.cdiv(T, block_m), H)
    hsa_swa_extend_kernel[grid](
        q_,
        k_,
        v_,
        idx_,
        out,
        lse,
        q_.stride(0), q_.stride(1), q_.stride(2),
        k_.stride(0), k_.stride(1), k_.stride(2),
        v_.stride(0), v_.stride(1), v_.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        sm_scale=float(sm_scale),
        T=T,
        HQ=HQ,
        H=H,
        D=D,
        TOTAL_KV=int(total_kv),
        PREFIX_LEN=int(prefix_len),
        SW=int(hsa_window),
        PAGE_SIZE=int(page_size),
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        num_warps=4,
        num_stages=1,
    )
    return out, lse
