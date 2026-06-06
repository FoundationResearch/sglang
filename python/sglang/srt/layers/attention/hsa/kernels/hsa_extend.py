"""
Q-batched paged-KV sparse-attention kernel for HSA *extend* (prefill).

The decode kernel `hsa_decode_paged_fwd_kernel` was designed for B=1 query per
program (decode = 1 token). Reusing it for prefill at T=16K issues T*HQ=64K
thread blocks per layer, each doing TOPK small sequential page-attentions on
CUDA cores (no tensor cores possible at 1xD QK).

This kernel packs BLOCK_M adjacent Q tokens into one program and uses the
block-diagonal-masking trick to recover tensor cores: concatenate the BLOCK_M
per-Q K-page loads into a single [BLOCK_M*PAGE_SIZE, D] matrix, do one
TC `tl.dot(Q, K.T) -> [BLOCK_M, BLOCK_M*PAGE_SIZE]`, then mask off the
(BLOCK_M-1)/BLOCK_M off-diagonal-block entries to -inf before the per-row
softmax. Same for `tl.dot(P, V)` on the V side.

The waste is (BLOCK_M-1)/BLOCK_M of the QK and PV FLOPs, but tensor cores are
~16x faster than CUDA-core elementwise, so the net is still ~5x at BLOCK_M=4
and improves further with BLOCK_M=8 (waste 7/8, but TC throughput dominates).

Output: bf16 [T, HQ, D]. Decode keeps the original kernel.
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
def hsa_extend_paged_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    PAGE_TABLE_ptr,
    PAGE_IDS_ptr,
    W_ptr,
    OUT_ptr,
    SEQ_ID_MAP_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_k_loc: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_v_loc: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_pt_b: tl.constexpr,
    stride_pt_t: tl.constexpr,
    stride_p_b: tl.constexpr,
    stride_p_h: tl.constexpr,
    stride_p_k: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_hq: tl.constexpr,
    stride_w_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_hq: tl.constexpr,
    stride_out_d: tl.constexpr,
    sm_scale: tl.constexpr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_T: tl.constexpr,
    mask_last_token: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hq = tl.program_id(1)

    G: tl.constexpr = HQ // H
    kv_h = pid_hq // G

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = offs_m < T

    offs_d = tl.arange(0, D)  # [D]
    offs_c = tl.arange(0, BLOCK_M * PAGE_SIZE)  # [BLOCK_M*PAGE_SIZE]
    col_to_row = offs_c // PAGE_SIZE  # [BLOCK_M*PAGE_SIZE] — which Q-row owns this col
    s_in_page = offs_c % PAGE_SIZE  # [BLOCK_M*PAGE_SIZE]

    # Block-diagonal column-row association mask
    row_ids = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    bd_mask = col_to_row[None, :] == row_ids[:, None]  # [BLOCK_M, BLOCK_M*PAGE_SIZE]

    # Load Q [BLOCK_M, D] (keep bf16 for tl.dot)
    q_offsets = (
        offs_m[:, None].to(tl.int64) * stride_qb
        + pid_hq * stride_qh
        + offs_d[None, :].to(tl.int64) * stride_qd
    )
    q = tl.load(Q_ptr + q_offsets, mask=mask_m[:, None], other=0.0)  # bf16 [BLOCK_M, D]

    seq_b = tl.load(SEQ_ID_MAP_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)  # [BLOCK_M]

    # For broadcasting per-row scalars to [BLOCK_M*PAGE_SIZE] via the bd_mask:
    # value_per_col[c] = sum_m (mask[m, c] * per_row_value[m])
    # since mask is one-hot in m for each c, this picks the right value.
    bd_mask_i = bd_mask.to(tl.int64)

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    ln2_inv = 1.4426950408889634

    for k_i in range(0, TOPK):
        page_ids = tl.load(
            PAGE_IDS_ptr
            + offs_m.to(tl.int64) * stride_p_b
            + kv_h.to(tl.int64) * stride_p_h
            + k_i * stride_p_k,
            mask=mask_m,
            other=-1,
        ).to(tl.int32)  # [BLOCK_M]
        weights = tl.load(
            W_ptr
            + offs_m.to(tl.int64) * stride_w_b
            + pid_hq * stride_w_hq
            + k_i * stride_w_k,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_M]

        is_valid = page_ids >= 0  # [BLOCK_M]
        page_ids_safe = tl.maximum(page_ids, 0).to(tl.int64)  # [BLOCK_M]
        weights_eff = tl.where(is_valid, weights, 0.0)  # [BLOCK_M]

        # Build per-col page_id and seq_b via bd_mask broadcast-and-sum
        page_id_per_col = tl.sum(bd_mask_i * page_ids_safe[:, None], axis=0)  # [BLOCK_M*PAGE_SIZE]
        seq_per_col = tl.sum(bd_mask_i * seq_b[:, None], axis=0)  # [BLOCK_M*PAGE_SIZE]
        valid_per_col = tl.sum(bd_mask_i * is_valid[:, None].to(tl.int64), axis=0) > 0  # [BLOCK_M*PAGE_SIZE]
        row_valid_per_col = tl.sum(bd_mask_i * mask_m[:, None].to(tl.int64), axis=0) > 0

        # tok positions per col, with LMK exclusion mask
        tok = page_id_per_col * PAGE_SIZE + s_in_page.to(tl.int64)  # [BLOCK_M*PAGE_SIZE]
        tok_in_range = tok < MAX_T
        col_load_mask = tok_in_range & valid_per_col & row_valid_per_col

        # token_loc via page_table
        pt_offsets = seq_per_col * stride_pt_b + tok * stride_pt_t
        token_loc = tl.load(
            PAGE_TABLE_ptr + pt_offsets,
            mask=col_load_mask,
            other=0,
        ).to(tl.int64)  # [BLOCK_M*PAGE_SIZE]

        # K [BLOCK_M*PAGE_SIZE, D] — single gather, becomes one tensor for tl.dot
        k_offsets = (
            token_loc[:, None] * stride_k_loc
            + kv_h.to(tl.int64) * stride_kh
            + offs_d[None, :].to(tl.int64) * stride_kd
        )
        k = tl.load(
            K_ptr + k_offsets,
            mask=col_load_mask[:, None],
            other=0.0,
        )  # bf16 [BLOCK_M*PAGE_SIZE, D]

        # Q @ K.T via tl.dot (tensor cores).
        # logits[m, c] = sum_d q[m, d] * k[c, d]
        # For m == col_to_row[c]: valid; otherwise off-diagonal block, must mask.
        logits = tl.dot(q, k.trans()) * sm_scale  # [BLOCK_M, BLOCK_M*PAGE_SIZE]

        # Mask: only block-diagonal entries are real; off-diagonal -> -inf
        # Also kill LMK column and out-of-range / invalid-page columns.
        col_valid_for_softmax = col_load_mask
        if mask_last_token:
            col_valid_for_softmax = col_valid_for_softmax & (s_in_page != (PAGE_SIZE - 1))

        valid_mask = bd_mask & col_valid_for_softmax[None, :]  # [BLOCK_M, BLOCK_M*PAGE_SIZE]
        valid_mask = valid_mask & mask_m[:, None]
        logits = tl.where(valid_mask, logits, -float("inf"))

        # Per-row stable softmax
        m_row = tl.max(logits, axis=1)  # [BLOCK_M]
        m_row_safe = tl.where(m_row == -float("inf"), 0.0, m_row)
        p = tl.exp2((logits - m_row_safe[:, None]) * ln2_inv)
        p = tl.where(valid_mask, p, 0.0)
        denom = tl.sum(p, axis=1)
        denom_safe = tl.where(denom == 0.0, 1.0, denom)
        p = p / denom_safe[:, None]  # [BLOCK_M, BLOCK_M*PAGE_SIZE]

        # V [BLOCK_M*PAGE_SIZE, D]
        v_offsets = (
            token_loc[:, None] * stride_v_loc
            + kv_h.to(tl.int64) * stride_vh
            + offs_d[None, :].to(tl.int64) * stride_vd
        )
        v = tl.load(
            V_ptr + v_offsets,
            mask=col_load_mask[:, None],
            other=0.0,
        )  # bf16 [BLOCK_M*PAGE_SIZE, D]

        # P @ V via tl.dot (tensor cores)
        out_page = tl.dot(p.to(v.dtype), v)  # [BLOCK_M, D]

        # Per-row weighted accumulate
        acc = acc + weights_eff[:, None] * out_page.to(tl.float32)

    out_offsets = (
        offs_m[:, None].to(tl.int64) * stride_out_b
        + pid_hq * stride_out_hq
        + offs_d[None, :].to(tl.int64) * stride_out_d
    )
    tl.store(
        OUT_ptr + out_offsets,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None],
    )


def hsa_extend_paged_fwd(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    selected_page_ids: torch.Tensor,
    hsa_weights: torch.Tensor,
    page_size: int,
    token_to_seq_id: torch.Tensor,
    sm_scale: Optional[float] = None,
    mask_last_token: bool = True,
    out: Optional[torch.Tensor] = None,
    block_m: int = 4,
) -> torch.Tensor:
    """Q-batched paged sparse-attention for extend (block-diagonal TC trick).

    Shapes:
        q                  [T, HQ, D]
        k_cache, v_cache   [Nloc, H, D]
        page_table_1       [B_seq, MAX_T]
        selected_page_ids  [T, H, K]   int32, -1-padded
        hsa_weights        [T, HQ, K]
        token_to_seq_id    [T]         int32
    """
    _require_triton()
    if q.dim() != 3:
        raise ValueError(f"q must be [T, HQ, D], got {tuple(q.shape)}")
    if k_cache.dim() != 3 or v_cache.dim() != 3:
        raise ValueError("k_cache/v_cache must be [Nloc, H, D]")
    if page_table_1.dim() != 2:
        raise ValueError("page_table_1 must be [B, MAX_T]")
    if selected_page_ids.dim() != 3:
        raise ValueError("selected_page_ids must be [T, H, K]")
    if hsa_weights.dim() != 3:
        raise ValueError("hsa_weights must be [T, HQ, K]")
    if token_to_seq_id is None or token_to_seq_id.dim() != 1:
        raise ValueError("token_to_seq_id must be [T]")

    T, HQ, D = q.shape
    Nloc, H, Dk = k_cache.shape
    assert v_cache.shape == (Nloc, H, Dk)
    assert Dk == D
    MAX_T = int(page_table_1.shape[1])
    assert selected_page_ids.shape[0] == T and selected_page_ids.shape[1] == H
    TOPK = int(selected_page_ids.shape[2])
    assert hsa_weights.shape == (T, HQ, TOPK)
    assert HQ % H == 0
    assert int(page_size) > 0
    assert token_to_seq_id.shape[0] == T

    if sm_scale is None:
        sm_scale = float(D) ** -0.5

    q_ = q.contiguous()
    k_ = k_cache.contiguous()
    v_ = v_cache.contiguous()
    pt_ = (
        page_table_1
        if page_table_1.dtype == torch.int32 and page_table_1.is_contiguous()
        else page_table_1.to(torch.int32).contiguous()
    )
    page_ids_ = (
        selected_page_ids
        if selected_page_ids.dtype == torch.int32 and selected_page_ids.is_contiguous()
        else selected_page_ids.to(torch.int32).contiguous()
    )
    w_ = hsa_weights if hsa_weights.is_contiguous() else hsa_weights.contiguous()
    seq_map_ = (
        token_to_seq_id
        if token_to_seq_id.dtype == torch.int32 and token_to_seq_id.is_contiguous()
        else token_to_seq_id.to(torch.int32).contiguous()
    )

    if out is None:
        out = torch.empty((T, HQ, D), device=q.device, dtype=torch.bfloat16)

    grid = (triton.cdiv(T, block_m), HQ)
    hsa_extend_paged_fwd_kernel[grid](
        q_,
        k_,
        v_,
        pt_,
        page_ids_,
        w_,
        out,
        seq_map_,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        pt_.stride(0),
        pt_.stride(1),
        page_ids_.stride(0),
        page_ids_.stride(1),
        page_ids_.stride(2),
        w_.stride(0),
        w_.stride(1),
        w_.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sm_scale=float(sm_scale),
        T=T,
        HQ=HQ,
        H=H,
        D=D,
        TOPK=TOPK,
        PAGE_SIZE=int(page_size),
        MAX_T=MAX_T,
        mask_last_token=bool(mask_last_token),
        BLOCK_M=int(block_m),
        num_warps=8,
        num_stages=2,
    )
    return out
