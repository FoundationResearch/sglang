"""
Q-batched paged-KV sparse-attention kernel for HSA *extend* (prefill).

R37: BLOCK_M=4 with block-diagonal masking — packs adjacent Q tokens for one
TC matmul per page, wastes 3/4 of the QK FLOPs but TC throughput beats CUDA
cores by enough to win net 1.3x at L=16K.

R38: Also pack all G q-heads of a kv-head into the same program. Since
`selected_page_ids` is [T, H_kv, K] (shared across q-heads within a GQA group),
the page-id loads, K-page loads, and the block-diagonal column structure can
be reused across G q-heads — only the per-q-head `hsa_weights` and the per-
row softmax differ. The QK and PV tl.dot's now run on a Q tile of shape
[BLOCK_M*G, D] = [16, 64] at BLOCK_M=4 G=4, hitting TC's natural 16x16x16
sweet spot, and the grid drops to (T/BLOCK_M, H_kv) — 4x fewer programs
than (T/BLOCK_M, HQ) at G=4.

Output: bf16 [T, HQ, D]. Decode still uses `hsa_decode_paged_fwd_kernel`.
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
    SWA_O_ptr,
    SWA_W_ptr,
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
    BLEND_SWA: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)  # kv-head index now (not q-head)

    G: tl.constexpr = HQ // H
    kv_h = pid_h
    hq_start = kv_h * G  # first q-head index for this kv group

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = offs_m < T

    offs_d = tl.arange(0, D)  # [D]
    offs_c = tl.arange(0, BLOCK_M * PAGE_SIZE)  # [BLOCK_M*PAGE_SIZE]
    col_to_row_m = offs_c // PAGE_SIZE  # [BLOCK_M*PAGE_SIZE] — which Q m-row owns this col
    s_in_page = offs_c % PAGE_SIZE  # [BLOCK_M*PAGE_SIZE]

    # Row index in the packed Q tile: r in [0, BLOCK_M*G).
    # r = m * G + g where m in [0, BLOCK_M), g in [0, G).
    # Page selection is per (m, kv_h), shared across g — so block-diagonal mask
    # is defined by `m` only: row r is valid for col c iff (c // PAGE_SIZE) == (r // G).
    offs_r = tl.arange(0, BLOCK_M * G)  # [BLOCK_M*G]
    row_to_m = offs_r // G  # [BLOCK_M*G]
    row_to_g = offs_r % G  # [BLOCK_M*G]
    bd_mask = col_to_row_m[None, :] == row_to_m[:, None]  # [BLOCK_M*G, BLOCK_M*PAGE_SIZE]

    # Row-validity from mask_m: row r is valid iff offs_m[row_to_m[r]] < T.
    # We'll fold this into the final softmax mask via broadcasting.

    # Load Q [BLOCK_M*G, D]: row r = m*G + g maps to (token offs_m[m], head hq_start+g).
    # Gather offs_m[row_to_m] via broadcast-and-sum on a one-hot identity:
    #   row_tok_flat[r] = sum_m (row_to_m[r] == m) * offs_m[m]
    row_eq_m = (row_to_m[:, None] == tl.arange(0, BLOCK_M)[None, :])  # [BLOCK_M*G, BLOCK_M]
    row_eq_m_i = row_eq_m.to(tl.int64)
    row_tok_flat = tl.sum(row_eq_m_i * offs_m[None, :].to(tl.int64), axis=1)  # [BLOCK_M*G]
    row_mask_flat = tl.sum(row_eq_m_i * mask_m[None, :].to(tl.int64), axis=1) > 0  # [BLOCK_M*G]
    row_hq_flat = hq_start.to(tl.int64) + row_to_g.to(tl.int64)  # [BLOCK_M*G]

    q_offsets = (
        row_tok_flat[:, None] * stride_qb
        + row_hq_flat[:, None] * stride_qh
        + offs_d[None, :].to(tl.int64) * stride_qd
    )  # [BLOCK_M*G, D]
    q = tl.load(Q_ptr + q_offsets, mask=row_mask_flat[:, None], other=0.0)  # bf16 [BLOCK_M*G, D]

    seq_b = tl.load(SEQ_ID_MAP_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)  # [BLOCK_M]

    # bd_mask_i used to broadcast per-m scalars to [BLOCK_M*PAGE_SIZE]
    col_eq_m = (col_to_row_m[:, None] == tl.arange(0, BLOCK_M)[None, :])  # [BLOCK_M*PAGE_SIZE, BLOCK_M]
    col_eq_m_i = col_eq_m.to(tl.int64)

    acc = tl.zeros((BLOCK_M * G, D), dtype=tl.float32)
    ln2_inv = 1.4426950408889634

    for k_i in range(0, TOPK):
        # Per-m page id (shared across G q-heads)
        page_ids = tl.load(
            PAGE_IDS_ptr
            + offs_m.to(tl.int64) * stride_p_b
            + kv_h.to(tl.int64) * stride_p_h
            + k_i * stride_p_k,
            mask=mask_m,
            other=-1,
        ).to(tl.int32)  # [BLOCK_M]
        # Per-(m, g) weight: weights[t, hq, k_i] for hq = hq_start..hq_start+G-1
        weights = tl.load(
            W_ptr
            + row_tok_flat * stride_w_b
            + row_hq_flat * stride_w_hq
            + k_i * stride_w_k,
            mask=row_mask_flat,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_M*G]

        is_valid = page_ids >= 0  # [BLOCK_M]
        page_ids_safe = tl.maximum(page_ids, 0).to(tl.int64)  # [BLOCK_M]

        # Build per-col scalars
        page_id_per_col = tl.sum(col_eq_m_i * page_ids_safe[None, :], axis=1)  # [BLOCK_M*PAGE_SIZE]
        seq_per_col = tl.sum(col_eq_m_i * seq_b[None, :], axis=1)
        valid_per_col = tl.sum(col_eq_m_i * is_valid[None, :].to(tl.int64), axis=1) > 0
        m_valid_per_col = tl.sum(col_eq_m_i * mask_m[None, :].to(tl.int64), axis=1) > 0

        tok = page_id_per_col * PAGE_SIZE + s_in_page.to(tl.int64)
        tok_in_range = tok < MAX_T
        col_load_mask = tok_in_range & valid_per_col & m_valid_per_col

        pt_offsets = seq_per_col * stride_pt_b + tok * stride_pt_t
        token_loc = tl.load(
            PAGE_TABLE_ptr + pt_offsets,
            mask=col_load_mask,
            other=0,
        ).to(tl.int64)

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

        # Q @ K.T — Q is [BLOCK_M*G, D], K is [BLOCK_M*PAGE_SIZE, D]
        # M=16 (BLOCK_M=4, G=4) hits TC sweet spot.
        logits = tl.dot(q, k.trans()) * sm_scale  # [BLOCK_M*G, BLOCK_M*PAGE_SIZE]

        # Mask: block-diagonal in (row_to_m, col_to_row_m); also kill LMK / OOB / invalid pages.
        col_valid_for_softmax = col_load_mask
        if mask_last_token:
            col_valid_for_softmax = col_valid_for_softmax & (s_in_page != (PAGE_SIZE - 1))

        valid_mask = bd_mask & col_valid_for_softmax[None, :]  # [BLOCK_M*G, BLOCK_M*PAGE_SIZE]
        valid_mask = valid_mask & row_mask_flat[:, None]
        logits = tl.where(valid_mask, logits, -float("inf"))

        # Per-row stable softmax
        m_row = tl.max(logits, axis=1)  # [BLOCK_M*G]
        m_row_safe = tl.where(m_row == -float("inf"), 0.0, m_row)
        p = tl.exp2((logits - m_row_safe[:, None]) * ln2_inv)
        p = tl.where(valid_mask, p, 0.0)
        denom = tl.sum(p, axis=1)
        denom_safe = tl.where(denom == 0.0, 1.0, denom)
        p = p / denom_safe[:, None]  # [BLOCK_M*G, BLOCK_M*PAGE_SIZE]

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

        out_page = tl.dot(p.to(v.dtype), v)  # [BLOCK_M*G, D]
        acc = acc + weights[:, None] * out_page.to(tl.float32)

    # R80: optional epilogue — fuse SWA blend
    #   out = bf16( acc + swa_o[t, hq, :] * swa_w[t, hq] )
    # Matches hsa_decode_paged_fwd_kernel's BLEND_SWA epilogue. Caller
    # guarantees swa_o_inner [T, HQ, D] bf16 contiguous and swa_w_q
    # [T, HQ] bf16 contiguous. Saves a standalone addcmul launch per layer.
    if BLEND_SWA:
        # swa_o is contiguous [T, HQ, D] — stride (HQ*D, D, 1).
        swa_o_offsets = (
            row_tok_flat[:, None].to(tl.int64) * (HQ * D)
            + row_hq_flat[:, None].to(tl.int64) * D
            + offs_d[None, :].to(tl.int64)
        )
        swa_o = tl.load(
            SWA_O_ptr + swa_o_offsets,
            mask=row_mask_flat[:, None],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_M*G, D] bf16 -> fp32
        # swa_w is contiguous [T, HQ] — stride (HQ, 1).
        swa_w_offsets = row_tok_flat.to(tl.int64) * HQ + row_hq_flat.to(tl.int64)
        swa_w = tl.load(
            SWA_W_ptr + swa_w_offsets, mask=row_mask_flat, other=0.0
        ).to(tl.float32)  # [BLOCK_M*G]
        acc = acc + swa_o * swa_w[:, None]

    # Store [BLOCK_M*G, D] -> Out[offs_m, hq_start..hq_start+G-1, :]
    out_offsets = (
        row_tok_flat[:, None] * stride_out_b
        + row_hq_flat[:, None] * stride_out_hq
        + offs_d[None, :].to(tl.int64) * stride_out_d
    )
    tl.store(
        OUT_ptr + out_offsets,
        acc.to(tl.bfloat16),
        mask=row_mask_flat[:, None],
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
    block_m: int = 1,
    num_warps: int = 2,
    num_stages: int = 2,
    # R80: optional fused SWA blend epilogue.
    # When provided, computes out = bf16(acc + swa_o_inner * swa_w_q) in one
    # kernel — saves the standalone addcmul launch in forward_extend.
    swa_o_inner: Optional[torch.Tensor] = None,
    swa_w_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Q-batched paged sparse-attention for extend (G-fused, block-diagonal TC).

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

    blend_swa = (swa_o_inner is not None) and (swa_w_q is not None)
    if blend_swa:
        assert swa_o_inner.shape == (T, HQ, D), (
            f"swa_o_inner {tuple(swa_o_inner.shape)} != ({T}, {HQ}, {D})"
        )
        assert swa_w_q.shape == (T, HQ), (
            f"swa_w_q {tuple(swa_w_q.shape)} != ({T}, {HQ})"
        )
        swa_o_ = swa_o_inner if swa_o_inner.is_contiguous() else swa_o_inner.contiguous()
        swa_w_ = swa_w_q if swa_w_q.is_contiguous() else swa_w_q.contiguous()
    else:
        swa_o_ = q_  # dummy pointer (unused under BLEND_SWA=False)
        swa_w_ = q_

    # Grid: (T/BLOCK_M, H_kv) — each program handles BLOCK_M tokens, all G q-heads
    grid = (triton.cdiv(T, block_m), H)
    hsa_extend_paged_fwd_kernel[grid](
        q_,
        k_,
        v_,
        pt_,
        page_ids_,
        w_,
        out,
        seq_map_,
        swa_o_,
        swa_w_,
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
        BLEND_SWA=bool(blend_swa),
        BLOCK_M=int(block_m),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out
