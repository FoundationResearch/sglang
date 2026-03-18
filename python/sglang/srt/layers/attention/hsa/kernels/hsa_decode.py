"""
FlashHSA-style Triton kernels (decode + extend).

This file intentionally mirrors the *shape conventions* of FlashHSA:
- LMK token participates in KV cache, but should be masked from attention reads.
- page_size == chunk_size.

We provide:
1) A real paged-KV kernel (`hsa_decode_paged_fwd`) that gathers KV via `page_table_1`.

Inputs:
  q:              [B, HQ, D] (query for a single decode step per sequence)
                  OR [T, HQ, D] for extend (T = total extend tokens)
  k_cache/v_cache:[Nloc, H, D] (KV cache for a single layer, paged by page_size)
  selected_page_ids: [B, H, K] int32, page ids, padded with -1
  hsa_weights:    [B, HQ, K] float16/float32 (per-q-head weights for each selected page)
  token_to_seq_id: Optional[T] int32 — for extend, maps token index → sequence index.
                   When provided, page_table_1 is indexed by seq_id instead of token index.

Output:
  out:            [B, HQ, D] or [T, HQ, D]

Notes:
- This kernel implements the per-page block attention (within a page) using a
  numerically-stable softmax, then combines K pages using provided weights.
- It masks the last token in each page (LMK slot) via `mask_last_token=True`.
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


### NOTE:
# We intentionally removed the earlier contiguous-page "scaffold" kernel. The only decode kernel
# we keep is the paged-KV version below.


@triton.jit
def hsa_decode_paged_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    PAGE_TABLE_ptr,  # [B, MAX_T] token->slot (token_loc)
    PAGE_IDS_ptr,
    W_ptr,
    OUT_ptr,
    SEQ_ID_MAP_ptr,  # [T] int32, token_idx -> seq_idx (or dummy when USE_SEQ_MAP=False)
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
    B: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_T: tl.constexpr,
    mask_last_token: tl.constexpr,
    USE_SEQ_MAP: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)
    if pid_b >= B or pid_hq >= HQ:
        return

    # For extend: page_table_1 is indexed by seq_id, not token index.
    if USE_SEQ_MAP:
        seq_b = tl.load(SEQ_ID_MAP_ptr + pid_b).to(tl.int64)
    else:
        seq_b = pid_b

    G = HQ // H
    kv_h = pid_hq // G

    # load q
    q_ptr = Q_ptr + pid_b * stride_qb + pid_hq * stride_qh
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + offs_d * stride_qd, mask=True, other=0.0).to(tl.float32)  # [D]

    acc = tl.zeros((D,), dtype=tl.float32)
    ln2_inv = 1.4426950408889634

    # vector of offsets in page
    offs_s = tl.arange(0, PAGE_SIZE)

    for k_i in range(0, TOPK):
        page_id = tl.load(
            PAGE_IDS_ptr + pid_b * stride_p_b + kv_h * stride_p_h + k_i * stride_p_k
        ).to(tl.int32)
        w = tl.load(
            W_ptr + pid_b * stride_w_b + pid_hq * stride_w_hq + k_i * stride_w_k
        ).to(tl.float32)

        is_valid = page_id >= 0
        page_id = tl.maximum(page_id, 0)
        w = tl.where(is_valid, w, 0.0)

        # token indices for this page: [PAGE_SIZE]
        tok = page_id.to(tl.int64) * PAGE_SIZE + offs_s.to(tl.int64)
        tok_in_range = tok < MAX_T

        # token_loc (KV slot id): [PAGE_SIZE]
        # Use seq_b (not pid_b) for page_table_1 lookup — critical for extend.
        token_loc = tl.load(
            PAGE_TABLE_ptr + seq_b * stride_pt_b + tok * stride_pt_t,
            mask=tok_in_range,
            other=0,
        ).to(tl.int64)

        # Load K: [PAGE_SIZE, D]
        k_ptrs = (
            K_ptr
            + token_loc[:, None] * stride_k_loc
            + kv_h.to(tl.int64) * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=tok_in_range[:, None], other=0.0).to(tl.float32)

        logits = tl.sum(k * q[None, :], axis=1) * sm_scale  # [PAGE_SIZE]
        if mask_last_token:
            logits = tl.where(offs_s == (PAGE_SIZE - 1), -float("inf"), logits)

        m = tl.max(logits, axis=0)
        p = tl.exp2((logits - m) * ln2_inv)
        denom = tl.sum(p, axis=0)
        p = p / denom

        # Load V: [PAGE_SIZE, D]
        v_ptrs = (
            V_ptr
            + token_loc[:, None] * stride_v_loc
            + kv_h.to(tl.int64) * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=tok_in_range[:, None], other=0.0).to(tl.float32)

        out_page = tl.sum(v * p[:, None], axis=0)  # [D]
        acc += w * out_page

    out_ptr = OUT_ptr + pid_b * stride_out_b + pid_hq * stride_out_hq
    # FlashHSA alignment: bf16-only output.
    tl.store(out_ptr + offs_d * stride_out_d, acc.to(tl.bfloat16), mask=True)


def hsa_decode_paged_fwd(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    selected_page_ids: torch.Tensor,
    hsa_weights: torch.Tensor,
    page_size: int,
    sm_scale: Optional[float] = None,
    mask_last_token: bool = True,
    out: Optional[torch.Tensor] = None,
    token_to_seq_id: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Paged-KV kernel (FlashHSA semantics) for decode and extend.

    page_table_1: [B_seq, MAX_T] token->slot mapping (token_loc).
    token_to_seq_id: Optional [T] int32 — for extend, maps each token in q's
        first dimension to the sequence index in page_table_1. When None (decode),
        q's first dimension directly indexes page_table_1.
    """
    _require_triton()
    if q.dim() != 3:
        raise ValueError(f"q must be [B/T,HQ,D], got {tuple(q.shape)}")
    if k_cache.dim() != 3 or v_cache.dim() != 3:
        raise ValueError("k_cache/v_cache must be [Nloc,H,D]")
    if page_table_1.dim() != 2:
        raise ValueError("page_table_1 must be [B,MAX_T]")
    if selected_page_ids.dim() != 3:
        raise ValueError("selected_page_ids must be [B/T,H,K]")
    if hsa_weights.dim() != 3:
        raise ValueError("hsa_weights must be [B/T,HQ,K]")

    N, HQ, D = q.shape  # N = B (decode) or T (extend)
    Nloc, H, Dk = k_cache.shape
    assert v_cache.shape == (Nloc, H, Dk)
    assert Dk == D
    MAX_T = int(page_table_1.shape[1])
    assert selected_page_ids.shape[0] == N and selected_page_ids.shape[1] == H
    TOPK = int(selected_page_ids.shape[2])
    assert hsa_weights.shape == (N, HQ, TOPK)
    assert HQ % H == 0
    assert int(page_size) > 0

    use_seq_map = token_to_seq_id is not None
    if use_seq_map:
        assert token_to_seq_id.shape[0] == N, (
            f"token_to_seq_id length {token_to_seq_id.shape[0]} != q batch dim {N}"
        )

    if sm_scale is None:
        sm_scale = float(D) ** -0.5

    q_ = q.contiguous()
    k_ = k_cache.contiguous()
    v_ = v_cache.contiguous()
    pt_ = page_table_1.to(torch.int32).contiguous()
    page_ids_ = selected_page_ids.to(torch.int32).contiguous()
    w_ = hsa_weights.contiguous()

    # For the seq_id_map, use the real tensor or a dummy scalar placeholder.
    if use_seq_map:
        seq_map_ = token_to_seq_id.to(torch.int32).contiguous()
    else:
        seq_map_ = torch.zeros(1, dtype=torch.int32, device=q.device)

    if out is None:
        out = torch.empty((N, HQ, D), device=q.device, dtype=torch.bfloat16)
    else:
        assert out.shape == (N, HQ, D)

    grid = (N, HQ)
    hsa_decode_paged_fwd_kernel[grid](
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
        B=N,
        HQ=HQ,
        H=H,
        D=D,
        TOPK=TOPK,
        PAGE_SIZE=int(page_size),
        MAX_T=MAX_T,
        mask_last_token=bool(mask_last_token),
        USE_SEQ_MAP=use_seq_map,
        num_warps=4,
    )
    return out


