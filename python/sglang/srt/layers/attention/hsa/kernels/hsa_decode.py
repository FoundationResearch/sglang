"""
FlashHSA-style Triton kernel scaffold (decode).

This file intentionally mirrors the *shape conventions* of FlashHSA:
- LMK token participates in KV cache, but should be masked from attention reads.
- page_size == chunk_size.

We provide a minimal, correct-by-shape decode kernel that you can extend into the
full fused SWA/HSA kernel later.

Inputs:
  q:              [B, HQ, D] (query for a single decode step per sequence)
  k_cache/v_cache:[Nloc, H, D] (KV cache for a single layer, paged by page_size)
  selected_page_ids: [B, H, K] int32, page ids (not token locs), padded with -1
  hsa_weights:    [B, HQ, K] float16/float32 (per-q-head weights for each selected page)

Output:
  out:            [B, HQ, D]

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


@triton.jit
def _softmax_block_attn_1q(
    q: tl.tensor,  # [D]
    k_ptr,  # [PAGE_SIZE, D]
    v_ptr,  # [PAGE_SIZE, D]
    sm_scale: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    mask_last_token: tl.constexpr,
):
    # Compute logits for one query over a single page (PAGE_SIZE tokens).
    offs_s = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, D)

    # k: [PAGE_SIZE, D]
    k = tl.load(k_ptr + offs_s[:, None] * D + offs_d[None, :], mask=True, other=0.0)
    # logits: [PAGE_SIZE]
    logits = tl.sum(k * q[None, :], axis=1) * sm_scale

    if mask_last_token:
        logits = tl.where(offs_s == (PAGE_SIZE - 1), -float("inf"), logits)

    # stable softmax via exp2
    m = tl.max(logits, axis=0)
    p = tl.exp2((logits - m) * 1.4426950408889634)  # 1/log(2)
    denom = tl.sum(p, axis=0)
    p = p / denom

    v = tl.load(v_ptr + offs_s[:, None] * D + offs_d[None, :], mask=True, other=0.0)
    # out: [D]
    out = tl.sum(v * p[:, None], axis=0)
    return out


@triton.jit
def hsa_decode_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    PAGE_IDS_ptr,
    W_ptr,
    OUT_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_k_loc: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_v_loc: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
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
    mask_last_token: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)

    if pid_b >= B or pid_hq >= HQ:
        return

    # Map q head -> kv head (GQA)
    G = HQ // H
    kv_h = pid_hq // G

    q_ptr = Q_ptr + pid_b * stride_qb + pid_hq * stride_qh
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + offs_d * stride_qd, mask=True, other=0.0).to(tl.float32)

    acc = tl.zeros((D,), dtype=tl.float32)
    for k_i in range(0, TOPK):
        page_id = tl.load(
            PAGE_IDS_ptr + pid_b * stride_p_b + kv_h * stride_p_h + k_i * stride_p_k
        ).to(tl.int32)
        w = tl.load(W_ptr + pid_b * stride_w_b + pid_hq * stride_w_hq + k_i * stride_w_k).to(
            tl.float32
        )
        # Triton doesn't support `continue`; use masked computation.
        is_valid = page_id >= 0
        page_id = tl.maximum(page_id, 0)
        w = tl.where(is_valid, w, 0.0)

        base_loc = page_id.to(tl.int64) * PAGE_SIZE
        k_base = K_ptr + base_loc * stride_k_loc + kv_h * stride_kh
        v_base = V_ptr + base_loc * stride_v_loc + kv_h * stride_vh

        # Treat each page as a contiguous block of PAGE_SIZE KV slots.
        # NOTE: This assumes page_id maps to contiguous KV locations. In SGLang, token_loc is the
        # KV slot id; for the upcoming paged-HSA kernel we will pass a real page_table mapping.
        #
        # For now this is a scaffold: the caller should ensure page_id corresponds to a contiguous
        # KV region of length PAGE_SIZE in the KV cache.
        out_page = _softmax_block_attn_1q(
            q,
            k_base,
            v_base,
            sm_scale=sm_scale,
            PAGE_SIZE=PAGE_SIZE,
            D=D,
            mask_last_token=mask_last_token,
        )
        acc += w * out_page

    out_ptr = OUT_ptr + pid_b * stride_out_b + pid_hq * stride_out_hq
    tl.store(out_ptr + offs_d * stride_out_d, acc.to(tl.float16), mask=True)


def hsa_decode_fwd(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_page_ids: torch.Tensor,
    hsa_weights: torch.Tensor,
    page_size: int,
    sm_scale: Optional[float] = None,
    mask_last_token: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Python wrapper for the decode kernel scaffold."""
    _require_triton()
    if q.dim() != 3:
        raise ValueError(f"q must be [B,HQ,D], got {tuple(q.shape)}")
    if k_cache.dim() != 3 or v_cache.dim() != 3:
        raise ValueError("k_cache/v_cache must be [Nloc,H,D]")
    if selected_page_ids.dim() != 3:
        raise ValueError("selected_page_ids must be [B,H,K]")
    if hsa_weights.dim() != 3:
        raise ValueError("hsa_weights must be [B,HQ,K]")

    B, HQ, D = q.shape
    Nloc, H, Dk = k_cache.shape
    assert v_cache.shape == (Nloc, H, Dk)
    assert Dk == D, "head dim mismatch"
    assert selected_page_ids.shape[0] == B and selected_page_ids.shape[1] == H
    TOPK = int(selected_page_ids.shape[2])
    assert hsa_weights.shape == (B, HQ, TOPK)
    assert HQ % H == 0, "GQA requires HQ % H == 0"
    assert int(page_size) > 0

    if sm_scale is None:
        sm_scale = float(D) ** -0.5

    q_ = q.contiguous()
    k_ = k_cache.contiguous()
    v_ = v_cache.contiguous()
    page_ids_ = selected_page_ids.to(torch.int32).contiguous()
    w_ = hsa_weights.contiguous()

    if out is None:
        out = torch.empty((B, HQ, D), device=q.device, dtype=torch.float16)
    else:
        assert out.shape == (B, HQ, D)

    grid = (B, HQ)
    hsa_decode_fwd_kernel[grid](
        q_,
        k_,
        v_,
        page_ids_,
        w_,
        out,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
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
        B=B,
        HQ=HQ,
        H=H,
        D=D,
        TOPK=TOPK,
        PAGE_SIZE=int(page_size),
        mask_last_token=bool(mask_last_token),
        num_warps=4,
    )
    return out


