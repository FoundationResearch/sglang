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
    W_ptr,            # [B, HQ, K] precomputed weights (used when not INLINE_WEIGHTS)
    SCORES_ptr,       # [B, H, K]  selected_scores (used when INLINE_WEIGHTS)
    LSE_ptr,          # [B, HQ]    lse_hq          (used when INLINE_WEIGHTS)
    OUT_ptr,
    SEQ_ID_MAP_ptr,  # [T] int32, token_idx -> seq_idx (or dummy when USE_SEQ_MAP=False)
    # R25: optional SWA blend inputs (pointers; None at Python side when blend disabled)
    SWA_O_ptr,        # [B, HQ, D] fp32  (internal-SWA output) — only used when BLEND_SWA
    SWA_W_ptr,        # [B, HQ]    fp32  (internal-SWA blend weight per q-head; used when not INLINE_WEIGHTS)
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
    BLEND_SWA: tl.constexpr,
    # R86: fused chunk-weight + page-aggregate path.  Eliminates the
    # separate `fused_chunk_weight_h_kv_kernel` launch per layer (16 layers
    # of HSA-345M ⇒ ~16 kernel launches/step saved under CG).
    INLINE_WEIGHTS: tl.constexpr,
    SOFTMAX1: tl.constexpr,
    SCORES_IS_BF16: tl.constexpr,
    LSE_IS_BF16: tl.constexpr,
    Gh: tl.constexpr,
    BLOCK_K: tl.constexpr,  # power-of-two ≥ TOPK; used by INLINE_WEIGHTS path only
    # R87: split-K parallelism — grid is (B, HQ, SPLIT_K) instead of (B, HQ).
    # Each program handles a slice of TOPK pages.  When SPLIT_K > 1 the kernel
    # writes partial fp32 [D] outputs to OUT_ptr at granularity [B, HQ, SPLIT_K]
    # and a separate reduction kernel sums them + applies the SWA blend.
    # When SPLIT_K == 1 the kernel behaves like the original single-program path.
    SPLIT_K: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)
    pid_split = tl.program_id(2)
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

    # ---- R86: INLINE chunk-weight softmax ----
    # Computes the per-(b, hq) page weights w[K] and swa_w blend weight in registers
    # from raw selected_scores + lse_hq.  Mirrors fused_chunk_weight_h_kv_kernel.
    # NOTE under SPLIT_K > 1: every split computes the FULL softmax over all K
    # scores so it can correctly extract its assigned weights.  The softmax is
    # tiny (K=32) so the redundant compute is negligible.
    if INLINE_WEIGHTS:
        k_offs_w = tl.arange(0, BLOCK_K)
        k_mask_w = k_offs_w < TOPK
        if SCORES_IS_BF16:
            scores_vec = tl.load(
                SCORES_ptr + (pid_b * H + kv_h) * TOPK + k_offs_w,
                mask=k_mask_w, other=float("-inf"),
            ).to(tl.float32)
        else:
            scores_vec = tl.load(
                SCORES_ptr + (pid_b * H + kv_h) * TOPK + k_offs_w,
                mask=k_mask_w, other=float("-inf"),
            )
        # Mask invalid pages (page_id<0) → -inf
        sel_pages_w = tl.load(
            PAGE_IDS_ptr + (pid_b * H + kv_h) * TOPK + k_offs_w,
            mask=k_mask_w, other=-1,
        )
        valid_w = sel_pages_w >= 0
        scores_vec = tl.where(valid_w & k_mask_w, scores_vec, float("-inf"))

        # Load Gh values of lse_hq for this h_kv group, reduce → lse_kv
        g_offs = kv_h * Gh + tl.arange(0, Gh)
        if LSE_IS_BF16:
            lse_hq_vec = tl.load(LSE_ptr + pid_b * HQ + g_offs).to(tl.float32)
        else:
            lse_hq_vec = tl.load(LSE_ptr + pid_b * HQ + g_offs)
        lse_max = tl.max(lse_hq_vec, axis=0)
        lse_sum_exp = tl.sum(tl.exp(lse_hq_vec - lse_max), axis=0)
        lse_kv = lse_max + tl.log(lse_sum_exp)

        scores_max = tl.max(scores_vec, axis=0)
        max_val = tl.maximum(scores_max, lse_kv)
        if SOFTMAX1:
            max_val = tl.maximum(max_val, 0.0)

        exp_scores = tl.where(k_mask_w, tl.exp(scores_vec - max_val), 0.0)
        exp_lse = tl.exp(lse_kv - max_val)
        total = tl.sum(exp_scores, axis=0) + exp_lse
        if SOFTMAX1:
            total = total + tl.exp(-max_val)
        inv_total = tl.where(total > 0.0, 1.0 / total, 0.0)

        w_vec = exp_scores * inv_total                   # [BLOCK_K]
        swa_w_inline = exp_lse * inv_total               # scalar (per b, hq)
    else:
        # Dead-code placeholders; will not run when INLINE_WEIGHTS=False.
        # Triton requires them to exist syntactically for the conditional uses below.
        k_offs_w = tl.arange(0, BLOCK_K)
        w_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
        swa_w_inline = 0.0

    acc = tl.zeros((D,), dtype=tl.float32)
    ln2_inv = 1.4426950408889634

    # vector of offsets in page
    offs_s = tl.arange(0, PAGE_SIZE)

    # R87: page iteration range for this split.
    k_start = pid_split * PAGES_PER_SPLIT
    k_end = tl.minimum(k_start + PAGES_PER_SPLIT, TOPK)

    for k_i in range(k_start, k_end):
        page_id = tl.load(
            PAGE_IDS_ptr + pid_b * stride_p_b + kv_h * stride_p_h + k_i * stride_p_k
        ).to(tl.int32)
        if INLINE_WEIGHTS:
            # Extract w_vec[k_i] from the [BLOCK_K] register tile via masked sum.
            # K is tiny (typically 32), so the reduction is essentially free.
            w = tl.sum(tl.where(k_offs_w == k_i, w_vec, 0.0), axis=0)
        else:
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

    if SPLIT_K == 1:
        # Single-split path: apply the optional SWA blend epilogue + bf16 cast
        # and write the final output as before.
        if BLEND_SWA:
            swa_o = tl.load(
                SWA_O_ptr + (pid_b * HQ + pid_hq) * D + offs_d, mask=True, other=0.0,
            ).to(tl.float32)
            if INLINE_WEIGHTS:
                swa_w = swa_w_inline
            else:
                swa_w = tl.load(SWA_W_ptr + pid_b * HQ + pid_hq).to(tl.float32)
            acc = acc + swa_o * swa_w
        out_ptr = OUT_ptr + pid_b * stride_out_b + pid_hq * stride_out_hq
        # FlashHSA alignment: bf16-only output.
        tl.store(out_ptr + offs_d * stride_out_d, acc.to(tl.bfloat16), mask=True)
    else:
        # R87: split-K path — write partial fp32 output to per-split slot.
        # Layout: [B, HQ, SPLIT_K, D] fp32 contiguous.  Output OUT_ptr here
        # actually points at this partial buffer (caller substitutes pointers
        # when SPLIT_K > 1).  Final reduction + BLEND_SWA happens in the
        # separate hsa_decode_reduce_kernel.
        partial_off = ((pid_b * HQ + pid_hq) * SPLIT_K + pid_split) * D + offs_d
        tl.store(OUT_ptr + partial_off, acc, mask=True)


@triton.jit
def hsa_decode_reduce_kernel(
    PARTIAL_ptr,     # [B, HQ, SPLIT_K, D] fp32
    OUT_ptr,         # [B, HQ, D] bf16
    SWA_O_ptr,       # [B, HQ, D] fp32   (BLEND_SWA only)
    SWA_W_ptr,       # [B, HQ]    fp32   (BLEND_SWA only, used when not INLINE_WEIGHTS)
    SCORES_ptr,      # [B, H, K]  scores (BLEND_SWA + INLINE_WEIGHTS path)
    LSE_ptr,         # [B, HQ]    lse    (BLEND_SWA + INLINE_WEIGHTS path)
    PAGE_IDS_ptr,    # [B, H, K]  page ids (used to mask invalid → -inf)
    B: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TOPK: tl.constexpr,
    Gh: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLEND_SWA: tl.constexpr,
    INLINE_WEIGHTS: tl.constexpr,
    SOFTMAX1: tl.constexpr,
    SCORES_IS_BF16: tl.constexpr,
    LSE_IS_BF16: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)
    if pid_b >= B or pid_hq >= HQ:
        return

    G = HQ // H
    kv_h = pid_hq // G

    offs_d = tl.arange(0, D)
    acc = tl.zeros((D,), dtype=tl.float32)
    for s in range(0, SPLIT_K):
        off = ((pid_b * HQ + pid_hq) * SPLIT_K + s) * D + offs_d
        acc += tl.load(PARTIAL_ptr + off, mask=True, other=0.0)

    if BLEND_SWA:
        swa_o = tl.load(
            SWA_O_ptr + (pid_b * HQ + pid_hq) * D + offs_d, mask=True, other=0.0,
        ).to(tl.float32)
        if INLINE_WEIGHTS:
            # Recompute the SWA blend weight inline (matches the decode kernel's
            # softmax).  Same cost: tiny K-element softmax in registers.
            k_offs = tl.arange(0, BLOCK_K)
            k_mask = k_offs < TOPK
            if SCORES_IS_BF16:
                scores_vec = tl.load(
                    SCORES_ptr + (pid_b * H + kv_h) * TOPK + k_offs,
                    mask=k_mask, other=float("-inf"),
                ).to(tl.float32)
            else:
                scores_vec = tl.load(
                    SCORES_ptr + (pid_b * H + kv_h) * TOPK + k_offs,
                    mask=k_mask, other=float("-inf"),
                )
            sel_pages = tl.load(
                PAGE_IDS_ptr + (pid_b * H + kv_h) * TOPK + k_offs,
                mask=k_mask, other=-1,
            )
            valid = sel_pages >= 0
            scores_vec = tl.where(valid & k_mask, scores_vec, float("-inf"))

            g_offs = kv_h * Gh + tl.arange(0, Gh)
            if LSE_IS_BF16:
                lse_hq_vec = tl.load(LSE_ptr + pid_b * HQ + g_offs).to(tl.float32)
            else:
                lse_hq_vec = tl.load(LSE_ptr + pid_b * HQ + g_offs)
            lse_max = tl.max(lse_hq_vec, axis=0)
            lse_sum_exp = tl.sum(tl.exp(lse_hq_vec - lse_max), axis=0)
            lse_kv = lse_max + tl.log(lse_sum_exp)

            scores_max = tl.max(scores_vec, axis=0)
            max_val = tl.maximum(scores_max, lse_kv)
            if SOFTMAX1:
                max_val = tl.maximum(max_val, 0.0)
            exp_scores = tl.where(k_mask, tl.exp(scores_vec - max_val), 0.0)
            exp_lse = tl.exp(lse_kv - max_val)
            total = tl.sum(exp_scores, axis=0) + exp_lse
            if SOFTMAX1:
                total = total + tl.exp(-max_val)
            inv_total = tl.where(total > 0.0, 1.0 / total, 0.0)
            swa_w = exp_lse * inv_total
        else:
            swa_w = tl.load(SWA_W_ptr + pid_b * HQ + pid_hq).to(tl.float32)
        acc = acc + swa_o * swa_w

    tl.store(OUT_ptr + (pid_b * HQ + pid_hq) * D + offs_d, acc.to(tl.bfloat16), mask=True)


# R87: cached split-K config — picked once per process.  GB200 has 144 SMs;
# with B=1, HQ=16 (HSA-345M), grid (1, 16, 8) = 128 programs covers most SMs.
# Heuristic chosen to keep PAGES_PER_SPLIT divisible into TOPK for simplicity.
def _pick_split_k(B: int, HQ: int, TOPK: int, num_sms: int = 132) -> int:
    """Pick a SPLIT_K that uses the GPU well without over-shooting.

    Aim for B*HQ*SPLIT_K up to ~2x num_sms.  TOPK must be divisible by SPLIT_K
    for the simple per-split iteration count.  `HSA_DECODE_SPLIT` overrides.
    """
    import os
    _env = os.getenv("HSA_DECODE_SPLIT")
    if _env is not None:
        s = int(_env)
        return s if (s >= 1 and TOPK % s == 0) else 1
    if B * HQ >= num_sms:
        return 1
    # H200 tuning: allow B*HQ*split up to ~2x num_sms (matches the docstring and
    # the prior fixed split=16).  At B*HQ=16 on 132 SMs this picks 16, measured
    # ~3% faster than the 8 the old 1x target gave (16K decode 3.74→3.62ms).
    target = max(2 * num_sms // (B * HQ), 1)
    best = 1
    for s in (16, 8, 4, 2, 1):
        if TOPK % s == 0 and s <= target:
            best = s
            break
    return best


def hsa_decode_paged_fwd(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    selected_page_ids: torch.Tensor,
    hsa_weights: Optional[torch.Tensor] = None,
    page_size: int,
    sm_scale: Optional[float] = None,
    mask_last_token: bool = True,
    out: Optional[torch.Tensor] = None,
    token_to_seq_id: Optional[torch.Tensor] = None,
    # R25: optional SWA blend in the same kernel
    swa_o_inner: Optional[torch.Tensor] = None,   # [B, HQ, D] fp32
    swa_w_q: Optional[torch.Tensor] = None,       # [B, HQ]    fp32
    # R86: in-kernel chunk-weight path.  When provided, the kernel skips the
    # separate `fused_chunk_weight_h_kv_decode` call and computes both the
    # per-page weights AND the SWA blend weight inline.  Pass selected_scores
    # + lse_hq (raw outputs of selector + internal_swa_decode) and leave
    # hsa_weights/swa_w_q as None.  Caller still needs to pass swa_o_inner
    # for the BLEND_SWA epilogue.
    selected_scores: Optional[torch.Tensor] = None,  # [B, H, K] bf16 or fp32
    lse_hq: Optional[torch.Tensor] = None,           # [B, HQ]   bf16 or fp32
    enable_softmax1: bool = False,
    # R87: split-K parallelism — defaults to autotuned based on B*HQ vs num_sms.
    split_k: Optional[int] = None,
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

    inline_weights = selected_scores is not None and lse_hq is not None
    if not inline_weights:
        if hsa_weights is None:
            raise ValueError(
                "Either hsa_weights or (selected_scores + lse_hq) must be provided."
            )
        if hsa_weights.dim() != 3:
            raise ValueError("hsa_weights must be [B/T,HQ,K]")

    N, HQ, D = q.shape  # N = B (decode) or T (extend)
    Nloc, H, Dk = k_cache.shape
    assert v_cache.shape == (Nloc, H, Dk)
    assert Dk == D
    MAX_T = int(page_table_1.shape[1])
    assert selected_page_ids.shape[0] == N and selected_page_ids.shape[1] == H
    TOPK = int(selected_page_ids.shape[2])
    if not inline_weights:
        assert hsa_weights.shape == (N, HQ, TOPK)
    else:
        assert selected_scores.shape == (N, H, TOPK), (
            f"selected_scores {tuple(selected_scores.shape)} != ({N}, {H}, {TOPK})"
        )
        assert lse_hq.shape == (N, HQ), (
            f"lse_hq {tuple(lse_hq.shape)} != ({N}, {HQ})"
        )
        assert selected_scores.dtype in (torch.bfloat16, torch.float32)
        assert lse_hq.dtype in (torch.bfloat16, torch.float32)
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
    pt_ = page_table_1 if page_table_1.dtype == torch.int32 and page_table_1.is_contiguous() \
        else page_table_1.to(torch.int32).contiguous()
    # R33: accept int32 OR int64 page_ids; kernel does `.to(tl.int32)` in-register
    # after the load, so an int64 pointer-dtype just doubles a tiny amount of
    # gmem traffic (32 ids × 8B = 256B/layer) and saves the explicit cast
    # (which under CG capture isn't free — it's a fresh int32 buffer alloc).
    page_ids_ = selected_page_ids if selected_page_ids.is_contiguous() \
        else selected_page_ids.contiguous()

    if inline_weights:
        # Placeholder W tensor (kernel won't read it). 1-element keeps strides defined.
        w_ = torch.zeros(1, dtype=torch.float32, device=q.device)
        scores_ = selected_scores if selected_scores.is_contiguous() \
            else selected_scores.contiguous()
        lse_ = lse_hq if lse_hq.is_contiguous() else lse_hq.contiguous()
        Gh_val = HQ // H
        BLOCK_K = max(triton.next_power_of_2(TOPK), 16)
        scores_is_bf16 = bool(scores_.dtype == torch.bfloat16)
        lse_is_bf16 = bool(lse_.dtype == torch.bfloat16)
    else:
        w_ = hsa_weights.contiguous()
        # Dummy scores/lse pointers; kernel skips them when INLINE_WEIGHTS=False.
        scores_ = torch.zeros(1, dtype=torch.float32, device=q.device)
        lse_ = torch.zeros(1, dtype=torch.float32, device=q.device)
        Gh_val = HQ // H
        BLOCK_K = 16
        scores_is_bf16 = False
        lse_is_bf16 = False

    # For the seq_id_map, use the real tensor or a dummy scalar placeholder.
    if use_seq_map:
        seq_map_ = token_to_seq_id.to(torch.int32).contiguous()
    else:
        seq_map_ = torch.zeros(1, dtype=torch.int32, device=q.device)

    if out is None:
        out = torch.empty((N, HQ, D), device=q.device, dtype=torch.bfloat16)
    else:
        assert out.shape == (N, HQ, D)

    # R25: blend-in-kernel inputs
    # When INLINE_WEIGHTS, the swa_w_q is computed inside the kernel — only
    # swa_o_inner is needed externally.
    if inline_weights:
        blend_swa = swa_o_inner is not None
        if blend_swa:
            assert swa_o_inner.shape == (N, HQ, D)
            swa_o_ = swa_o_inner
        else:
            swa_o_ = torch.zeros(1, dtype=torch.float32, device=q.device)
        swa_w_ = torch.zeros(1, dtype=torch.float32, device=q.device)
    else:
        blend_swa = (swa_o_inner is not None) and (swa_w_q is not None)
        if blend_swa:
            assert swa_o_inner.shape == (N, HQ, D), f"{swa_o_inner.shape} != ({N}, {HQ}, {D})"
            assert swa_w_q.shape == (N, HQ), f"{swa_w_q.shape} != ({N}, {HQ})"
            swa_o_ = swa_o_inner
            swa_w_ = swa_w_q
        else:
            swa_o_ = torch.zeros(1, dtype=torch.float32, device=q.device)
            swa_w_ = torch.zeros(1, dtype=torch.float32, device=q.device)

    # Provide a stride tuple for w_ that the kernel can read.  When INLINE_WEIGHTS
    # is True the strides are never used (kernel skips W loads), but Triton still
    # binds them as constexpr arguments, so any consistent integer triple is fine.
    if w_.dim() == 3:
        w_stride0, w_stride1, w_stride2 = w_.stride()
    else:
        w_stride0, w_stride1, w_stride2 = 0, 0, 0

    # R87: split-K decision.  Decode is the hot path (N small, HQ ≤ ~32 for
    # GQA models); split TOPK across SMs.  Extend is N-heavy already and
    # doesn't benefit, so leave SPLIT_K=1 there.
    if split_k is None:
        if use_seq_map:
            split_k = 1
        else:
            split_k = _pick_split_k(N, HQ, TOPK)
    if TOPK % split_k != 0:
        split_k = 1  # fallback if TOPK doesn't divide cleanly
    pages_per_split = TOPK // split_k

    if split_k == 1:
        # No partial buffer needed; kernel writes directly to `out`.
        out_for_kernel = out
        partial_buf = None
    else:
        # Partial buffer at fp32 granularity per (b, hq, split, d).
        partial_buf = torch.empty(
            (N, HQ, split_k, D), device=q.device, dtype=torch.float32
        )
        out_for_kernel = partial_buf

    # When SPLIT_K > 1, the per-split decode kernel does NOT apply the BLEND_SWA
    # epilogue — the reduction kernel does.  Tell the decode kernel BLEND_SWA=False
    # so it just writes the partial accumulator.
    decode_blend_swa = blend_swa and (split_k == 1)

    grid = (N, HQ, split_k)
    hsa_decode_paged_fwd_kernel[grid](
        q_,
        k_,
        v_,
        pt_,
        page_ids_,
        w_,
        scores_,
        lse_,
        out_for_kernel,
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
        w_stride0,
        w_stride1,
        w_stride2,
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
        BLEND_SWA=decode_blend_swa,
        INLINE_WEIGHTS=inline_weights,
        SOFTMAX1=bool(enable_softmax1),
        SCORES_IS_BF16=scores_is_bf16,
        LSE_IS_BF16=lse_is_bf16,
        Gh=Gh_val,
        BLOCK_K=BLOCK_K,
        SPLIT_K=split_k,
        PAGES_PER_SPLIT=pages_per_split,
        num_warps=4,
    )

    if split_k > 1:
        # R87: reduction kernel — sums per-split partials + applies BLEND_SWA + bf16 cast.
        hsa_decode_reduce_kernel[(N, HQ)](
            partial_buf,
            out,
            swa_o_,
            swa_w_,
            scores_,
            lse_,
            page_ids_,
            B=N,
            HQ=HQ,
            H=H,
            D=D,
            SPLIT_K=split_k,
            TOPK=TOPK,
            Gh=Gh_val,
            BLOCK_K=BLOCK_K,
            BLEND_SWA=blend_swa,
            INLINE_WEIGHTS=inline_weights,
            SOFTMAX1=bool(enable_softmax1),
            SCORES_IS_BF16=scores_is_bf16,
            LSE_IS_BF16=lse_is_bf16,
            num_warps=2,
        )
    return out


