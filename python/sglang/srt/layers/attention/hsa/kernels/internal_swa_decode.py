"""Fused streaming-flash-attention kernel for HSA's internal SWA at decode.

Replaces the R11/R16 chain (~15 ops per layer per decode step):

    seq_lens_i64 = cache_seqlens.to(int64)
    chunk_start = where(raw_start >= 0, (raw_start // ps) * ps, 0).clamp(min=0)
    pos_offsets = arange(W)
    tok_pos = chunk_start[:, None] + pos_offsets[None, :]
    valid_mask = (tok_pos < seqlen) & ((tok_pos % ps) != (ps-1)) & (seqlen > 0)
    tok_pos_safe = tok_pos.clamp(min=0, max=...)
    token_locs = gather(page_table_1.to(int64), 1, tok_pos_safe)
    k_hsa = k_cache[:, H_swa:H_swa+H_hsa, :]; v_hsa = v_cache[...]
    flat_locs = token_locs.reshape(-1)
    k_win = k_hsa[flat_locs].view(B, W, H_hsa, D)
    v_win = v_hsa[flat_locs].view(B, W, H_hsa, D)
    q_hgd = q_hsa.view(B, H_hsa, Gh, D)
    logits = einsum("bhgd,bwhd->bhgw", q_hgd, k_win).to(fp32) * sm_scale
    logits = logits.masked_fill(~valid_mask, -inf)
    lse_per_q = logsumexp(logits, -1)
    p = softmax(logits, -1)
    p = nan_to_num(p, 0)
    o = einsum("bhgw,bwhd->bhgd", p.to(bf16), v_win).to(fp32)

with one streaming flash-attention kernel that walks the window in
BLOCK_W chunks and maintains running (max, sum, output) state for
numerical-stable online softmax.

LMK exclusion happens inside the kernel via `(pos+1) % page_size != 0`.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_internal_swa_decode_kernel(
    q_ptr,                # [B*HQ_hsa*D] bf16
    k_cache_ptr,          # [Nloc*H_total*D] bf16
    v_cache_ptr,          # [Nloc*H_total*D] bf16
    page_table_1_ptr,     # [B*max_seqlen] int32
    cache_seqlens_ptr,    # [B] int32
    swa_o_ptr,            # [B*HQ_hsa*D] fp32  (output)
    lse_hq_ptr,           # [B*HQ_hsa]    bf16 (output)
    sm_scale,
    H_swa: tl.constexpr,
    H_hsa: tl.constexpr,
    HQ_hsa: tl.constexpr,
    H_total: tl.constexpr,
    D: tl.constexpr,
    max_seqlen: tl.constexpr,
    page_size: tl.constexpr,
    hsa_window: tl.constexpr,
    BLOCK_W: tl.constexpr,
    NUM_W_BLOCKS: tl.constexpr,
):
    """One program per (b, hq). Streaming flash attention over the window."""
    pid = tl.program_id(axis=0)
    b = pid // HQ_hsa
    hq = pid % HQ_hsa

    Gh: tl.constexpr = HQ_hsa // H_hsa
    h_kv_local = hq // Gh           # 0..H_hsa-1
    h_kv_global = H_swa + h_kv_local

    d_offs = tl.arange(0, D)
    q = tl.load(q_ptr + (b * HQ_hsa + hq) * D + d_offs).to(tl.float32)

    seqlen = tl.load(cache_seqlens_ptr + b)
    # chunk_start = max(0, ((seqlen - 1) - hsa_window + 1) // page_size * page_size)
    #             = max(0, (seqlen - hsa_window) // page_size * page_size)
    raw_start = seqlen - hsa_window
    chunk_start_pos = tl.where(
        raw_start >= 0, (raw_start // page_size) * page_size, 0
    )

    # Online flash-attention running state.
    m_i = float("-inf")     # running max of logits
    l_i = tl.zeros([], tl.float32)  # running sum of exp(logit - m)
    acc = tl.zeros([D], tl.float32)

    for w_idx in tl.static_range(NUM_W_BLOCKS):
        w_block_start = w_idx * BLOCK_W
        w_offs_in_window = w_block_start + tl.arange(0, BLOCK_W)
        w_global = chunk_start_pos + w_offs_in_window  # absolute positions

        # Validity: in [chunk_start, seqlen)  AND  not a LMK slot
        in_range = (w_global < seqlen) & (w_global >= chunk_start_pos) & (seqlen > 0)
        not_lmk = (w_global % page_size) != (page_size - 1)
        w_valid = in_range & not_lmk

        # Safe token positions for the page_table_1 gather.
        w_safe = tl.minimum(tl.maximum(w_global, 0), max_seqlen - 1)
        token_locs = tl.load(
            page_table_1_ptr + b * max_seqlen + w_safe,
            mask=in_range, other=0,
        ).to(tl.int64)

        # Gather K/V[token_locs, h_kv_global, :]
        kv_off = token_locs[:, None] * H_total * D + h_kv_global * D + d_offs[None, :]
        k = tl.load(
            k_cache_ptr + kv_off, mask=w_valid[:, None], other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_cache_ptr + kv_off, mask=w_valid[:, None], other=0.0,
        ).to(tl.float32)

        # Compute scaled scores [BLOCK_W]
        scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
        scores = tl.where(w_valid, scores, float("-inf"))

        # Online softmax update
        block_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        exp_scores = tl.exp(scores - m_new)
        exp_scores = tl.where(w_valid, exp_scores, 0.0)
        block_sum = tl.sum(exp_scores, axis=0)

        acc = acc * alpha + tl.sum(exp_scores[:, None] * v, axis=0)
        l_i = l_i * alpha + block_sum
        m_i = m_new

    # Finalise — guard against all-masked window (l_i == 0).
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o = acc / safe_l
    # When the window is empty, output stays at 0 (init) and lse stays at -inf.
    o = tl.where(l_i > 0.0, o, 0.0)
    lse = tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf"))

    tl.store(swa_o_ptr + (b * HQ_hsa + hq) * D + d_offs, o)
    tl.store(lse_hq_ptr + b * HQ_hsa + hq, lse.to(tl.bfloat16))


def fused_internal_swa_decode(
    *,
    q_hsa: torch.Tensor,           # [B, HQ_hsa, D] bf16
    k_cache_full: torch.Tensor,    # [Nloc, H_total, D] bf16
    v_cache_full: torch.Tensor,    # [Nloc, H_total, D] bf16
    page_table_1: torch.Tensor,    # [B, max_seqlen] int32
    cache_seqlens: torch.Tensor,   # [B] int32
    H_swa: int,
    H_hsa: int,
    HQ_hsa: int,
    page_size: int,
    hsa_window: int,
    sm_scale: float,
):
    """Returns:
      swa_o : [B, HQ_hsa, D] fp32
      lse_hq: [B, HQ_hsa]    bf16
    """
    assert q_hsa.dim() == 3
    B, HQ_, D = q_hsa.shape
    assert HQ_ == HQ_hsa
    Nloc, H_total, D2 = k_cache_full.shape
    assert D2 == D and v_cache_full.shape == k_cache_full.shape
    assert page_table_1.dim() == 2 and page_table_1.shape[0] == B
    max_seqlen = page_table_1.shape[1]

    device = q_hsa.device

    # R31: kernel always writes every (b, hq) slot (lines 132-133), so torch.empty
    # is correct here.  The early-exit below keeps the zeros/full path since the
    # kernel isn't launched.
    if hsa_window <= 0:
        swa_o = torch.zeros((B, HQ_hsa, D), device=device, dtype=torch.float32)
        lse_hq = torch.full((B, HQ_hsa), float("-inf"), device=device, dtype=torch.bfloat16)
        return swa_o, lse_hq

    swa_o = torch.empty((B, HQ_hsa, D), device=device, dtype=torch.float32)
    lse_hq = torch.empty((B, HQ_hsa), device=device, dtype=torch.bfloat16)

    # Window upper bound = hsa_window + page_size (chunk-aligned slack).
    W_total = int(hsa_window) + int(page_size)
    BLOCK_W = 128
    NUM_W_BLOCKS = (W_total + BLOCK_W - 1) // BLOCK_W

    fused_internal_swa_decode_kernel[(B * HQ_hsa,)](
        q_hsa.contiguous(),
        k_cache_full.contiguous(),
        v_cache_full.contiguous(),
        page_table_1.contiguous(),
        cache_seqlens.contiguous(),
        swa_o,
        lse_hq,
        float(sm_scale),
        H_swa=int(H_swa),
        H_hsa=int(H_hsa),
        HQ_hsa=int(HQ_hsa),
        H_total=int(H_total),
        D=int(D),
        max_seqlen=int(max_seqlen),
        page_size=int(page_size),
        hsa_window=int(hsa_window),
        BLOCK_W=int(BLOCK_W),
        NUM_W_BLOCKS=int(NUM_W_BLOCKS),
        num_warps=4,
    )

    return swa_o, lse_hq
