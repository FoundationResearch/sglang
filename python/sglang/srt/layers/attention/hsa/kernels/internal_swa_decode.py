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

R88: optional split-W parallelism — grid becomes (B*HQ_hsa, SPLIT_W) so we
can use ~140 SMs instead of just 16.  Each split walks its share of the
W blocks and writes a partial (m, l, acc) tuple; a small reduce kernel
combines them via the online-softmax merge.
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
    swa_o_ptr,            # [B*HQ_hsa*D] fp32  (output; SPLIT_W==1 path) OR
                          # [B*HQ_hsa*SPLIT_W*D] fp32 (partial accumulators; SPLIT_W>1)
    lse_hq_ptr,           # [B*HQ_hsa]    bf16 (output; SPLIT_W==1 path) OR
                          # [B*HQ_hsa*SPLIT_W*2] fp32 (partial (m, l); SPLIT_W>1)
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
    SPLIT_W: tl.constexpr,
    BLOCKS_PER_SPLIT: tl.constexpr,
):
    """One program per (b, hq, split). Streaming flash attention over the window."""
    pid = tl.program_id(axis=0)
    pid_split = tl.program_id(axis=1)
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

    # R88: each split processes its slice of NUM_W_BLOCKS.
    w_start = pid_split * BLOCKS_PER_SPLIT
    w_end = tl.minimum(w_start + BLOCKS_PER_SPLIT, NUM_W_BLOCKS)

    for w_idx in range(w_start, w_end):
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
        # Guard the all-masked window case: if no valid KV so far, both m_i and
        # block_max are -inf -> m_new=-inf -> exp(m_i - m_new)=exp(nan)=nan,
        # which poisons acc. Treat the rescale factor as 1.0 in that case (acc
        # is still 0). Short sequences / LMK-internal decode steps can hit this.
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
        exp_scores = tl.exp(scores - m_new)
        exp_scores = tl.where(w_valid, exp_scores, 0.0)
        block_sum = tl.sum(exp_scores, axis=0)

        acc = acc * alpha + tl.sum(exp_scores[:, None] * v, axis=0)
        l_i = l_i * alpha + block_sum
        m_i = m_new

    if SPLIT_W == 1:
        # Single-split path: finalize in-kernel.
        safe_l = tl.where(l_i > 0.0, l_i, 1.0)
        o = acc / safe_l
        # When the window is empty, output stays at 0 (init) and lse stays at -inf.
        o = tl.where(l_i > 0.0, o, 0.0)
        lse = tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf"))
        tl.store(swa_o_ptr + (b * HQ_hsa + hq) * D + d_offs, o)
        tl.store(lse_hq_ptr + b * HQ_hsa + hq, lse.to(tl.bfloat16))
    else:
        # Partial output: write (m_i, l_i, acc) so the reduce kernel can merge.
        # Layout per (b, hq):
        #   swa_o_ptr     : [B, HQ, SPLIT_W, D] fp32   (partial acc, unnormalized)
        #   lse_hq_ptr    : [B, HQ, SPLIT_W, 2] fp32   ((m, l) pair per split)
        out_off = ((b * HQ_hsa + hq) * SPLIT_W + pid_split) * D + d_offs
        tl.store(swa_o_ptr + out_off, acc)
        ml_off = ((b * HQ_hsa + hq) * SPLIT_W + pid_split) * 2
        tl.store(lse_hq_ptr + ml_off + 0, m_i)
        tl.store(lse_hq_ptr + ml_off + 1, l_i)


@triton.jit
def fused_internal_swa_decode_reduce_kernel(
    partial_o_ptr,        # [B, HQ_hsa, SPLIT_W, D] fp32
    partial_ml_ptr,       # [B, HQ_hsa, SPLIT_W, 2] fp32 (m, l per split)
    swa_o_ptr,            # [B, HQ_hsa, D] fp32     (final output)
    lse_hq_ptr,           # [B, HQ_hsa]    bf16     (final logsumexp)
    HQ_hsa: tl.constexpr,
    D: tl.constexpr,
    SPLIT_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // HQ_hsa
    hq = pid % HQ_hsa
    d_offs = tl.arange(0, D)

    # First pass: find global max m_global.
    m_global = float("-inf")
    for s in range(0, SPLIT_W):
        ml_off = ((b * HQ_hsa + hq) * SPLIT_W + s) * 2
        m_s = tl.load(partial_ml_ptr + ml_off)
        m_global = tl.maximum(m_global, m_s)

    # Second pass: rescale & sum.
    l_global = tl.zeros([], tl.float32)
    acc = tl.zeros([D], tl.float32)
    for s in range(0, SPLIT_W):
        ml_off = ((b * HQ_hsa + hq) * SPLIT_W + s) * 2
        m_s = tl.load(partial_ml_ptr + ml_off + 0)
        l_s = tl.load(partial_ml_ptr + ml_off + 1)
        out_off = ((b * HQ_hsa + hq) * SPLIT_W + s) * D + d_offs
        acc_s = tl.load(partial_o_ptr + out_off)

        # Guard the all-empty case: if every split saw no valid KV, m_global is
        # -inf and exp(m_s - m_global)=exp(nan)=nan. An empty split contributes
        # nothing, so use 0.0 there (a finite m_global gives the normal factor,
        # and a single empty split among valid ones already yields exp(-inf)=0).
        alpha = tl.where(m_global == float("-inf"), 0.0, tl.exp(m_s - m_global))
        l_global = l_global + l_s * alpha
        acc = acc + acc_s * alpha

    # Finalize.
    safe_l = tl.where(l_global > 0.0, l_global, 1.0)
    o = acc / safe_l
    o = tl.where(l_global > 0.0, o, 0.0)
    lse = tl.where(l_global > 0.0, m_global + tl.log(l_global), float("-inf"))

    tl.store(swa_o_ptr + (b * HQ_hsa + hq) * D + d_offs, o)
    tl.store(lse_hq_ptr + b * HQ_hsa + hq, lse.to(tl.bfloat16))


def _pick_split_w(B: int, HQ_hsa: int, num_w_blocks: int, num_sms: int = 132) -> int:
    """Pick SPLIT_W: cover SMs without going past num_w_blocks."""
    if B * HQ_hsa >= num_sms or num_w_blocks <= 1:
        return 1
    target = max(num_sms // (B * HQ_hsa), 1)
    # SPLIT_W shouldn't exceed num_w_blocks. Pick the smallest power-of-2
    # split that uses enough SMs.
    best = 1
    for s in (8, 4, 2, 1):
        if s <= target and s <= num_w_blocks:
            best = s
            break
    return best


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

    # R88: split-W decision.
    split_w = _pick_split_w(B, HQ_hsa, NUM_W_BLOCKS)
    blocks_per_split = (NUM_W_BLOCKS + split_w - 1) // split_w

    if split_w == 1:
        # No partial buffers needed; write directly to swa_o/lse_hq.
        partial_o_ptr = swa_o
        partial_ml_ptr = lse_hq  # ignored
    else:
        partial_o = torch.empty((B, HQ_hsa, split_w, D), device=device, dtype=torch.float32)
        partial_ml = torch.empty((B, HQ_hsa, split_w, 2), device=device, dtype=torch.float32)
        partial_o_ptr = partial_o
        partial_ml_ptr = partial_ml

    fused_internal_swa_decode_kernel[(B * HQ_hsa, split_w)](
        q_hsa.contiguous(),
        k_cache_full.contiguous(),
        v_cache_full.contiguous(),
        page_table_1.contiguous(),
        cache_seqlens.contiguous(),
        partial_o_ptr,
        partial_ml_ptr,
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
        SPLIT_W=int(split_w),
        BLOCKS_PER_SPLIT=int(blocks_per_split),
        num_warps=4,
    )

    if split_w > 1:
        fused_internal_swa_decode_reduce_kernel[(B * HQ_hsa,)](
            partial_o,
            partial_ml,
            swa_o,
            lse_hq,
            HQ_hsa=int(HQ_hsa),
            D=int(D),
            SPLIT_W=int(split_w),
            num_warps=2,
        )

    return swa_o, lse_hq
