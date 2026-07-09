import os
import torch
import tilelang
import tilelang.language as T
from typing import Optional
import math

def ref_topk_max_pooling(q, k_lmks, topk, block_size, window_size, is_causal=False, q_offset=0, bias=None, lse_swa=None, head_mask=None, mask_recompute=False):
    """
    Reference implementation for max-pooling Top-K (no online softmax).

    Selection rule: argmax_g (sm_scale * qk + bias - lse_swa) over groups G,
    then top-k over chunks S. The lse_swa term is optional and defaults to the
    old behavior when omitted.
    Output scores are raw scaled qk (sm_scale * qk), WITHOUT bias or lse_swa,
    to match the softmax-version's recompute output convention.

    Args:
        q: [B, L, h_kv, G, D]
        k_lmks: [B, S, h_kv, D] (per-KV-head shared lmks) or
                [B, S, h_kv * G, D] (per-q-head lmks)
        topk: int
        block_size: int
        window_size: int
        is_causal: bool
        q_offset: global query offset for inference causal masking.
        bias: optional additive bias with shape [B, S, h_kv, G] or [B, S, h_kv * G].
              When provided, selection logits add bias[b, chunk, head]; returned
              scores stay raw scaled qk and do not include bias.
        lse_swa: optional [B, L, h_kv, G] or [B, L, h_kv * G]. When provided,
                 selection logits subtract lse_swa[b, query, head]. Returned
                 scores stay raw scaled qk and do not include lse_swa.
        head_mask: optional [h_kv, G] or [h_kv * G]. 1 keeps a head active;
                   0 removes it from max-over-G selection. Recompute still
                   returns raw-qk scores for all heads at selected chunks.

    Returns:
        indices_sorted: [B, L, h_kv, topk]
        scores_sorted:  [B, L, h_kv, G, topk]   (raw scaled qk, NO bias/lse_swa)
    """
    B, L, h_kv, G, D = q.shape
    S = k_lmks.shape[1]
    sm_scale = 1.0 / math.sqrt(D)
    lmks_h = k_lmks.shape[2]
    if lmks_h == h_kv:
        per_qhead_lmks = False
    elif lmks_h == h_kv * G:
        per_qhead_lmks = True
    else:
        raise AssertionError(
            f"k_lmks h dim ({lmks_h}) must be either h_kv ({h_kv}) or h_kv*G ({h_kv * G})"
        )

    if per_qhead_lmks:
        k_lmks_v = k_lmks.view(B, S, h_kv, G, D)
        scores_all = torch.einsum("blhgd,bshgd->blhgs", q.float(), k_lmks_v.float())
    else:
        scores_all = torch.einsum("blhgd,bshd->blhgs", q.float(), k_lmks.float())

    # raw scaled qk (no bias) — used both for selection (after adding bias) and for output
    scores_all_scaled = scores_all * sm_scale

    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1)  # [L, 1]
        i_idx_global = i_idx + int(q_offset)
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)  # [1, S]
        threshold_idx = (i_idx_global - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= threshold_idx
        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all_scaled = scores_all_scaled.masked_fill(causal_mask_expanded, float('-inf'))

    # Build selection logits = scaled qk + bias (broadcast over L)
    if bias is not None:
        if bias.dim() == 3:
            assert bias.shape == (B, S, h_kv * G), (
                f"bias shape {tuple(bias.shape)} != ({B}, {S}, {h_kv * G})"
            )
            bias_4d = bias.float().reshape(B, S, h_kv, G)
        elif bias.dim() == 4:
            assert bias.shape == (B, S, h_kv, G), (
                f"bias shape {tuple(bias.shape)} != ({B}, {S}, {h_kv}, {G})"
            )
            bias_4d = bias.float()
        else:
            raise AssertionError(
                f"bias must be [B, S, h_q] or [B, S, h_kv, G], got {tuple(bias.shape)}"
            )
        # bias_4d: [B, S, h_kv, G] -> broadcast to [B, L, h_kv, G, S]
        bias_for_sel = bias_4d.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, h_kv, G, S]
        scores_for_sel = scores_all_scaled + bias_for_sel
        # Preserve -inf positions (causal masked)
        scores_for_sel = torch.where(
            scores_all_scaled == float('-inf'),
            scores_all_scaled,
            scores_for_sel,
        )
    else:
        scores_for_sel = scores_all_scaled

    if lse_swa is not None:
        lse_swa = lse_swa.to(device=q.device, dtype=torch.float32)
        if lse_swa.dim() == 3:
            assert lse_swa.shape == (B, L, h_kv * G), (
                f"lse_swa shape {tuple(lse_swa.shape)} != ({B}, {L}, {h_kv * G})"
            )
            lse_swa_view = lse_swa.reshape(B, L, h_kv, G)
        elif lse_swa.dim() == 4:
            assert lse_swa.shape == (B, L, h_kv, G), (
                f"lse_swa shape {tuple(lse_swa.shape)} != ({B}, {L}, {h_kv}, {G})"
            )
            lse_swa_view = lse_swa
        else:
            raise AssertionError(
                f"lse_swa must be [B, L, h_q] or [B, L, h_kv, G], got {tuple(lse_swa.shape)}"
            )
        scores_for_sel = torch.where(
            scores_for_sel == float('-inf'),
            scores_for_sel,
            scores_for_sel - lse_swa_view.unsqueeze(-1),
        )

    head_mask_view = None
    if head_mask is not None:
        head_mask_view = head_mask.to(device=q.device, dtype=torch.bool)
        if head_mask_view.dim() == 1:
            assert head_mask_view.numel() == h_kv * G, (
                f"head_mask shape {tuple(head_mask_view.shape)} != ({h_kv * G},)"
            )
            head_mask_view = head_mask_view.view(h_kv, G)
        elif head_mask_view.dim() == 2:
            assert tuple(head_mask_view.shape) == (h_kv, G), (
                f"head_mask shape {tuple(head_mask_view.shape)} != ({h_kv}, {G})"
            )
        else:
            raise AssertionError(
                f"head_mask must be [h_q] or [h_kv, G], got {tuple(head_mask_view.shape)}"
            )
        inactive = ~head_mask_view.view(1, 1, h_kv, G, 1)
        scores_for_sel = scores_for_sel.masked_fill(inactive, float('-inf'))

    # max over G with selection logits
    scores_max_pooling = scores_for_sel.max(dim=3).values  # [B, L, h_kv, S]

    # Handle topk > S or causal-masked rows where valid chunks < topk by
    # taking only actual_topk = min(topk, S), then padding with sentinel -1.
    # Invalid (-inf) positions are also replaced with -1 to mirror kernel behavior.
    actual_topk = min(topk, S)
    topk_scores, topk_indices = torch.topk(
        scores_max_pooling, k=actual_topk, dim=-1, sorted=False
    )
    topk_indices[topk_scores == float('-inf')] = -1
    if actual_topk < topk:
        pad = torch.full(
            (B, L, h_kv, topk - actual_topk), -1,
            dtype=topk_indices.dtype, device=topk_indices.device,
        )
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    # Sort with -1 routed to the end (use S+1000 as sentinel key, then restore)
    sort_temp = topk_indices.clone()
    sort_temp[sort_temp < 0] = S + 1000
    indices_sorted, _ = torch.sort(sort_temp, dim=-1)
    indices_sorted[indices_sorted >= S] = -1

    # Gather the RAW scaled qk (no bias) at the selected indices.
    # For -1 positions, use 0 as a safe gather index and mask out afterwards.
    safe_indices = indices_sorted.clone()
    safe_indices[safe_indices < 0] = 0
    indices_expanded = safe_indices.unsqueeze(3).expand(-1, -1, -1, G, -1)
    scores_sorted = torch.gather(scores_all_scaled, -1, indices_expanded)
    invalid_mask = indices_sorted.unsqueeze(3).expand(-1, -1, -1, G, -1) < 0
    scores_sorted = scores_sorted.masked_fill(invalid_mask, float('-inf'))
    if mask_recompute and head_mask_view is not None:
        inactive = ~head_mask_view.view(1, 1, h_kv, G, 1)
        scores_sorted = scores_sorted.masked_fill(inactive, float('-inf'))
    return indices_sorted, scores_sorted


@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_topk_max_pooling_kernel(batch, seq_len, s_len, h_kv, groups, head_dim, topk,
                                  block_size, window_size, is_causal,
                                  BLOCK_L=None, BLOCK_S=None, threads=None,
                                  use_bias=False, use_lse_swa=False,
                                  per_qhead_lmks=False, use_head_mask=False,
                                  is_training=True):
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    if not is_training:
        seq_len_var = T.dynamic("seq_len")
        s_len_var = T.dynamic("s_len")
    else:
        seq_len_var = seq_len
        s_len_var = s_len

    q_shape = [batch, seq_len_var, h_kv, groups, head_dim]
    if per_qhead_lmks:
        k_shape = [batch, s_len_var, h_kv * groups, head_dim]
    else:
        k_shape = [batch, s_len_var, h_kv, head_dim]
    out_indices_shape = [batch, seq_len_var, h_kv, topk]
    bias_shape = [batch, s_len_var, h_kv, groups] if use_bias else [1, 1, 1, 1]
    lse_swa_shape = [batch, seq_len_var, h_kv, groups] if use_lse_swa else [1, 1, 1, 1]
    head_mask_shape = [h_kv, groups] if use_head_mask else [1, 1]
    q_offset_shape = [1]

    if BLOCK_L is None:
        # GEMM requires M % 16 == 0. For shared-K path M = BLOCK_L*groups, so
        # use the smallest BLOCK_L that makes M >= 16. For per-q-head path the
        # GEMM is per-g with M = BLOCK_L, so BLOCK_L itself must be >= 16.
        BLOCK_L = int(os.environ.get("HSA_MP_BLOCK_L", "16"))
    if BLOCK_S is None:
        BLOCK_S = int(os.environ.get("HSA_MP_BLOCK_S", "16"))
    BLOCK_D = head_dim
    if threads is None:
        threads = int(os.environ.get("HSA_MP_THREADS", "64"))

    GEMM_M = BLOCK_L * groups
    num_s_blocks = tilelang.cdiv(s_len_var, BLOCK_S)

    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def fwd_kernel_max_pooling(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        LSE_SWA: T.Tensor(lse_swa_shape, accum_dtype),
        bias: T.Tensor(bias_shape, accum_dtype),
        HeadMask: T.Tensor(head_mask_shape, idx_dtype),
        Q_Offset: T.Tensor(q_offset_shape, idx_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            q_offset = T.if_then_else(is_training, 0, Q_Offset[0])
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L

            Q_shared = T.alloc_shared([GEMM_M, BLOCK_D], dtype)
            K_shared = T.alloc_shared([BLOCK_S, BLOCK_D], dtype)
            score_shared = T.alloc_shared([GEMM_M, BLOCK_S], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, BLOCK_S], accum_dtype)
            # R89: parallel max-over-G reduction buffer. Removes the serial
            # inner g-loop (groups iterations per chunk) from the per-query
            # topk scan, moving it onto all `threads` lanes instead of BLOCK_L.
            score_maxg = T.alloc_shared([BLOCK_L, BLOCK_S], accum_dtype)

            # Per-g GEMM staging (only used when per_qhead_lmks=True)
            Q_g_shared = T.alloc_shared([BLOCK_L, BLOCK_D], dtype)
            acc_s_g = T.alloc_fragment([BLOCK_L, BLOCK_S], accum_dtype)

            topk_max_scores_local = T.alloc_local([topk], accum_dtype)
            topk_indices_local = T.alloc_local([topk], idx_dtype)

            T.fill(topk_max_scores_local, -T.infinity(accum_dtype))
            T.fill(topk_indices_local, -1)

            for l_idx, g, d in T.Parallel(BLOCK_L, groups, BLOCK_D):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < seq_len_var:
                    Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                else:
                    Q_shared[flat_m, d] = 0

            loop_limit = T.alloc_var(idx_dtype)
            loop_limit = num_s_blocks
            if is_causal:
                limit_ts = tilelang.cdiv(q_offset + base_l + BLOCK_L, block_size)
                loop_limit = T.min(loop_limit, tilelang.cdiv(limit_ts, BLOCK_S) + 1)

            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S

                if per_qhead_lmks:
                    # Per-g GEMM: write into score_shared[l*G+g, s_idx].
                    # When use_head_mask=True, inactive local heads skip the
                    # GEMM entirely and are filled with -inf so they cannot win
                    # the following max-over-G selection.
                    for g in T.serial(groups):
                        if (not use_head_mask) or (HeadMask[i_h, g] != 0):
                            for s_idx, d in T.Parallel(BLOCK_S, BLOCK_D):
                                ts = base_s + s_idx
                                if ts < s_len_var:
                                    K_shared[s_idx, d] = K[i_b, ts, i_h * groups + g, d]
                                else:
                                    K_shared[s_idx, d] = 0
                            for l_idx, d in T.Parallel(BLOCK_L, BLOCK_D):
                                Q_g_shared[l_idx, d] = Q_shared[l_idx * groups + g, d]
                            T.sync_threads()
                            T.clear(acc_s_g)
                            T.gemm(Q_g_shared, K_shared, acc_s_g, transpose_B=True,
                                   policy=T.GemmWarpPolicy.FullRow)
                            for l_idx, s_idx in T.Parallel(BLOCK_L, BLOCK_S):
                                score_shared[l_idx * groups + g, s_idx] = acc_s_g[l_idx, s_idx]
                            T.sync_threads()
                        else:
                            for l_idx, s_idx in T.Parallel(BLOCK_L, BLOCK_S):
                                score_shared[l_idx * groups + g, s_idx] = -T.infinity(accum_dtype)
                            T.sync_threads()
                else:
                    for s_idx, d in T.Parallel(BLOCK_S, BLOCK_D):
                        ts = base_s + s_idx
                        if ts < s_len_var:
                            K_shared[s_idx, d] = K[i_b, ts, i_h, d]
                        else:
                            K_shared[s_idx, d] = 0
                    T.sync_threads()

                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow
                    )
                    T.copy(acc_s, score_shared)

                if is_causal:
                    for i, j in T.Parallel(GEMM_M, BLOCK_S):
                        l_idx = i // groups
                        tq = base_l + l_idx
                        tq_global = q_offset + tq
                        ts = base_s + j
                        if ts >= (tq_global - window_size + 1) // block_size:
                            score_shared[i, j] = -T.infinity(accum_dtype)

                T.sync_threads()

                # R89: parallel max-over-G reduction across all `threads` lanes.
                # For each (l_idx, s_idx) chunk, fold the per-g selection logit
                # (scaled qk + bias - lse_swa) into score_maxg[l_idx, s_idx].
                # Masked (-inf) groups are skipped; fully-masked / OOB chunks
                # stay -inf so they never win topk.
                for l_idx, s_idx in T.Parallel(BLOCK_L, BLOCK_S):
                    ts_p = base_s + s_idx
                    tq_p = base_l + l_idx
                    cur_p = T.alloc_var(accum_dtype)
                    cur_p = -T.infinity(accum_dtype)
                    if ts_p < s_len_var:
                        for g in T.serial(groups):
                            v_p = T.alloc_var(accum_dtype)
                            v_p = score_shared[l_idx * groups + g, s_idx] * sm_scale
                            if v_p != -T.infinity(accum_dtype):
                                if use_bias:
                                    v_p += bias[i_b, ts_p, i_h, g]
                                if use_lse_swa:
                                    v_p -= LSE_SWA[i_b, tq_p, i_h, g]
                                if v_p > cur_p:
                                    cur_p = v_p
                    score_maxg[l_idx, s_idx] = cur_p
                T.sync_threads()

                tx = T.get_thread_binding()
                my_l_idx = tx
                my_tq = base_l + my_l_idx
                cur_max_val = T.alloc_var(accum_dtype)

                if (my_tq < seq_len_var) and (tx < BLOCK_L):
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        if ts < s_len_var:
                            cur_max_val = score_maxg[my_l_idx, s_idx]
                            if cur_max_val > topk_max_scores_local[topk - 1]:
                                moving = T.alloc_var("bool")
                                moving = True
                                for kk in T.serial(topk):
                                    k = topk - 1 - kk
                                    if moving:
                                        if (k > 0) and (cur_max_val > topk_max_scores_local[k - 1]):
                                            topk_max_scores_local[k] = topk_max_scores_local[k - 1]
                                            topk_indices_local[k] = topk_indices_local[k - 1]
                                        else:
                                            topk_max_scores_local[k] = cur_max_val
                                            topk_indices_local[k] = ts
                                            moving = False
                T.sync_threads()

            tx = T.get_thread_binding()
            my_l_idx = tx
            my_tq = base_l + my_l_idx
            if (my_tq < seq_len_var) and (tx < BLOCK_L):
                for k in T.serial(topk):
                    OutIndices[i_b, my_tq, i_h, k] = topk_indices_local[k]


    return fwd_kernel_max_pooling



@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def recompute_topk_max_pooling_scores_kernel(
    batch, seq_len, s_len, h_kv, groups, head_dim, topk,
    BLOCK_L=None, BLOCK_TK=None, threads=None,
    per_qhead_lmks=False, use_head_mask=False,
    is_training=True,
):
    """
    Recompute scores at given topk indices. Outputs raw scaled qk (no bias),
    consistent with the softmax-version's recompute kernel.

    Q:        [B, L, h_kv, G, D]
    K:        [B, S, h_kv, D]   (per-KV-head)  or  [B, S, h_kv*G, D] (per-q-head)
    Indices:  [B, L, h_kv, topk]
    OutScores:[B, L, h_kv, G, topk]
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    if not is_training:
        seq_len_var = T.dynamic("seq_len")
        s_len_var = T.dynamic("s_len")
    else:
        seq_len_var = seq_len
        s_len_var = s_len

    q_shape = [batch, seq_len_var, h_kv, groups, head_dim]
    if per_qhead_lmks:
        k_shape = [batch, s_len_var, h_kv * groups, head_dim]
    else:
        k_shape = [batch, s_len_var, h_kv, head_dim]
    indices_shape = [batch, seq_len_var, h_kv, topk]
    out_scores_shape = [batch, seq_len_var, h_kv, groups, topk]
    head_mask_shape = [h_kv, groups] if use_head_mask else [1, 1]

    if BLOCK_L is None:
        # score_shared in the shared-K path is [BLOCK_L*G, BLOCK_L*BLOCK_TK]
        # fp32 (O(BLOCK_L^2) SMEM growth), so we cannot push BLOCK_L as high
        # as in the selection kernel. Per-q-head path keeps a per-g GEMM and
        # only allocates [BLOCK_L, GEMM_N] fragment, so 16 is fine there.
        BLOCK_L = 16 if per_qhead_lmks else 8
    if BLOCK_TK is None:
        BLOCK_TK = 16
    BLOCK_D = head_dim
    if threads is None:
        threads = 64

    GEMM_M = BLOCK_L * groups
    GEMM_N = BLOCK_L * BLOCK_TK
    tk_blocks = (topk + BLOCK_TK - 1) // BLOCK_TK

    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def fwd_recompute(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        Indices: T.Tensor(indices_shape, idx_dtype),
        HeadMask: T.Tensor(head_mask_shape, idx_dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L
            K_shared = T.alloc_shared([BLOCK_L * BLOCK_TK, BLOCK_D], dtype)

            if per_qhead_lmks:
                # Per-g GEMM staging to keep dynamic shared memory within limits.
                Q_g_shared = T.alloc_shared([BLOCK_L, BLOCK_D], dtype)
                acc_s_g = T.alloc_fragment([BLOCK_L, GEMM_N], accum_dtype)

                for tk_block in T.serial(tk_blocks):
                    tk_base = tk_block * BLOCK_TK
                    tk_size = T.min(BLOCK_TK, topk - tk_base)

                    for g in T.serial(groups):
                        if (not use_head_mask) or (HeadMask[i_h, g] != 0):
                            # Load K for this (tk_block, g)
                            for l_idx, tk_idx, d in T.Parallel(BLOCK_L, BLOCK_TK, BLOCK_D):
                                tq = base_l + l_idx
                                off = l_idx * BLOCK_TK + tk_idx
                                if (tq < seq_len_var) and (tk_idx < tk_size):
                                    k_id = tk_base + tk_idx
                                    idx = Indices[i_b, tq, i_h, k_id]
                                    if (idx >= 0) and (idx < s_len_var):
                                        K_shared[off, d] = K[i_b, idx, i_h * groups + g, d]
                                    else:
                                        K_shared[off, d] = T.Cast(dtype, 0.0)
                                else:
                                    if off < BLOCK_L * BLOCK_TK:
                                        K_shared[off, d] = T.Cast(dtype, 0.0)
                            # Load Q for this g
                            for l_idx, d in T.Parallel(BLOCK_L, BLOCK_D):
                                tq = base_l + l_idx
                                if tq < seq_len_var:
                                    Q_g_shared[l_idx, d] = Q[i_b, tq, i_h, g, d]
                                else:
                                    Q_g_shared[l_idx, d] = T.Cast(dtype, 0.0)
                            T.sync_threads()
                            T.clear(acc_s_g)
                            T.gemm(
                                Q_g_shared,
                                K_shared,
                                acc_s_g,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                            )
                            for l_idx, tk_idx in T.Parallel(BLOCK_L, BLOCK_TK):
                                tq = base_l + l_idx
                                if (tq < seq_len_var) and (tk_idx < tk_size):
                                    k_id = tk_base + tk_idx
                                    idx = Indices[i_b, tq, i_h, k_id]
                                    if idx < 0:
                                        OutScores[i_b, tq, i_h, g, k_id] = -T.infinity(accum_dtype)
                                    else:
                                        col = l_idx * BLOCK_TK + tk_idx
                                        OutScores[i_b, tq, i_h, g, k_id] = acc_s_g[l_idx, col] * sm_scale
                            T.sync_threads()
                        else:
                            for l_idx, tk_idx in T.Parallel(BLOCK_L, BLOCK_TK):
                                tq = base_l + l_idx
                                if (tq < seq_len_var) and (tk_idx < tk_size):
                                    k_id = tk_base + tk_idx
                                    OutScores[i_b, tq, i_h, g, k_id] = -T.infinity(accum_dtype)
                            T.sync_threads()
            else:
                Q_shared = T.alloc_shared([GEMM_M, BLOCK_D], dtype)
                score_shared = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)
                acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)

                for l_idx, g, d in T.Parallel(BLOCK_L, groups, BLOCK_D):
                    tq = base_l + l_idx
                    flat_m = l_idx * groups + g
                    if tq < seq_len_var:
                        Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                    else:
                        Q_shared[flat_m, d] = T.Cast(dtype, 0.0)

                for tk_block in T.serial(tk_blocks):
                    tk_base = tk_block * BLOCK_TK
                    tk_size = T.min(BLOCK_TK, topk - tk_base)

                    for l_idx, tk_idx, d in T.Parallel(BLOCK_L, BLOCK_TK, BLOCK_D):
                        tq = base_l + l_idx
                        off = l_idx * BLOCK_TK + tk_idx
                        if (tq < seq_len_var) and (tk_idx < tk_size):
                            k_id = tk_base + tk_idx
                            idx = Indices[i_b, tq, i_h, k_id]
                            if (idx >= 0) and (idx < s_len_var):
                                K_shared[off, d] = K[i_b, idx, i_h, d]
                            else:
                                K_shared[off, d] = T.Cast(dtype, 0.0)
                        else:
                            if off < BLOCK_L * BLOCK_TK:
                                K_shared[off, d] = T.Cast(dtype, 0.0)
                    T.sync_threads()

                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow
                    )
                    T.copy(acc_s, score_shared)
                    T.sync_threads()

                    for l_idx, g, tk_idx in T.Parallel(BLOCK_L, groups, BLOCK_TK):
                        tq = base_l + l_idx
                        if (tq < seq_len_var) and (tk_idx < tk_size):
                            k_id = tk_base + tk_idx
                            idx = Indices[i_b, tq, i_h, k_id]
                            if idx < 0:
                                OutScores[i_b, tq, i_h, g, k_id] = -T.infinity(accum_dtype)
                            else:
                                row = l_idx * groups + g
                                col = l_idx * BLOCK_TK + tk_idx
                                val = score_shared[row, col]
                                OutScores[i_b, tq, i_h, g, k_id] = val * sm_scale

    return fwd_recompute






# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [1,2,4,8,16,32]
# num_threads = [32,64,128,256]
# _configs = list(
#     itertools.product(
#         BLOCK_L,
#         num_threads,
#     ))

# configs = [
#     {
#         "BLOCK_L": c[0],
#         "num_threads": c[1],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
@tilelang.jit(
    out_idx=[1],
    pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
},
)
def sort_topk_indices_kernel(
    batch: int,
    seq_len: int,
    h_kv: int,
    topk: int,
    BLOCK_L: int = 16,
    num_threads: int = 64,
    is_training: bool = True,
):
    """
    对 IndicesIn: [B, L, h_kv, topk] 按最后一维进行 bitonic 升序排序（按 chunk id）。
    - 有效索引: idx >= 0
    - 无效索引: idx < 0（例如 -1），排序时会被视为最大 key，放在末尾。
    """
    if not is_training:
        seq_len_var = T.dynamic("seq_len")
    else:
        seq_len_var = seq_len
    INVALID_KEY = 0x7FFFFFFF
    BF16 = "bfloat16"
    FP32 = "float32"
    INT32 = "int32"
    assert topk == tilelang.math.next_power_of_2(topk)
    num_iters = int(round(math.log2(topk)))

    indices_shape = [batch, seq_len_var, h_kv, topk]

    @T.prim_func
    def sort_kernel(
        IndicesIn: T.Tensor(indices_shape, INT32),
        IndicesOut: T.Tensor(indices_shape, INT32),
    ):

        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=num_threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L

            idx_shared = T.alloc_shared([BLOCK_L, topk], dtype=INT32)

            for l_idx, k in T.Parallel(BLOCK_L, topk):
                lq = base_l + l_idx
                if lq < seq_len_var:
                    idx_shared[l_idx, k] = IndicesIn[i_b, lq, i_h, k]
                else:
                    idx_shared[l_idx, k] = -1
            T.sync_threads()

            k_step = T.alloc_var(INT32)
            j_step = T.alloc_var(INT32)
            # i_idx = T.alloc_var(INT32)  # 0.1.7.post1和0.1.7.post2需要注释掉
            # ixj = T.alloc_var(INT32)  # 0.1.7.post1和0.1.7.post2需要注释掉
            val_i = T.alloc_var(INT32)
            val_j = T.alloc_var(INT32)
            key_i = T.alloc_var(INT32)
            key_j = T.alloc_var(INT32)

            k_step = 2
            for _ in T.serial(num_iters):
                j_step = k_step // 2
                while j_step > 0:
                    for l_idx, i in T.Parallel(BLOCK_L, topk):
                        lq = base_l + l_idx
                        if lq < seq_len_var:
                            i_idx = i
                            ixj = i_idx ^ j_step
                            if (ixj > i_idx) and (ixj < topk):
                                val_i = idx_shared[l_idx, i_idx]
                                val_j = idx_shared[l_idx, ixj]

                                if val_i >= 0:
                                    key_i = val_i
                                else:
                                    key_i = INVALID_KEY
                                if val_j >= 0:
                                    key_j = val_j
                                else:
                                    key_j = INVALID_KEY

                                up = (i_idx & k_step) == 0

                                do_swap = T.alloc_var("bool")
                                do_swap = False
                                if up:
                                    if key_i > key_j:
                                        do_swap = True
                                else:
                                    if key_i < key_j:
                                        do_swap = True

                                if do_swap:
                                    idx_shared[l_idx, i_idx] = val_j
                                    idx_shared[l_idx, ixj] = val_i
                    T.sync_threads()
                    j_step = j_step // 2
                k_step = k_step * 2

            for l_idx, k in T.Parallel(BLOCK_L, topk):
                lq = base_l + l_idx
                if lq < seq_len_var:
                    IndicesOut[i_b, lq, i_h, k] = idx_shared[l_idx, k]

    return sort_kernel

class TopKMaxPoolingFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, lmks, topk: int,
                select_kernel,
                sort_kernel,
                recompute_kernel,
                lse_swa_arg,
                bias_arg,
                head_mask_arg,
                q_offset_tensor,
                per_qhead_lmks: bool,
                mask_recompute: bool,
                ):
        # q:    [B, L, h_kv, G, D]
        # lmks: [B, S, h_kv, D]  (per-KV-head)  or  [B, S, h_kv*G, D] (per-q-head)
        # lse_swa_arg: [B, L, h_kv, G] float32 when use_lse_swa else [1,1,1,1] dummy
        # bias_arg: [B, S, h_kv, G] float32 when use_bias else [1,1,1,1] dummy
        # head_mask_arg: [h_kv, G] int32 when use_head_mask else [1,1] dummy
        B, L, h_kv, G, D = q.shape
        B2, S, lmks_h, D2 = lmks.shape
        dtype = q.dtype

        assert B == B2 and D == D2
        if per_qhead_lmks:
            assert lmks_h == h_kv * G, (
                f"per_qhead_lmks expects lmks_h={h_kv*G}, got {lmks_h}"
            )
        else:
            assert lmks_h == h_kv, (
                f"shared-K expects lmks_h={h_kv}, got {lmks_h}"
            )

        q_in = q.contiguous()
        k_in = lmks.contiguous()
        lse_swa_in = lse_swa_arg.contiguous()
        bias_in = bias_arg.contiguous()
        head_mask_in = head_mask_arg.contiguous()
        q_offset_in = q_offset_tensor.contiguous()

        indices_raw = select_kernel(q_in, k_in, lse_swa_in, bias_in, head_mask_in, q_offset_in)  # int32
        indices_sorted = sort_kernel(indices_raw)
        indices_for_kernel = indices_sorted

        if mask_recompute:
            recompute_head_mask = head_mask_in
        else:
            # Selection-only mode: recompute intentionally runs all heads for
            # the selected chunks to match the old lse_swa=1e4 poison
            # semantics. If selection produces -1 indices (e.g. an all-local
            # group), recompute naturally outputs -inf.
            recompute_head_mask = torch.ones(1, 1, dtype=torch.int32, device=q_in.device)
        best_scores_buf = recompute_kernel(q_in, k_in, indices_for_kernel, recompute_head_mask)  # float32

        ctx.save_for_backward(q_in, k_in, indices_sorted)
        ctx.h_kv = h_kv
        ctx.G = G
        ctx.topk = topk
        ctx.shapes = (B, L, S, h_kv, D)
        ctx.per_qhead_lmks = per_qhead_lmks

        return indices_sorted, best_scores_buf.to(dtype)

    @staticmethod
    def backward(ctx, grad_indices_unused, grad_scores_selected):
        q_in, k_in, indices = ctx.saved_tensors
        indices = indices.long()
        B, L, S, h_kv, D = ctx.shapes
        G = ctx.G
        per_qhead_lmks = ctx.per_qhead_lmks

        sm_scale = 1.0 / math.sqrt(D)

        grad_scores_dense = torch.zeros(
            (B, L, h_kv, G, S),
            dtype=grad_scores_selected.dtype,
            device=grad_scores_selected.device,
        )

        indices_expanded = indices.unsqueeze(3).expand(-1, -1, -1, G, -1)

        valid_mask = (indices_expanded >= 0) & (indices_expanded < S)
        safe_indices = indices_expanded.clone()
        safe_indices[~valid_mask] = 0
        safe_grad = grad_scores_selected.clone()
        safe_grad[~valid_mask] = 0

        grad_scores_dense.scatter_(4, safe_indices, safe_grad)

        bs_hg = B * h_kv * G
        # dense_in: [B*h_kv*G, L, S]
        dense_in = grad_scores_dense.permute(0, 2, 3, 1, 4).reshape(bs_hg, L, S)
        # q_flat: [B*h_kv*G, L, D]
        q_flat = q_in.permute(0, 2, 3, 1, 4).reshape(bs_hg, L, D)

        if per_qhead_lmks:
            # k_in: [B, S, h_kv*G, D] -> [B, h_kv, G, S, D] -> [B*h_kv*G, S, D]
            k_view = k_in.view(B, S, h_kv, G, D)
            k_flat = k_view.permute(0, 2, 3, 1, 4).reshape(bs_hg, S, D)
        else:
            # k_in: [B, S, h_kv, D] -> [B, S, h_kv, G, D] (broadcast over G)
            k_expanded = k_in.unsqueeze(3).expand(-1, -1, -1, G, -1)
            k_flat = k_expanded.permute(0, 2, 3, 1, 4).reshape(bs_hg, S, D)

        grad_q_flat = torch.bmm(dense_in, k_flat)
        grad_q = grad_q_flat.view(B, h_kv, G, L, D).permute(0, 3, 1, 2, 4)
        grad_q = grad_q * sm_scale

        grad_k_flat = torch.bmm(dense_in.transpose(1, 2), q_flat)  # [B*h_kv*G, S, D]
        grad_k_grouped = grad_k_flat.view(B, h_kv, G, S, D)

        if per_qhead_lmks:
            # No sum over G; output shape [B, S, h_kv*G, D]
            grad_lmks = grad_k_grouped.permute(0, 3, 1, 2, 4).reshape(B, S, h_kv * G, D)
        else:
            grad_k_sum = grad_k_grouped.sum(dim=2)
            grad_lmks = grad_k_sum.permute(0, 2, 1, 3)  # [B, S, h_kv, D]

        grad_lmks = grad_lmks * sm_scale

        # forward signature: (q, lmks, topk, select_kernel, sort_kernel,
        #                    recompute_kernel, lse_swa_arg, bias_arg,
        #                    head_mask_arg, q_offset_tensor,
        #                    per_qhead_lmks, mask_recompute)
        return grad_q, grad_lmks, None, None, None, None, None, None, None, None, None, None


class TopKMaxPooling_Fused(torch.nn.Module):
    def __init__(self, topk, block_size, window_size, is_causal,
                 use_bias: bool = False, per_qhead_lmks: bool = False,
                 is_training: bool = True):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self.use_bias = use_bias
        self.per_qhead_lmks = per_qhead_lmks
        self.is_training = is_training
        self._cached_select_kernel = None
        self._cached_sort_kernel = None
        self._cached_recompute_kernel = None
        self._cached_shape = None

    def forward(self, q, lmks, bias=None, lse_swa=None, head_mask=None, mask_recompute: bool = False, q_offset: int = 0):
        # q:    [B, L, h_kv, G, D]
        # lmks: [B, S, h_kv, D]   (per-KV-head)  or  [B, S, h_kv*G, D] (per-q-head)
        # bias: optional [B, S, h_q] or [B, S, h_kv, G]
        # lse_swa: optional [B, L, h_q] or [B, L, h_kv, G]
        # head_mask: optional [h_q] or [h_kv, G], int/bool. Only the
        #            per_qhead_lmks=True kernels can use it to skip
        #            local-head GEMMs.
        # mask_recompute: if True, recompute also skips inactive heads and
        #                 returns -inf scores for them; default False keeps
        #                 old selection-only mask semantics.
        B, L, h_kv, G, D = q.shape
        _, S, _, _ = lmks.shape
        topk = self.topk
        block_size = self.block_size
        window_size = self.window_size
        is_causal = self.is_causal
        use_bias = self.use_bias
        per_qhead_lmks = self.per_qhead_lmks
        is_training = self.is_training

        # Build bias_arg [B, S, h_kv, G] float32 (or dummy [1,1,1,1] when not used)
        if use_bias:
            assert bias is not None, "use_bias=True but bias is None"
            bias_arg = bias.to(device=q.device, dtype=torch.float32)
            if bias_arg.dim() == 3:
                assert bias_arg.shape == (B, S, h_kv * G), (
                    f"bias shape {tuple(bias_arg.shape)} != ({B}, {S}, {h_kv * G})"
                )
                bias_arg = bias_arg.reshape(B, S, h_kv, G).contiguous()
            elif bias_arg.dim() == 4:
                assert bias_arg.shape == (B, S, h_kv, G), (
                    f"bias shape {tuple(bias_arg.shape)} != ({B}, {S}, {h_kv}, {G})"
                )
                bias_arg = bias_arg.contiguous()
            else:
                raise AssertionError(
                    f"bias must be [B, S, h_q] or [B, S, h_kv, G], got {tuple(bias_arg.shape)}"
                )
        else:
            bias_arg = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=q.device)

        use_lse_swa = lse_swa is not None
        if use_lse_swa:
            lse_swa_arg = lse_swa.to(device=q.device, dtype=torch.float32)
            if lse_swa_arg.dim() == 3:
                assert lse_swa_arg.shape == (B, L, h_kv * G), (
                    f"lse_swa shape {tuple(lse_swa_arg.shape)} != ({B}, {L}, {h_kv * G})"
                )
                lse_swa_arg = lse_swa_arg.reshape(B, L, h_kv, G).contiguous()
            elif lse_swa_arg.dim() == 4:
                assert lse_swa_arg.shape == (B, L, h_kv, G), (
                    f"lse_swa shape {tuple(lse_swa_arg.shape)} != ({B}, {L}, {h_kv}, {G})"
                )
                lse_swa_arg = lse_swa_arg.contiguous()
            else:
                raise AssertionError(
                    f"lse_swa must be [B, L, h_q] or [B, L, h_kv, G], got {tuple(lse_swa_arg.shape)}"
                )
        else:
            lse_swa_arg = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=q.device)

        # Only the per-q-head TileLang path can skip individual heads before
        # GEMM.  For shared-K, keep the old big-GEMM path even if a mask is
        # provided.
        use_head_mask = (head_mask is not None) and per_qhead_lmks
        use_recompute_head_mask = use_head_mask and mask_recompute
        if use_head_mask:
            head_mask_arg = head_mask.to(device=q.device, dtype=torch.int32)
            if head_mask_arg.dim() == 1:
                assert head_mask_arg.numel() == h_kv * G, (
                    f"head_mask shape {tuple(head_mask_arg.shape)} != ({h_kv * G},)"
                )
                head_mask_arg = head_mask_arg.view(h_kv, G)
            elif head_mask_arg.dim() == 2:
                assert tuple(head_mask_arg.shape) == (h_kv, G), (
                    f"head_mask shape {tuple(head_mask_arg.shape)} != ({h_kv}, {G})"
                )
            else:
                raise AssertionError(
                    f"head_mask must be [h_q] or [h_kv, G], got {tuple(head_mask_arg.shape)}"
                )
            head_mask_arg = head_mask_arg.contiguous()
        else:
            head_mask_arg = torch.ones(1, 1, dtype=torch.int32, device=q.device)

        if is_training:
            shape_key = (B, L, S, h_kv, G, D, topk, block_size, window_size,
                         is_causal, is_training, use_bias, use_lse_swa,
                         per_qhead_lmks, use_head_mask, use_recompute_head_mask)
        else:
            shape_key = (B, h_kv, G, D, topk, block_size, window_size,
                         is_causal, is_training, use_bias, use_lse_swa,
                         per_qhead_lmks, use_head_mask, use_recompute_head_mask)
        if self._cached_shape != shape_key:
            seq_len_param = L if is_training else None
            s_len_param = S if is_training else None

            self._cached_select_kernel = fused_topk_max_pooling_kernel(
                B, seq_len_param, s_len_param, h_kv, G, D, topk, block_size, window_size, is_causal,
                use_bias=use_bias, use_lse_swa=use_lse_swa,
                per_qhead_lmks=per_qhead_lmks, use_head_mask=use_head_mask,
                is_training=is_training,
            )

            self._cached_sort_kernel = sort_topk_indices_kernel(
                B, seq_len_param, h_kv, topk,
                is_training=is_training,
            )

            self._cached_recompute_kernel = recompute_topk_max_pooling_scores_kernel(
                B, seq_len_param, s_len_param, h_kv, G, D, topk,
                per_qhead_lmks=per_qhead_lmks,
                use_head_mask=use_recompute_head_mask,
                is_training=is_training,
            )
            self._cached_shape = shape_key

        select_kernel = self._cached_select_kernel
        sort_kernel = self._cached_sort_kernel
        recompute_kernel = self._cached_recompute_kernel
        q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=q.device)

        indices, scores = TopKMaxPoolingFusedFn.apply(
            q, lmks, topk, select_kernel, sort_kernel, recompute_kernel,
            lse_swa_arg, bias_arg, head_mask_arg, q_offset_tensor,
            per_qhead_lmks, use_recompute_head_mask,
        )
        # Reshape scores: [B, L, h_kv, G, topk] -> [B, L, h_q, topk]
        scores = scores.view(B, L, h_kv * G, -1)
        return indices, scores

_MODULE_CACHE = {}
def online_topk_head(q: torch.Tensor, lmks: torch.Tensor, topk: int,
                     block_size: int, window_size: int, is_causal: bool = False,
                     bias: Optional[torch.Tensor] = None,
                     G: Optional[int] = None,
                     lse_swa: Optional[torch.Tensor] = None,
                     head_mask: Optional[torch.Tensor] = None,
                     mask_recompute: bool = False,
                     q_offset: int = 0,
                     is_training: bool = True):
    """
    Functional API for TopKMaxPooling_Fused (max-pooling top-k).

    Args:
        q: [B, L, h_q, D] or [B, L, h_kv, G, D]
        lmks: [B, S, h_kv, D] (per-KV-head) or [B, S, h_q, D] (per-q-head)
        topk: int
        block_size: int
        window_size: int
        is_causal: bool
        bias: optional [B, S, h_q] or [B, S, h_kv, G] additive selection bias.
              Only affects argmax-over-G during selection; returned scores
              stay raw scaled qk and do NOT include bias.
        G: optional explicit query group size. When G>1 and lmks_h==h_q,
           takes the per-q-head path (no K sharing across G).
        lse_swa: optional [B, L, h_q] or [B, L, h_kv, G]. Only affects
                 argmax-over-G during selection; returned scores stay raw
                 scaled qk and do NOT include lse_swa.
        head_mask: optional [h_q] or [h_kv, G]. Only the per-q-head lmks
                   kernels can use it to skip local-head GEMMs.
        mask_recompute: if True, recompute also skips inactive heads and
                        returns -inf scores for them. Default False keeps
                        selection-only mask semantics.
        q_offset: global query offset for inference causal masking.
        is_training: False enables dynamic seq_len/s_len kernels for inference.

    Returns:
        indices: [B, L, h_kv, topk]
        scores:  [B, L, h_q, topk]   (raw scaled qk)
    """
    if q.dim() == 4:
        B, L, h_q, D = q.shape
        lmks_h = lmks.shape[2]
        if G is None or G == 1:
            assert h_q % lmks_h == 0, (
                f"h_q ({h_q}) must be divisible by lmks_h ({lmks_h})"
            )
            h_kv = lmks_h
            G_eff = h_q // h_kv
            per_qhead_lmks = False
        else:
            assert lmks_h == h_q, (
                f"when G>1, lmks must have h_q ({h_q}) heads, got {lmks_h}"
            )
            assert h_q % G == 0, f"h_q ({h_q}) must be divisible by G ({G})"
            h_kv = h_q // G
            G_eff = G
            per_qhead_lmks = True
        q = q.view(B, L, h_kv, G_eff, D)
    else:
        B, L, h_kv, G_in, D = q.shape
        h_q = h_kv * G_in
        lmks_h = lmks.shape[2]
        if G is None or G == 1:
            if lmks_h == h_kv:
                per_qhead_lmks = False
            elif lmks_h == h_q:
                per_qhead_lmks = True
            else:
                raise AssertionError(
                    f"lmks_h ({lmks_h}) must be h_kv ({h_kv}) or h_q ({h_q})"
                )
        else:
            assert lmks_h == h_q, (
                f"when G>1, lmks must have h_q ({h_q}) heads, got {lmks_h}"
            )
            assert h_q % G == 0
            new_h_kv = h_q // G
            if new_h_kv != h_kv:
                q = q.reshape(B, L, new_h_kv, G, D)
                h_kv = new_h_kv
            per_qhead_lmks = True

    use_bias = bias is not None
    use_lse_swa = lse_swa is not None
    cache_key = (topk, block_size, window_size, is_causal, is_training,
                 use_bias, use_lse_swa, per_qhead_lmks)
    if cache_key not in _MODULE_CACHE:
        _MODULE_CACHE[cache_key] = TopKMaxPooling_Fused(
            topk, block_size, window_size, is_causal,
            use_bias=use_bias, per_qhead_lmks=per_qhead_lmks,
            is_training=is_training,
        )

    return _MODULE_CACHE[cache_key](
        q, lmks, bias=bias, lse_swa=lse_swa,
        head_mask=head_mask, mask_recompute=mask_recompute,
        q_offset=q_offset,
    )


# ----------------------------------------------------------------------
# Tests (ported & adapted from topk_head_softmax.py to cover bias and
# per_qhead_lmks for the max-pooling top-k path).
# ----------------------------------------------------------------------
def test_train_inference_correctness(test_name, B, q_len, h_kv, G, D, topk,
                                     block_size, window_size,
                                     use_bias, per_qhead_lmks=False, is_causal=True,
                                     use_lse_swa=False):
    """
    Forward + backward correctness test for the max-pooling top-k kernel.
    Adapted from topk_head_softmax.py:test_train_inference_correctness.
    This test keeps training-style coverage of bias, lse_swa, and
    per_qhead_lmks combinations; dense inference coverage is below.
    """
    device = "cuda"
    dtype = torch.bfloat16
    h_q = h_kv * G

    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"Config: B={B}, q_len={q_len}, h_kv={h_kv}, G={G}, D={D}")
    print(f"        topk={topk}, block_size={block_size}, window_size={window_size}")
    print(f"        is_causal={is_causal}, use_bias={use_bias}, use_lse_swa={use_lse_swa}, per_qhead_lmks={per_qhead_lmks}")
    print(f"{'='*70}")

    torch.manual_seed(42)

    # S = q_len // block_size (kv landmarks)
    S = q_len // block_size

    # Q: [B, q_len, h_kv, G, D]
    q_raw = torch.randn(B, q_len, h_kv, G, D, dtype=dtype, device=device)
    # Lmks: per-KV-head [B, S, h_kv, D]; per-q-head [B, S, h_q, D]
    if per_qhead_lmks:
        lmks_raw = torch.randn(B, S, h_q, D, dtype=dtype, device=device)
    else:
        lmks_raw = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)

    # Input isolation
    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)
    q_ref = q_raw.clone().detach().requires_grad_(True)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(True)

    if use_bias:
        # Deterministic per-chunk per-head additive bias.
        chunk_axis = torch.linspace(-1.0, 1.0, S, device=device, dtype=torch.float32).view(1, S, 1)
        head_axis = torch.linspace(-0.5, 0.5, h_kv * G, device=device, dtype=torch.float32).view(1, 1, h_kv * G)
        batch_axis = torch.arange(B, device=device, dtype=torch.float32).view(B, 1, 1) * 0.01
        bias = (chunk_axis + head_axis + batch_axis).contiguous()
    else:
        bias = None

    if use_lse_swa:
        query_axis = torch.linspace(-0.75, 0.75, q_len, device=device, dtype=torch.float32).view(1, q_len, 1)
        head_axis = torch.linspace(-1.0, 1.0, h_q, device=device, dtype=torch.float32).view(1, 1, h_q)
        batch_axis = torch.arange(B, device=device, dtype=torch.float32).view(B, 1, 1) * 0.03
        lse_swa = (query_axis + head_axis + batch_axis).contiguous()
    else:
        lse_swa = None

    G_arg = G if per_qhead_lmks else None

    # ============ Forward ============
    print("\n--- Forward Pass ---")

    indices_ref, scores_ref = ref_topk_max_pooling(
        q_ref, lmks_ref, topk, block_size, window_size, is_causal,
        bias=bias, lse_swa=lse_swa,
    )

    indices_fused, scores_fused = online_topk_head(
        q_fused, lmks_fused, topk, block_size, window_size, is_causal,
        bias=bias, G=G_arg, lse_swa=lse_swa,
    )

    def get_abs_err(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0:
            return 0.0
        return (x[mask] - y[mask]).abs().max().item()

    def get_err_ratio(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0:
            return 0.0
        err = (x[mask] - y[mask]).square().mean().sqrt().item()
        base = x[mask].square().mean().sqrt().item()
        return err / (base + 1e-12)

    # Reshape fused scores to [B, q_len, h_kv, G, topk] for comparison
    scores_fused_reshaped = scores_fused.view(B, q_len, h_kv, G, topk)

    fwd_abs_err = get_abs_err(scores_ref.float(), scores_fused_reshaped.float())
    fwd_rel_err = get_err_ratio(scores_ref.float(), scores_fused_reshaped.float())

    print(f"Forward Scores - Abs Error: {fwd_abs_err:.6f}")
    print(f"Forward Scores - Rel Error: {fwd_rel_err:.6f}")

    # Indices match rate (where ref has valid pooled scores)
    indices_match = (indices_fused.long() == indices_ref.long())
    scores_ref_pooled = scores_ref.max(dim=3).values
    valid_mask = scores_ref_pooled > -1e5
    if valid_mask.sum() > 0:
        match_rate = indices_match[valid_mask].float().mean().item()
        print(f"Indices Match Rate: {match_rate*100:.2f}%")
    else:
        match_rate = 1.0
        print("Indices Match Rate: N/A (all masked)")

    assert fwd_rel_err < 0.02, f"Forward relative error too large: {fwd_rel_err}"
    print("✅ Forward PASSED")

    # ============ Backward ============
    print("\n--- Backward Pass ---")
    grad_output = torch.randn_like(scores_fused, dtype=dtype)
    with torch.no_grad():
        invalid_mask = scores_ref < -1e5
        grad_output_view = grad_output.view(B, q_len, h_kv, G, topk)
        grad_output_view[invalid_mask] = 0.0

    scores_fused.backward(grad_output)
    dq_fused = q_fused.grad.clone()
    dlmks_fused = lmks_fused.grad.clone()

    scores_ref.backward(grad_output.view(B, q_len, h_kv, G, topk))
    dq_ref = q_ref.grad.clone()
    dlmks_ref = lmks_ref.grad.clone()

    dq_fused = torch.nan_to_num(dq_fused, 0.0)
    dq_ref = torch.nan_to_num(dq_ref, 0.0)
    dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
    dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)

    dq_rel_err = get_err_ratio(dq_ref.float(), dq_fused.float())
    dlmks_rel_err = get_err_ratio(dlmks_ref.float(), dlmks_fused.float())

    print(f"dQ Rel Error: {dq_rel_err:.6f}")
    print(f"dLmks Rel Error: {dlmks_rel_err:.6f}")

    assert dq_rel_err < 0.05, f"dQ relative error too large: {dq_rel_err}"
    assert dlmks_rel_err < 0.05, f"dLmks relative error too large: {dlmks_rel_err}"
    print("✅ Backward PASSED")

    print(f"\n✅ Test '{test_name}' PASSED")


def test_fused_topk_max_pooling_correctness():
    print("\n" + "=" * 70)
    print("=== Testing Fused TopK Max-Pooling Kernel Correctness (legacy) ===")
    print("=" * 70)

    B, L, D = 2, 1024, 128
    h_kv = 2
    G = 8
    h_q = h_kv * G
    S = 64
    topk = 16
    is_causal = True
    block_size = 16
    window_size = 64

    dtype = torch.bfloat16
    device = "cuda"

    print(f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}")

    torch.manual_seed(4200)

    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)

    # ============ Forward Correctness ============
    print("\n--- Forward Correctness ---")

    ref_indices, ref_scores = ref_topk_max_pooling(q.detach(), lmks.detach(), topk, block_size, window_size, is_causal)
    fused_indices, fused_scores = online_topk_head(q, lmks, topk, block_size, window_size, is_causal)
    fused_scores_reshaped = fused_scores.view(B, L, h_kv, G, topk)

    diff_mask = (ref_scores > -1e9) & (fused_scores_reshaped.float() > -1e9)
    if diff_mask.sum() == 0:
        max_score_diff = 0.0
        rel_l2_score = 0.0
    else:
        score_diff = torch.abs(ref_scores[diff_mask] - fused_scores_reshaped.float()[diff_mask])
        max_score_diff = score_diff.max().item()
        rel_l2_score = score_diff.norm().item() / (ref_scores[diff_mask].norm().item() + 1e-6)

    print(f"Forward scores (valid only) - Max Diff: {max_score_diff:.6f}")
    print(f"Forward scores (valid only) - L2 RelErr: {rel_l2_score:.6f}")

    indices_match = (fused_indices.long() == ref_indices.long())
    ref_scores_pooled = ref_scores.max(dim=3).values
    valid_indices_mask = (ref_scores_pooled > -1e9)
    if valid_indices_mask.sum() > 0:
        match_rate = indices_match[valid_indices_mask].float().mean().item()
    else:
        match_rate = 1.0
    print(f"Indices Match Rate (Valid Elements): {match_rate*100:.6f}%")

    if match_rate >= 0.99 and rel_l2_score < 1e-2:
        print("✅ Fused Forward PASSED")
    else:
        print("❌ Fused Forward FAILED")


def test_head_mask_per_qhead_lmks_correctness():
    print("\n" + "=" * 70)
    print("=== Testing head_mask correctness (per_qhead_lmks=True) ===")
    print("=" * 70)

    device = "cuda"
    dtype = torch.bfloat16
    B, L, h_kv, G, D = 2, 512, 2, 8, 64
    S, topk = 64, 8
    block_size, window_size = 16, 64
    is_causal = True
    h_q = h_kv * G

    torch.manual_seed(20260601)
    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device)
    lmks = torch.randn(B, S, h_q, D, dtype=dtype, device=device)
    bias = torch.randn(B, S, h_q, dtype=torch.float32, device=device) * 0.25
    lse_swa = torch.randn(B, L, h_q, dtype=torch.float32, device=device) * 2.0 + 5.0

    # Per-layer mask semantics: one mask row per KV group.  Group 0 keeps
    # two heads; group 1 keeps three heads.  This intentionally exercises
    # padding-free kernel-side skipping instead of host-side compacting.
    head_mask = torch.tensor(
        [
            [1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1, 0],
        ],
        dtype=torch.int32,
        device=device,
    )

    indices_ref, scores_ref = ref_topk_max_pooling(
        q, lmks, topk, block_size, window_size, is_causal,
        bias=bias, lse_swa=lse_swa, head_mask=head_mask,
    )
    indices_fused, scores_fused = online_topk_head(
        q, lmks, topk, block_size, window_size, is_causal,
        bias=bias, G=G, lse_swa=lse_swa, head_mask=head_mask,
    )
    scores_fused = scores_fused.view(B, L, h_kv, G, topk).float()

    valid_scores = (scores_ref > -1e9) & (scores_fused > -1e9)
    if valid_scores.any():
        max_diff = (scores_ref[valid_scores] - scores_fused[valid_scores]).abs().max().item()
        rel_l2 = (
            (scores_ref[valid_scores] - scores_fused[valid_scores]).float().norm().item()
            / (scores_ref[valid_scores].float().norm().item() + 1e-6)
        )
    else:
        max_diff = 0.0
        rel_l2 = 0.0

    valid_indices = scores_ref.max(dim=3).values > -1e9
    if valid_indices.any():
        match_rate = (indices_ref.long() == indices_fused.long())[valid_indices].float().mean().item()
    else:
        match_rate = 1.0

    print(f"Indices Match Rate: {match_rate * 100:.4f}%")
    print(f"Scores Max Diff: {max_diff:.6f}")
    print(f"Scores Rel L2: {rel_l2:.6f}")

    assert match_rate >= 0.99, f"indices mismatch too high: {match_rate}"
    assert rel_l2 < 0.02, f"score rel_l2 too high: {rel_l2}"
    print("✅ head_mask per_qhead_lmks correctness PASSED")


def _assert_topk_outputs_close(name, indices_ref, scores_ref, indices_fused, scores_fused, h_kv, G, topk, ratio_thr=0.03):
    scores_fused = scores_fused.view(scores_ref.shape[0], scores_ref.shape[1], h_kv, G, topk).float()
    valid_scores = (scores_ref > -1e9) & (scores_fused > -1e9)
    if valid_scores.any():
        rel_l2 = (
            (scores_ref[valid_scores] - scores_fused[valid_scores]).float().norm().item()
            / (scores_ref[valid_scores].float().norm().item() + 1e-6)
        )
    else:
        rel_l2 = 0.0
    valid_indices = scores_ref.max(dim=3).values > -1e9
    if valid_indices.any():
        match_rate = (indices_ref.long() == indices_fused.long())[valid_indices].float().mean().item()
    else:
        match_rate = 1.0
    print(f"{name}: indices_match={match_rate * 100:.4f}% score_rel_l2={rel_l2:.6f}")
    assert match_rate >= 0.99, f"{name}: indices mismatch too high: {match_rate}"
    assert rel_l2 < ratio_thr, f"{name}: score rel_l2 too high: {rel_l2}"


def test_dense_inference_q_offset_matches_full_prefill():
    print("\n" + "=" * 70)
    print("=== Testing dense inference q_offset matches full prefill ===")
    print("=" * 70)

    device = "cuda"
    dtype = torch.bfloat16
    B, L_total, h_kv, G, D = 2, 1024, 2, 4, 64
    block_size, window_size, topk = 16, 64, 8
    h_q = h_kv * G
    S = L_total // block_size
    torch.manual_seed(20260602)

    q_full = torch.randn(B, L_total, h_kv, G, D, dtype=dtype, device=device)
    lmks_perq = torch.randn(B, S, h_q, D, dtype=dtype, device=device)
    lmks_shared = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)
    bias = torch.randn(B, S, h_q, dtype=torch.float32, device=device) * 0.1
    lse_swa_full = torch.randn(B, L_total, h_q, dtype=torch.float32, device=device) * 0.2 + 4.0

    for per_qhead_lmks, lmks, g_arg in [
        (True, lmks_perq, G),
        (False, lmks_shared, None),
    ]:
        label = "per_qhead" if per_qhead_lmks else "shared_k"
        indices_full, scores_full = online_topk_head(
            q_full, lmks, topk, block_size, window_size, is_causal=True,
            bias=bias, G=g_arg, lse_swa=lse_swa_full,
            is_training=True, q_offset=0,
        )
        scores_full = scores_full.view(B, L_total, h_kv, G, topk).float()
        for q_len in [1, 7, 33, 128]:
            q_offset = L_total - q_len
            q_slice = q_full[:, q_offset:]
            lse_swa_slice = lse_swa_full[:, q_offset:]
            indices_inf, scores_inf = online_topk_head(
                q_slice, lmks, topk, block_size, window_size, is_causal=True,
                bias=bias, G=g_arg, lse_swa=lse_swa_slice,
                is_training=False, q_offset=q_offset,
            )
            _assert_topk_outputs_close(
                f"{label}_q{q_len}",
                indices_full[:, q_offset:], scores_full[:, q_offset:],
                indices_inf, scores_inf, h_kv, G, topk,
            )
    print("✅ dense inference q_offset correctness PASSED")


def test_dense_inference_head_mask_recompute_correctness():
    print("\n" + "=" * 70)
    print("=== Testing dense inference head_mask + mask_recompute correctness ===")
    print("=" * 70)

    device = "cuda"
    dtype = torch.bfloat16
    B, L_total, q_len, h_kv, G, D = 1, 768, 97, 2, 4, 64
    block_size, window_size, topk = 16, 64, 8
    h_q = h_kv * G
    S = L_total // block_size
    q_offset = L_total - q_len
    torch.manual_seed(20260603)

    q = torch.randn(B, q_len, h_kv, G, D, dtype=dtype, device=device)
    lmks = torch.randn(B, S, h_q, D, dtype=dtype, device=device)
    bias = torch.randn(B, S, h_q, dtype=torch.float32, device=device) * 0.1
    lse_swa = torch.randn(B, q_len, h_q, dtype=torch.float32, device=device) * 0.2 + 4.0
    head_mask = torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0]], dtype=torch.int32, device=device)

    for mask_recompute in [False, True]:
        indices_ref, scores_ref = ref_topk_max_pooling(
            q, lmks, topk, block_size, window_size, is_causal=True,
            q_offset=q_offset, bias=bias, lse_swa=lse_swa,
            head_mask=head_mask, mask_recompute=mask_recompute,
        )
        indices_fused, scores_fused = online_topk_head(
            q, lmks, topk, block_size, window_size, is_causal=True,
            bias=bias, G=G, lse_swa=lse_swa, head_mask=head_mask,
            mask_recompute=mask_recompute, q_offset=q_offset,
            is_training=False,
        )
        _assert_topk_outputs_close(
            f"mask_recompute={mask_recompute}",
            indices_ref, scores_ref, indices_fused, scores_fused, h_kv, G, topk,
        )
        if mask_recompute:
            scores_view = scores_fused.view(B, q_len, h_kv, G, topk)
            inactive = ~head_mask.bool().view(1, 1, h_kv, G, 1)
            assert torch.isneginf(scores_view.masked_select(inactive.expand_as(scores_view))).all()
    print("✅ dense inference head_mask correctness PASSED")


def test_dense_inference_dynamic_shape_reuse():
    print("\n" + "=" * 70)
    print("=== Testing dense inference dynamic shape reuse ===")
    print("=" * 70)

    device = "cuda"
    dtype = torch.bfloat16
    B, h_kv, G, D = 1, 2, 4, 64
    block_size, window_size, topk = 16, 64, 8
    h_q = h_kv * G
    mod = TopKMaxPooling_Fused(
        topk, block_size, window_size, is_causal=True,
        use_bias=True, per_qhead_lmks=True, is_training=False,
    ).to(device)
    cached_key = None
    for L_total, q_len in [(512, 1), (1024, 17), (1536, 128)]:
        S = L_total // block_size
        q_offset = L_total - q_len
        torch.manual_seed(20260604 + L_total + q_len)
        q = torch.randn(B, q_len, h_kv, G, D, dtype=dtype, device=device)
        lmks = torch.randn(B, S, h_q, D, dtype=dtype, device=device)
        bias = torch.randn(B, S, h_q, dtype=torch.float32, device=device) * 0.1
        lse_swa = torch.randn(B, q_len, h_q, dtype=torch.float32, device=device) * 0.2 + 4.0
        indices_ref, scores_ref = ref_topk_max_pooling(
            q, lmks, topk, block_size, window_size, is_causal=True,
            q_offset=q_offset, bias=bias, lse_swa=lse_swa,
        )
        indices_fused, scores_fused = mod(
            q, lmks, bias=bias, lse_swa=lse_swa, q_offset=q_offset
        )
        _assert_topk_outputs_close(
            f"dynamic_L{q_len}_S{S}",
            indices_ref, scores_ref, indices_fused, scores_fused, h_kv, G, topk,
        )
        if cached_key is None:
            cached_key = mod._cached_shape
        else:
            assert mod._cached_shape == cached_key, "dynamic inference kernel should be reused across L/S"
    print("✅ dense inference dynamic shape reuse PASSED")


def test_fused_topk_max_pooling_memory_and_speed(
    name: str = "default",
    B: int = 4, L: int = 8192, D: int = 128, h_kv: int = 2, G: int = 16,
    S: int = 128, topk: int = 16, block_size: int = 64, window_size: int = 512,
    is_causal: bool = True,
    use_bias: bool = False,
    use_lse_swa: bool = False,
    per_qhead_lmks: bool = True,
    n_iters: int = 20, n_warmup: int = 5,
    skip_ref: bool = False,
    pass_G: bool = None,  # whether to pass G to online_topk_head; defaults to True iff per_qhead_lmks
):
    """Benchmark fused vs reference impl for max-pooling top-k (no online softmax)."""
    print("\n" + "=" * 70)
    print(f"=== Benchmark [{name}] Fused TopK Max-Pooling (no softmax) ===")
    print("=" * 70)

    h_q = h_kv * G

    dtype = torch.bfloat16
    device = "cuda"

    if pass_G is None:
        pass_G = per_qhead_lmks
    G_arg = G if pass_G else None

    print(
        f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, "
        f"topk={topk}, block_size={block_size}, window_size={window_size}, "
        f"use_bias={use_bias}, use_lse_swa={use_lse_swa}, "
        f"per_qhead_lmks={per_qhead_lmks}, pass_G={pass_G}"
    )

    torch.manual_seed(42)
    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device)
    if per_qhead_lmks:
        lmks = torch.randn(B, S, h_kv * G, D, dtype=dtype, device=device)
    else:
        lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)

    # lse_swa is optional for the max-pooling top-k path. When enabled, use
    # [B, L, h_q] shape (online_topk_head accepts h_q or h_kv*G).
    if use_lse_swa:
        lse_swa = torch.randn(B, L, h_q, dtype=torch.float32, device=device) * 5 + 10
    else:
        lse_swa = None

    # grad_output matches fused score shape: [B, L, h_q, topk]
    grad_output = torch.randn(B, L, h_q, topk, dtype=dtype, device=device)
    grad_output_ref = grad_output.view(B, L, h_kv, G, topk)

    bias = None
    if use_bias:
        # bias shape [B, S, h_q] is also accepted by both fused and ref
        bias = torch.randn(B, S, h_q, dtype=torch.float32, device=device) * 0.5

    def _make_inputs():
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)
        lse_swa_t = lse_swa.detach().clone() if lse_swa is not None else None
        return q_t, lmks_t, lse_swa_t

    def run_fused():
        q_t, lmks_t, lse_swa_t = _make_inputs()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # JIT compile + first call
        _, scores = online_topk_head(
            q_t, lmks_t, topk, block_size, window_size, is_causal,
            bias=bias, G=G_arg, lse_swa=lse_swa_t,
        )
        loss = (scores * grad_output).sum()
        loss.backward()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(n_warmup):
            q_t.grad = None
            lmks_t.grad = None
            _, scores = online_topk_head(
                q_t, lmks_t, topk, block_size, window_size, is_causal,
                bias=bias, G=G_arg, lse_swa=lse_swa_t,
            )
            loss = (scores * grad_output).sum()
            loss.backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        _, scores = online_topk_head(
            q_t, lmks_t, topk, block_size, window_size, is_causal,
            bias=bias, G=G_arg, lse_swa=lse_swa_t,
        )
        loss = (scores * grad_output).sum()
        loss.backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)

        # Fwd only
        start_fwd.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            _ = online_topk_head(
                q_t, lmks_t, topk, block_size, window_size, is_causal,
                bias=bias, G=G_arg, lse_swa=lse_swa_t,
            )
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters

        # Fwd + Bwd
        start_all.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            _, scores = online_topk_head(
                q_t, lmks_t, topk, block_size, window_size, is_causal,
                bias=bias, G=G_arg, lse_swa=lse_swa_t,
            )
            loss = (scores * grad_output).sum()
            loss.backward()
        end_all.record()
        torch.cuda.synchronize()
        avg_all_ms = start_all.elapsed_time(end_all) / n_iters

        avg_bwd_ms = avg_all_ms - avg_fwd_ms
        return peak_mem, avg_fwd_ms, avg_all_ms, avg_bwd_ms

    def run_ref():
        q_t, lmks_t, lse_swa_t = _make_inputs()
        # ref_topk_max_pooling internally upcasts to float; pass tensors as-is
        lse_swa_ref = lse_swa_t.float() if lse_swa_t is not None else None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def forward_only():
            _ = ref_topk_max_pooling(
                q_t, lmks_t, topk, block_size, window_size, is_causal,
                bias=bias, lse_swa=lse_swa_ref,
            )

        def forward_backward():
            _, scores = ref_topk_max_pooling(
                q_t, lmks_t, topk, block_size, window_size, is_causal,
                bias=bias, lse_swa=lse_swa_ref,
            )
            loss = (scores * grad_output_ref).sum()
            loss.backward()

        for _ in range(n_warmup):
            q_t.grad = None
            lmks_t.grad = None
            forward_only()
            forward_backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        forward_backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)

        start_fwd.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            forward_only()
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters

        start_all.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            forward_backward()
        end_all.record()
        torch.cuda.synchronize()
        avg_all_ms = start_all.elapsed_time(end_all) / n_iters

        avg_bwd_ms = avg_all_ms - avg_fwd_ms
        return peak_mem, avg_fwd_ms, avg_all_ms, avg_bwd_ms

    print("\nRunning benchmarks...")

    mem_fused, fwd_fused, all_fused, bwd_fused = run_fused()
    print(f"\n[Fused TopK Max-Pooling]")
    print(f"  Peak Memory: {mem_fused:.2f} MB")
    print(f"  Avg Fwd Latency: {fwd_fused:.2f} ms")
    print(f"  Avg Fwd+Bwd Latency: {all_fused:.2f} ms")
    print(f"  Derived Bwd Latency: {bwd_fused:.2f} ms")

    if skip_ref:
        return

    try:
        mem_ref, fwd_ref, all_ref, bwd_ref = run_ref()
        print(f"\n[Reference (PyTorch)]")
        print(f"  Peak Memory: {mem_ref:.2f} MB")
        print(f"  Avg Fwd Latency: {fwd_ref:.2f} ms")
        print(f"  Avg Fwd+Bwd Latency: {all_ref:.2f} ms")
        print(f"  Derived Bwd Latency: {bwd_ref:.2f} ms")

        print("\n" + "-" * 70)
        print("Comparison:")
        print("-" * 70)
        print(f"{'Method':<25} {'Memory (MB)':<15} {'Fwd (ms)':<12} {'Fwd+Bwd (ms)':<15} {'Bwd (ms)':<12}")
        print("-" * 70)
        print(f"{'Fused':<25} {mem_fused:<15.2f} {fwd_fused:<12.2f} {all_fused:<15.2f} {bwd_fused:<12.2f}")
        print(f"{'Reference':<25} {mem_ref:<15.2f} {fwd_ref:<12.2f} {all_ref:<15.2f} {bwd_ref:<12.2f}")
        print("-" * 70)
        print(f"Speedup (Fwd): {fwd_ref / max(fwd_fused, 1e-6):.2f}x")
        print(f"Speedup (Bwd): {bwd_ref / max(bwd_fused, 1e-6):.2f}x")
        print(f"Speedup (Fwd+Bwd): {all_ref / max(all_fused, 1e-6):.2f}x")
        print(f"Memory Saving: {mem_ref / max(mem_fused, 1e-6):.2f}x")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[Reference (PyTorch)] OOM - Cannot run with this config")
            print("\n" + "-" * 70)
            print("Comparison (Reference OOM):")
            print("-" * 70)
            print(f"{'Method':<25} {'Memory (MB)':<15} {'Fwd (ms)':<12} {'Fwd+Bwd (ms)':<15} {'Bwd (ms)':<12}")
            print("-" * 70)
            print(f"{'Fused':<25} {mem_fused:<15.2f} {fwd_fused:<12.2f} {all_fused:<15.2f} {bwd_fused:<12.2f}")
            print("-" * 70)
        else:
            raise e


if __name__ == "__main__":
    # print("\n" + "=" * 70)
    # print("Regression Suite for topk_head (max-pooling, with bias / per-q-head lmks)")
    # print("=" * 70)

    # # Legacy quick check
    # test_fused_topk_max_pooling_correctness()

    # # 1) Baseline: per-KV-head lmks, no bias
    # test_train_inference_correctness(
    #     "train_basic", 2, 1024, 2, 8, 128, 16, 64, 64, False, False
    # )

    # # 2) Per-KV-head lmks + bias  (the missing-bias path we are now fixing)
    # test_train_inference_correctness(
    #     "train_basic_bias", 2, 1024, 2, 8, 128, 16, 64, 64, True, False
    # )

    # # 3) Per-q-head lmks (no bias)
    # test_train_inference_correctness(
    #     "train_perqhead", 2, 1024, 2, 8, 128, 16, 64, 64, False, True
    # )

    # # 4) Per-q-head lmks + bias (most general path)
    # test_train_inference_correctness(
    #     "train_perqhead_bias", 2, 1024, 2, 8, 128, 16, 64, 64, True, True
    # )

    # # 5) Non-divisible q_len (padding inside kernel)
    # test_train_inference_correctness(
    #     "train_bias_nondiv_999", 2, 999, 2, 8, 128, 16, 64, 64, True, False
    # )

    # # 6) G=1 corner case + per_qhead + bias
    # test_train_inference_correctness(
    #     "train_perqhead_G1_bias", 2, 1024, 4, 1, 128, 16, 64, 64, True, True
    # )

    # # 7) Per-KV-head lmks + bias + lse_swa
    # test_train_inference_correctness(
    #     "train_basic_bias_lse_swa", 2, 1024, 2, 8, 128, 16, 64, 64, True, False,
    #     use_lse_swa=True,
    # )

    # # 8) Per-q-head lmks + bias + lse_swa
    # test_train_inference_correctness(
    #     "train_perqhead_bias_lse_swa", 2, 1024, 2, 8, 128, 16, 64, 64, True, True,
    #     use_lse_swa=True,
    # )

    # print("\n" + "=" * 70)
    # print("All regression tests finished.")
    # print("=" * 70)

    # ---------------- Speed / memory benchmarks ----------------
    test_head_mask_per_qhead_lmks_correctness()
    test_dense_inference_q_offset_matches_full_prefill()
    test_dense_inference_head_mask_recompute_correctness()
    test_dense_inference_dynamic_shape_reuse()


    # Same shape but with bias + lse_swa enabled (closer to real HSA usage).
    # test_fused_topk_max_pooling_memory_and_speed(
    #     name="default_perqhead_bias_lse_swa",
    #     B=2, L=32768, D=128, h_kv=2, G=16,
    #     S=512, topk=16, block_size=64, window_size=512,
    #     is_causal=True, use_bias=True, use_lse_swa=True,
    #     per_qhead_lmks=True,
    #     n_iters=20, n_warmup=5,
    #     skip_ref=True,
    # )
