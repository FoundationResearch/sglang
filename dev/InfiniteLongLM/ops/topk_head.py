import torch
import tilelang
import tilelang.language as T
from typing import Optional
import math

def ref_topk_max_pooling(q, k_lmks, topk, block_size, window_size, is_causal=False, bias=None):
    """
    Reference implementation for max-pooling Top-K (no online softmax, no SWA LSE).

    Selection rule: argmax_g (sm_scale * qk + bias) over groups G, then top-k over chunks S.
    Output scores are raw scaled qk (sm_scale * qk), WITHOUT bias, to match the
    softmax-version's recompute output convention.

    Args:
        q: [B, L, h_kv, G, D]
        k_lmks: [B, S, h_kv, D] (per-KV-head shared lmks) or
                [B, S, h_kv * G, D] (per-q-head lmks)
        topk: int
        block_size: int
        window_size: int
        is_causal: bool
        bias: optional additive bias with shape [B, S, h_kv, G] or [B, S, h_kv * G].
              When provided, selection logits add bias[b, chunk, head]; returned
              scores stay raw scaled qk and do not include bias.

    Returns:
        indices_sorted: [B, L, h_kv, topk]
        scores_sorted:  [B, L, h_kv, G, topk]   (raw scaled qk, NO bias)
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
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)  # [1, S]
        threshold_idx = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
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

    # max over G with bias-aware selection logits
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
    return indices_sorted, scores_sorted


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_topk_max_pooling_kernel(batch, seq_len, s_len, h_kv, groups, head_dim, topk,
                                  block_size, window_size, is_causal,
                                  BLOCK_L=None, BLOCK_S=None, threads=None,
                                  use_bias=False, per_qhead_lmks=False):
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    q_shape = [batch, seq_len, h_kv, groups, head_dim]
    if per_qhead_lmks:
        k_shape = [batch, s_len, h_kv * groups, head_dim]
    else:
        k_shape = [batch, s_len, h_kv, head_dim]
    out_indices_shape = [batch, seq_len, h_kv, topk]
    bias_shape = [batch, s_len, h_kv, groups] if use_bias else [1, 1, 1, 1]

    if BLOCK_L is None:
        # GEMM requires M % 16 == 0. For shared-K path M = BLOCK_L*groups, so
        # use the smallest BLOCK_L that makes M >= 16. For per-q-head path the
        # GEMM is per-g with M = BLOCK_L, so BLOCK_L itself must be >= 16.
        BLOCK_L = 16 if per_qhead_lmks else (16 + groups - 1) // groups
    if BLOCK_S is None:
        BLOCK_S = 16
    BLOCK_D = head_dim
    if threads is None:
        threads = 64

    GEMM_M = BLOCK_L * groups
    num_s_blocks = tilelang.cdiv(s_len, BLOCK_S)

    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def fwd_kernel_max_pooling(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        bias: T.Tensor(bias_shape, accum_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L

            Q_shared = T.alloc_shared([GEMM_M, BLOCK_D], dtype)
            K_shared = T.alloc_shared([BLOCK_S, BLOCK_D], dtype)
            score_shared = T.alloc_shared([GEMM_M, BLOCK_S], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, BLOCK_S], accum_dtype)

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
                if tq < seq_len:
                    Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                else:
                    Q_shared[flat_m, d] = 0

            loop_limit = num_s_blocks
            if is_causal:
                limit_ts = tilelang.cdiv(base_l + BLOCK_L, block_size)
                loop_limit = T.min(loop_limit, tilelang.cdiv(limit_ts, BLOCK_S) + 1)

            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S

                if per_qhead_lmks:
                    # Per-g GEMM: write into score_shared[l*G+g, s_idx]
                    for g in T.serial(groups):
                        for s_idx, d in T.Parallel(BLOCK_S, BLOCK_D):
                            ts = base_s + s_idx
                            if ts < s_len:
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
                    for s_idx, d in T.Parallel(BLOCK_S, BLOCK_D):
                        ts = base_s + s_idx
                        if ts < s_len:
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
                        ts = base_s + j
                        if ts >= (tq - window_size + 1) // block_size:
                            score_shared[i, j] = -T.infinity(accum_dtype)

                T.sync_threads()

                tx = T.get_thread_binding()
                my_l_idx = tx
                my_tq = base_l + my_l_idx
                cur_max_val = T.alloc_var(accum_dtype)
                val = T.alloc_var(accum_dtype)

                if (my_tq < seq_len) and (tx < BLOCK_L):
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        if ts < s_len:
                            cur_max_val = -T.infinity(accum_dtype)
                            # max over groups using selection logits = scaled qk + bias
                            for g in T.serial(groups):
                                val = score_shared[my_l_idx * groups + g, s_idx] * sm_scale
                                if use_bias:
                                    if val == -T.infinity(accum_dtype):
                                        # Keep -inf for causal-masked positions
                                        val = -T.infinity(accum_dtype)
                                    else:
                                        val += bias[i_b, ts, i_h, g]
                                if val > cur_max_val:
                                    cur_max_val = val

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
            if (my_tq < seq_len) and (tx < BLOCK_L):
                for k in T.serial(topk):
                    OutIndices[i_b, my_tq, i_h, k] = topk_indices_local[k]


    return fwd_kernel_max_pooling



@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def recompute_topk_max_pooling_scores_kernel(
    batch, seq_len, s_len, h_kv, groups, head_dim, topk,
    BLOCK_L=None, BLOCK_TK=None, threads=None,
    per_qhead_lmks=False,
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

    q_shape = [batch, seq_len, h_kv, groups, head_dim]
    if per_qhead_lmks:
        k_shape = [batch, s_len, h_kv * groups, head_dim]
    else:
        k_shape = [batch, s_len, h_kv, head_dim]
    indices_shape = [batch, seq_len, h_kv, topk]
    out_scores_shape = [batch, seq_len, h_kv, groups, topk]

    if BLOCK_L is None:
        # Same alignment constraint as the selection kernel: per-q-head path
        # needs BLOCK_L >= 16; shared-K path needs BLOCK_L*groups >= 16.
        BLOCK_L = 16 if per_qhead_lmks else (16 + groups - 1) // groups
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
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
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
                        # Load K for this (tk_block, g)
                        for l_idx, tk_idx, d in T.Parallel(BLOCK_L, BLOCK_TK, BLOCK_D):
                            tq = base_l + l_idx
                            off = l_idx * BLOCK_TK + tk_idx
                            if (tq < seq_len) and (tk_idx < tk_size):
                                k_id = tk_base + tk_idx
                                idx = Indices[i_b, tq, i_h, k_id]
                                if (idx >= 0) and (idx < s_len):
                                    K_shared[off, d] = K[i_b, idx, i_h * groups + g, d]
                                else:
                                    K_shared[off, d] = T.Cast(dtype, 0.0)
                            else:
                                if off < BLOCK_L * BLOCK_TK:
                                    K_shared[off, d] = T.Cast(dtype, 0.0)
                        # Load Q for this g
                        for l_idx, d in T.Parallel(BLOCK_L, BLOCK_D):
                            tq = base_l + l_idx
                            if tq < seq_len:
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
                            if (tq < seq_len) and (tk_idx < tk_size):
                                k_id = tk_base + tk_idx
                                idx = Indices[i_b, tq, i_h, k_id]
                                if idx < 0:
                                    OutScores[i_b, tq, i_h, g, k_id] = -T.infinity(accum_dtype)
                                else:
                                    col = l_idx * BLOCK_TK + tk_idx
                                    OutScores[i_b, tq, i_h, g, k_id] = acc_s_g[l_idx, col] * sm_scale
                        T.sync_threads()
            else:
                Q_shared = T.alloc_shared([GEMM_M, BLOCK_D], dtype)
                score_shared = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)
                acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)

                for l_idx, g, d in T.Parallel(BLOCK_L, groups, BLOCK_D):
                    tq = base_l + l_idx
                    flat_m = l_idx * groups + g
                    if tq < seq_len:
                        Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                    else:
                        Q_shared[flat_m, d] = T.Cast(dtype, 0.0)

                for tk_block in T.serial(tk_blocks):
                    tk_base = tk_block * BLOCK_TK
                    tk_size = T.min(BLOCK_TK, topk - tk_base)

                    for l_idx, tk_idx, d in T.Parallel(BLOCK_L, BLOCK_TK, BLOCK_D):
                        tq = base_l + l_idx
                        off = l_idx * BLOCK_TK + tk_idx
                        if (tq < seq_len) and (tk_idx < tk_size):
                            k_id = tk_base + tk_idx
                            idx = Indices[i_b, tq, i_h, k_id]
                            if (idx >= 0) and (idx < s_len):
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
                        if (tq < seq_len) and (tk_idx < tk_size):
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
    s_len: int,
    h_kv: int,
    topk: int,
    BLOCK_L: int = 16,
    num_threads: int = 64,
):
    """
    对 IndicesIn: [B, L, h_kv, topk] 按最后一维进行 bitonic 升序排序（按 chunk id）。
    - 有效索引: 0 <= idx < s_len
    - 无效索引: idx < 0（例如 -1），排序时会被视为 key = s_len，放在末尾。
    """
    
    BF16 = "bfloat16"
    FP32 = "float32"
    INT32 = "int32"
    assert topk == tilelang.math.next_power_of_2(topk)
    num_iters = int(round(math.log2(topk)))

    indices_shape = [batch, seq_len, h_kv, topk]

    @T.prim_func
    def sort_kernel(
        IndicesIn: T.Tensor(indices_shape, INT32),
        IndicesOut: T.Tensor(indices_shape, INT32),
    ):

        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), h_kv, batch, threads=num_threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L

            idx_shared = T.alloc_shared([BLOCK_L, topk], dtype=INT32)

            for l_idx, k in T.Parallel(BLOCK_L, topk):
                lq = base_l + l_idx
                if lq < seq_len:
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
                        if lq < seq_len:
                            i_idx = i
                            ixj = i_idx ^ j_step
                            if (ixj > i_idx) and (ixj < topk):
                                val_i = idx_shared[l_idx, i_idx]
                                val_j = idx_shared[l_idx, ixj]

                                if val_i >= 0:
                                    key_i = val_i
                                else:
                                    key_i = s_len
                                if val_j >= 0:
                                    key_j = val_j
                                else:
                                    key_j = s_len

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
                if lq < seq_len:
                    IndicesOut[i_b, lq, i_h, k] = idx_shared[l_idx, k]

    return sort_kernel

class TopKMaxPoolingFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, lmks, topk: int,
                select_kernel,
                sort_kernel,
                recompute_kernel,
                bias_arg,
                per_qhead_lmks: bool,
                ):
        # q:    [B, L, h_kv, G, D]
        # lmks: [B, S, h_kv, D]  (per-KV-head)  or  [B, S, h_kv*G, D] (per-q-head)
        # bias_arg: [B, S, h_kv, G] float32 when use_bias else [1,1,1,1] dummy
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
        bias_in = bias_arg.contiguous()

        indices_raw = select_kernel(q_in, k_in, bias_in)  # int32
        indices_sorted = sort_kernel(indices_raw)
        indices_for_kernel = indices_sorted

        best_scores_buf = recompute_kernel(q_in, k_in, indices_for_kernel)  # float32

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
        #                    recompute_kernel, bias_arg, per_qhead_lmks)
        return grad_q, grad_lmks, None, None, None, None, None, None


class TopKMaxPooling_Fused(torch.nn.Module):
    def __init__(self, topk, block_size, window_size, is_causal,
                 use_bias: bool = False, per_qhead_lmks: bool = False):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self.use_bias = use_bias
        self.per_qhead_lmks = per_qhead_lmks
        self._cached_select_kernel = None
        self._cached_sort_kernel = None
        self._cached_recompute_kernel = None
        self._cached_shape = None

    def forward(self, q, lmks, bias=None):
        # q:    [B, L, h_kv, G, D]
        # lmks: [B, S, h_kv, D]   (per-KV-head)  or  [B, S, h_kv*G, D] (per-q-head)
        # bias: optional [B, S, h_q] or [B, S, h_kv, G]
        B, L, h_kv, G, D = q.shape
        _, S, _, _ = lmks.shape
        topk = self.topk
        block_size = self.block_size
        window_size = self.window_size
        is_causal = self.is_causal
        use_bias = self.use_bias
        per_qhead_lmks = self.per_qhead_lmks

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

        shape_key = (B, L, S, h_kv, G, D, topk, block_size, window_size,
                     is_causal, use_bias, per_qhead_lmks)
        if self._cached_shape != shape_key:

            self._cached_select_kernel = fused_topk_max_pooling_kernel(
                B, L, S, h_kv, G, D, topk, block_size, window_size, is_causal,
                use_bias=use_bias, per_qhead_lmks=per_qhead_lmks,
            )

            self._cached_sort_kernel = sort_topk_indices_kernel(
                B, L, S, h_kv, topk
            )

            self._cached_recompute_kernel = recompute_topk_max_pooling_scores_kernel(
                B, L, S, h_kv, G, D, topk,
                per_qhead_lmks=per_qhead_lmks,
            )
            self._cached_shape = shape_key

        select_kernel = self._cached_select_kernel
        sort_kernel = self._cached_sort_kernel
        recompute_kernel = self._cached_recompute_kernel

        indices, scores = TopKMaxPoolingFusedFn.apply(
            q, lmks, topk, select_kernel, sort_kernel, recompute_kernel,
            bias_arg, per_qhead_lmks,
        )
        # Reshape scores: [B, L, h_kv, G, topk] -> [B, L, h_q, topk]
        scores = scores.view(B, L, h_kv * G, -1)
        return indices, scores

_MODULE_CACHE = {}
def online_topk_head(q: torch.Tensor, lmks: torch.Tensor, topk: int,
                     block_size: int, window_size: int, is_causal: bool = False,
                     bias: Optional[torch.Tensor] = None,
                     G: Optional[int] = None):
    """
    Functional API for TopKMaxPooling_Fused (max-pooling top-k, no SWA-LSE).

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
    cache_key = (topk, block_size, window_size, is_causal, use_bias, per_qhead_lmks)
    if cache_key not in _MODULE_CACHE:
        _MODULE_CACHE[cache_key] = TopKMaxPooling_Fused(
            topk, block_size, window_size, is_causal,
            use_bias=use_bias, per_qhead_lmks=per_qhead_lmks,
        )

    return _MODULE_CACHE[cache_key](q, lmks, bias=bias)


# ----------------------------------------------------------------------
# Tests (ported & adapted from topk_head_softmax.py to cover bias and
# per_qhead_lmks for the max-pooling top-k path).
# ----------------------------------------------------------------------
def test_train_inference_correctness(test_name, B, q_len, h_kv, G, D, topk,
                                     block_size, window_size,
                                     use_bias, per_qhead_lmks=False, is_causal=True):
    """
    Forward + backward correctness test for the max-pooling top-k kernel.
    Adapted from topk_head_softmax.py:test_train_inference_correctness.
    The max-pooling path has no SWA LSE / inference q_offset, so we keep
    only the training-style coverage of bias and per_qhead_lmks combinations.
    """
    device = "cuda"
    dtype = torch.bfloat16
    h_q = h_kv * G

    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"Config: B={B}, q_len={q_len}, h_kv={h_kv}, G={G}, D={D}")
    print(f"        topk={topk}, block_size={block_size}, window_size={window_size}")
    print(f"        is_causal={is_causal}, use_bias={use_bias}, per_qhead_lmks={per_qhead_lmks}")
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

    G_arg = G if per_qhead_lmks else None

    # ============ Forward ============
    print("\n--- Forward Pass ---")

    indices_ref, scores_ref = ref_topk_max_pooling(
        q_ref, lmks_ref, topk, block_size, window_size, is_causal, bias=bias,
    )

    indices_fused, scores_fused = online_topk_head(
        q_fused, lmks_fused, topk, block_size, window_size, is_causal,
        bias=bias, G=G_arg,
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


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Regression Suite for topk_head (max-pooling, with bias / per-q-head lmks)")
    print("=" * 70)

    # Legacy quick check
    test_fused_topk_max_pooling_correctness()

    # 1) Baseline: per-KV-head lmks, no bias
    test_train_inference_correctness(
        "train_basic", 2, 1024, 2, 8, 128, 16, 64, 64, False, False
    )

    # 2) Per-KV-head lmks + bias  (the missing-bias path we are now fixing)
    test_train_inference_correctness(
        "train_basic_bias", 2, 1024, 2, 8, 128, 16, 64, 64, True, False
    )

    # 3) Per-q-head lmks (no bias)
    test_train_inference_correctness(
        "train_perqhead", 2, 1024, 2, 8, 128, 16, 64, 64, False, True
    )

    # 4) Per-q-head lmks + bias (most general path)
    test_train_inference_correctness(
        "train_perqhead_bias", 2, 1024, 2, 8, 128, 16, 64, 64, True, True
    )

    # 5) Non-divisible q_len (padding inside kernel)
    test_train_inference_correctness(
        "train_bias_nondiv_999", 2, 999, 2, 8, 128, 16, 64, 64, True, False
    )

    # 6) G=1 corner case + per_qhead + bias
    test_train_inference_correctness(
        "train_perqhead_G1_bias", 2, 1024, 4, 1, 128, 16, 64, 64, True, True
    )

    print("\n" + "=" * 70)
    print("All regression tests finished.")
    print("=" * 70)
