"""
Tilelang flex attention with PoPE built into the kernel.

- PoPE applied inside the kernel: caller passes raw (already softplus'd) q/k
  plus `freqs` and `bias`; kernel materializes Qc/Qs/Kc/Ks and computes
  score = Qc.Kc + Qs.Ks (rotated dims) + Q.K (non-rotated dims).
- Softplus is NOT done inside the kernel (fused with rmsnorm upstream).
- Bias sign convention follows hsa_fwd_bwd_head_pope / topk_head_softmax_pope:
    theta_q = Freqs[pos_q] - Bias[h_kv, g]   (MINUS sign)
    theta_k = Freqs[pos_k]                    (no bias on K side)

Two boolean flags retained from flex_attn_tilelang.py:
- mask_lmk:        if True, mask kv columns where (kv_idx + 1) % chunk_size == 0
- expand_to_chunk: if True, sliding window left edge expands to start of containing chunk

Public API layout:
    q: (B, Lq, H_q,  D_qk), k: (B, Lk, H_kv, D_qk), v: (B, Lk, H_kv, D_v)
    freqs: (L_total, R)        fp32
    bias:  (H_q, R) or (H_kv, G, R)  fp32   Q-side additive bias (MINUS sign)
    -> o: (B, Lq, H_q, D_v),  lse: (B, Lq, H_q) natural log

Backward is atomic-add only, training only (use_cache must be False).
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


_TORCH_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}

def _torch_dtype_to_str(td: torch.dtype) -> str:
    if td not in _TORCH_DTYPE_TO_STR:
        raise ValueError(f"Unsupported torch dtype for tilelang kernel: {td}")
    return _TORCH_DTYPE_TO_STR[td]


# ---------------------------------------------------------------------------
# Forward kernel (fused train + inference via use_cache flag)
# ---------------------------------------------------------------------------
@tilelang.jit(
    out_idx=[5, 6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def flex_attn_pope_fwd(
    batch, heads, q_len, kv_len, dim_qk, dim_v, rotate_dim, freqs_len,
    window_size, chunk_size, mask_lmk, expand_to_chunk,
    block_M, block_N, groups=1, use_cache=False,
    dtype="bfloat16", accum_dtype="float",
    sm_scale=None,
):
    # Default sm_scale = 1/sqrt(dim_qk) (the original head_dim, not D+R) so
    # this op stays consistent with HSA_block_M_head_pope* and
    # online_softmax_topk_head_pope* whose defaults also use 1/sqrt(D).
    if sm_scale is None:
        sm_scale = (1.0 / dim_qk) ** 0.5
    scale = sm_scale * 1.44269504  # log2(e)
    head_kv = heads // groups

    if use_cache:
        q_len_var = T.dynamic("q_len")
        kv_len_var = T.dynamic("kv_len")
        freqs_len_var = T.dynamic("freqs_len")
    else:
        q_len_var = q_len
        kv_len_var = kv_len
        freqs_len_var = freqs_len

    q_shape = [batch, q_len_var, heads, dim_qk]
    k_shape = [batch, kv_len_var, head_kv, dim_qk]
    v_shape = [batch, kv_len_var, head_kv, dim_v]
    o_shape = [batch, q_len_var, heads, dim_v]
    lse_shape = [batch, q_len_var, heads]
    freqs_shape = [freqs_len_var, rotate_dim]
    bias_shape = [head_kv, groups, rotate_dim]

    @T.prim_func
    def fwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        Freqs: T.Tensor(freqs_shape, accum_dtype),  # type: ignore
        Bias: T.Tensor(bias_shape, accum_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(q_len_var, block_M), heads, batch, threads=256) as (bx, by, bz):
            Qc_shared = T.alloc_shared([block_M, dim_qk], dtype)
            Qs_shared = T.alloc_shared([block_M, rotate_dim], dtype)
            Kc_shared = T.alloc_shared([block_N, dim_qk], dtype)
            Ks_shared = T.alloc_shared([block_N, rotate_dim], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            O_shared = T.alloc_shared([block_M, dim_v], dtype)
            lse_out_shared = T.alloc_shared([block_M], accum_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            i_h_kv = by // groups
            i_g = by % groups
            q_offset = kv_len_var - q_len_var

            for i, d in T.Parallel(block_M, dim_qk):
                tq = bx * block_M + i
                if tq < q_len_var:
                    if d < rotate_dim:
                        pos_q = q_offset + tq
                        theta_q = Freqs[pos_q, d] - Bias[i_h_kv, i_g, d]
                        qx = T.Cast(accum_dtype, Q[bz, tq, by, d])
                        Qc_shared[i, d] = T.Cast(dtype, qx * T.cos(theta_q))
                    else:
                        Qc_shared[i, d] = Q[bz, tq, by, d]
                else:
                    Qc_shared[i, d] = T.Cast(dtype, 0)

            for i, d in T.Parallel(block_M, rotate_dim):
                tq = bx * block_M + i
                if tq < q_len_var:
                    pos_q = q_offset + tq
                    theta_q = Freqs[pos_q, d] - Bias[i_h_kv, i_g, d]
                    qx = T.Cast(accum_dtype, Q[bz, tq, by, d])
                    Qs_shared[i, d] = T.Cast(dtype, qx * T.sin(theta_q))
                else:
                    Qs_shared[i, d] = T.Cast(dtype, 0)

            q_block_start = bx * block_M + q_offset
            q_block_end = (bx + 1) * block_M + q_offset

            if expand_to_chunk:
                left_kv_raw = T.floordiv(q_block_start - window_size + 1, chunk_size) * chunk_size
            else:
                left_kv_raw = q_block_start - window_size + 1
            left_kv = T.max(left_kv_raw, 0)
            right_kv = T.min(q_block_end, kv_len_var)
            loop_st = T.floordiv(left_kv, block_N)
            loop_ed = T.ceildiv(right_kv, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                for i, d in T.Parallel(block_N, dim_qk):
                    kv_real_local = k * block_N + i
                    if kv_real_local < kv_len_var:
                        if d < rotate_dim:
                            theta_k = Freqs[kv_real_local, d]
                            kx = T.Cast(accum_dtype, K[bz, kv_real_local, by // groups, d])
                            Kc_shared[i, d] = T.Cast(dtype, kx * T.cos(theta_k))
                        else:
                            Kc_shared[i, d] = K[bz, kv_real_local, by // groups, d]
                    else:
                        Kc_shared[i, d] = T.Cast(dtype, 0)

                for i, d in T.Parallel(block_N, rotate_dim):
                    kv_real_local = k * block_N + i
                    if kv_real_local < kv_len_var:
                        theta_k = Freqs[kv_real_local, d]
                        kx = T.Cast(accum_dtype, K[bz, kv_real_local, by // groups, d])
                        Ks_shared[i, d] = T.Cast(dtype, kx * T.sin(theta_k))
                    else:
                        Ks_shared[i, d] = T.Cast(dtype, 0)

                for i, d in T.Parallel(block_N, dim_v):
                    kv_real_local = k * block_N + i
                    if kv_real_local < kv_len_var:
                        V_shared[i, d] = V[bz, kv_real_local, by // groups, d]
                    else:
                        V_shared[i, d] = T.Cast(dtype, 0)

                T.clear(acc_s)
                T.gemm(Qc_shared, Kc_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Qs_shared, Ks_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                if expand_to_chunk and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var)
                            & (T.floormod(kv_real + 1, chunk_size) != 0),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )
                elif expand_to_chunk and (not mask_lmk):
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )
                elif (not expand_to_chunk) and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = q_real - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var)
                            & (T.floormod(kv_real + 1, chunk_size) != 0),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = q_real - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                        1.0,
                    )
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(acc_s[i, j] * scale - scores_max[i] * scale),
                        0.0,
                    )
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            for i, j in T.Parallel(block_M, dim_v):
                acc_o[i, j] = T.if_then_else(logsum[i] > 0, acc_o[i, j] / logsum[i], 0.0)
            for i in T.Parallel(block_M):
                logsum[i] = T.if_then_else(
                    logsum[i] > 0,
                    T.log2(logsum[i]) + scores_max[i] * scale,
                    -T.infinity(accum_dtype),
                )
            T.copy(acc_o, O_shared)
            T.copy(logsum, lse_out_shared)
            for i, j in T.Parallel(block_M, dim_v):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Output[bz, tq, by, j] = O_shared[i, j]
            for i in T.Parallel(block_M):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Lse[bz, tq, by] = lse_out_shared[i]

    return fwd


# ---------------------------------------------------------------------------
# K-side prerotation + fused forward variant
# ---------------------------------------------------------------------------
@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def pope_pre_rotate_k_kernel(
    batch, head_kv, kv_len, dim_qk, rotate_dim, freqs_len,
    block_N=128, dtype="bfloat16", accum_dtype="float",
):
    """Pre-rotate K once into Kc/Ks.

    Kc keeps full D: [k*cos(theta), k_rest]. Ks keeps only R: [k*sin(theta)].
    This avoids repeated K-side cos/sin inside every Q-block of the attention
    kernel while keeping Q-side rotation on-chip.
    """
    k_shape = [batch, kv_len, head_kv, dim_qk]
    ks_shape = [batch, kv_len, head_kv, rotate_dim]
    freqs_shape = [freqs_len, rotate_dim]

    @T.prim_func
    def kernel(
        K: T.Tensor(k_shape, dtype),  # type: ignore
        Freqs: T.Tensor(freqs_shape, accum_dtype),  # type: ignore
        Kc: T.Tensor(k_shape, dtype),  # type: ignore
        Ks: T.Tensor(ks_shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(kv_len, block_N), head_kv, batch, threads=256) as (bx, by, bz):
            for i, d in T.Parallel(block_N, dim_qk):
                kv_real = bx * block_N + i
                if kv_real < kv_len:
                    if d < rotate_dim:
                        theta_k = Freqs[kv_real, d]
                        kx = T.Cast(accum_dtype, K[bz, kv_real, by, d])
                        Kc[bz, kv_real, by, d] = T.Cast(dtype, kx * T.cos(theta_k))
                    else:
                        Kc[bz, kv_real, by, d] = K[bz, kv_real, by, d]

            for i, d in T.Parallel(block_N, rotate_dim):
                kv_real = bx * block_N + i
                if kv_real < kv_len:
                    theta_k = Freqs[kv_real, d]
                    kx = T.Cast(accum_dtype, K[bz, kv_real, by, d])
                    Ks[bz, kv_real, by, d] = T.Cast(dtype, kx * T.sin(theta_k))

    return kernel


@tilelang.jit(
    out_idx=[6, 7],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def flex_attn_pope_kprerotated_fwd(
    batch, heads, q_len, kv_len, dim_qk, dim_v, rotate_dim, freqs_len,
    window_size, chunk_size, mask_lmk, expand_to_chunk,
    block_M, block_N, groups=1, use_cache=False,
    dtype="bfloat16", accum_dtype="float",
    sm_scale=None,
):
    """Fused PoPE forward with K-side rotation precomputed as Kc/Ks."""
    if sm_scale is None:
        sm_scale = (1.0 / dim_qk) ** 0.5
    scale = sm_scale * 1.44269504
    head_kv = heads // groups

    if use_cache:
        q_len_var = T.dynamic("q_len")
        kv_len_var = T.dynamic("kv_len")
        freqs_len_var = T.dynamic("freqs_len")
    else:
        q_len_var = q_len
        kv_len_var = kv_len
        freqs_len_var = freqs_len

    q_shape = [batch, q_len_var, heads, dim_qk]
    kc_shape = [batch, kv_len_var, head_kv, dim_qk]
    ks_shape = [batch, kv_len_var, head_kv, rotate_dim]
    v_shape = [batch, kv_len_var, head_kv, dim_v]
    o_shape = [batch, q_len_var, heads, dim_v]
    lse_shape = [batch, q_len_var, heads]
    freqs_shape = [freqs_len_var, rotate_dim]
    bias_shape = [head_kv, groups, rotate_dim]

    @T.prim_func
    def fwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        Kc: T.Tensor(kc_shape, dtype),  # type: ignore
        Ks: T.Tensor(ks_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        Freqs: T.Tensor(freqs_shape, accum_dtype),  # type: ignore
        Bias: T.Tensor(bias_shape, accum_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(q_len_var, block_M), heads, batch, threads=256) as (bx, by, bz):
            Qc_shared = T.alloc_shared([block_M, dim_qk], dtype)
            Qs_shared = T.alloc_shared([block_M, rotate_dim], dtype)
            Kc_shared = T.alloc_shared([block_N, dim_qk], dtype)
            Ks_shared = T.alloc_shared([block_N, rotate_dim], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            O_shared = T.alloc_shared([block_M, dim_v], dtype)
            lse_out_shared = T.alloc_shared([block_M], accum_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            i_h_kv = by // groups
            i_g = by % groups
            q_offset = kv_len_var - q_len_var

            for i, d in T.Parallel(block_M, dim_qk):
                tq = bx * block_M + i
                if tq < q_len_var:
                    if d < rotate_dim:
                        pos_q = q_offset + tq
                        theta_q = Freqs[pos_q, d] - Bias[i_h_kv, i_g, d]
                        qx = T.Cast(accum_dtype, Q[bz, tq, by, d])
                        Qc_shared[i, d] = T.Cast(dtype, qx * T.cos(theta_q))
                    else:
                        Qc_shared[i, d] = Q[bz, tq, by, d]
                else:
                    Qc_shared[i, d] = T.Cast(dtype, 0)

            for i, d in T.Parallel(block_M, rotate_dim):
                tq = bx * block_M + i
                if tq < q_len_var:
                    pos_q = q_offset + tq
                    theta_q = Freqs[pos_q, d] - Bias[i_h_kv, i_g, d]
                    qx = T.Cast(accum_dtype, Q[bz, tq, by, d])
                    Qs_shared[i, d] = T.Cast(dtype, qx * T.sin(theta_q))
                else:
                    Qs_shared[i, d] = T.Cast(dtype, 0)

            q_block_start = bx * block_M + q_offset
            q_block_end = (bx + 1) * block_M + q_offset

            if expand_to_chunk:
                left_kv_raw = T.floordiv(q_block_start - window_size + 1, chunk_size) * chunk_size
            else:
                left_kv_raw = q_block_start - window_size + 1
            left_kv = T.max(left_kv_raw, 0)
            right_kv = T.min(q_block_end, kv_len_var)
            loop_st = T.floordiv(left_kv, block_N)
            loop_ed = T.ceildiv(right_kv, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                for i, d in T.Parallel(block_N, dim_qk):
                    kv_real = k * block_N + i
                    if kv_real < kv_len_var:
                        Kc_shared[i, d] = Kc[bz, kv_real, by // groups, d]
                    else:
                        Kc_shared[i, d] = T.Cast(dtype, 0)

                for i, d in T.Parallel(block_N, rotate_dim):
                    kv_real = k * block_N + i
                    if kv_real < kv_len_var:
                        Ks_shared[i, d] = Ks[bz, kv_real, by // groups, d]
                    else:
                        Ks_shared[i, d] = T.Cast(dtype, 0)

                for i, d in T.Parallel(block_N, dim_v):
                    kv_real = k * block_N + i
                    if kv_real < kv_len_var:
                        V_shared[i, d] = V[bz, kv_real, by // groups, d]
                    else:
                        V_shared[i, d] = T.Cast(dtype, 0)

                T.clear(acc_s)
                T.gemm(Qc_shared, Kc_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Qs_shared, Ks_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                if expand_to_chunk and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var)
                            & (T.floormod(kv_real + 1, chunk_size) != 0),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )
                elif expand_to_chunk and (not mask_lmk):
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )
                elif (not expand_to_chunk) and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = q_real - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var)
                            & (T.floormod(kv_real + 1, chunk_size) != 0),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = q_real - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                        1.0,
                    )
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(acc_s[i, j] * scale - scores_max[i] * scale),
                        0.0,
                    )
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            for i, j in T.Parallel(block_M, dim_v):
                acc_o[i, j] = T.if_then_else(logsum[i] > 0, acc_o[i, j] / logsum[i], 0.0)
            for i in T.Parallel(block_M):
                logsum[i] = T.if_then_else(
                    logsum[i] > 0,
                    T.log2(logsum[i]) + scores_max[i] * scale,
                    -T.infinity(accum_dtype),
                )
            T.copy(acc_o, O_shared)
            T.copy(logsum, lse_out_shared)
            for i, j in T.Parallel(block_M, dim_v):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Output[bz, tq, by, j] = O_shared[i, j]
            for i in T.Parallel(block_M):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Lse[bz, tq, by] = lse_out_shared[i]

    return fwd


@tilelang.jit(
    out_idx=[3, 4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def pope_pre_rotate_q_kernel(
    batch, heads, q_len, kv_len, dim_qk, rotate_dim, freqs_len,
    block_M=128, groups=1, dtype="bfloat16", accum_dtype="float",
):
    """Pre-rotate Q once into Qc/Qs for the current q_len/kv_len window."""
    head_kv = heads // groups
    q_shape = [batch, q_len, heads, dim_qk]
    qs_shape = [batch, q_len, heads, rotate_dim]
    freqs_shape = [freqs_len, rotate_dim]
    bias_shape = [head_kv, groups, rotate_dim]

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        Freqs: T.Tensor(freqs_shape, accum_dtype),  # type: ignore
        Bias: T.Tensor(bias_shape, accum_dtype),  # type: ignore
        Qc: T.Tensor(q_shape, dtype),  # type: ignore
        Qs: T.Tensor(qs_shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(q_len, block_M), heads, batch, threads=256) as (bx, by, bz):
            i_h_kv = by // groups
            i_g = by % groups
            q_offset = kv_len - q_len
            for i, d in T.Parallel(block_M, dim_qk):
                tq = bx * block_M + i
                if tq < q_len:
                    if d < rotate_dim:
                        pos_q = q_offset + tq
                        theta_q = Freqs[pos_q, d] - Bias[i_h_kv, i_g, d]
                        qx = T.Cast(accum_dtype, Q[bz, tq, by, d])
                        Qc[bz, tq, by, d] = T.Cast(dtype, qx * T.cos(theta_q))
                    else:
                        Qc[bz, tq, by, d] = Q[bz, tq, by, d]

            for i, d in T.Parallel(block_M, rotate_dim):
                tq = bx * block_M + i
                if tq < q_len:
                    pos_q = q_offset + tq
                    theta_q = Freqs[pos_q, d] - Bias[i_h_kv, i_g, d]
                    qx = T.Cast(accum_dtype, Q[bz, tq, by, d])
                    Qs[bz, tq, by, d] = T.Cast(dtype, qx * T.sin(theta_q))

    return kernel


@tilelang.jit(
    out_idx=[5, 6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def flex_attn_pope_qkprerotated_fwd(
    batch, heads, q_len, kv_len, dim_qk, dim_v, rotate_dim,
    window_size, chunk_size, mask_lmk, expand_to_chunk,
    block_M, block_N, groups=1, use_cache=False,
    dtype="bfloat16", accum_dtype="float",
    sm_scale=None,
):
    """Fused PoPE forward with Q/K rotations precomputed as Qc/Qs/Kc/Ks."""
    if sm_scale is None:
        sm_scale = (1.0 / dim_qk) ** 0.5
    scale = sm_scale * 1.44269504
    head_kv = heads // groups

    if use_cache:
        q_len_var = T.dynamic("q_len")
        kv_len_var = T.dynamic("kv_len")
    else:
        q_len_var = q_len
        kv_len_var = kv_len

    qc_shape = [batch, q_len_var, heads, dim_qk]
    qs_shape = [batch, q_len_var, heads, rotate_dim]
    kc_shape = [batch, kv_len_var, head_kv, dim_qk]
    ks_shape = [batch, kv_len_var, head_kv, rotate_dim]
    v_shape = [batch, kv_len_var, head_kv, dim_v]
    o_shape = [batch, q_len_var, heads, dim_v]
    lse_shape = [batch, q_len_var, heads]

    @T.prim_func
    def fwd(
        Qc: T.Tensor(qc_shape, dtype),  # type: ignore
        Qs: T.Tensor(qs_shape, dtype),  # type: ignore
        Kc: T.Tensor(kc_shape, dtype),  # type: ignore
        Ks: T.Tensor(ks_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(q_len_var, block_M), heads, batch, threads=256) as (bx, by, bz):
            Qc_shared = T.alloc_shared([block_M, dim_qk], dtype)
            Qs_shared = T.alloc_shared([block_M, rotate_dim], dtype)
            Kc_shared = T.alloc_shared([block_N, dim_qk], dtype)
            Ks_shared = T.alloc_shared([block_N, rotate_dim], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            O_shared = T.alloc_shared([block_M, dim_v], dtype)
            lse_out_shared = T.alloc_shared([block_M], accum_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            q_offset = kv_len_var - q_len_var
            for i, d in T.Parallel(block_M, dim_qk):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Qc_shared[i, d] = Qc[bz, tq, by, d]
                else:
                    Qc_shared[i, d] = T.Cast(dtype, 0)
            for i, d in T.Parallel(block_M, rotate_dim):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Qs_shared[i, d] = Qs[bz, tq, by, d]
                else:
                    Qs_shared[i, d] = T.Cast(dtype, 0)

            q_block_start = bx * block_M + q_offset
            q_block_end = (bx + 1) * block_M + q_offset
            if expand_to_chunk:
                left_kv_raw = T.floordiv(q_block_start - window_size + 1, chunk_size) * chunk_size
            else:
                left_kv_raw = q_block_start - window_size + 1
            left_kv = T.max(left_kv_raw, 0)
            right_kv = T.min(q_block_end, kv_len_var)
            loop_st = T.floordiv(left_kv, block_N)
            loop_ed = T.ceildiv(right_kv, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                for i, d in T.Parallel(block_N, dim_qk):
                    kv_real = k * block_N + i
                    if kv_real < kv_len_var:
                        Kc_shared[i, d] = Kc[bz, kv_real, by // groups, d]
                    else:
                        Kc_shared[i, d] = T.Cast(dtype, 0)
                for i, d in T.Parallel(block_N, rotate_dim):
                    kv_real = k * block_N + i
                    if kv_real < kv_len_var:
                        Ks_shared[i, d] = Ks[bz, kv_real, by // groups, d]
                    else:
                        Ks_shared[i, d] = T.Cast(dtype, 0)
                for i, d in T.Parallel(block_N, dim_v):
                    kv_real = k * block_N + i
                    if kv_real < kv_len_var:
                        V_shared[i, d] = V[bz, kv_real, by // groups, d]
                    else:
                        V_shared[i, d] = T.Cast(dtype, 0)

                T.clear(acc_s)
                T.gemm(Qc_shared, Kc_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Qs_shared, Ks_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                if expand_to_chunk and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var)
                            & (T.floormod(kv_real + 1, chunk_size) != 0),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )
                elif expand_to_chunk and (not mask_lmk):
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )
                elif (not expand_to_chunk) and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = q_real - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var)
                            & (T.floormod(kv_real + 1, chunk_size) != 0),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = bx * block_M + i + q_offset
                        kv_real = k * block_N + j
                        chunk_start = q_real - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_real >= chunk_start) & (kv_real <= q_real)
                            & (kv_real < kv_len_var) & ((bx * block_M + i) < q_len_var),
                            acc_s[i, j], -T.infinity(accum_dtype),
                        )

                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                        1.0,
                    )
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(acc_s[i, j] * scale - scores_max[i] * scale),
                        0.0,
                    )
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            for i, j in T.Parallel(block_M, dim_v):
                acc_o[i, j] = T.if_then_else(logsum[i] > 0, acc_o[i, j] / logsum[i], 0.0)
            for i in T.Parallel(block_M):
                logsum[i] = T.if_then_else(
                    logsum[i] > 0,
                    T.log2(logsum[i]) + scores_max[i] * scale,
                    -T.infinity(accum_dtype),
                )
            T.copy(acc_o, O_shared)
            T.copy(logsum, lse_out_shared)
            for i, j in T.Parallel(block_M, dim_v):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Output[bz, tq, by, j] = O_shared[i, j]
            for i in T.Parallel(block_M):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Lse[bz, tq, by] = lse_out_shared[i]

    return fwd


# ---------------------------------------------------------------------------
# Backward preprocess
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flex_attn_pope_bwd_preprocess(batch, heads, seq_len, dim_v, dtype="bfloat16", accum_dtype="float"):
    shape = [batch, seq_len, heads, dim_v]
    blk = 32

    @T.prim_func
    def prep(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            delta_shared = T.alloc_shared([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim_v, blk)):
                for i, j in T.Parallel(blk, blk):
                    tq = by * blk + i
                    vd = k * blk + j
                    if (tq < seq_len) and (vd < dim_v):
                        o[i, j] = O[bz, tq, bx, vd]
                        do[i, j] = dO[bz, tq, bx, vd]
                    else:
                        o[i, j] = T.Cast(dtype, 0)
                        do[i, j] = T.Cast(dtype, 0)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, delta_shared)
            for i in T.Parallel(blk):
                tq = by * blk + i
                if tq < seq_len:
                    Delta[bz, bx, tq] = delta_shared[i]

    return prep


@tilelang.jit(out_idx=[1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flex_attn_pope_bwd_postprocess(batch, heads, seq_len, dim_qk, dtype="bfloat16", accum_dtype="float"):
    shape = [batch, seq_len, heads, dim_qk]
    blk = 64

    @T.prim_func
    def post(
        dQ: T.Tensor(shape, accum_dtype),  # type: ignore
        dQ_out: T.Tensor(shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            for i, d in T.Parallel(blk, dim_qk):
                tq = bx * blk + i
                if tq < seq_len:
                    dQ_out[bz, tq, by, d] = T.Cast(dtype, dQ[bz, tq, by, d])

    return post


# ---------------------------------------------------------------------------
# Backward kernel with PoPE (atomic-add, training only)
# ---------------------------------------------------------------------------
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def flex_attn_pope_bwd(
    batch, heads, seq_len, dim_qk, dim_v, rotate_dim, freqs_len,
    window_size, chunk_size, mask_lmk, expand_to_chunk,
    block_M, block_N, threads=256, num_stages=2, groups=1,
    dtype="bfloat16", accum_dtype="float",
    sm_scale=None,
):
    # Default sm_scale = 1/sqrt(dim_qk) (see flex_attn_pope_fwd for rationale).
    if sm_scale is None:
        sm_scale = (1.0 / dim_qk) ** 0.5
    scale = sm_scale * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    freqs_shape = [freqs_len, rotate_dim]
    bias_shape = [head_kv, groups, rotate_dim]
    dbias_shape = [head_kv, groups, rotate_dim]

    @T.prim_func
    def bwd(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        dO: T.Tensor([batch, seq_len, heads, dim_v], dtype),
        lse: T.Tensor([batch, seq_len, heads], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        Freqs: T.Tensor(freqs_shape, accum_dtype),
        Bias: T.Tensor(bias_shape, accum_dtype),
        dQ: T.Tensor(q_shape, accum_dtype),
        dK: T.Tensor(k_shape, accum_dtype),
        dV: T.Tensor(v_shape, accum_dtype),
        dBias: T.Tensor(dbias_shape, accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            Kc_shared = T.alloc_shared([block_M, dim_qk], dtype)
            Ks_shared = T.alloc_shared([block_M, rotate_dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            Qc_shared = T.alloc_shared([block_N, dim_qk], dtype)
            Qs_shared = T.alloc_shared([block_N, rotate_dim], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta_sh = T.alloc_shared([block_N], accum_dtype)
            do_sh = T.alloc_shared([block_N, dim_v], dtype)
            dqc_out_shared = T.alloc_shared([block_N, dim_qk], accum_dtype)
            dqs_out_shared = T.alloc_shared([block_N, rotate_dim], accum_dtype)
            dKc_out_shared = T.alloc_shared([block_M, dim_qk], accum_dtype)
            dKs_out_shared = T.alloc_shared([block_M, rotate_dim], accum_dtype)
            dKc_acc = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dKs_acc = T.alloc_fragment([block_M, rotate_dim], accum_dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            dqc_frag = T.alloc_fragment([block_N, dim_qk], accum_dtype)
            dqs_frag = T.alloc_fragment([block_N, rotate_dim], accum_dtype)

            i_h_kv = bx // groups
            i_g = bx % groups

            for i, d in T.Parallel(block_M, dim_qk):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    if d < rotate_dim:
                        theta_k = Freqs[kv_real, d]
                        kx = T.Cast(accum_dtype, K[bz, kv_real, bx // groups, d])
                        Kc_shared[i, d] = T.Cast(dtype, kx * T.cos(theta_k))
                    else:
                        Kc_shared[i, d] = K[bz, kv_real, bx // groups, d]
                else:
                    Kc_shared[i, d] = T.Cast(dtype, 0)

            for i, d in T.Parallel(block_M, rotate_dim):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    theta_k = Freqs[kv_real, d]
                    kx = T.Cast(accum_dtype, K[bz, kv_real, bx // groups, d])
                    Ks_shared[i, d] = T.Cast(dtype, kx * T.sin(theta_k))
                else:
                    Ks_shared[i, d] = T.Cast(dtype, 0)

            for i, d in T.Parallel(block_M, dim_v):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    V_shared[i, d] = V[bz, kv_real, bx // groups, d]
                else:
                    V_shared[i, d] = T.Cast(dtype, 0)

            T.clear(dv)
            T.clear(dKc_acc)
            T.clear(dKs_acc)

            kv_lo = by * block_M
            kv_hi = (by + 1) * block_M
            q_lo = kv_lo
            q_hi = T.min(kv_hi + window_size + chunk_size, seq_len)
            loop_st = T.floordiv(q_lo, block_N)
            loop_ed = T.ceildiv(q_hi, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                for i, d in T.Parallel(block_N, dim_qk):
                    tq = k * block_N + i
                    if tq < seq_len:
                        if d < rotate_dim:
                            theta_q = Freqs[tq, d] - Bias[i_h_kv, i_g, d]
                            qx = T.Cast(accum_dtype, Q[bz, tq, bx, d])
                            Qc_shared[i, d] = T.Cast(dtype, qx * T.cos(theta_q))
                        else:
                            Qc_shared[i, d] = Q[bz, tq, bx, d]
                    else:
                        Qc_shared[i, d] = T.Cast(dtype, 0)

                for i, d in T.Parallel(block_N, rotate_dim):
                    tq = k * block_N + i
                    if tq < seq_len:
                        theta_q = Freqs[tq, d] - Bias[i_h_kv, i_g, d]
                        qx = T.Cast(accum_dtype, Q[bz, tq, bx, d])
                        Qs_shared[i, d] = T.Cast(dtype, qx * T.sin(theta_q))
                    else:
                        Qs_shared[i, d] = T.Cast(dtype, 0)

                T.clear(qkT)
                T.gemm(Kc_shared, Qc_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Ks_shared, Qs_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        lse_shared[i] = lse[bz, tq, bx]
                    else:
                        lse_shared[i] = T.Cast(accum_dtype, 0)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])

                if expand_to_chunk and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len) & (T.floormod(kv_real + 1, chunk_size) != 0),
                            qkT[i, j], 0.0)
                elif expand_to_chunk and (not mask_lmk):
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len),
                            qkT[i, j], 0.0)
                elif (not expand_to_chunk) and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = q_real - window_size + 1
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len) & (T.floormod(kv_real + 1, chunk_size) != 0),
                            qkT[i, j], 0.0)
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = q_real - window_size + 1
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len),
                            qkT[i, j], 0.0)

                for i, d in T.Parallel(block_N, dim_v):
                    tq = k * block_N + i
                    if tq < seq_len:
                        do_sh[i, d] = dO[bz, tq, bx, d]
                    else:
                        do_sh[i, d] = T.Cast(dtype, 0)
                T.clear(dsT)
                T.gemm(V_shared, do_sh, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do_sh, dv, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        delta_sh[i] = Delta[bz, bx, tq]
                    else:
                        delta_sh[i] = T.Cast(accum_dtype, 0)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = T.Cast(dtype, qkT[i, j] * (dsT[i, j] - delta_sh[j]) * sm_scale)

                T.gemm(dsT_cast, Qc_shared, dKc_acc, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(dsT_cast, Qs_shared, dKs_acc, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dqc_frag)
                T.gemm(dsT_shared, Kc_shared, dqc_frag, transpose_A=True)
                T.copy(dqc_frag, dqc_out_shared)
                T.clear(dqs_frag)
                T.gemm(dsT_shared, Ks_shared, dqs_frag, transpose_A=True)
                T.copy(dqs_frag, dqs_out_shared)

                for i, d in T.Parallel(block_N, rotate_dim):
                    tq = k * block_N + i
                    if tq < seq_len:
                        theta_q = Freqs[tq, d] - Bias[i_h_kv, i_g, d]
                        cos_tq = T.cos(theta_q)
                        sin_tq = T.sin(theta_q)
                        dqc_val = dqc_out_shared[i, d]
                        dqs_val = dqs_out_shared[i, d]
                        dq_val = dqc_val * cos_tq + dqs_val * sin_tq
                        T.atomic_add(dQ[bz, tq, bx, d], dq_val)
                        qc_v = T.Cast(accum_dtype, Qc_shared[i, d])
                        qs_v = T.Cast(accum_dtype, Qs_shared[i, d])
                        dtheta = dqs_val * qc_v - dqc_val * qs_v
                        T.atomic_add(dBias[i_h_kv, i_g, d], -dtheta)

                for i, d in T.Parallel(block_N, dim_qk):
                    tq = k * block_N + i
                    if (tq < seq_len) and (d >= rotate_dim):
                        T.atomic_add(dQ[bz, tq, bx, d], dqc_out_shared[i, d])

            for i, d in T.Parallel(block_M, dim_v):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    T.atomic_add(dV[bz, kv_real, bx // groups, d], dv[i, d])

            T.copy(dKc_acc, dKc_out_shared)
            T.copy(dKs_acc, dKs_out_shared)
            for i, d in T.Parallel(block_M, rotate_dim):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    theta_k = Freqs[kv_real, d]
                    dk_val = dKc_out_shared[i, d] * T.cos(theta_k) + dKs_out_shared[i, d] * T.sin(theta_k)
                    T.atomic_add(dK[bz, kv_real, bx // groups, d], dk_val)

            for i, d in T.Parallel(block_M, dim_qk):
                kv_real = by * block_M + i
                if (kv_real < seq_len) and (d >= rotate_dim):
                    T.atomic_add(dK[bz, kv_real, bx // groups, d], dKc_out_shared[i, d])

    return bwd


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def flex_attn_pope_qkprerotated_bwd(
    batch, heads, seq_len, dim_qk, dim_v, rotate_dim, freqs_len,
    window_size, chunk_size, mask_lmk, expand_to_chunk,
    block_M, block_N, threads=256, num_stages=2, groups=1,
    dtype="bfloat16", accum_dtype="float",
    sm_scale=None,
):
    """Backward for Q/K-prerotated PoPE attention.

    The kernel reads precomputed Qc/Qs/Kc/Ks for score recomputation and
    converts dQc/dQs and dKc/dKs back to raw dQ/dK at the end.
    """
    if sm_scale is None:
        sm_scale = (1.0 / dim_qk) ** 0.5
    scale = sm_scale * 1.44269504
    head_kv = heads // groups
    qc_shape = [batch, seq_len, heads, dim_qk]
    qs_shape = [batch, seq_len, heads, rotate_dim]
    kc_shape = [batch, seq_len, head_kv, dim_qk]
    ks_shape = [batch, seq_len, head_kv, rotate_dim]
    v_shape = [batch, seq_len, head_kv, dim_v]
    freqs_shape = [freqs_len, rotate_dim]
    bias_shape = [head_kv, groups, rotate_dim]
    dbias_shape = [head_kv, groups, rotate_dim]

    @T.prim_func
    def bwd(
        Qc: T.Tensor(qc_shape, dtype),
        Qs: T.Tensor(qs_shape, dtype),
        Kc: T.Tensor(kc_shape, dtype),
        Ks: T.Tensor(ks_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        dO: T.Tensor([batch, seq_len, heads, dim_v], dtype),
        lse: T.Tensor([batch, seq_len, heads], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        Freqs: T.Tensor(freqs_shape, accum_dtype),
        Bias: T.Tensor(bias_shape, accum_dtype),
        dQ: T.Tensor(qc_shape, accum_dtype),
        dK: T.Tensor(kc_shape, accum_dtype),
        dV: T.Tensor(v_shape, accum_dtype),
        dBias: T.Tensor(dbias_shape, accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            Kc_shared = T.alloc_shared([block_M, dim_qk], dtype)
            Ks_shared = T.alloc_shared([block_M, rotate_dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            Qc_shared = T.alloc_shared([block_N, dim_qk], dtype)
            Qs_shared = T.alloc_shared([block_N, rotate_dim], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta_sh = T.alloc_shared([block_N], accum_dtype)
            do_sh = T.alloc_shared([block_N, dim_v], dtype)
            dqc_out_shared = T.alloc_shared([block_N, dim_qk], accum_dtype)
            dqs_out_shared = T.alloc_shared([block_N, rotate_dim], accum_dtype)
            dKc_out_shared = T.alloc_shared([block_M, dim_qk], accum_dtype)
            dKs_out_shared = T.alloc_shared([block_M, rotate_dim], accum_dtype)
            dKc_acc = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dKs_acc = T.alloc_fragment([block_M, rotate_dim], accum_dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            dqc_frag = T.alloc_fragment([block_N, dim_qk], accum_dtype)
            dqs_frag = T.alloc_fragment([block_N, rotate_dim], accum_dtype)

            i_h_kv = bx // groups
            i_g = bx % groups

            for i, d in T.Parallel(block_M, dim_qk):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    Kc_shared[i, d] = Kc[bz, kv_real, bx // groups, d]
                else:
                    Kc_shared[i, d] = T.Cast(dtype, 0)

            for i, d in T.Parallel(block_M, rotate_dim):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    Ks_shared[i, d] = Ks[bz, kv_real, bx // groups, d]
                else:
                    Ks_shared[i, d] = T.Cast(dtype, 0)

            for i, d in T.Parallel(block_M, dim_v):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    V_shared[i, d] = V[bz, kv_real, bx // groups, d]
                else:
                    V_shared[i, d] = T.Cast(dtype, 0)

            T.clear(dv)
            T.clear(dKc_acc)
            T.clear(dKs_acc)

            kv_lo = by * block_M
            kv_hi = (by + 1) * block_M
            q_lo = kv_lo
            q_hi = T.min(kv_hi + window_size + chunk_size, seq_len)
            loop_st = T.floordiv(q_lo, block_N)
            loop_ed = T.ceildiv(q_hi, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                for i, d in T.Parallel(block_N, dim_qk):
                    tq = k * block_N + i
                    if tq < seq_len:
                        Qc_shared[i, d] = Qc[bz, tq, bx, d]
                    else:
                        Qc_shared[i, d] = T.Cast(dtype, 0)

                for i, d in T.Parallel(block_N, rotate_dim):
                    tq = k * block_N + i
                    if tq < seq_len:
                        Qs_shared[i, d] = Qs[bz, tq, bx, d]
                    else:
                        Qs_shared[i, d] = T.Cast(dtype, 0)

                T.clear(qkT)
                T.gemm(Kc_shared, Qc_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Ks_shared, Qs_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        lse_shared[i] = lse[bz, tq, bx]
                    else:
                        lse_shared[i] = T.Cast(accum_dtype, 0)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])

                if expand_to_chunk and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len) & (T.floormod(kv_real + 1, chunk_size) != 0),
                            qkT[i, j], 0.0)
                elif expand_to_chunk and (not mask_lmk):
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len),
                            qkT[i, j], 0.0)
                elif (not expand_to_chunk) and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = q_real - window_size + 1
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len) & (T.floormod(kv_real + 1, chunk_size) != 0),
                            qkT[i, j], 0.0)
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = q_real - window_size + 1
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len),
                            qkT[i, j], 0.0)

                for i, d in T.Parallel(block_N, dim_v):
                    tq = k * block_N + i
                    if tq < seq_len:
                        do_sh[i, d] = dO[bz, tq, bx, d]
                    else:
                        do_sh[i, d] = T.Cast(dtype, 0)
                T.clear(dsT)
                T.gemm(V_shared, do_sh, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do_sh, dv, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        delta_sh[i] = Delta[bz, bx, tq]
                    else:
                        delta_sh[i] = T.Cast(accum_dtype, 0)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = T.Cast(dtype, qkT[i, j] * (dsT[i, j] - delta_sh[j]) * sm_scale)

                T.gemm(dsT_cast, Qc_shared, dKc_acc, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(dsT_cast, Qs_shared, dKs_acc, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dqc_frag)
                T.gemm(dsT_shared, Kc_shared, dqc_frag, transpose_A=True)
                T.copy(dqc_frag, dqc_out_shared)
                T.clear(dqs_frag)
                T.gemm(dsT_shared, Ks_shared, dqs_frag, transpose_A=True)
                T.copy(dqs_frag, dqs_out_shared)

                for i, d in T.Parallel(block_N, rotate_dim):
                    tq = k * block_N + i
                    if tq < seq_len:
                        theta_q = Freqs[tq, d] - Bias[i_h_kv, i_g, d]
                        cos_tq = T.cos(theta_q)
                        sin_tq = T.sin(theta_q)
                        dqc_val = dqc_out_shared[i, d]
                        dqs_val = dqs_out_shared[i, d]
                        dq_val = dqc_val * cos_tq + dqs_val * sin_tq
                        T.atomic_add(dQ[bz, tq, bx, d], dq_val)
                        qc_v = T.Cast(accum_dtype, Qc_shared[i, d])
                        qs_v = T.Cast(accum_dtype, Qs_shared[i, d])
                        dtheta = dqs_val * qc_v - dqc_val * qs_v
                        T.atomic_add(dBias[i_h_kv, i_g, d], -dtheta)

                for i, d in T.Parallel(block_N, dim_qk):
                    tq = k * block_N + i
                    if (tq < seq_len) and (d >= rotate_dim):
                        T.atomic_add(dQ[bz, tq, bx, d], dqc_out_shared[i, d])

            for i, d in T.Parallel(block_M, dim_v):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    T.atomic_add(dV[bz, kv_real, bx // groups, d], dv[i, d])

            T.copy(dKc_acc, dKc_out_shared)
            T.copy(dKs_acc, dKs_out_shared)
            for i, d in T.Parallel(block_M, rotate_dim):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    theta_k = Freqs[kv_real, d]
                    dk_val = dKc_out_shared[i, d] * T.cos(theta_k) + dKs_out_shared[i, d] * T.sin(theta_k)
                    T.atomic_add(dK[bz, kv_real, bx // groups, d], dk_val)

            for i, d in T.Parallel(block_M, dim_qk):
                kv_real = by * block_M + i
                if (kv_real < seq_len) and (d >= rotate_dim):
                    T.atomic_add(dK[bz, kv_real, bx // groups, d], dKc_out_shared[i, d])

    return bwd


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def flex_attn_pope_split_bwd(
    batch, heads, seq_len, dim_qk, dim_v, rotate_dim,
    window_size, chunk_size, mask_lmk, expand_to_chunk,
    block_M, block_N, threads=256, num_stages=2, groups=1,
    dtype="bfloat16", accum_dtype="float",
    sm_scale=None,
):
    """Backward for externally pre-rotated split inputs.

    Returns gradients w.r.t. Qc/Qs/Kc/Ks/V only. Gradients w.r.t. raw q/k/bias
    are handled by PyTorch autograd through the external prerotate graph.
    """
    if sm_scale is None:
        sm_scale = (1.0 / dim_qk) ** 0.5
    scale = sm_scale * 1.44269504
    head_kv = heads // groups
    qc_shape = [batch, seq_len, heads, dim_qk]
    qs_shape = [batch, seq_len, heads, rotate_dim]
    kc_shape = [batch, seq_len, head_kv, dim_qk]
    ks_shape = [batch, seq_len, head_kv, rotate_dim]
    v_shape = [batch, seq_len, head_kv, dim_v]

    @T.prim_func
    def bwd(
        Qc: T.Tensor(qc_shape, dtype),
        Qs: T.Tensor(qs_shape, dtype),
        Kc: T.Tensor(kc_shape, dtype),
        Ks: T.Tensor(ks_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        dO: T.Tensor([batch, seq_len, heads, dim_v], dtype),
        lse: T.Tensor([batch, seq_len, heads], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQc: T.Tensor(qc_shape, accum_dtype),
        dQs: T.Tensor(qs_shape, accum_dtype),
        dKc: T.Tensor(kc_shape, accum_dtype),
        dKs: T.Tensor(ks_shape, accum_dtype),
        dV: T.Tensor(v_shape, accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            Kc_shared = T.alloc_shared([block_M, dim_qk], dtype)
            Ks_shared = T.alloc_shared([block_M, rotate_dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            Qc_shared = T.alloc_shared([block_N, dim_qk], dtype)
            Qs_shared = T.alloc_shared([block_N, rotate_dim], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta_sh = T.alloc_shared([block_N], accum_dtype)
            do_sh = T.alloc_shared([block_N, dim_v], dtype)
            dKc_acc = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dKs_acc = T.alloc_fragment([block_M, rotate_dim], accum_dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            dqc_frag = T.alloc_fragment([block_N, dim_qk], accum_dtype)
            dqs_frag = T.alloc_fragment([block_N, rotate_dim], accum_dtype)

            for i, d in T.Parallel(block_M, dim_qk):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    Kc_shared[i, d] = Kc[bz, kv_real, bx // groups, d]
                else:
                    Kc_shared[i, d] = T.Cast(dtype, 0)

            for i, d in T.Parallel(block_M, rotate_dim):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    Ks_shared[i, d] = Ks[bz, kv_real, bx // groups, d]
                else:
                    Ks_shared[i, d] = T.Cast(dtype, 0)

            for i, d in T.Parallel(block_M, dim_v):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    V_shared[i, d] = V[bz, kv_real, bx // groups, d]
                else:
                    V_shared[i, d] = T.Cast(dtype, 0)

            T.clear(dv)
            T.clear(dKc_acc)
            T.clear(dKs_acc)

            kv_lo = by * block_M
            kv_hi = (by + 1) * block_M
            q_lo = kv_lo
            q_hi = T.min(kv_hi + window_size + chunk_size, seq_len)
            loop_st = T.floordiv(q_lo, block_N)
            loop_ed = T.ceildiv(q_hi, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                for i, d in T.Parallel(block_N, dim_qk):
                    tq = k * block_N + i
                    if tq < seq_len:
                        Qc_shared[i, d] = Qc[bz, tq, bx, d]
                    else:
                        Qc_shared[i, d] = T.Cast(dtype, 0)

                for i, d in T.Parallel(block_N, rotate_dim):
                    tq = k * block_N + i
                    if tq < seq_len:
                        Qs_shared[i, d] = Qs[bz, tq, bx, d]
                    else:
                        Qs_shared[i, d] = T.Cast(dtype, 0)

                T.clear(qkT)
                T.gemm(Kc_shared, Qc_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Ks_shared, Qs_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        lse_shared[i] = lse[bz, tq, bx]
                    else:
                        lse_shared[i] = T.Cast(accum_dtype, 0)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])

                if expand_to_chunk and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len) & (T.floormod(kv_real + 1, chunk_size) != 0),
                            qkT[i, j], 0.0)
                elif expand_to_chunk and (not mask_lmk):
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len),
                            qkT[i, j], 0.0)
                elif (not expand_to_chunk) and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = q_real - window_size + 1
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len) & (T.floormod(kv_real + 1, chunk_size) != 0),
                            qkT[i, j], 0.0)
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        cs = q_real - window_size + 1
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= cs) & (kv_real <= q_real) & (kv_real < seq_len)
                            & (q_real < seq_len),
                            qkT[i, j], 0.0)

                for i, d in T.Parallel(block_N, dim_v):
                    tq = k * block_N + i
                    if tq < seq_len:
                        do_sh[i, d] = dO[bz, tq, bx, d]
                    else:
                        do_sh[i, d] = T.Cast(dtype, 0)
                T.clear(dsT)
                T.gemm(V_shared, do_sh, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do_sh, dv, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        delta_sh[i] = Delta[bz, bx, tq]
                    else:
                        delta_sh[i] = T.Cast(accum_dtype, 0)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = T.Cast(dtype, qkT[i, j] * (dsT[i, j] - delta_sh[j]) * sm_scale)

                T.gemm(dsT_cast, Qc_shared, dKc_acc, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(dsT_cast, Qs_shared, dKs_acc, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dqc_frag)
                T.gemm(dsT_shared, Kc_shared, dqc_frag, transpose_A=True)
                T.clear(dqs_frag)
                T.gemm(dsT_shared, Ks_shared, dqs_frag, transpose_A=True)

                for i, d in T.Parallel(block_N, dim_qk):
                    tq = k * block_N + i
                    if tq < seq_len:
                        T.atomic_add(dQc[bz, tq, bx, d], dqc_frag[i, d])
                for i, d in T.Parallel(block_N, rotate_dim):
                    tq = k * block_N + i
                    if tq < seq_len:
                        T.atomic_add(dQs[bz, tq, bx, d], dqs_frag[i, d])

            for i, d in T.Parallel(block_M, dim_v):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    T.atomic_add(dV[bz, kv_real, bx // groups, d], dv[i, d])
            for i, d in T.Parallel(block_M, dim_qk):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    T.atomic_add(dKc[bz, kv_real, bx // groups, d], dKc_acc[i, d])
            for i, d in T.Parallel(block_M, rotate_dim):
                kv_real = by * block_M + i
                if kv_real < seq_len:
                    T.atomic_add(dKs[bz, kv_real, bx // groups, d], dKs_acc[i, d])

    return bwd


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------
class _FlexAttnPopeTL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_blhd, k_blhd, v_blhd, freqs, bias,
                window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache,
                sm_scale):
        BATCH, Q_LEN, H, D_QK = q_blhd.shape
        KV_LEN = k_blhd.shape[1]
        H_KV = k_blhd.shape[2]
        D_V = v_blhd.shape[-1]
        groups = H // H_KV
        rotate_dim = freqs.shape[-1]
        freqs_len = freqs.shape[0]

        block_M = 128
        block_N = 64

        # Auto-derive kernel dtype from q's torch dtype.
        dtype_str = _torch_dtype_to_str(q_blhd.dtype)

        mod = flex_attn_pope_fwd(
            BATCH, H, Q_LEN, KV_LEN, D_QK, D_V,
            rotate_dim, freqs_len,
            window_size, chunk_size, mask_lmk, expand_to_chunk,
            block_M, block_N, groups, use_cache=use_cache,
            dtype=dtype_str,
            sm_scale=sm_scale,
        )
        o, lse = mod(q_blhd, k_blhd, v_blhd, freqs, bias)

        ctx.save_for_backward(q_blhd, k_blhd, v_blhd, o, lse, freqs, bias)
        ctx.window_size = window_size
        ctx.chunk_size = chunk_size
        ctx.mask_lmk = mask_lmk
        ctx.expand_to_chunk = expand_to_chunk
        ctx.use_cache = use_cache
        ctx.dtype_str = dtype_str
        ctx.sm_scale = sm_scale
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q, k, v, o, lse, freqs, bias = ctx.saved_tensors
        if ctx.use_cache:
            raise RuntimeError("backward not supported in inference mode")

        BATCH, N_CTX, H, D_QK = q.shape
        H_KV = v.shape[-2]
        D_V = v.shape[-1]
        groups = H // H_KV
        rotate_dim = freqs.shape[-1]
        freqs_len = freqs.shape[0]
        dtype_str = ctx.dtype_str

        do = do.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()

        block_M = 128
        block_N = 32

        mod_prep = flex_attn_pope_bwd_preprocess(BATCH, H, N_CTX, D_V, dtype=dtype_str)
        mod_post = flex_attn_pope_bwd_postprocess(BATCH, H, N_CTX, D_QK, dtype=dtype_str)
        delta = mod_prep(o, do)

        kernel = flex_attn_pope_bwd(
            BATCH, H, N_CTX, D_QK, D_V,
            rotate_dim, freqs_len,
            ctx.window_size, ctx.chunk_size,
            ctx.mask_lmk, ctx.expand_to_chunk,
            block_M, block_N, threads=256, num_stages=1, groups=groups,
            dtype=dtype_str,
            sm_scale=ctx.sm_scale,
        )
        dq = torch.zeros([BATCH, N_CTX, H, D_QK], dtype=torch.float32, device=q.device)
        dk = torch.zeros([BATCH, N_CTX, H_KV, D_QK], dtype=torch.float32, device=q.device)
        dv = torch.zeros([BATCH, N_CTX, H_KV, D_V], dtype=torch.float32, device=q.device)
        dbias = torch.zeros_like(bias, dtype=torch.float32)
        kernel(q, k, v, do, lse, delta, freqs, bias, dq, dk, dv, dbias)
        dq = mod_post(dq)
        dk = dk.to(q.dtype)
        dv = dv.to(q.dtype)
        dbias = dbias.to(bias.dtype)
        # return order must match forward args
        return dq, dk, dv, None, dbias, None, None, None, None, None, None


class _FlexAttnPopeKPreRotTL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_blhd, k_blhd, v_blhd, freqs, bias,
                window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache,
                sm_scale):
        """Forward with K pre-rotated once; backward reuses raw fused bwd."""
        BATCH, Q_LEN, H, D_QK = q_blhd.shape
        KV_LEN = k_blhd.shape[1]
        H_KV = k_blhd.shape[2]
        D_V = v_blhd.shape[-1]
        groups = H // H_KV
        rotate_dim = freqs.shape[-1]
        freqs_len = freqs.shape[0]

        block_M = 128
        block_N = 64
        dtype_str = _torch_dtype_to_str(q_blhd.dtype)

        pre_mod = pope_pre_rotate_k_kernel(
            BATCH, H_KV, KV_LEN, D_QK, rotate_dim, freqs_len,
            block_N=128, dtype=dtype_str,
        )
        kc_blhd, ks_blhd = pre_mod(k_blhd, freqs)

        mod = flex_attn_pope_kprerotated_fwd(
            BATCH, H, Q_LEN, KV_LEN, D_QK, D_V,
            rotate_dim, freqs_len,
            window_size, chunk_size, mask_lmk, expand_to_chunk,
            block_M, block_N, groups, use_cache=use_cache,
            dtype=dtype_str,
            sm_scale=sm_scale,
        )
        o, lse = mod(q_blhd, kc_blhd, ks_blhd, v_blhd, freqs, bias)

        ctx.save_for_backward(q_blhd, k_blhd, v_blhd, o, lse, freqs, bias)
        ctx.window_size = window_size
        ctx.chunk_size = chunk_size
        ctx.mask_lmk = mask_lmk
        ctx.expand_to_chunk = expand_to_chunk
        ctx.use_cache = use_cache
        ctx.dtype_str = dtype_str
        ctx.sm_scale = sm_scale
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        return _FlexAttnPopeTL.backward(ctx, do, dlse)


class _FlexAttnPopeQKPreRotTL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_blhd, k_blhd, v_blhd, freqs, bias,
                window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache,
                sm_scale):
        """Forward/backward with Q and K pre-rotated once."""
        BATCH, Q_LEN, H, D_QK = q_blhd.shape
        KV_LEN = k_blhd.shape[1]
        H_KV = k_blhd.shape[2]
        D_V = v_blhd.shape[-1]
        groups = H // H_KV
        rotate_dim = freqs.shape[-1]
        freqs_len = freqs.shape[0]

        block_M = 128
        block_N = 64
        dtype_str = _torch_dtype_to_str(q_blhd.dtype)

        q_pre_mod = pope_pre_rotate_q_kernel(
            BATCH, H, Q_LEN, KV_LEN, D_QK, rotate_dim, freqs_len,
            block_M=128, groups=groups, dtype=dtype_str,
        )
        qc_blhd, qs_blhd = q_pre_mod(q_blhd, freqs, bias)

        k_pre_mod = pope_pre_rotate_k_kernel(
            BATCH, H_KV, KV_LEN, D_QK, rotate_dim, freqs_len,
            block_N=128, dtype=dtype_str,
        )
        kc_blhd, ks_blhd = k_pre_mod(k_blhd, freqs)

        mod = flex_attn_pope_qkprerotated_fwd(
            BATCH, H, Q_LEN, KV_LEN, D_QK, D_V,
            rotate_dim,
            window_size, chunk_size, mask_lmk, expand_to_chunk,
            block_M, block_N, groups, use_cache=use_cache,
            dtype=dtype_str,
            sm_scale=sm_scale,
        )
        o, lse = mod(qc_blhd, qs_blhd, kc_blhd, ks_blhd, v_blhd)

        ctx.save_for_backward(qc_blhd, qs_blhd, kc_blhd, ks_blhd, v_blhd, o, lse, freqs, bias)
        ctx.window_size = window_size
        ctx.chunk_size = chunk_size
        ctx.mask_lmk = mask_lmk
        ctx.expand_to_chunk = expand_to_chunk
        ctx.use_cache = use_cache
        ctx.dtype_str = dtype_str
        ctx.sm_scale = sm_scale
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        qc, qs, kc, ks, v, o, lse, freqs, bias = ctx.saved_tensors
        if ctx.use_cache:
            raise RuntimeError("backward not supported in inference mode")

        BATCH, N_CTX, H, D_QK = qc.shape
        H_KV = kc.shape[-2]
        D_V = v.shape[-1]
        groups = H // H_KV
        rotate_dim = freqs.shape[-1]
        freqs_len = freqs.shape[0]
        dtype_str = ctx.dtype_str

        do = do.contiguous()
        qc = qc.contiguous()
        qs = qs.contiguous()
        kc = kc.contiguous()
        ks = ks.contiguous()
        v = v.contiguous()
        o = o.contiguous()

        block_M = 128
        block_N = 32

        mod_prep = flex_attn_pope_bwd_preprocess(BATCH, H, N_CTX, D_V, dtype=dtype_str)
        mod_post = flex_attn_pope_bwd_postprocess(BATCH, H, N_CTX, D_QK, dtype=dtype_str)
        delta = mod_prep(o, do)

        kernel = flex_attn_pope_qkprerotated_bwd(
            BATCH, H, N_CTX, D_QK, D_V,
            rotate_dim, freqs_len,
            ctx.window_size, ctx.chunk_size,
            ctx.mask_lmk, ctx.expand_to_chunk,
            block_M, block_N, threads=256, num_stages=1, groups=groups,
            dtype=dtype_str,
            sm_scale=ctx.sm_scale,
        )
        dq = torch.zeros([BATCH, N_CTX, H, D_QK], dtype=torch.float32, device=qc.device)
        dk = torch.zeros([BATCH, N_CTX, H_KV, D_QK], dtype=torch.float32, device=qc.device)
        dv = torch.zeros([BATCH, N_CTX, H_KV, D_V], dtype=torch.float32, device=qc.device)
        dbias = torch.zeros_like(bias, dtype=torch.float32)
        kernel(qc, qs, kc, ks, v, do, lse, delta, freqs, bias, dq, dk, dv, dbias)
        dq = mod_post(dq)
        dk = dk.to(qc.dtype)
        dv = dv.to(qc.dtype)
        dbias = dbias.to(bias.dtype)
        return dq, dk, dv, None, dbias, None, None, None, None, None, None


def _pope_prerotate_qk_split_torch(q, k, freqs, bias_3d):
    """Pure-PyTorch PoPE split prerotate for flex-attn.

    Returns q_cos/q_sin/k_cos/k_sin. Because this runs in normal PyTorch, the
    gradients of raw q/k and bias are handled by autograd from the split grads
    returned by `_FlexAttnPopeTorchPreRotTL`.
    """
    B, Lq, Hq, D = q.shape
    _, Lk, Hkv, _ = k.shape
    R = freqs.shape[-1]
    G = Hq // Hkv
    q_dtype = q.dtype
    k_dtype = k.dtype
    q_offset = Lk - Lq

    common_theta_dtype = freqs.dtype if freqs.dtype == bias_3d.dtype else torch.float32

    q_pos = torch.arange(Lq, device=q.device, dtype=torch.long) + int(q_offset)
    fq = freqs.index_select(0, q_pos).to(common_theta_dtype)
    theta_q = fq.view(1, Lq, 1, R) - bias_3d.reshape(1, 1, Hq, R).to(common_theta_dtype)
    cos_q = torch.cos(theta_q.float()).to(q_dtype)
    sin_q = torch.sin(theta_q.float()).to(q_dtype)

    q_rot_f = q[..., :R].float()
    qc_rot = (q_rot_f * cos_q.float()).to(q_dtype)
    qs = (q_rot_f * sin_q.float()).to(q_dtype)
    if R < D:
        q_cos = torch.cat([qc_rot, q[..., R:]], dim=-1).contiguous()
    else:
        q_cos = qc_rot.contiguous()
    q_sin = qs.contiguous()

    k_pos = torch.arange(Lk, device=k.device, dtype=torch.long)
    fk = freqs.index_select(0, k_pos).to(common_theta_dtype)
    cos_k = torch.cos(fk.float()).to(k_dtype).view(1, Lk, 1, R)
    sin_k = torch.sin(fk.float()).to(k_dtype).view(1, Lk, 1, R)

    k_rot_f = k[..., :R].float()
    kc_rot = (k_rot_f * cos_k.float()).to(k_dtype)
    ks = (k_rot_f * sin_k.float()).to(k_dtype)
    if R < D:
        k_cos = torch.cat([kc_rot, k[..., R:]], dim=-1).contiguous()
    else:
        k_cos = kc_rot.contiguous()
    k_sin = ks.contiguous()

    return q_cos, q_sin, k_cos, k_sin


class _FlexAttnPopeTorchPreRotTL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_cos, q_sin, k_cos, k_sin, v_blhd,
                window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache,
                sm_scale):
        BATCH, Q_LEN, H, D_QK = q_cos.shape
        KV_LEN = k_cos.shape[1]
        H_KV = k_cos.shape[2]
        D_V = v_blhd.shape[-1]
        groups = H // H_KV
        rotate_dim = q_sin.shape[-1]

        block_M = 128
        block_N = 64
        dtype_str = _torch_dtype_to_str(q_cos.dtype)

        mod = flex_attn_pope_qkprerotated_fwd(
            BATCH, H, Q_LEN, KV_LEN, D_QK, D_V,
            rotate_dim,
            window_size, chunk_size, mask_lmk, expand_to_chunk,
            block_M, block_N, groups, use_cache=use_cache,
            dtype=dtype_str,
            sm_scale=sm_scale,
        )
        o, lse = mod(q_cos, q_sin, k_cos, k_sin, v_blhd)

        ctx.save_for_backward(q_cos, q_sin, k_cos, k_sin, v_blhd, o, lse)
        ctx.window_size = window_size
        ctx.chunk_size = chunk_size
        ctx.mask_lmk = mask_lmk
        ctx.expand_to_chunk = expand_to_chunk
        ctx.use_cache = use_cache
        ctx.dtype_str = dtype_str
        ctx.sm_scale = sm_scale
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q_cos, q_sin, k_cos, k_sin, v, o, lse = ctx.saved_tensors
        if ctx.use_cache:
            raise RuntimeError("backward not supported in inference mode")

        BATCH, N_CTX, H, D_QK = q_cos.shape
        H_KV = k_cos.shape[-2]
        D_V = v.shape[-1]
        groups = H // H_KV
        rotate_dim = q_sin.shape[-1]
        dtype_str = ctx.dtype_str

        do = do.contiguous()
        q_cos = q_cos.contiguous()
        q_sin = q_sin.contiguous()
        k_cos = k_cos.contiguous()
        k_sin = k_sin.contiguous()
        v = v.contiguous()
        o = o.contiguous()

        block_M = 128
        block_N = 32

        mod_prep = flex_attn_pope_bwd_preprocess(BATCH, H, N_CTX, D_V, dtype=dtype_str)
        mod_post_d = flex_attn_pope_bwd_postprocess(BATCH, H, N_CTX, D_QK, dtype=dtype_str)
        mod_post_r = flex_attn_pope_bwd_postprocess(BATCH, H, N_CTX, rotate_dim, dtype=dtype_str)
        delta = mod_prep(o, do)

        kernel = flex_attn_pope_split_bwd(
            BATCH, H, N_CTX, D_QK, D_V, rotate_dim,
            ctx.window_size, ctx.chunk_size,
            ctx.mask_lmk, ctx.expand_to_chunk,
            block_M, block_N, threads=256, num_stages=1, groups=groups,
            dtype=dtype_str,
            sm_scale=ctx.sm_scale,
        )
        dqc = torch.zeros([BATCH, N_CTX, H, D_QK], dtype=torch.float32, device=q_cos.device)
        dqs = torch.zeros([BATCH, N_CTX, H, rotate_dim], dtype=torch.float32, device=q_cos.device)
        dkc = torch.zeros([BATCH, N_CTX, H_KV, D_QK], dtype=torch.float32, device=q_cos.device)
        dks = torch.zeros([BATCH, N_CTX, H_KV, rotate_dim], dtype=torch.float32, device=q_cos.device)
        dv = torch.zeros([BATCH, N_CTX, H_KV, D_V], dtype=torch.float32, device=q_cos.device)
        kernel(q_cos, q_sin, k_cos, k_sin, v, do, lse, delta, dqc, dqs, dkc, dks, dv)
        dqc = mod_post_d(dqc)
        dqs = mod_post_r(dqs)
        dkc = dkc.to(q_cos.dtype)
        dks = dks.to(q_cos.dtype)
        dv = dv.to(v.dtype)
        return dqc, dqs, dkc, dks, dv, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def flex_attn_pope_tl(q, k, v, freqs, bias,
                      window_size, chunk_size,
                      training=True, mask_lmk=True, expand_to_chunk=True,
                      sm_scale=None):
    """Tilelang flex attention with PoPE.

    Args:
        q: (B, Lq, Hq, D)  already softplus'd magnitude (bf16)
        k: (B, Lk, Hkv, D)  already softplus'd magnitude (bf16)
        v: (B, Lk, Hkv, Dv)  (bf16)
        freqs: (L_total, R)  fp32
        bias:  (Hq, R) or (Hkv, G, R)  fp32, theta_q = freqs - bias
        sm_scale: optional softmax scale (natural base). Defaults to
            1/sqrt(D) (the original head_dim, NOT D+R), to stay consistent
            with HSA_block_M_head_pope* and online_softmax_topk_head_pope*.
    Returns: o (B,Lq,Hq,Dv), lse (B,Lq,Hq) natural log
    """
    Hq, Hkv, R = q.shape[2], k.shape[2], freqs.shape[-1]
    G = Hq // Hkv
    # Kernel declares Bias/Freqs as accum_dtype (fp32). Caller's bias may follow
    # the model dtype (e.g. bf16 Parameter), so force-cast here.
    bias = bias.float()
    freqs = freqs.float()
    if bias.dim() == 2:
        bias = bias.view(Hkv, G, R).contiguous()
    else:
        bias = bias.contiguous()
    freqs = freqs.contiguous()

    q_blhd = q.contiguous()
    k_blhd = k.contiguous()
    v_blhd = v.contiguous()

    use_cache = not training
    o_blhd, lse_log2 = _FlexAttnPopeTL.apply(
        q_blhd, k_blhd, v_blhd, freqs, bias,
        window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache,
        sm_scale,
    )
    lse = lse_log2 * 0.6931471805599453
    return o_blhd, lse


def flex_attn_pope_tl_k_prerotate(q, k, v, freqs, bias,
                                  window_size, chunk_size,
                                  training=True, mask_lmk=True, expand_to_chunk=True,
                                  sm_scale=None):
    """Tilelang flex attention with K-side PoPE pre-rotated once.

    Forward computes Kc/Ks with a standalone TileLang kernel, then runs a fused
    attention forward that only rotates Q on-chip. Backward intentionally reuses
    the raw fused backward to keep gradients identical to `flex_attn_pope_tl`.
    """
    Hq, Hkv, R = q.shape[2], k.shape[2], freqs.shape[-1]
    G = Hq // Hkv
    bias = bias.float()
    freqs = freqs.float()
    if bias.dim() == 2:
        bias = bias.view(Hkv, G, R).contiguous()
    else:
        bias = bias.contiguous()
    freqs = freqs.contiguous()

    q_blhd = q.contiguous()
    k_blhd = k.contiguous()
    v_blhd = v.contiguous()

    use_cache = not training
    o_blhd, lse_log2 = _FlexAttnPopeKPreRotTL.apply(
        q_blhd, k_blhd, v_blhd, freqs, bias,
        window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache,
        sm_scale,
    )
    lse = lse_log2 * 0.6931471805599453
    return o_blhd, lse


def flex_attn_pope_tl_qk_prerotate(q, k, v, freqs, bias,
                                   window_size, chunk_size,
                                   training=True, mask_lmk=True, expand_to_chunk=True,
                                   sm_scale=None):
    """Tilelang flex attention with both Q and K PoPE pre-rotated once."""
    Hq, Hkv, R = q.shape[2], k.shape[2], freqs.shape[-1]
    G = Hq // Hkv
    bias = bias.float()
    freqs = freqs.float()
    if bias.dim() == 2:
        bias = bias.view(Hkv, G, R).contiguous()
    else:
        bias = bias.contiguous()
    freqs = freqs.contiguous()

    q_blhd = q.contiguous()
    k_blhd = k.contiguous()
    v_blhd = v.contiguous()

    use_cache = not training
    o_blhd, lse_log2 = _FlexAttnPopeQKPreRotTL.apply(
        q_blhd, k_blhd, v_blhd, freqs, bias,
        window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache,
        sm_scale,
    )
    lse = lse_log2 * 0.6931471805599453
    return o_blhd, lse


def flex_attn_pope_tl_torch_prerotate(q, k, v, freqs, bias,
                                      window_size, chunk_size,
                                      training=True, mask_lmk=True, expand_to_chunk=True,
                                      sm_scale=None):
    """Flex-attn PoPE with PyTorch-visible split prerotation.

    This is the preparation path for sharing q_cos/q_sin/k_cos/k_sin across
    flex/topk/hsa. The custom attention kernel returns split gradients, and
    PyTorch autograd handles raw q/k/bias gradients through the prerotate ops.
    """
    Hq, Hkv, R = q.shape[2], k.shape[2], freqs.shape[-1]
    G = Hq // Hkv
    bias = bias.float()
    freqs = freqs.float()
    if bias.dim() == 2:
        bias = bias.view(Hkv, G, R).contiguous()
    else:
        bias = bias.contiguous()
    freqs = freqs.contiguous()

    q_blhd = q.contiguous()
    k_blhd = k.contiguous()
    v_blhd = v.contiguous()
    q_cos, q_sin, k_cos, k_sin = _pope_prerotate_qk_split_torch(q_blhd, k_blhd, freqs, bias)

    use_cache = not training
    o_blhd, lse_log2 = _FlexAttnPopeTorchPreRotTL.apply(
        q_cos, q_sin, k_cos, k_sin, v_blhd,
        window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache,
        sm_scale,
    )
    lse = lse_log2 * 0.6931471805599453
    return o_blhd, lse


# ---------------------------------------------------------------------------
# Prerotate wrapper variant
#
# This variant materializes the PoPE rotation on Q/K in PyTorch, concatenates
# the rotated components along the last dim (so dim_qk grows from D to D+R),
# and then delegates the attention itself to the non-PoPE `flex_attn_tl`
# kernel with `sm_scale=1/sqrt(D)` explicitly passed through. Mathematically
# it is exactly equivalent to `flex_attn_pope_tl`, but the attention kernel
# stays untouched and the PoPE logic lives in autograd-visible PyTorch ops.
#
# Representation (rotate dims R, non-rotate dims D-R; last dim becomes D+R):
#   q_eff = cat([q[..., :R] * cos(theta_q),
#                q[..., :R] * sin(theta_q),
#                q[..., R:]], dim=-1)
#   k_eff = cat([k[..., :R] * cos(theta_k),
#                k[..., :R] * sin(theta_k),
#                k[..., R:]], dim=-1)
#   theta_q = freqs[pos_q] - bias     (MINUS sign, same as fused kernel)
#   theta_k = freqs[pos_k]
#
# The resulting <q_eff, k_eff> equals the fused kernel's
# Qc.Kc + Qs.Ks + <q_rest, k_rest> dot product, so passing sm_scale=1/sqrt(D)
# to the base kernel reproduces the fused kernel's softmax exactly. Without
# the explicit sm_scale override the base kernel would use 1/sqrt(D+R) (its
# current last-dim), which is a DIFFERENT attention op.
# ---------------------------------------------------------------------------
def _pope_prerotate_qk(q_blhd, k_blhd, freqs, bias_3d):
    """PyTorch-side PoPE expansion D -> D+R (concat layout: [cos | sin | rest]).

    Inputs:
        q_blhd:   (B, Lq, Hq, D)    bf16/fp16/fp32
        k_blhd:   (B, Lk, Hkv, D)
        freqs:    (L_total, R)      fp32, non-learnable
        bias_3d:  (Hkv, G, R)       fp32, learnable (Q-side, MINUS sign)

    Returns:
        q_eff:    (B, Lq, Hq, D+R)  same dtype as q
        k_eff:    (B, Lk, Hkv, D+R) same dtype as k

    The rotation is done in fp32 then cast back to the input dtype so we get
    the same numerical path as the fused kernel (cos/sin in fp32).
    """
    B, Lq, Hq, D = q_blhd.shape
    _, Lk, Hkv, _ = k_blhd.shape
    R = freqs.shape[-1]
    G = Hq // Hkv
    q_offset = Lk - Lq

    q_dtype = q_blhd.dtype
    k_dtype = k_blhd.dtype

    # theta_q: (Hq, Lq, R); theta_k: (Lk, R). Freqs/bias are fp32 by contract.
    bias_hq = bias_3d.reshape(Hq, R)                                  # (Hq, R)
    pos_q = torch.arange(Lq, device=q_blhd.device) + q_offset
    pos_k = torch.arange(Lk, device=k_blhd.device)
    fq = freqs.index_select(0, pos_q)                                 # (Lq, R)
    fk = freqs.index_select(0, pos_k)                                 # (Lk, R)
    theta_q = fq.unsqueeze(0) - bias_hq.unsqueeze(1)                  # (Hq, Lq, R)
    theta_k = fk                                                       # (Lk, R)

    cos_q = theta_q.cos()
    sin_q = theta_q.sin()
    cos_k = theta_k.cos()
    sin_k = theta_k.sin()

    # Q side: rotate first R dims, pass-through the rest.
    # q_blhd[..., :R]: (B, Lq, Hq, R) -> perm to (B, Hq, Lq, R) for broadcasting
    # with theta_q (Hq, Lq, R). Then perm back to (B, Lq, Hq, R).
    q_rot = q_blhd[..., :R].float()
    q_rest = q_blhd[..., R:]
    # broadcast: cos_q/sin_q shape (Hq, Lq, R) -> (1, Lq, Hq, R) via permute.
    cos_q_blhr = cos_q.permute(1, 0, 2).unsqueeze(0)                  # (1, Lq, Hq, R)
    sin_q_blhr = sin_q.permute(1, 0, 2).unsqueeze(0)                  # (1, Lq, Hq, R)
    qc = (q_rot * cos_q_blhr).to(q_dtype)
    qs = (q_rot * sin_q_blhr).to(q_dtype)
    q_eff = torch.cat([qc, qs, q_rest], dim=-1)                       # (B, Lq, Hq, D+R)

    # K side: rotate first R dims, pass-through the rest. theta_k depends only
    # on position, not on head.
    k_rot = k_blhd[..., :R].float()
    k_rest = k_blhd[..., R:]
    cos_k_blhr = cos_k.unsqueeze(0).unsqueeze(2)                      # (1, Lk, 1, R)
    sin_k_blhr = sin_k.unsqueeze(0).unsqueeze(2)                      # (1, Lk, 1, R)
    kc = (k_rot * cos_k_blhr).to(k_dtype)
    ks = (k_rot * sin_k_blhr).to(k_dtype)
    k_eff = torch.cat([kc, ks, k_rest], dim=-1)                       # (B, Lk, Hkv, D+R)

    return q_eff, k_eff


def flex_attn_pope_tl_prerotate(q, k, v, freqs, bias,
                                 window_size, chunk_size,
                                 training=True, mask_lmk=True, expand_to_chunk=True,
                                 sm_scale=None, two_phase_bwd=False):
    """Wrapper variant of `flex_attn_pope_tl` that reuses the non-PoPE
    `flex_attn_tl` kernel by prerotating Q/K in PyTorch.

    Numerically equivalent to `flex_attn_pope_tl` (same sign conventions, same
    default sm_scale = 1/sqrt(D)). Useful as a reference implementation whose
    backward is driven by autograd (no custom PoPE backward kernel).

    Args:
        q, k, v, freqs, bias, window_size, chunk_size, training, mask_lmk,
        expand_to_chunk, sm_scale: identical semantics to `flex_attn_pope_tl`.
        two_phase_bwd:  if True, delegate attention to `flex_attn_tl_two_phase`
                        (atomic-free two-phase backward). Defaults to False
                        (the atomic backward, matching the original
                        `flex_attn_tl` default).

    Returns:
        o:   (B, Lq, Hq, Dv)
        lse: (B, Lq, Hq) natural log
    """
    from ops.flex_attn_tilelang import flex_attn_tl, flex_attn_tl_two_phase

    Hq, Hkv, R = q.shape[2], k.shape[2], freqs.shape[-1]
    G = Hq // Hkv
    D = q.shape[-1]
    # Freqs/bias kept fp32 (same contract as flex_attn_pope_tl).
    bias = bias.float()
    freqs = freqs.float()
    if bias.dim() == 2:
        bias_3d = bias.view(Hkv, G, R)
    else:
        bias_3d = bias
    bias_3d = bias_3d.contiguous()
    freqs = freqs.contiguous()

    q_blhd = q.contiguous()
    k_blhd = k.contiguous()
    v_blhd = v.contiguous()

    # Logical softmax scale lives on D (NOT D+R), matching flex_attn_pope_tl.
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    q_eff, k_eff = _pope_prerotate_qk(q_blhd, k_blhd, freqs, bias_3d)

    # flex_attn_tl expects (B, H, L, D) layout. It also auto-contiguousifies
    # internally via its _to_blhd helper, so simply transposing back is fine.
    q_bhld = q_eff.transpose(1, 2)
    k_bhld = k_eff.transpose(1, 2)
    v_bhld = v_blhd.transpose(1, 2)

    attn_fn = flex_attn_tl_two_phase if two_phase_bwd else flex_attn_tl
    o_blhd, lse = attn_fn(
        q_bhld, k_bhld, v_bhld,
        window_size=window_size, chunk_size=chunk_size,
        training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        sm_scale=sm_scale,
    )
    # flex_attn_tl returns o: (B, Lq, Hq, Dv), lse: (B, Lq, Hq) already in the
    # public BLHD layout, so no further transpose is needed.
    return o_blhd, lse


# ---------------------------------------------------------------------------
# Reference implementation (pure torch, minus-bias convention)
# ---------------------------------------------------------------------------
def _torch_ref_pope(q, k, v, freqs, bias,
                    window_size, chunk_size, mask_lmk, expand_to_chunk,
                    sm_scale=None):
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    G = Hq // Hkv
    R = freqs.shape[-1]
    # Default sm_scale = 1/sqrt(D) (the original head_dim, NOT D+R), matching
    # flex_attn_pope_tl / HSA_block_M_head_pope* / online_softmax_topk_head_pope*.
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    device = q.device
    q_offset = Lk - Lq

    q_f = q.float()
    k_f = k.float()
    # bias: (Hkv, G, R)
    bias_hq = bias.view(Hq, R)  # (Hq, R)

    pos_q = torch.arange(Lq, device=device) + q_offset
    pos_k = torch.arange(Lk, device=device)
    fq = freqs[pos_q]  # (Lq, R)
    fk = freqs[pos_k]  # (Lk, R)

    # theta_q = fq - bias  (Hq, Lq, R)
    theta_q = fq.unsqueeze(0) - bias_hq.unsqueeze(1)
    theta_k = fk  # (Lk, R)

    q_rot = q_f[..., :R]  # (B, Hq, Lq, R)
    k_rot = k_f[..., :R]  # (B, Hkv, Lk, R)
    qc = q_rot * theta_q.cos().unsqueeze(0)
    qs = q_rot * theta_q.sin().unsqueeze(0)
    kc = k_rot * theta_k.cos().view(1, 1, Lk, R)
    ks = k_rot * theta_k.sin().view(1, 1, Lk, R)

    # concat [qc, qs, q_rest] / [kc, ks, k_rest]
    q_ext = torch.cat([qc, qs, q_f[..., R:]], dim=-1)  # (B, Hq, Lq, 2R+(D-R))
    k_ext = torch.cat([kc, ks, k_f[..., R:]], dim=-1)  # (B, Hkv, Lk, 2R+(D-R))

    # GQA expand
    k_exp = k_ext.repeat_interleave(G, dim=1)
    v_exp = v.float().repeat_interleave(G, dim=1)

    scores = torch.einsum("bhqd,bhkd->bhqk", q_ext, k_exp) * sm_scale

    # mask
    real_q = torch.arange(Lq, device=device) + q_offset
    kv_idx = torch.arange(Lk, device=device)
    if expand_to_chunk:
        cs = (real_q - window_size + 1).div(chunk_size, rounding_mode="floor") * chunk_size
    else:
        cs = real_q - window_size + 1
    keep = (kv_idx[None, :] >= cs[:, None]) & (kv_idx[None, :] <= real_q[:, None])
    if mask_lmk:
        keep = keep & ((kv_idx + 1) % chunk_size != 0).unsqueeze(0)
    mask_bias = torch.where(keep, 0.0, float("-inf")).float()
    scores = scores + mask_bias[None, None, :, :]

    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    o = torch.einsum("bhqk,bhkd->bhqd", p, v_exp).to(q.dtype)
    return o, lse


def flex_attn_pope_tl_ref(q, k, v, freqs, bias,
                          window_size, chunk_size,
                          training=True, mask_lmk=True, expand_to_chunk=True,
                          sm_scale=None):
    """Pure-PyTorch reference matching flex_attn_pope_tl's API.

    Same signature/semantics as flex_attn_pope_tl, but built entirely from
    PyTorch ops (no tilelang kernel). Useful for fp32 verification on configs
    where the tilelang fwd kernel hits dtype/layout-inference limits.

    Autograd is supported automatically by torch.

    Args:
        q: (B, Lq, Hq, D)  - same layout as flex_attn_pope_tl input
        k: (B, Lk, Hkv, D)
        v: (B, Lk, Hkv, Dv)
        freqs: (L_total, R)  fp32
        bias:  (Hq, R) or (Hkv, G, R)  fp32, theta_q = freqs - bias
        sm_scale: optional softmax scale; defaults to 1/sqrt(D).
    Returns:
        o:   (B, Lq, Hq, Dv)  same dtype as q
        lse: (B, Lq, Hq)      natural log
    """
    Hq, Hkv, R = q.shape[2], k.shape[2], freqs.shape[-1]
    G = Hq // Hkv
    if bias.dim() == 2:
        bias = bias.view(Hkv, G, R).contiguous()
    else:
        bias = bias.contiguous()
    freqs = freqs.contiguous()

    # Note: training flag has no effect for the ref impl (no kv-cache fast path).
    o, lse = _torch_ref_pope(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), freqs, bias,
        window_size=window_size, chunk_size=chunk_size,
        mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        sm_scale=sm_scale,
    )
    return o.transpose(1, 2).contiguous(), lse.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Consistency test
# ---------------------------------------------------------------------------
import pytest


def _make_pope(L_total, R, Hq, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    freqs = torch.randn(L_total, R, generator=g, device=device, dtype=torch.float32) * 0.5
    bias = torch.randn(Hq, R, generator=g, device=device, dtype=torch.float32) * 0.5
    return freqs, bias


_REGRESSION_CASES = [
    # seq_len / q_len not aligned to block_N / block_M / swizzled-dQ tile assumptions.
    ("train_seq130_misaligned",  dict(training=True, seq_len=130, window_size=64,  chunk_size=32)),
    ("train_seq200_misaligned",  dict(training=True, seq_len=200, window_size=64,  chunk_size=32)),
    ("train_seq250_misaligned",  dict(training=True, seq_len=250, window_size=128, chunk_size=64)),
    ("train_seq333_misaligned",  dict(training=True, seq_len=333, window_size=128, chunk_size=64)),
    ("train_seq513_misaligned",  dict(training=True, seq_len=513, window_size=128, chunk_size=64)),
    ("train_seq2049_lmk_no_exp", dict(training=True, seq_len=2049, window_size=512, chunk_size=64,
                                       mask_lmk=True, expand_to_chunk=False)),
    ("train_misaligned_gqa",     dict(training=True, seq_len=333, heads_q=16, heads_kv=4,
                                       window_size=128, chunk_size=64,
                                       mask_lmk=True, expand_to_chunk=False)),
    ("train_misaligned_rd64",    dict(training=True, seq_len=333, dim_qk=128, rotate_dim=64,
                                       window_size=128, chunk_size=64,
                                       mask_lmk=True, expand_to_chunk=False)),
    ("infer_q_misaligned_70",    dict(training=False, seq_len=512, q_len_inf=70,
                                       window_size=128, chunk_size=64)),
    ("infer_q_misaligned_130",   dict(training=False, seq_len=512, q_len_inf=130,
                                       window_size=128, chunk_size=64)),
    ("infer_kv_misaligned",      dict(training=False, seq_len=1023, q_len_inf=100,
                                       window_size=256, chunk_size=64)),
    ("infer_decode_q1_kv333",    dict(training=False, seq_len=333, q_len_inf=1,
                                       window_size=128, chunk_size=64)),
]


_TEST_CASES = _REGRESSION_CASES + [
    # ("train_lmk_expand",     dict(training=True,  mask_lmk=True,  expand_to_chunk=True,  seq_len=512, window_size=128, chunk_size=64)),
    # ("train_no_flags",       dict(training=True,  mask_lmk=False, expand_to_chunk=False, seq_len=512, window_size=128, chunk_size=64)),
    # ("train_lmk_only",       dict(training=True,  mask_lmk=True,  expand_to_chunk=False, seq_len=512, window_size=128, chunk_size=64)),
    # ("train_expand_only",    dict(training=True,  mask_lmk=False, expand_to_chunk=True,  seq_len=512, window_size=128, chunk_size=64)),
    # ("train_gqa_g1",         dict(training=True,  heads_q=4, heads_kv=4, seq_len=512)),
    # ("train_gqa_g4",         dict(training=True,  heads_q=8, heads_kv=2, seq_len=512)),
    # ("train_gqa_g8",         dict(training=True,  heads_q=8, heads_kv=1, seq_len=512)),
    # ("train_batch2",         dict(training=True,  batch=2, seq_len=512)),
    # ("train_short",          dict(training=True,  seq_len=256, window_size=64, chunk_size=32)),
    # ("train_long",           dict(training=True,  seq_len=1024, window_size=256, chunk_size=64)),
    # ("train_w_eq_cs",        dict(training=True,  seq_len=512, window_size=64, chunk_size=64)),
    # ("train_w_gt_cs",        dict(training=True,  seq_len=512, window_size=256, chunk_size=64)),
    # ("train_w_lt_cs",        dict(training=True,  seq_len=512, window_size=32, chunk_size=64, expand_to_chunk=True, mask_lmk=True)),
    # ("train_rd_full",        dict(training=True,  seq_len=512, dim_qk=128, rotate_dim=128)),
    # ("train_rd_half",        dict(training=True,  seq_len=512, dim_qk=128, rotate_dim=64)),
    # ("train_rd_quarter",     dict(training=True,  seq_len=512, dim_qk=128, rotate_dim=32)),
    # ("train_seq384",         dict(training=True,  seq_len=384, window_size=128, chunk_size=64)),
    # ("infer_prefill_half",   dict(training=False, seq_len=512,  q_len_inf=256)),
    # ("infer_decode_q1",      dict(training=False, seq_len=512,  q_len_inf=1)),
    # ("infer_decode_long",    dict(training=False, seq_len=2048, q_len_inf=1, window_size=512, chunk_size=128)),
    # ("infer_gqa_q1",         dict(training=False, seq_len=512,  q_len_inf=1, heads_q=8, heads_kv=2)),
    # ("infer_no_flags",       dict(training=False, seq_len=512,  q_len_inf=128, mask_lmk=False, expand_to_chunk=False)),
    # ("infer_rd_half_q1",     dict(training=False, seq_len=512,  q_len_inf=1, dim_qk=128, rotate_dim=64)),
]


@pytest.mark.parametrize("test_name, cfg", _TEST_CASES, ids=[c[0] for c in _TEST_CASES])
@pytest.mark.parametrize("impl", ["fused", "fused_k_prerotate", "fused_qk_prerotate", "torch_prerotate", "prerotate"])
def test_flex_attn_pope_tl(test_name, cfg, impl):
    _run_case(test_name, impl=impl, **cfg)


def _run_case(
    test_name="default", batch=1, heads_q=8, heads_kv=2, seq_len=512,
    dim_qk=128, dim_v=128, rotate_dim=None,
    window_size=128, chunk_size=64,
    mask_lmk=True, expand_to_chunk=True,
    training=True, q_len_inf=None, ratio=1e-2, seed=0,
    impl="fused",
):
    """Consistency test runner.

    `impl` selects which implementation under test to compare against the
    pure-PyTorch reference:
      - "fused"               -> flex_attn_pope_tl
      - "fused_k_prerotate"   -> flex_attn_pope_tl_k_prerotate
      - "fused_qk_prerotate"  -> flex_attn_pope_tl_qk_prerotate
      - "torch_prerotate"     -> flex_attn_pope_tl_torch_prerotate
      - "prerotate"           -> flex_attn_pope_tl_prerotate (atomic bwd)
    """
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    if rotate_dim is None:
        rotate_dim = dim_qk

    L_k = seq_len
    L_q = seq_len if training else (q_len_inf if q_len_inf is not None else seq_len // 2)

    print(f"\n{'=' * 70}")
    print(f"Test: {test_name} [impl={impl}]")
    print(f"Config: B={batch}, Hq={heads_q}, Hkv={heads_kv}, Lq={L_q}, Lk={L_k}, "
          f"D_qk={dim_qk}, D_v={dim_v}, R={rotate_dim}")
    print(f"        window={window_size}, chunk={chunk_size}, "
          f"mask_lmk={mask_lmk}, expand_to_chunk={expand_to_chunk}, training={training}")
    print(f"{'=' * 70}")

    q = F.softplus(torch.randn(batch, L_q, heads_q, dim_qk, device=device, dtype=dtype) * 0.3)
    k = F.softplus(torch.randn(batch, L_k, heads_kv, dim_qk, device=device, dtype=dtype) * 0.3)
    v = torch.randn(batch, L_k, heads_kv, dim_v, device=device, dtype=dtype) * 0.3
    freqs, bias_raw = _make_pope(L_total=L_k + 16, R=rotate_dim, Hq=heads_q, device=device, seed=seed)

    if training:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        bias_raw.requires_grad_(True)

    # Inference path is not supported by the prerotate wrappers' backward, but
    # the forward IS supported; this matches flex_attn_tl's contract.
    if impl == "fused":
        o_tl, lse_tl = flex_attn_pope_tl(
            q, k, v, freqs, bias_raw,
            window_size=window_size, chunk_size=chunk_size,
            training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        )
    elif impl == "fused_k_prerotate":
        o_tl, lse_tl = flex_attn_pope_tl_k_prerotate(
            q, k, v, freqs, bias_raw,
            window_size=window_size, chunk_size=chunk_size,
            training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        )
    elif impl == "fused_qk_prerotate":
        o_tl, lse_tl = flex_attn_pope_tl_qk_prerotate(
            q, k, v, freqs, bias_raw,
            window_size=window_size, chunk_size=chunk_size,
            training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        )
    elif impl == "torch_prerotate":
        o_tl, lse_tl = flex_attn_pope_tl_torch_prerotate(
            q, k, v, freqs, bias_raw,
            window_size=window_size, chunk_size=chunk_size,
            training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        )
    elif impl == "prerotate":
        o_tl, lse_tl = flex_attn_pope_tl_prerotate(
            q, k, v, freqs, bias_raw,
            window_size=window_size, chunk_size=chunk_size,
            training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
            two_phase_bwd=False,
        )
    else:
        raise ValueError(f"Unknown impl: {impl}")

    if training:
        do = torch.randn_like(o_tl)
        o_tl.backward(do, retain_graph=False)
        dq_tl, q.grad = q.grad.clone(), None
        dk_tl, k.grad = k.grad.clone(), None
        dv_tl, v.grad = v.grad.clone(), None
        dbias_tl, bias_raw.grad = bias_raw.grad.clone(), None

    # ref
    q_ref = q.detach().clone().requires_grad_(training)
    k_ref = k.detach().clone().requires_grad_(training)
    v_ref = v.detach().clone().requires_grad_(training)
    bias_ref = bias_raw.detach().clone().requires_grad_(training)

    Hkv = heads_kv
    G = heads_q // heads_kv
    bias_3d = bias_ref.view(Hkv, G, rotate_dim)

    o_ref, lse_ref = _torch_ref_pope(
        q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2), freqs, bias_3d,
        window_size=window_size, chunk_size=chunk_size,
        mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
    )
    o_ref_blhd = o_ref.transpose(1, 2).contiguous()
    lse_ref_blh = lse_ref.transpose(1, 2).contiguous()

    if training:
        with torch.no_grad():
            invalid_lse_blh = ~torch.isfinite(lse_ref_blh)
            do_masked = do.clone()
            do_masked[invalid_lse_blh] = 0.0
        do_ref = do_masked.transpose(1, 2).contiguous()
        o_ref.backward(do_ref, retain_graph=False)
        dq_ref = q_ref.grad
        dk_ref = k_ref.grad
        dv_ref = v_ref.grad
        dbias_ref = bias_ref.grad

    def _get_abs_err(x, y):
        m = (x > -1e5) & (y > -1e5)
        if m.sum() == 0:
            return 0.0
        return (x[m] - y[m]).abs().max().item()

    def _get_err_ratio(x, y):
        m = (x > -1e5) & (y > -1e5)
        if m.sum() == 0:
            return 0.0
        err = (x[m] - y[m]).square().mean().sqrt().item()
        base = (x[m]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    def _assert_close(prefix, ref, tri, ratio_thr=ratio):
        ref_f = torch.nan_to_num(ref.detach().float(), 0.0, 0.0, 0.0)
        tri_f = torch.nan_to_num(tri.detach().float(), 0.0, 0.0, 0.0)
        abs_err = _get_abs_err(ref_f, tri_f)
        rel_ratio = _get_err_ratio(ref_f, tri_f)
        msg = f"  {prefix:<6s} abs_diff={abs_err:.6f}  rel_ratio={rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio_thr, f"[{test_name}] {prefix} failed: {msg}"

    valid_rows = torch.isfinite(lse_ref_blh).sum().item()
    print(f"  valid_rows={valid_rows}/{lse_ref_blh.numel()}")
    _assert_close("o",   o_ref_blhd, o_tl)
    _assert_close("lse", lse_ref_blh, lse_tl)

    if training:
        _assert_close("dv",    dv_ref,    dv_tl)
        _assert_close("dk",    dk_ref,    dk_tl)
        # dq accumulates via atomic_add across many KV-blocks and recomputes
        # cos/sin in bf16, so its precision is inherently lower.
        _assert_close("dq",    dq_ref,    dq_tl,    ratio_thr=ratio * 3)
        _assert_close("dbias", dbias_ref, dbias_tl, ratio_thr=ratio * 4)

    print(f"[PASS] {test_name}")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def _bench_one(fn, n_warmup=5, n_iter=20):
    """Run `fn` n_warmup + n_iter times; return median ms over n_iter runs."""
    import time
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times_ms[len(times_ms) // 2]


def bench_flex_attn_pope_tl(
    batch=4, heads_q=8, heads_kv=2, seq_len=4096,
    dim_qk=128, dim_v=128, rotate_dim=64,
    window_size=512, chunk_size=64,
    mask_lmk=True, expand_to_chunk=True,
    training=True, q_len_inf=None, seed=0,
    n_warmup=5, n_iter=20,
    impls=("fused", "fused_k_prerotate", "fused_qk_prerotate", "torch_prerotate", "prerotate"),
):
    """Compare fwd / fwd+bwd latency of fused variants and prerotate wrappers.

    Reports median latency in milliseconds over `n_iter` runs (after `n_warmup`
    warmup runs). When `training=True` also benchmarks the full fwd+bwd pass.
    """
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    L_k = seq_len
    L_q = seq_len if training else (q_len_inf if q_len_inf is not None else seq_len // 2)

    print(f"\n{'=' * 72}")
    print(f"Bench: B={batch}, Hq={heads_q}, Hkv={heads_kv}, Lq={L_q}, Lk={L_k}, "
          f"D={dim_qk}, Dv={dim_v}, R={rotate_dim}")
    print(f"       window={window_size}, chunk={chunk_size}, "
          f"mask_lmk={mask_lmk}, expand_to_chunk={expand_to_chunk}, training={training}")
    print(f"{'=' * 72}")

    def _make_inputs():
        q = F.softplus(torch.randn(batch, L_q, heads_q, dim_qk, device=device, dtype=dtype) * 0.3)
        k = F.softplus(torch.randn(batch, L_k, heads_kv, dim_qk, device=device, dtype=dtype) * 0.3)
        v = torch.randn(batch, L_k, heads_kv, dim_v, device=device, dtype=dtype) * 0.3
        freqs, bias = _make_pope(L_total=L_k + 16, R=rotate_dim, Hq=heads_q, device=device, seed=seed)
        return q, k, v, freqs, bias

    def _call(impl_name, q, k, v, freqs, bias):
        if impl_name == "fused":
            return flex_attn_pope_tl(
                q, k, v, freqs, bias,
                window_size=window_size, chunk_size=chunk_size,
                training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
            )
        elif impl_name == "fused_k_prerotate":
            return flex_attn_pope_tl_k_prerotate(
                q, k, v, freqs, bias,
                window_size=window_size, chunk_size=chunk_size,
                training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
            )
        elif impl_name == "fused_qk_prerotate":
            return flex_attn_pope_tl_qk_prerotate(
                q, k, v, freqs, bias,
                window_size=window_size, chunk_size=chunk_size,
                training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
            )
        elif impl_name == "torch_prerotate":
            return flex_attn_pope_tl_torch_prerotate(
                q, k, v, freqs, bias,
                window_size=window_size, chunk_size=chunk_size,
                training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
            )
        elif impl_name == "prerotate":
            return flex_attn_pope_tl_prerotate(
                q, k, v, freqs, bias,
                window_size=window_size, chunk_size=chunk_size,
                training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
                two_phase_bwd=False,
            )
        raise ValueError(f"Unknown impl: {impl_name}")

    results = {}

    # --- Forward-only benchmarks (no autograd graph). ---
    for impl_name in impls:
        q, k, v, freqs, bias = _make_inputs()
        with torch.no_grad():
            ms_fwd = _bench_one(
                lambda: _call(impl_name, q, k, v, freqs, bias),
                n_warmup=n_warmup, n_iter=n_iter,
            )
        results.setdefault(impl_name, {})["fwd_ms"] = ms_fwd

    # --- Forward + backward benchmarks (training only). ---
    if training:
        for impl_name in impls:
            q, k, v, freqs, bias = _make_inputs()
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            bias.requires_grad_(True)
            do = torch.randn(batch, L_q, heads_q, dim_v, device=device, dtype=dtype)

            def _fwdbwd():
                # Zero grads BEFORE the timed step so the timing measures one
                # complete fwd+bwd. zero_grad uses a fresh tensor each iter to
                # avoid mutating the leaves' .grad in-place between iters.
                if q.grad is not None: q.grad = None
                if k.grad is not None: k.grad = None
                if v.grad is not None: v.grad = None
                if bias.grad is not None: bias.grad = None
                o, _ = _call(impl_name, q, k, v, freqs, bias)
                o.backward(do)

            ms_fwdbwd = _bench_one(_fwdbwd, n_warmup=n_warmup, n_iter=n_iter)
            results[impl_name]["fwdbwd_ms"] = ms_fwdbwd

    # --- Pretty print. ---
    header = f"{'impl':<22s}  {'fwd ms':>10s}"
    if training:
        header += f"  {'fwd+bwd ms':>14s}"
    print(header)
    print("-" * len(header))
    fused_fwd = results.get("fused", {}).get("fwd_ms")
    fused_fb = results.get("fused", {}).get("fwdbwd_ms")
    for impl_name in impls:
        r = results[impl_name]
        line = f"{impl_name:<22s}  {r['fwd_ms']:>10.3f}"
        if fused_fwd is not None and impl_name != "fused":
            line += f"  ({r['fwd_ms']/fused_fwd:.2f}x fused)"
        if training:
            line += f"  {r['fwdbwd_ms']:>14.3f}"
            if fused_fb is not None and impl_name != "fused":
                line += f"  ({r['fwdbwd_ms']/fused_fb:.2f}x fused)"
        print(line)

    return results


_BENCH_PRESETS = [
    # (label, kwargs)
    ("L4k_W512_R64",
     dict(seq_len=8192, window_size=512, chunk_size=64, rotate_dim=32,
          dim_qk=64, dim_v=64, heads_q=16, heads_kv=2)),
    # ("L8k_W1024_R64",
    #  dict(seq_len=8192, window_size=1024, chunk_size=128, rotate_dim=64,
    #       dim_qk=128, dim_v=128, heads_q=8, heads_kv=2)),
    # ("L16k_W1024_R64",
    #  dict(seq_len=16384, window_size=1024, chunk_size=128, rotate_dim=64,
    #       dim_qk=128, dim_v=128, heads_q=8, heads_kv=2)),
    # ("L4k_W512_Rfull",
    #  dict(seq_len=4096, window_size=512, chunk_size=64, rotate_dim=128,
    #       dim_qk=128, dim_v=128, heads_q=8, heads_kv=2)),
    # ("L4k_W512_R64_inference_q1",
    #  dict(seq_len=4096, window_size=512, chunk_size=64, rotate_dim=64,
    #       dim_qk=128, dim_v=128, heads_q=8, heads_kv=2,
    #       training=False, q_len_inf=1)),
]


def _run_benchmarks(argv):
    """CLI entry: `python flex_attn_pope_tilelang.py bench [preset_name ...]`.

    With no preset names, runs every entry in `_BENCH_PRESETS`. With one or
    more names, runs only those.
    """
    selected = argv if argv else [name for name, _ in _BENCH_PRESETS]
    name_to_cfg = dict(_BENCH_PRESETS)
    for name in selected:
        if name not in name_to_cfg:
            print(f"[skip] unknown preset: {name}")
            continue
        bench_flex_attn_pope_tl(**name_to_cfg[name])


if __name__ == "__main__":
    import sys

    # CLI:  python flex_attn_pope_tilelang.py [bench|test] [extra args]
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "bench":
        _run_benchmarks(sys.argv[2:])
        sys.exit(0)

    failures = []
    impls = ["fused", "fused_k_prerotate", "fused_qk_prerotate", "torch_prerotate", "prerotate"]
    total = 0
    for impl in impls:
        for name, cfg in _TEST_CASES:
            total += 1
            label = f"{name}[{impl}]"
            try:
                _run_case(name, impl=impl, **cfg)
            except AssertionError as e:
                failures.append((label, str(e)))
                print(f"[FAIL] {label}: {e}\n")
            except Exception as e:
                failures.append((label, f"{type(e).__name__}: {e}"))
                print(f"[ERROR] {label}: {type(e).__name__}: {e}\n")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Summary: {total - len(failures)}/{total} passed")
    if failures:
        print("Failures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        raise SystemExit(1)
    print("All tests passed")



# python ops/flex_attn_pope_tilelang.py