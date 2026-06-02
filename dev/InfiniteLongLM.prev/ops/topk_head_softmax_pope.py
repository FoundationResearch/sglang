import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from typing import Optional
import math

try:
    from .topk_head_softmax import online_softmax_topk_head as _base_online_softmax_topk_head
except ImportError:
    from topk_head_softmax import online_softmax_topk_head as _base_online_softmax_topk_head


_TORCH_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}

def _torch_dtype_to_str(td: torch.dtype) -> str:
    if td not in _TORCH_DTYPE_TO_STR:
        raise ValueError(f"Unsupported torch dtype for tilelang kernel: {td}")
    return _TORCH_DTYPE_TO_STR[td]


def _pope_qk_logits(q, k_lmks, freqs, bias, block_size, q_offset, rotate_dim=None):
    """
    Args:
        q:      [B, L, h_kv, G, D]  (float32)
        k_lmks: [B, S, h_kv, D]     (float32)
        freqs:  [L_total, rotate_dim] (float32)
        bias:   [h_q, rotate_dim] or [h_kv, G, rotate_dim]   (float32)   — applied on Q side with minus sign
        block_size: int
        q_offset:   int
        rotate_dim: int (default=D, i.e. full rotation)
    Returns:
        logits: [B, L, h_kv, G, S]  (float32)
    """
    B, L, h_kv, G, D = q.shape
    S = k_lmks.shape[1]
    device = q.device
    if rotate_dim is None:
        rotate_dim = freqs.shape[-1]  
    R = rotate_dim

    # Auto-reshape bias: [h_q, R] -> [h_kv, G, R]
    if bias.dim() == 2:
        bias = bias.view(h_kv, G, R)

    q_pos = torch.arange(L, device=device) + q_offset     # [L]
    fq = freqs.index_select(0, q_pos)                     # [L, R]

    k_pos = torch.arange(S, device=device) * block_size + (block_size - 1)   # [S]
    fk = freqs.index_select(0, k_pos)                     # [S, R]

    # bias applied on Q side: theta_q = fq - bias[h_kv, g]  (minus sign for equivalence)
    # bias: [h_kv, G, R] -> theta_q: [L, h_kv, G, R]
    theta_q = fq.view(L, 1, 1, R) - bias.unsqueeze(0)    # [L, h_kv, G, R]

    # K side: pure positional frequency, no bias
    theta_k = fk                                           # [S, R]

    cos_tq = torch.cos(theta_q).unsqueeze(0)               # [1, L, h_kv, G, R]
    sin_tq = torch.sin(theta_q).unsqueeze(0)               # [1, L, h_kv, G, R]
    cos_tk = torch.cos(theta_k).view(1, S, 1, R)           # [1, S, h_kv=1, R]
    sin_tk = torch.sin(theta_k).view(1, S, 1, R)           # [1, S, h_kv=1, R]

    q_rot = q[..., :R]                                     # [B,L,h_kv,G,R]
    k_rot = k_lmks[..., :R]                                # [B,S,h_kv,R]
    aq = q_rot                                                 # [B,L,h_kv,G,R]
    bk = k_rot                                                 # [B,S,h_kv,R]

    qc = aq * cos_tq                                       # [B,L,h_kv,G,R]
    qs = aq * sin_tq
    kc = bk * cos_tk                                       # [B,S,h_kv,R]
    ks = bk * sin_tk

    logits = (
        torch.einsum("blhgd,bshd->blhgs", qc, kc)
      + torch.einsum("blhgd,bshd->blhgs", qs, ks)
    )

    if R < D:
        q_rest = q[..., R:]                                # [B,L,h_kv,G,D-R]
        k_rest = k_lmks[..., R:]                           # [B,S,h_kv,D-R]
        logits = logits + torch.einsum("blhgd,bshd->blhgs", q_rest, k_rest)

    return logits


def ref_softmax_topk_max_pooling(
    q, k_lmks, lse_swa, freqs, bias,
    topk, block_size, window_size,
    is_causal=False, q_offset=0, drop_mask=None,
):
    """
    Reference implementation for Softmax-then-Max Top-K strategy with PoPE similarity.

    Args:
        q: [B, L, h_kv, G, D]
        k_lmks: [B, S, h_kv, D]
        lse_swa: [B, L, h_q] or [B, L, h_kv, G]
        freqs: [L_total, D]      PoPE position-angle table, shared across batch/head
        bias:  [h_kv, G, D]     per-query-head additive bias on Q-side angle (learnable)
        topk: int
        block_size: int
        window_size: int
        is_causal: bool
        q_offset: int (for causal masking when q does not start from 0)
        drop_mask: [B, L, S] int32 tensor, 1 表示 drop 该 chunk

    Returns:
        indices_sorted: [B, L, h_kv, topk]
        scores_sorted: [B, L, h_kv, G, topk] (Raw Logits)
    """
    B, L, h_kv, G, D = q.shape
    S = k_lmks.shape[1]

    logits_hsa = _pope_qk_logits(
        q.float(), k_lmks.float(), freqs.float(), bias.float(),
        block_size=block_size, q_offset=q_offset,
    )
    
    sm_scale = 1.0 / math.sqrt(D)
    logits_hsa_scaled = logits_hsa * sm_scale

    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1)
        i_idx_global = i_idx + q_offset  
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)
        
        threshold_idx = (i_idx_global - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= threshold_idx 
        
        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        logits_hsa_scaled = logits_hsa_scaled.masked_fill(causal_mask_expanded, float('-inf'))

    lse_hsa = torch.logsumexp(logits_hsa_scaled, dim=-1)
    
    if lse_swa.dim() == 3:
        lse_swa_view = lse_swa.view(B, L, h_kv, G)
    else:
        lse_swa_view = lse_swa
        
    lse_total = torch.logaddexp(lse_swa_view, lse_hsa)
    
    log_probs = logits_hsa_scaled - lse_total.unsqueeze(-1)
    
    scores_max_pooling = log_probs.max(dim=3).values
    
    if drop_mask is not None:
        drop_bool = drop_mask.bool()  # [B, L, S]
        scores_max_pooling = scores_max_pooling.masked_fill(drop_bool.unsqueeze(2), float('-inf'))
    
    actual_topk = min(topk, S)
    
    topk_scores, topk_indices = torch.topk(scores_max_pooling, k=actual_topk, dim=-1, sorted=False)
    
    topk_indices[topk_scores == float('-inf')] = -1
    
    if actual_topk < topk:
        pad_size = topk - actual_topk
        pad_indices = torch.full(
            (B, L, h_kv, pad_size), -1, 
            dtype=topk_indices.dtype, device=topk_indices.device
        )
        topk_indices = torch.cat([topk_indices, pad_indices], dim=-1)
    
    indices_sorted, order = torch.sort(topk_indices, dim=-1)
    
    sort_temp = topk_indices.clone()
    sort_temp[sort_temp < 0] = S + 1000  
    indices_sorted, order = torch.sort(sort_temp, dim=-1)
    indices_sorted[indices_sorted >= S] = -1  
    
    order_expanded = order.unsqueeze(3).expand(-1, -1, -1, G, -1)
    
    safe_indices_sorted = indices_sorted.clone()
    safe_indices_sorted[safe_indices_sorted < 0] = 0
    indices_expanded = safe_indices_sorted.unsqueeze(3).expand(-1, -1, -1, G, -1)
    
    scores_sorted = torch.gather(logits_hsa_scaled, -1, indices_expanded)
    
    invalid_mask = indices_sorted.unsqueeze(3).expand(-1, -1, -1, G, -1) < 0
    scores_sorted = scores_sorted.masked_fill(invalid_mask, float('-inf'))
    
    return indices_sorted, scores_sorted




@tilelang.jit(
    out_idx=[5, 6, 7, 8],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def pope_pre_rotate_qk_kernel(
    batch, h_kv, groups, head_dim, block_size,
    is_training=True, seq_len=None, s_len=None, freqs_len=None, rotate_dim=None,
    BLOCK_L=None, BLOCK_S=None, threads=None,
    dtype="bfloat16", freqs_dtype="bfloat16", bias_dtype="bfloat16", accum_dtype="float",
):
    if not is_training:
        seq_len_var = T.dynamic("seq_len")
        s_len_var = T.dynamic("s_len")
        freqs_len_var = T.dynamic("freqs_len") if freqs_len is None else freqs_len
    else:
        seq_len_var = seq_len
        s_len_var = s_len
        freqs_len_var = freqs_len if freqs_len is not None else seq_len + (s_len - 1) * block_size + 1

    if rotate_dim is None:
        rotate_dim = head_dim
    if BLOCK_L is None:
        BLOCK_L = (16 + groups - 1) // groups
    if BLOCK_S is None:
        BLOCK_S = 64
    if threads is None:
        threads = 128

    q_shape = [batch, seq_len_var, h_kv, groups, head_dim]
    k_shape = [batch, s_len_var, h_kv, head_dim]
    q_sin_shape = [batch, seq_len_var, h_kv, groups, rotate_dim]
    k_sin_shape = [batch, s_len_var, h_kv, rotate_dim]
    freqs_shape = [freqs_len_var, rotate_dim]
    bias_shape = [h_kv, groups, rotate_dim]
    q_blocks = tilelang.cdiv(seq_len_var, BLOCK_L)
    k_blocks = tilelang.cdiv(s_len_var, BLOCK_S)

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        Freqs: T.Tensor(freqs_shape, freqs_dtype),
        Bias: T.Tensor(bias_shape, bias_dtype),
        Q_Offset: T.Tensor([1], "int32"),
        Qc: T.Tensor(q_shape, dtype),
        Qs: T.Tensor(q_sin_shape, dtype),
        Kc: T.Tensor(k_shape, dtype),
        Ks: T.Tensor(k_sin_shape, dtype),
    ):
        with T.Kernel(q_blocks + k_blocks, h_kv, batch, threads=threads) as (bx, by, bz):
            i_b, i_h = bz, by
            if bx < q_blocks:
                q_offset = T.if_then_else(is_training, 0, Q_Offset[0])
                base_l = bx * BLOCK_L
                for l_idx, g, d in T.Parallel(BLOCK_L, groups, head_dim):
                    tq = base_l + l_idx
                    if tq < seq_len_var:
                        Qc[i_b, tq, i_h, g, d] = Q[i_b, tq, i_h, g, d]

                for l_idx, g, d in T.Parallel(BLOCK_L, groups, rotate_dim):
                    tq = base_l + l_idx
                    if tq < seq_len_var:
                        qx = T.Cast(accum_dtype, Q[i_b, tq, i_h, g, d])
                        theta_q = Freqs[q_offset + tq, d] - Bias[i_h, g, d]
                        Qc[i_b, tq, i_h, g, d] = T.Cast(dtype, qx * T.cos(theta_q))
                        Qs[i_b, tq, i_h, g, d] = T.Cast(dtype, qx * T.sin(theta_q))
            else:
                k_bx = bx - q_blocks
                base_s = k_bx * BLOCK_S
                for s_idx, d in T.Parallel(BLOCK_S, head_dim):
                    ts = base_s + s_idx
                    if ts < s_len_var:
                        Kc[i_b, ts, i_h, d] = K[i_b, ts, i_h, d]

                for s_idx, d in T.Parallel(BLOCK_S, rotate_dim):
                    ts = base_s + s_idx
                    if ts < s_len_var:
                        kx = T.Cast(accum_dtype, K[i_b, ts, i_h, d])
                        theta_k = Freqs[ts * block_size + (block_size - 1), d]
                        Kc[i_b, ts, i_h, d] = T.Cast(dtype, kx * T.cos(theta_k))
                        Ks[i_b, ts, i_h, d] = T.Cast(dtype, kx * T.sin(theta_k))

    return kernel


# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [2,4,8,16]
# BLOCK_S = [16,32,64]
# threads = [64,128,256]
# _configs = list(
#     itertools.product(
#         BLOCK_L,
#         BLOCK_S,
#         threads,
#     ))

# configs = [
#     {
#         "BLOCK_L": c[0],
#         "BLOCK_S": c[1],
#         "threads": c[2],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hsa_lse_kernel(
    batch, h_kv, groups, head_dim, block_size, window_size,
    is_causal, is_training=True, seq_len=None, s_len=None,
    freqs_len=None, rotate_dim=None, sm_scale=None,
    BLOCK_L=None, BLOCK_S=None, threads=None,
    dtype="bfloat16", accum_dtype="float",
):
    if not is_training:
        seq_len_var = T.dynamic("seq_len")
        s_len_var = T.dynamic("s_len")
        freqs_len_var = T.dynamic("freqs_len") if freqs_len is None else freqs_len
    else:
        seq_len_var = seq_len
        s_len_var = s_len
        freqs_len_var = freqs_len if freqs_len is not None else seq_len + (s_len - 1) * block_size + 1

    # Q: [B, L, h_kv, G, D]
    q_shape = [batch, seq_len_var, h_kv, groups, head_dim]
    # K: [B, S, h_kv, D]
    k_shape = [batch, s_len_var, h_kv, head_dim]
    if rotate_dim is None:
        rotate_dim = head_dim
    q_sin_shape = [batch, seq_len_var, h_kv, groups, rotate_dim]
    k_sin_shape = [batch, s_len_var, h_kv, rotate_dim]
    # LSE Out: [B, L, h_kv, G]
    lse_shape = [batch, seq_len_var, h_kv, groups]

    if BLOCK_L is None: BLOCK_L = (16 + groups - 1) // groups
    if BLOCK_S is None: BLOCK_S = 64
    if threads is None: threads = 128
    

    GEMM_M = BLOCK_L * groups
    GEMM_N = BLOCK_S
    GEMM_K = head_dim

    num_s_blocks = tilelang.cdiv(s_len_var, BLOCK_S)
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def kernel(
        Qc: T.Tensor(q_shape, dtype),
        Kc: T.Tensor(k_shape, dtype),
        LSE_Out: T.Tensor(lse_shape, accum_dtype),
        Qs: T.Tensor(q_sin_shape, dtype),
        Ks: T.Tensor(k_sin_shape, dtype),
        Q_Offset: T.Tensor([1], "int32"),
    ):
        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            q_offset = T.if_then_else(is_training, 0, Q_Offset[0])
            i_b, i_h = bz, by
            base_l = bx * BLOCK_L
            
            Qc_shared = T.alloc_shared([GEMM_M, GEMM_K], dtype)
            Qs_shared = T.alloc_shared([GEMM_M, rotate_dim], dtype)
            Kc_shared = T.alloc_shared([GEMM_N, GEMM_K], dtype)
            Ks_shared = T.alloc_shared([GEMM_N, rotate_dim], dtype)
            
            score_shared = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)
            
            acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)
            
            m_curr = T.alloc_fragment([GEMM_M], accum_dtype)
            m_prev = T.alloc_fragment([GEMM_M], accum_dtype)
            l_prev = T.alloc_fragment([GEMM_M], accum_dtype)
            
            scores_max = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_sum = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_scale = T.alloc_fragment([GEMM_M], accum_dtype)

            T.annotate_layout({Qc_shared: tilelang.layout.make_swizzled_layout(Qc_shared)})
            T.annotate_layout({Qs_shared: tilelang.layout.make_swizzled_layout(Qs_shared)})

            T.fill(m_prev, -T.infinity(accum_dtype))
            T.fill(l_prev, 0.0)

            T.fill(Qc_shared, 0)
            T.fill(Qs_shared, 0)
            for i, j in T.Parallel(GEMM_M, GEMM_K):
                l_idx = i // groups
                g = i % groups
                tq = base_l + l_idx
                if tq < seq_len_var:
                    Qc_shared[i, j] = Qc[i_b, tq, i_h, g, j]

            for i, j in T.Parallel(GEMM_M, rotate_dim):
                l_idx = i // groups
                g = i % groups
                tq = base_l + l_idx
                if tq < seq_len_var:
                    Qs_shared[i, j] = Qs[i_b, tq, i_h, g, j]

            loop_limit_base = tilelang.cdiv(s_len_var, BLOCK_S)
            if is_causal:
                global_end = q_offset + base_l + BLOCK_L
                loop_limit = T.min(loop_limit_base, tilelang.cdiv(global_end, BLOCK_S))

            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S
                
                T.fill(Kc_shared, 0)
                T.fill(Ks_shared, 0)
                for i, j in T.Parallel(GEMM_N, GEMM_K):
                    ts = base_s + i
                    if ts < s_len_var:
                        Kc_shared[i, j] = Kc[i_b, ts, i_h, j]

                for i, j in T.Parallel(GEMM_N, rotate_dim):
                    ts = base_s + i
                    if ts < s_len_var:
                        Ks_shared[i, j] = Ks[i_b, ts, i_h, j]
                
                T.sync_threads()
                
                T.clear(acc_s)
                T.gemm(Qc_shared, Kc_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Qs_shared, Ks_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                T.copy(acc_s, score_shared)
                
                for i, j in T.Parallel(GEMM_M, GEMM_N):
                    ts = base_s + j
                    if ts >= s_len_var:
                        score_shared[i, j] = -T.infinity(accum_dtype)
                    elif is_causal:
                        l_idx = i // groups
                        tq_local = base_l + l_idx
                        tq_global = q_offset + tq_local
                        if tq_local < seq_len_var:
                            if ts >= (tq_global - window_size + 1) // block_size:
                                score_shared[i, j] = -T.infinity(accum_dtype)

                T.sync_threads()
                T.copy(score_shared, acc_s)
                
                T.copy(m_prev, m_curr)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                
                for i in T.Parallel(GEMM_M):
                    scores_max[i] = scores_max[i] * sm_scale
                    m_prev[i] = T.max(m_prev[i], scores_max[i])
                
                for i in T.Parallel(GEMM_M):
                    if m_prev[i] == -T.infinity(accum_dtype):
                        scores_scale[i] = 1.0
                    else:
                        scores_scale[i] = T.exp(m_curr[i] - m_prev[i])
                
                for i, j in T.Parallel(GEMM_M, GEMM_N):
                    ts = base_s + j
                    if ts < s_len_var:
                        val = acc_s[i, j] * sm_scale
                        if val == -T.infinity(accum_dtype) and m_prev[i] == -T.infinity(accum_dtype):
                            acc_s[i, j] = 0.0
                        else:
                            acc_s[i, j] = T.exp(val - m_prev[i])
                    else:
                        acc_s[i, j] = 0.0
                
                T.reduce_sum(acc_s, scores_sum, dim=1)
                
                for i in T.Parallel(GEMM_M):
                    l_prev[i] = l_prev[i] * scores_scale[i] + scores_sum[i]
                
                T.sync_threads()

            for i in T.Parallel(GEMM_M):
                l_idx = i // groups
                g = i % groups
                tq = base_l + l_idx
                if tq < seq_len_var:
                    # Log can be -inf if l_prev is 0, handled correctly by + operator usually
                    # But if l_prev is 0, it means everything masked.
                    if l_prev[i] == 0:
                         LSE_Out[i_b, tq, i_h, g] = -T.infinity(accum_dtype)
                    else:
                         LSE_Out[i_b, tq, i_h, g] = m_prev[i] + T.log(l_prev[i])

    return kernel




# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [2,4,8,16]
# BLOCK_S = [16,32,64]
# threads = [64,128,256]
# _configs = list(
#     itertools.product(
#         BLOCK_L,
#         BLOCK_S,
#         threads,
#     ))

# configs = [
#     {
#         "BLOCK_L": c[0],
#         "BLOCK_S": c[1],
#         "threads": c[2],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def weighted_select_kernel(
    batch, h_kv, groups, head_dim, topk, block_size, window_size,
    is_causal, is_training=True, seq_len=None, s_len=None,
    use_drop_mask=False,
    freqs_len=None, rotate_dim=None, sm_scale=None,
    BLOCK_L=None, BLOCK_S=None, threads=None,
    dtype="bfloat16", accum_dtype="float",
):
    if not is_training:
        seq_len_var = T.dynamic("seq_len")
        s_len_var = T.dynamic("s_len")
        freqs_len_var = T.dynamic("freqs_len") if freqs_len is None else freqs_len
    else:
        seq_len_var = seq_len
        s_len_var = s_len
        freqs_len_var = freqs_len if freqs_len is not None else seq_len + (s_len - 1) * block_size + 1
    idx_dtype = "int32"
    
    q_shape = [batch, seq_len_var, h_kv, groups, head_dim]
    k_shape = [batch, s_len_var, h_kv, head_dim]
    lse_shape = [batch, seq_len_var, h_kv, groups]
    
    out_indices_shape = [batch, seq_len_var, h_kv, topk]
    drop_mask_shape = [batch, seq_len_var, s_len_var] if use_drop_mask else [1, 1, 1]
    if rotate_dim is None:
        rotate_dim = head_dim
    q_sin_shape = [batch, seq_len_var, h_kv, groups, rotate_dim]
    k_sin_shape = [batch, s_len_var, h_kv, rotate_dim]
    if BLOCK_L is None: BLOCK_L = (16 + groups - 1) // groups
    if BLOCK_S is None: BLOCK_S = 16
    if threads is None: threads = 64
    
    GEMM_M = BLOCK_L * groups
    num_s_blocks = tilelang.cdiv(s_len_var, BLOCK_S)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def kernel(
        Qc: T.Tensor(q_shape, dtype),
        Kc: T.Tensor(k_shape, dtype),
        LSE_Total: T.Tensor(lse_shape, accum_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
        Qs: T.Tensor(q_sin_shape, dtype),
        Ks: T.Tensor(k_sin_shape, dtype),
        Q_Offset: T.Tensor([1], "int32"),
        DropMask: T.Tensor(drop_mask_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            q_offset = T.if_then_else(is_training, 0, Q_Offset[0])
            i_b, i_h = bz, by
            base_l = bx * BLOCK_L
            
            Qc_shared = T.alloc_shared([GEMM_M, head_dim], dtype)
            Qs_shared = T.alloc_shared([GEMM_M, rotate_dim], dtype)
            Kc_shared = T.alloc_shared([BLOCK_S, head_dim], dtype)
            Ks_shared = T.alloc_shared([BLOCK_S, rotate_dim], dtype)
            score_shared = T.alloc_shared([GEMM_M, BLOCK_S], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, BLOCK_S], accum_dtype)
            
            topk_max_scores = T.alloc_local([topk], accum_dtype)
            topk_indices = T.alloc_local([topk], idx_dtype)
            
            lse_local = T.alloc_local([groups], accum_dtype)
            
            T.fill(topk_max_scores, -T.infinity(accum_dtype))
            T.fill(topk_indices, -1)
            
            tx = T.get_thread_binding()
            if tx < BLOCK_L and (base_l + tx) < seq_len_var:
                for g in T.serial(groups):
                    lse_local[g] = LSE_Total[i_b, base_l + tx, i_h, g]
            
            T.fill(Qc_shared, 0)
            T.fill(Qs_shared, 0)
            for l_idx, g, d in T.Parallel(BLOCK_L, groups, head_dim):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < seq_len_var:
                    Qc_shared[flat_m, d] = Qc[i_b, tq, i_h, g, d]

            for l_idx, g, d in T.Parallel(BLOCK_L, groups, rotate_dim):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < seq_len_var:
                    Qs_shared[flat_m, d] = Qs[i_b, tq, i_h, g, d]
            
            loop_limit_base = num_s_blocks
            if is_causal:
                global_end = q_offset + base_l + BLOCK_L
                loop_limit = T.min(loop_limit_base, tilelang.cdiv(global_end, BLOCK_S))
            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S
                
                T.fill(Kc_shared, 0)
                T.fill(Ks_shared, 0)
                for s_idx, d in T.Parallel(BLOCK_S, head_dim):
                    ts = base_s + s_idx
                    if ts < s_len_var:
                        Kc_shared[s_idx, d] = Kc[i_b, ts, i_h, d]

                for s_idx, d in T.Parallel(BLOCK_S, rotate_dim):
                    ts = base_s + s_idx
                    if ts < s_len_var:
                        Ks_shared[s_idx, d] = Ks[i_b, ts, i_h, d]
                T.sync_threads()
                
                T.clear(acc_s)
                T.gemm(Qc_shared, Kc_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Qs_shared, Ks_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_s, score_shared)
                if is_causal:
                    for i, j in T.Parallel(GEMM_M, BLOCK_S):
                        l_idx = i // groups
                        tq_local = base_l + l_idx
                        tq_global = q_offset + tq_local
                        ts = base_s + j
                        # if ts >= (tq // block_size):
                        if ts >= (tq_global - window_size + 1) // block_size:
                            score_shared[i, j] = -T.infinity(accum_dtype)
                T.sync_threads()
                
                if tx < BLOCK_L and (base_l + tx) < seq_len_var:
                    my_l_idx = tx
                    tq = base_l + my_l_idx
                    tq_global = q_offset + tq
                    limit_chunk = (tq_global - window_size + 1) // block_size
                    val = T.alloc_var(accum_dtype)
                    norm_score = T.alloc_var(accum_dtype)
                    cur_max_norm_score = T.alloc_var(accum_dtype)
                    is_valid = T.alloc_var("bool")
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        in_range = T.alloc_var("bool")
                        in_range = (ts < s_len_var)
                        if in_range:
                            is_valid = (not is_causal) or (ts < limit_chunk)
                            if use_drop_mask:
                                is_valid = is_valid and (DropMask[i_b, tq, ts] == 0)
                            if is_valid:
                                cur_max_norm_score = -T.infinity(accum_dtype)
                                
                                for g in T.serial(groups):
                                    val = score_shared[my_l_idx * groups + g, s_idx] * sm_scale
                                    if val == -T.infinity(accum_dtype):
                                        norm_score = -T.infinity(accum_dtype)
                                    else:
                                        norm_score = val - lse_local[g] 
                                    cur_max_norm_score = T.max(cur_max_norm_score, norm_score)
                                
                                if cur_max_norm_score > topk_max_scores[topk - 1]:
                                    moving = T.alloc_var("bool")
                                    moving = True
                                    for kk in T.serial(topk):
                                        k = topk - 1 - kk
                                        if moving:
                                            if (k > 0) and (cur_max_norm_score > topk_max_scores[k - 1]):
                                                topk_max_scores[k] = topk_max_scores[k - 1]
                                                topk_indices[k] = topk_indices[k - 1]
                                            else:
                                                topk_max_scores[k] = cur_max_norm_score
                                                topk_indices[k] = ts
                                                moving = False
                T.sync_threads()
            
            if tx < BLOCK_L and (base_l + tx) < seq_len_var:
                for k in T.serial(topk):
                    OutIndices[i_b, base_l + tx, i_h, k] = topk_indices[k]

    return kernel



# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [2,4,8]
# BLOCK_TK = [16,32,64]
# threads = [64,128,256]
# _configs = list(
#     itertools.product(
#         BLOCK_L,
#         BLOCK_TK,
#         threads,
#     ))

# configs = [
#     {
#         "BLOCK_L": c[0],
#         "BLOCK_TK": c[1],
#         "threads": c[2],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
# ...existing code...
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def recompute_topk_max_pooling_scores_kernel(
    batch,  h_kv, groups, head_dim, topk, block_size, seq_len=None, s_len=None, is_training=True,
    freqs_len=None, rotate_dim=None, sm_scale=None,
    BLOCK_L=None, BLOCK_TK=None, threads=None,
    dtype="bfloat16", accum_dtype="float",
):

    idx_dtype = "int32"

    if not is_training:
        seq_len_var = T.dynamic("seq_len")
        s_len_var = T.dynamic("s_len")
        freqs_len_var = T.dynamic("freqs_len") if freqs_len is None else freqs_len
    else:
        seq_len_var = seq_len
        s_len_var = s_len
        freqs_len_var = freqs_len if freqs_len is not None else seq_len + (s_len - 1) * block_size + 1

    q_shape = [batch, seq_len_var, h_kv, groups, head_dim]
    k_shape = [batch, s_len_var, h_kv, head_dim]
    indices_shape = [batch, seq_len_var, h_kv, topk]
    out_scores_shape = [batch, seq_len_var, h_kv, groups, topk]
    if rotate_dim is None:
        rotate_dim = head_dim
    q_sin_shape = [batch, seq_len_var, h_kv, groups, rotate_dim]
    k_sin_shape = [batch, s_len_var, h_kv, rotate_dim]

    if BLOCK_L is None:
        BLOCK_L = (16 + groups - 1) // groups
    if BLOCK_TK is None:
        BLOCK_TK = 16
    BLOCK_D = head_dim
    if threads is None:
        threads = 64

    GEMM_M = BLOCK_L * groups
    GEMM_N = BLOCK_L * BLOCK_TK
    tk_blocks = (topk + BLOCK_TK - 1) // BLOCK_TK

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    @T.prim_func
    def fwd_recompute(
        Qc: T.Tensor(q_shape, dtype),
        Kc: T.Tensor(k_shape, dtype),
        Indices: T.Tensor(indices_shape, idx_dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
        Qs: T.Tensor(q_sin_shape, dtype),
        Ks: T.Tensor(k_sin_shape, dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L
            Qc_shared = T.alloc_shared([GEMM_M, BLOCK_D], dtype)
            Qs_shared = T.alloc_shared([GEMM_M, rotate_dim], dtype)
            Kc_shared = T.alloc_shared([BLOCK_L * BLOCK_TK, BLOCK_D], dtype)
            Ks_shared = T.alloc_shared([BLOCK_L * BLOCK_TK, rotate_dim], dtype)
            score_shared = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)

            T.fill(Qc_shared, 0)
            T.fill(Qs_shared, 0)
            for l_idx, g, d in T.Parallel(BLOCK_L, groups, BLOCK_D):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < seq_len_var:
                    Qc_shared[flat_m, d] = Qc[i_b, tq, i_h, g, d]

            for l_idx, g, d in T.Parallel(BLOCK_L, groups, rotate_dim):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < seq_len_var:
                    Qs_shared[flat_m, d] = Qs[i_b, tq, i_h, g, d]

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
                            Kc_shared[off, d] = Kc[i_b, idx, i_h, d]
                        else:
                            Kc_shared[off, d] = T.Cast(dtype, 0.0)
                    else:
                        if off < BLOCK_L * BLOCK_TK:
                            Kc_shared[off, d] = T.Cast(dtype, 0.0)

                for l_idx, tk_idx, d in T.Parallel(BLOCK_L, BLOCK_TK, rotate_dim):
                    tq = base_l + l_idx
                    off = l_idx * BLOCK_TK + tk_idx
                    if (tq < seq_len_var) and (tk_idx < tk_size):
                        k_id = tk_base + tk_idx
                        idx = Indices[i_b, tq, i_h, k_id]
                        if (idx >= 0) and (idx < s_len_var):
                            Ks_shared[off, d] = Ks[i_b, idx, i_h, d]
                        else:
                            Ks_shared[off, d] = T.Cast(dtype, 0.0)
                    else:
                        if off < BLOCK_L * BLOCK_TK:
                            Ks_shared[off, d] = T.Cast(dtype, 0.0)
                T.sync_threads()

                T.clear(acc_s)
                T.gemm(
                    Qc_shared,
                    Kc_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow
                )
                T.gemm(
                    Qs_shared,
                    Ks_shared,
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
    
    h_kv: int,
    topk: int,
    BLOCK_L: int = 16,
    num_threads: int = 64,
    seq_len=None,
    is_training: bool = True,
):

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



def _pope_pre_rotate_qk_torch(q, lmks, freqs, bias, q_offset, block_size, rotate_dim):
    """
    Pure-PyTorch equivalent of pope_pre_rotate_qk_kernel, used so autograd can
    handle the gradients of q / lmks / freqs / bias for the PoPE topk path.

    Inputs:
        q     : [B, L, h_kv, G, D]
        lmks  : [B, S, h_kv, D]
        freqs : [L_total, R]
        bias  : [h_kv, G, R]
        q_offset    : python int
        block_size  : python int
        rotate_dim  : python int (R)

    Outputs (matching the TileLang kernel layout):
        q_cos : [B, L, h_kv, G, D]   (rotated cos part on [..,:R], copy of q on [..,R:])
        q_sin : [B, L, h_kv, G, R]
        k_cos : [B, S, h_kv, D]
        k_sin : [B, S, h_kv, R]

    Dtype trajectory mirrors the kernel:
        theta = freqs - bias                       (in common_dtype)
        cos/sin = cos/sin in fp32, then cast to qk_dtype
        Qc = (qx_fp32 * cos_fp32).to(qk_dtype)     (i.e. fp32 mul, cast back)
    so the wrapper output stays close to the original kernel and gradients
    are computed by autograd via the same formula.
    """
    B, L, h_kv, G, D = q.shape
    _, S, _, _ = lmks.shape
    R = rotate_dim
    qk_dtype = q.dtype
    device = q.device

    common_theta_dtype = freqs.dtype if freqs.dtype == bias.dtype else torch.float32

    # ---- Q side: theta_q = freqs[q_offset + l] - bias[h, g] ----
    q_pos = torch.arange(L, device=device, dtype=torch.long) + int(q_offset)
    fq = freqs.index_select(0, q_pos).to(common_theta_dtype)         # [L, R]
    bias_t = bias.to(common_theta_dtype).view(1, 1, h_kv, G, R)
    theta_q = fq.view(1, L, 1, 1, R) - bias_t                        # [1, L, h_kv, G, R]
    cos_q = torch.cos(theta_q.float()).to(qk_dtype)                  # bf16
    sin_q = torch.sin(theta_q.float()).to(qk_dtype)

    q_rot_f = q[..., :R].float()                                     # fp32
    qc_rot = (q_rot_f * cos_q.float()).to(qk_dtype)                  # bf16
    qs = (q_rot_f * sin_q.float()).to(qk_dtype)                      # bf16

    if R < D:
        q_cos = torch.cat([qc_rot, q[..., R:]], dim=-1).contiguous() # [B, L, h_kv, G, D]
    else:
        q_cos = qc_rot.contiguous()
    q_sin = qs.contiguous()

    # ---- K side: theta_k = freqs[k_pos], no bias ----
    k_pos = torch.arange(S, device=device, dtype=torch.long) * block_size + (block_size - 1)
    fk = freqs.index_select(0, k_pos).to(common_theta_dtype)         # [S, R]
    cos_k = torch.cos(fk.float()).to(qk_dtype).view(1, S, 1, R)
    sin_k = torch.sin(fk.float()).to(qk_dtype).view(1, S, 1, R)

    k_rot_f = lmks[..., :R].float()                                  # fp32
    kc_rot = (k_rot_f * cos_k.float()).to(qk_dtype)                  # bf16
    ks = (k_rot_f * sin_k.float()).to(qk_dtype)                      # bf16

    if R < D:
        k_cos = torch.cat([kc_rot, lmks[..., R:]], dim=-1).contiguous()  # [B, S, h_kv, D]
    else:
        k_cos = kc_rot.contiguous()
    k_sin = ks.contiguous()

    return q_cos, q_sin, k_cos, k_sin


class SoftmaxTopKMaxPoolingFusedFn(torch.autograd.Function):
    """
    PoPE topk fused autograd function.

    Inputs are *already pre-rotated* tensors q_cos / q_sin / k_cos / k_sin,
    produced by a plain-PyTorch helper (see SoftmaxTopKMaxPooling_Fused.forward).
    Because the prerotate step lives in PyTorch, the gradients of q / lmks /
    freqs / bias are handled by autograd automatically -- this autograd.Function
    only needs to compute grads w.r.t. q_cos / q_sin / k_cos / k_sin.

    The downstream TileLang kernels (lse / select / recompute) compute, for
    each (b, l, h, g, s):

        score = sm_scale * ( sum_d Qc[b,l,h,g,d] * Kc[b,s,h,d]
                           + sum_r Qs[b,l,h,g,r] * Ks[b,s,h,r] )

    so the local gradients are two independent bmms (D-dim and R-dim). This
    mirrors the non-PoPE backward in topk_head_softmax.py exactly, just with
    an extra (R-dim) pair for the sin part.
    """
    @staticmethod
    def forward(ctx, q_cos, q_sin, k_cos, k_sin, lse_swa,
                lse_kernel,
                select_kernel,
                sort_kernel,
                recompute_kernel,
                q_offset_tensor,
                drop_mask,
                sm_scale,
                ):
        B, L, h_kv, G, D = q_cos.shape
        _, S, h_kv2, D2 = k_cos.shape
        _, _, _, _, R = q_sin.shape
        dtype = q_cos.dtype

        assert h_kv == h_kv2 and D == D2
        assert q_sin.shape == (B, L, h_kv, G, R), \
            f"q_sin shape mismatch: {q_sin.shape}, expect [{B},{L},{h_kv},{G},{R}]"
        assert k_sin.shape == (B, S, h_kv, R), \
            f"k_sin shape mismatch: {k_sin.shape}, expect [{B},{S},{h_kv},{R}]"

        # Downstream TileLang kernels require contiguous inputs.
        q_cos = q_cos.contiguous()
        q_sin = q_sin.contiguous()
        k_cos = k_cos.contiguous()
        k_sin = k_sin.contiguous()

        lse_hsa = lse_kernel(q_cos, k_cos, q_sin, k_sin, q_offset_tensor)

        if lse_swa.dim() == 3: # [B, L, h_q]
            lse_swa_view = lse_swa.view(B, L, h_kv, G)
        else:
            lse_swa_view = lse_swa

        lse_total = torch.logaddexp(lse_swa_view, lse_hsa)

        if drop_mask is None:
            drop_mask_in = torch.zeros(1, 1, 1, dtype=torch.int32, device=q_cos.device)
        else:
            drop_mask_in = drop_mask
        indices_raw = select_kernel(
            q_cos, k_cos, lse_total, q_sin, k_sin, q_offset_tensor, drop_mask_in
        )  # int32

        indices_sorted = sort_kernel(indices_raw)

        best_scores_buf = recompute_kernel(
            q_cos, k_cos, indices_sorted, q_sin, k_sin
        )  # float32

        ctx.save_for_backward(q_cos, q_sin, k_cos, k_sin, indices_sorted)
        ctx.h_kv = h_kv
        ctx.G = G
        ctx.shapes = (B, L, S, h_kv, D, R)
        ctx.sm_scale = sm_scale

        return indices_sorted, best_scores_buf.to(dtype)

    @staticmethod
    def backward(ctx, grad_indices_unused, grad_scores_selected):
        q_cos, q_sin, k_cos, k_sin, indices = ctx.saved_tensors
        indices = indices.long()
        B, L, S, h_kv, D, R = ctx.shapes
        G = ctx.G

        sm_scale = ctx.sm_scale
        device = q_cos.device

        grad_scores_dense = torch.zeros(
            (B, h_kv, G, L, S),
            dtype=grad_scores_selected.dtype,
            device=device,
        )

        indices_expanded = indices.permute(0, 2, 1, 3).unsqueeze(2).expand(-1, -1, G, -1, -1)
        valid_mask = (indices_expanded >= 0) & (indices_expanded < S)

        safe_indices = indices_expanded.clone()
        safe_indices[~valid_mask] = 0

        safe_grad = grad_scores_selected.permute(0, 2, 3, 1, 4).clone()
        safe_grad.mul_(sm_scale)
        safe_grad[~valid_mask] = 0

        grad_scores_dense.scatter_(4, safe_indices, safe_grad)
        del indices, indices_expanded, valid_mask, safe_indices, safe_grad

        bs_hg = B * h_kv * G
        dense_in = grad_scores_dense.reshape(bs_hg, L, S)
        dense_in_T = dense_in.transpose(1, 2).contiguous()

        # ---- D-dim grads (cos / "real" part) ----
        # q_cos: [B, L, h_kv, G, D] -> [bs_hg, L, D]
        qc_flat = q_cos.permute(0, 2, 3, 1, 4).reshape(bs_hg, L, D)
        # k_cos: [B, S, h_kv, D] -> expand groups -> [bs_hg, S, D]
        kc_expanded = k_cos.unsqueeze(3).expand(-1, -1, -1, G, -1)
        kc_flat = kc_expanded.permute(0, 2, 3, 1, 4).reshape(bs_hg, S, D)

        grad_qc_flat = torch.bmm(dense_in, kc_flat)
        grad_qc = grad_qc_flat.view(B, h_kv, G, L, D).permute(0, 3, 1, 2, 4).contiguous()

        grad_kc_flat = torch.bmm(dense_in_T, qc_flat)
        grad_kc_grouped = grad_kc_flat.view(B, h_kv, G, S, D)
        grad_kc = grad_kc_grouped.sum(dim=2).permute(0, 2, 1, 3).contiguous()  # [B, S, h_kv, D]
        del qc_flat, kc_flat, kc_expanded, grad_qc_flat, grad_kc_flat, grad_kc_grouped

        # ---- R-dim grads (sin part) ----
        qs_flat = q_sin.permute(0, 2, 3, 1, 4).reshape(bs_hg, L, R)
        ks_expanded = k_sin.unsqueeze(3).expand(-1, -1, -1, G, -1)
        ks_flat = ks_expanded.permute(0, 2, 3, 1, 4).reshape(bs_hg, S, R)

        grad_qs_flat = torch.bmm(dense_in, ks_flat)
        grad_qs = grad_qs_flat.view(B, h_kv, G, L, R).permute(0, 3, 1, 2, 4).contiguous()

        grad_ks_flat = torch.bmm(dense_in_T, qs_flat)
        grad_ks_grouped = grad_ks_flat.view(B, h_kv, G, S, R)
        grad_ks = grad_ks_grouped.sum(dim=2).permute(0, 2, 1, 3).contiguous()  # [B, S, h_kv, R]
        del qs_flat, ks_flat, ks_expanded, grad_qs_flat, grad_ks_flat, grad_ks_grouped
        del dense_in, dense_in_T, grad_scores_dense

        # forward signature:
        #   (q_cos, q_sin, k_cos, k_sin, lse_swa,
        #    lse_kernel, select_kernel, sort_kernel, recompute_kernel,
        #    q_offset_tensor, drop_mask, sm_scale)
        return grad_qc, grad_qs, grad_kc, grad_ks, None, \
               None, None, None, None, None, None, None


class SoftmaxTopKMaxPooling_Fused(torch.nn.Module):
    def __init__(self, topk, block_size, window_size, is_causal, is_training=True, use_drop_mask=False):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self.is_training = is_training
        self.use_drop_mask = use_drop_mask
        self._cached_lse_kernel = None
        self._cached_select_kernel = None
        self._cached_sort_kernel = None
        self._cached_recompute_kernel = None
        self._cached_shape = None

    def forward(self, q, lmks, lse_swa, freqs, bias, q_offset, drop_mask=None, sm_scale=None):
        # q:     [B, L, h_kv, G, D]
        # lmks:  [B, S, h_kv, D]
        # lse_swa: [B, L, h_q]
        # freqs: [L_total, rotate_dim]
        # bias:  [h_kv, G, rotate_dim]  (applied on Q side with minus sign)
        B, L, h_kv, G, D = q.shape
        _, S, _, _ = lmks.shape
        topk = self.topk
        is_causal = self.is_causal
        block_size = self.block_size
        window_size = self.window_size
        is_training = self.is_training
        use_drop_mask = self.use_drop_mask

        freqs_len_total = freqs.shape[0]
        rotate_dim = freqs.shape[1]

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)
        # Bake sm_scale into kernel cache_key so different scales recompile.
        sm_scale_key = float(sm_scale)

        # Auto-derive kernel dtype from q's torch dtype (so callers can pass fp32).
        dtype_str = _torch_dtype_to_str(q.dtype)
        freqs_dtype_str = _torch_dtype_to_str(freqs.dtype)
        bias_dtype_str = _torch_dtype_to_str(bias.dtype)

        if not is_training:
            shape_key = (B, h_kv, G, D, topk, block_size, window_size, is_causal, is_training, use_drop_mask, rotate_dim, sm_scale_key, dtype_str, freqs_dtype_str, bias_dtype_str)
        else:
            shape_key = (B, L, S, h_kv, G, D, topk, block_size, window_size, is_causal, is_training, use_drop_mask, freqs_len_total, rotate_dim, sm_scale_key, dtype_str, freqs_dtype_str, bias_dtype_str)

        if self._cached_shape != shape_key:
            seq_len_param = None if not is_training else L
            s_len_param = None if not is_training else S
            freqs_len_param = None if not is_training else freqs_len_total

            self._cached_lse_kernel = hsa_lse_kernel(
                B, seq_len=seq_len_param, s_len=s_len_param, h_kv=h_kv, groups=G, head_dim=D,
                block_size=block_size, window_size=window_size, is_causal=is_causal,
                is_training=is_training, freqs_len=freqs_len_param, rotate_dim=rotate_dim,
                sm_scale=sm_scale, dtype=dtype_str,
            )

            self._cached_select_kernel = weighted_select_kernel(
                B, seq_len=seq_len_param, s_len=s_len_param, h_kv=h_kv, groups=G, head_dim=D,
                topk=topk, block_size=block_size, window_size=window_size, is_causal=is_causal,
                is_training=is_training, use_drop_mask=use_drop_mask, freqs_len=freqs_len_param,
                rotate_dim=rotate_dim, sm_scale=sm_scale, dtype=dtype_str,
            )

            self._cached_sort_kernel = sort_topk_indices_kernel(
                B, seq_len=seq_len_param, h_kv=h_kv, topk=topk,
                is_training=is_training
            )

            self._cached_recompute_kernel = recompute_topk_max_pooling_scores_kernel(
                B, seq_len=seq_len_param, s_len=s_len_param, h_kv=h_kv, groups=G, head_dim=D,
                topk=topk, block_size=block_size,
                is_training=is_training, freqs_len=freqs_len_param, rotate_dim=rotate_dim,
                sm_scale=sm_scale, dtype=dtype_str,
            )
            self._cached_shape = shape_key

        lse_kernel = self._cached_lse_kernel
        select_kernel = self._cached_select_kernel
        sort_kernel = self._cached_sort_kernel
        recompute_kernel = self._cached_recompute_kernel

        q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=q.device)

        # TileLang kernels require contiguous input tensors. Callers often
        # pass strided slices (e.g. lmks = hsa_k[:, chunk_size-1::chunk_size]),
        # so make them contiguous here.
        q = q.contiguous()
        lmks = lmks.contiguous()

        # ---- PoPE pre-rotate in pure PyTorch ----
        # We delegate the gradients of q / lmks / freqs / bias to autograd by
        # doing the rotation in plain PyTorch. The downstream autograd Function
        # then only needs to compute grads w.r.t. q_cos / q_sin / k_cos / k_sin.
        #
        # We mirror the dtype trajectory of pope_pre_rotate_qk_kernel:
        #   qx       = float(q[bf16])              # promote to fp32
        #   theta    = freqs[bf16] - bias[bf16]    # subtraction in common dtype
        #   cos/sin  = T.cos / T.sin in fp32       # then cast back to qk dtype
        #   Qc       = bf16(qx * cos)              # fp32 mul, cast back bf16
        # so wrapper output stays close to the original kernel.
        q_cos, q_sin, k_cos, k_sin = _pope_pre_rotate_qk_torch(
            q, lmks, freqs, bias, q_offset, block_size, rotate_dim,
        )

        indices, scores = SoftmaxTopKMaxPoolingFusedFn.apply(
            q_cos, q_sin, k_cos, k_sin, lse_swa,
            lse_kernel, select_kernel, sort_kernel, recompute_kernel,
            q_offset_tensor, drop_mask, sm_scale,
        )
        # Reshape scores: [B, L, h_kv, G, topk] -> [B, L, h_q, topk]
        scores = scores.view(B, L, h_kv * G, -1)
        return indices, scores

_SOFTMAX_MODULE_CACHE = {}


def online_softmax_topk_head_pope(
    q: torch.Tensor,
    lmks: torch.Tensor,
    lse_swa: torch.Tensor,
    freqs: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    block_size: int,
    window_size: int,
    is_causal: bool = True,
    q_offset: int = 0,
    is_training: bool = True,
    drop_mask: torch.Tensor = None,
    sm_scale: float = None,
):
    """
    Args:
        q (torch.Tensor): shape = [B, L, h_q, D]
        lmks (torch.Tensor): shape = [B, S, h_kv, D]
        lse_swa (torch.Tensor): shape = [B, L, h_q] or [B, L, h_kv, G]
        freqs (torch.Tensor): shape = [L_total, rotate_dim]  (rotate_dim <= D)
        bias (torch.Tensor): shape = [h_q, rotate_dim]
        sm_scale (float, optional): softmax scale. Defaults to 1/sqrt(D) when None.
            Pass an explicit value (e.g. 1/sqrt(D+rotate_dim)) to match the naive
            PoPE path which feeds rotated q (last-dim = D+R) into the unfused op.
    """

    if q.dim() == 4:
        B, L, h_q, D = q.shape
        h_kv = lmks.shape[2]
        D_lmk = lmks.shape[3]

        if D != D_lmk:
            assert D_lmk % D == 0, f"lmks D dim ({D_lmk}) must be divisible by q D dim ({D})"
            d_ratio = D_lmk // D
            # lmks: [B, S, h_kv, D_lmk] -> [B, S, h_kv * d_ratio, D]
            lmks = lmks.reshape(lmks.shape[0], lmks.shape[1], h_kv * d_ratio, D)
            h_kv = h_kv * d_ratio

        assert h_q % h_kv == 0, f"h_q ({h_q}) must be divisible by h_kv ({h_kv})"
        G = h_q // h_kv
        q = q.view(B, L, h_kv, G, D)
    else:
        B, L, h_kv, G, D = q.shape
        D_lmk = lmks.shape[3]
        if D != D_lmk:
            assert D_lmk % D == 0, f"lmks D dim ({D_lmk}) must be divisible by q D dim ({D})"
            d_ratio = D_lmk // D
            lmks = lmks.reshape(lmks.shape[0], lmks.shape[1], lmks.shape[2] * d_ratio, D)
            new_h_kv = lmks.shape[2]
            new_G = (h_kv * G) // new_h_kv
            assert (h_kv * G) % new_h_kv == 0
            q = q.reshape(B, L, new_h_kv, new_G, D)
            if lse_swa.dim() == 3:
                lse_swa = lse_swa.view(B, L, new_h_kv, new_G)
            else:
                lse_swa = lse_swa.view(B, L, new_h_kv, new_G)
            h_kv = new_h_kv
            G = new_G

    rotate_dim = freqs.shape[1]
    # At this point q is always 5D: [B, L, h_kv, G, D]
    G = q.shape[3]
    # bias: [h_q, rotate_dim] -> reshape to [h_kv, G, rotate_dim] for kernel
    if bias.dim() == 2:
        assert bias.shape[0] == h_kv * G and bias.shape[1] == rotate_dim, \
            f"bias shape {bias.shape} does not match h_q={h_kv * G}, rotate_dim={rotate_dim}"
        bias = bias.view(h_kv, G, rotate_dim)
    elif bias.dim() == 3:
        assert bias.shape == (h_kv, G, rotate_dim), \
            f"bias shape {bias.shape} does not match [{h_kv}, {G}, {rotate_dim}]"
    else:
        raise ValueError(f"bias must be 2D [h_q, R] or 3D [h_kv, G, R], got dim={bias.dim()}")
    assert freqs.dim() == 2 and rotate_dim <= D, \
        f"freqs shape {freqs.shape}, rotate_dim={rotate_dim} must be <= D={D}"

    use_drop_mask = drop_mask is not None
    cache_key = (topk, block_size, window_size, is_causal, is_training, use_drop_mask)

    if cache_key not in _SOFTMAX_MODULE_CACHE:
        _SOFTMAX_MODULE_CACHE[cache_key] = SoftmaxTopKMaxPooling_Fused(
            topk, block_size, window_size, is_causal, is_training, use_drop_mask
        )

    return _SOFTMAX_MODULE_CACHE[cache_key](
        q, lmks, lse_swa, freqs, bias, q_offset=q_offset, drop_mask=drop_mask, sm_scale=sm_scale
    )


def online_softmax_topk_head_pope_prerotate(
    q: torch.Tensor,
    lmks: torch.Tensor,
    lse_swa: torch.Tensor,
    freqs: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    block_size: int,
    window_size: int,
    is_causal: bool = True,
    q_offset: int = 0,
    is_training: bool = True,
    drop_mask: torch.Tensor = None,
    sm_scale: float = None,
):
    """
    PoPE prerotate wrapper: pre-rotate Q/K with plain PyTorch ops (so autograd
    handles the gradients of q / lmks / bias automatically), concatenate the
    cosine and sine parts along the head_dim axis to obtain effective tensors
    of last-dim D + rotate_dim, then run the non-PoPE topk pipeline
    (online_softmax_topk_head) with an explicit sm_scale = 1/sqrt(D) so the
    resulting logits match the original PoPE math.

    Args / shapes mirror online_softmax_topk_head_pope.
    """
    assert drop_mask is None or drop_mask.dtype == torch.int32, \
        "drop_mask must be int32 if provided"

    # ---- Reshape q to 5D and align lmks h_kv (same logic as the fused path) ----
    if q.dim() == 4:
        B, L, h_q, D = q.shape
        h_kv = lmks.shape[2]
        D_lmk = lmks.shape[3]
        if D != D_lmk:
            assert D_lmk % D == 0, f"lmks D dim ({D_lmk}) must be divisible by q D dim ({D})"
            d_ratio = D_lmk // D
            lmks = lmks.reshape(lmks.shape[0], lmks.shape[1], h_kv * d_ratio, D)
            h_kv = h_kv * d_ratio
        assert h_q % h_kv == 0, f"h_q ({h_q}) must be divisible by h_kv ({h_kv})"
        G = h_q // h_kv
        q = q.view(B, L, h_kv, G, D)
    else:
        B, L, h_kv, G, D = q.shape
        D_lmk = lmks.shape[3]
        if D != D_lmk:
            assert D_lmk % D == 0, f"lmks D dim ({D_lmk}) must be divisible by q D dim ({D})"
            d_ratio = D_lmk // D
            lmks = lmks.reshape(lmks.shape[0], lmks.shape[1], lmks.shape[2] * d_ratio, D)
            new_h_kv = lmks.shape[2]
            new_G = (h_kv * G) // new_h_kv
            assert (h_kv * G) % new_h_kv == 0
            q = q.reshape(B, L, new_h_kv, new_G, D)
            if lse_swa.dim() == 3:
                lse_swa = lse_swa.view(B, L, new_h_kv, new_G)
            else:
                lse_swa = lse_swa.view(B, L, new_h_kv, new_G)
            h_kv = new_h_kv
            G = new_G

    rotate_dim = freqs.shape[1]
    G = q.shape[3]
    h_q = h_kv * G
    if bias.dim() == 2:
        assert bias.shape[0] == h_q and bias.shape[1] == rotate_dim, \
            f"bias shape {bias.shape} does not match h_q={h_q}, rotate_dim={rotate_dim}"
        bias = bias.view(h_kv, G, rotate_dim)
    elif bias.dim() == 3:
        assert bias.shape == (h_kv, G, rotate_dim), \
            f"bias shape {bias.shape} does not match [{h_kv}, {G}, {rotate_dim}]"
    else:
        raise ValueError(f"bias must be 2D [h_q, R] or 3D [h_kv, G, R], got dim={bias.dim()}")
    assert freqs.dim() == 2 and rotate_dim <= D, \
        f"freqs shape {freqs.shape}, rotate_dim={rotate_dim} must be <= D={D}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    S = lmks.shape[1]
    R = rotate_dim
    device = q.device
    qk_dtype = q.dtype

    # ---- PyTorch prerotate (autograd-friendly) ----
    # We try to mimic the dtype trajectory of the fused PoPE prerotate kernel
    # closely so wrapper output stays numerically aligned with the fused path:
    #
    #   In the kernel:
    #     qx       = float(Q[bf16])                  # promote q to fp32
    #     theta    = Freqs[bf16] - Bias[bf16]        # bf16 subtraction
    #     cos_t    = T.cos(theta)                    # bf16 cos (TileLang)
    #     Qc       = bf16(qx * cos_t)                # fp32 mul, cast back bf16
    #
    # In PyTorch we therefore:
    #   1) compute theta_q in *freqs.dtype* (bf16 by convention)
    #   2) take cos/sin in fp32 (PyTorch promotes bf16 inputs anyway), then
    #      cast cos/sin back to qk_dtype to match the bf16 truncation that
    #      happens at the boundary of T.cos in the kernel
    #   3) cast q[..,:R] to fp32 for the multiply and cast the product back
    #      to qk_dtype, matching the kernel's qx*cos_t -> bf16 cast.

    freqs_dtype = freqs.dtype
    bias_dtype = bias.dtype
    common_theta_dtype = freqs_dtype if freqs_dtype == bias_dtype else torch.float32

    # Q side: theta_q = freqs[q_pos+q_offset] - bias  (per-head, per-group)
    q_pos = torch.arange(L, device=device, dtype=torch.long) + int(q_offset)
    fq = freqs.index_select(0, q_pos).to(common_theta_dtype)         # [L, R]
    bias_t = bias.to(common_theta_dtype).view(1, 1, h_kv, G, R)
    theta_q = fq.view(1, L, 1, 1, R) - bias_t                        # [1, L, h_kv, G, R]
    cos_q = torch.cos(theta_q.float()).to(qk_dtype)                  # bf16 cos
    sin_q = torch.sin(theta_q.float()).to(qk_dtype)

    q_rot_f = q[..., :R].float()                                     # [B, L, h_kv, G, R] fp32
    qc_rot = (q_rot_f * cos_q.float()).to(qk_dtype)                  # bf16
    qs = (q_rot_f * sin_q.float()).to(qk_dtype)                      # bf16
    if R < D:
        qc = torch.cat([qc_rot, q[..., R:]], dim=-1)                 # [B, L, h_kv, G, D]
    else:
        qc = qc_rot
    q_eff = torch.cat([qc, qs], dim=-1)                              # [B, L, h_kv, G, D + R]

    # K side: theta_k = freqs[k_pos], no bias
    k_pos = torch.arange(S, device=device, dtype=torch.long) * block_size + (block_size - 1)
    fk = freqs.index_select(0, k_pos).to(common_theta_dtype)         # [S, R]
    cos_k = torch.cos(fk.float()).to(qk_dtype).view(1, S, 1, R)
    sin_k = torch.sin(fk.float()).to(qk_dtype).view(1, S, 1, R)

    k_rot_f = lmks[..., :R].float()                                  # [B, S, h_kv, R] fp32
    kc_rot = (k_rot_f * cos_k.float()).to(qk_dtype)                  # bf16
    ks = (k_rot_f * sin_k.float()).to(qk_dtype)                      # bf16
    if R < D:
        kc = torch.cat([kc_rot, lmks[..., R:]], dim=-1)              # [B, S, h_kv, D]
    else:
        kc = kc_rot
    k_eff = torch.cat([kc, ks], dim=-1)                              # [B, S, h_kv, D + R]

    return _base_online_softmax_topk_head(
        q_eff, k_eff, lse_swa,
        topk=topk, block_size=block_size, window_size=window_size,
        is_causal=is_causal, q_offset=q_offset, is_training=is_training,
        drop_mask=drop_mask, sm_scale=sm_scale,
    )



def test_fused_topk_softmax_max_pooling_correctness():
    print("\n" + "=" * 70)
    print("=== Testing Fused Softmax TopK Max-Pooling Kernel (PoPE) Correctness ===")
    print("=" * 70)

    # 配置参数
    B, L, D = 4, 8192, 128
    h_kv = 32
    G = 1
    h_q = h_kv * G
    S = 64
    topk = 16
    is_causal = True
    block_size = 64
    window_size = 512  # Add window_size

    dtype = torch.bfloat16
    device = "cuda"

    print(f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, topk={topk}, "
          f"is_causal={is_causal}, block_size={block_size}, window_size={window_size}")

    torch.manual_seed(4200)

    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)
    lse_swa = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10

    # PoPE 参数：freqs 用小幅度 (~0.1) 避免 cos/sin 饱和；bias 同样。
    # freqs 是预计算的位置频率（常量 buffer，不参与优化）；bias 是 learnable (per-query-head)。
    freqs_len_total = max(L, (S - 1) * block_size + 1)
    freqs = torch.randn(freqs_len_total, D, device=device, dtype=dtype) * 0.1
    bias = (torch.randn(h_q, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)

    # ============ Forward Correctness ============
    print("\n--- Forward Correctness ---")

    # Reference returns: indices [B, L, h_kv, topk], scores [B, L, h_q, topk]
    ref_indices, ref_scores = ref_softmax_topk_max_pooling(
        q.detach(), lmks.detach(), lse_swa.detach().float(),
        freqs.detach(), bias.detach(),
        topk, block_size, window_size, is_causal,
    )

    # Fused returns: indices [B, L, h_kv, topk], scores [B, L, h_q, topk]
    fused_indices, fused_scores = online_softmax_topk_head_pope(
        q, lmks, lse_swa, freqs, bias, topk, block_size, window_size, is_causal
    )

    # Reshape fused scores back to [B, L, h_kv, G, topk] for comparison
    fused_scores_reshaped = fused_scores.view(B, L, h_kv, G, topk)

    # Calculate ground truth scores for all candidates (PoPE)
    scores_all_ref = _pope_qk_logits(
        q.detach().float(), lmks.detach().float(), freqs.detach().float(), bias.detach().float(),
        block_size=block_size, q_offset=0,
    )
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D))

    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        # Update manual check to match window mechanism
        aligned_threshold = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= aligned_threshold

        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))

    safe_indices = fused_indices.clone()
    safe_indices[safe_indices < 0] = 0

    # Gather scores using fused indices to verify values
    indices_expanded = safe_indices.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded)

    # Compare valid scores (ignore masked/padding)
    valid_mask = (scores_gathered_ref > -1e9) & (fused_scores_reshaped.float() > -1e9)

    if valid_mask.sum() == 0:
        print("Warning: No valid scores to compare (all masked?)")
        max_score_diff = 0.0
        rel_l2_score = 0.0
    else:
        score_diff = torch.abs(scores_gathered_ref[valid_mask] - fused_scores_reshaped.float()[valid_mask])
        max_score_diff = score_diff.max().item()
        rel_l2_score = score_diff.norm().item() / (scores_gathered_ref[valid_mask].norm().item() + 1e-6)

    print(f"Forward scores (valid only) - Max Diff: {max_score_diff:.6f}")
    print(f"Forward scores (valid only) - L2 RelErr: {rel_l2_score:.6f}")

    indices_match = (fused_indices.long() == ref_indices.long())  # [B, L, h_kv, topk]

    ref_scores_pooled = ref_scores.max(dim=3).values  # [B, L, h_kv, topk]
    valid_indices_mask = (ref_scores_pooled > -1e9)   # [B, L, h_kv, topk]
    match_rate = indices_match[valid_indices_mask].float().mean().item()

    print(f"Indices Match Rate (Valid Elements): {match_rate*100:.6f}%")

    if match_rate >= 0.99 and max_score_diff < 1 and rel_l2_score < 1e-2:
        print("✅ Fused Forward PASSED")
    else:
        print("❌ Fused Forward FAILED")

    # ============ Backward Correctness ============
    print("\n--- Backward Correctness ---")

    q_fused = q.detach().clone().requires_grad_(True)
    lmks_fused = lmks.detach().clone().requires_grad_(True)
    # freqs 是常量 buffer（PoPE 预计算位置频率），不参与反向
    freqs_fused = freqs.detach().clone()
    bias_fused = bias.detach().clone().requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    lmks_ref = lmks.detach().clone().requires_grad_(True)
    freqs_ref = freqs.detach().clone()
    bias_ref = bias.detach().clone().requires_grad_(True)

    # grad_output shape matches fused output: [B, L, h_q, topk]
    grad_output = torch.randn(B, L, h_kv * G, topk, dtype=dtype, device=device)

    # Create a view for reference implementation: [B, L, h_kv, G, topk]
    grad_output_ref_view = grad_output.view(B, L, h_kv, G, topk)

    # Mask gradients for invalid positions based on reference forward pass
    with torch.no_grad():
        _, ref_scores_check = ref_softmax_topk_max_pooling(
            q_ref, lmks_ref, lse_swa, freqs_ref, bias_ref,
            topk, block_size, window_size, is_causal,
        )
        invalid_mask = (ref_scores_check < -1e9)
        grad_output_ref_view[invalid_mask] = 0.0

    # Fused Backward
    indices_fused_bwd, scores_fused_bwd = online_softmax_topk_head_pope(
        q_fused, lmks_fused, lse_swa, freqs_fused, bias_fused,
        topk, block_size, window_size, is_causal,
    )
    loss_fused = (scores_fused_bwd * grad_output).sum()
    loss_fused.backward()
    grad_q_fused = q_fused.grad.clone()
    grad_lmks_fused = lmks_fused.grad.clone()
    grad_bias_fused = bias_fused.grad.clone()

    # Ref Backward —— 用 PoPE 相似度直接算全量分数
    scores_all_ref = _pope_qk_logits(
        q_ref.float(), lmks_ref.float(), freqs_ref.float(), bias_ref.float(),
        block_size=block_size, q_offset=0,
    )
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D))

    if is_causal:
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))

    # Use fused indices for gather to match the graph
    safe_indices_bwd = indices_fused_bwd.clone()
    safe_indices_bwd[safe_indices_bwd < 0] = 0
    indices_expanded = safe_indices_bwd.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded)

    loss_ref = (scores_gathered_ref * grad_output_ref_view.float()).sum()
    loss_ref.backward()
    grad_q_ref = q_ref.grad.clone()
    grad_lmks_ref = lmks_ref.grad.clone()
    grad_bias_ref = bias_ref.grad.clone()

    # Handle NaNs in grads (from -inf scores)
    for name, g in [("grad_q_ref", grad_q_ref), ("grad_lmks_ref", grad_lmks_ref),
                    ("grad_q_fused", grad_q_fused), ("grad_lmks_fused", grad_lmks_fused),
                    ("grad_bias_ref", grad_bias_ref), ("grad_bias_fused", grad_bias_fused)]:
        if torch.isnan(g).any():
            g.copy_(torch.nan_to_num(g, 0.0))

    def _rel_err(a, b):
        diff = torch.abs(a.float() - b.float())
        return diff.max().item(), diff.norm().item() / (b.float().norm().item() + 1e-6)

    max_q, rel_q = _rel_err(grad_q_fused, grad_q_ref)
    max_k, rel_k = _rel_err(grad_lmks_fused, grad_lmks_ref)
    max_b, rel_b = _rel_err(grad_bias_fused, grad_bias_ref)

    print(f"Fused vs Ref - grad_q     Max={max_q:.6f}, L2Rel={rel_q*100:.4f}%")
    print(f"Fused vs Ref - grad_lmks  Max={max_k:.6f}, L2Rel={rel_k*100:.4f}%")
    print(f"Fused vs Ref - grad_bias  Max={max_b:.6f}, L2Rel={rel_b*100:.4f}%")

    if rel_q < 0.01 and rel_k < 0.01 and rel_b < 0.01:
        print("✅ Fused Backward PASSED")
    else:
        print("❌ Fused Backward FAILED")

def test_fused_softmax_topk_max_pooling_memory_and_speed():
    print("\n" + "=" * 70)
    print("=== Benchmark Fused Softmax TopK Max-Pooling Memory and Speed ===")
    print("=" * 70)
    
    B, L, D = 4, 8192, 128
    h_kv = 32
    G = 1
    h_q = h_kv * G
    S = 128
    topk = 32
    is_causal = True
    block_size = 64
    window_size = 512
    rotate_dim = 64
    
    dtype = torch.bfloat16
    device = "cuda"
    assert 0 < rotate_dim <= D, f"rotate_dim ({rotate_dim}) must be in (0, D={D}]"
    
    print(f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, rotate_dim={rotate_dim}, topk={topk}, block_size={block_size}, window_size={window_size}")
    
    torch.manual_seed(42)
    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)

    lse_swa = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10

    freqs_len_total = max(L, (S - 1) * block_size + 1)
    freqs = torch.randn(freqs_len_total, rotate_dim, device=device, dtype=dtype) * 0.1
    bias = (torch.randn(h_q, rotate_dim, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    
    grad_output = torch.randn(B, L, h_q, topk, dtype=dtype, device=device)
    
    n_iters = 20
    
    def run_fused():
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)
        bias_t = bias.detach().clone().requires_grad_(True)
        freqs_t = freqs.detach().clone()

        lse_swa_t = lse_swa.detach().clone()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        _, scores = online_softmax_topk_head_pope(q_t, lmks_t, lse_swa_t, freqs_t, bias_t, topk, block_size, window_size, is_causal)
        loss = (scores * grad_output).sum()
        loss.backward()
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(5):
            q_t.grad = None
            lmks_t.grad = None
            bias_t.grad = None
            _ = online_softmax_topk_head_pope(q_t, lmks_t, lse_swa_t, freqs_t, bias_t, topk, block_size, window_size, is_causal)
            _, scores = online_softmax_topk_head_pope(q_t, lmks_t, lse_swa_t, freqs_t, bias_t, topk, block_size, window_size, is_causal)
            loss = (scores * grad_output).sum()
            loss.backward()
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        bias_t.grad = None
        _, scores = online_softmax_topk_head_pope(q_t, lmks_t, lse_swa_t, freqs_t, bias_t, topk, block_size, window_size, is_causal)
        loss = (scores * grad_output).sum()
        loss.backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)
        
        # Fwd only
        start_fwd.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            bias_t.grad = None
            _ = online_softmax_topk_head_pope(q_t, lmks_t, lse_swa_t, freqs_t, bias_t, topk, block_size, window_size, is_causal)
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters
        
        # Fwd + Bwd
        start_all.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            bias_t.grad = None
            _, scores = online_softmax_topk_head_pope(q_t, lmks_t, lse_swa_t, freqs_t, bias_t, topk, block_size, window_size, is_causal)
            loss = (scores * grad_output).sum()
            loss.backward()
        end_all.record()
        torch.cuda.synchronize()
        avg_all_ms = start_all.elapsed_time(end_all) / n_iters
        
        # Bwd = (Fwd+Bwd) - Fwd
        avg_bwd_ms = avg_all_ms - avg_fwd_ms
        
        return peak_mem, avg_fwd_ms, avg_all_ms, avg_bwd_ms
    
    # ============ Benchmark Reference (ref_softmax_topk_max_pooling) ============
    grad_output_ref = grad_output.view(B, L, h_kv, G, topk)

    def run_ref():
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)
        bias_t = bias.detach().clone().requires_grad_(True)
        freqs_t = freqs.detach().clone()
        lse_swa_t = lse_swa.detach().clone().float()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        def forward_only():
            _ = ref_softmax_topk_max_pooling(q_t, lmks_t, lse_swa_t, freqs_t, bias_t, topk, block_size, window_size, is_causal)
        
        def forward_backward():
            _, scores = ref_softmax_topk_max_pooling(q_t, lmks_t, lse_swa_t, freqs_t, bias_t, topk, block_size, window_size, is_causal)
            loss = (scores * grad_output_ref).sum()
            loss.backward()
        
        for _ in range(5):
            q_t.grad = None
            lmks_t.grad = None
            bias_t.grad = None
            forward_only()
            forward_backward()
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        bias_t.grad = None
        forward_backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)
        
        # Fwd only
        start_fwd.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            bias_t.grad = None
            forward_only()
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters
        
        # Fwd + Bwd
        start_all.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            bias_t.grad = None
            forward_backward()
        end_all.record()
        torch.cuda.synchronize()
        avg_all_ms = start_all.elapsed_time(end_all) / n_iters
        
        avg_bwd_ms = avg_all_ms - avg_fwd_ms
        
        return peak_mem, avg_fwd_ms, avg_all_ms, avg_bwd_ms
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    
    mem_fused, fwd_fused, all_fused, bwd_fused = run_fused()
    print(f"\n[Fused Softmax TopK Max-Pooling]")
    print(f"  Peak Memory: {mem_fused:.2f} MB")
    print(f"  Avg Fwd Latency: {fwd_fused:.2f} ms")
    print(f"  Avg Fwd+Bwd Latency: {all_fused:.2f} ms")
    print(f"  Derived Bwd Latency: {bwd_fused:.2f} ms")
    
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
        print(f"Speedup (Fwd): {fwd_ref / fwd_fused:.2f}x")
        print(f"Speedup (Bwd): {bwd_ref / bwd_fused:.2f}x")
        print(f"Speedup (Fwd+Bwd): {all_ref / all_fused:.2f}x")
        print(f"Memory Saving: {mem_ref / mem_fused:.2f}x")
        
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

import pytest
@pytest.mark.parametrize("B, L, S, h_kv, G, D, topk, block_size, window_size, rotate_dim", [
    (2, 4096, 64, 2, 8, 64, 16, 64, 64, None),
    (2, 4096, 64, 8, 8, 64, 16, 64, 64, None),
    # rotate_dim < D 的 partial rotation 测试
    (2, 4096, 64, 2, 8, 64, 16, 64, 64, 32),
    (2, 4096, 64, 2, 8, 128, 16, 64, 64, 64),
    (2, 4096, 64, 2, 8, 128, 16, 64, 64, 32),
    (1, 2048, 64, 1, 8, 128, 16, 32, 40, 64),
])
def test_topk_correctness_robust(B, L, S, h_kv, G, D, topk, block_size, window_size, rotate_dim=None):
    device = "cuda"
    dtype = torch.bfloat16
    is_causal = True

    # rotate_dim 默认为 D（全维旋转，向后兼容）
    if rotate_dim is None:
        rotate_dim = D

    torch.manual_seed(42)
    
    print(f"\nTesting Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G}, D={D}, topk={topk}, BS={block_size}, WS={window_size}, rotate_dim={rotate_dim}")

    # 1.准备数据
    # Q: [B, L, h_kv, G, D]
    q_raw = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device)
    # Lmks: [B, S, h_kv, D]
    lmks_raw = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)
    # LSE SWA: [B, L, h_kv, G]
    lse_swa_raw = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10

    # PoPE 参数：freqs 宽度为 rotate_dim（仅前 rotate_dim 维做旋转），bias 是 per-query-head
    freqs_len_total = max(L, (S - 1) * block_size + 1)
    freqs_raw = torch.randn(freqs_len_total, rotate_dim, device=device, dtype=torch.float32) * 0.1
    bias_raw = torch.randn(h_kv * G, rotate_dim, device=device, dtype=torch.float32) * 0.1
    
    # 2.输入隔离 (Clone & Detach)
    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)
    bias_fused = bias_raw.clone().detach().requires_grad_(True)
    freqs_fused = freqs_raw.clone().detach()
    
    q_ref = q_raw.clone().detach().requires_grad_(True)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(True)
    bias_ref = bias_raw.clone().detach().requires_grad_(True)
    freqs_ref = freqs_raw.clone().detach()
    
    # LSE 不需要梯度
    lse_swa = lse_swa_raw.clone().detach()

    # 3.前向计算
    # Fused Kernel
    indices_fused, scores_fused = online_softmax_topk_head_pope(
        q_fused, lmks_fused, lse_swa, freqs_fused, bias_fused,
        topk, block_size, window_size, is_causal,
    )

    # Reference
    indices_ref, scores_ref = ref_softmax_topk_max_pooling(
        q_ref, lmks_ref, lse_swa.float(), freqs_ref, bias_ref,
        topk, block_size, window_size, is_causal,
    )

    # 4.辅助校验函数 (带 Mask 处理)
    def get_abs_err(x, y):
        # 过滤掉 -inf (Masked) 的位置，避免 NaN
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        return (x[mask] - y[mask]).abs().max().item()

    def get_err_ratio(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        err = (x[mask] - y[mask]).square().mean().sqrt().item()
        base = (x[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    def assert_close(prefix, ref, tri, ratio=0.01):
        abs_err = get_abs_err(ref, tri)
        rel_ratio = get_err_ratio(ref, tri)
        msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio, msg

    # 5.验证前向 Scores
    # 关键：scores_ref 与 scores_fused 分别按各自 indices gather。当 match rate < 100% 时
    # 两边 gather 位置不同会放大差异。这里用 fused_indices 从全量 ref scores 中 gather 对齐。
    with torch.no_grad():
        sm_scale_fwd = 1.0 / math.sqrt(D)
        scores_all_ref_fwd = _pope_qk_logits(
            q_ref.float(), lmks_ref.float(), freqs_ref.float(), bias_ref.float(),
            block_size=block_size, q_offset=0,
        ) * sm_scale_fwd
        if is_causal:
            i_idx_fwd = torch.arange(L, device=device).unsqueeze(1)
            j_idx_fwd = torch.arange(S, device=device).unsqueeze(0)
            causal_thr_fwd = (i_idx_fwd - window_size + 1).div(block_size, rounding_mode='floor')
            causal_mask_fwd = j_idx_fwd >= causal_thr_fwd
            scores_all_ref_fwd = scores_all_ref_fwd.masked_fill(
                causal_mask_fwd.view(1, L, 1, 1, S), float('-inf')
            )
        safe_idx_fwd = indices_fused.clone().detach().long()
        invalid_idx_fwd = safe_idx_fwd < 0
        safe_idx_fwd[invalid_idx_fwd] = 0
        scores_ref_aligned = torch.gather(
            scores_all_ref_fwd, -1, safe_idx_fwd.unsqueeze(3).expand(-1, -1, -1, G, -1)
        )
        scores_ref_aligned = scores_ref_aligned.masked_fill(
            invalid_idx_fwd.unsqueeze(3).expand(-1, -1, -1, G, -1), float('-inf')
        )
    
    assert_close("FWD Scores", scores_ref_aligned.float(), scores_fused.view(B, L, h_kv, G, topk).float())

    # 6.反向计算
    # 生成 grad_output [B, L, h_q, topk]
    grad_output = torch.randn_like(scores_fused, dtype=dtype)
    
    # 关键：Mask 掉无效位置的梯度，防止 -inf 导致 NaN 传播
    with torch.no_grad():
        # 基于 fused 的 scores 判定无效位置（与 fused 反向路径一致）
        invalid_mask = scores_fused.view(B, L, h_kv, G, topk).float() < -1e5
        grad_output_view = grad_output.view(B, L, h_kv, G, topk)
        grad_output_view[invalid_mask] = 0.0

    # Fused Backward
    scores_fused.backward(grad_output)
    dq_fused = q_fused.grad.clone()
    dlmks_fused = lmks_fused.grad.clone()
    dbias_fused = bias_fused.grad.clone()

    # Ref Backward：用 fused_indices 从全量 ref scores gather 构造 loss，保证与 fused 在相同位置反传。
    scores_all_ref_bwd = _pope_qk_logits(
        q_ref.float(), lmks_ref.float(), freqs_ref.float(), bias_ref.float(),
        block_size=block_size, q_offset=0,
    ) * sm_scale_fwd
    if is_causal:
        i_idx_bwd = torch.arange(L, device=device).unsqueeze(1)
        j_idx_bwd = torch.arange(S, device=device).unsqueeze(0)
        causal_thr_bwd = (i_idx_bwd - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask_bwd = j_idx_bwd >= causal_thr_bwd
        scores_all_ref_bwd = scores_all_ref_bwd.masked_fill(
            causal_mask_bwd.view(1, L, 1, 1, S), float('-inf')
        )
    safe_idx_bwd = indices_fused.clone().detach().long()
    safe_idx_bwd[safe_idx_bwd < 0] = 0
    scores_gathered_ref_bwd = torch.gather(
        scores_all_ref_bwd, -1, safe_idx_bwd.unsqueeze(3).expand(-1, -1, -1, G, -1)
    )
    loss_ref = (scores_gathered_ref_bwd * grad_output.view(B, L, h_kv, G, topk).float()).sum()
    loss_ref.backward()
    dq_ref = q_ref.grad.clone()
    dlmks_ref = lmks_ref.grad.clone()
    dbias_ref = bias_ref.grad.clone()

    # 7.验证梯度
    # 处理可能的 NaN (虽然上面已经 mask 了 grad_output，但为了稳健性)
    dq_fused = torch.nan_to_num(dq_fused, 0.0)
    dq_ref = torch.nan_to_num(dq_ref, 0.0)
    dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
    dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)
    dbias_fused = torch.nan_to_num(dbias_fused, 0.0)
    dbias_ref = torch.nan_to_num(dbias_ref, 0.0)

    assert_close("DQ", dq_ref.float(), dq_fused.float(), ratio=0.05)
    assert_close("DLmks", dlmks_ref.float(), dlmks_fused.float(), ratio=0.05)
    assert_close("DBias", dbias_ref.float(), dbias_fused.float(), ratio=0.05)

    print(f"Test Passed: B={B}, L={L}, S={S}, G={G}, D={D}, topk={topk}, rotate_dim={rotate_dim}")



import pytest
@pytest.mark.parametrize(
    "test_name, B, q_len, kv_len, h_kv, G, D, topk, block_size, window_size, is_training, q_offset",
    [
        ("train_basic", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0),
    ]
)
def test_train_inference_correctness(test_name, B, q_len, kv_len, h_kv, G, D, topk, block_size, window_size, is_training, q_offset):
    """
    测试 TopK 模块在训练和推理场景下的正确性
    
    训练场景：
    - q_len == kv_len
    - is_training = True
    - q_offset = 0
    - 支持任意 batch size
    
    推理场景：
    - q_len <= kv_len
    - is_training = False
    - q_offset = kv_len - q_len (Q 对应 KV 中最新的部分)
    - batch 必须为 1
    """
    device = "cuda"
    dtype = torch.bfloat16
    is_causal = True
    
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"Config: B={B}, q_len={q_len}, kv_len={kv_len}, h_kv={h_kv}, G={G}, D={D}")
    print(f"        topk={topk}, block_size={block_size}, window_size={window_size}")
    print(f"        is_training={is_training}, q_offset={q_offset}")
    print(f"{'='*70}")
    
    # 推理模式下 batch 必须为 1
    if not is_training:
        assert B == 1, f"Inference mode requires B=1, got B={B}"
    
    # 验证 q_offset 设置正确（Q 对应 KV 中最新的部分）
    if not is_training and q_len < kv_len:
        expected_q_offset = kv_len - q_len
        assert q_offset == expected_q_offset, f"q_offset should be {expected_q_offset}, got {q_offset}"
    
    torch.manual_seed(42)
    
    # S = kv_len // block_size (向下取整)
    S = kv_len // block_size
    
    # 准备数据
    # Q: [B, q_len, h_kv, G, D]
    q_raw = torch.randn(B, q_len, h_kv, G, D, dtype=dtype, device=device)
    # Lmks: [B, S, h_kv, D]
    lmks_raw = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)
    # LSE SWA: [B, q_len, h_kv, G]
    lse_swa_raw = torch.randn(B, q_len, h_kv, G, dtype=dtype, device=device) * 5 + 10

    # PoPE 参数：freqs 涵盖 q_offset..q_offset+q_len 以及 K 侧 (S-1)*block_size
    freqs_len_total = max(q_offset + q_len, (S - 1) * block_size + 1)
    freqs_raw = torch.randn(freqs_len_total, D, device=device, dtype=torch.float32) * 0.1
    h_q = h_kv * G
    bias_raw = torch.randn(h_q, D, device=device, dtype=torch.float32) * 0.1
    
    # 输入隔离
    q_fused = q_raw.clone().detach().requires_grad_(is_training)  # 推理模式不需要梯度
    lmks_fused = lmks_raw.clone().detach().requires_grad_(is_training)
    bias_fused = bias_raw.clone().detach().requires_grad_(is_training)
    freqs_fused = freqs_raw.clone().detach()
    
    q_ref = q_raw.clone().detach().requires_grad_(is_training)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(is_training)
    bias_ref = bias_raw.clone().detach().requires_grad_(is_training)
    freqs_ref = freqs_raw.clone().detach()
    
    lse_swa = lse_swa_raw.clone().detach()
    
    # ============ 前向计算 ============
    print("\n--- Forward Pass ---")
    
    # Reference
    indices_ref, scores_ref = ref_softmax_topk_max_pooling(
        q_ref, lmks_ref, lse_swa.float(), freqs_ref, bias_ref,
        topk, block_size, window_size, is_causal, q_offset=q_offset
    )
    
    # Fused Kernel
    indices_fused, scores_fused = online_softmax_topk_head_pope(
        q_fused, lmks_fused, lse_swa, freqs_fused, bias_fused,
        topk, block_size, window_size, is_causal,
        q_offset=q_offset, is_training=is_training
    )
    
    # ============ 辅助函数 ============
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
    
    # ============ 验证前向 Scores ============
    # Reshape fused scores to match reference [B, q_len, h_kv, G, topk]
    scores_fused_reshaped = scores_fused.view(B, q_len, h_kv, G, topk)
    
    # 关键：scores_ref 是 ref 按 ref_indices gather 出来的，scores_fused 是 fused 按 fused_indices
    # gather 出来的。当 match rate < 100% 时，两边 gather 的位置不同，直接对比会放大 abs 误差。
    # 正确做法：用 fused_indices 从全量 ref scores 中重新 gather，与 fused scores 在相同 (s) 位置对齐。
    with torch.no_grad():
        sm_scale_fwd = 1.0 / math.sqrt(D)
        scores_all_ref_fwd = _pope_qk_logits(
            q_ref.float(), lmks_ref.float(), freqs_ref.float(), bias_ref.float(),
            block_size=block_size, q_offset=q_offset,
        ) * sm_scale_fwd  # [B, q_len, h_kv, G, S]
        
        if is_causal:
            i_idx_fwd = torch.arange(q_len, device=device).unsqueeze(1) + q_offset
            j_idx_fwd = torch.arange(S, device=device).unsqueeze(0)
            causal_thr_fwd = (i_idx_fwd - window_size + 1).div(block_size, rounding_mode='floor')
            causal_mask_fwd = j_idx_fwd >= causal_thr_fwd
            scores_all_ref_fwd = scores_all_ref_fwd.masked_fill(
                causal_mask_fwd.view(1, q_len, 1, 1, S), float('-inf')
            )
        
        safe_idx_fwd = indices_fused.clone().detach().long()
        invalid_idx_fwd = safe_idx_fwd < 0
        safe_idx_fwd[invalid_idx_fwd] = 0
        scores_ref_aligned = torch.gather(
            scores_all_ref_fwd, -1, safe_idx_fwd.unsqueeze(3).expand(-1, -1, -1, G, -1)
        )
        # 将 fused 标记为无效 (-1) 的位置也设为 -inf，与 fused 的 -inf 对齐后会被 mask 函数过滤
        scores_ref_aligned = scores_ref_aligned.masked_fill(
            invalid_idx_fwd.unsqueeze(3).expand(-1, -1, -1, G, -1), float('-inf')
        )
    
    fwd_abs_err = get_abs_err(scores_ref_aligned.float(), scores_fused_reshaped.float())
    fwd_rel_err = get_err_ratio(scores_ref_aligned.float(), scores_fused_reshaped.float())
    
    print(f"Forward Scores - Abs Error: {fwd_abs_err:.6f}")
    print(f"Forward Scores - Rel Error: {fwd_rel_err:.6f}")
    
    # 验证 indices 匹配率
    indices_match = (indices_fused.long() == indices_ref.long())
    scores_ref_pooled = scores_ref.max(dim=3).values
    valid_mask = scores_ref_pooled > -1e5
    if valid_mask.sum() > 0:
        match_rate = indices_match[valid_mask].float().mean().item()
        print(f"Indices Match Rate: {match_rate*100:.2f}%")
    else:
        match_rate = 1.0
        print("Indices Match Rate: N/A (all masked)")
    
    # 前向断言
    assert fwd_rel_err < 0.02, f"Forward relative error too large: {fwd_rel_err}"
    print("✅ Forward PASSED")
    
    # ============ 反向计算 (仅训练模式) ============
    if is_training:
        print("\n--- Backward Pass ---")
        
        grad_output = torch.randn_like(scores_fused, dtype=dtype)
        
        # Mask 掉无效位置的梯度（基于 fused 的 scores 是否有效）
        with torch.no_grad():
            scores_fused_view = scores_fused.view(B, q_len, h_kv, G, topk)
            invalid_mask_bwd = scores_fused_view.float() < -1e5
            grad_output_view = grad_output.view(B, q_len, h_kv, G, topk)
            grad_output_view[invalid_mask_bwd] = 0.0
        
        # Fused Backward
        scores_fused.backward(grad_output)
        dq_fused = q_fused.grad.clone()
        dlmks_fused = lmks_fused.grad.clone()
        dbias_fused = bias_fused.grad.clone()
        
        # Ref Backward：为了避免因 indices 不对齐（match rate < 100%）放大 dLmks/dBias 误差，
        # 采用"用 fused 的 indices 从全量 ref scores 中 gather，再反传"的方式。
        # 这样 fused 和 ref 在完全相同的 (b, l, h, g, s) 位置上接收梯度，
        # 误差就只来自于数值精度本身，与原版 topk_head_softmax.py 的做法一致。
        sm_scale_ref = 1.0 / math.sqrt(D)
        scores_all_ref_bwd = _pope_qk_logits(
            q_ref.float(), lmks_ref.float(), freqs_ref.float(), bias_ref.float(),
            block_size=block_size, q_offset=q_offset,
        ) * sm_scale_ref  # [B, q_len, h_kv, G, S]
        
        if is_causal:
            i_idx_bwd = torch.arange(q_len, device=device).unsqueeze(1) + q_offset
            j_idx_bwd = torch.arange(S, device=device).unsqueeze(0)
            causal_thr_bwd = (i_idx_bwd - window_size + 1).div(block_size, rounding_mode='floor')
            causal_mask_bwd = j_idx_bwd >= causal_thr_bwd
            scores_all_ref_bwd = scores_all_ref_bwd.masked_fill(
                causal_mask_bwd.view(1, q_len, 1, 1, S), float('-inf')
            )
        
        # 用 fused indices gather，保证与 fused 的反向路径在相同位置对齐
        safe_indices_bwd = indices_fused.clone().detach().long()
        safe_indices_bwd[safe_indices_bwd < 0] = 0
        indices_expanded_bwd = safe_indices_bwd.unsqueeze(3).expand(-1, -1, -1, G, -1)
        scores_gathered_ref_bwd = torch.gather(scores_all_ref_bwd, -1, indices_expanded_bwd)
        
        loss_ref = (scores_gathered_ref_bwd * grad_output.view(B, q_len, h_kv, G, topk).float()).sum()
        loss_ref.backward()
        dq_ref = q_ref.grad.clone()
        dlmks_ref = lmks_ref.grad.clone()
        dbias_ref = bias_ref.grad.clone()
        
        # 处理 NaN
        dq_fused = torch.nan_to_num(dq_fused, 0.0)
        dq_ref = torch.nan_to_num(dq_ref, 0.0)
        dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
        dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)
        dbias_fused = torch.nan_to_num(dbias_fused, 0.0)
        dbias_ref = torch.nan_to_num(dbias_ref, 0.0)
        
        dq_rel_err = get_err_ratio(dq_ref.float(), dq_fused.float())
        dlmks_rel_err = get_err_ratio(dlmks_ref.float(), dlmks_fused.float())
        dbias_rel_err = get_err_ratio(dbias_ref.float(), dbias_fused.float())
        
        print(f"dQ Rel Error: {dq_rel_err:.6f}")
        print(f"dLmks Rel Error: {dlmks_rel_err:.6f}")
        print(f"dBias Rel Error: {dbias_rel_err:.6f}")
        
        assert dq_rel_err < 0.05, f"dQ relative error too large: {dq_rel_err}"
        assert dlmks_rel_err < 0.05, f"dLmks relative error too large: {dlmks_rel_err}"
        assert dbias_rel_err < 0.05, f"dBias relative error too large: {dbias_rel_err}"
        print("✅ Backward PASSED")
    else:
        print("\n--- Backward Pass ---")
        print("⏭️  Skipped (inference mode)")
    
    print(f"\n✅ Test '{test_name}' PASSED")


def test_drop_mask_correctness():
    """
    测试 DropMask 功能的正确性。
    验证：当某些 chunk 被 drop 时，fused kernel 和 ref 实现的结果一致。
    """
    print("\n" + "=" * 70)
    print("=== Testing DropMask Correctness ===")
    print("=" * 70)
    
    B, L, D = 2, 2050, 128
    h_kv = 2
    G = 8
    h_q = h_kv * G
    S = 64
    topk = 8
    is_causal = True
    block_size = 32
    window_size = 32
    
    dtype = torch.bfloat16
    device = "cuda"
    
    print(f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, topk={topk}")
    
    torch.manual_seed(42)
    
    q_raw = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device)
    lmks_raw = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)
    lse_swa_raw = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10

    # PoPE 参数
    freqs_len_total = max(L, (S - 1) * block_size + 1)
    freqs_raw = torch.randn(freqs_len_total, D, device=device, dtype=torch.float32) * 0.1
    bias_raw = torch.randn(h_q, D, device=device, dtype=torch.float32) * 0.1
    
    # 创建 drop_mask: [B, L, S], int32, 随机 drop 约 30% 的 chunks
    drop_mask = (torch.rand(B, L, S, device=device) < 0.3).to(torch.int32)
    
    # 输入隔离
    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)
    bias_fused = bias_raw.clone().detach().requires_grad_(True)
    freqs_fused = freqs_raw.clone().detach()
    q_ref = q_raw.clone().detach().requires_grad_(True)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(True)
    bias_ref = bias_raw.clone().detach().requires_grad_(True)
    freqs_ref = freqs_raw.clone().detach()
    lse_swa = lse_swa_raw.clone().detach()
    
    # ============ Forward ============
    print("\n--- Forward Correctness ---")

    # Reference (带 drop_mask)
    ref_indices, ref_scores = ref_softmax_topk_max_pooling(
        q_ref, lmks_ref, lse_swa.float(), freqs_ref, bias_ref,
        topk, block_size, window_size, is_causal, drop_mask=drop_mask
    )
    
    # Fused Kernel (带 drop_mask)
    fused_indices, fused_scores = online_softmax_topk_head_pope(
        q_fused, lmks_fused, lse_swa, freqs_fused, bias_fused,
        topk, block_size, window_size, is_causal,
        is_training=True, drop_mask=drop_mask
    )
    
    # Reshape fused scores for comparison
    fused_scores_reshaped = fused_scores.view(B, L, h_kv, G, topk)

    def get_err_ratio(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        err = (x[mask] - y[mask]).square().mean().sqrt().item()
        base = (x[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)
    
    # ---- 计算全量 ref scores 用于对齐比较 (PoPE) ----
    scores_all_ref = _pope_qk_logits(
        q_raw.float(), lmks_raw.float(), freqs_raw.float(), bias_raw.float(),
        block_size=block_size, q_offset=0,
    )
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D))
    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        aligned_threshold = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= aligned_threshold
        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))

    # ---- 用 fused indices 从全量 scores 中 gather，验证 score 正确性 ----
    safe_indices = fused_indices.clone()
    safe_indices[safe_indices < 0] = 0
    # fused_indices: [B, L, h_kv, topk] -> expand to [B, L, h_kv, G, topk]
    indices_expanded = safe_indices.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
    # scores_all_ref: [B, L, h_kv, G, S] -> gather on S dim
    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded)

    valid_mask = (scores_gathered_ref > -1e9) & (fused_scores_reshaped.float() > -1e9)
    if valid_mask.sum() == 0:
        max_score_diff = 0.0
        rel_l2_score = 0.0
    else:
        score_diff = torch.abs(scores_gathered_ref[valid_mask] - fused_scores_reshaped.float()[valid_mask])
        max_score_diff = score_diff.max().item()
        rel_l2_score = score_diff.norm().item() / (scores_gathered_ref[valid_mask].norm().item() + 1e-6)

    print(f"Forward scores (aligned by fused indices) - Max Diff: {max_score_diff:.6f}")
    print(f"Forward scores (aligned by fused indices) - L2 RelErr: {rel_l2_score:.6f}")
    
    # ---- 验证 indices 匹配率（仅供参考，head 版本因 LSE 归一化精度差异可能较低） ----
    indices_match = (fused_indices.long() == ref_indices.long())
    scores_ref_pooled = ref_scores.max(dim=3).values
    valid_indices_mask = scores_ref_pooled > -1e5
    if valid_indices_mask.sum() > 0:
        match_rate = indices_match[valid_indices_mask].float().mean().item()
        print(f"Indices Match Rate: {match_rate*100:.2f}% (head 版本因 LSE 归一化精度差异可能 < 100%)")
    else:
        match_rate = 1.0
    
    # ---- 核心验证：被 drop 的 chunk 确实没有被选中 ----
    # fused_indices: [B, L, h_kv, topk]
    # drop_mask: [B, L, S]
    dropped_selected = 0
    total_valid = 0
    for b in range(B):
        for l in range(L):
            for h in range(h_kv):
                for k in range(topk):
                    idx = fused_indices[b, l, h, k].item()
                    if idx >= 0 and idx < S:
                        total_valid += 1
                        if drop_mask[b, l, idx].item() == 1:
                            dropped_selected += 1
    
    if total_valid > 0:
        drop_violation_rate = dropped_selected / total_valid
        print(f"Drop Violation Rate: {drop_violation_rate*100:.4f}% ({dropped_selected}/{total_valid})")
    else:
        drop_violation_rate = 0.0
        print("No valid indices to check")
    
    # 判断条件：score 正确 + drop 排除有效
    if max_score_diff < 1.0 and rel_l2_score < 1e-2 and drop_violation_rate < 0.001:
        print("✅ DropMask Forward PASSED")
    else:
        print("❌ DropMask Forward FAILED")
    
    # ============ Backward ============
    print("\n--- Backward Correctness ---")
    
    grad_output = torch.randn_like(fused_scores, dtype=dtype)
    
    # Mask 掉无效位置的梯度（基于 fused 的 indices）
    with torch.no_grad():
        fused_scores_view = fused_scores.view(B, L, h_kv, G, topk)
        invalid_mask_bwd = fused_scores_view.float() < -1e5
        grad_output_view = grad_output.view(B, L, h_kv, G, topk)
        grad_output_view[invalid_mask_bwd] = 0.0
    
    # Fused Backward
    loss_fused = (fused_scores * grad_output).sum()
    loss_fused.backward()
    dq_fused = q_fused.grad.clone()
    dlmks_fused = lmks_fused.grad.clone()
    dbias_fused = bias_fused.grad.clone()
    
    # Ref Backward: 用 fused indices 从全量 scores 中 gather，确保和 fused 对齐
    scores_all_ref_bwd = _pope_qk_logits(
        q_ref.float(), lmks_ref.float(), freqs_ref.float(), bias_ref.float(),
        block_size=block_size, q_offset=0,
    )
    scores_all_ref_bwd = scores_all_ref_bwd * (1.0 / math.sqrt(D))
    if is_causal:
        scores_all_ref_bwd = scores_all_ref_bwd.masked_fill(causal_mask_expanded, float('-inf'))
    
    safe_indices_bwd = fused_indices.clone().detach()
    safe_indices_bwd[safe_indices_bwd < 0] = 0
    indices_expanded_bwd = safe_indices_bwd.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
    scores_gathered_ref_bwd = torch.gather(scores_all_ref_bwd, -1, indices_expanded_bwd)
    
    loss_ref = (scores_gathered_ref_bwd * grad_output.view(B, L, h_kv, G, topk).float()).sum()
    loss_ref.backward()
    dq_ref = q_ref.grad.clone()
    dlmks_ref = lmks_ref.grad.clone()
    dbias_ref = bias_ref.grad.clone()
    
    dq_fused = torch.nan_to_num(dq_fused, 0.0)
    dq_ref = torch.nan_to_num(dq_ref, 0.0)
    dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
    dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)
    dbias_fused = torch.nan_to_num(dbias_fused, 0.0)
    dbias_ref = torch.nan_to_num(dbias_ref, 0.0)
    
    dq_rel_err = get_err_ratio(dq_ref.float(), dq_fused.float())
    dlmks_rel_err = get_err_ratio(dlmks_ref.float(), dlmks_fused.float())
    dbias_rel_err = get_err_ratio(dbias_ref.float(), dbias_fused.float())
    
    print(f"dQ Rel Error: {dq_rel_err:.6f}")
    print(f"dLmks Rel Error: {dlmks_rel_err:.6f}")
    print(f"dBias Rel Error: {dbias_rel_err:.6f}")
    
    if dq_rel_err < 0.05 and dlmks_rel_err < 0.05 and dbias_rel_err < 0.05:
        print("✅ DropMask Backward PASSED")
    else:
        print("❌ DropMask Backward FAILED")


def test_gqa_d_reshape_correctness():
    """
    测试 GQA 自适应 (D_lmk != D_q) 的正确性。
    当 lmks 的 D 维度是 q 的整数倍时，自动 reshape 为更多的 head。
    """
    print("\n" + "=" * 70)
    print("=== Testing GQA D-Reshape (D_lmk != D_q) Correctness ===")
    print("=" * 70)
    
    B, L = 2, 512
    D_q = 64       # query 的 head dim
    D_lmk = 128    # lmk 的 head dim = D_q * 2
    h_kv_orig = 2  # 原始 h_kv
    G_orig = 4     # 原始 G
    h_q = h_kv_orig * G_orig  # = 8
    S = 16
    topk = 8
    is_causal = True
    block_size = 32
    window_size = 32
    
    dtype = torch.bfloat16
    device = "cuda"
    
    # D_lmk / D_q = 2, 所以 reshape 后 h_kv_new = h_kv_orig * 2 = 4, G_new = h_q / h_kv_new = 2
    d_ratio = D_lmk // D_q
    h_kv_new = h_kv_orig * d_ratio
    G_new = h_q // h_kv_new
    
    print(f"Config: B={B}, L={L}, S={S}")
    print(f"  D_q={D_q}, D_lmk={D_lmk}, d_ratio={d_ratio}")
    print(f"  h_kv_orig={h_kv_orig}, G_orig={G_orig}, h_q={h_q}")
    print(f"  After reshape: h_kv_new={h_kv_new}, G_new={G_new}")
    print(f"  topk={topk}, block_size={block_size}, window_size={window_size}")
    
    torch.manual_seed(42)
    
    # q: [B, L, h_q, D_q] (4维输入，API 内部会 reshape)
    q_raw = torch.randn(B, L, h_q, D_q, dtype=dtype, device=device)
    # lmks: [B, S, h_kv_orig, D_lmk]
    lmks_raw = torch.randn(B, S, h_kv_orig, D_lmk, dtype=dtype, device=device)
    # lse_swa: [B, L, h_q]
    lse_swa_raw = torch.randn(B, L, h_q, dtype=dtype, device=device) * 5 + 10

    # PoPE 参数：reshape 后 head_dim = D_q, bias 是 per-query-head [h_q, D_q]
    freqs_len_total = max(L, (S - 1) * block_size + 1)
    freqs_raw = torch.randn(freqs_len_total, D_q, device=device, dtype=torch.float32) * 0.1
    bias_raw = torch.randn(h_q, D_q, device=device, dtype=torch.float32) * 0.1
    
    # 输入隔离
    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)
    bias_fused = bias_raw.clone().detach().requires_grad_(True)
    freqs_fused = freqs_raw.clone().detach()
    
    q_ref = q_raw.clone().detach()
    lmks_ref = lmks_raw.clone().detach()
    lse_swa = lse_swa_raw.clone().detach()
    
    # ============ Forward ============
    print("\n--- Forward ---")
    
    # Fused Kernel (自动处理 D_lmk != D_q)
    fused_indices, fused_scores = online_softmax_topk_head_pope(
        q_fused, lmks_fused, lse_swa, freqs_fused, bias_fused,
        topk, block_size, window_size, is_causal,
        is_training=True
    )
    
    # 手动 reshape 后用 ref 验证
    # lmks reshape: [B, S, h_kv_orig, D_lmk] -> [B, S, h_kv_new, D_q]
    lmks_reshaped = lmks_ref.reshape(B, S, h_kv_new, D_q)
    # q reshape: [B, L, h_q, D_q] -> [B, L, h_kv_new, G_new, D_q]
    q_reshaped = q_ref.view(B, L, h_kv_new, G_new, D_q)
    # lse_swa reshape: [B, L, h_q] -> [B, L, h_kv_new, G_new]
    lse_swa_reshaped = lse_swa.view(B, L, h_kv_new, G_new)
    
    bias_ref_for_ref = bias_raw.clone().detach()
    freqs_ref_for_ref = freqs_raw.clone().detach()
    ref_indices, ref_scores = ref_softmax_topk_max_pooling(
        q_reshaped, lmks_reshaped, lse_swa_reshaped.float(),
        freqs_ref_for_ref, bias_ref_for_ref,
        topk, block_size, window_size, is_causal
    )
    
    # fused_scores: [B, L, h_q, topk] -> [B, L, h_kv_new, G_new, topk]
    fused_scores_reshaped = fused_scores.view(B, L, h_kv_new, G_new, topk)
    
    def get_err_ratio(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        err = (x[mask] - y[mask]).square().mean().sqrt().item()
        base = (x[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)
    
    fwd_rel_err = get_err_ratio(ref_scores.float(), fused_scores_reshaped.float())
    print(f"Forward Scores Rel Error: {fwd_rel_err:.6f}")
    
    # 验证 indices 匹配率
    indices_match = (fused_indices.long() == ref_indices.long())
    scores_ref_pooled = ref_scores.max(dim=3).values
    valid_mask = scores_ref_pooled > -1e5
    if valid_mask.sum() > 0:
        match_rate = indices_match[valid_mask].float().mean().item()
        print(f"Indices Match Rate: {match_rate*100:.2f}%")
    else:
        match_rate = 1.0
    
    if fwd_rel_err < 0.02 and match_rate >= 0.99:
        print("✅ GQA D-Reshape Forward PASSED")
    else:
        print("❌ GQA D-Reshape Forward FAILED")
    
    # ============ Backward ============
    print("\n--- Backward ---")
    
    grad_output = torch.randn_like(fused_scores, dtype=dtype)
    
    # Mask 掉无效位置的梯度（基于 fused 的 scores）
    with torch.no_grad():
        fused_scores_view = fused_scores.view(B, L, h_kv_new, G_new, topk)
        invalid_mask_bwd = fused_scores_view.float() < -1e5
        grad_output_view = grad_output.view(B, L, h_kv_new, G_new, topk)
        grad_output_view[invalid_mask_bwd] = 0.0
    
    # Fused Backward
    loss_fused = (fused_scores * grad_output).sum()
    loss_fused.backward()
    dq_fused = q_fused.grad.clone()
    dlmks_fused = lmks_fused.grad.clone()
    dbias_fused = bias_fused.grad.clone()
    
    # 验证梯度形状正确
    assert dq_fused.shape == q_raw.shape, f"q grad shape mismatch: {dq_fused.shape} vs {q_raw.shape}"
    assert dlmks_fused.shape == lmks_raw.shape, f"lmks grad shape mismatch: {dlmks_fused.shape} vs {lmks_raw.shape}"
    assert dbias_fused.shape == bias_raw.shape, f"bias grad shape mismatch: {dbias_fused.shape} vs {bias_raw.shape}"
    print(f"q grad shape: {dq_fused.shape} (expected {q_raw.shape})")
    print(f"lmks grad shape: {dlmks_fused.shape} (expected {lmks_raw.shape})")
    print(f"bias grad shape: {dbias_fused.shape} (expected {bias_raw.shape})")
    
    # Ref Backward: 用 fused indices 从全量 scores 中 gather，确保和 fused 对齐
    # 手动 reshape 后构建 ref 的计算图
    q_ref_bwd = q_raw.clone().detach().requires_grad_(True)
    lmks_ref_bwd = lmks_raw.clone().detach().requires_grad_(True)
    bias_ref_bwd = bias_raw.clone().detach().requires_grad_(True)
    freqs_ref_bwd = freqs_raw.clone().detach()
    
    # reshape: [B, L, h_q, D_q] -> [B, L, h_kv_new, G_new, D_q]
    q_reshaped_bwd = q_ref_bwd.view(B, L, h_kv_new, G_new, D_q)
    # reshape: [B, S, h_kv_orig, D_lmk] -> [B, S, h_kv_new, D_q]
    lmks_reshaped_bwd = lmks_ref_bwd.reshape(B, S, h_kv_new, D_q)
    
    # 计算全量 scores (PoPE): [B, L, h_kv_new, G_new, S]
    scores_all_ref = _pope_qk_logits(
        q_reshaped_bwd.float(), lmks_reshaped_bwd.float(),
        freqs_ref_bwd.float(), bias_ref_bwd.float(),
        block_size=block_size, q_offset=0,
    )
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D_q))
    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        aligned_threshold = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= aligned_threshold
        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))
    
    # 用 fused indices gather: fused_indices [B, L, h_kv_new, topk]
    safe_indices_bwd = fused_indices.clone().detach()
    safe_indices_bwd[safe_indices_bwd < 0] = 0
    indices_expanded_bwd = safe_indices_bwd.unsqueeze(3).expand(-1, -1, -1, G_new, -1).long()
    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded_bwd)
    
    # 用相同的 grad_output 计算 ref loss
    loss_ref = (scores_gathered_ref * grad_output.view(B, L, h_kv_new, G_new, topk).float()).sum()
    loss_ref.backward()
    dq_ref = q_ref_bwd.grad.clone()
    dlmks_ref = lmks_ref_bwd.grad.clone()
    dbias_ref = bias_ref_bwd.grad.clone()
    
    # 处理 NaN
    dq_fused = torch.nan_to_num(dq_fused, 0.0)
    dq_ref = torch.nan_to_num(dq_ref, 0.0)
    dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
    dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)
    dbias_fused = torch.nan_to_num(dbias_fused, 0.0)
    dbias_ref = torch.nan_to_num(dbias_ref, 0.0)
    
    # 梯度数值一致性检测
    dq_rel_err = get_err_ratio(dq_ref.float(), dq_fused.float())
    dlmks_rel_err = get_err_ratio(dlmks_ref.float(), dlmks_fused.float())
    dbias_rel_err = get_err_ratio(dbias_ref.float(), dbias_fused.float())
    
    print(f"dQ Rel Error: {dq_rel_err:.6f}")
    print(f"dLmks Rel Error: {dlmks_rel_err:.6f}")
    print(f"dBias Rel Error: {dbias_rel_err:.6f}")
    
    if dq_rel_err < 0.05 and dlmks_rel_err < 0.05 and dbias_rel_err < 0.05:
        print("✅ GQA D-Reshape Backward PASSED")
    else:
        print("❌ GQA D-Reshape Backward FAILED")


def test_prerotate_correctness(
    B=2, L=4096, h_kv=8, G=2, D=128, S=64,
    topk=16, block_size=64, window_size=512,
    rotate_dim=64, is_causal=True,
):
    """
    Validate online_softmax_topk_head_pope_prerotate (PyTorch concat + non-PoPE
    topk kernel + sm_scale=1/sqrt(D)) against:
      1) the fused PoPE kernel online_softmax_topk_head_pope (numerical reference),
      2) the math reference _pope_qk_logits + topk-max-pooling (semantic reference).
    """
    print("\n" + "=" * 70)
    print(f"[Prerotate Wrapper Correctness] B={B}, L={L}, h_kv={h_kv}, G={G}, "
          f"D={D}, S={S}, topk={topk}, rotate_dim={rotate_dim}")
    print("=" * 70)

    device = "cuda"
    dtype = torch.bfloat16
    h_q = h_kv * G

    torch.manual_seed(42)
    q_orig = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device) * 0.5
    lmks_orig = torch.randn(B, S, h_kv, D, dtype=dtype, device=device) * 0.5
    lse_swa = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10
    freqs_len_total = max(L, (S - 1) * block_size + 1)
    freqs_orig = torch.randn(freqs_len_total, rotate_dim, dtype=dtype, device=device) * 0.1
    bias_orig = torch.randn(h_q, rotate_dim, dtype=dtype, device=device) * 0.1

    # ---- Forward: fused PoPE ----
    q_f = q_orig.detach().clone().requires_grad_(True)
    lmks_f = lmks_orig.detach().clone().requires_grad_(True)
    bias_f = bias_orig.detach().clone().requires_grad_(True)
    freqs_f = freqs_orig.detach().clone()

    idx_fused, scores_fused = online_softmax_topk_head_pope(
        q_f, lmks_f, lse_swa, freqs_f, bias_f,
        topk, block_size, window_size, is_causal,
    )

    # ---- Forward: prerotate wrapper ----
    q_w = q_orig.detach().clone().requires_grad_(True)
    lmks_w = lmks_orig.detach().clone().requires_grad_(True)
    bias_w = bias_orig.detach().clone().requires_grad_(True)
    freqs_w = freqs_orig.detach().clone()

    idx_wrap, scores_wrap = online_softmax_topk_head_pope_prerotate(
        q_w, lmks_w, lse_swa, freqs_w, bias_w,
        topk, block_size, window_size, is_causal,
    )

    # ---- Indices match rate (wrapper vs fused) ----
    indices_match = (idx_wrap.long() == idx_fused.long())
    valid_mask = (idx_fused >= 0)
    if valid_mask.sum() == 0:
        match_rate = 1.0
    else:
        match_rate = indices_match[valid_mask].float().mean().item()
    print(f"Indices match rate (wrapper vs fused, valid only): {match_rate*100:.4f}%")

    # ---- Score diff on common (valid) positions ----
    s_fused = scores_fused.float()
    s_wrap = scores_wrap.float()
    finite_mask = torch.isfinite(s_fused) & torch.isfinite(s_wrap)
    if finite_mask.sum() > 0:
        diff = (s_fused[finite_mask] - s_wrap[finite_mask]).abs()
        print(f"Score (wrapper vs fused) max diff (finite only): {diff.max().item():.6e}")
        print(f"Score (wrapper vs fused) L2 rel-err            : "
              f"{diff.norm().item() / (s_fused[finite_mask].norm().item() + 1e-6):.6e}")
    else:
        print("Score (wrapper vs fused): no finite overlap")

    # ---- Compare BOTH fused and wrapper to the math reference _pope_qk_logits.
    # This tells us each path's absolute precision so we can isolate where the
    # wrapper diverges (vs fused vs the true math).
    with torch.no_grad():
        q_ref = q_orig.detach().float()
        lmks_ref = lmks_orig.detach().float()
        freqs_ref = freqs_orig.detach().float()
        bias_ref = bias_orig.detach().float().view(h_kv, G, rotate_dim)
        logits_ref = _pope_qk_logits(
            q_ref, lmks_ref, freqs_ref, bias_ref,
            block_size=block_size, q_offset=0, rotate_dim=rotate_dim,
        )                                                      # [B, L, h_kv, G, S]
        logits_ref = logits_ref * (1.0 / math.sqrt(D))

        if is_causal:
            i_idx = torch.arange(L, device=device).unsqueeze(1)
            j_idx = torch.arange(S, device=device).unsqueeze(0)
            aligned_threshold = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
            causal_mask = j_idx >= aligned_threshold
            logits_ref = logits_ref.masked_fill(causal_mask.view(1, L, 1, 1, S), float('-inf'))

        def _gather_ref(idx_path):
            safe_idx = idx_path.clone()
            safe_idx[safe_idx < 0] = 0
            ie = safe_idx.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
            return torch.gather(logits_ref, -1, ie)            # [B, L, h_kv, G, topk]

        ref_for_fused = _gather_ref(idx_fused)
        ref_for_wrap = _gather_ref(idx_wrap)

        fused_view = scores_fused.float().view(B, L, h_kv, G, topk)
        wrap_view = scores_wrap.float().view(B, L, h_kv, G, topk)

        m1 = (ref_for_fused > -1e9) & (fused_view > -1e9)
        m2 = (ref_for_wrap > -1e9) & (wrap_view > -1e9)
        if m1.sum() > 0:
            d1 = (ref_for_fused[m1] - fused_view[m1]).abs()
            print(f"Score (fused   vs ref) max diff: {d1.max().item():.6e}, "
                  f"L2 rel-err: {d1.norm().item() / (ref_for_fused[m1].norm().item() + 1e-6):.6e}")
        if m2.sum() > 0:
            d2 = (ref_for_wrap[m2] - wrap_view[m2]).abs()
            print(f"Score (wrapper vs ref) max diff: {d2.max().item():.6e}, "
                  f"L2 rel-err: {d2.norm().item() / (ref_for_wrap[m2].norm().item() + 1e-6):.6e}")

    # ---- Backward gradient compare ----
    grad_output = torch.randn(B, L, h_q, topk, dtype=dtype, device=device)
    # Mask invalid positions in grad_output (selected indices == -1)
    invalid_grad_mask_4d = (idx_fused < 0).unsqueeze(3).expand(-1, -1, -1, G, -1)
    invalid_grad_mask_4d = invalid_grad_mask_4d.reshape(B, L, h_q, topk)
    grad_output_masked = grad_output.clone()
    grad_output_masked[invalid_grad_mask_4d] = 0.0

    loss_fused = (scores_fused * grad_output_masked).sum()
    loss_fused.backward()
    grad_q_f = q_f.grad.clone()
    grad_lmks_f = lmks_f.grad.clone()
    grad_bias_f = bias_f.grad.clone()

    loss_wrap = (scores_wrap * grad_output_masked).sum()
    loss_wrap.backward()
    grad_q_w = q_w.grad.clone()
    grad_lmks_w = lmks_w.grad.clone()
    grad_bias_w = bias_w.grad.clone()

    def _safe_nan(g):
        return torch.nan_to_num(g, 0.0)

    grad_q_f = _safe_nan(grad_q_f)
    grad_lmks_f = _safe_nan(grad_lmks_f)
    grad_bias_f = _safe_nan(grad_bias_f)
    grad_q_w = _safe_nan(grad_q_w)
    grad_lmks_w = _safe_nan(grad_lmks_w)
    grad_bias_w = _safe_nan(grad_bias_w)

    def _rel_err(a, b):
        diff = (a.float() - b.float()).abs()
        return diff.max().item(), diff.norm().item() / (b.float().norm().item() + 1e-6)

    mq, rq = _rel_err(grad_q_w, grad_q_f)
    mk, rk = _rel_err(grad_lmks_w, grad_lmks_f)
    mb, rb = _rel_err(grad_bias_w, grad_bias_f)
    print(f"Wrapper vs Fused - grad_q     Max={mq:.6e}, L2Rel={rq*100:.4f}%")
    print(f"Wrapper vs Fused - grad_lmks  Max={mk:.6e}, L2Rel={rk*100:.4f}%")
    print(f"Wrapper vs Fused - grad_bias  Max={mb:.6e}, L2Rel={rb*100:.4f}%")

    ok = (match_rate >= 0.99 and rq < 0.05 and rk < 0.05 and rb < 0.05)
    print("Result:", "PASSED" if ok else "FAILED (check tolerances above)")


def test_prerotate_latency(
    B=4, L=8192, h_kv=32, G=1, D=128, S=128,
    topk=32, block_size=64, window_size=512,
    rotate_dim=64, is_causal=True, n_iters=20, n_warmup=20,
):
    """
    Compare end-to-end FWD / FWD+BWD latency of:
      1) Fused PoPE kernel:    online_softmax_topk_head_pope
      2) Prerotate wrapper:    online_softmax_topk_head_pope_prerotate
    Also reports the standalone PyTorch-prerotate cost so we can split the
    wrapper FWD into prerotate vs base-kernel time.
    """
    print("\n" + "=" * 70)
    print(f"[Prerotate Wrapper Latency] B={B}, L={L}, h_kv={h_kv}, G={G}, D={D}, "
          f"S={S}, topk={topk}, rotate_dim={rotate_dim}")
    print("=" * 70)

    device = "cuda"
    dtype = torch.bfloat16
    h_q = h_kv * G
    torch.manual_seed(0)

    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)
    lse_swa = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10
    freqs_len_total = max(L, (S - 1) * block_size + 1)
    freqs = torch.randn(freqs_len_total, rotate_dim, dtype=dtype, device=device) * 0.1
    bias = torch.randn(h_q, rotate_dim, dtype=dtype, device=device) * 0.1
    grad_output = torch.randn(B, L, h_q, topk, dtype=dtype, device=device)

    def _bench(name, fn):
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)
        bias_t = bias.detach().clone().requires_grad_(True)
        freqs_t = freqs.detach().clone()

        # Warmup (covers TileLang JIT cold compile and CUDA stream warmup).
        for _ in range(n_warmup):
            q_t.grad = None
            lmks_t.grad = None
            bias_t.grad = None
            _ = fn(q_t, lmks_t, lse_swa, freqs_t, bias_t)
            _, scores = fn(q_t, lmks_t, lse_swa, freqs_t, bias_t)
            (scores * grad_output).sum().backward()
        torch.cuda.synchronize()

        # FWD-only
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e_start.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            bias_t.grad = None
            _ = fn(q_t, lmks_t, lse_swa, freqs_t, bias_t)
        e_end.record()
        torch.cuda.synchronize()
        fwd_ms = e_start.elapsed_time(e_end) / n_iters

        # FWD+BWD
        torch.cuda.synchronize()
        e_start.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            bias_t.grad = None
            _, scores = fn(q_t, lmks_t, lse_swa, freqs_t, bias_t)
            (scores * grad_output).sum().backward()
        e_end.record()
        torch.cuda.synchronize()
        fb_ms = e_start.elapsed_time(e_end) / n_iters
        bwd_ms = fb_ms - fwd_ms
        print(f"  {name:<28} FWD={fwd_ms:8.3f}  BWD={bwd_ms:8.3f}  TOTAL={fb_ms:8.3f} ms")
        return dict(name=name, fwd=fwd_ms, bwd=bwd_ms, total=fb_ms)

    def _fused_call(q_, lmks_, lse_, freqs_, bias_):
        return online_softmax_topk_head_pope(
            q_, lmks_, lse_, freqs_, bias_,
            topk, block_size, window_size, is_causal,
        )

    def _wrap_call(q_, lmks_, lse_, freqs_, bias_):
        return online_softmax_topk_head_pope_prerotate(
            q_, lmks_, lse_, freqs_, bias_,
            topk, block_size, window_size, is_causal,
        )

    # Standalone benchmark: PyTorch prerotate piece only (so we can attribute
    # wrapper FWD = prerotate_ms + base_kernel_ms).
    R = rotate_dim

    def _bench_prerotate_only():
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)
        bias_t = bias.detach().clone().requires_grad_(True)
        freqs_t = freqs.detach().clone()

        def _prerotate(q_in, lmks_in, freqs_in, bias_in):
            qk_dtype = q_in.dtype
            q_pos = torch.arange(L, device=device, dtype=torch.long)
            fq = freqs_in.index_select(0, q_pos)
            theta_q = fq.view(1, L, 1, 1, R) - bias_in.view(1, 1, h_kv, G, R)
            cos_q = torch.cos(theta_q.float()).to(qk_dtype)
            sin_q = torch.sin(theta_q.float()).to(qk_dtype)

            q_rot_f = q_in[..., :R].float()
            qc_rot = (q_rot_f * cos_q.float()).to(qk_dtype)
            qs = (q_rot_f * sin_q.float()).to(qk_dtype)
            if R < D:
                qc = torch.cat([qc_rot, q_in[..., R:]], dim=-1)
            else:
                qc = qc_rot
            q_eff = torch.cat([qc, qs], dim=-1)

            k_pos = torch.arange(S, device=device, dtype=torch.long) * block_size + (block_size - 1)
            fk = freqs_in.index_select(0, k_pos)
            cos_k = torch.cos(fk.float()).to(qk_dtype).view(1, S, 1, R)
            sin_k = torch.sin(fk.float()).to(qk_dtype).view(1, S, 1, R)
            k_rot_f = lmks_in[..., :R].float()
            kc_rot = (k_rot_f * cos_k.float()).to(qk_dtype)
            ks = (k_rot_f * sin_k.float()).to(qk_dtype)
            if R < D:
                kc = torch.cat([kc_rot, lmks_in[..., R:]], dim=-1)
            else:
                kc = kc_rot
            k_eff = torch.cat([kc, ks], dim=-1)
            return q_eff, k_eff

        # Warmup
        for _ in range(n_warmup):
            _ = _prerotate(q_t, lmks_t, freqs_t, bias_t.view(h_kv, G, R))
        torch.cuda.synchronize()

        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        e_start.record()
        for _ in range(n_iters):
            _ = _prerotate(q_t, lmks_t, freqs_t, bias_t.view(h_kv, G, R))
        e_end.record()
        torch.cuda.synchronize()
        return e_start.elapsed_time(e_end) / n_iters

    print("Latency (lower is better, all numbers averaged over n_iters):")
    r_fused = _bench("Fused PoPE", _fused_call)
    r_wrap = _bench("Prerotate wrapper", _wrap_call)

    prerot_ms = _bench_prerotate_only()
    base_ms = max(r_wrap["fwd"] - prerot_ms, 0.0)
    print(f"  PyTorch prerotate only       FWD={prerot_ms:8.3f} ms  "
          f"(==> base non-PoPE kernel ~ {base_ms:.3f} ms inside wrapper FWD)")

    print(f"  Wrapper / Fused (TOTAL):     {r_wrap['total'] / max(r_fused['total'], 1e-6):.3f}x  "
          f"(>1 means wrapper slower)")


if __name__ == "__main__":
    test_fused_topk_softmax_max_pooling_correctness()
    test_fused_softmax_topk_max_pooling_memory_and_speed()

    # Wrapper correctness + latency: PyTorch concat + non-PoPE topk kernel
    # test_prerotate_correctness()
    # test_prerotate_latency()
    
    # # 运行训练和推理正确性测试
    # print("\n" + "=" * 70)
    # print("Running Train/Inference Correctness Tests")
    # print("=" * 70)
    
    # test_cases = [# test_name, B, q_len, kv_len, h_kv, G, D, topk, block_size, window_size, is_training, q_offset
    #     # 训练场景
    #             ("train_basic", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0),
    #             ("train_large_batch", 4, 2048, 2048, 2, 8, 128, 16, 64, 64, True, 0),
    #             ("train_non_divisible", 2, 999, 999, 2, 8, 128, 16, 64, 64, True, 0),
                
    #             # 推理场景
    #             ("prefill_chunk1", 1, 1024, 1024, 2, 8, 128, 16, 64, 64, False, 0),
    #             ("prefill_chunk2", 1, 1024, 2048, 2, 8, 128, 16, 64, 64, False, 1024),
    #             ("prefill_chunk3", 1, 1024, 3072, 2, 8, 128, 16, 64, 64, False, 2048),
    #             ("prefill_non_divisible", 1, 512, 1536, 2, 8, 128, 16, 64, 64, False, 1024),
    #             ("decode_step1", 1, 1, 1025, 2, 8, 128, 16, 64, 64, False, 1024),
    #             ("decode_step2", 1, 1, 2049, 2, 8, 128, 16, 64, 64, False, 2048),
    #             ("edge_small_topk", 1, 512, 1024, 2, 8, 128, 4, 64, 64, False, 512),
    #             ("edge_large_window", 1, 1024, 2048, 2, 8, 128, 16, 64, 128, False, 1024),
    # ]
    
    # for params in test_cases:
    #     test_train_inference_correctness(*params)
    

    # params_list = [
    #     # (B, L, S, h_kv, G, D, topk, block_size, window_size, rotate_dim)
    #     # rotate_dim == D
    #     (2, 4096, 64, 2, 8, 128, 16, 64, 64, None),
    #     (1, 2048, 64, 1, 8, 128, 16, 32, 40, None),
    #     (3, 2048, 64, 1, 8, 128, 8, 32, 33, None),
    #     (2, 4096, 64, 1, 8, 128, 32, 64, 100, None),
    #     # rotate_dim < D
    #     (2, 4096, 64, 2, 8, 128, 16, 64, 64, 64),
    #     (2, 4096, 64, 2, 8, 128, 16, 64, 64, 32),
    #     (1, 2048, 64, 1, 8, 128, 16, 32, 40, 64),
    #     (3, 2050, 64, 1, 8, 128, 8, 32, 33, 32),
    # ]
    # for p in params_list:
    #     test_topk_correctness_robust(*p)

    # # 运行 DropMask 正确性测试
    # test_drop_mask_correctness()
    
    # # 运行 GQA D-Reshape 正确性测试
    # test_gqa_d_reshape_correctness()


# pkill -f "burner.*--gpu 7"; python ops/topk_head_softmax_pope.py
