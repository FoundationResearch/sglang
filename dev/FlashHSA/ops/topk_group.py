
import torch
import tilelang
import tilelang.language as T
from typing import Optional
import math




# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [32,64,128]
# BLOCK_S = [16,32,64,128]
# threads = [32,64,128,256,512]
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
    out_idx=[2, 3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_topk_forward_kernel_insert(
    batch, seq_len, s_len, heads, head_dim, topk,
    block_size, window_size,is_causal,
    BLOCK_L=None, BLOCK_S=None, threads=None
):
    """
    Q: [B, L, H, D]
    K: [B, S, H, D]
    OutScores:  [B, L, H, topk] (fp32)
    OutIndices: [B, L, H, topk] (int32)

    Causal规则（与 topk_head_softmax.py 对齐）:
      令 block_q = tq // block_size
      只允许 ts < block_q
      屏蔽 ts >= block_q
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    q_shape = [batch, seq_len, heads, head_dim]
    k_shape = [batch, s_len, heads, head_dim]
    out_scores_shape = [batch, seq_len, heads, topk]
    out_indices_shape = [batch, seq_len, heads, topk]

    if BLOCK_L is None:
        BLOCK_L = 64
    if BLOCK_S is None:
        BLOCK_S = 16
    BLOCK_D = head_dim
    if threads is None:
        threads = BLOCK_L

    num_s_blocks = tilelang.cdiv(s_len, BLOCK_S)
    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def fwd_kernel_insert(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), heads, batch, threads=threads) as (bx, by, bz):
            b = bz
            h = by
            base_l = bx * BLOCK_L

            Q_shared = T.alloc_shared([BLOCK_L, BLOCK_D], dtype)
            K_shared = T.alloc_shared([BLOCK_S, BLOCK_D], dtype)
            score_shared = T.alloc_shared([BLOCK_L, BLOCK_S], accum_dtype)
            acc_s = T.alloc_fragment([BLOCK_L, BLOCK_S], accum_dtype)

            topk_scores = T.alloc_local([topk], accum_dtype)
            topk_indices = T.alloc_local([topk], idx_dtype)

            # init
            for kk in T.serial(topk):
                topk_scores[kk] = -T.infinity(accum_dtype)
                topk_indices[kk] = -1

            # load Q block
            for l_idx, d in T.Parallel(BLOCK_L, BLOCK_D):
                tq = base_l + l_idx
                if tq < seq_len:
                    Q_shared[l_idx, d] = Q[b, tq, h, d]
                else:
                    Q_shared[l_idx, d] = T.Cast(dtype, 0.0)

            loop_limit = num_s_blocks
            if is_causal:
                tq_max = T.min(seq_len - 1, base_l + (BLOCK_L - 1))
                block_q_max = tq_max // block_size  # int
                loop_limit = T.min(loop_limit, tilelang.cdiv(block_q_max, BLOCK_S))

            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S

                for s_idx, d in T.Parallel(BLOCK_S, BLOCK_D):
                    ts = base_s + s_idx
                    if ts < s_len:
                        K_shared[s_idx, d] = K[b, ts, h, d]
                    else:
                        K_shared[s_idx, d] = T.Cast(dtype, 0.0)
                T.sync_threads()

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_s, score_shared)
                T.sync_threads()

                tx = T.get_thread_binding()
                my_l = tx
                tq = base_l + my_l

                if (my_l < BLOCK_L) and (tq < seq_len):
                    # block_q = tq // block_size
                    limit_chunk = (tq - window_size + 1) // block_size
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        if ts < s_len:
                            # if (not is_causal) or (ts < block_q):
                            if (not is_causal) or (ts < limit_chunk):
                                cur = score_shared[my_l, s_idx] * sm_scale

                                if cur > topk_scores[topk - 1]:
                                    moving = T.alloc_var("bool")
                                    moving = True
                                    for rkk in T.serial(topk):
                                        kpos = topk - 1 - rkk
                                        if moving:
                                            if (kpos > 0) and (cur > topk_scores[kpos - 1]):
                                                topk_scores[kpos] = topk_scores[kpos - 1]
                                                topk_indices[kpos] = topk_indices[kpos - 1]
                                            else:
                                                topk_scores[kpos] = cur
                                                topk_indices[kpos] = ts
                                                moving = False
                T.sync_threads()

            tx2 = T.get_thread_binding()
            tq2 = base_l + tx2
            if (tx2 < BLOCK_L) and (tq2 < seq_len):
                for kk in T.serial(topk):
                    OutScores[b, tq2, h, kk] = topk_scores[kk]
                    OutIndices[b, tq2, h, kk] = topk_indices[kk]

    return fwd_kernel_insert
  





# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [32,64]
# BLOCK_S = [32,64,128]
# _configs = list(
#     itertools.product(
#         BLOCK_L,
#         BLOCK_S,
#     ))

# configs = [
#     {
#         "BLOCK_L": c[0],
#         "BLOCK_S": c[1],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_topk_forward_kernel_heap(
    batch, seq_len, s_len, heads, head_dim, topk,
    BLOCK_L=None, BLOCK_S=None
):
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"
    
    q_shape = [batch, seq_len, heads, head_dim]
    k_shape = [batch, s_len, heads, head_dim]
    out_scores_shape = [batch, seq_len, heads, topk]
    out_indices_shape = [batch, seq_len, heads, topk]

    if BLOCK_L is None:
        BLOCK_L = 64  # {'BLOCK_L': 64, 'BLOCK_S': 64}
    if BLOCK_S is None:
        BLOCK_S = 64
    BLOCK_D = head_dim
    threads = BLOCK_L
    
    num_s_blocks = tilelang.cdiv(s_len, BLOCK_S)
    max_sift_iters = int(math.ceil(math.log2(topk)))
    
    @T.prim_func
    def fwd_kernel_heap(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), heads, batch, threads=threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L

            tx = T.get_thread_binding()
            my_l_idx = tx
            my_tq = base_l + my_l_idx
            
            Q_shared = T.alloc_shared([BLOCK_L, BLOCK_D], dtype)
            K_shared = T.alloc_shared([BLOCK_S, BLOCK_D], dtype)
            score_shared = T.alloc_shared([BLOCK_L, BLOCK_S], accum_dtype)
            
            topk_scores = T.alloc_local([topk], accum_dtype)
            topk_indices = T.alloc_local([topk], idx_dtype)
            
            acc_s = T.alloc_fragment([BLOCK_L, BLOCK_S], accum_dtype)
            
            new_score = T.alloc_var(accum_dtype)
            new_idx = T.alloc_var(idx_dtype)
            old_idx = T.alloc_var(idx_dtype)
            temp_idx = T.alloc_var(idx_dtype)

            for k_idx in T.serial(topk):
                topk_scores[k_idx] = -T.infinity(accum_dtype)
                topk_indices[k_idx] = -1
            
            T.fill(Q_shared, 0)
            for l_idx in T.serial(BLOCK_L):
                tq = base_l + l_idx
                if tq < seq_len:
                    for d in T.Parallel(BLOCK_D):
                        Q_shared[l_idx, d] = Q[i_b, tq, i_h, d]
            
            for s_block in T.serial(num_s_blocks):
                base_s = s_block * BLOCK_S
                
                T.fill(K_shared, 0)
                for s_idx in T.serial(BLOCK_S):
                    ts = base_s + s_idx
                    if ts < s_len:
                        for d in T.Parallel(BLOCK_D):
                            K_shared[s_idx, d] = K[i_b, ts, i_h, d]

                T.sync_threads()
                
                T.clear(acc_s)
                T.gemm(
                    Q_shared, K_shared, acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow
                )
                T.copy(acc_s, score_shared)
                
                T.sync_threads()
                
                if my_tq < seq_len:
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        if ts < s_len:
                            cur_score = score_shared[my_l_idx, s_idx]
                            
                            if cur_score > topk_scores[0]:
                                topk_scores[0] = cur_score
                                topk_indices[0] = ts
                                
                                new_idx = 0
                                for ki in T.serial(max_sift_iters):
                                    old_idx = new_idx
                                    
                                    if new_idx * 2 + 1 < topk and \
                                       topk_scores[new_idx * 2 + 1] < topk_scores[old_idx]:
                                        old_idx = new_idx * 2 + 1
                                    
                                    if new_idx * 2 + 2 < topk and \
                                       topk_scores[new_idx * 2 + 2] < topk_scores[old_idx]:
                                        old_idx = new_idx * 2 + 2
                                        
                                    if old_idx == new_idx:
                                        T.loop_break()
                                    
                                    new_score = topk_scores[new_idx]
                                    temp_idx = topk_indices[new_idx]
                                    topk_scores[new_idx] = topk_scores[old_idx]
                                    topk_indices[new_idx] = topk_indices[old_idx]
                                    topk_scores[old_idx] = new_score
                                    topk_indices[old_idx] = temp_idx
                                    new_idx = old_idx
                                        
                T.sync_threads()
            
            if my_tq < seq_len:
                for k in T.serial(topk):
                    OutScores[i_b, my_tq, i_h, k] = topk_scores[k]
                    OutIndices[i_b, my_tq, i_h, k] = topk_indices[k]

    return fwd_kernel_heap
                        



import torch
import tilelang
import tilelang.language as T
from tvm import DataType

@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_topk_forward_kernel_rtopk(
    batch: int,
    seq_len: int,
    s_len: int,
    heads: int,
    head_dim: int,
    topk: int,
    BLOCK_L: int = 16,
    BLOCK_S: int = 32,
    num_threads: int = 32,
    max_iter: int = 32,
    precision: float = 1e-6,
):

    FP32 = "float32"
    INT32 = "int32"
    dtype = "bfloat16"
    
    q_shape = [batch, seq_len, heads, head_dim]
    k_shape = [batch, s_len, heads, head_dim]
    out_scores_shape = [batch, seq_len, heads, topk]
    out_indices_shape = [batch, seq_len, heads, topk]
    
    # 候选池大小
    cand_size = topk + BLOCK_S
    cand_dim_len = tilelang.cdiv(cand_size, 32)
    
    # K 的分块数
    num_k_blocks = tilelang.cdiv(s_len, BLOCK_S)
    num_l_blocks = tilelang.cdiv(seq_len, BLOCK_L)
    
    NEG_INF_REPLACEMENT = -1e30
    sm_scale = 1.0 / math.sqrt(head_dim)
    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        OutScores: T.Tensor(out_scores_shape, FP32),
        OutIndices: T.Tensor(out_indices_shape, INT32),
    ):
        with T.Kernel(num_l_blocks, heads, batch, threads=num_threads) as (bx, by, bz):
            b_idx = bz
            h_idx = by
            l_base = bx * BLOCK_L
            
            tx = T.get_thread_binding()
            lane_id = tx % 32
            
            Q_shared = T.alloc_shared([BLOCK_L, head_dim], dtype)
            
            K_shared = T.alloc_shared([BLOCK_S, head_dim], dtype)
            
            scores_shared = T.alloc_shared([BLOCK_L, BLOCK_S], FP32)
            
            scores_frag = T.alloc_fragment([BLOCK_L, BLOCK_S], FP32)
            
            topk_scores_all = T.alloc_shared([BLOCK_L, topk], FP32)
            topk_indices_all = T.alloc_shared([BLOCK_L, topk], INT32)
            
            cand_scores = T.alloc_shared([cand_size], FP32)
            cand_indices = T.alloc_shared([cand_size], INT32)
            
            max_val = T.alloc_shared([1], FP32)
            min_val = T.alloc_shared([1], FP32)
            mid_val = T.alloc_shared([1], FP32)
            close_flag = T.alloc_shared([1], INT32)
            
            for li, d in T.Parallel(BLOCK_L, head_dim):
                q_idx = l_base + li
                if q_idx < seq_len:
                    Q_shared[li, d] = Q[b_idx, q_idx, h_idx, d]
                else:
                    Q_shared[li, d] = T.Cast(dtype, 0.0)
            
            for li, ki in T.Parallel(BLOCK_L, topk):
                topk_scores_all[li, ki] = NEG_INF_REPLACEMENT
                topk_indices_all[li, ki] = -1
            
            T.sync_threads()
            
            for k_block in T.serial(num_k_blocks):
                k_start = k_block * BLOCK_S
                
                for si, d in T.Parallel(BLOCK_S, head_dim):
                    k_pos = k_start + si
                    if k_pos < s_len:
                        K_shared[si, d] = K[b_idx, k_pos, h_idx, d]
                    else:
                        K_shared[si, d] = T.Cast(dtype, 0.0)
                
                T.sync_threads()
                
                T.clear(scores_frag)
                T.gemm(Q_shared, K_shared, scores_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(scores_frag, scores_shared)
                
                T.sync_threads()
                
                for li, si in T.Parallel(BLOCK_L, BLOCK_S):
                    q_idx = l_base + li
                    k_pos = k_start + si
                    if q_idx >= seq_len or k_pos >= s_len:
                        scores_shared[li, si] = NEG_INF_REPLACEMENT
                
                T.sync_threads()
                
                for l_offset in T.serial(BLOCK_L):
                    q_idx = l_base + l_offset
                    
                    if q_idx < seq_len:
                        if tx < topk:
                            cand_scores[tx] = topk_scores_all[l_offset, tx]
                            cand_indices[tx] = topk_indices_all[l_offset, tx]
                        
                        if tx < BLOCK_S:
                            cand_scores[topk + tx] = scores_shared[l_offset, tx] * sm_scale
                            cand_indices[topk + tx] = k_start + tx
                        
                        T.sync_threads()
                        
                        if tx < 32:
                            local_max = T.alloc_var(FP32)
                            local_min = T.alloc_var(FP32)
                            local_max = NEG_INF_REPLACEMENT
                            local_min = -NEG_INF_REPLACEMENT
                            
                            for ext in T.serial(cand_dim_len):
                                idx = lane_id + ext * 32
                                if idx < cand_size:
                                    val = cand_scores[idx]
                                    if val > NEG_INF_REPLACEMENT + 1e20:
                                        if val > local_max:
                                            local_max = val
                                        if val < local_min:
                                            local_min = val
                            
                            other_val = T.alloc_var(FP32)
                            
                            other_val = T.shfl_down(local_max, 16)
                            if other_val > local_max:
                                local_max = other_val
                            other_val = T.shfl_down(local_min, 16)
                            if other_val < local_min:
                                local_min = other_val
                            
                            other_val = T.shfl_down(local_max, 8)
                            if other_val > local_max:
                                local_max = other_val
                            other_val = T.shfl_down(local_min, 8)
                            if other_val < local_min:
                                local_min = other_val
                            
                            other_val = T.shfl_down(local_max, 4)
                            if other_val > local_max:
                                local_max = other_val
                            other_val = T.shfl_down(local_min, 4)
                            if other_val < local_min:
                                local_min = other_val
                            
                            other_val = T.shfl_down(local_max, 2)
                            if other_val > local_max:
                                local_max = other_val
                            other_val = T.shfl_down(local_min, 2)
                            if other_val < local_min:
                                local_min = other_val
                            
                            other_val = T.shfl_down(local_max, 1)
                            if other_val > local_max:
                                local_max = other_val
                            other_val = T.shfl_down(local_min, 1)
                            if other_val < local_min:
                                local_min = other_val
                            
                            if lane_id == 0:
                                if local_min > 1e29:
                                    local_min = local_max
                                max_val[0] = local_max
                                min_val[0] = local_min
                                mid_val[0] = (local_max + local_min) * 0.5
                                close_flag[0] = 0
                        
                        T.sync_threads()
                        
                        for iter_idx in T.serial(max_iter):
                            if tx < 32:
                                local_count = T.alloc_var(INT32)
                                local_count = 0
                                
                                cur_mid = mid_val[0]
                                
                                for ext in T.serial(cand_dim_len):
                                    idx = lane_id + ext * 32
                                    if idx < cand_size:
                                        val = cand_scores[idx]
                                        if val > NEG_INF_REPLACEMENT + 1e20 and val >= cur_mid:
                                            local_count += 1
                                
                                other_count = T.alloc_var(INT32)
                                
                                other_count = T.shfl_down(local_count, 16)
                                local_count += other_count
                                other_count = T.shfl_down(local_count, 8)
                                local_count += other_count
                                other_count = T.shfl_down(local_count, 4)
                                local_count += other_count
                                other_count = T.shfl_down(local_count, 2)
                                local_count += other_count
                                other_count = T.shfl_down(local_count, 1)
                                local_count += other_count
                                
                                if lane_id == 0:
                                    count = local_count
                                    cur_max = max_val[0]
                                    cur_min = min_val[0]
                                    
                                    if close_flag[0] == 0:
                                        if count < topk:
                                            max_val[0] = cur_mid
                                        elif count > topk:
                                            min_val[0] = cur_mid
                                        else:
                                            close_flag[0] = 1
                                        
                                        new_max = max_val[0]
                                        new_min = min_val[0]
                                        new_mid = (new_max + new_min) * 0.5
                                        
                                        if new_mid == new_max or new_mid == new_min or (new_max - new_min) <= precision:
                                            close_flag[0] = 2
                                        else:
                                            mid_val[0] = new_mid
                            
                            T.sync_threads()
                        
                        T.sync_threads()
                        
                        if tx == 0:
                            thres = T.alloc_var(FP32)
                            if close_flag[0] == 2:
                                thres = max_val[0]
                            else:
                                thres = mid_val[0]
                            
                            for i in T.serial(topk):
                                topk_scores_all[l_offset, i] = NEG_INF_REPLACEMENT
                                topk_indices_all[l_offset, i] = -1
                            
                            cnt = T.alloc_var(INT32)
                            cnt = 0
                            
                            for ci in T.serial(cand_size):
                                if cnt < topk:
                                    val = cand_scores[ci]
                                    c_idx = cand_indices[ci]
                                    if val > NEG_INF_REPLACEMENT + 1e20 and val >= thres and c_idx >= 0:
                                        topk_scores_all[l_offset, cnt] = val
                                        topk_indices_all[l_offset, cnt] = c_idx
                                        cnt += 1
                            
                            cur_min = min_val[0]
                            for ci in T.serial(cand_size):
                                if cnt < topk:
                                    val = cand_scores[ci]
                                    c_idx = cand_indices[ci]
                                    if val > NEG_INF_REPLACEMENT + 1e20 and val >= cur_min and val < thres and c_idx >= 0:
                                        topk_scores_all[l_offset, cnt] = val
                                        topk_indices_all[l_offset, cnt] = c_idx
                                        cnt += 1
                        
                        T.sync_threads()
            
            for li, ki in T.Parallel(BLOCK_L, topk):
                q_idx = l_base + li
                if q_idx < seq_len:
                    OutScores[b_idx, q_idx, h_idx, ki] = topk_scores_all[li, ki]
                    OutIndices[b_idx, q_idx, h_idx, ki] = topk_indices_all[li, ki]
    
    return kernel







                        
# from tilelang.autotuner import autotune
# import itertools
# block_K = [16,32]
# block_L = [32,64]
# num_threads = [256,512]
# r_merge = [1]
# gemm_bd=[64,128]
# _configs = list(
#     itertools.product(
#         block_K,
#         block_L,
#         num_threads,
#         r_merge,
#         gemm_bd,
#     ))

# configs = [
#     {
#         "block_K": c[0],
#         "block_L": c[1],
#         "num_threads": c[2],
#         "r_merge": c[3],
#         "gemm_bd": c[4],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )

@tilelang.jit(out_idx=[2, 3],
        pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def fused_topk_forward_kernel_bitonic(
    batch: int,
    seq_len: int,
    s_len: int,
    heads: int,
    head_dim: int,
    topk: int,
    block_K: int = 16,  # {'block_K': 16, 'block_L': 32, 'num_threads': 256, 'r_merge': 1, 'gemm_bd': 128}
    block_L: int = 16,
    dtype: str = "bfloat16",
    num_threads: int = 64,
    r_merge: int = 1,       # 每处理 r_merge 个块后再进行一次合并排序，topk+block_K*r_merge最好是2的幂，否则会很慢
    gemm_bd: int = 128,     # 用于控制参与 GEMM 的线程数
):
    BF16 = "bfloat16"
    FP32 = "float32"
    INT32 = "int32"
    assert topk == tilelang.math.next_power_of_2(topk)

    candidate_raw = topk + r_merge * block_K
    cand_capacity = tilelang.math.next_power_of_2(candidate_raw)
    num_iters_bitonic = int(round(math.log2(cand_capacity)))

    q_shape = [batch, seq_len, heads, head_dim]
    k_shape = [batch, s_len, heads, head_dim]
    out_scores_shape = [batch, seq_len, heads, topk]
    out_indices_shape = [batch, seq_len, heads, topk]

    num_L_blocks = T.ceildiv(seq_len, block_L)
    num_K_blocks = T.ceildiv(s_len, block_K)
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    @T.prim_func
    def fwd_kernel_bitonic(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        OutScores: T.Tensor(out_scores_shape, FP32),
        OutIndices: T.Tensor(out_indices_shape, INT32),
    ):
        with T.Kernel(num_L_blocks, heads, batch, threads=num_threads) as (bx, by, bz):
            b = bz
            h = by
            l_base = bx * block_L

            Q_shared = T.alloc_shared([block_L, head_dim], dtype=dtype)
            logits_frag = T.alloc_fragment([block_L, block_K], FP32)
            logits_shared = T.alloc_shared([block_L, block_K], FP32)
            K_shared = T.alloc_shared([block_K, head_dim], dtype=dtype)
            cand_index_row = T.alloc_shared([block_L, cand_capacity], dtype=INT32)
            cand_value_row = T.alloc_shared([block_L, cand_capacity], dtype=FP32)
            cand_count_row = T.alloc_shared([block_L], dtype=INT32)
            
            
            for li, d in T.Parallel(block_L, head_dim):
                lq = l_base + li
                if lq < seq_len:
                    Q_shared[li, d] = Q[b, lq, h, d]
                else:
                    Q_shared[li, d] = 0
            

            for li, i in T.Parallel(block_L, cand_capacity):
                if i < topk:
                    cand_index_row[li, i] = -1
                    cand_value_row[li, i] = -T.infinity(FP32)
            for li in T.Parallel(block_L):
                cand_count_row[li] = topk
            T.sync_threads()

            for kb in T.serial(num_K_blocks):
                k_start = kb * block_K
                k_end = T.min((kb + 1) * block_K, s_len)
                this_block = k_end - k_start

                for ki, d in T.Parallel(block_K, head_dim):
                    if ki < this_block:
                        K_shared[ki, d] = K[b, k_start + ki, h, d]
                    else:
                        K_shared[ki, d] = 0
                T.sync_threads()

                T.clear(logits_frag)
                tx = T.get_thread_binding()
                if tx < gemm_bd:
                    T.gemm(
                        Q_shared,
                        K_shared,
                        logits_frag,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow
                    )
                T.sync_threads()
                T.copy(logits_frag, logits_shared)
                T.sync_threads()
                
                for li, ki in T.Parallel(block_L, block_K):
                    lq = l_base + li
                    if lq < seq_len:
                        base = cand_count_row[li]
                        if ki < this_block and base + ki < cand_capacity:
                            cand_index_row[li, base + ki] = k_start + ki
                            cand_value_row[li, base + ki] = logits_shared[li, ki] * sm_scale

                T.sync_threads()
                for li in T.Parallel(block_L):
                    lq = l_base + li
                    if lq < seq_len:
                        cand_count_row[li] = T.min(cand_capacity,
                                                   cand_count_row[li] + this_block)
                
                T.sync_threads()
                # 到达 r_merge 或最后一个块触发排序 + 截断保留前 topk
                if ((kb + 1) % r_merge == 0) or (kb + 1 == num_K_blocks):
                    for li, i in T.Parallel(block_L, cand_capacity):
                        lq = l_base + li
                        if lq < seq_len:
                            cur = cand_count_row[li]
                            if i >= cur:
                                cand_index_row[li, i] = -1
                                cand_value_row[li, i] = -T.infinity(FP32)
                    
                    T.sync_threads()
                    # bitonic sort（降序）
                    for stage in T.serial(num_iters_bitonic):
                        for step in T.serial(stage + 1):
                            for li, i in T.Parallel(block_L, cand_capacity):
                                j = i ^ (1 << (stage - step))
                                if i < j:
                                    up = (i & (1 << (stage + 1))) == 0
                                    vi = cand_value_row[li, i]
                                    vj = cand_value_row[li, j]
                                    cond_swap = (up and vi < vj) or ((not up) and vi > vj)
                                    if cond_swap:
                                        # 交换值与索引
                                        cand_value_row[li, i] = vj
                                        cand_value_row[li, j] = vi
                                        ii = cand_index_row[li, i]
                                        ij = cand_index_row[li, j]
                                        cand_index_row[li, i] = ij
                                        cand_index_row[li, j] = ii
                            T.sync_threads()
                            
                    T.sync_threads()
                    # 将计数重置为 topk
                    for li in T.Parallel(block_L):
                        cand_count_row[li] = topk

            T.sync_threads()
            for li, tk in T.Parallel(block_L, topk):
                lq = l_base + li
                if lq < seq_len:
                    OutIndices[b, lq, h, tk] = cand_index_row[li, tk]
                    OutScores[b, lq, h, tk] = cand_value_row[li, tk]
                else:
                    OutIndices[b, lq, h, tk] = -1
                    OutScores[b, lq, h, tk] = 0.0

    return fwd_kernel_bitonic



# from tilelang.autotuner import autotune
# import itertools
# BLOCK_M = [32, 64, 128]
# threads = [64, 128, 256, 512]
# _configs = list(
#     itertools.product(
#         BLOCK_M,
#         threads,
#     ))

# configs = [
#     {
#         "BLOCK_M": c[0],
#         "threads": c[1],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_topk_backward_kernel(batch, seq_len, s_len, heads, head_dim, topk, BLOCK_M=None, threads=None):
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_input_dtype = "int32"
    
    q_shape = [batch, seq_len, heads, head_dim]
    k_shape = [batch, s_len, heads, head_dim]
    grad_out_shape = [batch, seq_len, heads, topk]
    indices_shape = [batch, seq_len, heads, topk]
    
    BK = head_dim
    if BLOCK_M is None:
        BLOCK_M = 32
    if threads is None:
        threads = 256
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    @T.prim_func
    def fused_bwd(
        GradOutScores: T.Tensor(grad_out_shape, dtype),
        Indices: T.Tensor(indices_shape, idx_input_dtype),
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        GradQ: T.Tensor(q_shape, dtype),
        GradK: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_M), heads, batch, threads=threads) as (bx, by, bz):
            i_q_base = bx * BLOCK_M
            h_idx = by
            b_idx = bz
            
            dQ_accum = T.alloc_fragment([BLOCK_M, BK], dtype)
            
            indices_shared = T.alloc_shared([BLOCK_M, topk], "int32")
            grad_scores_shared = T.alloc_shared([BLOCK_M, topk], dtype)
            
            b = bz; h = by
            q_base = bx * BLOCK_M
            Q_shared = T.alloc_shared([BLOCK_M, BK], dtype)
            for m, d in T.Parallel(BLOCK_M, BK):
                q_idx = q_base + m
                if q_idx < seq_len:
                    Q_shared[m, d] = Q[b, q_idx, h, d]
                else:
                    Q_shared[m, d] = 0
            
            for m, t in T.Parallel(BLOCK_M, topk):
                tq = i_q_base + m
                if tq < seq_len:
                    indices_shared[m, t] = Indices[b_idx, tq, h_idx, t]
                    grad_scores_shared[m, t] = GradOutScores[b_idx, tq, h_idx, t]
                else:
                    indices_shared[m, t] = -1
                    grad_scores_shared[m, t] = 0.0
            
            T.fill(dQ_accum, 0.0)
            T.sync_threads()
            
            for m, d in T.Parallel(BLOCK_M, BK):
                q_idx = q_base + m
                if q_idx < seq_len:
                    q_val = T.Cast(accum_dtype, Q_shared[m, d])                  
                    for t in T.serial(topk):
                        k_idx = indices_shared[m, t]
                        if k_idx >= 0:
                            g_score = T.Cast(accum_dtype, grad_scores_shared[m, t])
                            k_val = T.Cast(accum_dtype, K[b, k_idx, h, d])
                            dQ_accum[m, d] += g_score * k_val * sm_scale
                            T.atomic_add(GradK[b, k_idx, h, d], g_score * q_val)
                            
            for m, d in T.Parallel(BLOCK_M, BK):
                q_idx = q_base + m
                if q_idx < seq_len:
                    GradQ[b, q_idx, h, d] = dQ_accum[m, d]

    return fused_bwd







# class OnlineTopKFusedFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q, lmks, topk: int, fwd_kernel, bwd_kernel, sort_kernel):
#         B, L, h_q, D = q.shape
#         B2, S, h_kv, D2 = lmks.shape
#         dtype = q.dtype
#         assert B == B2 and D == D2 and h_q % h_kv == 0
#         G = h_q // h_kv

#         # GQA 将 Query 分组并沿着 G 求和
#         q_group_sum = q.view(B, L, h_kv, G, D).sum(dim=3)  # [B, L, h_kv, D]

#         q_in = q_group_sum.contiguous()
#         k_in = lmks.contiguous()
        
#         best_scores_buf, best_indices_buf = fwd_kernel(q_in, k_in)  # 8.82 ms
        
#         ctx.save_for_backward(q_in, k_in, best_indices_buf)
#         ctx.h_kv = h_kv
#         ctx.G = G
#         ctx.topk = topk
#         ctx.bwd_kernel = bwd_kernel
#         ctx.shapes = (B, L, S, h_kv, D)
        
#         # 将fp32的scores转成bf16返回
#         return best_scores_buf.to(dtype), best_indices_buf

#     # @staticmethod
#     # def backward(ctx, grad_scores_selected, grad_indices_unused):
#     #     q_group_sum, lmks, indices = ctx.saved_tensors
#     #     B, L, S, h_kv, D = ctx.shapes
#     #     G = ctx.G
#     #     bwd_kernel = ctx.bwd_kernel
        
#     #     # GradQ 用 bf16，GradK 仍用 fp32, 因为需要原子加
#     #     grad_q_group_sum = torch.zeros_like(q_group_sum, dtype=q_group_sum.dtype)
#     #     grad_lmks = torch.zeros_like(lmks, dtype=torch.float32)
        
#     #     bwd_kernel(
#     #         grad_scores_selected.contiguous(),
#     #         indices.contiguous(),
#     #         q_group_sum.contiguous(),
#     #         lmks.contiguous(),
#     #         grad_q_group_sum,   # bf16
#     #         grad_lmks           # fp32
#     #     )
        
#     #     # GQA 梯度广播
#     #     grad_q_grouped = grad_q_group_sum.unsqueeze(3).expand(B, L, h_kv, G, D)
#     #     grad_q = grad_q_grouped.reshape(B, L, h_kv * G, D)
        
#     #     return grad_q.to(q_group_sum.dtype), grad_lmks.to(lmks.dtype), None, None, None
    
#     @staticmethod
#     def backward(ctx, grad_scores_selected, grad_indices_unused):
#         q_group_sum, lmks, indices = ctx.saved_tensors
#         B, L, S, h_kv, D = ctx.shapes
#         G = ctx.G
#         bs_h = B * h_kv
        
#         grad_scores_dense = torch.zeros(
#             (B, L, h_kv, S), 
#             dtype=grad_scores_selected.dtype, 
#             device=grad_scores_selected.device
#         )

#         indices = indices.to(torch.int64)
#         grad_scores_dense.scatter_(3, indices, grad_scores_selected)

#         dense_in = grad_scores_dense.permute(0, 2, 1, 3).reshape(bs_h, L, S)
        
#         del grad_scores_dense 

#         q_in = q_group_sum.permute(0, 2, 1, 3).reshape(bs_h, L, D)
        
#         k_in = lmks.permute(0, 2, 1, 3).reshape(bs_h, S, D)
        
#         grad_q_flat = torch.bmm(dense_in, k_in)
        
#         grad_k_flat = torch.bmm(dense_in.transpose(1, 2), q_in)
        
#         del dense_in, q_in, k_in
        
#         grad_q_group_sum = grad_q_flat.view(B, h_kv, L, D).permute(0, 2, 1, 3)
        
#         del grad_q_flat 

#         grad_lmks = grad_k_flat.view(B, h_kv, S, D).permute(0, 2, 1, 3)
#         del grad_k_flat

#         grad_q_grouped = grad_q_group_sum.unsqueeze(3).expand(B, L, h_kv, G, D)
#         grad_q = grad_q_grouped.reshape(B, L, h_kv * G, D)
        
#         return grad_q, grad_lmks, None, None, None, None

    




# class OnlineTopK_Fused(torch.nn.Module):
#     def __init__(self, topk):
#         super().__init__()
#         self.topk = topk
#         self._cached_fwd = None
#         self._cached_sort_kernel = None
#         self._cached_bwd = None
#         self._cached_shape = None

#     def forward(self, q, lmks):
#         B, L, h_q, D = q.shape
#         _, S, h_kv, _ = lmks.shape
#         topk = self.topk

#         shape_key = (B, L, S, h_kv, D, topk)
#         if self._cached_fwd is None or self._cached_shape != shape_key:
            
#             self._cached_fwd = fused_topk_forward_kernel_insert(B, L, S, h_kv, D, topk)
#             print("using insert fused topk kernel")
            
#             # self._cached_fwd = fused_topk_forward_kernel_heap(B, L, S, h_kv, D, topk)
#             # print("using heap fused topk kernel")
            
#             # self._cached_fwd = fused_topk_forward_kernel_rtopk(B, L, S, h_kv, D, topk)
#             # print("using fused rtopk kernel")
            
#             # self._cached_fwd = fused_topk_forward_kernel_bitonic(B, L, S, h_kv, D, topk)
#             # print("using bitonic fused topk kernel")
            
#             self._cached_bwd = fused_topk_backward_kernel(B, L, S, h_kv, D, topk)
#             self._cached_sort_kernel = sort_topk_indices_kernel(B, L, S, h_kv, topk)
            
#             # print(self._cached_fwd.config)
#             # print(self._cached_bwd.config)
            
#             self._cached_shape = shape_key

#         fwd_kernel = self._cached_fwd
#         bwd_kernel = self._cached_bwd
#         sort_kernel = self._cached_sort_kernel

#         scores, indices = OnlineTopKFusedFn.apply(
#             q, lmks, topk, fwd_kernel, bwd_kernel, sort_kernel,
#         )
#         return indices, scores





@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
},
)
def sort_topk_indices_scores_kernel(
    batch: int,
    seq_len: int,
    s_len: int,
    h_kv: int,
    topk: int,
    BLOCK_L: int = 16,
    num_threads: int = 64,
):
    """
    对 IndicesIn: [B, L, h_kv, topk] 按最后一维进行 bitonic 升序排序。
    同时对 ScoresIn 进行相同的重排。
    """
    
    FP32 = "float32"
    INT32 = "int32"
    assert topk == tilelang.math.next_power_of_2(topk)
    num_iters = int(round(math.log2(topk)))

    indices_shape = [batch, seq_len, h_kv, topk]
    scores_shape = [batch, seq_len, h_kv, topk]

    @T.prim_func
    def sort_kernel(
        IndicesIn: T.Tensor(indices_shape, INT32),
        ScoresIn: T.Tensor(scores_shape, FP32),
        IndicesOut: T.Tensor(indices_shape, INT32),
        ScoresOut: T.Tensor(scores_shape, FP32),
    ):

        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), h_kv, batch, threads=num_threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L

            idx_shared = T.alloc_shared([BLOCK_L, topk], dtype=INT32)
            score_shared = T.alloc_shared([BLOCK_L, topk], dtype=FP32)

            for l_idx, k in T.Parallel(BLOCK_L, topk):
                lq = base_l + l_idx
                if lq < seq_len:
                    idx_shared[l_idx, k] = IndicesIn[i_b, lq, i_h, k]
                    score_shared[l_idx, k] = ScoresIn[i_b, lq, i_h, k]
                else:
                    idx_shared[l_idx, k] = -1
                    score_shared[l_idx, k] = -T.infinity(FP32)
            T.sync_threads()

            k_step = T.alloc_var(INT32)
            j_step = T.alloc_var(INT32)
            # i_idx = T.alloc_var(INT32)  # 0.1.7.post1和0.1.7.post2需要注释掉
            # ixj = T.alloc_var(INT32)  # 0.1.7.post1和0.1.7.post2需要注释掉
            
            val_i = T.alloc_var(INT32)
            val_j = T.alloc_var(INT32)
            score_val_i = T.alloc_var(FP32)
            score_val_j = T.alloc_var(FP32)
            
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
                                
                                score_val_i = score_shared[l_idx, i_idx]
                                score_val_j = score_shared[l_idx, ixj]

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
                                    score_shared[l_idx, i_idx] = score_val_j
                                    score_shared[l_idx, ixj] = score_val_i
                                    
                    T.sync_threads()
                    j_step = j_step // 2
                k_step = k_step * 2

            for l_idx, k in T.Parallel(BLOCK_L, topk):
                lq = base_l + l_idx
                if lq < seq_len:
                    IndicesOut[i_b, lq, i_h, k] = idx_shared[l_idx, k]
                    ScoresOut[i_b, lq, i_h, k] = score_shared[l_idx, k]

    return sort_kernel


class OnlineTopKFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, lmks, topk: int, fwd_kernel, bwd_kernel, sort_kernel):
        B, L, h_q, D = q.shape
        B2, S, h_kv, D2 = lmks.shape
        dtype = q.dtype
        assert B == B2 and D == D2 and h_q % h_kv == 0
        G = h_q // h_kv

        q_group_sum = q.view(B, L, h_kv, G, D).sum(dim=3)  # [B, L, h_kv, D]

        q_in = q_group_sum.contiguous()
        k_in = lmks.contiguous()
        
        best_scores_buf, best_indices_buf = fwd_kernel(q_in, k_in)  
        
        indices_sorted, scores_sorted = sort_kernel(best_indices_buf, best_scores_buf) 

        ctx.save_for_backward(q_in, k_in, indices_sorted)
        ctx.h_kv = h_kv
        ctx.G = G
        ctx.topk = topk
        ctx.bwd_kernel = bwd_kernel
        ctx.shapes = (B, L, S, h_kv, D)
        
        return scores_sorted.to(dtype), indices_sorted

    @staticmethod
    def backward(ctx, grad_scores_selected, grad_indices_unused):
        q_group_sum, lmks, indices = ctx.saved_tensors
        B, L, S, h_kv, D = ctx.shapes
        G = ctx.G
        bs_h = B * h_kv
        
        sm_scale = 1.0 / math.sqrt(D)
        
        grad_scores_dense = torch.zeros(
            (B, L, h_kv, S), 
            dtype=grad_scores_selected.dtype, 
            device=grad_scores_selected.device
        )

        indices = indices.to(torch.int64)
        
        valid_mask = (indices >= 0) & (indices < S)
        safe_indices = indices.clone()
        safe_indices[~valid_mask] = 0
        safe_grad = grad_scores_selected.clone()
        safe_grad[~valid_mask] = 0
        
        grad_scores_dense.scatter_(3, safe_indices, safe_grad)

        dense_in = grad_scores_dense.permute(0, 2, 1, 3).reshape(bs_h, L, S)
        
        del grad_scores_dense 

        q_in = q_group_sum.permute(0, 2, 1, 3).reshape(bs_h, L, D)
        
        k_in = lmks.permute(0, 2, 1, 3).reshape(bs_h, S, D)
        
        grad_q_flat = torch.bmm(dense_in, k_in)
        
        grad_k_flat = torch.bmm(dense_in.transpose(1, 2), q_in)
        
        del dense_in, q_in, k_in
        
        grad_q_group_sum = grad_q_flat.view(B, h_kv, L, D).permute(0, 2, 1, 3)
        
        del grad_q_flat 

        grad_lmks = grad_k_flat.view(B, h_kv, S, D).permute(0, 2, 1, 3)
        del grad_k_flat

        grad_q_grouped = grad_q_group_sum.unsqueeze(3).expand(B, L, h_kv, G, D)
        grad_q = grad_q_grouped.reshape(B, L, h_kv * G, D)
        
        grad_q = grad_q * sm_scale
        grad_lmks = grad_lmks * sm_scale
        
        return grad_q, grad_lmks, None, None, None, None


class OnlineTopK_Fused(torch.nn.Module):
    def __init__(self, topk, block_size, window_size, is_causal):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self._cached_fwd = None
        self._cached_sort_kernel = None
        self._cached_bwd = None
        self._cached_shape = None

    def forward(self, q, lmks):
        B, L, h_q, D = q.shape
        _, S, h_kv, _ = lmks.shape
        topk = self.topk
        block_size = self.block_size
        window_size = self.window_size
        is_causal = self.is_causal

        shape_key = (B, L, S, h_kv, D, topk, block_size, window_size, is_causal)
        if self._cached_fwd is None or self._cached_shape != shape_key:
            
            self._cached_fwd = fused_topk_forward_kernel_insert(
                B, L, S, h_kv, D, topk, block_size, window_size, is_causal
            )
            print("using insert fused topk kernel")
            
            self._cached_bwd = fused_topk_backward_kernel(B, L, S, h_kv, D, topk)
            
            self._cached_sort_kernel = sort_topk_indices_scores_kernel(B, L, S, h_kv, topk)
            
            self._cached_shape = shape_key

        fwd_kernel = self._cached_fwd
        bwd_kernel = self._cached_bwd
        sort_kernel = self._cached_sort_kernel

        scores, indices = OnlineTopKFusedFn.apply(
            q, lmks, topk, fwd_kernel, bwd_kernel, sort_kernel,
        )
        return indices, scores


_MODULE_CACHE = {}

def online_topk_group(q: torch.Tensor, lmks: torch.Tensor, topk: int, block_size: int, window_size: int, is_causal: bool = False):
    """
    Functional API for OnlineTopK_Fused
    
    Args:
        q: [B, L, h_q, D]
        lmks: [B, S, h_kv, D]
        topk: int
        block_size: int
        window_size: int
        is_causal: bool

    Returns:
        indices: [B, L, h_kv, topk]
        scores: [B, L, h_kv, topk]
    """
    cache_key = (topk, block_size, window_size, is_causal)
    if cache_key not in _MODULE_CACHE:
        _MODULE_CACHE[cache_key] = OnlineTopK_Fused(topk, block_size, window_size, is_causal)

    return _MODULE_CACHE[cache_key](q, lmks)






def ref_topk_forward_with_grad(q, lmks, topk, block_size, window_size, is_causal, dtype):
    B, L, h_q, D = q.shape
    _, S, h_kv, _ = lmks.shape
    sm_scale = 1.0 / math.sqrt(D)
    G = h_q // h_kv
    q_group_sum = q.view(B, L, h_kv, G, D).sum(dim=3)
    scores_ref = torch.einsum("blkd,bskd->blks", q_group_sum.to(dtype), lmks.to(dtype)) * sm_scale
    
    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1) # [L, 1]
        j_idx = torch.arange(S, device=q.device).unsqueeze(0) # [1, S]
        
        # New Mask Logic
        limit_chunk_idx = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= limit_chunk_idx
        
        # scores_ref: [B, L, h_kv, S]
        # mask: [1, L, 1, S]
        scores_ref = scores_ref.masked_fill(causal_mask.unsqueeze(0).unsqueeze(2), float('-inf'))

    scores_topk, indices_topk = torch.topk(scores_ref, k=topk, dim=-1, sorted=False)
    indices_sorted, order = torch.sort(indices_topk, dim=-1)
    scores_sorted = torch.gather(scores_topk, -1, order)
    return indices_sorted, scores_sorted



def test_online_topk_fused_correctness():
    print("\n" + "=" * 70)
    print("=== Testing OnlineTopK_Fused Correctness ===")
    print("=" * 70)
    
    B, L, D = 64, 4096, 128
    h_q = 16
    h_kv = 2
    S = 64
    topk = 16
    is_causal =True
    block_size = 64
    window_size = 100
    
    dtype = torch.bfloat16
    device = "cuda"
    
    print(f"Config: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}, window_size={window_size}")
    
    torch.manual_seed(42)
    q = torch.randn(B, L, h_q, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)
    
    # ============ Forward Correctness ============
    print("\n--- Forward Correctness ---")
    
    indices_fused, scores_fused = online_topk_group(q, lmks, topk, block_size, window_size, is_causal)
    indices_ref, scores_ref = ref_topk_forward_with_grad(q, lmks, topk, block_size, window_size, is_causal, dtype=torch.float32)
    
    print("indices_fused sample:", indices_fused[0,500,0,:])
    print("indices_ref sample:  ", indices_ref[0,500,0,:])

    valid_mask = (scores_ref > -1e9) & (scores_fused.float() > -1e9)
    
    if valid_mask.sum() == 0:
        diff_scores = 0.0
    else:
        diff_scores = torch.abs(scores_fused.float()[valid_mask] - scores_ref[valid_mask]).max().item()
        
    print(f"Max Scores Diff (Element-wise, Valid): {diff_scores:.6f}")
    
    indices_match = (indices_fused == indices_ref)
    if valid_mask.sum() > 0:
        match_rate = indices_match[valid_mask].float().mean().item()
    else:
        match_rate = 1.0
        
    print(f"Indices Match Rate (Element-wise, Valid): {match_rate*100:.8f}%")
    
    if match_rate >= 0.99 and diff_scores < 1.0:
        print("✅ Fused Forward PASSED")
    else:
        print("❌ Fused Forward FAILED")
    
    # ============ Backward Correctness ============
    print("\n--- Backward Correctness ---")
    
    q_fused = q.detach().clone().requires_grad_(True)
    lmks_fused = lmks.detach().clone().requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    lmks_ref = lmks.detach().clone().requires_grad_(True)
    
    grad_output = torch.randn(B, L, h_kv, topk, dtype=dtype, device=device)
    
    with torch.no_grad():
        _, scores_ref_check = ref_topk_forward_with_grad(q_ref, lmks_ref, topk, block_size, window_size, is_causal, dtype=torch.float32)
        invalid_mask = (scores_ref_check < -1e9)
        grad_output[invalid_mask] = 0.0

    indices_fused_bwd, scores_fused_bwd = online_topk_group(q_fused, lmks_fused, topk, block_size, window_size, is_causal)
    loss_fused = (scores_fused_bwd * grad_output).sum()
    loss_fused.backward()
    grad_q_fused = q_fused.grad.clone()
    grad_lmks_fused = lmks_fused.grad.clone()
    
    G = h_q // h_kv
    q_grouped_ref = q_ref.view(B, L, h_kv, G, D)
    lmks_expanded = lmks_ref.unsqueeze(3).expand(B, S, h_kv, G, D)
    
    scores_per_group = torch.einsum("blkgd,bskgd->blkgs", q_grouped_ref.float(), lmks_expanded.float())
    scores_full = scores_per_group.sum(dim=3) * (1.0 / math.sqrt(D))
    
    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        
        # New Mask Logic
        limit_chunk_idx = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= limit_chunk_idx

        scores_full = scores_full.masked_fill(causal_mask.unsqueeze(0).unsqueeze(2), float('-inf'))

    safe_indices = indices_fused_bwd.long().clone()
    safe_indices[safe_indices < 0] = 0
    scores_gathered = torch.gather(scores_full, -1, safe_indices)
    
    loss_ref = (scores_gathered * grad_output.float()).sum()
    loss_ref.backward()
    grad_q_ref = q_ref.grad.clone()
    grad_lmks_ref = lmks_ref.grad.clone()
    
    if torch.isnan(grad_q_ref).any(): grad_q_ref = torch.nan_to_num(grad_q_ref, 0.0)
    if torch.isnan(grad_lmks_ref).any(): grad_lmks_ref = torch.nan_to_num(grad_lmks_ref, 0.0)
    if torch.isnan(grad_q_fused).any(): grad_q_fused = torch.nan_to_num(grad_q_fused, 0.0)
    if torch.isnan(grad_lmks_fused).any(): grad_lmks_fused = torch.nan_to_num(grad_lmks_fused, 0.0)

    diff_grad_q = torch.abs(grad_q_fused - grad_q_ref)
    diff_grad_lmks = torch.abs(grad_lmks_fused - grad_lmks_ref)
    
    max_diff_q = diff_grad_q.max().item()
    max_diff_lmks = diff_grad_lmks.max().item()
    
    norm_q_ref = torch.norm(grad_q_ref).item()
    norm_lmks_ref = torch.norm(grad_lmks_ref).item()
    norm_diff_q = torch.norm(diff_grad_q).item()
    norm_diff_lmks = torch.norm(diff_grad_lmks).item()
    
    rel_err_q = norm_diff_q / (norm_q_ref + 1e-6)
    rel_err_lmks = norm_diff_lmks / (norm_lmks_ref + 1e-6)
    
    print(f"Fused vs Ref - Max grad_q Diff: {max_diff_q:.6f}")
    print(f"Fused vs Ref - Max grad_lmks Diff: {max_diff_lmks:.6f}")
    print(f"Fused vs Ref - L2 Relative Error grad_q: {rel_err_q:.6f} ({rel_err_q*100:.4f}%)")
    print(f"Fused vs Ref - L2 Relative Error grad_lmks: {rel_err_lmks:.6f} ({rel_err_lmks*100:.4f}%)")

    passed = True
    if rel_err_q > 0.01 or rel_err_lmks > 0.01:
        passed = False
        print(f"\n⚠️  L2 Relative Error too large!")
    
    if passed:
        print("✅ Fused Backward PASSED")
    else:
        print("❌ Fused Backward FAILED")


def test_online_topk_fused_memory_and_speed():
    print("\n" + "=" * 70)
    print("=== Benchmark OnlineTopK_Fused Memory and Speed ===")
    print("=" * 70)
    
    B, L, D = 64, 4096, 128
    h_q = 16
    h_kv = 2
    S = 64
    topk = 16
    is_causal =True
    block_size = 64
    window_size = 64
    
    dtype = torch.bfloat16
    device = "cuda"
    
    print(f"Config: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}")
    
    torch.manual_seed(42)
    q = torch.randn(B, L, h_q, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(B, L, h_kv, topk, dtype=dtype, device=device)
    
    n_iters = 20
    
    def run_fused():
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        _, scores = online_topk_group(q_t, lmks_t, topk, block_size, window_size, is_causal)
        loss = (scores * grad_output).sum()
        loss.backward()
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(5):
            q_t.grad = None
            lmks_t.grad = None
            _ = online_topk_group(q_t, lmks_t, topk, block_size, window_size, is_causal)
            _, scores = online_topk_group(q_t, lmks_t, topk, block_size, window_size, is_causal)
            loss = (scores * grad_output).sum()
            loss.backward()
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        _ = online_topk_group(q_t, lmks_t, topk, block_size, window_size, is_causal)
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
            _, scores = online_topk_group(q_t, lmks_t, topk, block_size, window_size, is_causal)
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters
        
        # Fwd + Bwd
        start_all.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            _, scores = online_topk_group(q_t, lmks_t, topk, block_size, window_size, is_causal)
            loss = (scores * grad_output).sum()
            loss.backward()
        end_all.record()
        torch.cuda.synchronize()
        avg_all_ms = start_all.elapsed_time(end_all) / n_iters
        
        # Bwd = (Fwd+Bwd) - Fwd
        avg_bwd_ms = avg_all_ms - avg_fwd_ms
        
        return peak_mem, avg_fwd_ms, avg_all_ms, avg_bwd_ms
    
    def run_ref():
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        def forward_only():
            _ = ref_topk_forward_with_grad(q_t, lmks_t, topk, block_size, window_size, is_causal, dtype=torch.bfloat16)
        
        def forward_backward():
            _, scores = ref_topk_forward_with_grad(q_t, lmks_t, topk, block_size, window_size, is_causal, dtype=torch.bfloat16)
            loss = (scores * grad_output).sum()
            loss.backward()
        
        for _ in range(5):
            q_t.grad = None
            lmks_t.grad = None
            forward_only()
            forward_backward()
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
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
            forward_only()
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters
        
        # Fwd + Bwd
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
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    
    mem_fused, fwd_fused, all_fused, bwd_fused = run_fused()
    print(f"\n[OnlineTopK_Fused]")
    print(f"  Peak Memory: {mem_fused:.2f} MB")
    print(f"  Avg Fwd Latency: {fwd_fused:.2f} ms")
    print(f"  Avg Fwd+Bwd Latency: {all_fused:.2f} ms")
    print(f"  Derived Bwd Latency: {bwd_fused:.2f} ms")
    
    try:
        mem_ref, fwd_ref, all_ref, bwd_ref = run_ref()
        print(f"\n[Reference (torch.topk)]")
        print(f"  Peak Memory: {mem_ref:.2f} MB")
        print(f"  Avg Fwd Latency: {fwd_ref:.2f} ms")
        print(f"  Avg Fwd+Bwd Latency: {all_ref:.2f} ms")
        print(f"  Derived Bwd Latency: {bwd_ref:.2f} ms")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[Reference (torch.topk)] OOM - Cannot run with this config")
            print("\n" + "-" * 70)
            print("Comparison (Reference OOM):")
            print("-" * 70)
            print(f"{'Method':<25} {'Memory (MB)':<15} {'Fwd (ms)':<12} {'Fwd+Bwd (ms)':<15} {'Bwd (ms)':<12}")
            print("-" * 70)
            print(f"{'OnlineTopK_Fused':<25} {mem_fused:<15.2f} {fwd_fused:<12.2f} {all_fused:<15.2f} {bwd_fused:<12.2f}")
            print("-" * 70)
        else:
            raise e



# ...existing code...
import pytest
@pytest.mark.parametrize("B, L, S, h_kv, G, D, topk, block_size", [
    (2, 4096, 64, 2, 8, 64, 16, 64),
])
def test_topk_correctness_robust(B, L, S, h_kv, G, D, topk, block_size):
    device = "cuda"
    dtype = torch.bfloat16
    is_causal = True
    window_size = block_size # default robust test
    h_q = h_kv * G
    
    torch.manual_seed(42)
    
    print(f"\nTesting Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, topk={topk}, BS={block_size}")

    # 1.准备数据
    # Q: [B, L, h_q, D]
    q_raw = torch.randn(B, L, h_q, D, dtype=dtype, device=device)
    # Lmks: [B, S, h_kv, D]
    lmks_raw = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)
    
    # 2.输入隔离 (Clone & Detach)
    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)
    
    q_ref = q_raw.clone().detach().requires_grad_(True)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(True)

    # 3.前向计算
    # Fused Kernel
    # indices: [B, L, h_kv, topk]
    # scores:  [B, L, h_kv, topk]
    indices_fused, scores_fused = online_topk_group(q_fused, lmks_fused, topk, block_size, window_size, is_causal)


    # Reference
    # indices: [B, L, h_kv, topk]
    # scores:  [B, L, h_kv, topk]
    # 注意：ref_topk_forward_with_grad 内部会对 indices 进行排序，fused kernel 也有排序步骤，因此可以直接对比
    indices_ref, scores_ref = ref_topk_forward_with_grad(q_ref, lmks_ref, topk, block_size, window_size, is_causal, dtype=torch.float32)

    # 4.辅助校验函数
    def get_abs_err(x, y):
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
    assert_close("FWD Scores", scores_ref.float(), scores_fused.float())

    # 6.反向计算
    # 生成 grad_output [B, L, h_kv, topk]
    grad_output = torch.randn_like(scores_fused, dtype=dtype)
    
    # Mask 掉无效位置的梯度
    with torch.no_grad():
        invalid_mask = scores_ref < -1e5
        grad_output[invalid_mask] = 0.0

    # Fused Backward
    scores_fused.backward(grad_output)
    dq_fused, dlmks_fused = q_fused.grad.clone(), lmks_fused.grad.clone()

    # Ref Backward
    # 需要手动构建计算图来验证 backward
    # 1. 计算 Full Scores [B, L, h_kv, S]
    q_grouped_ref = q_ref.view(B, L, h_kv, G, D)
    q_group_sum_ref = q_grouped_ref.sum(dim=3) # [B, L, h_kv, D]
    scores_full = torch.einsum("blkd,bskd->blks", q_group_sum_ref.float(), lmks_ref.float())
    scores_full = scores_full * (1.0 / math.sqrt(D))
    
    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        
        # New Mask Logic
        limit_chunk_idx = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= limit_chunk_idx
        
        scores_full = scores_full.masked_fill(causal_mask.unsqueeze(0).unsqueeze(2), float('-inf'))

    # 2. Gather using FUSED indices (to match the graph path)
    safe_indices = indices_fused.long().clone()
    safe_indices[safe_indices < 0] = 0
    scores_gathered = torch.gather(scores_full, -1, safe_indices)
    
    # 3. Backward
    loss_ref = (scores_gathered * grad_output.float()).sum()
    loss_ref.backward()
    dq_ref, dlmks_ref = q_ref.grad.clone(), lmks_ref.grad.clone()

    # 7.验证梯度
    dq_fused = torch.nan_to_num(dq_fused, 0.0)
    dq_ref = torch.nan_to_num(dq_ref, 0.0)
    dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
    dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)

    assert_close("DQ", dq_ref.float(), dq_fused.float(), ratio=0.05)
    assert_close("DLmks", dlmks_ref.float(), dlmks_fused.float(), ratio=0.05)

    print(f"Test Passed: B={B}, L={L}, S={S}, G={G}, topk={topk}")

if __name__ == "__main__":
    test_online_topk_fused_correctness()
    # test_online_topk_fused_memory_and_speed()
    params_list = [
        (2, 4096, 64, 2, 8, 64, 16, 64),
        (1, 2048, 64, 1, 8, 64, 16, 32),
        (3, 2048, 64, 1, 8, 64, 8, 32),
    ]
    for p in params_list:
        test_topk_correctness_robust(*p)
