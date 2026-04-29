import torch
import tilelang
import tilelang.language as T
from typing import Optional
import math





@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_topk_forward_kernel_insert_unified(
    batch, heads, head_dim, topk,
    block_size, window_size, is_causal,
    seq_len=None, s_len=None,
    BLOCK_L=None, BLOCK_S=None, threads=None,
    is_training=True,
    use_drop_mask=False,
    is_varlen=False,
    force_recent_chunks=0,
):
    """
    统一的 TopK 前向 kernel，同时支持 dense (pretrain) 和 varlen (SFT packing) 场景。

    以 dense kernel 为主干代码，varlen 通过 5 个残差注入点融合：
      注入点1: seq_len/s_len dynamic 处理
      注入点2: s_block 循环前 — seq_id 查找 + local 坐标计算
      注入点3: loop_limit 计算（varlen 需遍历全局 s_len）
      注入点4: is_valid 判断前 — 坐标转换 (effective_q, effective_s, in_range)
      注入点5: topk_indices 写入 — varlen 存 global index

    is_varlen=False (dense 模式):
        - Grid: (cdiv(seq_len, BLOCK_L), heads, batch)
        - 输入: Q [B, L, h, D], K [B, S, h, D], Q_Offset [1], DropMask [B, L, S]
        - 输出 indices: batch 内的 lmk index

    is_varlen=True (varlen 模式):
        - Grid: (cdiv(seq_len, BLOCK_L), heads, 1)，batch 固定为 1
        - 输入: Q [1, L_total, h, D], K [1, S_total, h, D], CuSeqLensQ [N+1], CuSeqLensK [N+1]
        - 输出 indices: global packed landmark index
        - 内部 causal/window 判断使用子序列内的 local index

    所有 is_varlen 分支都是编译时常量，不引入运行时开销。
    """
    if is_varlen:
        seq_len = T.dynamic("seq_len")
        s_len = T.dynamic("s_len")
        num_seqs = T.dynamic("num_seqs")
    else:
        if not is_training:
            seq_len = T.dynamic("seq_len")
            s_len = T.dynamic("s_len")

    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    if is_varlen:
        q_shape = [1, seq_len, heads, head_dim]
        k_shape = [1, s_len, heads, head_dim]
        out_scores_shape = [1, seq_len, heads, topk]
        out_indices_shape = [1, seq_len, heads, topk]
        cu_q_shape = [num_seqs + 1]
        cu_k_shape = [num_seqs + 1]
        drop_mask_shape = [1, seq_len, s_len] if use_drop_mask else [1, 1, 1]
    else:
        q_shape = [batch, seq_len, heads, head_dim]
        k_shape = [batch, s_len, heads, head_dim]
        out_scores_shape = [batch, seq_len, heads, topk]
        out_indices_shape = [batch, seq_len, heads, topk]
        drop_mask_shape = [batch, seq_len, s_len] if use_drop_mask else [1, 1, 1]
        cu_q_shape = [1]
        cu_k_shape = [1]

    if BLOCK_L is None:
        BLOCK_L = 32
    if BLOCK_S is None:
        BLOCK_S = 16
    BLOCK_D = head_dim
    if threads is None:
        threads = BLOCK_L

    num_s_blocks = tilelang.cdiv(s_len, BLOCK_S)
    sm_scale = 1.0 / math.sqrt(head_dim)

    grid_batch = 1 if is_varlen else batch
    effective_topk = topk - force_recent_chunks

    # debug
    print(f"batch={batch}, seq_len={seq_len}, s_len={s_len}, heads={heads}, head_dim={head_dim}, topk={topk}, BLOCK_L={BLOCK_L}, BLOCK_S={BLOCK_S}, force_recent_chunks={force_recent_chunks}")

    @T.prim_func
    def fwd_kernel_insert_unified(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
        Q_Offset_or_CuQ: T.Tensor(cu_q_shape if is_varlen else [1], "int32"),
        CuSeqLensK: T.Tensor(cu_k_shape, idx_dtype),
        DropMask: T.Tensor(drop_mask_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), heads, grid_batch, threads=threads) as (bx, by, bz):
            q_offset = T.if_then_else(is_training, 0, Q_Offset_or_CuQ[0])
            b = bz
            h = by
            base_l = bx * BLOCK_L

            Q_shared = T.alloc_shared([BLOCK_L, BLOCK_D], dtype)
            K_shared = T.alloc_shared([BLOCK_S, BLOCK_D], dtype)
            score_shared = T.alloc_shared([BLOCK_L, BLOCK_S], accum_dtype)
            acc_s = T.alloc_fragment([BLOCK_L, BLOCK_S], accum_dtype)

            topk_scores = T.alloc_local([topk], accum_dtype)
            topk_indices = T.alloc_local([topk], idx_dtype)

            for kk in T.serial(topk):
                topk_scores[kk] = -T.infinity(accum_dtype)
                topk_indices[kk] = -1

            if force_recent_chunks > 0:
                recent_scores = T.alloc_local([force_recent_chunks], accum_dtype)
                for ri in T.serial(force_recent_chunks):
                    recent_scores[ri] = -T.infinity(accum_dtype)
                recent_start = T.alloc_var(idx_dtype)
                recent_start = 0

            for l_idx, d in T.Parallel(BLOCK_L, BLOCK_D):
                tq = base_l + l_idx
                if tq < seq_len:
                    Q_shared[l_idx, d] = Q[b, tq, h, d]
                else:
                    Q_shared[l_idx, d] = T.Cast(dtype, 0.0)

            if is_varlen:
                tx_pre = T.get_thread_binding()
                tq_packed = base_l + tx_pre

                seq_id = T.alloc_var(idx_dtype)
                seq_id = 0
                for si in T.serial(num_seqs):
                    if tq_packed >= Q_Offset_or_CuQ[si + 1]:
                        seq_id = si + 1

                q_start = T.alloc_var(idx_dtype)
                q_start = Q_Offset_or_CuQ[seq_id]
                local_q = T.alloc_var(idx_dtype)
                local_q = tq_packed - q_start

                k_start = T.alloc_var(idx_dtype)
                k_end = T.alloc_var(idx_dtype)
                k_start = CuSeqLensK[seq_id]
                k_end = CuSeqLensK[seq_id + 1]

            loop_limit = T.alloc_var("int32")
            loop_limit = num_s_blocks
            if is_causal and (not is_varlen):
                tq_max = T.min(seq_len - 1, base_l + (BLOCK_L - 1))
                tq_max_global = q_offset + tq_max
                block_q_max = tq_max_global // block_size
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

                varlen_thread_ok = T.alloc_var("bool")
                varlen_thread_ok = True
                if is_varlen:
                    varlen_thread_ok = (seq_id < num_seqs)

                if (my_l < BLOCK_L) and (tq < seq_len) and varlen_thread_ok:
                    tq_global = q_offset + tq
                    effective_q = T.alloc_var(idx_dtype)
                    effective_q = tq_global
                    if is_varlen:
                        effective_q = local_q

                    limit_chunk = (effective_q - window_size + 1) // block_size

                    if force_recent_chunks > 0:
                        recent_start = T.max(limit_chunk - force_recent_chunks, 0)

                    is_valid = T.alloc_var("bool")
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        effective_s = T.alloc_var(idx_dtype)
                        effective_s = ts
                        in_range = T.alloc_var("bool")
                        in_range = (ts < s_len)
                        if is_varlen:
                            effective_s = ts - k_start
                            in_range = (ts >= k_start) and (ts < k_end)

                        if in_range:
                            if force_recent_chunks > 0:
                                is_valid = (not is_causal) or (effective_s < recent_start)
                            else:
                                is_valid = (not is_causal) or (effective_s < limit_chunk)
                            if use_drop_mask:
                                is_valid = is_valid and (DropMask[b, tq, ts] == 0)
                            if is_valid:
                                cur = score_shared[my_l, s_idx] * sm_scale

                                if cur > topk_scores[effective_topk - 1]:
                                    moving = T.alloc_var("bool")
                                    moving = True
                                    for rkk in T.serial(effective_topk):
                                        kpos = effective_topk - 1 - rkk
                                        if moving:
                                            if (kpos > 0) and (cur > topk_scores[kpos - 1]):
                                                topk_scores[kpos] = topk_scores[kpos - 1]
                                                topk_indices[kpos] = topk_indices[kpos - 1]
                                            else:
                                                topk_scores[kpos] = cur
                                                topk_indices[kpos] = ts
                                                moving = False

                            if force_recent_chunks > 0:
                                if (effective_s >= recent_start) and (effective_s < limit_chunk):
                                    ri = effective_s - recent_start
                                    recent_scores[ri] = score_shared[my_l, s_idx] * sm_scale
                T.sync_threads()

            tx2 = T.get_thread_binding()
            tq2 = base_l + tx2
            if (tx2 < BLOCK_L) and (tq2 < seq_len):
                for kk in T.serial(effective_topk):
                    OutScores[b, tq2, h, kk] = topk_scores[kk]
                    OutIndices[b, tq2, h, kk] = topk_indices[kk]
                if force_recent_chunks > 0:
                    for ri in T.serial(force_recent_chunks):
                        OutScores[b, tq2, h, effective_topk + ri] = recent_scores[ri]
                        if is_varlen:
                            OutIndices[b, tq2, h, effective_topk + ri] = k_start + recent_start + ri
                        else:
                            OutIndices[b, tq2, h, effective_topk + ri] = recent_start + ri

    return fwd_kernel_insert_unified
  





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




# _sort_BLOCK_L = [16, 32, 64]
# _sort_num_threads = [32, 64, 128]
# _sort_configs = list(
#     itertools.product(
#         _sort_BLOCK_L,
#         _sort_num_threads,
#     ))

# sort_configs = [
#     {
#         "BLOCK_L": c[0],
#         "num_threads": c[1],
#     } for c in _sort_configs
# ]

# @autotune(
#     configs=sort_configs,
#     warmup=5,
#     rep=10,
# )
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
    h_kv: int,
    topk: int,
    seq_len: int = None,
    s_len: int = None,
    BLOCK_L: int = 16,
    num_threads: int = 64,
    is_training: bool = True,
):
    """
    对 IndicesIn: [B, L, h_kv, topk] 按最后一维进行 bitonic 升序排序。
    同时对 ScoresIn 进行相同的重排。
    """
    
    FP32 = "float32"
    INT32 = "int32"
    assert topk == tilelang.math.next_power_of_2(topk)
    num_iters = int(round(math.log2(topk)))

    if not is_training:
        seq_len = T.dynamic("seq_len")
    
    if s_len is None:
        s_len = 2**30 

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




class OnlineTopKUnifiedFn(torch.autograd.Function):
    """
    统一的 TopK autograd Function，同时支持 dense 和 varlen 路径。
    通过 is_varlen 标志在 forward/backward 中分支。
    """

    @staticmethod
    def forward(ctx, q, lmks, topk: int, fwd_kernel, sort_kernel,
                is_varlen: bool,
                q_offset_tensor=None, drop_mask=None,
                cu_seq_lens_q=None, cu_seq_lens_k=None):
        B, L, h_q, D = q.shape
        B2, S, h_kv, D2 = lmks.shape
        dtype = q.dtype
        assert B == B2

        if is_varlen:
            assert B == 1, "Varlen TopK 要求 B=1"

        d_reshape = False
        orig_h_kv = h_kv
        orig_D2 = D2
        if D != D2:
            assert D2 % D == 0
            d_ratio = D2 // D
            lmks = lmks.reshape(B, S, h_kv * d_ratio, D)
            h_kv = h_kv * d_ratio
            D2 = D
            d_reshape = True

        if h_q >= h_kv:
            assert h_q % h_kv == 0
            G = h_q // h_kv
            sum_q = True
            h_shared = h_kv
            q_group_sum = q.view(B, L, h_kv, G, D).sum(dim=3)
            q_in = q_group_sum.to(dtype).contiguous()
            k_in = lmks.contiguous()
        else:
            assert h_kv % h_q == 0
            G = h_kv // h_q
            sum_q = False
            h_shared = h_q
            lmk_group_sum = lmks.view(B, S, h_q, G, D).sum(dim=3)
            q_in = q.contiguous()
            k_in = lmk_group_sum.to(dtype).contiguous()

        if is_varlen:
            dm_placeholder = torch.zeros(1, 1, 1, dtype=torch.int32, device=q.device)
            best_scores_buf, best_indices_buf = fwd_kernel(
                q_in, k_in,
                cu_seq_lens_q.contiguous(),
                cu_seq_lens_k.contiguous(),
                dm_placeholder,
            )
        else:
            if drop_mask is None:
                drop_mask_in = torch.zeros(1, 1, 1, dtype=torch.int32, device=q.device)
            else:
                drop_mask_in = drop_mask
            cu_k_placeholder = torch.zeros(1, dtype=torch.int32, device=q.device)
            best_scores_buf, best_indices_buf = fwd_kernel(
                q_in, k_in, q_offset_tensor, cu_k_placeholder, drop_mask_in,
            )

        indices_sorted, scores_sorted = sort_kernel(best_indices_buf, best_scores_buf)

        ctx.save_for_backward(q_in, k_in, indices_sorted)

        ctx.h_kv = h_kv
        ctx.h_q = h_q
        ctx.G = G
        ctx.sum_q = sum_q
        ctx.h_shared = h_shared
        ctx.topk = topk
        ctx.shapes = (B, L, S, D)
        ctx.d_reshape = d_reshape
        ctx.orig_h_kv = orig_h_kv
        ctx.orig_D2 = orig_D2

        return scores_sorted.to(dtype), indices_sorted

    @staticmethod
    def backward(ctx, grad_scores_selected, grad_indices_unused):
        """
        统一的 backward 路径（方案 B: 全局 scatter + bmm）。

        dense 和 varlen 共用同一套逻辑：
        - dense: B>=1, L=seq_len, S=num_chunks
        - varlen: B=1, L=L_total, S=S_total, indices 是 global index
          scatter 到 [1, L_total, h, S_total] 后 bmm，跨序列位置自然为 0
        """
        q_saved, k_saved, indices = ctx.saved_tensors
        B, L, S, D = ctx.shapes
        G = ctx.G
        sum_q = ctx.sum_q
        h_shared = ctx.h_shared
        h_q = ctx.h_q
        h_kv = ctx.h_kv
        d_reshape = ctx.d_reshape
        orig_h_kv = ctx.orig_h_kv
        orig_D2 = ctx.orig_D2
        sm_scale = 1.0 / math.sqrt(D)
        bs_h = B * h_shared

        grad_scores_dense = torch.zeros(
            (B, L, h_shared, S),
            dtype=grad_scores_selected.dtype,
            device=grad_scores_selected.device,
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

        q_in = q_saved.permute(0, 2, 1, 3).reshape(bs_h, L, D)
        k_in = k_saved.permute(0, 2, 1, 3).reshape(bs_h, S, D)

        grad_q_flat = torch.bmm(dense_in, k_in)
        grad_k_flat = torch.bmm(dense_in.transpose(1, 2), q_in)
        del dense_in, q_in, k_in

        if sum_q:
            grad_q_group_sum = grad_q_flat.view(B, h_kv, L, D).permute(0, 2, 1, 3)
            del grad_q_flat
            grad_lmks = grad_k_flat.view(B, h_kv, S, D).permute(0, 2, 1, 3)
            del grad_k_flat
            grad_q_grouped = grad_q_group_sum.unsqueeze(3).expand(B, L, h_kv, G, D)
            grad_q = grad_q_grouped.reshape(B, L, h_kv * G, D)
        else:
            grad_q = grad_q_flat.view(B, h_q, L, D).permute(0, 2, 1, 3)
            del grad_q_flat
            grad_lmk_group_sum = grad_k_flat.view(B, h_q, S, D).permute(0, 2, 1, 3)
            del grad_k_flat
            grad_lmk_grouped = grad_lmk_group_sum.unsqueeze(3).expand(B, S, h_q, G, D)
            grad_lmks = grad_lmk_grouped.reshape(B, S, h_q * G, D)

        grad_q = grad_q * sm_scale
        grad_lmks = grad_lmks * sm_scale

        if d_reshape:
            grad_lmks = grad_lmks.reshape(B, S, orig_h_kv, orig_D2)

        return grad_q, grad_lmks, None, None, None, None, None, None, None, None


class OnlineTopK_Unified(torch.nn.Module):
    """
    统一的 TopK Module，同时支持 dense (pretrain) 和 varlen (SFT packing) 场景。

    dense 模式: forward(q, lmks, q_offset, drop_mask=None)
    varlen 模式: forward(q, lmks, cu_seq_lens)
    """

    def __init__(self, topk, block_size, window_size, is_causal,
                 is_training=True,
                 use_drop_mask=False, is_varlen=False,
                 force_recent_chunks=0):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self.is_training = is_training
        self.use_drop_mask = use_drop_mask
        self.is_varlen = is_varlen
        self.force_recent_chunks = force_recent_chunks

    def forward(self, q, lmks, *args, **kwargs):
        B, L, h_q, D = q.shape
        _, S, h_kv, _ = lmks.shape
        h_shared = min(h_q, h_kv)

        fwd_kernel = fused_topk_forward_kernel_insert_unified(
            1 if self.is_varlen else B,
            h_shared, D, self.topk,
            self.block_size, self.window_size, self.is_causal,
            seq_len=None if (self.is_varlen or not self.is_training) else L,
            s_len=None if (self.is_varlen or not self.is_training) else S,
            is_training=self.is_training,
            use_drop_mask=self.use_drop_mask,
            is_varlen=self.is_varlen,
            force_recent_chunks=self.force_recent_chunks,
        )

        sort_kernel = sort_topk_indices_scores_kernel(
            1 if self.is_varlen else B,
            h_shared, self.topk,
            seq_len=None if (self.is_varlen or not self.is_training) else L,
            s_len=S if self.is_varlen else (S if self.is_training else None),
            is_training=self.is_training if not self.is_varlen else False,
        )

        if self.is_varlen:
            cu_seq_lens = args[0] if len(args) > 0 else kwargs['cu_seq_lens']

            # 从 token 级别的 cu_seq_lens 自动构造 landmark 级别的 cu_seq_lens_k
            sub_lengths = cu_seq_lens[1:] - cu_seq_lens[:-1]
            sub_lmk_counts = sub_lengths // self.block_size
            cu_seq_lens_k = torch.zeros_like(cu_seq_lens)
            cu_seq_lens_k[1:] = sub_lmk_counts.cumsum(0).to(cu_seq_lens.dtype)

            scores, indices = OnlineTopKUnifiedFn.apply(
                q, lmks, self.topk, fwd_kernel, sort_kernel,
                True,
                None, None,
                cu_seq_lens.contiguous(), cu_seq_lens_k.contiguous(),
            )
        else:
            q_offset = args[0] if len(args) > 0 else kwargs.get('q_offset', 0)
            drop_mask = args[1] if len(args) > 1 else kwargs.get('drop_mask', None)

            q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=q.device)

            scores, indices = OnlineTopKUnifiedFn.apply(
                q, lmks, self.topk, fwd_kernel, sort_kernel,
                False,
                q_offset_tensor, drop_mask,
                None, None,
            )

        return indices, scores


def online_topk_group(
    q: torch.Tensor,
    lmks: torch.Tensor,
    topk: int,
    block_size: int,
    window_size: int,
    is_causal: bool = True,
    q_offset: int = 0,
    is_training: bool = True,
    drop_mask: torch.Tensor = None,
    cu_seq_lens: torch.Tensor = None,
    force_recent_chunks: int = 0,
):
    """
    统一的 TopK Functional API，同时支持 dense 和 varlen 两种路径。

    根据是否传入 cu_seq_lens 自动判断模式：
    - 不传 → dense 模式 (pretrain / inference)
    - 传入 → varlen 模式 (SFT packing，多条子序列打包成一个 packed tensor)

    功能说明:
        对每个 query token，在其可见的 landmark key chunks 中选出 score 最高的 topk 个，
        返回排序后的 chunk indices 和对应 scores。支持 causal mask、sliding window、
        memory window 等约束。支持 autograd 反向传播。

    Args:
        q (torch.Tensor):
            Query tensor。
            - dense 模式: shape = [B, L, h_q, D]，B 为 batch size，L 为 query 序列长度
            - varlen 模式: shape = [1, L_total, h_q, D]，L_total 为所有子序列 query 长度之和
        lmks (torch.Tensor):
            Landmark key tensor（每个 chunk 的代表性 key）。
            - dense 模式: shape = [B, S, h_kv, D]，S = ceil(seq_len / block_size) 为 chunk 数
            - varlen 模式: shape = [1, S_total, h_kv, D]，S_total 为所有子序列 landmark 数之和
            注意: 如果 lmks 的 D 维度是 q 的整数倍（D_lmk = D_q * ratio），会自动
            reshape 为 [B, S, h_kv * ratio, D_q] 来对齐头维度。
        topk (int):
            每个 query token 选出的 top-k chunk 数量。
        block_size (int):
            Chunk 大小（= chunk_size），即每个 landmark 代表的 token 数。
        window_size (int):
            Sliding window 大小（= hsa_sliding_window）。causal 模式下，query 只能看到
            chunk_idx >= (q_global_pos - window_size + 1) // block_size 的 chunks。
        is_causal (bool, default=True):
            是否启用 causal mask。True 时 query 只能看到它之前的 chunks。
        q_offset (int, default=0):
            Dense 模式专用。query 在 KV cache 中的全局偏移量，用于 inference 时
            正确计算 causal 边界。训练时通常为 0。varlen 模式下忽略此参数。
        is_training (bool, default=True):
            是否为训练模式。False 时 seq_len/s_len 使用 dynamic shape 以避免
            不同序列长度导致的 kernel 重编译。
        drop_mask (torch.Tensor, optional):
            Dense 模式专用。shape = [B, L, S]，dtype = int32，0/1 bitmap。
            1 表示该 chunk 被 drop，不参与 topk 选择。varlen 模式下忽略此参数。
        cu_seq_lens (torch.Tensor, optional):
            Varlen 模式专用。Token 级别的 cumulative sequence lengths。
            shape = [num_seqs + 1]，dtype = int32。
            例如 3 条子序列长度分别为 [100, 200, 150]，则 cu_seq_lens = [0, 100, 300, 450]。
            接口内部会根据 cu_seq_lens 和 block_size 自动计算 landmark 级别的
            cu_seq_lens_k：每个子序列的 landmark 数 = L_i // block_size。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (indices, scores)
        - indices (torch.Tensor):
            选出的 chunk indices，按升序排列。
            - dense 模式: shape = [B, L, h_shared, topk]，值域 [0, S)
            - varlen 模式: shape = [1, L_total, h_shared, topk]，值域为 **global packed index**，
              可直接用于在 packed lmks tensor 上索引。
            - 无效位（causal mask 导致可见 chunks < topk）填充为 -1。
            - h_shared = min(h_q, h_kv)
        - scores (torch.Tensor):
            对应的 attention scores（经过 softmax scale 但未经 softmax）。
            shape 同 indices。无效位填充为 -inf。

    使用示例:
        # === Dense 模式 (pretrain) ===
        indices, scores = online_topk_group(
            q,              # [B, L, h_q, D]
            lmks,           # [B, S, h_kv, D]
            topk=4,
            block_size=64,
            window_size=512,
            is_causal=True,
        )

        # === Dense 模式 (inference, 带 q_offset) ===
        indices, scores = online_topk_group(
            q,              # [B, 1, h_q, D]  (单 token decode)
            lmks,           # [B, S, h_kv, D]
            topk=4,
            block_size=64,
            window_size=512,
            is_causal=True,
            q_offset=1023,  # 当前 token 在 KV cache 中的位置
            is_training=False,
        )

        # === Varlen 模式 (SFT packing) ===
        indices, scores = online_topk_group(
            q_packed,       # [1, L_total, h_q, D]
            lmks_packed,    # [1, S_total, h_kv, D]
            topk=4,
            block_size=64,
            window_size=512,
            is_causal=True,
            cu_seq_lens=cu_q,  # [num_seqs + 1], e.g. [0, 100, 300, 450]
        )
        # indices 是 global packed index，可直接索引: lmks_packed[:, indices, ...]

    注意事项:
        1. varlen 模式下 batch 维度必须为 1（packed tensor）。
        2. varlen 模式下 num_seqs、seq_len、s_len 均为 dynamic 变量，
           子序列个数或长度变化不会触发 kernel 重编译。
        3. varlen 模式下输出的 indices 是 global packed index（不是子序列内的 local index）。
        4. 此接口支持 autograd 反向传播（梯度流回 q 和 lmks）。
        5. h_q 和 h_kv 可以不同（GQA/MQA），内部会自动做 group sum。
        6. cu_seq_lens_k 由接口内部自动构造：S_i = L_i // block_size。
           调用方只需传入 token 级别的 cu_seq_lens 即可。
    """
    is_varlen = cu_seq_lens is not None
    use_drop_mask = drop_mask is not None
    module = OnlineTopK_Unified(
        topk, block_size, window_size, is_causal,
        is_training=is_training,
        use_drop_mask=use_drop_mask,
        is_varlen=is_varlen,
        force_recent_chunks=force_recent_chunks,
    )

    if is_varlen:
        return module(q, lmks, cu_seq_lens)
    else:
        return module(q, lmks, q_offset, drop_mask)







import torch.nn.functional as F
def ref_topk_forward_with_grad(q, lmks, topk, block_size, window_size, is_causal, dtype, q_offset: int = 0, drop_mask: Optional[torch.Tensor] = None, force_recent_chunks: int = 0):
    B, L, h_q, D = q.shape
    _, S, h_kv, D2 = lmks.shape
    sm_scale = 1.0 / math.sqrt(D)
    # 处理D维度不等的情况：按q的D维度对lmk切分，等效转换为头维度的差异
    if D != D2:
        assert D2 % D == 0, f"lmks D dim ({D2}) must be divisible by q D dim ({D})"
        d_ratio = D2 // D
        lmks = lmks.reshape(B, S, h_kv * d_ratio, D)
        h_kv = h_kv * d_ratio
    h_shared = min(h_q, h_kv)
    # 自动判断谁的头维度大就sum谁
    if h_q >= h_kv:
        G = h_q // h_kv
        q_group_sum = q.view(B, L, h_kv, G, D).sum(dim=3)  # [B, L, h_kv, D]
        scores_ref = torch.einsum("blkd,bskd->blks", q_group_sum.to(dtype), lmks.to(dtype)) * sm_scale
    else:
        G = h_kv // h_q
        lmk_group_sum = lmks.view(B, S, h_q, G, D).sum(dim=3)  # [B, S, h_q, D]
        scores_ref = torch.einsum("blkd,bskd->blks", q.to(dtype), lmk_group_sum.to(dtype)) * sm_scale
    
    # scores_ref: [B, L, h_shared, S]
    effective_topk = topk - force_recent_chunks

    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1)
        i_idx_global = i_idx + q_offset
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)

        limit_chunk_idx = (i_idx_global - window_size + 1).div(block_size, rounding_mode='floor')  # [L, 1]
        mask = j_idx >= limit_chunk_idx  # causal mask: 不可见的 chunk

        if force_recent_chunks > 0:
            # recent 区间: [recent_start, limit_chunk)，从竞争部分排除
            # 与 kernel 对齐: recent_start = max(limit_chunk - force_recent_chunks, 0)
            recent_start_idx = torch.clamp(limit_chunk_idx - force_recent_chunks, min=0)  # [L, 1]
            recent_mask = (j_idx >= recent_start_idx) & (j_idx < limit_chunk_idx)  # [L, S]
            # 竞争部分的 mask: 原始 causal mask + recent 区间
            compete_mask = mask | recent_mask  # [L, S]
        else:
            compete_mask = mask

        scores_compete = scores_ref.masked_fill(compete_mask.unsqueeze(0).unsqueeze(2), float('-inf'))
    else:
        scores_compete = scores_ref.clone()
    
    if drop_mask is not None:
        # drop_mask: (B, L, S), 1表示drop
        drop_bool = drop_mask.bool()  # (B, L, S)
        scores_compete = scores_compete.masked_fill(drop_bool.unsqueeze(2), float('-inf'))

    # 竞争部分 topk
    scores_topk, indices_topk = torch.topk(scores_compete, k=effective_topk, dim=-1, sorted=False)

    if force_recent_chunks > 0 and is_causal:
        # 收集 recent chunk 的真实 score
        # recent_start_idx: [L, 1], limit_chunk_idx: [L, 1]
        # 对每个 query，recent chunk indices = [recent_start, recent_start+1, ..., recent_start+F-1]
        recent_offsets = torch.arange(force_recent_chunks, device=q.device)  # [F]
        recent_indices_per_q = recent_start_idx.squeeze(1).unsqueeze(-1) + recent_offsets  # [L, F]
        recent_indices_per_q = recent_indices_per_q.long()

        # 标记无效的 recent index（< 0 或 >= S），与 kernel 行为对齐：
        # kernel 中 recent_scores 初始化为 -inf，只有 effective_s 在 [recent_start, limit_chunk) 
        # 且 effective_s 在 [0, S) 范围内时才会被填充真实 score
        invalid_recent = (recent_indices_per_q < 0) | (recent_indices_per_q >= S)  # [L, F]

        # 从 scores_ref [B, L, h_shared, S] 中 gather recent scores
        # 扩展 recent_indices_per_q 到 [B, L, h_shared, F]
        ri_expanded = recent_indices_per_q.unsqueeze(0).unsqueeze(2).expand(B, L, h_shared, force_recent_chunks).clone()
        # clamp 防止越界（仅用于 gather，无效位后续会被覆盖为 -inf）
        ri_safe = ri_expanded.clamp(0, S - 1)
        recent_scores = torch.gather(scores_ref, dim=-1, index=ri_safe)  # [B, L, h_shared, F]

        # 将无效的 recent 位设为 -inf（与 kernel 中 recent_scores 初始化为 -inf 对齐）
        invalid_recent_expanded = invalid_recent.unsqueeze(0).unsqueeze(2).expand(B, L, h_shared, force_recent_chunks)
        recent_scores[invalid_recent_expanded] = float('-inf')
        ri_expanded[invalid_recent_expanded] = -1

        # 拼接: [竞争部分 | recent 部分]
        indices_topk = torch.cat([indices_topk, ri_expanded], dim=-1)  # [B, L, h_shared, topk]
        scores_topk = torch.cat([scores_topk, recent_scores], dim=-1)  # [B, L, h_shared, topk]
    
    is_invalid = scores_topk == float('-inf')
    
    sort_keys = indices_topk.clone()
    sort_keys[is_invalid] = S + 1
    
    _, order = torch.sort(sort_keys, dim=-1)
    
    indices_sorted = torch.gather(indices_topk, -1, order)
    scores_sorted = torch.gather(scores_topk, -1, order)
    
    indices_sorted[scores_sorted == float('-inf')] = -1
    
    return indices_sorted, scores_sorted




def test_online_topk_fused_correctness():
    print("\n" + "=" * 70)
    print("=== Testing OnlineTopK_Fused Correctness ===")
    print("=" * 70)
    
    B, L, D = 64, 4096, 128
    h_q = 16
    h_kv = 2
    S = 64+2
    topk = 16
    is_causal =True
    block_size = 64
    window_size = 100
    q_offset = 128
    
    dtype = torch.bfloat16
    device = "cuda"
    
    print(f"Config: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}, window_size={window_size}, q_offset={q_offset}")
    
    torch.manual_seed(42)
    q = torch.randn(B, L, h_q, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)
    
    # ============ Forward Correctness ============
    print("\n--- Forward Correctness ---")
    
    # 修改 1: 传入 q_offset
    indices_fused, scores_fused = online_topk_group(q, lmks, topk, block_size, window_size, is_causal, q_offset=q_offset, is_training=False)
    indices_ref, scores_ref = ref_topk_forward_with_grad(q, lmks, topk, block_size, window_size, is_causal, dtype=torch.float32, q_offset=q_offset)
    
    print("indices_fused sample:", indices_fused[0,2500,0,:])
    print("scores_fused sample:", scores_fused[0,2500,0,:])

    print("indices_ref sample:", indices_ref[0,2500,0,:])
    print("scores_ref sample:", scores_ref[0,2500,0,:])


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
        # 修改 2: Ref 前向检查也传入 q_offset
        _, scores_ref_check = ref_topk_forward_with_grad(q_ref, lmks_ref, topk, block_size, window_size, is_causal, dtype=torch.float32, q_offset=q_offset)
        invalid_mask = (scores_ref_check < -1e9)
        grad_output[invalid_mask] = 0.0

    # 修改 3: Fused 后向验证传入 q_offset
    indices_fused_bwd, scores_fused_bwd = online_topk_group(q_fused, lmks_fused, topk, block_size, window_size, is_causal, q_offset=q_offset, is_training=False)
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
        
        # 修改 4: 手动构建 Mask 时加入 q_offset 偏移
        i_idx_global = i_idx + q_offset 
        
        # New Mask Logic
        limit_chunk_idx = (i_idx_global - window_size + 1).div(block_size, rounding_mode='floor')
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
    
    B, L, D = 16, 4096*2, 64
    h_q = 16
    h_kv = 4
    topk = 32
    is_causal =True
    block_size = 64
    S = L // block_size
    window_size = 512
    
    dtype = torch.bfloat16
    device = "cuda"
    
    print(f"Config: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}, window_size={window_size}")
    
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
        
        # Warmup
        for _ in range(5):
            q_t.grad = None
            lmks_t.grad = None
            # 修复点1: 确保每次前向的结果都被用来计算loss，不引用旧变量
            _, scores = online_topk_group(q_t, lmks_t, topk, block_size, window_size, is_causal)
            loss = (scores * grad_output).sum()
            loss.backward()
        torch.cuda.synchronize()
        
        # Memory Check
        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        # 修复点2: 正确获取当前Graph的 scores
        _, scores = online_topk_group(q_t, lmks_t, topk, block_size, window_size, is_causal)
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
            # 注意: 如果不做backward，PyTorch会累计计算图。对于测速来说，最好是在no_grad下测纯算子时间，
            # 或者这里只是测建图+执行时间。为了避免OOM，这里不backward时就不保留Graph。
            # 但由于online_topk_group是autograd function，不使用no_grad会建图。
            # 对于20次迭代一般没事。
            _, _ = online_topk_group(q_t, lmks_t, topk, block_size, window_size, is_causal)
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
    
    # try:
    #     mem_ref, fwd_ref, all_ref, bwd_ref = run_ref()
    #     print(f"\n[Reference (torch.topk)]")
    #     print(f"  Peak Memory: {mem_ref:.2f} MB")
    #     print(f"  Avg Fwd Latency: {fwd_ref:.2f} ms")
    #     print(f"  Avg Fwd+Bwd Latency: {all_ref:.2f} ms")
    #     print(f"  Derived Bwd Latency: {bwd_ref:.2f} ms")
        
    # except RuntimeError as e:
    #     if "out of memory" in str(e).lower():
    #         print(f"\n[Reference (torch.topk)] OOM - Cannot run with this config")
    #         print("\n" + "-" * 70)
    #         print("Comparison (Reference OOM):")
    #         print("-" * 70)
    #         print(f"{'Method':<25} {'Memory (MB)':<15} {'Fwd (ms)':<12} {'Fwd+Bwd (ms)':<15} {'Bwd (ms)':<12}")
    #         print("-" * 70)
    #         print(f"{'OnlineTopK_Fused':<25} {mem_fused:<15.2f} {fwd_fused:<12.2f} {all_fused:<15.2f} {bwd_fused:<12.2f}")
    #         print("-" * 70)
    #     else:
    #         raise e


try:
    import pytest
except ImportError:
    pytest = None

_parametrize = pytest.mark.parametrize if pytest is not None else lambda *a, **kw: (lambda fn: fn)

@_parametrize("B, L, S, h_kv, G, D, topk, block_size, window_size, q_offset", [
    (2, 4096, 64, 2, 8, 64, 16, 64, 64, 0),       # 原始测试: 偏移 0
    (2, 4096, 64, 2, 8, 64, 16, 64, 64, 128),    # 新增测试: 带偏移
    (1, 2048, 64, 1, 8, 64, 16, 32, 32, 1024),   # 窄窗口 + 大偏移
    (1, 2500, 64, 1, 8, 64, 8, 32, 100, 64),    # 小偏移
])
def test_topk_correctness_robust(B, L, S, h_kv, G, D, topk, block_size, window_size, q_offset, sum_kv=False, D_kv=None):
    """
    Args:
        sum_kv: 如果为True，表示h_kv > h_q的场景（对lmk做sum），此时G表示h_kv/h_q的倍数。
                如果为False（默认），表示h_q >= h_kv的场景（对q做sum），此时G表示h_q/h_kv的倍数。
        D_kv: lmk的D维度，如果为None则等于D（q的D维度）。当D_kv != D时，测试D维度不等的场景。
    """
    device = "cuda"
    dtype = torch.bfloat16
    is_causal = True
    if D_kv is None:
        D_kv = D
    if sum_kv:
        # h_kv > h_q: h_kv = h_kv参数, h_q = h_kv / G
        h_q = h_kv // G
        assert h_kv % h_q == 0, f"sum_kv模式下 h_kv={h_kv} 必须能被 G={G} 整除"
    else:
        # h_q >= h_kv: 原有逻辑
        h_q = h_kv * G
    is_training = False
    
    torch.manual_seed(42)
    
    sum_mode = "sum_kv(lmk)" if sum_kv else "sum_q"
    d_info = f", D_kv={D_kv}" if D_kv != D else ""
    print(f"\nTesting Config: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, G={G}, D={D}{d_info}, topk={topk}, BS={block_size}, Win={window_size}, q_offset={q_offset}, mode={sum_mode}")

    # 1.准备数据
    # Q: [B, L, h_q, D]
    q_raw = torch.randn(B, L, h_q, D, dtype=dtype, device=device)
    # Lmks: [B, S, h_kv, D_kv]
    lmks_raw = torch.randn(B, S, h_kv, D_kv, dtype=dtype, device=device)
    
    # 2.输入隔离 (Clone & Detach)
    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)
    
    q_ref = q_raw.clone().detach().requires_grad_(True)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(True)

    # 3.前向计算
    # Fused Kernel - 传入 q_offset
    indices_fused, scores_fused = online_topk_group(
        q_fused, lmks_fused, topk, block_size, window_size, is_causal, q_offset=q_offset, is_training=is_training
    )

    # Reference - 传入 q_offset
    indices_ref, scores_ref = ref_topk_forward_with_grad(
        q_ref, lmks_ref, topk, block_size, window_size, is_causal, dtype=torch.float32, q_offset=q_offset
    )

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
    # 生成 grad_output [B, L, h_shared, topk]  (h_shared = min(h_q, h_kv))
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
    # 先处理D维度不等的情况：与OnlineTopKFusedFn.forward一致
    lmks_ref_proc = lmks_ref
    h_kv_proc = h_kv
    if D_kv != D:
        d_ratio = D_kv // D
        lmks_ref_proc = lmks_ref.reshape(B, S, h_kv * d_ratio, D)
        h_kv_proc = h_kv * d_ratio
    # 1. 计算 Full Scores [B, L, h_shared, S]
    h_shared = min(h_q, h_kv_proc)
    G_proc = max(h_q, h_kv_proc) // h_shared
    if h_q >= h_kv_proc:
        # 对q做sum
        q_grouped_ref = q_ref.view(B, L, h_kv_proc, G_proc, D)
        q_group_sum_ref = q_grouped_ref.sum(dim=3) # [B, L, h_kv_proc, D]
        scores_full = torch.einsum("blkd,bskd->blks", q_group_sum_ref.float(), lmks_ref_proc.float())
    else:
        # 对lmk做sum
        lmk_grouped_ref = lmks_ref_proc.view(B, S, h_q, G_proc, D)
        lmk_group_sum_ref = lmk_grouped_ref.sum(dim=3) # [B, S, h_q, D]
        scores_full = torch.einsum("blkd,bskd->blks", q_ref.float(), lmk_group_sum_ref.float())
    scores_full = scores_full * (1.0 / math.sqrt(D))
    
    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        
        # New Mask Logic (Apply q_offset)
        i_idx_global = i_idx + q_offset  # 全局索引
        
        limit_chunk_idx = (i_idx_global - window_size + 1).div(block_size, rounding_mode='floor')
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

    print(f"Test Passed: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, G={G}, D={D}, D_kv={D_kv}, topk={topk}, q_offset={q_offset}, mode={sum_mode}")



def test_recompile_different_seq_len():
    """
    测试两次调用 fused_topk_forward_kernel_insert 但 seq_len/s_len 不同时，
    kernel 是否会重新编译。
    使用 is_training=False 开启动态 shape，只测试前向。
    """
    import time
    import logging

    # 开启 tilelang debug 日志来观察编译信息
    tl_logger = logging.getLogger("tilelang")
    old_level = tl_logger.level
    tl_logger.setLevel(logging.DEBUG)
    if not tl_logger.handlers:
        tl_logger.addHandler(logging.StreamHandler())

    device = "cuda"
    dtype = torch.bfloat16

    B = 2
    h_q = 16
    h_kv = 2
    G = h_q // h_kv
    D = 64
    topk = 16
    block_size = 64
    window_size = 64
    is_causal = True
    q_offset = 0
    is_training = False
    SEQ_LEN_1 = 512*2
    SEQ_LEN_2 = 1024*2

    S_1 = SEQ_LEN_1 // block_size
    S_2 = SEQ_LEN_2 // block_size

    def make_inputs(seq_len, s_len):
        torch.manual_seed(42)
        q = torch.randn(B, seq_len, h_q, D, dtype=dtype, device=device)
        lmks = torch.randn(B, s_len, h_kv, D, dtype=dtype, device=device)
        return q, lmks

    print("=" * 70)
    print("测试 topk kernel 是否因 seq_len 变化而重新编译")
    print(f"is_training={is_training}（动态 shape）")
    print("=" * 70)

    # ========== 第1次调用：SEQ_LEN_1 ==========
    print(f"\n[第1次调用] SEQ_LEN={SEQ_LEN_1}, S={S_1}")
    q1, lmks1 = make_inputs(SEQ_LEN_1, S_1)
    torch.cuda.synchronize()
    t0 = time.time()
    indices1, scores1 = online_topk_group(q1, lmks1, topk, block_size, window_size, is_causal,
                                   q_offset=q_offset, is_training=is_training)
    torch.cuda.synchronize()
    t1 = time.time()
    time_1 = t1 - t0
    print(f"  耗时: {time_1:.4f}s")

    # ========== 第2次调用：SEQ_LEN_2（不同长度） ==========
    print(f"\n[第2次调用] SEQ_LEN={SEQ_LEN_2}, S={S_2}")
    q2, lmks2 = make_inputs(SEQ_LEN_2, S_2)
    torch.cuda.synchronize()
    t0 = time.time()
    indices2, scores2 = online_topk_group(q2, lmks2, topk, block_size, window_size, is_causal,
                                   q_offset=q_offset, is_training=is_training)
    torch.cuda.synchronize()
    t1 = time.time()
    time_2 = t1 - t0
    print(f"  耗时: {time_2:.4f}s")

    # ========== 第3次调用：再次 SEQ_LEN_1（测试缓存命中） ==========
    print(f"\n[第3次调用] SEQ_LEN={SEQ_LEN_1}, S={S_1}（与第1次相同，测试缓存命中）")
    q3, lmks3 = make_inputs(SEQ_LEN_1, S_1)
    torch.cuda.synchronize()
    t0 = time.time()
    indices3, scores3 = online_topk_group(q3, lmks3, topk, block_size, window_size, is_causal,
                                   q_offset=q_offset, is_training=is_training)
    torch.cuda.synchronize()
    t1 = time.time()
    time_3 = t1 - t0
    print(f"  耗时: {time_3:.4f}s")

    # ========== 汇总 ==========
    print()
    print("=" * 70)
    print("汇总对比（耗时包含编译时间，编译通常需要数秒）")
    print("=" * 70)
    print(f"  {'调用':<40} {'耗时 (s)':<12}")
    print(f"  {'-'*52}")
    print(f"  {'第1次 SEQ_LEN=' + str(SEQ_LEN_1) + ', S=' + str(S_1):<40} {time_1:<12.4f}")
    print(f"  {'第2次 SEQ_LEN=' + str(SEQ_LEN_2) + ', S=' + str(S_2):<40} {time_2:<12.4f}")
    print(f"  {'第3次 SEQ_LEN=' + str(SEQ_LEN_1) + ', S=' + str(S_1) + '(重复)':<40} {time_3:<12.4f}")
    print()
    print("如果第2次耗时远小于第1次（< 0.5s），说明动态 shape 生效，没有重新编译。")
    print("如果第2次耗时与第1次接近（数秒），说明触发了重新编译。")

    # 恢复日志级别
    tl_logger.setLevel(old_level)


def test_kernel_vs_ref_with_drop_mask(
    B: int = 2,
    L: int = 4096,
    S: int = 64,
    h_kv: int = 2,
    G: int = 8,
    D: int = 64,
    topk: int = 16,
    block_size: int = 64,
    window_size: int = 64,
    topk_dropout: float = 0.3,
    q_offset: int = 0,
    seed: int = 42,
    device: str = "cuda",
):
    """
    验证 kernel 的 DropMask 功能与 ref_topk_forward_with_grad(drop_mask=...) 的一致性。

    流程：
    1. 在 S 维做 Bernoulli 采样生成 0/1 bitmap (B, L, S)
    2. 将相同的 bitmap 分别传给 kernel 和 ref
    3. 比较 kernel 和 ref 的 indices/scores 是否一致
    """
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    h_q = h_kv * G
    is_causal = True

    q = torch.randn(B, L, h_q, D, dtype=dtype, device=device)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)

    # ====== Step 1: 直接生成 0/1 bitmap ======
    drop_mask = (torch.rand(B, L, S, device=device) < topk_dropout).to(torch.int32)  # (B, L, S)

    print("=" * 100)
    print("[test_kernel_vs_ref_with_drop_mask] Kernel DropMask vs Ref 一致性验证")
    print(f"B={B}, L={L}, S={S}, h_kv={h_kv}, G={G}, D={D}, topk={topk}")
    print(f"block_size={block_size}, window_size={window_size}, topk_dropout={topk_dropout}")
    print(f"q_offset={q_offset}")
    print(f"drop_mask中被mask的chunk占比: {drop_mask.float().mean().item():.4f}")

    # ====== Step 2: 调用 kernel（传入 0/1 bitmap）======
    indices_kernel, scores_kernel = online_topk_group(
        q, lmks, topk, block_size, window_size, is_causal,

        q_offset=q_offset,
        is_training=True,
        drop_mask=drop_mask,
    )

    # ====== Step 3: 调用 ref（传入相同的 0/1 bitmap）======
    indices_ref, scores_ref = ref_topk_forward_with_grad(
        q, lmks, topk, block_size, window_size, is_causal,
        dtype=torch.float32,
        q_offset=q_offset,
        drop_mask=drop_mask,
    )

    # ====== Step 5: 比较 ======
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
        base = (x[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    # scores 对比
    scores_abs_err = get_abs_err(scores_ref.float(), scores_kernel.float())
    scores_ratio = get_err_ratio(scores_ref.float(), scores_kernel.float())
    print(f"\nScores  abs_err={scores_abs_err:.6f}, ratio={scores_ratio:.6f}")

    # indices 对比
    valid_mask = (scores_ref > -1e5) & (scores_kernel.float() > -1e5)
    if valid_mask.sum() > 0:
        idx_match_rate = (indices_kernel[valid_mask] == indices_ref[valid_mask]).float().mean().item()
    else:
        idx_match_rate = 1.0
    print(f"Indices match_rate={idx_match_rate*100:.4f}%")

    # 确认被 drop 的 chunk 不会出现在 kernel 结果中
    # 对每个 (b, l) 检查 kernel 返回的 indices 是否包含被 drop 的 chunk
    leaked_count = 0
    checked = 0
    for b in range(B):
        for l_idx in range(L):
            dropped_set = set()
            for s_idx in range(S):
                if int(drop_mask[b, l_idx, s_idx].item()) == 1:
                    dropped_set.add(s_idx)
            for h_idx in range(h_kv):
                for k_idx in range(topk):
                    idx_val = int(indices_kernel[b, l_idx, h_idx, k_idx].item())
                    if idx_val in dropped_set:
                        leaked_count += 1
                checked += 1
    print(f"Leaked drop chunks in kernel results: {leaked_count} (checked {checked} entries)")

     # 打印几个样例
    print("\n--- 样例对比 ---")
    for b in range(min(B, 1)):
        for l_idx in [0, L // 4, L // 2, L - 1]:
            for h_idx in range(min(h_kv, 1)):
                idx_k = indices_kernel[b, l_idx, h_idx].detach().cpu().tolist()
                idx_r = indices_ref[b, l_idx, h_idx].detach().cpu().tolist()
                sc_k = scores_kernel[b, l_idx, h_idx].detach().cpu().tolist()
                sc_r = scores_ref[b, l_idx, h_idx].detach().cpu().tolist()
                dropped = [s for s in range(S) if int(drop_mask[b, l_idx, s].item()) == 1]
                print(f"(b={b}, l={l_idx}, h={h_idx})")
                print(f"  dropped_chunks: {dropped}")
                print(f"  kernel indices: {idx_k}")
                print(f"  ref    indices: {idx_r}")
                print(f"  kernel scores : {[f'{s:.4f}' for s in sc_k]}")
                print(f"  ref    scores : {[f'{s:.4f}' for s in sc_r]}")

    print("=" * 100)

    # 断言
    assert scores_ratio < 0.01, f"Scores ratio too large: {scores_ratio}"
    assert idx_match_rate >= 0.99, f"Indices match rate too low: {idx_match_rate}"
    assert leaked_count == 0, f"Kernel returned {leaked_count} dropped chunks — DropMask not working!"

    print("所有断言通过 ✓\n")



def test_drop_mask_forward_speed():
    """
    测速函数：只测前向，对比不开 drop 和开 drop 的 fused topk 速度。
    开 drop 时在每个 iter 内用 rand < dropout 生成 0/1 bitmap，模拟实际训练开销。
    """
    device = "cuda"
    dtype = torch.bfloat16

    # 参数配置
    B = 64
    L = 8192
    h_q = 16
    h_kv = 4
    D = 64
    topk = 16
    block_size = 64
    window_size = 512
    is_causal = True
    q_offset = 0
    is_training = True
    topk_dropout = 0.2

    S = L // block_size  # 128

    warmup_iters = 20
    bench_iters = 100

    print("=" * 80)
    print("[test_drop_mask_forward_speed] 前向测速：无 drop vs 有 drop (含 Python 端 mask 生成)")
    print(f"B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, D={D}, topk={topk}")
    print(f"block_size={block_size}, window_size={window_size}")
    print(f"topk_dropout={topk_dropout}")
    print(f"mask生成方式: (rand < dropout).to(int32), 形状 (B, L, S)")
    print(f"warmup={warmup_iters}, bench={bench_iters}")
    print("=" * 80)

    torch.manual_seed(42)
    q = torch.randn(B, L, h_q, D, dtype=dtype, device=device)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)

    # ====== Benchmark: 无 drop ======
    print("\n--- 无 DropMask ---")
    for _ in range(warmup_iters):
        _ = online_topk_group(q, lmks, topk, block_size, window_size, is_causal,
q_offset=q_offset,
                              is_training=is_training, drop_mask=None)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(bench_iters):
        _ = online_topk_group(q, lmks, topk, block_size, window_size, is_causal,
q_offset=q_offset,
                              is_training=is_training, drop_mask=None)
    end.record()
    torch.cuda.synchronize()
    fwd_no_drop_ms = start.elapsed_time(end) / bench_iters
    print(f"  Avg Fwd Latency: {fwd_no_drop_ms:.4f} ms")

    # ====== Benchmark: 有 drop (含每iter生成mask) ======
    print(f"\n--- 有 DropMask (topk_dropout={topk_dropout}, 含每iter生成mask) ---")
    for _ in range(warmup_iters):
        drop_mask = (torch.rand(B, L, S, device=device) < topk_dropout).to(torch.int32)
        _ = online_topk_group(q, lmks, topk, block_size, window_size, is_causal,
q_offset=q_offset,
                              is_training=is_training,
                              drop_mask=drop_mask)
    torch.cuda.synchronize()

    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for _ in range(bench_iters):
        drop_mask = (torch.rand(B, L, S, device=device) < topk_dropout).to(torch.int32)
        _ = online_topk_group(q, lmks, topk, block_size, window_size, is_causal,
q_offset=q_offset,
                              is_training=is_training,
                              drop_mask=drop_mask)
    end2.record()
    torch.cuda.synchronize()
    fwd_drop_ms = start2.elapsed_time(end2) / bench_iters
    print(f"  Avg Fwd Latency: {fwd_drop_ms:.4f} ms")

    # ====== 汇总 ======
    overhead = fwd_drop_ms - fwd_no_drop_ms
    overhead_pct = overhead / fwd_no_drop_ms * 100
    print(f"\n{'=' * 80}")
    print(f"| {'配置':<50} | {'Fwd (ms)':>10} |")
    print(f"|{'-'*52}|{'-'*12}|")
    print(f"| {'无 DropMask':<50} | {fwd_no_drop_ms:>10.4f} |")
    print(f"| {f'有 DropMask (dropout={topk_dropout}, 含mask生成)':<50} | {fwd_drop_ms:>10.4f} |")
    print(f"|{'-'*52}|{'-'*12}|")
    print(f"| {'开销 (绝对)':<50} | {overhead:>+10.4f} |")
    print(f"| {'开销 (相对)':<50} | {overhead_pct:>+9.2f}% |")
    print(f"{'=' * 80}")


def test_visualize_dropout_mask_effect():
    """
    在 1 个 8k 长度样本上，测试不同 dropout 概率的实际 mask 效果。

    Mask 生成逻辑 (0/1 bitmap):
      drop_mask = (rand(L, S) < dropout).to(int32)
      每个位置独立以 dropout 概率被 mask。
      期望每 token mask 数 = S * dropout。
    """
    S = 128
    topk = 16
    L = 8192  # 8k token
    dropout_list = [0.05, 0.1, 0.2, 0.3]

    torch.manual_seed(42)

    print("=" * 70)
    print(f"[Dropout Mask 效果分析]  S={S}, topk={topk}, L={L}")
    print(f"生成方式: 0/1 bitmap, drop_mask = (rand(L, S) < dropout).to(int32)")
    print("=" * 70)

    for dropout in dropout_list:
        # === 直接生成 0/1 bitmap ===
        drop_mask = (torch.rand(L, S) < dropout).int()  # (L, S)

        # 统计每 token 被 mask 的 chunk 数
        mask_count = drop_mask.sum(dim=-1)  # (L,)

        print(f"\ndropout={dropout}  (理论期望mask数={S*dropout:.2f})")
        print(f"  每token mask数: 平均={mask_count.float().mean():.4f}, 最小={mask_count.min()}, 最大={mask_count.max()}")

        # mask 数分布
        dist = {}
        for i in range(S + 1):
            cnt = (mask_count == i).sum().item()
            if cnt > 0:
                dist[i] = cnt
        # 只展示前几个和后几个
        dist_items = sorted(dist.items())
        if len(dist_items) > 10:
            shown = dist_items[:5] + [("...", None)] + dist_items[-5:]
        else:
            shown = dist_items
        dist_str = ", ".join(f"{k}个:{v}({v/L*100:.1f}%)" if v is not None else "..." for k, v in shown)
        print(f"  mask数分布: {dist_str}")

        # 抽 5 个 token 展示
        sample_tokens = [0, L//4, L//2, 3*L//4, L-1]
        for t in sample_tokens:
            masked_ids = drop_mask[t].nonzero(as_tuple=True)[0].tolist()
            cnt = len(masked_ids)
            preview = masked_ids[:8]
            suffix = f" ... (共{cnt}个)" if cnt > 8 else ""
            print(f"    token[{t:>5d}]: mask数={cnt}, 被mask的chunk={preview}{suffix}")

    print(f"\n{'=' * 70}")



def test_drop_mask_topk_exclusion():
    """
    验证测试：用 0/1 bitmap 生成 mask，调用 topk kernel，
    检查被 mask 的 chunk 是否确实没有出现在 topk 结果中。
    对几个 token 位置打印详细信息，方便直观观察。
    """
    device = "cuda"
    dtype = torch.bfloat16

    B = 1
    L = 8192
    h_q = 16
    h_kv = 4
    D = 64
    topk = 16
    block_size = 64
    window_size = 512
    is_causal = True
    q_offset = 0
    is_training = True
    topk_dropout = 0.2

    S = L // block_size  # 128

    torch.manual_seed(42)
    q = torch.randn(B, L, h_q, D, dtype=dtype, device=device)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)

    print("=" * 80)
    print("[test_drop_mask_topk_exclusion] 验证被mask的chunk不会出现在topk结果中")
    print(f"B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, D={D}, topk={topk}")
    print(f"block_size={block_size}, window_size={window_size}")
    print(f"topk_dropout={topk_dropout}")
    print("=" * 80)

    # --- 生成 0/1 bitmap mask ---
    drop_mask = (torch.rand(B, L, S, device=device) < topk_dropout).to(torch.int32)  # (B, L, S)

    # --- 调用topk (有mask) ---
    indices_drop, scores_drop = online_topk_group(
        q, lmks, topk, block_size, window_size, is_causal,
q_offset=q_offset,
        is_training=is_training,
        drop_mask=drop_mask,
    )

    # --- 调用topk (无mask) 作为对照 ---
    indices_nodrop, scores_nodrop = online_topk_group(
        q, lmks, topk, block_size, window_size, is_causal,
q_offset=q_offset,
        is_training=is_training,
    )

    # --- 检查所有位置：被mask的chunk是否出现在topk结果中 ---
    leaked_total = 0
    checked_total = 0
    leaked_positions = []

    for b in range(B):
        for l in range(L):
            # 获取该token被mask的chunk id集合
            masked_set = set()
            for s in range(S):
                if int(drop_mask[b, l, s].item()) == 1:
                    masked_set.add(s)

            if len(masked_set) == 0:
                continue  # 该token没有被mask, 跳过

            # 检查所有head的topk结果
            for h in range(h_kv):
                for k in range(topk):
                    idx_val = int(indices_drop[b, l, h, k].item())
                    if idx_val in masked_set:
                        leaked_total += 1
                        if len(leaked_positions) < 10:  # 最多记10个
                            leaked_positions.append((b, l, h, k, idx_val, masked_set))
                    checked_total += 1

    has_mask_count = (drop_mask.sum(dim=-1) > 0).sum().item()  # 有至少1个mask的token数

    print(f"\n统计:")
    print(f"  有mask的token数: {has_mask_count}/{B*L} ({has_mask_count/(B*L)*100:.1f}%)")
    print(f"  每token平均mask chunk数: {drop_mask.float().sum(dim=-1).mean().item():.2f}")
    print(f"  检查的 (token, head, topk) 总数: {checked_total}")
    print(f"  泄露数 (被mask的chunk出现在topk中): {leaked_total}")
    if leaked_total > 0:
        print(f"  ❌ 发现泄露! 前{len(leaked_positions)}个:")
        for (b, l, h, k, idx_val, mset) in leaked_positions:
            print(f"     b={b}, l={l}, h={h}, k={k}: topk中出现了chunk {idx_val}, masked_set={mset}")
    else:
        print(f"  ✅ 无泄露，所有被mask的chunk都未出现在topk结果中")

    # --- 打印几个具体位置的详细信息 ---
    # 从序列后半段选取样例，优先选mask的chunk确实命中了无mask topk的token（有对比效果）
    print(f"\n--- 详细样例 (从序列后半段选取，优先展示mask命中topk的token) ---")

    # 先收集所有候选: 序列后半段、有mask的token
    start_l = L // 2  # 从后半段开始找，确保有足够chunk
    hit_samples = []    # mask命中了无mask topk的 (有对比效果)
    normal_samples = [] # mask没命中topk的 (普通样例)

    for l in range(start_l, L):
        valid_ids = [s for s in range(S) if int(drop_mask[0, l, s].item()) == 1]
        if len(valid_ids) == 0:
            continue

        h = 0
        idx_nodrop = indices_nodrop[0, l, h].detach().cpu().tolist()
        in_nodrop = [c for c in valid_ids if c in idx_nodrop]

        if len(in_nodrop) > 0:
            hit_samples.append(l)
        else:
            normal_samples.append(l)

        # 收集够了就停
        if len(hit_samples) >= 6 and len(normal_samples) >= 2:
            break

    # 优先展示命中的，再补充普通的，共8个
    sample_positions = hit_samples[:6] + normal_samples[:2]
    if len(sample_positions) < 8:
        sample_positions = (hit_samples + normal_samples)[:8]

    print(f"  (共找到 {len(hit_samples)} 个mask命中topk的token, "
          f"{len(normal_samples)} 个未命中的, 展示 {len(sample_positions)} 个)\n")

    for l in sample_positions:
        valid_ids = sorted(s for s in range(S) if int(drop_mask[0, l, s].item()) == 1)

        # 取head 0作为展示
        h = 0
        idx_drop = indices_drop[0, l, h].detach().cpu().tolist()
        idx_nodrop = indices_nodrop[0, l, h].detach().cpu().tolist()
        sc_drop = scores_drop[0, l, h].detach().cpu().tolist()
        sc_nodrop = scores_nodrop[0, l, h].detach().cpu().tolist()

        # 只展示有效的topk (去掉尾部的-1)
        idx_drop_valid = [x for x in idx_drop if x >= 0]
        idx_nodrop_valid = [x for x in idx_nodrop if x >= 0]

        # 检查无mask时该chunk的排名
        in_nodrop = [c for c in valid_ids if c in idx_nodrop_valid]

        # 找出有mask后"顶替上来"的chunk
        set_nodrop = set(idx_nodrop_valid)
        set_drop = set(idx_drop_valid)
        replaced_out = sorted(set_nodrop - set_drop)  # 被mask后消失的
        new_in = sorted(set_drop - set_nodrop)         # 有mask后新进入的

        hit_tag = " ← mask命中topk!" if in_nodrop else ""
        print(f"  token[{l:>5d}] (h=0):{hit_tag}")
        print(f"    被mask的chunk ({len(valid_ids)}个): {valid_ids[:16]}{'...' if len(valid_ids) > 16 else ''}")
        print(f"    无mask topk ({len(idx_nodrop_valid)}个有效): {idx_nodrop_valid}")
        print(f"    有mask topk ({len(idx_drop_valid)}个有效): {idx_drop_valid}")
        if in_nodrop:
            print(f"    被mask的chunk在无mask topk中: {in_nodrop}")
            print(f"    被踢出: {replaced_out}, 新顶替进入: {new_in}")
        else:
            print(f"    被mask的chunk不在无mask topk中, 两次结果相同")
        print()

    print(f"{'=' * 80}")




def test_force_recent_chunks():
    """
    验证 force_recent_chunks 功能的正确性。

    核心验证方案：直接对比 torch ref (ref_topk_forward_with_grad) 和 fused kernel (online_topk_group) 的结果。
    1. force_recent_chunks=0 时，fused 结果与 ref 完全一致（基线验证）
    2. force_recent_chunks>0 时，fused 结果与 ref 的 indices/scores 一致
    3. 不同 force_recent_chunks 值的边界测试
    """
    print("\n" + "=" * 70)
    print("=== Testing force_recent_chunks Correctness ===")
    print("=" * 70)

    dtype = torch.bfloat16
    device = "cuda"
    torch.manual_seed(42)

    # ============ 测试参数 ============
    B, L, D = 2, 2048, 128
    h_q, h_kv = 4, 2
    S = 32  # 32 个 chunk
    topk = 8
    block_size = 64
    window_size = 100
    is_causal = True
    q_offset = 0

    h_shared = min(h_q, h_kv)

    q = torch.randn(B, L, h_q, D, dtype=dtype, device=device)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)

    print(f"Config: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, D={D}")
    print(f"topk={topk}, block_size={block_size}, window_size={window_size}")

    # ============ 验证1: force_recent_chunks=0 基线验证 ============
    # 注意：ref 用 float32 dtype 计算 score（与 kernel 的 float32 累加对齐），
    # 由于 bf16 GEMM 和 float32 GEMM 的精度差异，indices 不会完全一致，
    # 所以用集合匹配率和 score 差异来判断。
    print("\n--- 验证1: force_recent_chunks=0 基线验证 (fused vs ref) ---")
    indices_fused_0, scores_fused_0 = online_topk_group(
        q, lmks, topk, block_size, window_size, is_causal,
q_offset=q_offset,
        is_training=True, force_recent_chunks=0,
    )
    indices_ref_0, scores_ref_0 = ref_topk_forward_with_grad(
        q, lmks, topk, block_size, window_size, is_causal, torch.float32,
q_offset=q_offset,
        force_recent_chunks=0,
    )
    # 用集合匹配率来比较（逐 query 逐 head 比较 index 集合）
    set_match_0 = 0
    set_total_0 = 0
    for b_idx in range(B):
        for l_idx in range(L):
            for h_idx in range(h_shared):
                fused_set = set(indices_fused_0[b_idx, l_idx, h_idx, :].tolist())
                fused_set.discard(-1)
                ref_set = set(indices_ref_0[b_idx, l_idx, h_idx, :].tolist())
                ref_set.discard(-1)
                set_total_0 += 1
                if fused_set == ref_set:
                    set_match_0 += 1
    set_rate_0 = set_match_0 / set_total_0 if set_total_0 > 0 else 1.0
    # scores 比较：过滤掉 -inf 位（避免 -inf - (-inf) = nan）
    valid_scores_mask_0 = (scores_fused_0 > -1e30) & (scores_ref_0 > -1e30)
    if valid_scores_mask_0.any():
        scores_diff_0 = (scores_fused_0.float() - scores_ref_0.float()).abs()[valid_scores_mask_0].max().item()
    else:
        scores_diff_0 = 0.0
    print(f"  indices 集合匹配率: {set_rate_0*100:.2f}% ({set_match_0}/{set_total_0})")
    print(f"  scores 最大差异 (有效位): {scores_diff_0:.6f}")
    if set_rate_0 >= 0.99 and scores_diff_0 < 1.0:
        print("✅ 基线验证通过")
    elif set_rate_0 >= 0.95:
        print("⚠️  基线验证通过 (在 bf16 精度容差内)")
    else:
        print("❌ 基线验证失败")

    # ============ 验证2: force_recent_chunks>0 对比验证 ============
    for force_recent in [1, 2, 3, 4, topk - 1]:
        if force_recent >= topk:
            continue
        print(f"\n--- 验证2: force_recent_chunks={force_recent} (fused vs ref) ---")
        indices_fused, scores_fused = online_topk_group(
            q, lmks, topk, block_size, window_size, is_causal,
q_offset=q_offset,
            is_training=True, force_recent_chunks=force_recent,
        )
        indices_ref, scores_ref = ref_topk_forward_with_grad(
            q, lmks, topk, block_size, window_size, is_causal, torch.float32,
q_offset=q_offset,
            force_recent_chunks=force_recent,
        )

        # indices 对比
        total_elements = indices_fused.numel()
        indices_match = (indices_fused == indices_ref).sum().item()
        indices_rate = indices_match / total_elements

        # scores 对比：只比较两边都有效（非 -inf）的位置，避免 -inf - (-inf) = nan
        valid_scores_mask = (scores_fused > -1e30) & (scores_ref > -1e30)
        if valid_scores_mask.any():
            scores_diff = (scores_fused.float() - scores_ref.float()).abs()
            scores_diff_valid = scores_diff[valid_scores_mask].max().item()
            scores_diff_mean = scores_diff[valid_scores_mask].mean().item()
        else:
            scores_diff_valid = 0.0
            scores_diff_mean = 0.0

        print(f"  indices 匹配率 (element-wise): {indices_rate*100:.2f}% ({indices_match}/{total_elements})")
        print(f"  scores 最大差异 (有效位): {scores_diff_valid:.6f}, 平均差异: {scores_diff_mean:.6f}")

        # 逐 query 详细分析不匹配的情况
        mismatch_count = 0
        mismatch_only_compete = 0  # 仅竞争部分不同（recent 部分一致）
        q_positions = torch.arange(L, device=device) + q_offset
        limit_chunks = (q_positions - window_size + 1) // block_size
        recent_starts = limit_chunks - force_recent

        for b_idx in range(B):
            for l_idx in range(L):
                rs = recent_starts[l_idx].item()
                lc = limit_chunks[l_idx].item()
                if rs < 0 or lc <= 0:
                    continue
                recent_set = set(range(rs, rs + force_recent))
                for h_idx in range(h_shared):
                    fused_set = set(indices_fused[b_idx, l_idx, h_idx, :].tolist())
                    fused_set.discard(-1)
                    ref_set = set(indices_ref[b_idx, l_idx, h_idx, :].tolist())
                    ref_set.discard(-1)
                    if fused_set != ref_set:
                        mismatch_count += 1
                        # 检查 recent 部分是否一致
                        fused_recent = fused_set & recent_set
                        ref_recent = ref_set & recent_set
                        if fused_recent == ref_recent == recent_set:
                            mismatch_only_compete += 1
                        elif mismatch_count <= 3:
                            print(f"    ❌ b={b_idx}, l={l_idx}, h={h_idx}: "
                                  f"fused={sorted(fused_set)}, ref={sorted(ref_set)}, "
                                  f"recent={sorted(recent_set)}")

        if mismatch_count > 0:
            print(f"  集合不匹配数: {mismatch_count}, 其中仅竞争部分不同: {mismatch_only_compete}")
            print(f"  (竞争部分不同通常由 bf16 GEMM 精度差异导致的 tie-breaking 不同)")

        # 判断条件：基于集合匹配（mismatch_count 少）+ scores 差异小
        set_total_2 = B * L * h_shared  # 粗略估计（包含无效 query）
        set_mismatch_rate = mismatch_count / set_total_2 if set_total_2 > 0 else 0.0
        if mismatch_count == 0 and scores_diff_valid < 0.1:
            print(f"✅ force_recent_chunks={force_recent} 验证通过")
        elif set_mismatch_rate < 0.01 and scores_diff_valid < 1.0:
            print(f"✅ force_recent_chunks={force_recent} 验证通过 (集合不匹配 {mismatch_count} 个, 在 bf16 精度容差内)")
        else:
            print(f"❌ force_recent_chunks={force_recent} 验证失败 (集合不匹配率: {set_mismatch_rate*100:.2f}%)")

    # ============ 验证3: 强制选中验证 (recent chunk 必须出现在结果中) ============
    force_recent = 3
    print(f"\n--- 验证3: 强制选中验证 (force_recent_chunks={force_recent}) ---")
    indices_forced, _ = online_topk_group(
        q, lmks, topk, block_size, window_size, is_causal,
q_offset=q_offset,
        is_training=True, force_recent_chunks=force_recent,
    )
    q_positions = torch.arange(L, device=device) + q_offset
    limit_chunks = (q_positions - window_size + 1) // block_size
    recent_starts = limit_chunks - force_recent

    all_correct = True
    num_checked = 0
    num_wrong = 0
    for b_idx in range(B):
        for l_idx in range(L):
            rs = recent_starts[l_idx].item()
            lc = limit_chunks[l_idx].item()
            if rs < 0 or lc <= 0:
                continue
            num_checked += 1
            expected_set = set(range(rs, rs + force_recent))
            for h_idx in range(h_shared):
                actual_set = set(indices_forced[b_idx, l_idx, h_idx, :].tolist())
                actual_set.discard(-1)
                if not expected_set.issubset(actual_set):
                    missing = expected_set - actual_set
                    if num_wrong < 5:
                        print(f"  ❌ b={b_idx}, l={l_idx}, h={h_idx}: "
                              f"expected recent={sorted(expected_set)}, "
                              f"actual topk={sorted(actual_set)}, "
                              f"missing={sorted(missing)}")
                    num_wrong += 1
                    all_correct = False

    if all_correct:
        print(f"✅ 强制选中验证通过 (检查了 {num_checked} 个有效 query × {h_shared} 个 head)")
    else:
        print(f"❌ 强制选中验证失败 ({num_wrong}/{num_checked * h_shared} 个不匹配)")

    # ============ 验证4: ref 自身一致性 (force_recent=0 的 ref 应与不带参数的 ref 一致) ============
    print(f"\n--- 验证4: ref 自身一致性验证 ---")
    indices_ref_base, scores_ref_base = ref_topk_forward_with_grad(
        q, lmks, topk, block_size, window_size, is_causal, torch.float32,
q_offset=q_offset,
    )
    indices_ref_0, scores_ref_0 = ref_topk_forward_with_grad(
        q, lmks, topk, block_size, window_size, is_causal, torch.float32,
q_offset=q_offset,
        force_recent_chunks=0,
    )
    eq_idx = torch.equal(indices_ref_base, indices_ref_0)
    eq_sco = torch.equal(scores_ref_base, scores_ref_0)
    print(f"  indices 完全一致: {eq_idx}")
    print(f"  scores 完全一致: {eq_sco}")
    if eq_idx and eq_sco:
        print("✅ ref 自身一致性验证通过")
    else:
        print("❌ ref 自身一致性验证失败")

    # ============ 演示: 开启/不开启 force_recent_chunks 的选 chunk 区别 ============
    demo_force = 3
    print(f"\n--- 演示: force_recent_chunks=0 vs {demo_force} 的选 chunk 区别 ---")
    # 用 ref 的 float32 结果做演示（更直观）
    indices_demo_0, scores_demo_0 = ref_topk_forward_with_grad(
        q, lmks, topk, block_size, window_size, is_causal, torch.float32,
q_offset=q_offset,
        force_recent_chunks=0,
    )
    indices_demo_f, scores_demo_f = ref_topk_forward_with_grad(
        q, lmks, topk, block_size, window_size, is_causal, torch.float32,
q_offset=q_offset,
        force_recent_chunks=demo_force,
    )
    # 选一个靠后的 query（有足够多可见 chunk），batch=0, head=0
    demo_b, demo_h = 0, 0
    demo_l = L - 1  # 最后一个 query
    q_pos = demo_l + q_offset
    limit_chunk = (q_pos - window_size + 1) // block_size
    recent_start = limit_chunk - demo_force

    idx_0 = indices_demo_0[demo_b, demo_l, demo_h].tolist()
    sco_0 = scores_demo_0[demo_b, demo_l, demo_h].tolist()
    idx_f = indices_demo_f[demo_b, demo_l, demo_h].tolist()
    sco_f = scores_demo_f[demo_b, demo_l, demo_h].tolist()

    # 过滤掉无效位（index < 0 或 score == -inf）
    valid_0 = [(i, f"{s:.4f}") for i, s in zip(idx_0, sco_0) if i >= 0 and s > -1e30]
    valid_f = [(i, f"{s:.4f}") for i, s in zip(idx_f, sco_f) if i >= 0 and s > -1e30]

    recent_range = list(range(max(0, recent_start), limit_chunk))
    set_0 = set(i for i, _ in valid_0)
    set_f = set(i for i, _ in valid_f)

    print(f"  query 位置: B={demo_b}, L={demo_l}, H={demo_h}")
    print(f"  可见 chunk 范围: [0, {limit_chunk})  (共 {limit_chunk} 个)")
    print(f"  recent chunk 范围: [{max(0, recent_start)}, {limit_chunk})  = {recent_range}")
    print(f"")
    print(f"  force_recent_chunks=0 选中的 chunk (按位置排序):")
    print(f"    indices: {[i for i, _ in valid_0]}")
    print(f"    scores:  {[s for _, s in valid_0]}")
    print(f"")
    print(f"  force_recent_chunks={demo_force} 选中的 chunk (按位置排序):")
    print(f"    indices: {[i for i, _ in valid_f]}")
    print(f"    scores:  {[s for _, s in valid_f]}")

    # 高亮差异
    only_in_0 = set_0 - set_f
    only_in_f = set_f - set_0
    forced_in = set(recent_range) & set_f
    forced_not_in_0 = set(recent_range) - set_0

    print(f"")
    print(f"  差异分析:")
    print(f"    recent chunk {recent_range} 在 force=0 中被选中: {set(recent_range) & set_0 or '无'}")
    print(f"    recent chunk {recent_range} 在 force={demo_force} 中被选中: {forced_in or '无'} (强制保证)")
    if forced_not_in_0:
        print(f"    → 原本未被选中但被强制加入的 recent chunk: {forced_not_in_0}")
    if only_in_0:
        print(f"    → 因腾出位置而被挤掉的 chunk: {only_in_0}")
    if not forced_not_in_0 and not only_in_0:
        print(f"    → recent chunk 本身就在 topk 中，结果无差异")

    print("\n" + "=" * 70)
    print("=== force_recent_chunks 测试完成 ===")
    print("=" * 70)


if __name__ == "__main__":
    test_online_topk_fused_correctness()
    test_online_topk_fused_memory_and_speed()
    params_list = [
      # (B,  L,   S, h_kv, G, D, topk, block_size, window_size, q_offset)
        (2, 4096, 64, 2, 8, 64, 16, 64, 64, 0),         # (4096+0)/64 = 64
        (2, 4096, 66, 2, 8, 64, 16, 64, 64, 128),       # (4096+128)/64 = 66
        (1, 2048, 96, 1, 8, 64, 16, 32, 32, 1024),      # (2048+1024)/32 = 96
        (1, 2500, 81, 1, 1, 128, 8, 32, 100, 64),       # (2500+64)/32 ≈ 80.125 -> 81
    ]
    for p in params_list:
        test_topk_correctness_robust(*p)

    # test_recompile_different_seq_len()
    # test_kernel_vs_ref_with_drop_mask()
    # test_drop_mask_forward_speed()
    # test_visualize_dropout_mask_effect()
    # test_drop_mask_topk_exclusion()

    # # === sum_kv 测试: h_kv > h_q, 对lmk做sum ===
    # # 参数说明: (B, L, S, h_kv, G, D, topk, block_size, window_size, memory_window_size, q_offset, sum_kv)
    # # 注意sum_kv=True时: h_kv是lmk的头数, G是倍数, h_q = h_kv // G
    # sum_kv_params = [
    #   # h_kv=8, G=4, h_q=2
    #     (2, 4096, 64, 8, 4, 64, 16, 64, 64, -1, 0, True),
    #   # h_kv=16, G=8, h_q=2
    #     (2, 4096, 66, 16, 8, 64, 16, 64, 64, 512, 128, True),
    #   # h_kv=8, G=8, h_q=1
    #     (1, 2048, 96, 8, 8, 64, 16, 32, 32, 128, 1024, True),
    # ]
    # for p in sum_kv_params:
    #     test_topk_correctness_robust(*p)

    # # === D维度不等测试: h_q == h_kv 但 D_kv != D_q, 在topk内部自动reshape ===
    # # 参数说明: (B, L, S, h_kv, G, D_q, topk, block_size, window_size, memory_window_size, q_offset, sum_kv, D_kv)
    # # 例如 unified_retrieval 场景: q=(B,L,1,1024), lmks=(B,S,1,4096), D_kv/D_q=4
    # d_reshape_params = [
    #   # h_q=1, h_kv=1, D_q=64, D_kv=256 (d_ratio=4)
    #     dict(B=2, L=4096, S=64, h_kv=1, G=1, D=64, topk=16, block_size=64, window_size=64, memory_window_size=-1, q_offset=0, sum_kv=False, D_kv=256),
    #   # h_q=1, h_kv=1, D_q=128, D_kv=512 (d_ratio=4)
    #     dict(B=1, L=2048, S=96, h_kv=1, G=1, D=128, topk=16, block_size=32, window_size=32, memory_window_size=128, q_offset=1024, sum_kv=False, D_kv=512),
    #   # h_q=2, h_kv=2, D_q=64, D_kv=256 (d_ratio=4), 多头+D不等
    #     dict(B=2, L=4096, S=64, h_kv=2, G=1, D=64, topk=16, block_size=64, window_size=64, memory_window_size=-1, q_offset=0, sum_kv=False, D_kv=256),
    #   # h_q=1, h_kv=1, D_q=128, D_kv=1024 (d_ratio=8)
    #     dict(B=1, L=2048, S=96, h_kv=1, G=1, D=128, topk=16, block_size=32, window_size=32, memory_window_size=-1, q_offset=0, sum_kv=False, D_kv=1024),
    # ]
    # for p in d_reshape_params:
    #     test_topk_correctness_robust(**p)

    test_force_recent_chunks()




# python ops/topk_group.py
