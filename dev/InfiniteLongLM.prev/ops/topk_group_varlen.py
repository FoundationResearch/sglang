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
    batch, heads, head_dim, topk,
    block_size, window_size, is_causal,
    seq_len=None, s_len=None,
    memory_window_size=-1,
    BLOCK_L=None, BLOCK_S=None, threads=None,
    is_training=True,
    use_drop_mask=False,
):
    if not is_training:
        seq_len = T.dynamic("seq_len")
        s_len = T.dynamic("s_len")
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    q_shape = [batch, seq_len, heads, head_dim]
    k_shape = [batch, s_len, heads, head_dim]
    out_scores_shape = [batch, seq_len, heads, topk]
    out_indices_shape = [batch, seq_len, heads, topk]
    drop_mask_shape = [batch, seq_len, s_len]

    if BLOCK_L is None:
        BLOCK_L = 32
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
        Q_Offset: T.Tensor([1], "int32"),
        DropMask: T.Tensor(drop_mask_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), heads, batch, threads=threads) as (bx, by, bz):
            q_offset = T.if_then_else(is_training, 0, Q_Offset[0])
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

            for l_idx, d in T.Parallel(BLOCK_L, BLOCK_D):
                tq = base_l + l_idx
                if tq < seq_len:
                    Q_shared[l_idx, d] = Q[b, tq, h, d]
                else:
                    Q_shared[l_idx, d] = T.Cast(dtype, 0.0)

            loop_limit = T.alloc_var("int32")
            loop_limit = num_s_blocks
            if is_causal:
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

                if (my_l < BLOCK_L) and (tq < seq_len):
                    tq_global = q_offset + tq
                    limit_chunk = (tq_global - window_size + 1) // block_size

                    min_chunk = -1
                    if memory_window_size > 0:
                        min_chunk = (tq_global - memory_window_size) // block_size

                    is_valid = T.alloc_var("bool")
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        if ts < s_len:
                            is_valid = (ts >= min_chunk) and ((not is_causal) or (ts < limit_chunk))
                            if use_drop_mask:
                                is_valid = is_valid and (DropMask[b, tq, ts] == 0)
                            if is_valid:
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


class OnlineTopKFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, lmks, topk: int, fwd_kernel, sort_kernel, q_offset_tensor, drop_mask=None):
        B, L, h_q, D = q.shape
        B2, S, h_kv, D2 = lmks.shape
        dtype = q.dtype
        assert B == B2

        # 处理D维度不等的情况：按q的D维度对lmk切分，等效转换为头维度的差异
        d_reshape = False
        orig_h_kv = h_kv
        orig_D2 = D2
        if D != D2:
            assert D2 % D == 0, f"lmks D dim ({D2}) must be divisible by q D dim ({D})"
            d_ratio = D2 // D
            # (B, S, h_kv, D2) -> (B, S, h_kv * d_ratio, D)
            lmks = lmks.reshape(B, S, h_kv * d_ratio, D)
            h_kv = h_kv * d_ratio
            D2 = D
            d_reshape = True

        if h_q >= h_kv:
            assert h_q % h_kv == 0
            G = h_q // h_kv
            sum_q = True
            h_shared = h_kv

            q_group_sum = q.view(B, L, h_kv, G, D).sum(dim=3)  # [B, L, h_kv, D]
            q_in = q_group_sum.to(dtype).contiguous()
            k_in = lmks.contiguous()
        else:
            assert h_kv % h_q == 0
            G = h_kv // h_q
            sum_q = False
            h_shared = h_q

            lmk_group_sum = lmks.view(B, S, h_q, G, D).sum(dim=3)  # [B, S, h_q, D]
            q_in = q.contiguous()
            k_in = lmk_group_sum.to(dtype).contiguous()

        best_scores_buf, best_indices_buf = fwd_kernel(q_in, k_in, q_offset_tensor, drop_mask)

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
        bs_h = B * h_shared
        
        sm_scale = 1.0 / math.sqrt(D)
        
        grad_scores_dense = torch.zeros(
            (B, L, h_shared, S), 
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

        # 如果forward中对lmks做了D维度的reshape，需要将梯度恢复到原始形状
        if d_reshape:
            # grad_lmks当前形状: (B, S, h_kv, D)，其中h_kv是reshape后的
            # 需要恢复为: (B, S, orig_h_kv, orig_D2)
            grad_lmks = grad_lmks.reshape(B, S, orig_h_kv, orig_D2)
        
        return grad_q, grad_lmks, None, None, None, None, None


class OnlineTopK_Fused(torch.nn.Module):
    def __init__(self, topk, block_size, window_size, is_causal, memory_window_size, is_training: bool = True, use_drop_mask: bool = False):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self.memory_window_size = memory_window_size
        self.is_training = is_training
        self.use_drop_mask = use_drop_mask

    def forward(self, q, lmks, q_offset, drop_mask=None):
        B, L, h_q, D = q.shape
        _, S, h_kv, _ = lmks.shape
        h_shared = min(h_q, h_kv)
        topk = self.topk
        block_size = self.block_size
        window_size = self.window_size
        is_causal = self.is_causal
        memory_window_size = self.memory_window_size
        is_training = self.is_training
        use_drop_mask = self.use_drop_mask

        fwd_kernel = fused_topk_forward_kernel_insert(
            B, h_shared, D, topk, block_size, window_size, is_causal,
            seq_len=L if is_training else None,
            s_len=S if is_training else None,
            memory_window_size=memory_window_size,
            is_training=is_training,
            use_drop_mask=use_drop_mask,
        )

        sort_kernel = sort_topk_indices_scores_kernel(
            B, h_shared, topk,
            seq_len=L if is_training else None,
            s_len=S if is_training else None,
            is_training=is_training,
        )
        
        q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=q.device)

        scores, indices = OnlineTopKFusedFn.apply(
            q, lmks, topk, fwd_kernel, sort_kernel, q_offset_tensor, drop_mask
        )
        return indices, scores


def online_topk_group(q: torch.Tensor, lmks: torch.Tensor, topk: int, block_size: int, window_size: int, is_causal: bool = False, memory_window_size: int = -1, q_offset: int = 0, is_training: bool = True, drop_mask: torch.Tensor = None):
    """
    Functional API for OnlineTopK_Fused
    
    Args:
        q: [B, L, h_q, D]
        lmks: [B, S, h_kv, D]
        topk: int
        block_size: int
        window_size: int
        is_causal: bool
        memory_window_size: int, optional. If > 0, limits the memory retrieval to [current - memory_window_size, current].
        q_offset: int, optional. The global offset of the first query in the kv cache.
        is_training: bool, optional. If False, use dynamic shape to avoid recompilation for different seq_len.
        drop_mask: [B, L, S] int32 tensor, optional. 0/1 bitmap, 1 means drop this chunk from topk selection.

    Returns:
        indices: [B, L, min(h_q,h_kv), topk]
        scores: [B, L, min(h_q,h_kv), topk]
    """
    if is_causal:
        B, L, _, _ = q.shape
        _, S, _, _ = lmks.shape
        min_required_s = (q_offset + L - 1) // block_size
        
        assert S >= min_required_s, (
            f"Input mismatch: lmks (S={S}) is too short for query (L={L}) with offset {q_offset}. "
            f"Expected S >= {min_required_s} (calculated from (L + q_offset) / block_size)"
        )

    use_drop_mask = drop_mask is not None
    module = OnlineTopK_Fused(topk, block_size, window_size, is_causal, memory_window_size, is_training=is_training, use_drop_mask=use_drop_mask)
    return module(q, lmks, q_offset, drop_mask=drop_mask)


# ============================================================================
# Varlen TopK Kernel: 独立的 SFT packing / varlen 版本
# ============================================================================
#
# 设计要点：
#   - 输入: packed Q [1, L_total, h, D], packed lmk_k [1, S_total, h_kv, D],
#           cu_seq_lens_q [num_seqs+1], cu_seq_lens_k [num_seqs+1]
#   - 输出: indices [1, L_total, h, topk] (global packed landmark index),
#           scores  [1, L_total, h, topk]
#   - 内部 causal/window/memory_window 判断用 local index
#   - 最终 OutIndices 写 global packed landmark index
#
# 与 dense 版本的区别：
#   - 不使用 batch 维度（B 固定为 1）
#   - 每个 query token 需要通过二分查找确定所属子序列
#   - 每个子序列的 lmk 范围由 cu_seq_lens_k 确定
#   - loop_limit 和 base_s 都是子序列级别的
# ============================================================================

@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_topk_forward_kernel_insert_varlen(
    heads, head_dim, topk,
    block_size, window_size, is_causal,
    seq_len=None, s_len=None,
    memory_window_size=-1,
    BLOCK_L=None, BLOCK_S=None, threads=None,
    use_drop_mask=False,
):
    """
    Varlen TopK 前向 kernel（SFT packing 场景）。

    与 dense 版本的关键区别：
    1. B 固定为 1，通过 cu_seq_lens 区分子序列
    2. 每个 query token 通过线性扫描 cu_seq_lens_q 确定所属子序列
    3. causal/window 判断基于子序列内的 local index
    4. 输出 indices 是 global packed landmark index
    """
    # varlen 场景下 L_total 和 S_total 在训练时已知（打包后的总长度）
    seq_len = T.dynamic("seq_len")
    s_len = T.dynamic("s_len")
    # num_seqs 也设为 dynamic，避免子序列个数变化时重编译
    num_seqs = T.dynamic("num_seqs")

    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    # Q: [1, L_total, heads, head_dim]
    # K: [1, S_total, heads, head_dim]
    # OutScores: [1, L_total, heads, topk]
    # OutIndices: [1, L_total, heads, topk]
    # CuSeqLensQ: [num_seqs + 1]
    # CuSeqLensK: [num_seqs + 1]
    q_shape = [1, seq_len, heads, head_dim]
    k_shape = [1, s_len, heads, head_dim]
    out_scores_shape = [1, seq_len, heads, topk]
    out_indices_shape = [1, seq_len, heads, topk]
    cu_q_shape = [num_seqs + 1]
    cu_k_shape = [num_seqs + 1]
    drop_mask_shape = [1, seq_len, s_len] if use_drop_mask else [1, 1, 1]

    if BLOCK_L is None:
        BLOCK_L = 32
    if BLOCK_S is None:
        BLOCK_S = 16
    BLOCK_D = head_dim
    if threads is None:
        threads = BLOCK_L

    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def fwd_kernel_insert_varlen(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
        CuSeqLensQ: T.Tensor(cu_q_shape, idx_dtype),
        CuSeqLensK: T.Tensor(cu_k_shape, idx_dtype),
        DropMask: T.Tensor(drop_mask_shape, idx_dtype),
    ):
        # Grid: (cdiv(L_total, BLOCK_L), heads, 1)
        # 每个 thread block 处理 BLOCK_L 个连续的 packed query token
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), heads, 1, threads=threads) as (bx, by, bz):
            h = by
            base_l = bx * BLOCK_L

            Q_shared = T.alloc_shared([BLOCK_L, BLOCK_D], dtype)
            K_shared = T.alloc_shared([BLOCK_S, BLOCK_D], dtype)
            score_shared = T.alloc_shared([BLOCK_L, BLOCK_S], accum_dtype)
            acc_s = T.alloc_fragment([BLOCK_L, BLOCK_S], accum_dtype)

            topk_scores = T.alloc_local([topk], accum_dtype)
            topk_indices = T.alloc_local([topk], idx_dtype)

            # 初始化 topk 缓冲
            for kk in T.serial(topk):
                topk_scores[kk] = -T.infinity(accum_dtype)
                topk_indices[kk] = -1

            # 加载 Q tile 到 shared memory
            for l_idx, d in T.Parallel(BLOCK_L, BLOCK_D):
                tq = base_l + l_idx
                if tq < seq_len:
                    Q_shared[l_idx, d] = Q[0, tq, h, d]
                else:
                    Q_shared[l_idx, d] = T.Cast(dtype, 0.0)

            # ============================================================
            # 每个 thread 确定自己的 query token 所属的子序列
            # 然后只在该子序列的 lmk 范围内做 topk
            # ============================================================
            tx = T.get_thread_binding()
            my_l = tx
            tq_global = base_l + my_l  # global packed query index

            # 通过线性扫描 cu_seq_lens_q 确定所属子序列 seq_id
            # 以及子序列内的 local query index
            seq_id = T.alloc_var(idx_dtype)
            seq_id = 0
            for si in T.serial(num_seqs):
                if tq_global >= CuSeqLensQ[si + 1]:
                    seq_id = si + 1

            # 子序列的 q 范围: [q_start, q_end)
            q_start = T.alloc_var(idx_dtype)
            q_start = CuSeqLensQ[seq_id]
            # local query index (子序列内)
            local_q = T.alloc_var(idx_dtype)
            local_q = tq_global - q_start

            # 子序列的 lmk 范围: [k_start, k_end)
            k_start = T.alloc_var(idx_dtype)
            k_end = T.alloc_var(idx_dtype)
            k_start = CuSeqLensK[seq_id]
            k_end = CuSeqLensK[seq_id + 1]
            local_s_len = T.alloc_var(idx_dtype)
            local_s_len = k_end - k_start

            # 计算该子序列需要遍历的 s_block 数量
            num_s_blocks_local = T.alloc_var(idx_dtype)
            num_s_blocks_local = (local_s_len + BLOCK_S - 1) // BLOCK_S

            # Causal: 限制最大可见 chunk (local index)
            local_loop_limit = T.alloc_var(idx_dtype)
            local_loop_limit = num_s_blocks_local
            if is_causal:
                local_block_q_max = local_q // block_size
                local_loop_limit = T.min(local_loop_limit, (local_block_q_max + BLOCK_S) // BLOCK_S)

            # ============================================================
            # 遍历该子序列的 lmk blocks
            # 注意：这里所有 thread 必须一起参与 GEMM，
            # 但不同 thread 可能属于不同子序列。
            # 因此我们用全局 s_len 范围做 GEMM（加载 K tile），
            # 然后在 thread 级别用 local 坐标做 mask 判断。
            # ============================================================

            # 计算整个 BLOCK_L 中涉及的最大 s_block 数量
            # 保守估计：取全局 s_len 的 block 数
            num_s_blocks_global = T.alloc_var(idx_dtype)
            num_s_blocks_global = (s_len + BLOCK_S - 1) // BLOCK_S

            for s_block in T.serial(num_s_blocks_global):
                # 每个 thread 需要判断这个 s_block 是否在自己子序列的范围内
                # 但 GEMM 需要所有 thread 协作加载 K tile
                # 策略：加载 K tile 时用全局坐标，GEMM 后在 thread 级别做 mask

                # 为了让 GEMM 有意义，我们需要加载一个合理的 K tile
                # 这里加载全局位置 s_block * BLOCK_S 处的 K
                base_s_global = s_block * BLOCK_S

                for s_idx, d in T.Parallel(BLOCK_S, BLOCK_D):
                    ts_global = base_s_global + s_idx
                    if ts_global < s_len:
                        K_shared[s_idx, d] = K[0, ts_global, h, d]
                    else:
                        K_shared[s_idx, d] = T.Cast(dtype, 0.0)
                T.sync_threads()

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_s, score_shared)
                T.sync_threads()

                # 每个 thread 独立判断 score 是否有效
                if (my_l < BLOCK_L) and (tq_global < seq_len) and (seq_id < num_seqs):
                    # Causal mask 的 limit_chunk (local index)
                    limit_chunk = T.alloc_var(idx_dtype)
                    limit_chunk = (local_q - window_size + 1) // block_size

                    min_chunk = T.alloc_var(idx_dtype)
                    min_chunk = -1
                    if memory_window_size > 0:
                        min_chunk = (local_q - memory_window_size) // block_size

                    is_valid = T.alloc_var("bool")
                    for s_idx in T.serial(BLOCK_S):
                        ts_global = base_s_global + s_idx
                        # 将全局 lmk index 转换为该子序列内的 local lmk index
                        local_s = ts_global - k_start

                        if (ts_global >= k_start) and (ts_global < k_end):
                            # 在当前子序列的 lmk 范围内
                            is_valid = (local_s >= min_chunk) and ((not is_causal) or (local_s < limit_chunk))
                            if use_drop_mask:
                                is_valid = is_valid and (DropMask[0, tq_global, ts_global] == 0)
                            if is_valid:
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
                                                # 输出 global packed landmark index
                                                topk_indices[kpos] = ts_global
                                                moving = False
                T.sync_threads()

            # 写回结果
            tx2 = T.get_thread_binding()
            tq2 = base_l + tx2
            if (tx2 < BLOCK_L) and (tq2 < seq_len):
                for kk in T.serial(topk):
                    OutScores[0, tq2, h, kk] = topk_scores[kk]
                    OutIndices[0, tq2, h, kk] = topk_indices[kk]

    return fwd_kernel_insert_varlen


# ============================================================================
# Varlen Sort Kernel: 复用 dense 版本的 sort kernel（形状兼容）
# sort_topk_indices_scores_kernel 已经支持 [B, L, h_kv, topk]，
# varlen 场景下 B=1, L=L_total，直接复用即可。
# ============================================================================


class OnlineTopKVarlenFn(torch.autograd.Function):
    """
    Varlen TopK 的 autograd Function。
    
    与 dense 版本的区别：
    1. 输入多了 cu_seq_lens_q 和 cu_seq_lens_k
    2. B 固定为 1
    3. 输出 indices 是 global packed landmark index
    4. 反向传播暂时用 PyTorch 实现（后续可优化为 kernel）
    """
    @staticmethod
    def forward(ctx, q, lmks, topk, fwd_kernel, sort_kernel,
                cu_seq_lens_q, cu_seq_lens_k, drop_mask=None):
        B, L, h_q, D = q.shape
        B2, S, h_kv, D2 = lmks.shape
        assert B == 1 and B2 == 1, "Varlen TopK 要求 B=1"

        # 处理 D 维度不等的情况
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

        # 自动判断 group 方向
        if h_q >= h_kv:
            G = h_q // h_kv
            h_shared = h_kv
            q_group_sum = q.view(B, L, h_kv, G, D).sum(dim=3)
            q_in = q_group_sum.to(q.dtype).contiguous()
            k_in = lmks.contiguous()
        else:
            G = h_kv // h_q
            h_shared = h_q
            lmk_group_sum = lmks.view(B, S, h_q, G, D).sum(dim=3)
            q_in = q.contiguous()
            k_in = lmk_group_sum.to(q.dtype).contiguous()

        # 调用 varlen fwd kernel
        if drop_mask is None:
            drop_mask = torch.zeros(1, 1, 1, dtype=torch.int32, device=q.device)
        best_scores_buf, best_indices_buf = fwd_kernel(
            q_in, k_in,
            cu_seq_lens_q.contiguous(),
            cu_seq_lens_k.contiguous(),
            drop_mask,
        )

        # 排序
        indices_sorted, scores_sorted = sort_kernel(best_indices_buf, best_scores_buf)

        ctx.save_for_backward(q_in, k_in, indices_sorted, cu_seq_lens_q, cu_seq_lens_k)
        ctx.h_kv = h_kv
        ctx.h_q = h_q
        ctx.G = G
        ctx.sum_q = (h_q >= h_kv)
        ctx.h_shared = h_shared
        ctx.topk = topk
        ctx.shapes = (B, L, S, D)
        ctx.d_reshape = d_reshape
        ctx.orig_h_kv = orig_h_kv
        ctx.orig_D2 = orig_D2

        return scores_sorted.to(q.dtype), indices_sorted

    @staticmethod
    def backward(ctx, grad_scores_selected, grad_indices_unused):
        q_saved, k_saved, indices, cu_seq_lens_q, cu_seq_lens_k = ctx.saved_tensors
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
        num_seqs = cu_seq_lens_q.shape[0] - 1

        # 逐子序列计算梯度（因为不同子序列的 lmk 范围不同）
        # indices 是 global index，需要转为 local index 来做 scatter
        grad_q_list = []
        grad_k_list = []

        for seq_idx in range(num_seqs):
            q_start = cu_seq_lens_q[seq_idx].item()
            q_end = cu_seq_lens_q[seq_idx + 1].item()
            k_start = cu_seq_lens_k[seq_idx].item()
            k_end = cu_seq_lens_k[seq_idx + 1].item()
            L_i = q_end - q_start
            S_i = k_end - k_start

            if L_i == 0 or S_i == 0:
                if L_i > 0:
                    grad_q_list.append(torch.zeros(L_i, h_shared, D, device=q_saved.device, dtype=grad_scores_selected.dtype))
                if S_i > 0:
                    grad_k_list.append(torch.zeros(S_i, h_shared, D, device=k_saved.device, dtype=grad_scores_selected.dtype))
                continue

            # 提取子序列的 q, k, indices, grad
            q_i = q_saved[0, q_start:q_end]  # [L_i, h_shared, D]
            k_i = k_saved[0, k_start:k_end]  # [S_i, h_shared, D]
            idx_i = indices[0, q_start:q_end]  # [L_i, h_shared, topk]
            grad_i = grad_scores_selected[0, q_start:q_end]  # [L_i, h_shared, topk]

            # 将 global index 转为 local index
            local_idx = idx_i.clone().to(torch.int64)
            valid_mask = (local_idx >= k_start) & (local_idx < k_end)
            local_idx[valid_mask] -= k_start
            local_idx[~valid_mask] = 0

            safe_grad = grad_i.clone()
            safe_grad[~valid_mask] = 0

            # scatter 到 dense grad
            grad_scores_dense = torch.zeros(
                L_i, h_shared, S_i,
                dtype=grad_scores_selected.dtype,
                device=grad_scores_selected.device,
            )
            grad_scores_dense.scatter_(2, local_idx, safe_grad)

            # grad_q = grad_scores_dense @ k_i
            # grad_k = grad_scores_dense^T @ q_i
            # [L_i, h_shared, S_i] x [S_i, h_shared, D] -> [L_i, h_shared, D]
            bs_h = h_shared
            dense_flat = grad_scores_dense.permute(1, 0, 2).reshape(bs_h, L_i, S_i)
            q_flat = q_i.permute(1, 0, 2).reshape(bs_h, L_i, D)
            k_flat = k_i.permute(1, 0, 2).reshape(bs_h, S_i, D)

            grad_q_flat = torch.bmm(dense_flat, k_flat)  # [bs_h, L_i, D]
            grad_k_flat = torch.bmm(dense_flat.transpose(1, 2), q_flat)  # [bs_h, S_i, D]

            grad_q_list.append(grad_q_flat.reshape(h_shared, L_i, D).permute(1, 0, 2))  # [L_i, h_shared, D]
            grad_k_list.append(grad_k_flat.reshape(h_shared, S_i, D).permute(1, 0, 2))  # [S_i, h_shared, D]

        # 拼接回 packed 形状
        grad_q_packed = torch.cat(grad_q_list, dim=0).unsqueeze(0)  # [1, L, h_shared, D]
        grad_k_packed = torch.cat(grad_k_list, dim=0).unsqueeze(0)  # [1, S, h_shared, D]

        grad_q_packed = grad_q_packed * sm_scale
        grad_k_packed = grad_k_packed * sm_scale

        # 如果 forward 中对 q 做了 group sum，需要 expand 梯度
        if sum_q:
            # grad_q_packed: [1, L, h_kv, D] -> expand to [1, L, h_q, D]
            grad_q_expanded = grad_q_packed.unsqueeze(3).expand(1, L, h_kv, G, D)
            grad_q_packed = grad_q_expanded.reshape(1, L, h_kv * G, D)
            grad_lmks = grad_k_packed
        else:
            grad_q_packed = grad_q_packed
            grad_lmk_expanded = grad_k_packed.unsqueeze(3).expand(1, S, h_q, G, D)
            grad_lmks = grad_lmk_expanded.reshape(1, S, h_q * G, D)

        if d_reshape:
            grad_lmks = grad_lmks.reshape(1, S, orig_h_kv, orig_D2)

        return grad_q_packed, grad_lmks, None, None, None, None, None, None


class OnlineTopK_Varlen(torch.nn.Module):
    """
    Varlen TopK Module（SFT packing 场景）。
    
    接口语义：
    - q: [1, L_total, h_q, D]  packed 后所有 query
    - lmks: [1, S_total, h_kv, D]  packed 后所有 landmark keys
    - cu_seq_lens_q: [num_seqs + 1]  query 子序列边界
    - cu_seq_lens_k: [num_seqs + 1]  lmk key 子序列边界
    
    输出：
    - indices: [1, L_total, h_shared, topk]  global packed landmark index
    - scores:  [1, L_total, h_shared, topk]
    """
    def __init__(self, topk, block_size, window_size, is_causal,
                 memory_window_size=-1, use_drop_mask=False):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self.memory_window_size = memory_window_size
        self.use_drop_mask = use_drop_mask

    def forward(self, q, lmks, cu_seq_lens_q, cu_seq_lens_k, drop_mask=None):
        B, L, h_q, D = q.shape
        _, S, h_kv, _ = lmks.shape
        assert B == 1, "Varlen TopK 要求 B=1"
        h_shared = min(h_q, h_kv)
        fwd_kernel = fused_topk_forward_kernel_insert_varlen(
            h_shared, D, self.topk,
            self.block_size, self.window_size, self.is_causal,
            memory_window_size=self.memory_window_size,
            use_drop_mask=self.use_drop_mask,
        )

        sort_kernel = sort_topk_indices_scores_kernel(
            1, h_shared, self.topk,
            seq_len=None,
            s_len=S,
            is_training=False,
        )

        scores, indices = OnlineTopKVarlenFn.apply(
            q, lmks, self.topk, fwd_kernel, sort_kernel,
            cu_seq_lens_q, cu_seq_lens_k, drop_mask,
        )
        return indices, scores


def online_topk_group_varlen(
    q: torch.Tensor,
    lmks: torch.Tensor,
    topk: int,
    block_size: int,
    window_size: int,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    is_causal: bool = True,
    memory_window_size: int = -1,
    drop_mask: torch.Tensor = None,
):
    """
    Varlen TopK 的 Functional API（SFT packing 场景）。

    Args:
        q: [1, L_total, h_q, D]  packed 后所有 query（或 query landmark）
        lmks: [1, S_total, h_kv, D]  packed 后所有 landmark keys
        topk: int
        block_size: int (= chunk_size)
        window_size: int (= hsa_sliding_window)
        cu_seq_lens_q: [num_seqs + 1], int32  query 子序列边界
        cu_seq_lens_k: [num_seqs + 1], int32  lmk key 子序列边界
        is_causal: bool
        memory_window_size: int, optional
        drop_mask: [1, L_total, S_total] int32 tensor, optional. 0/1 bitmap, 1 means drop.

    Returns:
        indices: [1, L_total, min(h_q,h_kv), topk]  global packed landmark index
        scores:  [1, L_total, min(h_q,h_kv), topk]

    语义约定：
        - 内部 causal/window/memory_window 判断使用子序列内的 local index
        - 输出 indices 是 global packed landmark index（直接可用于 varlen HSA 下游）
        - 无效位置 indices = -1, scores = -inf
    """
    assert q.shape[0] == 1, "Varlen TopK 要求 B=1"
    use_drop_mask = drop_mask is not None
    module = OnlineTopK_Varlen(
        topk, block_size, window_size, is_causal,
        memory_window_size=memory_window_size,
        use_drop_mask=use_drop_mask,
    )
    return module(q, lmks, cu_seq_lens_q, cu_seq_lens_k, drop_mask=drop_mask)


# ============================================================================
# 统一的 TopK 前向 Kernel（合并 dense + varlen）
# ============================================================================
#
# 以 dense (pretrain) kernel 为主干，varlen (SFT packing) 通过残差注入方式融合：
#   在 5 个注入点用 `if is_varlen:` 微调变量值，不改变主干代码结构。
#   类似于 `if use_drop_mask:` 对 is_valid 的处理方式。
#
# is_varlen 是编译时常量，所有分支在 JIT 时被消除，不引入运行时开销。
#
# 参数布局（7 个 tensor）：
#   Q, K, OutScores, OutIndices,
#   第5个=Q_Offset_or_CuQ:  dense → Q_Offset [1],       varlen → CuSeqLensQ [N+1]
#   第6个=CuSeqLensK:       dense → 占位 [1],            varlen → CuSeqLensK [N+1]
#   第7个=DropMask:          dense/varlen 均可使用 [B, L, S]（不使用时占位 [1,1,1]）
# ============================================================================

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
    memory_window_size=-1,
    BLOCK_L=None, BLOCK_S=None, threads=None,
    is_training=True,
    use_drop_mask=False,
    # ---- varlen 新增参数 ----
    is_varlen=False,
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
    # ---- 注入点1: seq_len/s_len dynamic 处理 ----
    if is_varlen:
        # varlen 场景: B=1, seq_len/s_len 始终 dynamic（打包后总长度运行时才知道）
        seq_len = T.dynamic("seq_len")
        s_len = T.dynamic("s_len")
        # num_seqs 也设为 dynamic，避免子序列个数变化时重编译
        num_seqs = T.dynamic("num_seqs")
    else:
        if not is_training:
            seq_len = T.dynamic("seq_len")
            s_len = T.dynamic("s_len")

    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    # Tensor shape: varlen 时 batch 维固定为 1
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
        cu_q_shape = [1]  # dense 不使用 cu_seq_lens，占位
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

    @T.prim_func
    def fwd_kernel_insert_unified(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
        # dense: Q_Offset [1] / varlen: CuSeqLensQ [num_seqs+1]
        Q_Offset_or_CuQ: T.Tensor(cu_q_shape if is_varlen else [1], "int32"),
        # varlen: CuSeqLensK [num_seqs+1] / dense: 占位 [1]
        CuSeqLensK: T.Tensor(cu_k_shape, idx_dtype),
        # DropMask [B, L, S]（不使用时占位 [1,1,1]），dense/varlen 均可使用
        DropMask: T.Tensor(drop_mask_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), heads, grid_batch, threads=threads) as (bx, by, bz):
            q_offset = T.if_then_else(is_training, 0, Q_Offset_or_CuQ[0])
            b = bz  # varlen 时 b 始终为 0
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

            # 加载 Q tile（主干代码，dense/varlen 共用，b=0 时自然兼容）
            for l_idx, d in T.Parallel(BLOCK_L, BLOCK_D):
                tq = base_l + l_idx
                if tq < seq_len:
                    Q_shared[l_idx, d] = Q[b, tq, h, d]
                else:
                    Q_shared[l_idx, d] = T.Cast(dtype, 0.0)

            # ---- 注入点2: varlen 子序列查找（在 s_block 循环前） ----
            # 以下变量在 dense 模式下不会被使用，编译时消除
            if is_varlen:
                tx_pre = T.get_thread_binding()
                tq_packed = base_l + tx_pre  # global packed query index

                # 线性扫描 cu_seq_lens_q 确定所属子序列 seq_id
                seq_id = T.alloc_var(idx_dtype)
                seq_id = 0
                for si in T.serial(num_seqs):
                    if tq_packed >= Q_Offset_or_CuQ[si + 1]:
                        seq_id = si + 1

                # 子序列的 q 范围
                q_start = T.alloc_var(idx_dtype)
                q_start = Q_Offset_or_CuQ[seq_id]
                local_q = T.alloc_var(idx_dtype)
                local_q = tq_packed - q_start

                # 子序列的 lmk 范围: [k_start, k_end)
                k_start = T.alloc_var(idx_dtype)
                k_end = T.alloc_var(idx_dtype)
                k_start = CuSeqLensK[seq_id]
                k_end = CuSeqLensK[seq_id + 1]

            # ---- 注入点3: loop_limit 计算 ----
            # dense: 编译时 num_s_blocks，causal 时可缩小
            # varlen: 必须遍历全局 s_len（不同 thread 属于不同子序列，无法统一缩小）
            loop_limit = T.alloc_var("int32")
            loop_limit = num_s_blocks
            if is_causal and (not is_varlen):
                tq_max = T.min(seq_len - 1, base_l + (BLOCK_L - 1))
                tq_max_global = q_offset + tq_max
                block_q_max = tq_max_global // block_size
                loop_limit = T.min(loop_limit, tilelang.cdiv(block_q_max, BLOCK_S))

            # ============================================================
            # 主干 s_block 循环（dense/varlen 共用同一份代码）
            # ============================================================
            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S

                # 加载 K tile（dense 用 K[b,...], varlen 时 b=0 自然兼容）
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

                # varlen 额外条件: seq_id < num_seqs（防止越界 padding token）
                varlen_thread_ok = T.alloc_var("bool")
                varlen_thread_ok = True
                if is_varlen:
                    varlen_thread_ok = (seq_id < num_seqs)

                if (my_l < BLOCK_L) and (tq < seq_len) and varlen_thread_ok:
                    # ---- 注入点4: 坐标转换 ----
                    # effective_q: 用于 causal/window 判断的 query 坐标
                    #   dense:  q_offset + tq (全局 token 位置)
                    #   varlen: local_q (子序列内 local index)
                    tq_global = q_offset + tq
                    effective_q = T.alloc_var(idx_dtype)
                    effective_q = tq_global
                    if is_varlen:
                        effective_q = local_q

                    limit_chunk = (effective_q - window_size + 1) // block_size

                    min_chunk = -1
                    if memory_window_size > 0:
                        min_chunk = (effective_q - memory_window_size) // block_size

                    is_valid = T.alloc_var("bool")
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx

                        # effective_s: 用于 causal/window 判断的 key 坐标
                        #   dense:  ts 本身就是 local index
                        #   varlen: ts - k_start (转为子序列内 local index)
                        # in_range: key 是否在有效范围内
                        #   dense:  ts < s_len（已由外层保证）
                        #   varlen: ts 必须在 [k_start, k_end) 内
                        effective_s = T.alloc_var(idx_dtype)
                        effective_s = ts
                        in_range = T.alloc_var("bool")
                        in_range = (ts < s_len)
                        if is_varlen:
                            effective_s = ts - k_start
                            in_range = (ts >= k_start) and (ts < k_end)

                        if in_range:
                            is_valid = (effective_s >= min_chunk) and ((not is_causal) or (effective_s < limit_chunk))
                            if use_drop_mask:
                                is_valid = is_valid and (DropMask[b, tq, ts] == 0)
                            if is_valid:
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
                                                # ---- 注入点5: 存储的 index ----
                                                # dense: ts (batch 内 lmk index)
                                                # varlen: ts (global packed lmk index，恰好也是 ts)
                                                topk_indices[kpos] = ts
                                                moving = False
                T.sync_threads()

            # 写回结果（dense/varlen 共用，b=0 时自然兼容）
            tx2 = T.get_thread_binding()
            tq2 = base_l + tx2
            if (tx2 < BLOCK_L) and (tq2 < seq_len):
                for kk in T.serial(topk):
                    OutScores[b, tq2, h, kk] = topk_scores[kk]
                    OutIndices[b, tq2, h, kk] = topk_indices[kk]

    return fwd_kernel_insert_unified


# ============================================================================
# Unified TopK autograd Function / Module / Functional API
# ============================================================================
# 统一封装 dense (pretrain) 和 varlen (SFT packing) 两条路径。
# backward 复用现有 OnlineTopKFusedFn.backward (dense) 和
# OnlineTopKVarlenFn.backward (varlen) 的逻辑。
# ============================================================================

class OnlineTopKUnifiedFn(torch.autograd.Function):
    """
    统一的 TopK autograd Function，同时支持 dense 和 varlen 路径。
    通过 is_varlen 标志在 forward/backward 中分支。
    """

    @staticmethod
    def forward(ctx, q, lmks, topk: int, fwd_kernel, sort_kernel,
                is_varlen: bool,
                # dense 专用
                q_offset_tensor=None, drop_mask=None,
                # varlen 专用
                cu_seq_lens_q=None, cu_seq_lens_k=None):
        B, L, h_q, D = q.shape
        B2, S, h_kv, D2 = lmks.shape
        dtype = q.dtype
        assert B == B2

        if is_varlen:
            assert B == 1, "Varlen TopK 要求 B=1"

        # 处理 D 维度不等
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

        # group sum
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

        # 调用 unified fwd kernel
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

        # sort
        indices_sorted, scores_sorted = sort_kernel(best_indices_buf, best_scores_buf)

        # 保存 backward 所需张量（统一路径，不区分 dense/varlen）
        # varlen 时 B=1, L=L_total, S=S_total，indices 是 global index，
        # 可以直接 scatter 到 [B, L, h_shared, S] 的全局 dense 矩阵，
        # 跨序列位置因 indices 不指向其他序列的 k 而自然为 0，数学上等价。
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

        # 1. scatter: 将 sparse [B, L, h_shared, topk] 展开为 dense [B, L, h_shared, S]
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

        # 2. bmm: grad_q = dense @ k, grad_k = dense^T @ q
        dense_in = grad_scores_dense.permute(0, 2, 1, 3).reshape(bs_h, L, S)
        del grad_scores_dense

        q_in = q_saved.permute(0, 2, 1, 3).reshape(bs_h, L, D)
        k_in = k_saved.permute(0, 2, 1, 3).reshape(bs_h, S, D)

        grad_q_flat = torch.bmm(dense_in, k_in)
        grad_k_flat = torch.bmm(dense_in.transpose(1, 2), q_in)
        del dense_in, q_in, k_in

        # 3. group expand + scale
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

        # 返回: q, lmks, topk, fwd_kernel, sort_kernel, is_varlen,
        #        q_offset_tensor, drop_mask, cu_seq_lens_q, cu_seq_lens_k
        return grad_q, grad_lmks, None, None, None, None, None, None, None, None


class OnlineTopK_Unified(torch.nn.Module):
    """
    统一的 TopK Module，同时支持 dense (pretrain) 和 varlen (SFT packing) 场景。

    dense 模式: forward(q, lmks, q_offset, drop_mask=None)
    varlen 模式: forward(q, lmks, cu_seq_lens_q, cu_seq_lens_k, drop_mask=None)
    """

    def __init__(self, topk, block_size, window_size, is_causal,
                 memory_window_size=-1, is_training=True,
                 use_drop_mask=False, is_varlen=False):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self.memory_window_size = memory_window_size
        self.is_training = is_training
        self.use_drop_mask = use_drop_mask
        self.is_varlen = is_varlen

    def forward(self, q, lmks, *args, **kwargs):
        B, L, h_q, D = q.shape
        _, S, h_kv, _ = lmks.shape
        h_shared = min(h_q, h_kv)

        # 编译 unified fwd kernel
        fwd_kernel = fused_topk_forward_kernel_insert_unified(
            1 if self.is_varlen else B,
            h_shared, D, self.topk,
            self.block_size, self.window_size, self.is_causal,
            seq_len=None if (self.is_varlen or not self.is_training) else L,
            s_len=None if (self.is_varlen or not self.is_training) else S,
            memory_window_size=self.memory_window_size,
            is_training=self.is_training,
            use_drop_mask=self.use_drop_mask,
            is_varlen=self.is_varlen,
        )

        # 编译 sort kernel
        sort_kernel = sort_topk_indices_scores_kernel(
            1 if self.is_varlen else B,
            h_shared, self.topk,
            seq_len=None if (self.is_varlen or not self.is_training) else L,
            s_len=S if self.is_varlen else (S if self.is_training else None),
            is_training=self.is_training if not self.is_varlen else False,
        )

        if self.is_varlen:
            # varlen 模式: args = (cu_seq_lens_q, cu_seq_lens_k[, drop_mask])
            cu_seq_lens_q = args[0] if len(args) > 0 else kwargs['cu_seq_lens_q']
            cu_seq_lens_k = args[1] if len(args) > 1 else kwargs['cu_seq_lens_k']

            scores, indices = OnlineTopKUnifiedFn.apply(
                q, lmks, self.topk, fwd_kernel, sort_kernel,
                True,  # is_varlen
                None, None,  # q_offset_tensor, drop_mask (dense 专用)
                cu_seq_lens_q, cu_seq_lens_k,
            )
        else:
            # dense 模式: args = (q_offset[, drop_mask])
            q_offset = args[0] if len(args) > 0 else kwargs.get('q_offset', 0)
            drop_mask = args[1] if len(args) > 1 else kwargs.get('drop_mask', None)

            q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=q.device)

            scores, indices = OnlineTopKUnifiedFn.apply(
                q, lmks, self.topk, fwd_kernel, sort_kernel,
                False,  # is_varlen
                q_offset_tensor, drop_mask,
                None, None,  # cu_seq_lens_q, cu_seq_lens_k (varlen 专用)
            )

        return indices, scores


def online_topk_group_unified(
    q: torch.Tensor,
    lmks: torch.Tensor,
    topk: int,
    block_size: int,
    window_size: int,
    is_causal: bool = True,
    memory_window_size: int = -1,
    q_offset: int = 0,
    is_training: bool = True,
    drop_mask: torch.Tensor = None,
    # varlen 参数（传入则走 varlen 路径）
    cu_seq_lens_q: torch.Tensor = None,
    cu_seq_lens_k: torch.Tensor = None,
):
    """
    统一的 TopK Functional API，同时支持 dense 和 varlen 两种路径。

    根据是否传入 cu_seq_lens_q / cu_seq_lens_k 自动判断模式：
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
        memory_window_size (int, default=-1):
            Memory window 大小。> 0 时，query 只能看到
            chunk_idx >= (q_global_pos - memory_window_size) // block_size 的 chunks，
            即限制最远可回溯的距离。-1 表示不限制。
        q_offset (int, default=0):
            Dense 模式专用。query 在 KV cache 中的全局偏移量，用于 inference 时
            正确计算 causal 边界。训练时通常为 0。varlen 模式下忽略此参数。
        is_training (bool, default=True):
            是否为训练模式。False 时 seq_len/s_len 使用 dynamic shape 以避免
            不同序列长度导致的 kernel 重编译。
        drop_mask (torch.Tensor, optional):
            Dense 模式专用。shape = [B, L, S]，dtype = int32，0/1 bitmap。
            1 表示该 chunk 被 drop，不参与 topk 选择。varlen 模式下忽略此参数。
        cu_seq_lens_q (torch.Tensor, optional):
            Varlen 模式专用。Cumulative sequence lengths for queries。
            shape = [num_seqs + 1]，dtype = int32。
            例如 3 条子序列长度分别为 [100, 200, 150]，则 cu_seq_lens_q = [0, 100, 300, 450]。
        cu_seq_lens_k (torch.Tensor, optional):
            Varlen 模式专用。Cumulative sequence lengths for landmark keys。
            shape = [num_seqs + 1]，dtype = int32。
            例如 3 条子序列的 landmark 数分别为 [2, 4, 3]，则 cu_seq_lens_k = [0, 2, 6, 9]。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (indices, scores)
        - indices (torch.Tensor):
            选出的 chunk indices，按升序排列。
            - dense 模式: shape = [B, L, h_shared, topk]，值域 [0, S)
            - varlen 模式: shape = [1, L_total, h_shared, topk]，值域为 **global packed index**，
              即 [cu_seq_lens_k[i], cu_seq_lens_k[i+1]) 范围内的值。
              可直接用于在 packed lmks tensor 上索引。
            - 无效位（causal mask 导致可见 chunks < topk）填充为 -1。
            - h_shared = min(h_q, h_kv)
        - scores (torch.Tensor):
            对应的 attention scores（经过 softmax scale 但未经 softmax）。
            shape 同 indices。无效位填充为 -inf。

    使用示例:
        # === Dense 模式 (pretrain) ===
        indices, scores = online_topk_group_unified(
            q,              # [B, L, h_q, D]
            lmks,           # [B, S, h_kv, D]
            topk=4,
            block_size=64,
            window_size=512,
            is_causal=True,
        )

        # === Dense 模式 (inference, 带 q_offset) ===
        indices, scores = online_topk_group_unified(
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
        indices, scores = online_topk_group_unified(
            q_packed,       # [1, L_total, h_q, D]
            lmks_packed,    # [1, S_total, h_kv, D]
            topk=4,
            block_size=64,
            window_size=512,
            is_causal=True,
            cu_seq_lens_q=cu_q,  # [num_seqs + 1], e.g. [0, 100, 300, 450]
            cu_seq_lens_k=cu_k,  # [num_seqs + 1], e.g. [0, 2, 6, 9]
        )
        # indices 是 global packed index，可直接索引: lmks_packed[:, indices, ...]
        # 若需 local index: local_idx = indices - cu_seq_lens_k[seq_id]

    注意事项:
        1. varlen 模式下 batch 维度必须为 1（packed tensor）。
        2. varlen 模式下 num_seqs、seq_len、s_len 均为 dynamic 变量，
           子序列个数或长度变化不会触发 kernel 重编译。
        3. varlen 模式下输出的 indices 是 global packed index（不是子序列内的 local index）。
        4. 此接口支持 autograd 反向传播（梯度流回 q 和 lmks）。
        5. h_q 和 h_kv 可以不同（GQA/MQA），内部会自动做 group sum。
    """
    is_varlen = (cu_seq_lens_q is not None) and (cu_seq_lens_k is not None)
    use_drop_mask = drop_mask is not None
    module = OnlineTopK_Unified(
        topk, block_size, window_size, is_causal,
        memory_window_size=memory_window_size,
        is_training=is_training,
        use_drop_mask=use_drop_mask,
        is_varlen=is_varlen,
    )

    if is_varlen:
        return module(q, lmks, cu_seq_lens_q, cu_seq_lens_k)
    else:
        return module(q, lmks, q_offset, drop_mask)


import torch.nn.functional as F
def ref_topk_forward_with_grad(q, lmks, topk, block_size, window_size, is_causal, dtype, memory_window_size=-1, q_offset: int = 0, drop_mask: Optional[torch.Tensor] = None):
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
    
    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1)
        i_idx_global = i_idx + q_offset
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)

        limit_chunk_idx = (i_idx_global - window_size + 1).div(block_size, rounding_mode='floor')
        mask = j_idx >= limit_chunk_idx
        if memory_window_size > 0:
            min_chunk_idx = (i_idx_global - memory_window_size).div(block_size, rounding_mode='floor')
            mask = mask | (j_idx < min_chunk_idx)
        scores_ref = scores_ref.masked_fill(mask.unsqueeze(0).unsqueeze(2), float('-inf'))
    
    if drop_mask is not None:
        # drop_mask: (B, L, S), 1表示drop
        drop_bool = drop_mask.bool()  # (B, L, S)
        scores_ref = scores_ref.masked_fill(drop_bool.unsqueeze(2), float('-inf'))

    scores_topk, indices_topk = torch.topk(scores_ref, k=topk, dim=-1, sorted=False)
    
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
    memory_window_size = 512
    q_offset = 128
    
    dtype = torch.bfloat16
    device = "cuda"
    
    print(f"Config: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}, window_size={window_size}, memory_window_size={memory_window_size}, q_offset={q_offset}")
    
    torch.manual_seed(42)
    q = torch.randn(B, L, h_q, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)
    
    # ============ Forward Correctness ============
    print("\n--- Forward Correctness ---")
    
    # 修改 1: 传入 q_offset
    indices_fused, scores_fused = online_topk_group_unified(q, lmks, topk, block_size, window_size, is_causal, memory_window_size=memory_window_size, q_offset=q_offset, is_training=False)
    indices_ref, scores_ref = ref_topk_forward_with_grad(q, lmks, topk, block_size, window_size, is_causal, dtype=torch.float32, memory_window_size=memory_window_size, q_offset=q_offset)
    
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
        _, scores_ref_check = ref_topk_forward_with_grad(q_ref, lmks_ref, topk, block_size, window_size, is_causal, dtype=torch.float32, memory_window_size=memory_window_size, q_offset=q_offset)
        invalid_mask = (scores_ref_check < -1e9)
        grad_output[invalid_mask] = 0.0

    # 修改 3: Fused 后向验证传入 q_offset
    indices_fused_bwd, scores_fused_bwd = online_topk_group_unified(q_fused, lmks_fused, topk, block_size, window_size, is_causal, memory_window_size=memory_window_size, q_offset=q_offset, is_training=False)
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
        
        # 同时补全 memory_window_size 的逻辑
        if memory_window_size > 0:
            min_chunk_idx = (i_idx_global - memory_window_size).div(block_size, rounding_mode='floor')
            causal_mask = causal_mask | (j_idx < min_chunk_idx)

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
    memory_window_size = -1
    
    dtype = torch.bfloat16
    device = "cuda"
    
    print(f"Config: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}, window_size={window_size}, memory_window_size={memory_window_size}")
    
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
            _, scores = online_topk_group_unified(q_t, lmks_t, topk, block_size, window_size, is_causal, memory_window_size=memory_window_size)
            loss = (scores * grad_output).sum()
            loss.backward()
        torch.cuda.synchronize()
        
        # Memory Check
        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        # 修复点2: 正确获取当前Graph的 scores
        _, scores = online_topk_group_unified(q_t, lmks_t, topk, block_size, window_size, is_causal, memory_window_size=memory_window_size)
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
            _, _ = online_topk_group_unified(q_t, lmks_t, topk, block_size, window_size, is_causal, memory_window_size=memory_window_size)
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters
        
        # Fwd + Bwd
        start_all.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            _, scores = online_topk_group_unified(q_t, lmks_t, topk, block_size, window_size, is_causal, memory_window_size=memory_window_size)
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
            _ = ref_topk_forward_with_grad(q_t, lmks_t, topk, block_size, window_size, is_causal, dtype=torch.bfloat16, memory_window_size=memory_window_size)
        
        def forward_backward():
            _, scores = ref_topk_forward_with_grad(q_t, lmks_t, topk, block_size, window_size, is_causal, dtype=torch.bfloat16, memory_window_size=memory_window_size)
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


import pytest
@pytest.mark.parametrize("B, L, S, h_kv, G, D, topk, block_size, window_size, memory_window_size, q_offset", [
    (2, 4096, 64, 2, 8, 64, 16, 64, 64, -1, 0),       # 原始测试: 偏移 0
    (2, 4096, 64, 2, 8, 64, 16, 64, 64, 512, 128),    # 新增测试: 带偏移和记忆窗口
    (1, 2048, 64, 1, 8, 64, 16, 32, 32, 128, 1024),   # 窄窗口 + 大偏移
    (1, 2500, 64, 1, 8, 64, 8, 32, 100, 2000, 64),    # 长记忆窗口 + 小偏移
])
def test_topk_correctness_robust(B, L, S, h_kv, G, D, topk, block_size, window_size, memory_window_size, q_offset, sum_kv=False, D_kv=None):
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
    print(f"\nTesting Config: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, G={G}, D={D}{d_info}, topk={topk}, BS={block_size}, Win={window_size}, MemWin={memory_window_size}, q_offset={q_offset}, mode={sum_mode}")

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
    indices_fused, scores_fused = online_topk_group_unified(
        q_fused, lmks_fused, topk, block_size, window_size, is_causal, memory_window_size=memory_window_size, q_offset=q_offset, is_training=is_training
    )

    # Reference - 传入 q_offset
    indices_ref, scores_ref = ref_topk_forward_with_grad(
        q_ref, lmks_ref, topk, block_size, window_size, is_causal, dtype=torch.float32, memory_window_size=memory_window_size, q_offset=q_offset
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
        
        # --- Memory Window Mask Logic ---
        if memory_window_size > 0:
             min_chunk_idx = (i_idx_global - memory_window_size).div(block_size, rounding_mode='floor')
             causal_mask = causal_mask | (j_idx < min_chunk_idx)
        # --------------------------------
        
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

    print(f"Test Passed: B={B}, L={L}, S={S}, h_q={h_q}, h_kv={h_kv}, G={G}, D={D}, D_kv={D_kv}, topk={topk}, MemWin={memory_window_size}, q_offset={q_offset}, mode={sum_mode}")



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
    memory_window_size = -1
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
    indices1, scores1 = online_topk_group_unified(q1, lmks1, topk, block_size, window_size, is_causal,
                                          memory_window_size=memory_window_size, q_offset=q_offset, is_training=is_training)
    torch.cuda.synchronize()
    t1 = time.time()
    time_1 = t1 - t0
    print(f"  耗时: {time_1:.4f}s")

    # ========== 第2次调用：SEQ_LEN_2（不同长度） ==========
    print(f"\n[第2次调用] SEQ_LEN={SEQ_LEN_2}, S={S_2}")
    q2, lmks2 = make_inputs(SEQ_LEN_2, S_2)
    torch.cuda.synchronize()
    t0 = time.time()
    indices2, scores2 = online_topk_group_unified(q2, lmks2, topk, block_size, window_size, is_causal,
                                          memory_window_size=memory_window_size, q_offset=q_offset, is_training=is_training)
    torch.cuda.synchronize()
    t1 = time.time()
    time_2 = t1 - t0
    print(f"  耗时: {time_2:.4f}s")

    # ========== 第3次调用：再次 SEQ_LEN_1（测试缓存命中） ==========
    print(f"\n[第3次调用] SEQ_LEN={SEQ_LEN_1}, S={S_1}（与第1次相同，测试缓存命中）")
    q3, lmks3 = make_inputs(SEQ_LEN_1, S_1)
    torch.cuda.synchronize()
    t0 = time.time()
    indices3, scores3 = online_topk_group_unified(q3, lmks3, topk, block_size, window_size, is_causal,
                                          memory_window_size=memory_window_size, q_offset=q_offset, is_training=is_training)
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
    memory_window_size: int = -1,
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
    print(f"memory_window_size={memory_window_size}, q_offset={q_offset}")
    print(f"drop_mask中被mask的chunk占比: {drop_mask.float().mean().item():.4f}")

    # ====== Step 2: 调用 kernel（传入 0/1 bitmap）======
    indices_kernel, scores_kernel = online_topk_group_unified(
        q, lmks, topk, block_size, window_size, is_causal,
        memory_window_size=memory_window_size,
        q_offset=q_offset,
        is_training=True,
        drop_mask=drop_mask,
    )

    # ====== Step 3: 调用 ref（传入相同的 0/1 bitmap）======
    indices_ref, scores_ref = ref_topk_forward_with_grad(
        q, lmks, topk, block_size, window_size, is_causal,
        dtype=torch.float32,
        memory_window_size=memory_window_size,
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
    memory_window_size = -1
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
        _ = online_topk_group_unified(q, lmks, topk, block_size, window_size, is_causal,
                              memory_window_size=memory_window_size, q_offset=q_offset,
                              is_training=is_training, drop_mask=None)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(bench_iters):
        _ = online_topk_group_unified(q, lmks, topk, block_size, window_size, is_causal,
                              memory_window_size=memory_window_size, q_offset=q_offset,
                              is_training=is_training, drop_mask=None)
    end.record()
    torch.cuda.synchronize()
    fwd_no_drop_ms = start.elapsed_time(end) / bench_iters
    print(f"  Avg Fwd Latency: {fwd_no_drop_ms:.4f} ms")

    # ====== Benchmark: 有 drop (含每iter生成mask) ======
    print(f"\n--- 有 DropMask (topk_dropout={topk_dropout}, 含每iter生成mask) ---")
    for _ in range(warmup_iters):
        drop_mask = (torch.rand(B, L, S, device=device) < topk_dropout).to(torch.int32)
        _ = online_topk_group_unified(q, lmks, topk, block_size, window_size, is_causal,
                              memory_window_size=memory_window_size, q_offset=q_offset,
                              is_training=is_training,
                              drop_mask=drop_mask)
    torch.cuda.synchronize()

    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for _ in range(bench_iters):
        drop_mask = (torch.rand(B, L, S, device=device) < topk_dropout).to(torch.int32)
        _ = online_topk_group_unified(q, lmks, topk, block_size, window_size, is_causal,
                              memory_window_size=memory_window_size, q_offset=q_offset,
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
    memory_window_size = -1
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
    indices_drop, scores_drop = online_topk_group_unified(
        q, lmks, topk, block_size, window_size, is_causal,
        memory_window_size=memory_window_size, q_offset=q_offset,
        is_training=is_training,
        drop_mask=drop_mask,
    )

    # --- 调用topk (无mask) 作为对照 ---
    indices_nodrop, scores_nodrop = online_topk_group_unified(
        q, lmks, topk, block_size, window_size, is_causal,
        memory_window_size=memory_window_size, q_offset=q_offset,
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



if __name__ == "__main__":
    test_online_topk_fused_correctness()
    test_online_topk_fused_memory_and_speed()
    params_list = [
      # (B,  L,   S, h_kv, G, D, topk, block_size, window_size, memory_window_size, q_offset)
        (2, 4096, 64, 2, 8, 64, 16, 64, 64, -1, 0),         # (4096+0)/64 = 64
        (2, 4096, 66, 2, 8, 64, 16, 64, 64, 512, 128),      # (4096+128)/64 = 66
        (1, 2048, 96, 1, 8, 64, 16, 32, 32, 128, 1024),     # (2048+1024)/32 = 96
        (1, 2500, 81, 1, 1, 128, 8, 32, 100, 2000, 64),      # (2500+64)/32 ≈ 80.125 -> 81
    ]
    for p in params_list:
        test_topk_correctness_robust(*p)

    test_recompile_different_seq_len()
    test_kernel_vs_ref_with_drop_mask()
    test_drop_mask_forward_speed()
    test_visualize_dropout_mask_effect()
    test_drop_mask_topk_exclusion()

    # === sum_kv 测试: h_kv > h_q, 对lmk做sum ===
    # 参数说明: (B, L, S, h_kv, G, D, topk, block_size, window_size, memory_window_size, q_offset, sum_kv)
    # 注意sum_kv=True时: h_kv是lmk的头数, G是倍数, h_q = h_kv // G
    sum_kv_params = [
      # h_kv=8, G=4, h_q=2
        (2, 4096, 64, 8, 4, 64, 16, 64, 64, -1, 0, True),
      # h_kv=16, G=8, h_q=2
        (2, 4096, 66, 16, 8, 64, 16, 64, 64, 512, 128, True),
      # h_kv=8, G=8, h_q=1
        (1, 2048, 96, 8, 8, 64, 16, 32, 32, 128, 1024, True),
    ]
    for p in sum_kv_params:
        test_topk_correctness_robust(*p)

    # === D维度不等测试: h_q == h_kv 但 D_kv != D_q, 在topk内部自动reshape ===
    # 参数说明: (B, L, S, h_kv, G, D_q, topk, block_size, window_size, memory_window_size, q_offset, sum_kv, D_kv)
    # 例如 unified_retrieval 场景: q=(B,L,1,1024), lmks=(B,S,1,4096), D_kv/D_q=4
    d_reshape_params = [
      # h_q=1, h_kv=1, D_q=64, D_kv=256 (d_ratio=4)
        dict(B=2, L=4096, S=64, h_kv=1, G=1, D=64, topk=16, block_size=64, window_size=64, memory_window_size=-1, q_offset=0, sum_kv=False, D_kv=256),
      # h_q=1, h_kv=1, D_q=128, D_kv=512 (d_ratio=4)
        dict(B=1, L=2048, S=96, h_kv=1, G=1, D=128, topk=16, block_size=32, window_size=32, memory_window_size=128, q_offset=1024, sum_kv=False, D_kv=512),
      # h_q=2, h_kv=2, D_q=64, D_kv=256 (d_ratio=4), 多头+D不等
        dict(B=2, L=4096, S=64, h_kv=2, G=1, D=64, topk=16, block_size=64, window_size=64, memory_window_size=-1, q_offset=0, sum_kv=False, D_kv=256),
      # h_q=1, h_kv=1, D_q=128, D_kv=1024 (d_ratio=8)
        dict(B=1, L=2048, S=96, h_kv=1, G=1, D=128, topk=16, block_size=32, window_size=32, memory_window_size=-1, q_offset=0, sum_kv=False, D_kv=1024),
    ]
    for p in d_reshape_params:
        test_topk_correctness_robust(**p)


# python ops/topk_group_varlen.py
