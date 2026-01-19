import torch
import math
import logging
logging.getLogger("tilelang.jit.kernel").setLevel(logging.WARNING)
logging.getLogger("tilelang").setLevel(logging.WARNING)
import tilelang
from tilelang import language as T


from einops import rearrange
def hsa_torch_ref(q, k, v, weights, indices, *, chunk_size: int, sm_scale: float, block_q: int, mask_last_token: bool = False):
    """
    参考实现（与 test_group_qa 一致的数学公式）:
    - 对于每个 query block 和被选中的 K 块:
      p = softmax_s(q @ k^T * sm_scale)  # 在块内 S 维做 softmax
      o_k = p @ v
    - 最终输出: o = sum_k (weights[:, :, :, k] * o_k)

    形状约定:
    - q: (B, L, HQ, D)
    - k, v: (B, L, H, D)
    - weights: (B, q_blocks, HQ, K)
    - indices: (B, q_blocks, H, K) 或 (B, L, H, K) 且 L == q_blocks * block_q
    - 返回: o_ref: (B, L, HQ, D) float32
    """
    B, L, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    q_blocks = L // block_q
    device = q.device

    if indices.shape[1] != q_blocks:
        idx_view = indices.view(B, q_blocks, block_q, H, -1)
        indices_q = idx_view[:, :, 0, :, :].contiguous()
    else:
        indices_q = indices

    valid_mask = (indices_q >= 0)  # (B, q_blocks, H, K)
    safe_indices = indices_q.clamp_min(0)

    N = L // chunk_size
    k_chunks = rearrange(k, 'B (N S) h d -> B N S h d', S=chunk_size)
    v_chunks = rearrange(v, 'B (N S) h d -> B N S h d', S=chunk_size)

    idx_flat = rearrange(safe_indices, 'B Bq h K -> B (Bq K) h').unsqueeze(2).unsqueeze(-1)  # (B, BqK, 1, h, 1)
    idx_flat = idx_flat.expand(-1, -1, chunk_size, -1, D)                                   # (B, BqK, S, h, D)
    idx_flat = idx_flat.long()  
    gather_k = k_chunks.gather(dim=1, index=idx_flat)  # (B, BqK, S, h, D)
    gather_v = v_chunks.gather(dim=1, index=idx_flat)

    gather_k = rearrange(gather_k, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)
    gather_v = rearrange(gather_v, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)

    k_ = torch.repeat_interleave(gather_k, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)
    v_ = torch.repeat_interleave(gather_v, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)

    q_chunked = rearrange(q, 'B (Bq X) hq d -> B Bq X hq d', X=block_q)

    # qk: (B, Bq, X, S, K, HQ)
    qk = torch.einsum('b q x h d, b q s k h d -> b q x s k h', q_chunked.float(), k_.float())
    qk = qk * float(sm_scale)
    
    if mask_last_token:
        qk[:, :, :, -1, :, :] = float("-inf")

    p = torch.softmax(qk, dim=3)

    # o_k: (B, Bq, X, K, HQ, D)
    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    w_masked = weights.clone()
    valid_mask_expanded = torch.repeat_interleave(valid_mask, dim=-2, repeats=G)
    w_masked = w_masked.masked_fill(~valid_mask_expanded, 0)
    w_exp = w_masked.float() # (B, Bq, HQ, K)
    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'b q x h d -> b (q x) h d')
    return o_ref.to(torch.float32)




def make_dq_layout_hsa(dQ):

    NV, B, L, HQ, D = dQ.shape
    return T.Layout(dQ.shape,
    lambda nv, b, l, h, d:   [nv,b,l, h//8, d//16, (d%16)//2, (h%8), (d%2)]
 )

@tilelang.jit(
    out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def hsa_bwd_postprocess(nv, batch, q_len, heads, head_dim):
    shape = [nv, batch, q_len, heads, head_dim]
    accum_dtype = "float"
    dtype = "bfloat16"
    blk = 64 # 可以调整的块大小

    @T.prim_func
    def hsa_post(
            dQ_swizzled: T.Tensor(shape, accum_dtype),
            dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(q_len, blk), heads, batch * nv, threads=32) as (bx, by, bz):
            i_nv = bz // batch
            i_b = bz % batch
            
            T.annotate_layout({dQ_swizzled: make_dq_layout_hsa(dQ_swizzled)})
            
            T.copy(
                dQ_swizzled[i_nv, i_b, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[i_nv, i_b, bx * blk:(bx + 1) * blk, by, :],
            )
    return hsa_post



@tilelang.jit(
    pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hsa_kernel_block_mask(batch, heads, q_len, kv_len, selected_blocks, block_size, dtype="int32"):
    from tilelang import language as T

    S = selected_blocks
    NS_kv = tilelang.cdiv(kv_len, block_size)

    block_indices_shape = [batch, q_len, heads, selected_blocks]
    block_mask_shape = [batch, q_len, heads, NS_kv]

    @T.prim_func
    def build_block_mask(
        BlockIndices: T.Tensor(block_indices_shape, dtype),
        BlockMask: T.Tensor(block_mask_shape, dtype),
    ):
        with T.Kernel(q_len, batch, heads * S, threads=32) as (i_t, i_b, i_hs):
            i_h = i_hs // S
            i_s = i_hs % S

            block_idx = BlockIndices[i_b, i_t, i_h, i_s]

            if block_idx >= 0 and block_idx < NS_kv:
                BlockMask[i_b, i_t, i_h, block_idx] = i_s

    return build_block_mask




import torch
def build_block_indices_block_M(
    B: int,
    SEQ_LEN: int,
    H: int,
    S: int,
    block_size: int,
    overlap_ratio: float = 0.5,
    block_M: int = 2,
    device: str = "cuda",
) -> torch.Tensor:
    """
    构造 block_indices 张量：
    - 在每个长度为 block_M 的 token 窗口内，相邻 token (t, t+1) 的选中块集合满足给定的重叠度。
    - 每个 query 的选中索引升序排列。
    - 不足 S 的填充为 -1。

    参数:
        B: batch 大小
        SEQ_LEN: 序列长度
        H: head 数
        S: 每个 query 选择的 block 数量
        block_size: 每个 block 的大小
        overlap_ratio: 相邻 token 之间的重叠比例 [0,1]
        block_M: 每个窗口内的 token 数（例如 pair=2，对 block_M kernel 可设为 M）
        device: 输出所在设备
    """
    import torch

    assert 0.0 <= overlap_ratio <= 1.0, "overlap_ratio 必须在 [0, 1]"
    assert block_M >= 1, "block_M 必须 >= 1"

    num_blocks = SEQ_LEN // block_size
    block_indices = torch.full((B, SEQ_LEN, H, S), -1, dtype=torch.int32, device=device)

    for b in range(B):
        for h in range(H):
            # 按 block_M 为一组滑动
            t = 0
            while t < SEQ_LEN:
                block_start = t
                block_end = min(t + block_M, SEQ_LEN)

                # 对这个窗口里的第一个 token 先生成索引
                t0 = block_start
                max_blocks_t0 = min(t0 // block_size + 1, num_blocks)
                if max_blocks_t0 <= 0:
                    # 这个 token 没有可用 block，直接跳过到下一组
                    t = block_end
                    continue

                num_select = min(S, max_blocks_t0)
                # 第一个 token 随机选
                idx_prev = torch.randperm(max_blocks_t0, device=device)[:num_select]
                idx_prev_sorted = torch.sort(idx_prev)[0]
                block_indices[b, t0, h, :len(idx_prev_sorted)] = idx_prev_sorted

                # 对窗口内其余 token：保证与前一个 token 保持 overlap_ratio
                for tt in range(t0 + 1, block_end):
                    max_blocks_tt = min(tt // block_size + 1, num_blocks)
                    if max_blocks_tt <= 0:
                        continue

                    num_select_tt = min(S, max_blocks_tt)

                    # 允许重叠的最大候选：当前 token 可用 block 与上一个的交集
                    # 这里简化为：从 idx_prev 中选 overlapped，再从其余可用 block 中选新块
                    num_overlap = int(overlap_ratio * num_select_tt)
                    num_overlap = min(num_overlap, len(idx_prev))

                    # 重叠部分：从 idx_prev 中随机取 num_overlap 个
                    if num_overlap > 0:
                        perm_prev = torch.randperm(len(idx_prev), device=device)
                        overlap_blocks = idx_prev[perm_prev[:num_overlap]]
                    else:
                        overlap_blocks = idx_prev.new_empty((0,), dtype=idx_prev.dtype)

                    # 剩余 block 候选：当前 token 可用的所有 block 中，剔除 overlap_blocks
                    remaining_blocks_all = torch.arange(max_blocks_tt, device=device)
                    mask = torch.ones(max_blocks_tt, dtype=torch.bool, device=device)
                    if overlap_blocks.numel() > 0:
                        mask[overlap_blocks] = False
                    candidates = remaining_blocks_all[mask]

                    num_new = num_select_tt - num_overlap
                    if num_new > 0 and candidates.numel() > 0:
                        perm_cand = torch.randperm(candidates.numel(), device=device)
                        new_blocks = candidates[perm_cand[:num_new]]
                        idx_curr = torch.cat([overlap_blocks, new_blocks], dim=0)
                    else:
                        idx_curr = overlap_blocks.clone()

                    # 升序写入
                    idx_curr_sorted = torch.sort(idx_curr)[0]
                    block_indices[b, tt, h, :len(idx_curr_sorted)] = idx_curr_sorted

                    # 下一轮的“上一个 token”索引
                    idx_prev = idx_curr

                # 跳到下一个窗口
                t = block_end

    return block_indices









@tilelang.jit(
    out_idx=[-1, -2], # 输出 Output 和 LSE
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_block_M(batch, heads, q_len, kv_len, head_dim,
                                          scale=None, block_size=64, groups=16,
                                          selected_blocks=16, block_M = 0, mask_last_token=False):
    
    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True
    
    # === Scale 处理 (参考 NSA) ===
    log2e = 1.44269504
    if scale is None:
        scale_origin = (1.0 / head_dim)**0.5 
    else:
        scale_origin = scale

    # Kernel 内部使用 base-2 scale
    scale = scale_origin * log2e

    head_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    kv_shape = [batch, kv_len, head_kv, head_dim]
    weight_shape = [batch, q_len, heads, selected_blocks]
    block_indices_shape = [batch, q_len, head_kv, selected_blocks]
    
    # LSE 形状: [batch, q_len, heads, selected_blocks]
    # 每个 Block 都有自己独立的 LSE
    lse_shape = [batch, q_len, heads, selected_blocks] 
    
    block_indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    
    step_S = 32  # 32比64快很多
    num_inner_steps = T.ceildiv(block_size, step_S)
    block_T = min(128, tilelang.math.next_power_of_2(head_dim))

    NV = tilelang.cdiv(head_dim, block_T)
    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, groups)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)
    
    M_G = M * groups
    S = selected_blocks
    BK = BV = block_T
    threads = 128

    @T.prim_func
    def hsa_block_M(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            LSE: T.Tensor(lse_shape, accum_dtype), # 新增 LSE 参数
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(tilelang.cdiv(q_len, M), NV, batch * head_kv, threads=threads) as (bx, by, bz):

            Q_shared = T.alloc_shared([M_G, BK], dtype)
            K_shared = T.alloc_shared([step_S, BK], dtype) 
            V_shared = T.alloc_shared([step_S, BV], dtype) 
            O_shared = T.alloc_shared([M_G, BV], dtype)

            acc_o = T.alloc_fragment([M_G, BV], accum_dtype)
            block_acc = T.alloc_fragment([M_G, BV], accum_dtype)
            
            # 状态变量
            running_max = T.alloc_fragment([M_G], accum_dtype)
            running_sum = T.alloc_fragment([M_G], accum_dtype)
            local_max_all = T.alloc_fragment([M_G], accum_dtype) 
            local_sum_all = T.alloc_fragment([M_G], accum_dtype) 
            weight_vec = T.alloc_fragment([M_G], dtype) 
            
            acc_s = T.alloc_fragment([M_G, step_S], accum_dtype)
            acc_s_cast = T.alloc_fragment([M_G, step_S], dtype)
            P_shared = T.alloc_shared([M_G, step_S], accum_dtype) 
            block_acc_tmp = T.alloc_fragment([M_G, BV], accum_dtype)

            # 归并排序相关
            merged_indices = T.alloc_shared([S * M], block_indices_dtype)
            block_ownership = T.alloc_shared([S * M], "int32")
            merged_len = T.alloc_shared([1], "int32")
            chunk_weights = T.alloc_shared([S * M, M_G], dtype)
            
            # === 新增：用于存储原始 block index 的 Shared Memory ===
            # 用于最后将 LSE 写回正确的位置
            chunk_s_indices = T.alloc_shared([S * M, M], "int32")

            i_t_base_idx, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv
            base_t = i_t_base_idx * M

            # 1. 加载 Q
            T.fill(Q_shared, 0)
            for q_idx1 in T.serial(M):
                tq = base_t + q_idx1
                if tq < q_len:
                    T.copy(Q[i_b, tq, i_h * groups:(i_h + 1) * groups, :],
                           Q_shared[q_idx1 * groups:(q_idx1 + 1) * groups, :])

            T.fill(acc_o, 0)
            
            # --- 归并 Block Indices 逻辑 ---
            T.fill(merged_indices, -1)
            T.fill(block_ownership, 0)
            T.fill(chunk_weights, 0)
            T.fill(chunk_s_indices, -1) 
            T.fill(merged_len, 0)
            
            W_local_shared = T.alloc_shared([M_G, S], dtype)
            T.fill(W_local_shared, 0.0)
            for i, s_idx in T.Parallel(M_G, S): # 修改加载逻辑
                q_idx = i // groups
                g = i % groups
                tq = base_t + q_idx
                if tq < q_len:
                    W_local_shared[i, s_idx] = W[i_b, tq, i_h * groups + g, s_idx]

            if T.get_thread_binding() == 0:
                valid_lens = T.alloc_fragment([M], "int32")
                pointers = T.alloc_fragment([M], "int32")
                cur_val = T.alloc_fragment([M], "int32")
                
                k = T.alloc_var("int32")
                
                for q_idx3 in T.Parallel(M):
                    tq = base_t + q_idx3
                    valid_lens[q_idx3] = 0
                    pointers[q_idx3] = 0
                    if tq < q_len:
                        for j in T.serial(S):
                            if BlockIndices[i_b, tq, i_h, j] >= 0:
                                valid_lens[q_idx3] = valid_lens[q_idx3] + 1
                            else:
                                T.loop_break()
                                
                k = 0
                for _ in T.serial(S * M):
                    min_val = T.alloc_var("int32")
                    has_valid = T.alloc_var("int32")
                    min_val = 2147483647
                    has_valid = 0
                    
                    for q_idx4 in T.serial(M):
                        tq = base_t + q_idx4
                        if tq < q_len and pointers[q_idx4] < valid_lens[q_idx4]:
                                has_valid = 1
                                val_q = BlockIndices[i_b, tq, i_h, pointers[q_idx4]]
                                cur_val[q_idx4] = val_q
                                if val_q < min_val:
                                    min_val = val_q
                        else:
                            cur_val[q_idx4] = 2147483647

                    if has_valid == 0:
                        T.loop_break()

                    merged_indices[k] = min_val
                    ownership_mask = T.alloc_var("int32")
                    ownership_mask = 0
                    for q_idx5 in T.serial(M):
                        if cur_val[q_idx5] == min_val:
                            ownership_mask = ownership_mask | (1 << q_idx5)
                            for g in T.serial(groups):
                                chunk_weights[k, q_idx5 * groups + g] = W_local_shared[q_idx5 * groups + g, pointers[q_idx5]]
                            # 保存原始 s_idx
                            chunk_s_indices[k, q_idx5] = pointers[q_idx5]
                            
                            pointers[q_idx5] = pointers[q_idx5] + 1
                    block_ownership[k] = ownership_mask
                    k = k + 1
                    
                merged_len[0] = k
            T.sync_threads()
            
            # --- 核心计算 ---
            merged_len_local = merged_len[0]

            for i in T.serial(merged_len_local):
                blk_idx = merged_indices[i]
                ownership = block_ownership[i]
                
                # 广播权重
                for r in T.Parallel(M_G):
                    q_idx = r // groups
                    if (ownership & (1 << q_idx)) != 0:
                        weight_vec[r] = chunk_weights[i, r] # 直接从 r 索引
                    else:
                        weight_vec[r] = 0.0
                
                # 初始化 Block 状态
                T.fill(block_acc, 0)
                T.fill(running_max, -T.infinity(accum_dtype))
                T.fill(running_sum, 0)

                for step_idx in T.serial(num_inner_steps):
                    s_offset = step_idx * step_S
                    i_s_start = blk_idx * block_size + s_offset

                    T.copy(K[i_b, i_s_start : i_s_start + step_S, i_h, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # === 修复：使用位运算和 if_then_else 处理 Mask 和 -inf ===
                    if enable_last_token_mask:
                        for r, s in T.Parallel(M_G, step_S):
                            q_idx = r // groups
                            own = (ownership & (1 << q_idx)) != 0
                            not_last = (s_offset + s) != (block_size - 1)
                            valid = own & not_last
                            acc_s[r, s] = T.if_then_else(valid, acc_s[r, s], -T.infinity(accum_dtype))
                    else:
                        for r, s in T.Parallel(M_G, step_S):
                            q_idx = r // groups
                            own = (ownership & (1 << q_idx)) != 0
                            acc_s[r, s] = T.if_then_else(own, acc_s[r, s], -T.infinity(accum_dtype))
                    
                    T.reduce_max(acc_s, local_max_all, dim=1, clear=True)
                    
                    # === Scale 处理 (参考 NSA) ===
                    scale_prev_buffer = T.alloc_fragment([M_G], dtype)
                    for r in T.Parallel(M_G):  
                        m_prev = running_max[r]  
                        m_curr = local_max_all[r]  
                        m_new = T.max(m_prev, m_curr)  
                        
                        # NSA 风格: exp2(m_prev * scale - m_new * scale)
                        # 并包含数值稳定性保护
                        scale_prev = T.if_then_else(
                            m_new == -T.infinity(accum_dtype),
                            1.0,
                            T.exp2(m_prev * scale - m_new * scale)
                        )
                        
                        running_max[r] = m_new  
                        running_sum[r] = running_sum[r] * scale_prev  
                        scale_prev_buffer[r] = scale_prev 

                    for r, v in T.Parallel(M_G, BV):  
                        block_acc[r, v] *= scale_prev_buffer[r]

                    for r, s in T.Parallel(M_G, step_S):
                        # NSA 风格: exp2(acc_s * scale - m_new * scale)
                        val = T.if_then_else(
                            acc_s[r, s] == -T.infinity(accum_dtype),
                            0.0,
                            T.exp2(acc_s[r, s] * scale - running_max[r] * scale)
                        )
                        acc_s[r, s] = val
                    
                    T.reduce_sum(acc_s, local_sum_all, dim=1, clear=True)
                    
                    for r in T.Parallel(M_G):
                        running_sum[r] += local_sum_all[r]

                    T.copy(acc_s, P_shared) 
                    T.copy(V[i_b, i_s_start : i_s_start + step_S, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    
                    T.clear(block_acc_tmp)
                    T.copy(P_shared, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, block_acc_tmp, policy=T.GemmWarpPolicy.FullRow)
                    
                    for r, c in T.Parallel(M_G, BV):
                        block_acc[r, c] += block_acc_tmp[r, c]

                # --- Block 结束：写回 LSE ---
                # LSE = (m * scale + log2(sum)) / log2(e) 
                # 这里 scale 已经是 scale_origin * log2e，所以公式等价于: m * scale_origin + ln(sum)
                # 使用 chunk_s_indices 获取原始 s_idx
                for r in T.Parallel(M_G):
                    q_idx = r // groups
                    if (ownership & (1 << q_idx)) != 0:
                        s_idx = chunk_s_indices[i, q_idx]
                        
                        m_val = running_max[r]
                        sum_val = running_sum[r]
                        
                        # 计算 LSE (Base e)
                        lse_val = T.if_then_else(
                            m_val == -T.infinity(accum_dtype),
                            -T.infinity(accum_dtype),
                            (m_val * scale + T.log2(sum_val)) / log2e
                        )
                        
                        tq = base_t + q_idx
                        g_idx = r % groups
                        h_idx = i_h * groups + g_idx
                        LSE[i_b, tq, h_idx, s_idx] = lse_val


                # --- Block 结束：归一化并加权累加到全局 acc_o ---
                for r, v in T.Parallel(M_G, BV):
                    norm_val = block_acc[r, v] / T.max(running_sum[r], 1e-6) 
                    acc_o[r, v] += norm_val * weight_vec[r]

            T.copy(acc_o, O_shared)
            for q_idx8 in T.serial(M):
                tq = base_t + q_idx8
                if tq < q_len:
                    h_start = q_idx8 * groups
                    for g, v in T.Parallel(groups, BV):
                        Output[i_b, tq, i_h * groups + g, i_v * BV + v] = O_shared[h_start + g, v]

    return hsa_block_M




# @tilelang.jit(
#     pass_configs={
#         tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
#         tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
#         tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
#     }
# )
# def hierarchical_sparse_attention_bwd_dqkv_block_M_inverse(
#     batch, heads, q_len, kv_len, head_dim,
#     scale=None, block_size=64, groups=16, selected_blocks=16,
#     dtype="bfloat16", accum_dtype="float", block_M = 0, mask_last_token=False,
# ):
#     from tilelang import language as T
    
#     enable_last_token_mask = False
#     if mask_last_token:
#         enable_last_token_mask = True

#     if scale is None:
#         sm_scale = (1.0 / head_dim)**0.5
#     else:
#         sm_scale = scale
#     log2e = 1.44269504
#     scale_log2 = sm_scale * log2e

#     B = batch
#     BS = block_size
#     G = groups
#     Vdim = head_dim
#     Kdim = head_dim
#     BK = tilelang.next_power_of_2(Kdim)
#     BV = min(128, tilelang.next_power_of_2(head_dim))
#     NV = tilelang.cdiv(Vdim, BV)
#     S = selected_blocks

#     heads_kv = heads // groups
#     q_shape = [batch, q_len, heads, head_dim]
#     k_shape = [batch, kv_len, heads_kv, head_dim]
#     v_shape = [batch, kv_len, heads_kv, head_dim]
#     do_shape = [batch, q_len, heads, head_dim]
#     dq_shape = [NV, batch, q_len, heads, head_dim]
#     dk_shape = [NV, batch, kv_len, heads_kv, head_dim]
#     dv_shape = [batch, kv_len, heads_kv, head_dim]
#     weight_shape = [batch, q_len, heads, selected_blocks] # 修改为 heads
#     dw_shape = [batch, q_len, heads, selected_blocks]     # 修改为 heads
#     block_indices_shape = [batch, q_len, heads_kv, selected_blocks]
#     lse_shape = [batch, q_len, heads, selected_blocks]

#     # 动态计算 M
#     MIN_GEMM_ROWS = 16
#     M_min = tilelang.cdiv(MIN_GEMM_ROWS, G)
#     if block_M is None or block_M <= 0:
#         M = M_min
#     else:
#         M = max(block_M, M_min)
#     print("Using M =", M, "for bwd_block_M_inverse kernel")
#     M_G = M * G

#     step_S = 64
#     print("step_S",step_S)
#     num_inner_steps = T.ceildiv(BS, step_S)

#     num_threads = 256
#     num_stages = 0

#     @T.prim_func
#     def hsa_bwd_dqkv_block_M_inverse(
#         Q: T.Tensor(q_shape, dtype),
#         K: T.Tensor(k_shape, dtype),
#         V: T.Tensor(v_shape, dtype),
#         W: T.Tensor(weight_shape, dtype),
#         LSE: T.Tensor(lse_shape, accum_dtype),
#         DO: T.Tensor(do_shape, dtype),
#         BlockIndices: T.Tensor(block_indices_shape, "int32"),
#         DQ: T.Tensor(dq_shape, dtype),
#         DK: T.Tensor(dk_shape, accum_dtype),
#         DV: T.Tensor(dv_shape, accum_dtype),
#         DW: T.Tensor(dw_shape, accum_dtype),
#     ):
#         with T.Kernel(tilelang.cdiv(q_len, M), NV, B * heads_kv, threads=num_threads) as (bx, by, bz):
#             i_t_base_idx, i_v, i_bh = bx, by, bz
#             i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
#             base_t = i_t_base_idx * M

#             # === Shared Memory ===
#             # 关键修改：K/V shared 大小改为 step_S
#             Q_shared = T.alloc_shared([M_G, BK], dtype)
#             K_shared = T.alloc_shared([step_S, BK], dtype)
#             V_shared = T.alloc_shared([step_S, BV], dtype)
            
#             dO_shared = T.alloc_shared([M_G, BV], dtype)
#             dO_weighted_shared = T.alloc_shared([M_G, BV], dtype)
            
#             # P_shared 大小改为 step_S
#             P_shared = T.alloc_shared([M_G, step_S], dtype)
#             dS_shared = T.alloc_shared([M_G, step_S], dtype)

#             # === Fragments ===
#             dQ_accum = T.alloc_fragment([M_G, BK], dtype)
#             qk_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
#             P_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
            
#             T_raw_frag = T.alloc_fragment([M_G, BV], accum_dtype)
#             T_raw_tmp = T.alloc_fragment([M_G, BV], accum_dtype)
            
#             dV_tmp = T.alloc_fragment([step_S, BV], accum_dtype)
#             dK_tmp = T.alloc_fragment([step_S, BK], accum_dtype)
#             dQ_tmp = T.alloc_fragment([M_G, BK], accum_dtype)
            
#             dP_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
#             pd_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
#             dS_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
            
#             delta_tmp = T.alloc_fragment([M_G], accum_dtype)
#             delta_rows = T.alloc_fragment([M_G], accum_dtype)

#             # === Accum Buffers for Atomic Add ===
#             # 使用小块 shared memory 进行中转，比直接用fragment进行atomic add更快
#             # dV_accum_shared = T.alloc_shared([step_S, BV], accum_dtype)
#             # dK_accum_shared = T.alloc_shared([step_S, BK], accum_dtype)
            
#             # T_raw_shared = T.alloc_shared([M_G, BV], dtype)
#             # dw_partial = T.alloc_fragment([G, BV], accum_dtype)
#             dw_g_frag = T.alloc_fragment([G], accum_dtype)
#             sum_w = T.alloc_fragment([1], accum_dtype)

#             # === Meta Data ===
#             merged_indices = T.alloc_shared([S * M], "int32")
#             block_ownership = T.alloc_shared([S * M], "int32")
#             merged_len = T.alloc_shared([1], "int32")
#             chunk_weights = T.alloc_shared([S * M, M_G], "bfloat16") # 修改为 M_G
#             W_local_shared = T.alloc_shared([M_G, S], "bfloat16")   # 修改为 M_G
#             merged_s_indices = T.alloc_shared([S * M, M], "int32")
            
#             lse_vec = T.alloc_fragment([M_G], accum_dtype)
            
#             # [新增] 用于 dW 计算的中间变量
#             dw_row_sum_frag = T.alloc_fragment([M_G], accum_dtype)
#             # [新增] 只需要一个很小的 shared buffer 来存规约后的结果，以便支持动态索引
#             dw_row_sum_shared = T.alloc_shared([M_G], accum_dtype)

#             # === Load Q & dO ===
#             T.fill(Q_shared, 0)
#             T.fill(dO_shared, 0)
#             for q_idx in T.serial(M):
#                 tq = base_t + q_idx
#                 if tq < q_len:
#                     h_start = q_idx * G
#                     T.copy(Q[i_b, tq, i_h * G:(i_h + 1) * G, :], Q_shared[h_start:h_start + G, :])
#                     T.copy(DO[i_b, tq, i_h * G:(i_h + 1) * G, i_v * BV:(i_v + 1) * BV], dO_shared[h_start:h_start + G, :])

#             T.fill(dQ_accum, 0)
            
#             # === Merging Logic (Same as before) ===
#             T.fill(merged_indices, -1)
#             T.fill(block_ownership, 0)
#             T.fill(chunk_weights, 0)
#             T.fill(merged_len, 0)
#             T.fill(merged_s_indices, -1)
#             T.fill(W_local_shared, 0.0)
#             for i, s_idx in T.Parallel(M_G, S): # 修改加载逻辑
#                 q_idx = i // G
#                 g = i % G
#                 tq = base_t + q_idx
#                 if tq < q_len:
#                     W_local_shared[i, s_idx] = W[i_b, tq, i_h * G + g, s_idx]

#             if T.get_thread_binding() == 0:
#                 valid_lens = T.alloc_fragment([M], "int32")
#                 pointers = T.alloc_fragment([M], "int32")
#                 merge_k = T.alloc_var("int32")
#                 cur_val = T.alloc_fragment([M], "int32")
                
#                 for q_idx in T.Parallel(M):
#                     tq = base_t + q_idx
#                     valid_lens[q_idx] = 0
#                     pointers[q_idx] = 0
#                     if tq < q_len:
#                         for j in T.serial(S):
#                             if BlockIndices[i_b, tq, i_h, j] >= 0:
#                                 valid_lens[q_idx] = valid_lens[q_idx] + 1
#                             else:
#                                 T.loop_break()
                    
#                 # M 路归并
#                 merge_k = 0
#                 ownership_mask = T.alloc_var("int32")
#                 for _ in T.serial(S * M):
#                     min_val = T.alloc_var("int32")
#                     min_val = 2147483647  # INT_MAX
#                     has_valid = T.alloc_var("int32")
#                     has_valid = 0
                    
#                     for q_idx in T.serial(M):
#                         tq = base_t + q_idx
#                         if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
#                             has_valid = 1
#                             val_q = BlockIndices[i_b, tq, i_h, pointers[q_idx]]
#                             cur_val[q_idx] = val_q
#                             if val_q < min_val:
#                                 min_val = val_q
#                         else:
#                             cur_val[q_idx] = 2147483647

#                     if has_valid == 0:
#                         T.loop_break()

#                     merged_indices[merge_k] = min_val
#                     ownership_mask = 0
#                     for q_idx in T.serial(M):
#                         tq = base_t + q_idx
#                         if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
#                             s_idx = pointers[q_idx]
#                             val_q = cur_val[q_idx]
#                             if val_q == min_val:
#                                 ownership_mask = ownership_mask | (1 << q_idx)
#                                 for g in T.serial(G): # 记录每个 head 的权重
#                                     chunk_weights[merge_k, q_idx * G + g] = W_local_shared[q_idx * G + g, s_idx]
#                                 merged_s_indices[merge_k, q_idx] = s_idx
#                                 pointers[q_idx] = pointers[q_idx] + 1
#                     block_ownership[merge_k] = ownership_mask
#                     merge_k = merge_k + 1
                    
#                 merged_len[0] = merge_k

#             T.sync_threads()

#             merged_len_local = merged_len[0]
#             h_start = T.alloc_var("int32")

#             # === Main Loop over Merged Blocks ===
#             for i in T.Pipelined(merged_len_local, num_stages=num_stages):
#                 blk_idx = merged_indices[i]
#                 i_s_global = blk_idx * BS
#                 ownership = block_ownership[i]

#                 if blk_idx >= 0:
#                     # 1. Prepare dO_weighted and LSE for this block
#                     for g, v in T.Parallel(M_G, BV):
#                         dO_weighted_shared[g, v] = chunk_weights[i, g] * dO_shared[g, v]
                    
#                     for r in T.Parallel(M_G):
#                         q_idx = r // G
#                         g_local = r - q_idx * G
#                         tq = base_t + q_idx
#                         s_idx_local = merged_s_indices[i, q_idx]
#                         if (ownership & (1 << q_idx)) != 0 and s_idx_local != -1:
#                             lse_vec[r] = LSE[i_b, tq, i_h * G + g_local, s_idx_local]
#                         else:
#                             lse_vec[r] = 0.0 # Should be masked out anyway

#                     T.fill(delta_rows, 0)
#                     T.fill(T_raw_frag, 0)

#                     # # === Pass 1: Accumulate O & dV ===
#                     for step_idx in T.serial(num_inner_steps):
#                         s_offset = step_idx * step_S
#                         i_kv = i_s_global + s_offset

#                         # Load K, V sub-blocks
#                         T.copy(K[i_b, i_kv:i_kv + step_S, i_h, :], K_shared)
#                         T.copy(V[i_b, i_kv:i_kv + step_S, i_h, i_v * BV:(i_v + 1) * BV], V_shared)

#                         # Calc P
#                         T.clear(qk_frag)
#                         T.gemm(Q_shared, K_shared, qk_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        
#                         # Masking logic
#                         if enable_last_token_mask:
#                             for r, s in T.Parallel(M_G, step_S):
#                                 q_idx = r // G
#                                 valid_row = (ownership & (1 << q_idx)) != 0
#                                 not_last = (s_offset + s) != (BS - 1)
#                                 valid = valid_row & not_last
#                                 qk_frag[r, s] = T.if_then_else(valid, qk_frag[r, s], -T.infinity(accum_dtype))
#                         else:
#                             for r, s in T.Parallel(M_G, step_S):
#                                 q_idx = r // G
#                                 valid_row = (ownership & (1 << q_idx)) != 0
#                                 qk_frag[r, s] = T.if_then_else(valid_row, qk_frag[r, s], -T.infinity(accum_dtype))

#                         for r, s in T.Parallel(M_G, step_S):
#                             P_frag[r, s] = T.exp2(qk_frag[r, s] * scale_log2 - lse_vec[r] * log2e)
                        
#                         T.copy(P_frag, P_shared)

#                         # Accumulate O (T_raw)
#                         T.clear(T_raw_tmp)
#                         T.gemm(P_shared, V_shared, T_raw_tmp, policy=T.GemmWarpPolicy.FullRow)
#                         for r, v in T.Parallel(M_G, BV):
#                             T_raw_frag[r, v] += T_raw_tmp[r, v]

#                         # Calc dP part for Delta
#                         T.clear(dP_frag)
#                         T.gemm(dO_weighted_shared, V_shared, dP_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
#                         for r, s in T.Parallel(M_G, step_S):
#                             pd_frag[r, s] = P_frag[r, s] * dP_frag[r, s]
#                         T.reduce_sum(pd_frag, delta_tmp, dim=1, clear=True)
#                         for r in T.Parallel(M_G):
#                             delta_rows[r] += delta_tmp[r]
                        
#                         # Calc dV & Atomic Add
#                         T.clear(dV_tmp)
#                         T.gemm(P_shared, dO_weighted_shared, dV_tmp, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                        
#                         # Copy to shared for atomic add
#                         T.copy(dV_tmp, V_shared) 
                        
#                         for s, v in T.Parallel(step_S, BV):
#                             T.atomic_add(DV[i_b, i_kv + s, i_h, i_v * BV + v], V_shared[s, v])

                        
                    
#                     # === Middle: Calc dW ===
#                     # 优化策略：
#                     # 1. 直接在 Fragment 上做逐元素乘法 (dO * T_raw)，不落 Shared
#                     # 2. 先把 BV 维度 reduce 掉，得到 [M_G] 大小的向量
#                     # 3. 只把这个小向量存入 Shared，供后续动态索引使用
                    
#                     # Step 1: Element-wise Mul (In-place update T_raw_frag)
#                     # T_raw_frag 是 fp32, dO_shared 是 bf16，直接相乘结果为 fp32
#                     for r, v in T.Parallel(M_G, BV):
#                         T_raw_frag[r, v] = T_raw_frag[r, v] * dO_shared[r, v]

#                     # Step 2: Reduce over BV dimension -> [M_G]
#                     T.reduce_sum(T_raw_frag, dw_row_sum_frag, dim=1, clear=True)
                    
#                     # Step 3: Copy small result to Shared Memory
#                     T.copy(dw_row_sum_frag, dw_row_sum_shared)

#                     # Step 4: Write output for each query and each head in parallel
#                     for idx in T.Parallel(M * G):
#                         q_idx = idx // G
#                         g = idx % G
                        
#                         # 检查ownership时只需要检查query维度
#                         if (ownership & (1 << q_idx)) != 0:
#                             tq = base_t + q_idx
#                             s_idx_local = merged_s_indices[i, q_idx]
                            
#                             # 有效性检查：query在范围内且s索引有效
#                             if tq < q_len and s_idx_local != -1:
#                                 DW[i_b, tq, i_h * G + g, s_idx_local] = dw_row_sum_shared[idx]


#                     # === Pass 2: Calc dS, dK, dQ ===
#                     for step_idx in T.serial(num_inner_steps):
#                         s_offset = step_idx * step_S
#                         i_kv = i_s_global + s_offset

#                         # Reload K, V
#                         T.copy(K[i_b, i_kv:i_kv + step_S, i_h, :], K_shared)
#                         # V is not needed for dK/dQ, but was needed for dP in Pass 1. 
#                         # In Pass 2 we only need K for dQ.

#                         # Re-calc P
#                         T.clear(qk_frag)
#                         T.gemm(Q_shared, K_shared, qk_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        
#                         if enable_last_token_mask:
#                             for r, s in T.Parallel(M_G, step_S):
#                                 q_idx = r // G
#                                 valid_row = (ownership & (1 << q_idx)) != 0
#                                 not_last = (s_offset + s) != (BS - 1)
#                                 valid = valid_row & not_last
#                                 qk_frag[r, s] = T.if_then_else(valid, qk_frag[r, s], -T.infinity(accum_dtype))
#                         else:
#                             for r, s in T.Parallel(M_G, step_S):
#                                 q_idx = r // G
#                                 valid_row = (ownership & (1 << q_idx)) != 0
#                                 qk_frag[r, s] = T.if_then_else(valid_row, qk_frag[r, s], -T.infinity(accum_dtype))

#                         for r, s in T.Parallel(M_G, step_S):
#                             P_frag[r, s] = T.exp2(qk_frag[r, s] * scale_log2 - lse_vec[r] * log2e)

#                         # Re-calc dP (Wait, we need dP = dO_weighted @ V.T)
#                         # So we DO need V_shared in Pass 2 if we didn't save dP.
#                         # To save shared memory, we re-load V and re-compute dP.
#                         T.copy(V[i_b, i_kv:i_kv + step_S, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
                        
#                         T.clear(dP_frag)
#                         T.gemm(dO_weighted_shared, V_shared, dP_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

#                         # Calc dS
#                         for r, s in T.Parallel(M_G, step_S):
#                             dS_frag[r, s] = sm_scale * (P_frag[r, s] * (dP_frag[r, s] - delta_rows[r]))
                        
#                         T.copy(dS_frag, dS_shared)
                        
                        
#                         # Calc dQ (Accumulate locally)
#                         T.clear(dQ_tmp)
#                         T.gemm(dS_shared, K_shared, dQ_tmp, policy=T.GemmWarpPolicy.FullRow)
#                         for r, k in T.Parallel(M_G, BK):
#                             dQ_accum[r, k] += dQ_tmp[r, k]

#                         # Calc dK & Atomic Add
#                         T.clear(dK_tmp)
#                         T.gemm(dS_shared, Q_shared, dK_tmp, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                        
#                         T.copy(dK_tmp, K_shared)
                        
#                         for s, k in T.Parallel(step_S, BK):
#                             T.atomic_add(DK[i_v, i_b, i_kv + s, i_h, k], K_shared[s, k])

                        

#             # === Write back dQ (Non-atomic) ===
#             # Reuse Q_shared buffer
#             T.copy(dQ_accum, Q_shared)
#             for q_idx in T.serial(M):
#                 tq = base_t + q_idx
#                 if tq < q_len:
#                     h_start = q_idx * G
#                     T.copy(Q_shared[h_start:h_start + G, :], DQ[i_v, i_b, tq, i_h * G:(i_h + 1) * G, :])

#     return hsa_bwd_dqkv_block_M_inverse



@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_bwd_dqkv_block_M_inverse(
    batch, heads, q_len, kv_len, head_dim,
    scale=None, block_size=64, groups=16, selected_blocks=16,
    dtype="bfloat16", accum_dtype="float", block_M = 0, mask_last_token=False,
):
    from tilelang import language as T
    
    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True

    if scale is None:
        sm_scale = (1.0 / head_dim)**0.5
    else:
        sm_scale = scale
    log2e = 1.44269504
    scale_log2 = sm_scale * log2e

    B = batch
    BS = block_size
    G = groups
    Vdim = head_dim
    Kdim = head_dim
    BK = tilelang.next_power_of_2(Kdim)
    BV = min(128, tilelang.next_power_of_2(head_dim))
    NV = tilelang.cdiv(Vdim, BV)
    S = selected_blocks

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    do_shape = [batch, q_len, heads, head_dim]
    dq_shape = [NV, batch, q_len, heads, head_dim]
    dk_shape = [NV, batch, kv_len, heads_kv, head_dim]
    dv_shape = [batch, kv_len, heads_kv, head_dim]
    weight_shape = [batch, q_len, heads, selected_blocks] # 修改为 heads
    dw_shape = [batch, q_len, heads, selected_blocks]     # 修改为 heads
    block_indices_shape = [batch, q_len, heads_kv, selected_blocks]
    lse_shape = [batch, q_len, heads, selected_blocks]

    # 动态计算 M
    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, G)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)
    print("Using M =", M, "for bwd_block_M_inverse kernel")
    M_G = M * G

    step_S = 64
    print("step_S",step_S)
    num_inner_steps = T.ceildiv(BS, step_S)

    num_threads = 256
    num_stages = 0

    @T.prim_func
    def hsa_bwd_dqkv_block_M_inverse(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        W: T.Tensor(weight_shape, dtype),
        LSE: T.Tensor(lse_shape, accum_dtype),
        DO: T.Tensor(do_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, "int32"),
        DQ: T.Tensor(dq_shape, dtype),
        DK: T.Tensor(dk_shape, accum_dtype),
        DV: T.Tensor(dv_shape, accum_dtype),
        DW: T.Tensor(dw_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(q_len, M), NV, B * heads_kv, threads=num_threads) as (bx, by, bz):
            i_t_base_idx, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
            base_t = i_t_base_idx * M

            # === Shared Memory ===
            # 关键修改：K/V shared 大小改为 step_S
            Q_shared = T.alloc_shared([M_G, BK], dtype)
            K_shared = T.alloc_shared([step_S, BK], dtype)
            V_shared = T.alloc_shared([step_S, BV], dtype)
            
            dO_shared = T.alloc_shared([M_G, BV], dtype)
            dO_weighted_shared = T.alloc_shared([M_G, BV], dtype)
            
            # P_shared 大小改为 step_S
            P_shared = T.alloc_shared([M_G, step_S], dtype)
            dS_shared = T.alloc_shared([M_G, step_S], dtype)

            # === Fragments ===
            dQ_accum = T.alloc_fragment([M_G, BK], dtype)
            qk_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
            P_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
            
            T_raw_frag = T.alloc_fragment([M_G, BV], accum_dtype)
            T_raw_tmp = T.alloc_fragment([M_G, BV], accum_dtype)
            
            dV_tmp = T.alloc_fragment([step_S, BV], accum_dtype)
            dK_tmp = T.alloc_fragment([step_S, BK], accum_dtype)
            dQ_tmp = T.alloc_fragment([M_G, BK], accum_dtype)
            
            dP_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
            pd_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
            dS_frag = T.alloc_fragment([M_G, step_S], accum_dtype)
            
            delta_tmp = T.alloc_fragment([M_G], accum_dtype)
            delta_rows = T.alloc_fragment([M_G], accum_dtype)

            # === Accum Buffers for Atomic Add ===
            # 使用小块 shared memory 进行中转，比直接用fragment进行atomic add更快
            # dV_accum_shared = T.alloc_shared([step_S, BV], accum_dtype)
            # dK_accum_shared = T.alloc_shared([step_S, BK], accum_dtype)
            
            # T_raw_shared = T.alloc_shared([M_G, BV], dtype)
            # dw_partial = T.alloc_fragment([G, BV], accum_dtype)
            dw_g_frag = T.alloc_fragment([G], accum_dtype)
            sum_w = T.alloc_fragment([1], accum_dtype)

            # === Meta Data ===
            merged_indices = T.alloc_shared([S * M], "int32")
            block_ownership = T.alloc_shared([S * M], "int32")
            merged_len = T.alloc_shared([1], "int32")
            chunk_weights = T.alloc_shared([S * M, M_G], "bfloat16") # 修改为 M_G
            W_local_shared = T.alloc_shared([M_G, S], "bfloat16")   # 修改为 M_G
            merged_s_indices = T.alloc_shared([S * M, M], "int32")
            
            lse_vec = T.alloc_fragment([M_G], accum_dtype)
            
            # [新增] 用于 dW 计算的中间变量
            dw_row_sum_frag = T.alloc_fragment([M_G], accum_dtype)
            # [新增] 只需要一个很小的 shared buffer 来存规约后的结果，以便支持动态索引
            dw_row_sum_shared = T.alloc_shared([M_G], accum_dtype)

            # === Load Q & dO ===
            T.fill(Q_shared, 0)
            T.fill(dO_shared, 0)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * G
                    T.copy(Q[i_b, tq, i_h * G:(i_h + 1) * G, :], Q_shared[h_start:h_start + G, :])
                    T.copy(DO[i_b, tq, i_h * G:(i_h + 1) * G, i_v * BV:(i_v + 1) * BV], dO_shared[h_start:h_start + G, :])

            T.fill(dQ_accum, 0)
            
            # === Merging Logic (Same as before) ===
            T.fill(merged_indices, -1)
            T.fill(block_ownership, 0)
            T.fill(chunk_weights, 0)
            T.fill(merged_len, 0)
            T.fill(merged_s_indices, -1)
            T.fill(W_local_shared, 0.0)
            for i, s_idx in T.Parallel(M_G, S): # 修改加载逻辑
                q_idx = i // G
                g = i % G
                tq = base_t + q_idx
                if tq < q_len:
                    W_local_shared[i, s_idx] = W[i_b, tq, i_h * G + g, s_idx]

            if T.get_thread_binding() == 0:
                valid_lens = T.alloc_fragment([M], "int32")
                pointers = T.alloc_fragment([M], "int32")
                merge_k = T.alloc_var("int32")
                cur_val = T.alloc_fragment([M], "int32")
                
                for q_idx in T.Parallel(M):
                    tq = base_t + q_idx
                    valid_lens[q_idx] = 0
                    pointers[q_idx] = 0
                    if tq < q_len:
                        for j in T.serial(S):
                            if BlockIndices[i_b, tq, i_h, j] >= 0:
                                valid_lens[q_idx] = valid_lens[q_idx] + 1
                            else:
                                T.loop_break()
                    
                # M 路归并
                merge_k = 0
                ownership_mask = T.alloc_var("int32")
                for _ in T.serial(S * M):
                    min_val = T.alloc_var("int32")
                    min_val = 2147483647  # INT_MAX
                    has_valid = T.alloc_var("int32")
                    has_valid = 0
                    
                    for q_idx in T.serial(M):
                        tq = base_t + q_idx
                        if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
                            has_valid = 1
                            val_q = BlockIndices[i_b, tq, i_h, pointers[q_idx]]
                            cur_val[q_idx] = val_q
                            if val_q < min_val:
                                min_val = val_q
                        else:
                            cur_val[q_idx] = 2147483647

                    if has_valid == 0:
                        T.loop_break()

                    merged_indices[merge_k] = min_val
                    ownership_mask = 0
                    for q_idx in T.serial(M):
                        tq = base_t + q_idx
                        if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
                            s_idx = pointers[q_idx]
                            val_q = cur_val[q_idx]
                            if val_q == min_val:
                                ownership_mask = ownership_mask | (1 << q_idx)
                                for g in T.serial(G): # 记录每个 head 的权重
                                    chunk_weights[merge_k, q_idx * G + g] = W_local_shared[q_idx * G + g, s_idx]
                                merged_s_indices[merge_k, q_idx] = s_idx
                                pointers[q_idx] = pointers[q_idx] + 1
                    block_ownership[merge_k] = ownership_mask
                    merge_k = merge_k + 1
                    
                merged_len[0] = merge_k

            T.sync_threads()

            merged_len_local = merged_len[0]
            h_start = T.alloc_var("int32")

            # === Main Loop over Merged Blocks ===
            for i in T.Pipelined(merged_len_local, num_stages=num_stages):
                blk_idx = merged_indices[i]
                i_s_global = blk_idx * BS
                ownership = block_ownership[i]

                if blk_idx >= 0:
                    # 1. Prepare dO_weighted and LSE for this block
                    # [Optimization] We will update dO_weighted_shared in-place later,
                    # so here we just load the weights.
                    # Wait, we need dO_weighted for dV calculation.
                    # Let's compute dO_weighted = W * dO first as before.
                    for g, v in T.Parallel(M_G, BV):
                        dO_weighted_shared[g, v] = chunk_weights[i, g] * dO_shared[g, v]
                    
                    for r in T.Parallel(M_G):
                        q_idx = r // G
                        g_local = r - q_idx * G
                        tq = base_t + q_idx
                        s_idx_local = merged_s_indices[i, q_idx]
                        if (ownership & (1 << q_idx)) != 0 and s_idx_local != -1:
                            lse_vec[r] = LSE[i_b, tq, i_h * G + g_local, s_idx_local]
                        else:
                            lse_vec[r] = 0.0 # Should be masked out anyway

                    T.fill(delta_rows, 0)
                    T.fill(dw_row_sum_frag, 0) # Init dW accumulator

                    # # === Pass 1: Accumulate dW & dV ===
                    for step_idx in T.serial(num_inner_steps):
                        s_offset = step_idx * step_S
                        i_kv = i_s_global + s_offset

                        # Load K, V sub-blocks
                        T.copy(K[i_b, i_kv:i_kv + step_S, i_h, :], K_shared)
                        T.copy(V[i_b, i_kv:i_kv + step_S, i_h, i_v * BV:(i_v + 1) * BV], V_shared)

                        # Calc P
                        T.clear(qk_frag)
                        T.gemm(Q_shared, K_shared, qk_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        
                        # Masking logic
                        if enable_last_token_mask:
                            for r, s in T.Parallel(M_G, step_S):
                                q_idx = r // G
                                valid_row = (ownership & (1 << q_idx)) != 0
                                not_last = (s_offset + s) != (BS - 1)
                                valid = valid_row & not_last
                                qk_frag[r, s] = T.if_then_else(valid, qk_frag[r, s], -T.infinity(accum_dtype))
                        else:
                            for r, s in T.Parallel(M_G, step_S):
                                q_idx = r // G
                                valid_row = (ownership & (1 << q_idx)) != 0
                                qk_frag[r, s] = T.if_then_else(valid_row, qk_frag[r, s], -T.infinity(accum_dtype))

                        for r, s in T.Parallel(M_G, step_S):
                            P_frag[r, s] = T.exp2(qk_frag[r, s] * scale_log2 - lse_vec[r] * log2e)
                        
                        T.copy(P_frag, P_shared)

                        # [Optimization] Calc Z = dO @ V.T (using unweighted dO)
                        # Reuse dP_frag for Z
                        T.clear(dP_frag)
                        T.gemm(dO_shared, V_shared, dP_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        
                        # [Optimization] Calc dW = Sum(P * Z)
                        # Reuse qk_frag for P * Z
                        for r, s in T.Parallel(M_G, step_S):
                            qk_frag[r, s] = P_frag[r, s] * dP_frag[r, s]
                        
                        # Accumulate dW to row sum
                        T.reduce_sum(qk_frag, delta_tmp, dim=1, clear=True)
                        for r in T.Parallel(M_G):
                            dw_row_sum_frag[r] += delta_tmp[r]

                        # [Optimization] Calc dS part: delta = Sum(P * (W * Z))
                        # W * Z = W * dP_frag.
                        # We can compute P * (W * Z) = (P * Z) * W = qk_frag * W
                        # qk_frag currently holds P * Z.
                        for r, s in T.Parallel(M_G, step_S):
                            # In-place update qk_frag to be P * (W * Z)
                            qk_frag[r, s] = qk_frag[r, s] * chunk_weights[i, r]
                        
                        T.reduce_sum(qk_frag, delta_tmp, dim=1, clear=True)
                        for r in T.Parallel(M_G):
                            delta_rows[r] += delta_tmp[r]
                        
                        # Calc dV & Atomic Add
                        # dV = P.T @ dO_weighted
                        T.clear(dV_tmp)
                        T.gemm(P_shared, dO_weighted_shared, dV_tmp, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                        
                        # Copy to shared for atomic add
                        T.copy(dV_tmp, V_shared) 
                        
                        for s, v in T.Parallel(step_S, BV):
                            T.atomic_add(DV[i_b, i_kv + s, i_h, i_v * BV + v], V_shared[s, v])

                    
                    # === Write dW ===
                    T.copy(dw_row_sum_frag, dw_row_sum_shared)
                    for idx in T.Parallel(M * G):
                        q_idx = idx // G
                        g = idx % G
                        if (ownership & (1 << q_idx)) != 0:
                            tq = base_t + q_idx
                            s_idx_local = merged_s_indices[i, q_idx]
                            if tq < q_len and s_idx_local != -1:
                                DW[i_b, tq, i_h * G + g, s_idx_local] = dw_row_sum_shared[idx]


                    # === Pass 2: Calc dS, dK, dQ ===
                    for step_idx in T.serial(num_inner_steps):
                        s_offset = step_idx * step_S
                        i_kv = i_s_global + s_offset

                        # Reload K, V
                        T.copy(K[i_b, i_kv:i_kv + step_S, i_h, :], K_shared)
                        # Need V for Z calculation again
                        T.copy(V[i_b, i_kv:i_kv + step_S, i_h, i_v * BV:(i_v + 1) * BV], V_shared)

                        # Re-calc P
                        T.clear(qk_frag)
                        T.gemm(Q_shared, K_shared, qk_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        
                        if enable_last_token_mask:
                            for r, s in T.Parallel(M_G, step_S):
                                q_idx = r // G
                                valid_row = (ownership & (1 << q_idx)) != 0
                                not_last = (s_offset + s) != (BS - 1)
                                valid = valid_row & not_last
                                qk_frag[r, s] = T.if_then_else(valid, qk_frag[r, s], -T.infinity(accum_dtype))
                        else:
                            for r, s in T.Parallel(M_G, step_S):
                                q_idx = r // G
                                valid_row = (ownership & (1 << q_idx)) != 0
                                qk_frag[r, s] = T.if_then_else(valid_row, qk_frag[r, s], -T.infinity(accum_dtype))

                        for r, s in T.Parallel(M_G, step_S):
                            P_frag[r, s] = T.exp2(qk_frag[r, s] * scale_log2 - lse_vec[r] * log2e)

                        # [Optimization] Re-calc Z = dO @ V.T (unweighted)
                        T.clear(dP_frag)
                        T.gemm(dO_shared, V_shared, dP_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        # Calc dS
                        # dS = P * (W * Z - delta)
                        #    = P * (W * dP_frag - delta)
                        for r, s in T.Parallel(M_G, step_S):
                            dS_frag[r, s] = sm_scale * (P_frag[r, s] * (chunk_weights[i, r] * dP_frag[r, s] - delta_rows[r]))
                        
                        T.copy(dS_frag, dS_shared)
                        
                        # Calc dQ (Accumulate locally)
                        T.clear(dQ_tmp)
                        T.gemm(dS_shared, K_shared, dQ_tmp, policy=T.GemmWarpPolicy.FullRow)
                        for r, k in T.Parallel(M_G, BK):
                            dQ_accum[r, k] += dQ_tmp[r, k]

                        # Calc dK & Atomic Add
                        T.clear(dK_tmp)
                        T.gemm(dS_shared, Q_shared, dK_tmp, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                        
                        T.copy(dK_tmp, K_shared)
                        
                        for s, k in T.Parallel(step_S, BK):
                            T.atomic_add(DK[i_v, i_b, i_kv + s, i_h, k], K_shared[s, k])

            # === Write back dQ (Non-atomic) ===
            # Reuse Q_shared buffer
            T.copy(dQ_accum, Q_shared)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * G
                    T.copy(Q_shared[h_start:h_start + G, :], DQ[i_v, i_b, tq, i_h * G:(i_h + 1) * G, :])

    return hsa_bwd_dqkv_block_M_inverse







class _hsa_block_M_attention_inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, weights, indices, block_size, sm_scale, block_M, mask_last_token):
        # Forward pass is the same
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert weights.is_contiguous() and indices.is_contiguous()

        B, L, HQ, D = q.shape
        L_kv = k.shape[1]
        H = k.shape[2]
        S = indices.shape[-1]
        G = HQ // H

        assert HQ % H == 0 and L_kv % block_size == 0
        if sm_scale is None: sm_scale = 1.0 / math.sqrt(D)

        fwd_kernel = hierarchical_sparse_attention_block_M(
            batch=B, heads=HQ, q_len=L, kv_len=L_kv, head_dim=D,
            block_size=block_size, groups=G, selected_blocks=S,
            scale=sm_scale, block_M=block_M,
            mask_last_token=mask_last_token,
        )
        o, lse = fwd_kernel(q, k, v, weights, indices)

        ctx.save_for_backward(q, k, v, lse, weights, indices)
        ctx.mask_last_token = mask_last_token
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.block_M = block_M
        ctx.G = G
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, lse, weights, indices = ctx.saved_tensors
        mask_last_token = ctx.mask_last_token
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        block_M = ctx.block_M
        G = ctx.G

        B, L, HQ, D = q.shape
        L_kv = k.shape[1]
        H = k.shape[2]
        S = indices.shape[-1]
        
        bwd_kernel = hierarchical_sparse_attention_bwd_dqkv_block_M_inverse(
            batch=B, heads=HQ, q_len=L, kv_len=L_kv, head_dim=D,
            block_size=block_size, groups=G, selected_blocks=S,
            scale=sm_scale, block_M=block_M,
            mask_last_token=mask_last_token,
        )

        BV = min(128, tilelang.next_power_of_2(D))
        NV = tilelang.cdiv(D, BV)
        
        dq = torch.zeros((NV, B, L, HQ, D), dtype=q.dtype, device=q.device)
        dk = torch.zeros((NV, B, L_kv, H, D), dtype=torch.float32, device=k.device)
        dv = torch.zeros((B, L_kv, H, D), dtype=torch.float32, device=v.device)
        dw = torch.zeros_like(weights, dtype=torch.float32)

        # Note: The new kernel expects indices, not block_mask
        bwd_kernel(q, k, v, weights, lse, do, indices, dq, dk, dv, dw)
        
        dq = dq.sum(0)
        dk = dk.sum(0)

        return dq, dk, dv, dw, None, None, None, None, None


def HSA_block_M_inverse(q, k, v, weights, indices, block_size=32, sm_scale=None, block_M=0, mask_last_token=False):
    return _hsa_block_M_attention_inverse.apply(q, k, v, weights, indices, block_size, sm_scale, block_M, mask_last_token)




def main_block_M_correctness():
    """
    检验 HSA_pair 封装类的前向和反向传播数值正确性
    与 hsa_torch_ref 进行对比
    """
    import math
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    from ops.hsa_fwd_bwd_triton import HSA
    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 4, 128
    dtype = torch.bfloat16
    device = "cuda"
    block_M=4
    mask_last_token=True
    G = HQ // H
    scale = 1.0 / math.sqrt(D)
    
    print(f"Correctness Config: Batch={B}, SeqLen={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, G={G}, S={S}, BlockSize={block_size}")
    
    # ---------- 生成测试数据 ----------
    torch.manual_seed(42)
    
    # 创建 block_indices（先生成 index 再生成 weight，使得我们可以在构造 W 时直接屏蔽非法位置）
    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.int32, device=device)
    num_blocks = SEQ_LEN // block_size
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                max_blocks = min(t // block_size + 1, num_blocks)
                if max_blocks > 0:
                    num_select = min(S, max_blocks)
                    selected = torch.randperm(max_blocks, device=device)[:num_select]
                    block_indices[b, t, h, :num_select] = selected

    # pair fwd kernel要求必须传入排好序的block_indices
    block_indices = block_indices.sort(-1)[0]
    
    # 替换block_indices中的SEQ_LEN值为-1（表示无效）
    block_indices[block_indices == SEQ_LEN] = -1

    # 创建 requires_grad=True 的输入（这些是 leaf tensors）
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成权重：仅在合法块上有非零概率，非法位置直接被置为非常小的 logit（softmax 后约为 0）
    # 这样 W 就天然在非法块上为 0，且仍为 leaf tensor（requires_grad=True）
    logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, float('-inf'))  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1)  # leaf tensor，非法位近似为 0
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)
    
    
    # 用于反向传播的梯度
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ========== 测试前向传播 ==========
    
    # HSA_pair 前向
    O_hsa = HSA_block_M_inverse(Q, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
    
    # Torch reference 前向
    O_ref = hsa_torch_ref(
        Q.float().detach(), 
        K.float().detach(), 
        V.float().detach(), 
        W.detach(), 
        block_indices,
        chunk_size=block_size, 
        sm_scale=scale, 
        block_q=1,
        mask_last_token=mask_last_token
    )

    print("[Tilelang HSA_block_M] vs [Torch Reference]:")
    # 比较前向输出
    fwd_max_diff = (O_hsa.float() - O_ref.float()).abs().max().item()
    print(f"前向最大误差: {fwd_max_diff:.6e}")


    # ========== 测试反向传播 ==========
    
    # ====== 先计算 Torch reference 反向 ======
    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None
    
    O_ref_bwd= hsa_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices,
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=mask_last_token
    )
    O_ref_bwd.backward(grad_output.float())
    
    # 保存 reference 梯度并清空
    DQ_ref = Q.grad.clone()
    DK_ref = K.grad.clone()
    DV_ref = V.grad.clone()
    DW_ref = W.grad.clone()
    
    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None
    
    # ====== 再计算 HSA_block_M 反向 ======
    O_hsa_bwd = HSA_block_M_inverse(Q, K, V, W, block_indices, 
                         block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
    O_hsa_bwd.backward(grad_output)
    
    # 获取 HSA_block_M 梯度
    DQ_hsa = Q.grad.clone()
    DK_hsa = K.grad.clone()
    DV_hsa = V.grad.clone()
    DW_hsa = W.grad.clone()
    
    # 比较梯度
    def compute_grad_diff(grad_hsa, grad_ref, name):
        if grad_hsa is None or grad_ref is None:
            print(f"{name}: 梯度为None")
            return None
        diff = (grad_hsa.float() - grad_ref.float()).abs()
        max_diff = diff.max().item()
        print(f"{name}最大误差: {max_diff:.6e}")
        # 打印几个位置的梯度检查
        # sample_indices = [(0, SEQ_LEN//2-3, 0, 0)]
        # for idx in sample_indices:
        #     hsa_val = grad_hsa[idx].float().item()
        #     ref_val = grad_ref[idx].float().item()
        #     val_diff = abs(hsa_val - ref_val)
        #     print(f"  {name}梯度检查 at {idx}: HSA={hsa_val:.6e}, Ref={ref_val:.6e}, Diff={val_diff:.6e}")
    dq_diff = compute_grad_diff(DQ_hsa, DQ_ref, "DQ")
    dk_diff = compute_grad_diff(DK_hsa, DK_ref, "DK")
    dv_diff = compute_grad_diff(DV_hsa, DV_ref, "DV")
    dw_diff = compute_grad_diff(DW_hsa, DW_ref, "DW")
    




def main_block_M_latency():
    """
    对比 tilelang HSA_block_M 和 triton HSA 的 FWD、BWD、以及综合 (FWD+BWD) 延迟
    """
    import torch
    import torch.nn.functional as F
    import time
    import math
    from einops import rearrange
    from ops.hsa_fwd_bwd_triton import HSA

    # ---------- 配置参数 ----------
    # B, SEQ_LEN, H, HQ, D, S, block_size = 128, 1024*4, 1, 8, 128, 8, 64
    B, SEQ_LEN, H, HQ, D, S, block_size = 32, 1024*4, 1, 8, 128, 8, 64
    block_M=4
    mask_last_token=True
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    G_padded = HQ_padded // H
    need_padding = (HQ_padded > HQ)
    scale = 1.0 / math.sqrt(D)

    print(f"Latency Config: B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block={block_size}, G={G}, HQ_padded={HQ_padded}, block_M={block_M}, mask_last_token={mask_last_token}")

    ### 生成成对重叠的 block_indices ###
    overlap_ratio = 0.8  # 可以调整重叠度
    block_indices = build_block_indices_block_M(
    B=B,
    SEQ_LEN=SEQ_LEN,
    H=H,
    S=S,
    block_size=block_size,
    overlap_ratio=overlap_ratio,
    block_M=block_M,
    device=device,
)
    print("Overlap_ratio:", overlap_ratio)

    # 创建 requires_grad=True 的输入（这些是 leaf tensors）
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成权重：仅在合法块上有非零概率，非法位置直接被置为非常小的 logit（softmax 后约为 0）
    # 这样 W 就天然在非法块上为 0，且仍为 leaf tensor（requires_grad=True）
    logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=torch.bfloat16, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, float('-inf'))  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1)  # leaf tensor，非法位近似为 0
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)
    

    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ---------- Triton block_indices ----------
    block_indices_triton = block_indices.clone()
    num_blocks = SEQ_LEN // block_size
    safe_block = num_blocks - 1
    block_indices_triton[block_indices_triton < 0] = safe_block

    # ---------- Triton padding ----------
    if need_padding:
        pad_heads = HQ_padded - HQ
        Q_triton = torch.cat(
            [Q, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2
        ).detach().clone().requires_grad_(True)
        grad_output_triton = torch.cat(
            [grad_output, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2
        )
    else:
        Q_triton = Q.detach().clone().requires_grad_(True)
        grad_output_triton = grad_output

    # ---------- TileLang ----------
    Q_tile = Q.detach().clone().requires_grad_(True)
    grad_output_tile = grad_output

    num_warmup = 50
    num_iters = 100

    # =========================================================
    # Helper function for timing
    # =========================================================
    def measure_time(func, *args, **kwargs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters  # ms per iter
    

    # =========================================================
    # Triton 部分
    # =========================================================
    # for _ in range(num_warmup):
    #     O = HSA(Q_triton, K, V, W, block_indices_triton, sm_n=0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
    #     O.backward(grad_output_triton)

    # # FWD only
    # def triton_fwd():
    #     with torch.no_grad():
    #         O = HSA(Q_triton, K, V, W, block_indices_triton, sm_n=0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
    # triton_fwd_ms = measure_time(triton_fwd)

    # # FWD+BWD
    # def triton_fwd_bwd():
    #     Q_triton.grad = K.grad = V.grad = W.grad = None
    #     O = HSA(Q_triton, K, V, W, block_indices_triton, sm_n=0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
    #     O.backward(grad_output_triton)
    # triton_total_ms = measure_time(triton_fwd_bwd)

    # triton_bwd_ms = triton_total_ms - triton_fwd_ms

    # =========================================================
    # TileLang 部分
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_block_M_inverse(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_inverse(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_inverse(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # 输出结果
    # =========================================================
    # print(f"[Triton]   FWD: {triton_fwd_ms:.3f} ms | BWD: {triton_bwd_ms:.3f} ms | Total: {triton_total_ms:.3f} ms")
    print(f"[TileLang] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")
    print()





    

if __name__ == "__main__":
    ### 验证 HSA_block_M (TileLang) 的正确性
    main_block_M_correctness()
    
    ### 对比 HSA_block_M (TileLang) 与 Triton HSA 的延迟
    main_block_M_latency()
    