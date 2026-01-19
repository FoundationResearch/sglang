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



# def make_dq_layout_hsa(dQ):

#     NV, B, L, HQ, D = dQ.shape
#     return T.Layout(dQ.shape,
#     lambda nv, b, l, h, d:   [nv,b,l, h//8, d//16, (d%16)//2, (h%8), (d%2)]
#  )

def make_dq_layout_hsa(DQ):
    # DQ.shape = [NV, batch, q_len, heads, head_dim]
    G = 8  # group size = tile size，完美匹配！
    return T.Layout(DQ.shape,
        lambda nv, b, l, h, d: [
            nv,
            b,
            l,
            h // 8,                           # tile 在 h 维的索引
            d // 8,                           # tile 在 d 维的索引
            d % 2,                            # elem_id (每个线程 2 个元素)
            4 * (h % 8) + (d % 8) // 2        # lane_id
        ])

@tilelang.jit(
    out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def hsa_bwd_postprocess(nv, batch, q_len, heads, head_dim):
    shape = [nv, batch, q_len, heads, head_dim]
    accum_dtype = "float"
    dtype = "bfloat16"
    blk = 64

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
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_block_M(batch, heads, q_len, kv_len, head_dim,
                                          scale=None, block_size=64, groups=16,
                                          selected_blocks=16, num_weights=None, block_M = 0, mask_last_token=False, dtype = "bfloat16", accum_dtype = "float", num_threads = 128):
    
    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504
    else:
        scale = scale * 1.44269504
    # 允许 num_weights (输入的 weights 张量的最后一维) 大于 selected_blocks (K)
    if num_weights is None:
        num_weights = selected_blocks
    head_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    kv_shape = [batch, kv_len, head_kv, head_dim]
    # weight_shape = [batch, q_len, heads, selected_blocks]
    weight_shape = [batch, q_len, heads, num_weights]
    block_indices_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    # dtype = "bfloat16"
    # accum_dtype = "float"
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(head_dim))

    NV = tilelang.cdiv(head_dim, block_T)
    assert tilelang.cdiv(head_dim, block_T) == 1, "The key dimension can not be larger than 256"

    # M: 每个线程块处理的query 数量，目标是让 M * G >= 16
    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, groups)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)
    # M=4
    print("Using M =", M, "for fwd_block_M kernel")
    M_G = M * groups  # 一次 GEMM 处理的总 head 数

    S = selected_blocks
    BS = block_S
    BK = BV = block_T
    num_stages = 0
    # num_threads = 128

    @T.prim_func
    def hsa_block_M(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(tilelang.cdiv(q_len, M), NV, batch * head_kv, threads=num_threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([M_G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([M_G, BV], dtype)

            acc_s = T.alloc_fragment([M_G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([M_G, BS], dtype)
            acc_o = T.alloc_fragment([M_G, BV], accum_dtype)
            
            P_shared = T.alloc_shared([M_G, BS], dtype)
            

            acc_s_tmp = T.alloc_fragment([groups, BS], accum_dtype)
            # scores_max = T.alloc_fragment([groups], accum_dtype)
            # scores_sum = T.alloc_fragment([groups], accum_dtype)
            scores_max = T.alloc_fragment([M_G], accum_dtype)
            scores_sum = T.alloc_fragment([M_G], accum_dtype)

            merged_indices = T.alloc_shared([S * M], block_indices_dtype)
            block_ownership = T.alloc_shared([S * M], "int32") # Bitmask
            merged_len = T.alloc_shared([1], "int32")
            chunk_weights = T.alloc_shared([S * M, M_G], dtype)
            

            i_t_base_idx, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv
            base_t = i_t_base_idx * M

            T.fill(Q_shared, 0)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    T.copy(Q[i_b, tq, i_h * groups:(i_h + 1) * groups, :],
                           Q_shared[q_idx * groups:(q_idx + 1) * groups, :])

            T.fill(acc_o, 0)
            T.fill(merged_indices, -1)
            T.fill(block_ownership, 0)
            T.fill(chunk_weights, 0)
            T.fill(merged_len, 0)
            
            # # 连续加载 W 到 shared memory备用
            # W_local_shared = T.alloc_shared([M, S], dtype)
            # T.fill(W_local_shared, 0.0)
            # for q_idx, s_idx in T.Parallel(M, S):
            #     tq = base_t + q_idx
            #     if tq < q_len:
            #         W_local_shared[q_idx, s_idx] = W[i_b, tq, i_h, s_idx]
            
            W_local_shared = T.alloc_shared([M_G, S], dtype)
            T.fill(W_local_shared, 0.0)
            # 并行加载所有 Head 的权重
            for i, s_idx in T.Parallel(M_G, S):
                q_idx = i // groups
                g = i % groups
                tq = base_t + q_idx
                if tq < q_len:
                    # W 的 shape 是 [batch, q_len, heads, selected_blocks]
                    # 当前 head index = i_h * groups + g
                    W_local_shared[i, s_idx] = W[i_b, tq, i_h * groups + g, s_idx]

            if T.get_thread_binding() == 0:
                valid_lens = T.alloc_fragment([M], "int32")
                pointers = T.alloc_fragment([M], "int32")
                k = T.alloc_var("int32")
                cur_val = T.alloc_fragment([M], "int32")
                
                # 记录每个query的有效block数量
                for q_idx in T.Parallel(M):
                    tq = base_t + q_idx
                    valid_lens[q_idx] = 0
                    pointers[q_idx] = 0
                    if tq < q_len:
                        for j in T.serial(S):
                            if BlockIndices[i_b, tq, i_h, j] >= 0:
                                valid_lens[q_idx] = valid_lens[q_idx] + 1
                            # else:
                            #     T.loop_break()
                    
                # M 路归并
                k = 0
                ownership_mask = T.alloc_var("int32")
                for _ in T.serial(S * M):
                    min_val = T.alloc_var("int32")
                    min_val = 2147483647  # INT_MAX
                    has_valid = T.alloc_var("int32")
                    has_valid = 0
                    
                    # 找到所有query当前还没处理的最小的块索引
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

                    merged_indices[k] = min_val
                    ownership_mask = 0
                    # for q_idx in T.serial(M):
                    for q_idx in T.unroll(M):
                        tq = base_t + q_idx
                        if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
                            # pointers[q_idx] 此时就是 W 在 S 维的索引
                            s_idx = pointers[q_idx]
                            val_q = cur_val[q_idx]
                            if val_q == min_val:
                                ownership_mask = ownership_mask | (1 << q_idx)
                                # [修改] 记录合并后第k个块对应于第q_idx个query下所有head的权重
                                # for g in T.serial(groups):
                                for g in T.unroll(groups):
                                    chunk_weights[k, q_idx * groups + g] = W_local_shared[q_idx * groups + g, s_idx]
                                pointers[q_idx] = pointers[q_idx] + 1
                    # 记录合并后的第k个块属于哪些query，用bitmask表示：0b0101表示属于第0和第2个query
                    block_ownership[k] = ownership_mask
                    k = k + 1
                    
                # 记录归并后的总块数
                merged_len[0] = k

            T.sync_threads()

            merged_len_local = T.alloc_var("int32")
            merged_len_local = merged_len[0]
            h_start = T.alloc_var("int32")

            for i in T.Pipelined(merged_len_local, num_stages=num_stages):
                blk_idx = merged_indices[i]
                i_s = blk_idx * BS
                ownership = block_ownership[i]

                if (blk_idx >= 0):
                    T.copy(K[i_b, i_s:i_s + BS, i_h, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    
                    # T.copy(acc_s, P_shared)
                    # for q_idx in T.serial(M):
                    #     if (ownership & (1 << q_idx)) != 0:
                    #         h_start = q_idx * groups

                    #         for g, s in T.Parallel(groups, BS):
                    #             acc_s_tmp[g, s] = T.if_then_else(s == BS - 1 and enable_last_token_mask, -T.infinity(accum_dtype),
                    #                                              P_shared[h_start + g, s]) 

                    #         T.fill(scores_max, -T.infinity(accum_dtype))
                    #         T.reduce_max(acc_s_tmp, scores_max, dim=1, clear=True)
                    #         for g, s in T.Parallel(groups, BS):
                    #             acc_s_tmp[g, s] = T.exp2(acc_s_tmp[g, s] * scale - scores_max[g] * scale)
                    #         T.reduce_sum(acc_s_tmp, scores_sum, dim=1, clear=True)
                            
                    #         for g, s in T.Parallel(groups, BS):
                    #             weight_q_g = chunk_weights[i, h_start + g]
                    #             acc_s_tmp[g, s] = weight_q_g * acc_s_tmp[g, s] / scores_sum[g]
                            
                    #         for g, s in T.Parallel(groups, BS):
                    #             P_shared[h_start + g, s] = acc_s_tmp[g, s]
                    #     else:
                    #         h_start = q_idx * groups
                    #         for g, s in T.Parallel(groups, BS):
                    #             P_shared[h_start + g, s] = 0.0
                                

                    # T.copy(P_shared, acc_s_cast)
                    # T.copy(V[i_b, i_s:i_s + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
                    # T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    
                    # 不再使用 T.serial(M)，而是直接并行处理所有 M*G 行
                    for r, c in T.Parallel(M_G, BS):
                        q_idx = r // groups
                        # # 检查当前行对应的 Query 是否拥有这个 Block
                        # is_valid_q = (ownership & (1 << q_idx)) != 0
                        # # 检查 Last Token Mask
                        # is_last = (c == BS - 1) and enable_last_token_mask
                        
                        # if not is_valid_q or is_last:
                        #     acc_s[r, c] = -T.infinity(accum_dtype)
                        acc_s[r, c] = T.if_then_else(  
                                                    ((ownership & (1 << q_idx)) == 0) or ((c == BS - 1) and enable_last_token_mask),  
                                                    -T.infinity(accum_dtype),  
                                                    acc_s[r, c]  
                                                )

                    # 4. Compute Softmax (Max -> Exp -> Sum -> Div)
                    # 4.1 Reduce Max
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    
                    # 4.2 Exp & Apply Scale
                    for r, c in T.Parallel(M_G, BS):
                        q_idx = r // groups
                        # is_valid_q = (ownership & (1 << q_idx)) != 0
                        
                        # # [修复] 只有有效的 Query 才进行 exp 计算，避免 -inf - (-inf) 产生 NaN
                        # if is_valid_q:
                        #     # 注意：如果 scores_max 仍为 -inf (例如 block_size=1 且 mask_last_token=True)，
                        #     # 这里仍可能出问题，但通常 block_size > 1。
                        #     # 更安全的写法是检查 scores_max[r] > -inf
                        #     acc_s[r, c] = T.exp2(acc_s[r, c] * scale - scores_max[r] * scale)
                        # else:
                        #     acc_s[r, c] = 0.0
                        acc_s[r, c] = T.if_then_else(
                                                    (ownership & (1 << q_idx)) != 0,
                                                    T.exp2(acc_s[r, c] * scale - scores_max[r] * scale),
                                                    0.0
                                                )
                    
                    # 4.3 Reduce Sum
                    T.fill(scores_sum, 0.0)
                    T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)
                    
                    # 4.4 Normalize & Apply Weights
                    for r, c in T.Parallel(M_G, BS):
                        q_idx = r // groups
                        # is_valid_q = (ownership & (1 << q_idx)) != 0
                        
                        # # [修复] 同样只对有效行进行归一化
                        # if is_valid_q:
                        #     # 获取当前 Head 的权重 (chunk_weights 形状为 [S*M, M_G])
                        #     # i 是当前处理的 merged block index
                        #     # weight = T.cast(chunk_weights[i, r], accum_dtype)
                        #     weight = chunk_weights[i, r]
                        #     # P = exp(x) * weight / sum
                        #     acc_s[r, c] = acc_s[r, c] * weight / scores_sum[r]
                        # else:
                        #     acc_s[r, c] = 0.0
                        acc_s[r, c] = T.if_then_else(
                                                    (ownership & (1 << q_idx)) != 0,
                                                    acc_s[r, c] * chunk_weights[i, r] / scores_sum[r],
                                                    0.0
                                                )

                    # 5. Store P to Shared Memory (仅一次写操作)
                    # P_shared 用于下一次 GEMM (P * V)
                    T.copy(acc_s, P_shared)

                    # 6. Load V & Compute O
                    T.copy(V[i_b, i_s:i_s + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
                    T.gemm(P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)
            
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * groups
                    for g, v in T.Parallel(groups, BV):
                        Output[i_b, tq, i_h * groups + g, i_v * BV + v] = O_shared[h_start + g, v]

    return hsa_block_M







@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_bwd_dqkv_block_M(
    batch, heads, q_len, kv_len, head_dim,
    scale=None, block_size=64, groups=16, selected_blocks=16, num_weights=None,
    block_M = 0, mask_last_token=False, dtype="bfloat16", accum_dtype="float", num_threads = 256
):
    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True
        
    if scale is None:
        sm_scale = (1.0 / head_dim)**0.5
    else:
        sm_scale = scale
    scale_log2 = sm_scale * 1.44269504

    B = batch
    BS = block_size
    G = groups
    Vdim = head_dim
    Kdim = head_dim
    BK = tilelang.next_power_of_2(Kdim)
    BV = min(128, tilelang.next_power_of_2(head_dim))
    NS_kv = tilelang.cdiv(kv_len, BS)
    NV = tilelang.cdiv(Vdim, BV)
    S = selected_blocks
    # 允许 num_weights (输入的 weights 张量的最后一维) 大于 selected_blocks (K)
    if num_weights is None:
        num_weights = S

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    do_shape = [batch, q_len, heads, head_dim]

    dq_shape = [NV, batch, q_len, heads, head_dim]
    dk_shape = [NV, batch, kv_len, heads_kv, head_dim]
    dv_shape = [batch, kv_len, heads_kv, head_dim]
    # weight_shape = [batch, q_len, heads, selected_blocks]
    weight_shape = [batch, q_len, heads, num_weights]
    # dw_shape = [batch, q_len, heads, selected_blocks]
    dw_shape = [batch, q_len, heads, num_weights]
    block_mask_shape = [batch, q_len, heads_kv, NS_kv]

    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, G)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)   
    # M=8  
    print("Using M =", M, "for bwd_block_M kernel")
    M_G = M * G
    NP = tilelang.cdiv(q_len, M)

    # num_threads = 256
    num_stages = 0

    @T.prim_func
    def hsa_bwd_dqkv_block_M(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        W: T.Tensor(weight_shape, dtype),
        DO: T.Tensor(do_shape, dtype),
        DQ: T.Tensor(dq_shape, accum_dtype),
        DK: T.Tensor(dk_shape, dtype),
        DV: T.Tensor(dv_shape, dtype),
        DW: T.Tensor(dw_shape, dtype),
        BlockMask: T.Tensor(block_mask_shape, "int32"),
    ):
        with T.Kernel(NV, NS_kv, B * heads_kv, threads=num_threads) as (i_v, i_s, i_bh):
            i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
            i_s_global = i_s * BS

            # === Shared Memory Optimization ===
            S_buf = T.alloc_shared([M_G, BS], dtype)
            dO_buf = T.alloc_shared([M_G, BV], dtype)
            Q_buf = T.alloc_shared([M_G, BK], dtype)
            
            logits_shared = S_buf
            P_shared = S_buf
            dS_shared = S_buf
            
            dO_shared = dO_buf
            dO_weighted_shared = dO_buf
            
            Q_shared = Q_buf

            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            
            dK_shared = T.alloc_shared([BS, BK], dtype)
            dV_shared = T.alloc_shared([BS, BV], dtype)

            qk_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            dV_PdO_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            dS_frag = T.alloc_fragment([M_G, BS], dtype)
            dV_accum = T.alloc_fragment([BS, BV], accum_dtype)
            dK_accum = T.alloc_fragment([BS, BK], accum_dtype)
            dQ_local = T.alloc_fragment([M_G, BK], accum_dtype)
            delta_rows = T.alloc_fragment([M_G], accum_dtype)
            
            acc_s_tmp = T.alloc_fragment([M_G, BS], accum_dtype)
            scores_max = T.alloc_fragment([M_G], accum_dtype)
            scores_sum = T.alloc_fragment([M_G], accum_dtype)
            
            has_q_shared = T.alloc_shared([M], "int32")

            dw_row_sum_frag = T.alloc_fragment([M_G], accum_dtype)
            dw_row_sum_shared = T.alloc_shared([M_G], accum_dtype)
            
            W_local = T.alloc_shared([M_G], dtype)
            
            pos = T.alloc_shared([M], "int32")
            has_q = T.alloc_shared([M], "int32")
            any_valid = T.alloc_var("int32")
            
            T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :], K_shared)
            T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
            T.fill(dK_accum, 0)
            T.fill(dV_accum, 0)

            T.annotate_layout({
                DQ: make_dq_layout_hsa(DQ),
            })

            for ip in T.Pipelined(NP, num_stages=num_stages):
                base_t = ip * M

                T.fill(pos, -1)
                T.fill(has_q, 0)
                
                any_valid = 0
                for qi1 in T.serial(M):
                    tq = base_t + qi1
                    if tq < q_len:
                        pos[qi1] = BlockMask[i_b, tq, i_h, i_s]
                        if pos[qi1] != -1:
                            has_q[qi1] = 1
                            any_valid = 1
                
                
                if any_valid != 0:
                    T.fill(Q_shared, 0)
                    T.fill(dO_shared, 0)
                    for qi3 in T.serial(M):
                        tq = base_t + qi3
                        if tq < q_len and has_q[qi3] == 1:
                            h_start = qi3 * G
                            T.copy(Q[i_b, tq, i_h * G:(i_h + 1) * G, :], Q_shared[h_start:h_start + G, :])
                            T.copy(DO[i_b, tq, i_h * G:(i_h + 1) * G, i_v * BV:(i_v + 1) * BV],
                                dO_shared[h_start:h_start + G, :])
                    
                    # T.copy(has_q, has_q_shared)

                    T.clear(qk_frag)
                    T.gemm(Q_shared, K_shared, qk_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(qk_frag, logits_shared)
                    
                    # 1. Load data from Shared Memory to Fragment
                    for i, s in T.Parallel(M_G, BS):
                        acc_s_tmp[i, s] = logits_shared[i, s]

                    # 2. Apply Masking using IF statements (avoids T.if_then_else expression bug)
                    for i, s in T.Parallel(M_G, BS):
                        q_idx = i // G
                        
                        # Use control flow to overwrite with -inf
                        if has_q[q_idx] == 0:
                            acc_s_tmp[i, s] = -T.infinity(accum_dtype)
                            
                        # Last token masking
                        if enable_last_token_mask:
                            if s == BS - 1:
                                acc_s_tmp[i, s] = -T.infinity(accum_dtype)

                    # 3. Reduction Max
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s_tmp, scores_max, dim=1, clear=True)

                    # 4. Exp with NaN Guard
                    for i, s in T.Parallel(M_G, BS):
                        q_idx = i // G
                        # 将 has_q 检查和 -inf 检查合并为一个 if_then_else
                        acc_s_tmp[i, s] = T.if_then_else(
                            has_q[q_idx] == 1 and scores_max[i] > -T.infinity(accum_dtype),
                            T.exp2(acc_s_tmp[i, s] * scale_log2 - scores_max[i] * scale_log2),
                            0.0
                        )

                    
                    # 5. Sum
                    T.reduce_sum(acc_s_tmp, scores_sum, dim=1, clear=True)

                    # 6. Normalize and Write Back
                    for i, s in T.Parallel(M_G, BS):
                        # inv_sum = T.if_then_else(scores_sum[i] > 0, 1.0 / scores_sum[i], 0.0)
                        # P_shared[i, s] = acc_s_tmp[i, s] * inv_sum
                        acc_s_tmp[i, s] = T.if_then_else(scores_sum[i] > 0,
                                                        acc_s_tmp[i, s] / scores_sum[i],
                                                        0.0)
                    # === dV & dW Optimization ===
                    # 优化策略：
                    # 1. 提前加载 W
                    # 2. 计算 Z = dO @ V.T
                    # 3. dW = P * Z (避免了 P @ V 的 GEMM)
                    # 4. dS = P * (W * Z - delta) (复用 Z)

                    # [Step 1] Load W early
                    T.fill(W_local, 0.0)
                    for i in T.Parallel(M_G):
                        qi_w = i // G
                        g_local = i % G
                        tq = base_t + qi_w
                        if tq < q_len and has_q[qi_w] == 1 and pos[qi_w] != -1:
                            W_local[i] = W[i_b, tq, i_h * G + g_local, pos[qi_w]]

                    # [Step 2] Compute Z = dO @ V.T
                    # Store in dV_PdO_frag (accum_dtype)
                    T.clear(dV_PdO_frag)
                    T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # [Step 3] Compute dW = Sum(P * Z)
                    # Reuse qk_frag (accum_dtype) for P * Z
                    for r, s in T.Parallel(M_G, BS):
                        qk_frag[r, s] = acc_s_tmp[r, s] * dV_PdO_frag[r, s]
                    
                    T.reduce_sum(qk_frag, dw_row_sum_frag, dim=1, clear=True)
                    T.copy(dw_row_sum_frag, dw_row_sum_shared)
                    
                    # Write back dW
                    for i in T.Parallel(M * G):  
                        qi5 = i // G  
                        g_w = i % G  
                        if has_q[qi5] == 1:
                            h_start = qi5 * G  
                            tq = base_t + qi5  
                            if tq < q_len and pos[qi5] != -1:  
                                DW[i_b, tq, i_h * G + g_w, pos[qi5]] = dw_row_sum_shared[h_start + g_w]

                    # [Step 4] Compute dV
                    # Need dO_weighted = W * dO
                    for row_idx, v in T.Parallel(M_G, BV):
                        dO_weighted_shared[row_idx, v] = W_local[row_idx] * dO_shared[row_idx, v]
                    T.copy(acc_s_tmp, P_shared)
                    T.gemm(P_shared, dO_weighted_shared, dV_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    # [Step 5] Compute dS
                    # We need Y = W * Z = W * (dO @ V.T)
                    # Z is currently in dV_PdO_frag
                    for g_row, s in T.Parallel(M_G, BS):
                        dV_PdO_frag[g_row, s] = dV_PdO_frag[g_row, s] * W_local[g_row]
                    
                    # Now dV_PdO_frag holds Y.
                    # Compute P * Y for delta calculation. Reuse qk_frag again.
                    for g_row, s in T.Parallel(M_G, BS):
                        qk_frag[g_row, s] = P_shared[g_row, s] * dV_PdO_frag[g_row, s]
                        
                    T.reduce_sum(qk_frag, delta_rows, dim=1, clear=True)
                    
                    for g_row, s in T.Parallel(M_G, BS):
                        # dS = scale * (P * Y - P * delta)
                        #    = scale * (qk_frag - P * delta)
                        dS_frag[g_row, s] = sm_scale * (qk_frag[g_row, s] - P_shared[g_row, s] * delta_rows[g_row])
                    
                    T.copy(dS_frag, dS_shared)  
                    
                    T.gemm(dS_shared, Q_shared, dK_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    
                    T.clear(dQ_local)
                    T.gemm(dS_shared, K_shared, dQ_local, policy=T.GemmWarpPolicy.FullRow)
                    
                    # T.copy(dQ_local, dQ_shared)
                    T.copy(has_q, has_q_shared)
                    for g_row, k in T.Parallel(M_G, BK):
                        q_idx = g_row // G
                        g_local = g_row % G
                        tq = base_t + q_idx
                        if tq < q_len and has_q_shared[q_idx] == 1:
                            T.atomic_add(DQ[i_v, i_b, tq, i_h * G + g_local, k], dQ_local[g_row, k])

            T.copy(dK_accum, dK_shared)
            T.copy(dV_accum, dV_shared)
            T.copy(dK_shared, DK[i_v, i_b, i_s_global:i_s_global + BS, i_h, :])
            T.copy(dV_shared, DV[i_b, i_s_global:i_s_global + BS, i_h, i_v * BV:(i_v + 1) * BV])

    return hsa_bwd_dqkv_block_M


# class _hsa_block_M_attention(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q, k, v, weights, indices, block_size, sm_scale, block_M, mask_last_token):
#         assert q.is_contiguous()
#         assert k.is_contiguous()
#         assert v.is_contiguous()
#         assert weights.is_contiguous()
#         assert indices.is_contiguous()

#         B, L, HQ, D = q.shape
#         L_kv = k.shape[1]
#         H = k.shape[2]
#         S = indices.shape[-1]
#         G = HQ // H

#         assert HQ % H == 0, f"HQ={HQ} must be divisible by H={H}"
#         assert L_kv % block_size == 0, f"L_kv={L_kv} must be divisible by block_size={block_size}"

#         if sm_scale is None:
#             sm_scale = 1.0 / math.sqrt(D)

#         # 创建前向 kernel（block_M）
#         fwd_kernel = hierarchical_sparse_attention_block_M(
#             batch=B,
#             heads=HQ,
#             q_len=L,
#             kv_len=L_kv,
#             head_dim=D,
#             block_size=block_size,
#             groups=G,
#             selected_blocks=S,
#             scale=sm_scale,
#             block_M=block_M,
#             mask_last_token=mask_last_token,
#         )

#         # 执行前向
#         o = fwd_kernel(q, k, v, weights, indices)

#         # 保存用于反向的张量
#         ctx.save_for_backward(q, k, v, weights, indices)
#         ctx.block_size = block_size
#         ctx.sm_scale = sm_scale
#         ctx.block_M = block_M
#         ctx.mask_last_token = mask_last_token
#         ctx.B = B
#         ctx.L = L
#         ctx.HQ = HQ
#         ctx.H = H
#         ctx.D = D
#         ctx.S = S
#         ctx.G = G

#         return o

#     @staticmethod
#     def backward(ctx, do):
#         q, k, v, weights, indices = ctx.saved_tensors
#         block_size = ctx.block_size
#         sm_scale = ctx.sm_scale
#         block_M = ctx.block_M
#         mask_last_token = ctx.mask_last_token
#         B, L, HQ, H, D, S, G = ctx.B, ctx.L, ctx.HQ, ctx.H, ctx.D, ctx.S, ctx.G

#         L_kv = k.shape[1]
#         NS_kv = L_kv // block_size
#         NV = tilelang.cdiv(D, min(128, tilelang.next_power_of_2(D)))

#         build_mask = hsa_kernel_block_mask(
#             batch=B, heads=H, q_len=L, kv_len=L_kv,
#             selected_blocks=S, block_size=block_size
#         )
#         block_mask = torch.full((B, L, H, NS_kv), -1, dtype=torch.int32, device=q.device)
#         build_mask(indices, block_mask)
        
        
#         bwd_kernel = hierarchical_sparse_attention_bwd_dqkv_block_M(
#             batch=B,
#             heads=HQ,
#             q_len=L,
#             kv_len=L_kv,
#             head_dim=D,
#             block_size=block_size,
#             groups=G,
#             selected_blocks=S,
#             scale=sm_scale,
#             block_M=block_M,
#             mask_last_token=mask_last_token,
#         )

#         # 分配梯度缓冲
#         dq = torch.zeros((NV, B, L, HQ, D), dtype=torch.float32, device=q.device)
#         dk = torch.zeros((NV, B, L_kv, H, D), dtype=k.dtype, device=k.device)
#         dv = torch.zeros((B, L_kv, H, D), dtype=v.dtype, device=v.device)
#         dw = torch.zeros((B, L, HQ, S), dtype=weights.dtype, device=weights.device)

#         # 执行反向
#         bwd_kernel(q, k, v, weights, do, dq, dk, dv, dw, block_mask)

#         post_kernel = hsa_bwd_postprocess(NV, B, L, HQ, D)
#         dq = post_kernel(dq)

#         dq = dq.sum(0)
#         dk = dk.sum(0)

#         dq = dq.to(q.dtype)
#         dk = dk.to(k.dtype)
#         dv = dv.to(v.dtype)
#         dw = dw.to(weights.dtype)

#         return dq, dk, dv, dw, None, None, None, None, None


# def HSA_block_M_head(q, k, v, weights, indices, block_size=32, sm_scale=None, block_M=0, mask_last_token=False):
#     return _hsa_block_M_attention.apply(q, k, v, weights, indices, block_size, sm_scale, block_M, mask_last_token)
    
    


from liger_kernel.transformers.rope import liger_rotary_pos_emb
from ops.rope_tilelang_fp32 import rope_rotary_pos_emb
class _hsa_block_M_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, weights, indices,
        block_size, sm_scale, block_M, mask_last_token,dtype, accum_dtype, num_threads,
        enable_inverse_rope,
        cos, sin,
    ):
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert v.is_contiguous()
        assert weights.is_contiguous()
        assert indices.is_contiguous()
        
        B, L, HQ, D = q.shape
        L_kv = k.shape[1]
        H = k.shape[2]
        S = indices.shape[-1]
        G = HQ // H
        
        # 捕获 weights 的真实 shape，num_weights 可能大于 S
        num_weights = weights.shape[-1]

        if enable_inverse_rope:
            assert cos.shape[1] == q.shape[1], f"cos seq_len {cos.shape[1]} != q seq_len {q.shape[1]}"
            q_in, k_in = rope_rotary_pos_emb(q, k, cos, -sin)
        else:
            q_in, k_in = q, k
            
        # if enable_inverse_rope:
        #     assert cos.shape[1] == q.shape[1], f"cos seq_len {cos.shape[1]} != q seq_len {q.shape[1]}"
            
        #     orig_dtype = q.dtype
            
        #     q_t = q.transpose(1, 2).contiguous().float()
        #     k_t = k.transpose(1, 2).contiguous().float()
            
        #     q_rope, k_rope = liger_rotary_pos_emb(q_t, k_t, cos, -sin)
            
        #     q_in = q_rope.to(orig_dtype).transpose(1, 2).contiguous()
        #     k_in = k_rope.to(orig_dtype).transpose(1, 2).contiguous()
        # else:
        #     q_in, k_in = q, k

        assert HQ % H == 0, f"HQ={HQ} must be divisible by H={H}"
        assert L_kv % block_size == 0, f"L_kv={L_kv} must be divisible by block_size={block_size}"

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        # 创建前向 kernel（block_M）
        fwd_kernel = hierarchical_sparse_attention_block_M(
            batch=B,
            heads=HQ,
            q_len=L,
            kv_len=L_kv,
            head_dim=D,
            block_size=block_size,
            groups=G,
            selected_blocks=S,
            num_weights=num_weights,  # 传入真实的权重维度
            scale=sm_scale,
            block_M=block_M,
            mask_last_token=mask_last_token,
            dtype=dtype, accum_dtype=accum_dtype,num_threads=num_threads,
        )

        # 执行前向
        o = fwd_kernel(q_in, k_in, v, weights, indices)

        # 保存用于反向的张量
        ctx.save_for_backward(q, k, v, weights, indices, cos, sin)
        ctx.enable_inverse_rope = bool(enable_inverse_rope)
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.block_M = block_M
        ctx.mask_last_token = mask_last_token
        ctx.dtype = dtype
        ctx.accum_dtype = accum_dtype
        ctx.num_threads = num_threads
        ctx.B = B
        ctx.L = L
        ctx.HQ = HQ
        ctx.H = H
        ctx.D = D
        ctx.S = S
        ctx.G = G
        ctx.num_weights = num_weights # 保存 num_weights

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, weights, indices, cos, sin = ctx.saved_tensors
        enable_inverse_rope = ctx.enable_inverse_rope
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        block_M = ctx.block_M
        mask_last_token = ctx.mask_last_token
        dtype = ctx.dtype
        accum_dtype = ctx.accum_dtype
        num_threads = ctx.num_threads
        B, L, HQ, H, D, S, G = ctx.B, ctx.L, ctx.HQ, ctx.H, ctx.D, ctx.S, ctx.G
        num_weights = ctx.num_weights

        if enable_inverse_rope:
            q_in, k_in = rope_rotary_pos_emb(q, k, cos, -sin)
        else:
            q_in, k_in = q, k
        
        # if enable_inverse_rope:
        #     orig_dtype = q.dtype
        #     q_t = q.transpose(1, 2).contiguous().float()
        #     k_t = k.transpose(1, 2).contiguous().float()
        #     q_rope, k_rope = liger_rotary_pos_emb(q_t, k_t, cos, -sin)
        #     q_in = q_rope.to(orig_dtype).transpose(1, 2).contiguous()
        #     k_in = k_rope.to(orig_dtype).transpose(1, 2).contiguous()
        # else:
        #     q_in, k_in = q, k

        L_kv = k.shape[1]
        NS_kv = L_kv // block_size
        NV = tilelang.cdiv(D, min(128, tilelang.next_power_of_2(D)))

        build_mask = hsa_kernel_block_mask(
            batch=B, heads=H, q_len=L, kv_len=L_kv,
            selected_blocks=S, block_size=block_size
        )
        block_mask = torch.full((B, L, H, NS_kv), -1, dtype=torch.int32, device=q.device)
        build_mask(indices, block_mask)
        
        bwd_kernel = hierarchical_sparse_attention_bwd_dqkv_block_M(
            batch=B,
            heads=HQ,
            q_len=L,
            kv_len=L_kv,
            head_dim=D,
            block_size=block_size,
            groups=G,
            selected_blocks=S,
            num_weights=num_weights,  # 传入真实的权重维度
            scale=sm_scale,
            block_M=block_M,
            mask_last_token=mask_last_token,
            dtype=dtype, accum_dtype=accum_dtype, num_threads=num_threads,
        )

        # 分配梯度缓冲
        dq_in = torch.zeros((NV, B, L, HQ, D), dtype=torch.float32, device=q.device)
        dk_in = torch.zeros((NV, B, L_kv, H, D), dtype=k.dtype, device=k.device)
        dv = torch.zeros((B, L_kv, H, D), dtype=v.dtype, device=v.device)
        # dw = torch.zeros((B, L, HQ, S), dtype=weights.dtype, device=weights.device)
        dw = torch.zeros((B, L, HQ, num_weights), dtype=weights.dtype, device=weights.device)

        # 执行反向
        bwd_kernel(q_in, k_in, v, weights, do, dq_in, dk_in, dv, dw, block_mask)

        post_kernel = hsa_bwd_postprocess(NV, B, L, HQ, D)
        dq_in = post_kernel(dq_in)

        dq_in = dq_in.sum(0)
        dk_in = dk_in.sum(0)

        dq_in = dq_in.to(q.dtype)
        dk_in = dk_in.to(k.dtype)
        dv = dv.to(v.dtype)
        dw = dw.to(weights.dtype)

        if enable_inverse_rope:
            dq, dk = rope_rotary_pos_emb(dq_in, dk_in, cos, sin)
        else:
            dq, dk = dq_in, dk_in
        
        # if enable_inverse_rope:
        #     dq_t = dq_in.transpose(1, 2).contiguous().float()
        #     dk_t = dk_in.transpose(1, 2).contiguous().float()
            
        #     dq_rope, dk_rope = liger_rotary_pos_emb(dq_t, dk_t, cos, sin)
            
        #     dq = dq_rope.to(orig_dtype).transpose(1, 2).contiguous()
        #     dk = dk_rope.to(orig_dtype).transpose(1, 2).contiguous()
        # else:
        #     dq, dk = dq_in, dk_in

        return dq, dk, dv, dw, None, None, None, None, None, None, None, None, None, None, None


def HSA_block_M_head(
    q, k, v, weights, indices,
    block_size=32, sm_scale=None, block_M=0, mask_last_token=False, dtype="bfloat16", accum_dtype="float", num_threads=128,
    enable_inverse_rope: bool = False,
    cos=None, sin=None,
):
    if enable_inverse_rope and (cos is None or sin is None):
        raise ValueError("cos and sin cannot be None when enable_inverse_rope is True")
    return _hsa_block_M_attention.apply(
        q, k, v, weights, indices,
        block_size, sm_scale, block_M, mask_last_token, dtype, accum_dtype,num_threads,
        enable_inverse_rope,
        cos, sin,
    )




# from ops.topk import online_topk 

# def main_block_M_correctness():
#     """
#     检验 HSA_pair 封装类的前向和反向传播数值正确性
#     与 hsa_torch_ref 进行对比
#     使用 online_topk 生成真实的 indices 和 weights
#     """
#     import math
#     import torch
#     import torch.nn.functional as F
#     from einops import rearrange
    
#     # ---------- 配置参数 ----------
#     B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 8, 64
#     dtype = torch.bfloat16
#     device = "cuda"
#     block_M = 4
#     G = HQ // H
#     scale = 1.0 / math.sqrt(D)
    
#     print(f"Correctness Config: Batch={B}, SeqLen={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, G={G}, S={S}, BlockSize={block_size}")
    
#     # ---------- 生成测试数据 ----------
#     torch.manual_seed(42)
    
#     # 创建 requires_grad=True 的输入
#     Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
#     K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
#     V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
#     # [新增] 1. 提取 Landmarks (lmks)
#     # 规定：lmks 是 K 中每个 chunk 的最后一个 token
#     # K shape: [B, L, H, D] -> lmks shape: [B, num_chunks, H, D]
#     # 切片索引: block_size-1, 2*block_size-1, ...
#     # 注意：确保 SEQ_LEN 是 block_size 的倍数，否则最后一个 chunk 可能取不到末尾
#     assert SEQ_LEN % block_size == 0, "SEQ_LEN must be divisible by block_size for this test extraction"
#     lmks = K[:, block_size-1::block_size, :, :].detach().clone() # Detach to treat as fixed input for TopK
    
#     # [新增] 2. 构造 lse_swa
#     # Shape: [B, L, HQ] (或者 [B, L, H, G])
#     # 这里随机生成模拟值
#     lse_swa = torch.randn((B, SEQ_LEN, HQ), dtype=dtype, device=device)
    
#     # [新增] 3. 调用 Online TopK 生成 Indices 和 Scores
#     print("Running Online TopK to generate indices and weights...")
#     # 使用 'softmax_head' 策略
#     topk_indices, topk_scores = online_topk(
#         q=Q.detach(),       # Detach Q for generation
#         lmks=lmks, 
#         topk=S, 
#         selection_strategy='head', 
#         block_size=block_size, 
#         is_causal=True, 
#         lse_swa=lse_swa
#     )
    
#     # topk_indices: [B, L, H, S]
#     # topk_scores:  [B, L, HQ, S] (注意: online_softmax_topk_head 返回的是 [B, L, H, G, S] 或 [B, L, HQ, S])
#     # 假设 online_topk 内部处理了维度，返回 [B, L, HQ, S] 或者 [B, L, H, G, S]
#     # 如果是 [B, L, H, G, S]，我们需要 flatten 成 [B, L, HQ, S]
#     if topk_scores.dim() == 5:
#         topk_scores = rearrange(topk_scores, 'b l h g s -> b l (h g) s')
        
#     sorted_indices = topk_indices
#     sorted_scores = topk_scores
    
#     # 生成 W (Weights): 对 Scores 做 Softmax
#     # 注意：TopK 返回的 scores 通常是 logits，需要 softmax 归一化作为 Attention 权重
#     W_gen = F.softmax(sorted_scores.float(), dim=-1).to(dtype)
    
#     # [新增] 5. 准备 HSA Kernel 的输入
#     # 将生成的 indices 和 W detach 出来，作为 HSA Kernel 测试的固定输入
#     # 这样我们只测试 HSA Kernel 的正确性，而不测试 TopK 的梯度回传
#     block_indices = sorted_indices.int().detach()
#     W = W_gen.detach().clone().requires_grad_(True) # W 需要是 leaf tensor 以测试 dW
    
#     # 处理无效索引：online_topk 可能返回 -1 (或 0 且 score=-inf)
#     # 确保 block_indices 中的无效位置标记为 -1 (如果 topk 没填满)
#     # 这里假设 online_topk 返回 -1 表示无效。
#     # 另外，HSA Kernel 中 valid_mask = (block_indices != -1)
#     # 如果 topk 逻辑正确，causal mask 会导致某些位置 score 为 -inf，但 indices 可能是 0 或 -1
#     # 我们显式处理一下：如果 score 非常小，认为无效？
#     # 或者直接信任 online_topk 的 indices 输出。
#     # 为了安全，如果 indices < 0，确保 W 为 0 (softmax 后 -inf 会变成 0，所以 W 应该是对的)
    
#     # 用于反向传播的梯度
#     grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
#     # ========== 测试前向传播 ==========
    
#     # HSA_pair 前向
#     O_hsa = HSA_block_M_head(Q, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=True)
    
#     # Torch reference 前向
#     O_ref = hsa_torch_ref(
#         Q.float().detach(), 
#         K.float().detach(), 
#         V.float().detach(), 
#         W.detach(), 
#         block_indices,
#         chunk_size=block_size, 
#         sm_scale=scale, 
#         block_q=1,
#         mask_last_token=True
#     )

#     print("[Tilelang HSA_block_M] vs [Torch Reference]:")
#     # 比较前向输出
#     fwd_max_diff = (O_hsa.float() - O_ref.float()).abs().max().item()
#     print(f"前向最大误差: {fwd_max_diff:.6e}")

#     # ========== 测试反向传播 ==========
    
#     # ====== 先计算 Torch reference 反向 ======
#     Q.grad = None
#     K.grad = None
#     V.grad = None
#     W.grad = None
    
#     O_ref_bwd = hsa_torch_ref(
#         Q.float(), K.float(), V.float(), W.float(), block_indices,
#         chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=True
#     )
#     O_ref_bwd.backward(grad_output.float())
    
#     # 保存 reference 梯度并清空
#     DQ_ref = Q.grad.clone()
#     DK_ref = K.grad.clone()
#     DV_ref = V.grad.clone()
#     DW_ref = W.grad.clone()
    
#     Q.grad = None
#     K.grad = None
#     V.grad = None
#     W.grad = None
    
#     # ====== 再计算 HSA_block_M 反向 ======
#     O_hsa_bwd = HSA_block_M_head(Q, K, V, W, block_indices, 
#                          block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=True)
#     O_hsa_bwd.backward(grad_output)
    
#     # 获取 HSA_block_M 梯度
#     DQ_hsa = Q.grad.clone()
#     DK_hsa = K.grad.clone()
#     DV_hsa = V.grad.clone()
#     DW_hsa = W.grad.clone()
    
#     # 比较梯度
#     def compute_grad_diff(grad_hsa, grad_ref, name):
#         if grad_hsa is None or grad_ref is None:
#             print(f"{name}: 梯度为None")
#             return None
#         diff = (grad_hsa.float() - grad_ref.float()).abs()
#         max_diff = diff.max().item()
#         print(f"{name}最大误差: {max_diff:.6e}")
#         return max_diff
    
#     dq_diff = compute_grad_diff(DQ_hsa, DQ_ref, "DQ")
#     dk_diff = compute_grad_diff(DK_hsa, DK_ref, "DK")
#     dv_diff = compute_grad_diff(DV_hsa, DV_ref, "DV")
#     dw_diff = compute_grad_diff(DW_hsa, DW_ref, "DW")






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
    B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 4, 64
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
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)
    
    
    # 用于反向传播的梯度
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ========== 测试前向传播 ==========
    
    # HSA_pair 前向
    O_hsa = HSA_block_M_head(Q, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
    
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
    O_hsa_bwd = HSA_block_M_head(Q, K, V, W, block_indices, 
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
        sample_indices = [(0, SEQ_LEN//2-3, 0, 0)]
        for idx in sample_indices:
            hsa_val = grad_hsa[idx].float().item()
            ref_val = grad_ref[idx].float().item()
            val_diff = abs(hsa_val - ref_val)
            print(f"  {name}梯度检查 at {idx}: HSA={hsa_val:.6e}, Ref={ref_val:.6e}, Diff={val_diff:.6e}")
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
    B, SEQ_LEN, H, HQ, D, S, block_size = 16, 4096, 1, 8, 128, 8, 64
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
    W = F.softmax(logits, dim=-1) # leaf tensor，非法位近似为 0
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
        O = HSA_block_M_head(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_head(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_head(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # 输出结果
    # =========================================================
    # print(f"[Triton]   FWD: {triton_fwd_ms:.3f} ms | BWD: {triton_bwd_ms:.3f} ms | Total: {triton_total_ms:.3f} ms")
    print(f"[TileLang] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")
    print()


def main_rope_correctness():
    """
    验证 enable_inverse_rope 开启后的数值正确性
    逻辑：RoPE(q_nope) -> HSA(enable_inverse_rope=True) -> 内部还原为 q_nope -> 与 Ref(q_nope) 对比
    """
    import torch
    import torch.nn.functional as F
    import math

    # ---------- 配置参数 ----------
    B, L, H, HQ, D, S, block_size = 1, 512, 1, 8, 128, 4, 64
    dtype = torch.bfloat16
    device = "cuda"
    block_M = 4
    scale = 1.0 / math.sqrt(D)
    
    print(f"\nRoPE Correctness Test: B={B}, L={L}, HQ={HQ}, D={D}, S={S}, block_size={block_size}")

    # ---------- 准备数据 ----------
    torch.manual_seed(42)
    q_nope = torch.randn((B, L, HQ, D), dtype=dtype, device=device, requires_grad=True)
    k_nope = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成 cos/sin (Liger 格式: 1, L, D)
    angle = torch.randn((1, L, D), dtype=dtype, device=device)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    cos_f32 = cos.float()
    sin_f32 = sin.float()

    # 生成 block_indices 和 weights
    block_indices = torch.full((B, L, H, S), -1, dtype=torch.int32, device=device)
    for t in range(L):
        max_blocks = min(t // block_size + 1, L // block_size)
        if max_blocks > 0:
            num_select = min(S, max_blocks)
            block_indices[0, t, 0, :num_select] = torch.sort(torch.randperm(max_blocks, device=device)[:num_select])[0]
    
    weights = torch.randn((B, L, HQ, S), dtype=dtype, device=device)
    valid_mask = (block_indices != -1)
    weights_base = F.softmax(weights.masked_fill(~valid_mask, float('-inf')), dim=-1)
    weights_base = torch.nan_to_num(weights_base, 0.0)

    # ---------- 1. 构造输入的 q_rope, k_rope (使用 float32 计算) ----------
    q_t = q_nope.transpose(1, 2).contiguous().float()
    k_t = k_nope.transpose(1, 2).contiguous().float()
    
    q_rope_t, k_rope_t = liger_rotary_pos_emb(q_t, k_t, cos_f32, sin_f32)
    
    q_rope = q_rope_t.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    k_rope = k_rope_t.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)

    # ---------- 2. 准备两组独立的输入 ----------
    # HSA with inverse RoPE: 输入 q_rope, k_rope
    v_hsa = v.detach().clone().requires_grad_(True)
    weights_hsa = weights_base.detach().clone().requires_grad_(True)
    
    # Torch Ref: 输入 q_nope, k_nope
    q_nope_ref = q_nope.detach().clone().requires_grad_(True)
    k_nope_ref = k_nope.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    weights_ref = weights_base.detach().clone().requires_grad_(True)

    # ---------- 3. 前向测试 ----------
    
    # HSA 开启逆向 RoPE (内部会把 q_rope 变回 q_nope)
    o_hsa = HSA_block_M_head(
        q_rope, k_rope, v_hsa, weights_hsa, block_indices, 
        block_size=block_size, sm_scale=scale, block_M=block_M, 
        mask_last_token=True, enable_inverse_rope=True, cos=cos, sin=sin
    )

    # Torch Ref: 直接用 nope 数据
    o_ref = hsa_torch_ref(
        q_nope_ref.float(), k_nope_ref.float(), v_ref.float(), 
        weights_ref, block_indices, 
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=True
    )

    # 找到前向最大误差位置
    diff_fwd = (o_hsa.float() - o_ref.float()).abs()
    max_fwd_err = diff_fwd.max().item()
    idx_fwd = tuple(i[0].item() for i in torch.where(diff_fwd == max_fwd_err))
    
    print(f"\n[Forward] Max Diff: {max_fwd_err:.6e} at {idx_fwd}")
    print(f"  HSA Val: {o_hsa[idx_fwd].item():.10f}")
    print(f"  Ref Val: {o_ref[idx_fwd].item():.10f}")

    # ---------- 4. 反向测试 ----------
    grad_output = torch.randn_like(o_hsa)
    
    # HSA with inverse RoPE 反向
    o_hsa.backward(grad_output)
    dq_rope_hsa = q_rope.grad.clone()
    dk_rope_hsa = k_rope.grad.clone()
    dv_hsa = v_hsa.grad.clone()
    dw_hsa = weights_hsa.grad.clone()

    # Torch Ref 反向
    o_ref.backward(grad_output.float())
    dq_nope_ref = q_nope_ref.grad.clone()
    dk_nope_ref = k_nope_ref.grad.clone()
    dv_ref = v_ref.grad.clone()
    dw_ref = weights_ref.grad.clone()

    # 转换 Ref 梯度
    dq_ref_t = dq_nope_ref.transpose(1, 2).contiguous().float()
    dk_ref_t = dk_nope_ref.transpose(1, 2).contiguous().float()
    dq_rope_ref_t, dk_rope_ref_t = liger_rotary_pos_emb(dq_ref_t, dk_ref_t, cos_f32, sin_f32)
    dq_rope_ref = dq_rope_ref_t.to(dtype).transpose(1, 2).contiguous()
    dk_rope_ref = dk_rope_ref_t.to(dtype).transpose(1, 2).contiguous()

    # 辅助函数：打印最大误差位置的对比
    def print_max_err_compare(name, tensor_hsa, tensor_ref):
        hsa_f = tensor_hsa.float()
        ref_f = tensor_ref.float()
        diff_abs = (hsa_f - ref_f).abs()
        
        # 1. 绝对误差最大位置
        max_abs_err = diff_abs.max().item()
        idx_abs = tuple(i[0].item() for i in torch.where(diff_abs == max_abs_err))
        
        # 2. 相对误差最大位置
        diff_rel = diff_abs / (ref_f.abs() + 1e-6)
        max_rel_err = diff_rel.max().item()
        idx_rel = tuple(i[0].item() for i in torch.where(diff_rel == max_rel_err))
        
        print(f"\n[{name} Error Analysis]:")
        print(f"  -> Max Absolute Error: {max_abs_err:.6e} at {idx_abs}")
        print(f"     HSA Val: {hsa_f[idx_abs].item():.10f} | Ref Val: {ref_f[idx_abs].item():.10f}")
        
        print(f"  -> Max Relative Error: {max_rel_err:.6e} at {idx_rel}")
        print(f"     HSA Val: {hsa_f[idx_rel].item():.10f} | Ref Val: {ref_f[idx_rel].item():.10f}")

    print("\n[Backward Gradients Comparison at Max Error Position]:")
    print_max_err_compare("DQ", dq_rope_hsa, dq_rope_ref)
    print_max_err_compare("DK", dk_rope_hsa, dk_rope_ref)
    print_max_err_compare("DV", dv_hsa, dv_ref)
    print_max_err_compare("DW", dw_hsa, dw_ref)




def main_rope_latency_memory():
    """
    对比开启和不开启 enable_inverse_rope 的延迟和显存
    """
    import torch
    import torch.nn.functional as F
    import math
    import gc

    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 64, 4096, 2, 16, 128, 8, 64
    block_M = 4
    mask_last_token = True
    dtype = torch.bfloat16
    device = "cuda"
    scale = 1.0 / math.sqrt(D)

    print("=" * 70)
    print(f"RoPE Latency & Memory Test")
    print(f"Config: B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block={block_size}, block_M={block_M}")
    print("=" * 70)

    # ---------- 生成 block_indices ----------
    overlap_ratio = 0.8
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

    # ---------- 生成 cos/sin ----------
    angle = torch.randn((1, SEQ_LEN, D), dtype=dtype, device=device)
    cos = torch.cos(angle).expand(B, -1, -1).contiguous()
    sin = torch.sin(angle).expand(B, -1, -1).contiguous()

    num_warmup = 20
    num_iters = 50

    def measure_time(func, num_iters=num_iters):
        """测量函数执行时间"""
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters  # ms per iter

    def measure_memory(func, description=""):
        """测量函数执行的峰值显存"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始显存
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        
        # 执行函数
        result = func()
        
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        mem_peak = torch.cuda.max_memory_allocated()
        
        activation_mem = mem_after - mem_before  # 常驻激活值
        peak_mem = mem_peak - mem_before  # 峰值显存增量
        
        return result, activation_mem, peak_mem

    def create_inputs(requires_grad=True):
        """创建输入张量"""
        Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=requires_grad)
        K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=requires_grad)
        V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=requires_grad)
        
        logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
        valid_mask = (block_indices != -1)
        repeat_factor = HQ // H  # 8
        valid_mask = valid_mask.unsqueeze(2)
        valid_mask = valid_mask.expand(B, SEQ_LEN, repeat_factor, H, S)
        valid_mask = valid_mask.reshape(B, SEQ_LEN, HQ, S)

        logits = logits.masked_fill(~valid_mask, float('-inf'))
        W = F.softmax(logits, dim=-1)
        W = torch.nan_to_num(W, 0.0).detach().requires_grad_(requires_grad)
        
        grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
        
        return Q, K, V, W, grad_output

    def bytes_to_mb(b):
        return b / (1024 * 1024)

    # ==================== 测试 1: 不开启 RoPE ====================
    print("\n" + "-" * 50)
    print("[Test 1] enable_inverse_rope = False")
    print("-" * 50)
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    Q, K, V, W, grad_output = create_inputs()
    
    # Warmup
    for _ in range(num_warmup):
        O = HSA_block_M_head(Q, K, V, W, block_indices, 
                            block_size=block_size, sm_scale=scale, 
                            block_M=block_M, mask_last_token=mask_last_token,
                            enable_inverse_rope=False)
        O.backward(grad_output)
        Q.grad = K.grad = V.grad = W.grad = None

    # 测量 FWD 延迟
    def no_rope_fwd():
        with torch.no_grad():
            O = HSA_block_M_head(Q, K, V, W, block_indices, 
                                block_size=block_size, sm_scale=scale, 
                                block_M=block_M, mask_last_token=mask_last_token,
                                enable_inverse_rope=False)
    
    fwd_ms_no_rope = measure_time(no_rope_fwd)

    # 测量 FWD+BWD 延迟
    def no_rope_fwd_bwd():
        Q.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_head(Q, K, V, W, block_indices, 
                            block_size=block_size, sm_scale=scale, 
                            block_M=block_M, mask_last_token=mask_last_token,
                            enable_inverse_rope=False)
        O.backward(grad_output)
    
    total_ms_no_rope = measure_time(no_rope_fwd_bwd)
    bwd_ms_no_rope = total_ms_no_rope - fwd_ms_no_rope

    # 测量显存
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    Q1, K1, V1, W1, grad_output1 = create_inputs()
    mem_before = torch.cuda.memory_allocated()
    
    O1 = HSA_block_M_head(Q1, K1, V1, W1, block_indices, 
                         block_size=block_size, sm_scale=scale, 
                         block_M=block_M, mask_last_token=mask_last_token,
                         enable_inverse_rope=False)
    
    torch.cuda.synchronize()
    mem_after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()
    
    activation_fwd_no_rope = mem_after_fwd - mem_before
    peak_fwd_no_rope = peak_fwd - mem_before
    
    torch.cuda.reset_peak_memory_stats()
    O1.backward(grad_output1)
    
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated()
    peak_bwd_no_rope = peak_bwd - mem_before

    print(f"延迟: FWD={fwd_ms_no_rope:.3f}ms | BWD={bwd_ms_no_rope:.3f}ms | Total={total_ms_no_rope:.3f}ms")
    print(f"显存: FWD激活={bytes_to_mb(activation_fwd_no_rope):.2f}MB | FWD峰值={bytes_to_mb(peak_fwd_no_rope):.2f}MB | BWD峰值={bytes_to_mb(peak_bwd_no_rope):.2f}MB")

    del Q, K, V, W, O, Q1, K1, V1, W1, O1, grad_output, grad_output1

    # ==================== 测试 2: 开启 RoPE ====================
    print("\n" + "-" * 50)
    print("[Test 2] enable_inverse_rope = True")
    print("-" * 50)
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 创建 rope 版本的输入
    Q_nope, K_nope, V, W, grad_output = create_inputs()
    
    # 生成 rope 版本
    Q_t = Q_nope.transpose(1, 2).contiguous().float()
    K_t = K_nope.transpose(1, 2).contiguous().float()
    cos_f32 = cos.float()
    sin_f32 = sin.float()
    
    Q_rope_t, K_rope_t = liger_rotary_pos_emb(Q_t, K_t, cos_f32, sin_f32)
    Q_rope = Q_rope_t.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    K_rope = K_rope_t.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    
    del Q_nope, K_nope, Q_t, K_t, Q_rope_t, K_rope_t
    gc.collect()
    torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(num_warmup):
        O = HSA_block_M_head(Q_rope, K_rope, V, W, block_indices, 
                            block_size=block_size, sm_scale=scale, 
                            block_M=block_M, mask_last_token=mask_last_token,
                            enable_inverse_rope=True, cos=cos, sin=sin)
        O.backward(grad_output)
        Q_rope.grad = K_rope.grad = V.grad = W.grad = None

    # 测量 FWD 延迟
    def rope_fwd():
        with torch.no_grad():
            O = HSA_block_M_head(Q_rope, K_rope, V, W, block_indices, 
                                block_size=block_size, sm_scale=scale, 
                                block_M=block_M, mask_last_token=mask_last_token,
                                enable_inverse_rope=True, cos=cos, sin=sin)
    
    fwd_ms_rope = measure_time(rope_fwd)

    # 测量 FWD+BWD 延迟
    def rope_fwd_bwd():
        Q_rope.grad = K_rope.grad = V.grad = W.grad = None
        O = HSA_block_M_head(Q_rope, K_rope, V, W, block_indices, 
                            block_size=block_size, sm_scale=scale, 
                            block_M=block_M, mask_last_token=mask_last_token,
                            enable_inverse_rope=True, cos=cos, sin=sin)
        O.backward(grad_output)
    
    total_ms_rope = measure_time(rope_fwd_bwd)
    bwd_ms_rope = total_ms_rope - fwd_ms_rope

    # 测量显存
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    Q_nope2, K_nope2, V2, W2, grad_output2 = create_inputs()
    Q_t2 = Q_nope2.transpose(1, 2).contiguous().float()
    K_t2 = K_nope2.transpose(1, 2).contiguous().float()
    Q_rope_t2, K_rope_t2 = liger_rotary_pos_emb(Q_t2, K_t2, cos_f32, sin_f32)
    Q_rope2 = Q_rope_t2.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    K_rope2 = K_rope_t2.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    del Q_nope2, K_nope2, Q_t2, K_t2, Q_rope_t2, K_rope_t2
    gc.collect()
    torch.cuda.empty_cache()
    
    mem_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    
    O2 = HSA_block_M_head(Q_rope2, K_rope2, V2, W2, block_indices, 
                         block_size=block_size, sm_scale=scale, 
                         block_M=block_M, mask_last_token=mask_last_token,
                         enable_inverse_rope=True, cos=cos, sin=sin)
    
    torch.cuda.synchronize()
    mem_after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()
    
    activation_fwd_rope = mem_after_fwd - mem_before
    peak_fwd_rope = peak_fwd - mem_before
    
    torch.cuda.reset_peak_memory_stats()
    O2.backward(grad_output2)
    
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated()
    peak_bwd_rope = peak_bwd - mem_before

    print(f"延迟: FWD={fwd_ms_rope:.3f}ms | BWD={bwd_ms_rope:.3f}ms | Total={total_ms_rope:.3f}ms")
    print(f"显存: FWD激活={bytes_to_mb(activation_fwd_rope):.2f}MB | FWD峰值={bytes_to_mb(peak_fwd_rope):.2f}MB | BWD峰值={bytes_to_mb(peak_bwd_rope):.2f}MB")

    # ==================== 对比汇总 ====================
    print("\n" + "=" * 70)
    print("对比汇总")
    print("=" * 70)
    
    print(f"\n{'指标':<20} {'无RoPE':<20} {'有RoPE':<20} {'开销':<20}")
    print("-" * 70)
    
    fwd_overhead = fwd_ms_rope - fwd_ms_no_rope
    bwd_overhead = bwd_ms_rope - bwd_ms_no_rope
    total_overhead = total_ms_rope - total_ms_no_rope
    
    print(f"{'FWD 延迟 (ms)':<20} {fwd_ms_no_rope:<20.3f} {fwd_ms_rope:<20.3f} {fwd_overhead:+.3f} ({fwd_overhead/fwd_ms_no_rope*100:+.1f}%)")
    print(f"{'BWD 延迟 (ms)':<20} {bwd_ms_no_rope:<20.3f} {bwd_ms_rope:<20.3f} {bwd_overhead:+.3f} ({bwd_overhead/bwd_ms_no_rope*100:+.1f}%)")
    print(f"{'Total 延迟 (ms)':<20} {total_ms_no_rope:<20.3f} {total_ms_rope:<20.3f} {total_overhead:+.3f} ({total_overhead/total_ms_no_rope*100:+.1f}%)")
    
    print()
    activation_overhead = activation_fwd_rope - activation_fwd_no_rope
    peak_fwd_overhead = peak_fwd_rope - peak_fwd_no_rope
    peak_bwd_overhead = peak_bwd_rope - peak_bwd_no_rope
    
    print(f"{'FWD 激活 (MB)':<20} {bytes_to_mb(activation_fwd_no_rope):<20.2f} {bytes_to_mb(activation_fwd_rope):<20.2f} {bytes_to_mb(activation_overhead):+.2f}")
    print(f"{'FWD 峰值 (MB)':<20} {bytes_to_mb(peak_fwd_no_rope):<20.2f} {bytes_to_mb(peak_fwd_rope):<20.2f} {bytes_to_mb(peak_fwd_overhead):+.2f}")
    print(f"{'BWD 峰值 (MB)':<20} {bytes_to_mb(peak_bwd_no_rope):<20.2f} {bytes_to_mb(peak_bwd_rope):<20.2f} {bytes_to_mb(peak_bwd_overhead):+.2f}")



import pytest
import torch.nn.functional as F
@pytest.mark.parametrize("B, SEQ_LEN, H, HQ, D, S, block_size", [(1, 1024, 1, 8, 64, 4, 32)])
def test_correctness_fp32(B, SEQ_LEN, H, HQ, D, S, block_size):
    device = "cuda"
    dtype = torch.float32
    scale = 1.0 / math.sqrt(D)
    block_M = 2
    mask_last_token = True
    torch.manual_seed(42)

    # 1. 构造 block_indices
    block_indices = torch.full((B, SEQ_LEN, H, S), -1, dtype=torch.int32, device=device)
    num_blocks = SEQ_LEN // block_size
    for t in range(SEQ_LEN):
        max_blocks = min(t // block_size + 1, num_blocks)
        if max_blocks > 0:
            num_select = min(S, max_blocks)
            selected = torch.randperm(max_blocks, device=device)[:num_select]
            block_indices[:, t, :, :num_select] = selected.sort()[0]

    # 2. 生成输入数据
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
    logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
    logits.masked_fill_(block_indices.repeat_interleave(HQ//H, dim=2) == -1, float('-inf'))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    # 3. 前向计算
    O_hsa = HSA_block_M_head(Q, K, V, W, block_indices, block_size, scale, block_M, mask_last_token, "float", "float", 64)
    O_ref = hsa_torch_ref(Q.detach(), K.detach(), V.detach(), W.detach(), block_indices,
                        chunk_size=block_size, 
                        sm_scale=scale, 
                        block_q=1,
                        mask_last_token=mask_last_token)

    # 4. 验证前向
    torch.testing.assert_close(O_hsa, O_ref, atol=0.005, rtol=0.005, msg="Forward mismatch")

    # 5. 反向计算
    O_hsa.backward(grad_output)
    DQ_hsa, DK_hsa, DV_hsa, DW_hsa = Q.grad.clone(), K.grad.clone(), V.grad.clone(), W.grad.clone()
    
    Q.grad, K.grad, V.grad, W.grad = None, None, None, None
    O_ref_bwd = hsa_torch_ref(Q, K, V, W, block_indices, 
                            chunk_size=block_size, 
                            sm_scale=scale, 
                            block_q=1,
                            mask_last_token=mask_last_token)
    O_ref_bwd.backward(grad_output)

    # 6. 验证反向梯度
    def get_abs_err(x, y):
        return (x-y).flatten().abs().max().item()
    def get_err_ratio(x, y):
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / base
    def assert_close(prefix, ref, tri, ratio):
        msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
        print(msg)
        assert get_err_ratio(ref, tri) < ratio, msg
    assert_close("DQ", Q.grad, DQ_hsa, 0.005)
    assert_close("DK", K.grad, DK_hsa, 0.005)
    assert_close("DV", V.grad, DV_hsa, 0.005)
    assert_close("DW", W.grad, DW_hsa, 0.005)
    # torch.testing.assert_close(DQ_hsa, Q.grad, atol=0.005, rtol=0.005, msg="DQ mismatch")
    # torch.testing.assert_close(DK_hsa, K.grad, atol=0.005, rtol=0.005, msg="DK mismatch")
    # torch.testing.assert_close(DV_hsa, V.grad, atol=0.005, rtol=0.005, msg="DV mismatch")
    # torch.testing.assert_close(DW_hsa, W.grad, atol=0.005, rtol=0.005, msg="DW mismatch")
    print(f"FP32 Correctness Test Passed for B={B}, SEQ_LEN={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block_size={block_size}")

if __name__ == "__main__":
    # ### 验证 HSA_block_M (TileLang) 的正确性
    main_block_M_correctness()
    
    
    
    # ### 对比 HSA_block_M (TileLang) 与 Triton HSA 的延迟
    main_block_M_latency()
    
    # main_rope_correctness()
    
    # main_rope_latency_memory()
    
    params_list = [
        (1, 1024, 1, 8, 64, 4, 32),
        (2, 2048, 2, 16, 128, 4, 32),
        (3, 512, 1, 8, 64, 4, 32),
        (4, 512, 2, 16, 128, 4, 64),
        (5, 256, 1, 8, 64, 4, 64),
    ]
    for p in params_list:
        test_correctness_fp32(*p)
    