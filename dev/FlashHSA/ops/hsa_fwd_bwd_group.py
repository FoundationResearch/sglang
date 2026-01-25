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
    - weights: (B, q_blocks, H, K)
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

    qk = torch.einsum('b q x h d, b q s k h d -> b q x s k h', q_chunked.float(), k_.float())
    qk = qk * float(sm_scale)
    
    if mask_last_token:
        qk[:, :, :, -1, :, :] = float("-inf")

    p = torch.softmax(qk, dim=3)  # S 维

    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    w_masked = weights.clone()
    w_masked = w_masked.masked_fill(~valid_mask, 0)
    w_exp = torch.repeat_interleave(w_masked, dim=-2, repeats=G).float()  # (B, Bq, HQ, K)

    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'B Bq X hq d -> B (Bq X) hq d')  # (B, L, HQ, D)
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
    out_idx=[-1],
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
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504
    else:
        scale = scale * 1.44269504

    head_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    kv_shape = [batch, kv_len, head_kv, head_dim]
    weight_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
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
    print("Using M =", M, "for fwd_block_M kernel")
    M_G = M * groups  # 一次 GEMM 处理的总 head 数

    S = selected_blocks
    BS = block_S
    BK = BV = block_T
    num_stages = 0
    threads = 128

    @T.prim_func
    def hsa_block_M(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(tilelang.cdiv(q_len, M), NV, batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([M_G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([M_G, BV], dtype)

            acc_s = T.alloc_fragment([M_G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([M_G, BS], dtype)
            acc_o = T.alloc_fragment([M_G, BV], accum_dtype)
            
            P_shared = T.alloc_shared([M_G, BS], accum_dtype)
            

            acc_s_tmp = T.alloc_fragment([groups, BS], accum_dtype)
            scores_max = T.alloc_fragment([groups], accum_dtype)
            scores_sum = T.alloc_fragment([groups], accum_dtype)

            merged_indices = T.alloc_shared([S * M], block_indices_dtype)
            block_ownership = T.alloc_shared([S * M], "int32") # Bitmask
            merged_len = T.alloc_shared([1], "int32")
            chunk_weights = T.alloc_shared([S * M, M], dtype)
            

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
            
            # 连续加载 W 到 shared memory备用
            W_local_shared = T.alloc_shared([M, S], dtype)
            T.fill(W_local_shared, 0.0)
            for q_idx, s_idx in T.Parallel(M, S):
                tq = base_t + q_idx
                if tq < q_len:
                    W_local_shared[q_idx, s_idx] = W[i_b, tq, i_h, s_idx]

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
                            else:
                                T.loop_break()
                    
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
                    for q_idx in T.serial(M):
                        tq = base_t + q_idx
                        if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
                            # pointers[q_idx] 此时就是 W 在 S 维的索引
                            s_idx = pointers[q_idx]
                            val_q = cur_val[q_idx]
                            if val_q == min_val:
                                ownership_mask = ownership_mask | (1 << q_idx)
                                # 记录合并后第k个块对应于第q_idx个query的权重
                                chunk_weights[k, q_idx] = W_local_shared[q_idx, s_idx]
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
                    T.copy(acc_s, P_shared)
                    
                    for q_idx in T.serial(M):
                        if (ownership & (1 << q_idx)) != 0:
                            h_start = q_idx * groups

                            for g, s in T.Parallel(groups, BS):
                                acc_s_tmp[g, s] = T.if_then_else(s == BS - 1 and enable_last_token_mask, -T.infinity(accum_dtype),
                                                                 P_shared[h_start + g, s]) 

                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s_tmp, scores_max, dim=1, clear=True)
                            for g, s in T.Parallel(groups, BS):
                                acc_s_tmp[g, s] = T.exp2(acc_s_tmp[g, s] * scale - scores_max[g] * scale)
                            T.reduce_sum(acc_s_tmp, scores_sum, dim=1, clear=True)

                            weight_q = chunk_weights[i, q_idx]
                            for g, s in T.Parallel(groups, BS):
                                acc_s_tmp[g, s] = weight_q * acc_s_tmp[g, s] / scores_sum[g]

                            for g, s in T.Parallel(groups, BS):
                                P_shared[h_start + g, s] = acc_s_tmp[g, s]
                        else:
                            h_start = q_idx * groups
                            for g, s in T.Parallel(groups, BS):
                                P_shared[h_start + g, s] = 0.0
                                

                    T.copy(P_shared, acc_s_cast)
                    T.copy(V[i_b, i_s:i_s + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

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
    scale=None, block_size=64, groups=16, selected_blocks=16,
    dtype="bfloat16", accum_dtype="float", block_M = 0, mask_last_token=False,
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

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    do_shape = [batch, q_len, heads, head_dim]

    dq_shape = [NV, batch, q_len, heads, head_dim]
    dk_shape = [NV, batch, kv_len, heads_kv, head_dim]
    dv_shape = [batch, kv_len, heads_kv, head_dim]
    weight_shape = [batch, q_len, heads_kv, selected_blocks]
    dw_shape = [batch, q_len, heads_kv, selected_blocks]
    block_mask_shape = [batch, q_len, heads_kv, NS_kv]

    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, G)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)     
    print("Using M =", M, "for bwd_block_M kernel")
    M_G = M * G
    NP = tilelang.cdiv(q_len, M)

    num_threads = 128
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
        DW: T.Tensor(dw_shape, accum_dtype),
        BlockMask: T.Tensor(block_mask_shape, "int32"),
    ):
        with T.Kernel(NV, NS_kv, B * heads_kv, threads=num_threads) as (i_v, i_s, i_bh):
            i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
            i_s_global = i_s * BS

            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            Q_shared = T.alloc_shared([M_G, BK], dtype)
            dO_shared = T.alloc_shared([M_G, BV], dtype)
            
            P_shared = T.alloc_shared([M_G, BS], dtype)
            dO_weighted_shared = T.alloc_shared([M_G, BV], dtype)
            dS_shared = T.alloc_shared([M_G, BS], dtype)
            dQ_shared = T.alloc_shared([M_G, BK], dtype)
            dK_shared = T.alloc_shared([BS, BK], dtype)
            dV_shared = T.alloc_shared([BS, BV], dtype)

            qk_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            T_raw_frag = T.alloc_fragment([M_G, BV], accum_dtype)
            dV_PdO_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            dS_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            dV_accum = T.alloc_fragment([BS, BV], accum_dtype)
            dK_accum = T.alloc_fragment([BS, BK], accum_dtype)
            dQ_local = T.alloc_fragment([M_G, BK], accum_dtype)
            delta_rows = T.alloc_fragment([M_G], accum_dtype)
            logits_shared = T.alloc_shared([M_G, BS], accum_dtype)

            acc_s_tmp = T.alloc_fragment([G, BS], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            
            T_raw_shared = T.alloc_shared([M_G, BV], accum_dtype)
            
            dw_partial = T.alloc_fragment([G, BV], accum_dtype)
            dw_g_frag = T.alloc_fragment([G], accum_dtype)
            sum_w = T.alloc_fragment([1], accum_dtype)
            
            W_local = T.alloc_shared([M], dtype)
            
            pos = T.alloc_fragment([M], "int32")
            has_q = T.alloc_fragment([M], "int32")
            any_valid = T.alloc_var("int32")
            
            T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :], K_shared)
            T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
            T.fill(dK_accum, 0)
            T.fill(dV_accum, 0)

            T.annotate_layout({
                DQ: make_dq_layout_hsa(DQ),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                dO_shared: tilelang.layout.make_swizzled_layout(dO_shared),
                P_shared: tilelang.layout.make_swizzled_layout(P_shared),
                dS_shared: tilelang.layout.make_swizzled_layout(dS_shared),
                dO_weighted_shared: tilelang.layout.make_swizzled_layout(dO_weighted_shared),
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
                    
                    T.clear(qk_frag)
                    T.gemm(Q_shared, K_shared, qk_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(qk_frag, logits_shared)
                    T.fill(P_shared, 0.0)

                    for qi4 in T.serial(M):
                        h_start = qi4 * G
                        if has_q[qi4] == 1:
                            for g, s in T.Parallel(G, BS):
                                acc_s_tmp[g, s] = T.if_then_else(s == BS - 1 and enable_last_token_mask, -T.infinity(accum_dtype), logits_shared[h_start + g, s])
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s_tmp, scores_max, dim=1, clear=True)
                            for g, s in T.Parallel(G, BS):
                                acc_s_tmp[g, s] = T.exp2(acc_s_tmp[g, s] * scale_log2 - scores_max[g] * scale_log2)
                            T.reduce_sum(acc_s_tmp, scores_sum, dim=1, clear=True)
                            for g, s in T.Parallel(G, BS):
                                acc_s_tmp[g, s] = acc_s_tmp[g, s] / scores_sum[g]
                                P_shared[h_start + g, s] = acc_s_tmp[g, s]


                    T.clear(T_raw_frag)
                    T.gemm(P_shared, V_shared, T_raw_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(T_raw_frag, T_raw_shared)
                    
                    for qi5 in T.serial(M):
                        if has_q[qi5] == 1:
                            h_start = qi5 * G
                            for g, v in T.Parallel(G, BV):
                                dw_partial[g, v] = dO_shared[h_start + g, v] * T_raw_shared[h_start + g, v]
                            T.reduce_sum(dw_partial, dw_g_frag, dim=1, clear=True)
                            T.reduce_sum(dw_g_frag, sum_w, dim=0, clear=True)

                            tq = base_t + qi5
                            if tq < q_len and pos[qi5] != -1:
                                DW[i_b, tq, i_h, pos[qi5]] = sum_w[0]
            

                    T.fill(W_local, 0.0)
                    
                    for qi_w in T.Parallel(M):
                        tq = base_t + qi_w
                        if tq < q_len and has_q[qi_w] == 1 and pos[qi_w] != -1:
                            W_local[qi_w] = W[i_b, tq, i_h, pos[qi_w]]

                    for g, v in T.Parallel(M_G, BV):
                        q_idx = g // G
                        dO_weighted_shared[g, v] = W_local[q_idx] * dO_shared[g, v]

                    T.gemm(P_shared, dO_weighted_shared, dV_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    T.clear(dV_PdO_frag)
                    T.gemm(dO_weighted_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for g_row, s in T.Parallel(M_G, BS):
                        dS_frag[g_row, s] = P_shared[g_row, s] * dV_PdO_frag[g_row, s]
                    T.reduce_sum(dS_frag, delta_rows, dim=1, clear=True)
                    for g_row, s in T.Parallel(M_G, BS):
                        dS_frag[g_row, s] = sm_scale * (dS_frag[g_row, s] - P_shared[g_row, s] * delta_rows[g_row])

                    T.copy(dS_frag, dS_shared)  
                    
                    T.gemm(dS_shared, Q_shared, dK_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    
                    T.clear(dQ_local)
                    T.gemm(dS_shared, K_shared, dQ_local, policy=T.GemmWarpPolicy.FullRow)
                    
                    T.copy(dQ_local, dQ_shared)
                    
                    has_q_shared = T.alloc_shared([M], "int32")
                    T.copy(has_q, has_q_shared)
                    for g_row, k in T.Parallel(M_G, BK):
                        q_idx = g_row // G
                        g_local = g_row % G
                        tq = base_t + q_idx
                        if tq < q_len and has_q_shared[q_idx] == 1:
                            T.atomic_add(DQ[i_v, i_b, tq, i_h * G + g_local, k], dQ_shared[g_row, k])

            T.copy(dK_accum, dK_shared)
            T.copy(dV_accum, dV_shared)
            T.copy(dK_shared, DK[i_v, i_b, i_s_global:i_s_global + BS, i_h, :])
            T.copy(dV_shared, DV[i_b, i_s_global:i_s_global + BS, i_h, i_v * BV:(i_v + 1) * BV])

    return hsa_bwd_dqkv_block_M



class _hsa_block_M_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, weights, indices, block_size, sm_scale, block_M, mask_last_token):
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
            scale=sm_scale,
            block_M=block_M,
            mask_last_token=mask_last_token,
        )

        # 执行前向
        o = fwd_kernel(q, k, v, weights, indices)

        # 保存用于反向的张量
        ctx.save_for_backward(q, k, v, weights, indices)
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.block_M = block_M
        ctx.mask_last_token = mask_last_token
        ctx.B = B
        ctx.L = L
        ctx.HQ = HQ
        ctx.H = H
        ctx.D = D
        ctx.S = S
        ctx.G = G

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, weights, indices = ctx.saved_tensors
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        block_M = ctx.block_M
        mask_last_token = ctx.mask_last_token
        B, L, HQ, H, D, S, G = ctx.B, ctx.L, ctx.HQ, ctx.H, ctx.D, ctx.S, ctx.G

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
            scale=sm_scale,
            block_M=block_M,
            mask_last_token=mask_last_token,
        )

        # 分配梯度缓冲
        dq = torch.zeros((NV, B, L, HQ, D), dtype=torch.float32, device=q.device)
        dk = torch.zeros((NV, B, L_kv, H, D), dtype=k.dtype, device=k.device)
        dv = torch.zeros((B, L_kv, H, D), dtype=v.dtype, device=v.device)
        dw = torch.zeros((B, L, H, S), dtype=torch.float32, device=weights.device)

        # 执行反向
        bwd_kernel(q, k, v, weights, do, dq, dk, dv, dw, block_mask)

        post_kernel = hsa_bwd_postprocess(NV, B, L, HQ, D)
        dq = post_kernel(dq)

        dq = dq.sum(0)
        dk = dk.sum(0)

        dq = dq.to(q.dtype)
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)
        dw = dw.to(weights.dtype)

        return dq, dk, dv, dw, None, None, None, None, None


def HSA_block_M_group(q, k, v, weights, indices, block_size=32, sm_scale=None, block_M=0, mask_last_token=False):
    return _hsa_block_M_attention.apply(q, k, v, weights, indices, block_size, sm_scale, block_M, mask_last_token)
    




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
    dtype="bfloat16", accum_dtype="float", block_M = 0,
):
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
    weight_shape = [batch, q_len, heads_kv, selected_blocks]
    dw_shape = [batch, q_len, heads_kv, selected_blocks]
    block_indices_shape = [batch, q_len, heads_kv, selected_blocks]

    # 动态计算 M
    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, G)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)
    print("Using M =", M, "for bwd_block_M_inverse kernel")
    M_G = M * G

    num_threads = 256
    num_stages = 0

    @T.prim_func
    def hsa_bwd_dqkv_block_M_inverse(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        W: T.Tensor(weight_shape, accum_dtype),
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

            Q_shared = T.alloc_shared([M_G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            dO_shared = T.alloc_shared([M_G, BV], dtype)
            P_shared = T.alloc_shared([M_G, BS], dtype)
            logits_shared = T.alloc_shared([M_G, BS], accum_dtype)
            dO_weighted_shared = T.alloc_shared([M_G, BV], dtype)

            dQ_accum = T.alloc_fragment([M_G, BK], accum_dtype)
            qk_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            T_raw_frag = T.alloc_fragment([M_G, BV], accum_dtype)
            dV_PdO_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            dS_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            delta_rows = T.alloc_fragment([M_G], accum_dtype)

            acc_s_tmp = T.alloc_fragment([G, BS], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            
            T_raw_shared = T.alloc_shared([M_G, BV], accum_dtype)
            dw_partial = T.alloc_fragment([G, BV], accum_dtype)
            dw_g_frag = T.alloc_fragment([G], accum_dtype)
            sum_w = T.alloc_fragment([1], accum_dtype)

            merged_indices = T.alloc_shared([S * M], "int32")
            block_ownership = T.alloc_shared([S * M], "int32")
            merged_len = T.alloc_shared([1], "int32")
            chunk_weights = T.alloc_shared([S * M, M], accum_dtype)
            W_local_shared = T.alloc_shared([M, S], accum_dtype)
            merged_s_indices = T.alloc_shared([S * M, M], "int32")

            T.fill(Q_shared, 0)
            T.fill(dO_shared, 0)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * G
                    T.copy(Q[i_b, tq, i_h * G:(i_h + 1) * G, :], Q_shared[h_start:h_start + G, :])
                    T.copy(DO[i_b, tq, i_h * G:(i_h + 1) * G, i_v * BV:(i_v + 1) * BV], dO_shared[h_start:h_start + G, :])

            T.fill(dQ_accum, 0)
            T.fill(merged_indices, -1)
            T.fill(block_ownership, 0)
            T.fill(chunk_weights, 0)
            T.fill(merged_len, 0)
            T.fill(merged_s_indices, -1)
            T.fill(W_local_shared, 0.0)
            for q_idx, s_idx in T.Parallel(M, S):
                tq = base_t + q_idx
                if tq < q_len:
                    W_local_shared[q_idx, s_idx] = W[i_b, tq, i_h, s_idx]

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
                                chunk_weights[merge_k, q_idx] = W_local_shared[q_idx, s_idx]
                                merged_s_indices[merge_k, q_idx] = s_idx
                                pointers[q_idx] = pointers[q_idx] + 1
                    block_ownership[merge_k] = ownership_mask
                    merge_k = merge_k + 1
                    
                merged_len[0] = merge_k

            T.sync_threads()

            merged_len_local = merged_len[0]
            h_start = T.alloc_var("int32")

            for i in T.Pipelined(merged_len_local, num_stages=num_stages):
                blk_idx = merged_indices[i]
                i_s_global = blk_idx * BS
                ownership = block_ownership[i]

                if blk_idx >= 0:
                    T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :], K_shared)
                    T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)

                    T.clear(qk_frag)
                    T.gemm(Q_shared, K_shared, qk_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(qk_frag, logits_shared)
                    T.fill(P_shared, 0.0) 
                    
                    
                    for q_idx in T.serial(M):
                        if (ownership & (1 << q_idx)) != 0:
                            h_start = q_idx * G
                            for g, s in T.Parallel(G, BS):
                                acc_s_tmp[g, s] = logits_shared[h_start + g, s]
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s_tmp, scores_max, dim=1, clear=True)
                            for g, s in T.Parallel(G, BS):
                                acc_s_tmp[g, s] = T.exp2(acc_s_tmp[g, s] * scale_log2 - scores_max[g] * scale_log2)
                            T.reduce_sum(acc_s_tmp, scores_sum, dim=1, clear=True)
                            for g, s in T.Parallel(G, BS):
                                P_shared[h_start + g, s] = acc_s_tmp[g, s] / scores_sum[g]


                    T.clear(T_raw_frag)
                    T.gemm(P_shared, V_shared, T_raw_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(T_raw_frag, T_raw_shared)

                    for q_idx in T.serial(M):
                        if (ownership & (1 << q_idx)) != 0:
                            h_start = q_idx * G
                            for g, v in T.Parallel(G, BV): dw_partial[g, v] = dO_shared[h_start + g, v] * T_raw_shared[h_start + g, v]
                            T.reduce_sum(dw_partial, dw_g_frag, dim=1, clear=True)
                            T.reduce_sum(dw_g_frag, sum_w, dim=0, clear=True)
                            
                            tq = base_t + q_idx
                            s_idx_local = T.alloc_var("int32")
                            s_idx_local = -1
                            if tq < q_len:
                                s_idx_local = merged_s_indices[i, q_idx]
                                if s_idx_local != -1:
                                    DW[i_b, tq, i_h, s_idx_local] = sum_w[0]
                                
                                
                    for g, v in T.Parallel(M_G, BV):
                        q_idx = g // G
                        weight_q = chunk_weights[i, q_idx]
                        dO_weighted_shared[g, v] = weight_q * dO_shared[g, v]

                    dV_accum_local = T.alloc_fragment([BS, BV], accum_dtype)
                    T.clear(dV_accum_local)
                    T.gemm(P_shared, dO_weighted_shared, dV_accum_local, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    
                    T.clear(dV_PdO_frag)
                    T.gemm(dO_weighted_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for g_row, s in T.Parallel(M_G, BS): dS_frag[g_row, s] = P_shared[g_row, s] * dV_PdO_frag[g_row, s]
                    T.reduce_sum(dS_frag, delta_rows, dim=1, clear=True)
                    for g_row, s in T.Parallel(M_G, BS): dS_frag[g_row, s] = sm_scale * (dS_frag[g_row, s] - P_shared[g_row, s] * delta_rows[g_row])
                    # T.copy(dS_frag, dS_cast)
                    T.copy(dS_frag, P_shared)  # 复用 P_shared 存放 dS

                    dK_accum_local = T.alloc_fragment([BS, BK], accum_dtype)
                    T.clear(dK_accum_local)
                    T.gemm(P_shared, Q_shared, dK_accum_local, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(P_shared, K_shared, dQ_accum, policy=T.GemmWarpPolicy.FullRow)

                    for s, k in T.Parallel(BS, BK):
                        T.atomic_add(DK[i_v, i_b, i_s_global + s, i_h, k], dK_accum_local[s, k])
                    for s, v in T.Parallel(BS, BV):
                        T.atomic_add(DV[i_b, i_s_global + s, i_h, i_v * BV + v], dV_accum_local[s, v])

            # 复用 Q_shared 来写回 dQ
            T.copy(dQ_accum, Q_shared)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * G
                    T.copy(Q_shared[h_start:h_start + G, :], DQ[i_v, i_b, tq, i_h * G:(i_h + 1) * G, :])

    return hsa_bwd_dqkv_block_M_inverse


class _hsa_block_M_attention_inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, weights, indices, block_size, sm_scale, block_M):
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
        )
        o = fwd_kernel(q, k, v, weights, indices)

        ctx.save_for_backward(q, k, v, weights, indices)
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.block_M = block_M
        ctx.G = G
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, weights, indices = ctx.saved_tensors
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
        )

        BV = min(128, tilelang.next_power_of_2(D))
        NV = tilelang.cdiv(D, BV)
        
        dq = torch.zeros((NV, B, L, HQ, D), dtype=q.dtype, device=q.device)
        dk = torch.zeros((NV, B, L_kv, H, D), dtype=torch.float32, device=k.device)
        dv = torch.zeros((B, L_kv, H, D), dtype=torch.float32, device=v.device)
        dw = torch.zeros_like(weights, dtype=torch.float32)

        # Note: The new kernel expects indices, not block_mask
        bwd_kernel(q, k, v, weights, do, indices, dq, dk, dv, dw)
        
        dq = dq.sum(0)
        dk = dk.sum(0)

        return dq, dk, dv, dw, None, None, None, None


def HSA_block_M_inverse(q, k, v, weights, indices, block_size=32, sm_scale=None, block_M=0):
    return _hsa_block_M_attention_inverse.apply(q, k, v, weights, indices, block_size, sm_scale, block_M)




def main_block_M_correctness():
    """
    检验 HSA_pair 封装类的前向和反向传播数值正确性
    与 hsa_torch_ref 进行对比
    """
    import math
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    from hsa_fwd_bwd_triton import HSA
    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 8, 32
    dtype = torch.bfloat16
    device = "cuda"
    block_M=4
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
    logits = torch.randn((B, SEQ_LEN, H, S), dtype=dtype, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, -1e9)  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1).requires_grad_(True)  # leaf tensor，非法位近似为 0
    
    
    # 用于反向传播的梯度
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ========== 测试前向传播 ==========
    
    # HSA_pair 前向
    O_hsa = HSA_block_M_group(Q, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=True)
    
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
        mask_last_token=True
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
    
    O_ref_bwd = hsa_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices,
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=True
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
    O_hsa_bwd = HSA_block_M_group(Q, K, V, W, block_indices, 
                         block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=True)
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
        return max_diff
    
    dq_diff = compute_grad_diff(DQ_hsa, DQ_ref, "DQ")
    dk_diff = compute_grad_diff(DK_hsa, DK_ref, "DK")
    dv_diff = compute_grad_diff(DV_hsa, DV_ref, "DV")
    dw_diff = compute_grad_diff(DW_hsa, DW_ref, "DW")
    
    BLOCK_Q = 1  # 与 tilelang/测试中保持一致（若你想用 block_q>1，请相应修改聚合逻辑）
    q_blocks = SEQ_LEN // BLOCK_Q

    # 聚合：按 block 取第一个 token 的 indices / weight 作为该 q_block 的 representative
    indices_blocks = block_indices.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :].contiguous()
    weights_blocks = W.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :].contiguous()

    # Triton 要求 indices >= 0，所以把无效位指向一个 safe block，同时把对应权重置为 0
    indices_blocks_hsa = indices_blocks.clone()
    safe_block = max(0, (SEQ_LEN // block_size) - 1)
    invalid_mask_blocks = (indices_blocks_hsa < 0)
    indices_blocks_hsa[indices_blocks_hsa < 0] = safe_block

    weights_blocks_hsa = weights_blocks.detach().clone()
    weights_blocks_hsa.requires_grad_(True)
    indices_blocks_hsa = indices_blocks_hsa.contiguous()
    weights_blocks_hsa = weights_blocks_hsa.contiguous()

    # HQ padding（与之前一致，保证 GROUP_NUM * BLOCK_M >= 16）
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    need_padding = (HQ_padded > HQ)

    if need_padding:
        pad_heads = HQ_padded - HQ
        Q_hsa = torch.cat([Q.detach(), torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2).clone().requires_grad_(True)
        grad_output_hsa = torch.cat([grad_output, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2)
    else:
        Q_hsa = Q.detach().clone().requires_grad_(True)
        grad_output_hsa = grad_output

    # 清理可能遗留的 grads
    K.grad = None
    V.grad = None

    # 调用封装 HSA（注意 chunk_size=block_size, sm_n 与 test 中一致用 0）
    O_triton_hsa_padded = HSA(Q_hsa, K, V, weights_blocks_hsa, indices_blocks_hsa, sm_n=0.0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
    O_triton_hsa = O_triton_hsa_padded[:, :, :HQ, :].float()
    print()
    print("[Triton HSA] vs [Torch Reference]:")
    fwd_err = (O_triton_hsa - O_ref.float()).abs().max().item()
    print(f"前向最大误差: {fwd_err:.6e}")

    # backward
    O_triton_hsa_padded.backward(grad_output_hsa)

    DQ_triton = Q_hsa.grad[:, :, :HQ, :].clone() if Q_hsa.grad is not None else None
    DK_triton = K.grad.clone() if K.grad is not None else None
    DV_triton = V.grad.clone() if V.grad is not None else None
    DW_triton_blocks = weights_blocks_hsa.grad.clone() if weights_blocks_hsa.grad is not None else None

    DW_ref_blocks = DW_ref.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :]

    compute_grad_diff(DQ_triton, DQ_ref, "DQ")
    compute_grad_diff(DK_triton, DK_ref, "DK")
    compute_grad_diff(DV_triton, DV_ref, "DV")
    DW_triton_blocks = weights_blocks_hsa.grad.clone()
    DW_triton_blocks[invalid_mask_blocks] = 0  # 对无效位置做 mask
    compute_grad_diff(DW_triton_blocks, DW_ref_blocks, "DW")


def main_block_M_latency():
    """
    对比 tilelang HSA_block_M 和 triton HSA 的 FWD、BWD、以及综合 (FWD+BWD) 延迟
    """
    import torch
    import torch.nn.functional as F
    import time
    import math
    from einops import rearrange
    from hsa_fwd_bwd_triton import HSA

    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 128, 4096, 1, 8, 128, 8, 32
    # B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 8, 32
    block_M=4
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    G_padded = HQ_padded // H
    need_padding = (HQ_padded > HQ)
    scale = 1.0 / math.sqrt(D)

    print(f"Latency Config: B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block={block_size}, G={G}, HQ_padded={HQ_padded}")

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
    logits = torch.randn((B, SEQ_LEN, H, S), dtype=torch.float32, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, -1e9)  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1).requires_grad_(True)  # leaf tensor，非法位近似为 0
    

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
    for _ in range(num_warmup):
        O = HSA(Q_triton, K, V, W, block_indices_triton, sm_n=0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
        O.backward(grad_output_triton)

    # FWD only
    def triton_fwd():
        with torch.no_grad():
            O = HSA(Q_triton, K, V, W, block_indices_triton, sm_n=0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
    triton_fwd_ms = measure_time(triton_fwd)

    # FWD+BWD
    def triton_fwd_bwd():
        Q_triton.grad = K.grad = V.grad = W.grad = None
        O = HSA(Q_triton, K, V, W, block_indices_triton, sm_n=0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
        O.backward(grad_output_triton)
    triton_total_ms = measure_time(triton_fwd_bwd)

    triton_bwd_ms = triton_total_ms - triton_fwd_ms

    # =========================================================
    # TileLang 部分
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_block_M_group(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_group(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_group(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # 输出结果
    # =========================================================
    print(f"[Triton]   FWD: {triton_fwd_ms:.3f} ms | BWD: {triton_bwd_ms:.3f} ms | Total: {triton_total_ms:.3f} ms")
    print(f"[TileLang] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")





def main_block_M_inverse_correctness():
    """
    检验 HSA_pair 封装类的前向和反向传播数值正确性
    与 hsa_torch_ref 进行对比
    """
    import math
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    from hsa_fwd_bwd_triton import HSA
    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 8, 32
    dtype = torch.bfloat16
    device = "cuda"
    block_M=4
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
    logits = torch.randn((B, SEQ_LEN, H, S), dtype=torch.float32, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, -1e9)  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1).requires_grad_(True)  # leaf tensor，非法位近似为 0
    
    
    # 用于反向传播的梯度
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ========== 测试前向传播 ==========
    
    # HSA_pair 前向
    O_hsa = HSA_block_M_inverse(Q, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
    
    # Torch reference 前向
    O_ref = hsa_torch_ref(
        Q.float().detach(), 
        K.float().detach(), 
        V.float().detach(), 
        W.detach(), 
        block_indices,
        chunk_size=block_size, 
        sm_scale=scale, 
        block_q=1
    )

    print("[Tilelang HSA_block_M_inverse] vs [Torch Reference]:")
    # 比较前向输出
    fwd_max_diff = (O_hsa.float() - O_ref.float()).abs().max().item()
    print(f"前向最大误差: {fwd_max_diff:.6e}")

    # ========== 测试反向传播 ==========
    
    # ====== 先计算 Torch reference 反向 ======
    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None
    
    O_ref_bwd = hsa_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices,
        chunk_size=block_size, sm_scale=scale, block_q=1
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
    
    # ====== 再计算 HSA_block_M_inverse 反向 ======
    O_hsa_bwd = HSA_block_M_inverse(Q, K, V, W, block_indices, 
                         block_size=block_size, sm_scale=scale, block_M=block_M)
    O_hsa_bwd.backward(grad_output)
    
    # 获取 HSA_block_M_inverse 梯度
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
        return max_diff
    
    dq_diff = compute_grad_diff(DQ_hsa, DQ_ref, "DQ")
    dk_diff = compute_grad_diff(DK_hsa, DK_ref, "DK")
    dv_diff = compute_grad_diff(DV_hsa, DV_ref, "DV")
    dw_diff = compute_grad_diff(DW_hsa, DW_ref, "DW")
    
    BLOCK_Q = 1  # 与 tilelang/测试中保持一致（若你想用 block_q>1，请相应修改聚合逻辑）
    q_blocks = SEQ_LEN // BLOCK_Q

    # 聚合：按 block 取第一个 token 的 indices / weight 作为该 q_block 的 representative
    indices_blocks = block_indices.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :].contiguous()
    weights_blocks = W.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :].contiguous()

    # Triton 要求 indices >= 0，所以把无效位指向一个 safe block，同时把对应权重置为 0
    indices_blocks_hsa = indices_blocks.clone()
    safe_block = max(0, (SEQ_LEN // block_size) - 1)
    invalid_mask_blocks = (indices_blocks_hsa < 0)
    indices_blocks_hsa[indices_blocks_hsa < 0] = safe_block

    weights_blocks_hsa = weights_blocks.detach().clone()
    weights_blocks_hsa.requires_grad_(True)
    indices_blocks_hsa = indices_blocks_hsa.contiguous()
    weights_blocks_hsa = weights_blocks_hsa.contiguous()

    # HQ padding（与之前一致，保证 GROUP_NUM * BLOCK_M >= 16）
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    need_padding = (HQ_padded > HQ)

    if need_padding:
        pad_heads = HQ_padded - HQ
        Q_hsa = torch.cat([Q.detach(), torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2).clone().requires_grad_(True)
        grad_output_hsa = torch.cat([grad_output, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2)
    else:
        Q_hsa = Q.detach().clone().requires_grad_(True)
        grad_output_hsa = grad_output

    # 清理可能遗留的 grads
    K.grad = None
    V.grad = None

    # 调用封装 HSA（注意 chunk_size=block_size, sm_n 与 test 中一致用 0）
    O_triton_hsa_padded = HSA(Q_hsa, K, V, weights_blocks_hsa, indices_blocks_hsa, sm_n=0.0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
    O_triton_hsa = O_triton_hsa_padded[:, :, :HQ, :].float()
    print()
    print("[Triton HSA] vs [Torch Reference]:")
    fwd_err = (O_triton_hsa - O_ref.float()).abs().max().item()
    print(f"前向最大误差: {fwd_err:.6e}")

    # backward
    O_triton_hsa_padded.backward(grad_output_hsa)

    DQ_triton = Q_hsa.grad[:, :, :HQ, :].clone() if Q_hsa.grad is not None else None
    DK_triton = K.grad.clone() if K.grad is not None else None
    DV_triton = V.grad.clone() if V.grad is not None else None
    DW_triton_blocks = weights_blocks_hsa.grad.clone() if weights_blocks_hsa.grad is not None else None

    DW_ref_blocks = DW_ref.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :]

    compute_grad_diff(DQ_triton, DQ_ref, "DQ")
    compute_grad_diff(DK_triton, DK_ref, "DK")
    compute_grad_diff(DV_triton, DV_ref, "DV")
    DW_triton_blocks = weights_blocks_hsa.grad.clone()
    DW_triton_blocks[invalid_mask_blocks] = 0  # 对无效位置做 mask
    compute_grad_diff(DW_triton_blocks, DW_ref_blocks, "DW")


def main_block_M_inverse_latency():
    """
    对比 tilelang HSA_block_M_inverse 和 triton HSA 的 FWD、BWD、以及综合 (FWD+BWD) 延迟
    """
    import torch
    import torch.nn.functional as F
    import time
    import math
    from einops import rearrange
    from hsa_fwd_bwd_triton import HSA

    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 128, 1024*4, 1, 8, 128, 8, 32
    # B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 8, 32
    block_M=2
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    G_padded = HQ_padded // H
    need_padding = (HQ_padded > HQ)
    scale = 1.0 / math.sqrt(D)

    print(f"Latency Config: B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block={block_size}, G={G}, HQ_padded={HQ_padded}")

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
    logits = torch.randn((B, SEQ_LEN, H, S), dtype=torch.float32, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, -1e9)  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1).requires_grad_(True)  # leaf tensor，非法位近似为 0
    

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
    for _ in range(num_warmup):
        O = HSA(Q_triton, K, V, W, block_indices_triton, sm_n=0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
        O.backward(grad_output_triton)

    # FWD only
    def triton_fwd():
        with torch.no_grad():
            O = HSA(Q_triton, K, V, W, block_indices_triton, sm_n=0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
    triton_fwd_ms = measure_time(triton_fwd)

    # FWD+BWD
    def triton_fwd_bwd():
        Q_triton.grad = K.grad = V.grad = W.grad = None
        O = HSA(Q_triton, K, V, W, block_indices_triton, sm_n=0, chunk_size=block_size, sm_scale=scale, reg_lamda=0.0, reg_C=0.0)
        O.backward(grad_output_triton)
    triton_total_ms = measure_time(triton_fwd_bwd)

    triton_bwd_ms = triton_total_ms - triton_fwd_ms

    # =========================================================
    # TileLang 部分
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_block_M_inverse(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_inverse(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_inverse(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # 输出结果
    # =========================================================
    print(f"[Triton]   FWD: {triton_fwd_ms:.3f} ms | BWD: {triton_bwd_ms:.3f} ms | Total: {triton_total_ms:.3f} ms")
    print(f"[TileLang] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")


from ops.hsa_fwd_bwd_single_tilelang import HSA_single

def main_block_M_within_tilelang_latency():
    """
    对比 tilelang HSA_block_M 和 HSA_single 的 FWD、BWD、以及综合 (FWD+BWD) 延迟
    """
    import torch
    import torch.nn.functional as F
    import time
    import math
    from einops import rearrange

    # ---------- 配置参数 ----------
    # B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 64, 8, 32
    B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 8, 32
    block_M=8
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    G_padded = HQ_padded // H
    need_padding = (HQ_padded > HQ)
    scale = 1.0 / math.sqrt(D)

    print(f"Latency Config: B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block={block_size}, G={G}, HQ_padded={HQ_padded}")

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
    logits = torch.randn((B, SEQ_LEN, H, S), dtype=torch.float32, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, -1e9)  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1).requires_grad_(True)  # leaf tensor，非法位近似为 0
    

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
    for _ in range(num_warmup):
        O = HSA_single(Q_triton, K, V, W, block_indices_triton, block_size=block_size, sm_scale=scale)
        O.backward(grad_output_triton)

    # FWD only
    def triton_fwd():
        with torch.no_grad():
            O = HSA_single(Q_triton, K, V, W, block_indices_triton, block_size=block_size, sm_scale=scale)
    triton_fwd_ms = measure_time(triton_fwd)

    # FWD+BWD
    def triton_fwd_bwd():
        Q_triton.grad = K.grad = V.grad = W.grad = None
        O = HSA_single(Q_triton, K, V, W, block_indices_triton, block_size=block_size, sm_scale=scale)
        O.backward(grad_output_triton)
    triton_total_ms = measure_time(triton_fwd_bwd)

    triton_bwd_ms = triton_total_ms - triton_fwd_ms

    # =========================================================
    # TileLang 部分
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_block_M_group(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_group(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_group(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # 输出结果
    # =========================================================
    print(f"[TileLang Single]  FWD: {triton_fwd_ms:.3f} ms | BWD: {triton_bwd_ms:.3f} ms | Total: {triton_total_ms:.3f} ms")
    print(f"[TileLang Block_M] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")
    
    
    
    
def main_block_M_inverse_within_tilelang_latency():
    """
    对比 tilelang HSA_block_M_inverse 和 HSA_single 的 FWD、BWD、以及综合 (FWD+BWD) 延迟
    """
    import torch
    import torch.nn.functional as F
    import time
    import math
    from einops import rearrange

    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 64, 8, 32
    # B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 8, 32
    block_M=8
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    G_padded = HQ_padded // H
    need_padding = (HQ_padded > HQ)
    scale = 1.0 / math.sqrt(D)

    print(f"Latency Config: B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block={block_size}, G={G}, HQ_padded={HQ_padded}")

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
    logits = torch.randn((B, SEQ_LEN, H, S), dtype=torch.float32, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, -1e9)  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1).requires_grad_(True)  # leaf tensor，非法位近似为 0
    

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
    for _ in range(num_warmup):
        O = HSA_single(Q_triton, K, V, W, block_indices_triton, block_size=block_size, sm_scale=scale)
        O.backward(grad_output_triton)

    # FWD only
    def triton_fwd():
        with torch.no_grad():
            O = HSA_single(Q_triton, K, V, W, block_indices_triton, block_size=block_size, sm_scale=scale)
    triton_fwd_ms = measure_time(triton_fwd)

    # FWD+BWD
    def triton_fwd_bwd():
        Q_triton.grad = K.grad = V.grad = W.grad = None
        O = HSA_single(Q_triton, K, V, W, block_indices_triton, block_size=block_size, sm_scale=scale)
        O.backward(grad_output_triton)
    triton_total_ms = measure_time(triton_fwd_bwd)

    triton_bwd_ms = triton_total_ms - triton_fwd_ms

    # =========================================================
    # TileLang 部分
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_block_M_inverse(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_inverse(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_inverse(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # 输出结果
    # =========================================================
    print(f"[Tilelang Single]          FWD: {triton_fwd_ms:.3f} ms | BWD: {triton_bwd_ms:.3f} ms | Total: {triton_total_ms:.3f} ms")
    print(f"[TileLang Block_M_Inverse] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")
    
    
    

def main_block_M_vs_single_latency():
    """
    对比三种 TileLang 实现的 FWD/BWD 延迟：
      - Baseline:   HSA_single
      - Block_M:    HSA_block_M
      - Block_M_Inv:HSA_block_M_inverse（仅 BWD 不同，FWD 与 Block_M 相同）

    FWD 基准: HSA_single FWD
    BWD 基准: HSA_single BWD
    """
    import torch
    import torch.nn.functional as F
    import math

    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 128, 4096, 1, 1, 128, 8, 64
    B, SEQ_LEN, H, HQ, D, S, block_size = 4, 4096, 1, 1, 128, 8, 64
    block_M = 32
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    need_padding = (HQ_padded > HQ)
    scale = 1.0 / math.sqrt(D)

    print(f"Latency Config: B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, "
          f"D={D}, S={S}, BS={block_size}, G={G}, HQ_padded={HQ_padded}")

    # ---------- 生成成对重叠的 block_indices ----------
    overlap_ratio = 0.1
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

    # ---------- 生成输入 ----------
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)

    logits = torch.randn((B, SEQ_LEN, H, S), dtype=torch.float32, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, -1e9)
    W = F.softmax(logits, dim=-1).requires_grad_(True)

    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    # single 版本的 indices 要求 >=0，所以把 -1 映射到 safe_block
    num_blocks = SEQ_LEN // block_size
    safe_block = num_blocks - 1
    block_indices_single = block_indices.clone()
    block_indices_single[block_indices_single < 0] = safe_block

    # ---------- padding（baseline HSA_single 用到 HQ_padded 时才需要） ----------
    if need_padding:
        pad_heads = HQ_padded - HQ
        Q_single = torch.cat(
            [Q, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2
        ).detach().clone().requires_grad_(True)
        grad_output_single = torch.cat(
            [grad_output, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2
        )
    else:
        Q_single = Q.detach().clone().requires_grad_(True)
        grad_output_single = grad_output

    # Block_M / Inverse 使用原始 HQ 形状（两者 FWD 相同）
    Q_blockM = Q.detach().clone().requires_grad_(True)
    Q_blockM_inv = Q.detach().clone().requires_grad_(True)
    grad_output_tile = grad_output

    num_warmup = 50
    num_iters = 100

    def measure_time(func, *args, **kwargs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters  # ms

    # =========================================================
    # Baseline: HSA_single
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_single(Q_single, K, V, W, block_indices_single,
                       block_size=block_size, sm_scale=scale)
        O.backward(grad_output_single)

    def single_fwd():
        with torch.no_grad():
            _ = HSA_single(Q_single, K, V, W, block_indices_single,
                           block_size=block_size, sm_scale=scale)

    def single_fwd_bwd():
        Q_single.grad = K.grad = V.grad = W.grad = None
        O = HSA_single(Q_single, K, V, W, block_indices_single,
                       block_size=block_size, sm_scale=scale)
        O.backward(grad_output_single)

    single_fwd_ms = measure_time(single_fwd)
    single_total_ms = measure_time(single_fwd_bwd)
    single_bwd_ms = single_total_ms - single_fwd_ms

    # =========================================================
    # HSA_block_M（FWD+BWD）
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_block_M_group(Q_blockM, K, V, W, block_indices,
                        block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)

    def blockM_fwd():
        with torch.no_grad():
            _ = HSA_block_M_group(Q_blockM, K, V, W, block_indices,
                            block_size=block_size, sm_scale=scale, block_M=block_M)

    def blockM_fwd_bwd():
        Q_blockM.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_group(Q_blockM, K, V, W, block_indices,
                        block_size=block_size, sm_scale=scale, block_M=block_M)
        O.backward(grad_output_tile)

    blockM_fwd_ms = measure_time(blockM_fwd)
    blockM_total_ms = measure_time(blockM_fwd_bwd)
    blockM_bwd_ms = blockM_total_ms - blockM_fwd_ms

    # =========================================================
    # HSA_block_M_inverse（只比较 BWD；FWD 与 HSA_block_M 相同）
    # =========================================================
    # for _ in range(num_warmup):
    #     O = HSA_block_M_inverse(Q_blockM_inv, K, V, W, block_indices,
    #                             block_size=block_size, sm_scale=scale, block_M=block_M)
    #     O.backward(grad_output_tile)

    # def blockM_inv_fwd_bwd():
    #     Q_blockM_inv.grad = K.grad = V.grad = W.grad = None
    #     O = HSA_block_M_inverse(Q_blockM_inv, K, V, W, block_indices,
    #                             block_size=block_size, sm_scale=scale, block_M=block_M)
    #     O.backward(grad_output_tile)

    # # 不单独计 FWD，直接测 FWD+BWD 总时间
    # blockM_inv_total_ms = measure_time(blockM_inv_fwd_bwd)
    # # 这里无法把 FWD/BWD 拆开，因为 FWD kernel 与 HSA_block_M 完全相同；可以用 blockM_fwd_ms 近似 FWD，
    # # 于是 BWD 时间近似为：
    # blockM_inv_bwd_ms = blockM_inv_total_ms - blockM_fwd_ms

    # =========================================================
    # 打印结果（以 HSA_single 作为 baseline）
    # =========================================================
    print("\n=== FWD Latency (baseline: HSA_single) ===")
    print(f"[HSA_single]   FWD: {single_fwd_ms:.3f} ms")
    print(f"[HSA_block_M]  FWD: {blockM_fwd_ms:.3f} ms")

    print("\n=== BWD Latency (baseline: HSA_single) ===")
    print(f"[HSA_single]           BWD: {single_bwd_ms:.3f} ms")
    print(f"[HSA_block_M]          BWD: {blockM_bwd_ms:.3f} ms")
    # print(f"[HSA_block_M_inverse]  BWD: {blockM_inv_bwd_ms:.3f} ms")

    print("\n=== Total (FWD+BWD) ===")
    print(f"[HSA_single]          Total: {single_total_ms:.3f} ms")
    print(f"[HSA_block_M]         Total: {blockM_total_ms:.3f} ms")
    # print(f"[HSA_block_M_inverse] Total: {blockM_inv_total_ms:.3f} ms")

    

if __name__ == "__main__":
    ### 验证 HSA_block_M (TileLang) 的正确性
    main_block_M_correctness()
    
    ### 对比 HSA_block_M (TileLang) 与 Triton HSA 的延迟
    # main_block_M_latency()
    
    ### 验证 HSA_block_M_inverse (TileLang) 的正确性
    # main_block_M_inverse_correctness()
    
    ### 对比 HSA_block_M_inverse (TileLang) 与 Triton HSA 的延迟
    # main_block_M_inverse_latency()
    
    ### 对比 HSA_block_M (TileLang) 与 HSA_single (TileLang) 的延迟
    # main_block_M_within_tilelang_latency()
    
    ### 对比 HSA_block_M_inverse (TileLang) 与 HSA_single (TileLang) 的延迟
    # main_block_M_inverse_within_tilelang_latency()
    
    ### 对比 HSA_single, HSA_block_M, 和 HSA_block_M_inverse 三者的延迟
    # main_block_M_vs_single_latency()

    