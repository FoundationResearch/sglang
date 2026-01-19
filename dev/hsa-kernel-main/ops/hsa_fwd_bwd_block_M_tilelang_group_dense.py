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
    参考实现:
    - q: (B, L, HQ, D)
    - k, v: (B, L, H, D)
    - weights: (B, L, H, K)  <-- [Modified] Shared across group, per-token
    - indices: (B, q_blocks, H, K)
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

    idx_flat = rearrange(safe_indices, 'B Bq h K -> B (Bq K) h').unsqueeze(2).unsqueeze(-1)
    idx_flat = idx_flat.expand(-1, -1, chunk_size, -1, D)
    idx_flat = idx_flat.long()  
    gather_k = k_chunks.gather(dim=1, index=idx_flat)
    gather_v = v_chunks.gather(dim=1, index=idx_flat)

    gather_k = rearrange(gather_k, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)
    gather_v = rearrange(gather_v, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)

    k_ = torch.repeat_interleave(gather_k, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)
    v_ = torch.repeat_interleave(gather_v, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)

    q_chunked = rearrange(q, 'B (Bq X) hq d -> B Bq X hq d', X=block_q)

    # qk: (B, Bq, X, S, K, HQ)
    qk = torch.einsum('b q x h d, b q s k h d -> b q x s k h', q_chunked.float(), k_.float())
    qk = qk * float(sm_scale)
    
    # Causal Mask Logic
    q_indices = torch.arange(L, device=device).view(1, q_blocks, block_q, 1, 1, 1)
    q_real_blk_ids = q_indices // chunk_size
    k_blk_ids = rearrange(indices_q, 'b q h k -> b q 1 1 k h')
    k_blk_ids = torch.repeat_interleave(k_blk_ids, repeats=G, dim=-1)
    
    mask = k_blk_ids < q_real_blk_ids
    qk = qk.masked_fill(~mask, float("-inf"))
    
    if mask_last_token:
        qk[:, :, :, -1, :, :] = float("-inf")

    p = torch.softmax(qk, dim=3)
    p = torch.nan_to_num(p, nan=0.0)

    # o_k: (B, Bq, X, K, HQ, D)
    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    # [Modified] Process weights: (B, L, H, K) -> (B, Bq, X, HQ, K)
    w_chunked = rearrange(weights, 'b (q x) h k -> b q x h k', x=block_q)
    w_exp = torch.repeat_interleave(w_chunked, dim=-2, repeats=G).float()
    
    # [Modified] Apply valid mask to expanded weights
    # valid_mask is (B, Bq, H, K), expand to (B, Bq, X, HQ, K)
    v_mask_exp = valid_mask.unsqueeze(2).expand(-1, -1, block_q, -1, -1)
    v_mask_exp = torch.repeat_interleave(v_mask_exp, dim=-2, repeats=G)
    w_exp = w_exp.masked_fill(~v_mask_exp, 0.0)

    # [Modified] Final einsum: o_k (b q x k h d) * w_exp (b q x h k) -> o_ref (b q x h d)
    o_ref = torch.einsum('b q x k h d, b q x h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'b q x h d -> b (q x) h d')
    return o_ref.to(torch.float32)



def make_dq_layout_hsa(DQ):
    return T.Layout(DQ.shape,
        lambda b, l, h, d: [
            b,
            h,                            # h 放在外层，因为它是 block 内常数
            l // 8,                       # l 参与 swizzle
            d // 8,                       # d 参与 swizzle
            (d % 2),
            4 * (l % 8) + (d % 8) // 2    # 混合 l 和 d
        ])

@tilelang.jit(
    out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def hsa_bwd_postprocess(batch, q_len, heads, head_dim):
    shape = [batch, q_len, heads, head_dim]
    accum_dtype = "float"
    dtype = "bfloat16"
    blk = 64

    @T.prim_func
    def hsa_post(
            dQ_swizzled: T.Tensor(shape, accum_dtype),
            dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(q_len, blk), heads, batch, threads=32) as (bx, by, bz):
            i_b = bz
            
            T.annotate_layout({dQ_swizzled: make_dq_layout_hsa(dQ_swizzled)})
            
            T.copy(
                dQ_swizzled[i_b, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[i_b, bx * blk:(bx + 1) * blk, by, :],
            )
    return hsa_post




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
                                          selected_blocks=16,num_weights=None, block_M = 64, mask_last_token=False, dtype = "bfloat16", accum_dtype = "float", num_threads = 128):
    
    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504
    else:
        scale = scale * 1.44269504

    # [Modified] Grid setup uses total query heads
    q_shape = [batch, q_len, heads, head_dim]
    
    # KV shape uses kv_heads
    head_kv = heads // groups
    kv_shape = [batch, kv_len, head_kv, head_dim]
    
    num_kv_blocks = tilelang.cdiv(kv_len, block_size)
    # weight_shape = [batch, q_len, head_kv, num_kv_blocks]
    if num_weights is None:
        num_weights = num_kv_blocks
    weight_shape = [batch, q_len, head_kv, num_weights]
    
    # dtype = "bfloat16"
    # accum_dtype = "float"
    
    # [Modified] block_M is now independent of groups
    if block_M is None or block_M <= 0:
        block_M = 64 
    # block_M = 64
    print("Using block_M =", block_M, "for fwd_block_M kernel (Head-Parallel)")

    BS = block_size
    BK = BV = min(128, tilelang.math.next_power_of_2(head_dim))
    
    num_stages = 2
    # threads = 128

    @T.prim_func
    def hsa_block_M(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            Output: T.Tensor(q_shape, dtype),
    ):
        # [Modified] Grid Z = batch * heads (One block per Query Head)
        with T.Kernel(tilelang.cdiv(q_len, block_M), batch * heads, threads=num_threads) as (bx, bz):
            # Shared Memory (No groups dimension)
            Q_shared = T.alloc_shared([block_M, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([block_M, BV], dtype)

            acc_s = T.alloc_fragment([block_M, BS], accum_dtype)
            acc_o = T.alloc_fragment([block_M, BV], accum_dtype)
            
            P_shared = T.alloc_shared([block_M, BS], dtype)
            W_curr_shared = T.alloc_fragment([block_M], dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)

            i_t_base_idx, i_bh = bx, bz
            
            # [Modified] Decode Batch and Query Head Index
            i_b = i_bh // heads
            i_h = i_bh % heads
            
            # [Modified] Calculate corresponding KV Head Index
            i_h_kv = i_h // groups

            base_t = i_t_base_idx * block_M
            q_blk_idx = base_t // BS

            # 1. Load Q (for current Query Head)
            T.copy(Q[i_b, base_t:base_t + block_M, i_h, :], Q_shared)

            T.fill(acc_o, 0)
            T.sync_threads()

            # Loop over KV blocks
            for k_blk_idx in T.Pipelined(q_blk_idx, num_stages=num_stages):
                
                i_s = k_blk_idx * BS
                
                # 2. Load K, V (from corresponding KV Head)
                T.copy(K[i_b, i_s:i_s + BS, i_h_kv, :], K_shared)
                T.copy(V[i_b, i_s:i_s + BS, i_h_kv, :], V_shared)
                
                T.copy(W[i_b, base_t:base_t + block_M, i_h_kv, k_blk_idx], W_curr_shared)

                # 4. GEMM QK
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                # 5. Softmax & Apply Weights
                for r, c in T.Parallel(block_M, BS):
                    acc_s[r, c] = T.if_then_else(c == BS - 1 and enable_last_token_mask, -T.infinity(accum_dtype), acc_s[r, c])

                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                
                for r, c in T.Parallel(block_M, BS):
                    acc_s[r, c] = T.exp2(acc_s[r, c] * scale - scores_max[r] * scale)
                
                T.fill(scores_sum, 0.0)
                T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)
                
                for r, c in T.Parallel(block_M, BS):
                    # Apply HSA Weight
                    acc_s[r, c] = acc_s[r, c] * W_curr_shared[r] / scores_sum[r]

                # 6. GEMM PV
                T.copy(acc_s, P_shared)
                T.gemm(P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)
            
            T.copy(O_shared, Output[i_b, base_t : base_t + block_M, i_h, :])

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
    scale=None, block_size=64, groups=16, selected_blocks=16,num_weights=None,
    block_M = 0, mask_last_token=False, dtype="bfloat16", accum_dtype="float", num_threads = 128
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
    
    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    do_shape = [batch, q_len, heads, head_dim]

    dq_shape = [batch, q_len, heads, head_dim]
    
    # [Modified] DK/DV shape: Add groups dimension for split accumulation
    # Shape: [NV, groups, batch, kv_len, heads_kv, head_dim]
    dk_shape = [groups, batch, kv_len, heads_kv, head_dim]
    dv_shape = [groups, batch, kv_len, heads_kv, head_dim]
    
    # weight_shape = [batch, q_len,  heads_kv, NS_kv]
    # dw_shape = [groups, batch, q_len, heads_kv, NS_kv]
    if num_weights is None:
        num_weights = NS_kv
        
    weight_shape = [batch, q_len,  heads_kv, num_weights]
    dw_shape = [groups, batch, q_len, heads_kv, num_weights]
    
    # [Modified] block_M is independent of groups
    if block_M is None or block_M <= 0:
        block_M = 64
    # block_M = 64
    print("Using block_M =", block_M, "for bwd_block_M kernel (Head-Parallel)")
    
    # [Modified] M_G is just block_M now
    M = block_M
    NP = tilelang.cdiv(q_len, M)

    # num_threads = 128
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
    ):
        # [Modified] Grid Z = batch * heads (Query Heads)
        with T.Kernel(NS_kv, B * heads, threads=num_threads) as (i_s, i_bh):
            # [Modified] Decode indices
            i_b = i_bh // heads
            i_h = i_bh % heads
            i_h_kv = i_h // groups
            g_idx = i_h % groups # Index within the group
            
            i_s_global = i_s * BS

            # === Shared Memory Optimization ===
            # [Modified] Sizes are [M, BS] instead of [M*G, BS]\
            S_buf = T.alloc_shared([M, BS], dtype)
            dO_buf = T.alloc_shared([M, BV], dtype)
            
            P_shared = S_buf
            dS_shared = S_buf
            
            dO_shared =  dO_buf
            dO_weighted_shared = dO_buf
            
            Q_shared = T.alloc_shared([M, BK], dtype)

            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            
            dK_shared = T.alloc_shared([BS, BK], dtype)
            dV_shared = T.alloc_shared([BS, BV], dtype)

            T_raw_frag = T.alloc_fragment([M, BV], accum_dtype)
            dV_PdO_frag = T.alloc_fragment([M, BS], accum_dtype)
            dS_frag = T.alloc_fragment([M, BS], dtype)
            dV_accum = T.alloc_fragment([BS, BV], accum_dtype)
            dK_accum = T.alloc_fragment([BS, BK], accum_dtype)
            dQ_local = T.alloc_fragment([M, BK], accum_dtype)
            delta_rows = T.alloc_fragment([M], accum_dtype)
            
            acc_s_tmp = T.alloc_fragment([M, BS], accum_dtype)
            scores_max = T.alloc_fragment([M], accum_dtype)
            scores_sum = T.alloc_fragment([M], accum_dtype)
            
            dw_row_sum_frag = T.alloc_fragment([M], accum_dtype)
            dw_row_sum_shared = T.alloc_shared([M], accum_dtype)
            
            W_local = T.alloc_shared([M], dtype)
            
            # Load K, V (using i_h_kv)
            T.copy(K[i_b, i_s_global:i_s_global + BS, i_h_kv, :], K_shared)
            T.copy(V[i_b, i_s_global:i_s_global + BS, i_h_kv,:], V_shared)
            T.fill(dK_accum, 0)
            T.fill(dV_accum, 0)

            T.annotate_layout({
                DQ: make_dq_layout_hsa(DQ),
            })
            ip_start = T.floordiv(i_s * BS, M) + 1
            for ip in T.Pipelined(ip_start, NP, num_stages=num_stages):
                base_t = ip * M
                q_blk_idx = base_t // BS
                
                if q_blk_idx > i_s:
                    # [Modified] Load Q, DO using i_h (Slicing)
                    T.copy(Q[i_b, base_t:base_t+M, i_h, :], Q_shared)
                    T.copy(DO[i_b, base_t:base_t+M, i_h, :], dO_shared)
                    
                    T.clear(acc_s_tmp)
                    T.gemm(Q_shared, K_shared, acc_s_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # 2. Apply Masking
                    for i, s in T.Parallel(M, BS):
                        tq = base_t + i
                        acc_s_tmp[i, s] = T.if_then_else(
                            (tq >= q_len) or (enable_last_token_mask & (s == BS - 1)),
                            -T.infinity(accum_dtype),
                            acc_s_tmp[i, s]
                        )

                    # 3. Reduction Max
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s_tmp, scores_max, dim=1, clear=True)

                    # 4. Exp with NaN Guard
                    for i, s in T.Parallel(M, BS):
                        tq = base_t + i
                        mx = scores_max[i]
                        acc_s_tmp[i, s] = T.if_then_else(
                            (tq < q_len) & (mx != -T.infinity(accum_dtype)),
                            T.exp2(acc_s_tmp[i, s] * scale_log2 - mx * scale_log2),
                            0.0
                        )
                    
                    # 5. Sum
                    T.reduce_sum(acc_s_tmp, scores_sum, dim=1, clear=True)

                    # 6. Normalize and Write Back
                    for i, s in T.Parallel(M, BS):
                        # P_shared[i, s] = T.if_then_else(scores_sum[i] > 0,
                        #                                 acc_s_tmp[i, s] / scores_sum[i],
                        #                                 0.0)
                        p_val = T.if_then_else(scores_sum[i] > 0,
                                                        acc_s_tmp[i, s] / scores_sum[i],
                                                        0.0)
                        # 更新寄存器 (用于后续 DW 计算)
                        acc_s_tmp[i, s] = p_val
                        # 写入 Shared Memory (用于后续 dV GEMM 和 dS 计算)
                        P_shared[i, s] = p_val

                    # === dV & dW Optimization ===
                    
                    # [Step 1] 提前加载 W
                    T.copy(W[i_b, base_t:base_t + M, i_h_kv, i_s], W_local)

                    # [Step 2] 计算 Z = dO @ V.T
                    # 此时 dO_shared 存储的是原始 dO
                    T.clear(dV_PdO_frag)
                    T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # [Step 3] 计算 dW = Sum(P * Z)
                    # 复用 acc_s_tmp (之前存 logits，现在空闲)
                    for r, s in T.Parallel(M, BS):
                        acc_s_tmp[r, s] = acc_s_tmp[r, s] * dV_PdO_frag[r, s]
                    
                    T.reduce_sum(acc_s_tmp, dw_row_sum_frag, dim=1, clear=True)
                    T.copy(dw_row_sum_frag, dw_row_sum_shared)
                    
                    T.copy(dw_row_sum_shared, DW[g_idx, i_b, base_t:base_t + M, i_h_kv, i_s])

                    # [Step 4] 准备 dV 和 dS 的输入
                    # 更新 dO_shared 为 dO_weighted (原地更新)
                    for row_idx, v in T.Parallel(M, BV):
                        dO_weighted_shared[row_idx, v] = W_local[row_idx] * dO_shared[row_idx, v]
                    
                    # 更新 Z (dV_PdO_frag) 为 W * Z
                    for g_row, s in T.Parallel(M, BS):
                        dV_PdO_frag[g_row, s] = dV_PdO_frag[g_row, s] * W_local[g_row]

                    # [Step 5] 计算 dV
                    # dV += P.T @ dO_weighted
                    T.gemm(P_shared, dO_weighted_shared, dV_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    # [Step 6] 计算 dS
                    # dS = P * (W*Z - delta)
                    # W*Z 已经在 dV_PdO_frag 中
                    # 计算 delta = Sum(P * (W*Z))
                    # 复用 acc_s_tmp
                    for g_row, s in T.Parallel(M, BS):
                        acc_s_tmp[g_row, s] = P_shared[g_row, s] * dV_PdO_frag[g_row, s]
                    T.reduce_sum(acc_s_tmp, delta_rows, dim=1, clear=True)
                    
                    for g_row, s in T.Parallel(M, BS):
                        dS_frag[g_row, s] = sm_scale * (acc_s_tmp[g_row, s] - P_shared[g_row, s] * delta_rows[g_row])
                    
                    T.copy(dS_frag, dS_shared)  
                    
                    T.gemm(dS_shared, Q_shared, dK_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    
                    T.clear(dQ_local)
                    T.gemm(dS_shared, K_shared, dQ_local, policy=T.GemmWarpPolicy.FullRow)
                    
                    # [Modified] Atomic Add DQ using i_h
                    for i, k in T.Parallel(M, BK):
                        tq = base_t + i
                        if tq < q_len:
                            T.atomic_add(DQ[i_b, tq, i_h, k], dQ_local[i, k])

            T.copy(dK_accum, dK_shared)
            T.copy(dV_accum, dV_shared)
            
            # [Modified] Store DK/DV with group index offset
            # DK shape: [NV, groups, batch, kv_len, heads_kv, head_dim]
            T.copy(dK_shared, DK[g_idx, i_b, i_s_global:i_s_global + BS, i_h_kv, :])
            T.copy(dV_shared, DV[g_idx, i_b, i_s_global:i_s_global + BS, i_h_kv, :])

    return hsa_bwd_dqkv_block_M


from ops.rope_tilelang_fp32 import rope_rotary_pos_emb
class _hsa_block_M_attention_group_dense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, W, block_size, sm_scale, block_M, mask_last_token, 
                dtype, accum_dtype, num_threads,
                enable_inverse_rope, cos, sin):
        # Q: [B, L, HQ, D]
        # K, V: [B, L, H, D]
        # W: [B, L, H, S]
        
        B, SEQ_LEN, HQ, D = Q.shape
        H = K.shape[2]
        groups = HQ // H
        
        num_weights = W.shape[-1]
        
        # 确保 block_M 是合理的
        if block_M is None or block_M <= 0:
            block_M = 64

        if enable_inverse_rope:
            assert cos.shape[1] == Q.shape[1], f"cos seq_len {cos.shape[1]} != q seq_len {Q.shape[1]}"
            Q_in, K_in = rope_rotary_pos_emb(Q, K, cos, -sin)
        else:
            Q_in, K_in = Q, K
            
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.block_M = block_M
        ctx.mask_last_token = mask_last_token
        ctx.groups = groups
        ctx.enable_inverse_rope = enable_inverse_rope
        ctx.dtype = dtype
        ctx.accum_dtype = accum_dtype
        ctx.num_threads = num_threads
        ctx.num_weights = num_weights # Save num_weights
        
        # 前向计算
        O = hierarchical_sparse_attention_block_M(
            batch=B, heads=HQ, q_len=SEQ_LEN, kv_len=SEQ_LEN, head_dim=D,
            scale=sm_scale, block_size=block_size, groups=groups,
            selected_blocks=0, num_weights=num_weights, # Pass num_weights
            block_M=block_M, mask_last_token=mask_last_token,
            dtype=dtype, accum_dtype=accum_dtype,num_threads=num_threads,
        )(Q_in, K_in, V, W)
        
        ctx.save_for_backward(Q, K, V, W, O, cos, sin)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, W, O, cos, sin = ctx.saved_tensors
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        block_M = ctx.block_M
        mask_last_token = ctx.mask_last_token
        groups = ctx.groups
        enable_inverse_rope = ctx.enable_inverse_rope
        dtype = ctx.dtype
        accum_dtype = ctx.accum_dtype
        num_threads = ctx.num_threads
        num_weights = ctx.num_weights # load num_weights
        
        B, SEQ_LEN, HQ, D = Q.shape
        H = K.shape[2]

        if enable_inverse_rope:
            Q_in, K_in = rope_rotary_pos_emb(Q, K, cos, -sin)
        else:
            Q_in, K_in = Q, K
        
        # Shapes for intermediate gradients
        # NV = ceil(D / 128)
        BV = min(128, 1 << (D - 1).bit_length())
        NS_kv = (SEQ_LEN + block_size - 1) // block_size
        
        dq_shape = [B, SEQ_LEN, HQ, D]
        dk_shape = [groups, B, SEQ_LEN, H, D]
        dv_shape = [groups, B, SEQ_LEN, H, D]
        # dw_shape = [groups, B, SEQ_LEN, H, NS_kv]
        dw_shape = [groups, B, SEQ_LEN, H, num_weights] # use num_weights
        
        # [Fix] Explicitly allocate and ZERO-INITIALIZE output tensors
        # DQ uses atomic_add, so it MUST be zeroed.
        DQ = torch.zeros(dq_shape, dtype=torch.float32, device=Q.device)
        
        # DW is sparsely written (masked blocks skipped), so it MUST be zeroed.
        DW = torch.zeros(dw_shape, dtype=W.dtype, device=Q.device)
        
        # DK/DV are fully overwritten by the grid, but using empty/zeros is safer.
        DK = torch.empty(dk_shape, dtype=K.dtype, device=Q.device)
        DV = torch.empty(dv_shape, dtype=V.dtype, device=Q.device)
        
        # Call Kernel with explicit output tensors
        hierarchical_sparse_attention_bwd_dqkv_block_M(
            batch=B, heads=HQ, q_len=SEQ_LEN, kv_len=SEQ_LEN, head_dim=D,
            scale=sm_scale, block_size=block_size, groups=groups,
            selected_blocks=0, num_weights=num_weights, # Pass num_weights
            block_M=block_M, mask_last_token=mask_last_token,
            dtype=dtype, accum_dtype=accum_dtype, num_threads=num_threads,
        )(Q_in, K_in, V, W, grad_output, DQ, DK, DV, DW)
        
        post_kernel = hsa_bwd_postprocess(B, SEQ_LEN, HQ, D)
        DQ = post_kernel(DQ)
        
        # Reduce gradients
        # DQ: [B, L, HQ, D] -> sum(0)
        DQ = DQ.to(Q.dtype)
        
        
        # DK: [groups, B, L, H, D] -> sum(0).sum(0)
        DK = DK.sum(dim=0).to(K.dtype)
        
        # DV: [groups, B, L, H, D] -> sum(0)
        DV = DV.sum(dim=0).to(V.dtype)
        DW = DW.sum(dim=0).to(W.dtype)

        if enable_inverse_rope:
            DQ, DK = rope_rotary_pos_emb(DQ, DK, cos, sin)
        
        return DQ, DK, DV, DW, None, None, None, None, None, None, None, None, None, None

def HSA_block_M_group_dense(Q, K, V, W, block_size=64, sm_scale=None, block_M=64, mask_last_token=False, 
                            dtype="bfloat16", accum_dtype="float", num_threads=128,
                            enable_inverse_rope=False, cos=None, sin=None):
    if enable_inverse_rope and (cos is None or sin is None):
        raise ValueError("cos and sin cannot be None when enable_inverse_rope is True")
    return _hsa_block_M_attention_group_dense.apply(Q, K, V, W, block_size, sm_scale, block_M, mask_last_token, 
                                                    dtype, accum_dtype,num_threads,
                                                    enable_inverse_rope, cos, sin)




def main_block_M_correctness():
    """
    检验 HSA_pair 封装类的前向和反向传播数值正确性
    与 hsa_torch_ref 进行对比 (Dense Mode)
    """
    import math
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    
    # ---------- 辅助函数 ----------
    def print_max_err_compare(name, tensor_hsa, tensor_ref):
        if tensor_hsa is None or tensor_ref is None:
            print(f"\n[{name} Error Analysis]: Tensor is None")
            return

        hsa_f = tensor_hsa.float()
        ref_f = tensor_ref.float()
        diff_abs = (hsa_f - ref_f).abs()
        
        # 1. 绝对误差最大位置
        max_abs_err = diff_abs.max().item()
        # torch.where 返回的是 tuple of tensors，取第一个元素并转为 item
        # 注意：如果有多个最大值，只取第一个
        indices = torch.where(diff_abs == max_abs_err)
        idx_abs = tuple(idx[0].item() for idx in indices)
        
        # 2. 相对误差最大位置
        diff_rel = diff_abs / (ref_f.abs() + 1e-6)
        max_rel_err = diff_rel.max().item()
        indices_rel = torch.where(diff_rel == max_rel_err)
        idx_rel = tuple(idx[0].item() for idx in indices_rel)
        
        print(f"\n[{name} Error Analysis]:")
        print(f"  -> Max Absolute Error: {max_abs_err:.6e} at {idx_abs}")
        print(f"     HSA Val: {hsa_f[idx_abs].item():.10f} | Ref Val: {ref_f[idx_abs].item():.10f}")
        
        print(f"  -> Max Relative Error: {max_rel_err:.6e} at {idx_rel}")
        print(f"     HSA Val: {hsa_f[idx_rel].item():.10f} | Ref Val: {ref_f[idx_rel].item():.10f}")

    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, block_size = 1, 512, 1, 8, 128, 64
    dtype = torch.bfloat16
    device = "cuda"
    block_M=64
    mask_last_token=True
    G = HQ // H
    scale = 1.0 / math.sqrt(D)
    
    num_kv_blocks = SEQ_LEN // block_size
    
    print(f"Correctness Config (Dense): Batch={B}, SeqLen={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, G={G}, NumKVBlocks={num_kv_blocks}, BlockSize={block_size}")
    
    # ---------- 生成测试数据 ----------
    torch.manual_seed(42)
    
    # 创建 requires_grad=True 的输入（这些是 leaf tensors）
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成 Dense 权重: [B, SEQ_LEN, HQ, num_kv_blocks]
    logits = torch.randn((B, SEQ_LEN, H, num_kv_blocks), dtype=dtype, device=device)
    
    # Causal Mask for Weights
    q_indices = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)
    q_blk_indices = q_indices // block_size
    
    # Valid condition: k_blk < q_blk (Strictly Past)
    weight_mask = k_blk_indices < q_blk_indices
    
    logits = logits.masked_fill(~weight_mask, float('-inf'))  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1)  # leaf tensor
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)
    
    # 构造全量的 block_indices 用于 torch ref
    block_indices_dense = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    block_indices_dense = block_indices_dense.view(1, 1, 1, num_kv_blocks).expand(B, SEQ_LEN, H, num_kv_blocks)
    
    # 用于反向传播的梯度
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ========== 测试前向传播 ==========
    
    # HSA_pair 前向 (Dense)
    O_hsa = HSA_block_M_group_dense(Q, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
    
    # Torch reference 前向
    O_ref = hsa_torch_ref(
        Q.float().detach(), 
        K.float().detach(), 
        V.float().detach(), 
        W.detach(), 
        block_indices_dense, 
        chunk_size=block_size, 
        sm_scale=scale, 
        block_q=1,
        mask_last_token=mask_last_token
    )

    print("[Tilelang HSA_block_M Dense] vs [Torch Reference]:")
    print_max_err_compare("Forward Output", O_hsa, O_ref)


    # ========== 测试反向传播 ==========
    
    # ====== 先计算 Torch reference 反向 ======
    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None
    
    O_ref_bwd= hsa_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices_dense,
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
    O_hsa_bwd = HSA_block_M_group_dense(Q, K, V, W, 
                         block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
    O_hsa_bwd.backward(grad_output)
    
    # 获取 HSA_block_M 梯度
    DQ_hsa = Q.grad.clone()
    DK_hsa = K.grad.clone()
    DV_hsa = V.grad.clone()
    DW_hsa = W.grad.clone()
    
    # 比较梯度
    print_max_err_compare("DQ", DQ_hsa, DQ_ref)
    print_max_err_compare("DK", DK_hsa, DK_ref)
    print_max_err_compare("DV", DV_hsa, DV_ref)
    print_max_err_compare("DW", DW_hsa, DW_ref)



def main_block_M_latency():
    """
    测试 tilelang HSA_block_M (Dense) 的 FWD、BWD、以及综合 (FWD+BWD) 延迟
    """
    import torch
    import torch.nn.functional as F
    import time
    import math
    from einops import rearrange

    # ---------- 配置参数 ----------
    # Dense 模式计算量大，适当减小 Batch 或 SeqLen 以防 OOM 或运行过慢
    B, SEQ_LEN, H, HQ, D, block_size = 16, 4096, 1, 8, 128, 64
    block_M=64
    mask_last_token=True
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    scale = 1.0 / math.sqrt(D)
    
    num_kv_blocks = SEQ_LEN // block_size

    print(f"Latency Config (Dense): B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, NumKVBlocks={num_kv_blocks}, block={block_size}, G={G}, block_M={block_M}, mask_last_token={mask_last_token}")

    # 创建 requires_grad=True 的输入（这些是 leaf tensors）
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成 Dense 权重: [B, SEQ_LEN, HQ, num_kv_blocks]
    logits = torch.randn((B, SEQ_LEN, H, num_kv_blocks), dtype=torch.bfloat16, device=device)
    
    # Causal Mask for Weights
    q_indices = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)
    q_blk_indices = q_indices // block_size
    weight_mask = k_blk_indices < q_blk_indices
    
    logits = logits.masked_fill(~weight_mask, float('-inf'))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)
    

    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ---------- TileLang ----------
    Q_tile = Q.detach().clone().requires_grad_(True)
    grad_output_tile = grad_output

    num_warmup = 20
    num_iters = 50

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
    # TileLang 部分
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_block_M_group_dense(Q_tile, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_group_dense(Q_tile, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_group_dense(Q_tile, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # 输出结果
    # =========================================================
    print(f"[TileLang Dense] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")
    print()

def main_rope_correctness_dense():
    """
    验证 enable_inverse_rope 开启后的数值正确性 (Dense Mode)
    逻辑：RoPE(q_nope) -> HSA_Dense(enable_inverse_rope=True) -> 内部还原为 q_nope -> 与 Ref(q_nope) 对比
    """
    import torch
    import torch.nn.functional as F
    import math
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    # ---------- 配置参数 ----------
    B, L, H, HQ, D, block_size = 1, 512, 1, 8, 128, 64
    dtype = torch.bfloat16
    device = "cuda"
    block_M = 64
    scale = 1.0 / math.sqrt(D)
    num_kv_blocks = L // block_size
    
    print(f"\nRoPE Correctness Test (Dense): B={B}, L={L}, HQ={HQ}, D={D}, NumKVBlocks={num_kv_blocks}, block_size={block_size}")

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

    # 生成 Dense 权重: [B, L, HQ, num_kv_blocks]
    logits = torch.randn((B, L, H, num_kv_blocks), dtype=dtype, device=device)
    
    # Causal Mask for Weights
    q_indices = torch.arange(L, device=device).view(1, L, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)
    q_blk_indices = q_indices // block_size
    weight_mask = k_blk_indices < q_blk_indices
    
    logits = logits.masked_fill(~weight_mask, float('-inf'))
    weights_base = F.softmax(logits, dim=-1)
    weights_base = torch.nan_to_num(weights_base, 0.0).detach().requires_grad_(True)

    # 构造全量的 block_indices 用于 torch ref
    block_indices_dense = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    block_indices_dense = block_indices_dense.view(1, 1, 1, num_kv_blocks).expand(B, L, H, num_kv_blocks)

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
    o_hsa = HSA_block_M_group_dense(
        q_rope, k_rope, v_hsa, weights_hsa, 
        block_size=block_size, sm_scale=scale, block_M=block_M, 
        mask_last_token=True, enable_inverse_rope=True, cos=cos, sin=sin
    )

    # Torch Ref: 直接用 nope 数据
    o_ref = hsa_torch_ref(
        q_nope_ref.float(), k_nope_ref.float(), v_ref.float(), 
        weights_ref, block_indices_dense, 
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

    # 转换 Ref 梯度: Ref 算出的是 d(q_nope)，我们需要将其 RoPE 化以对比 d(q_rope)
    # 因为 d(Loss)/d(q_rope) = RoPE( d(Loss)/d(q_nope) )
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
    
    
import pytest

@pytest.mark.parametrize("B, SEQ_LEN, H, HQ, D, block_size", [
    (1, 512, 1, 8, 64, 64),
    (2, 1024, 2, 16, 128, 64),
])
def test_correctness_fp32(B, SEQ_LEN, H, HQ, D, block_size):
    device = "cuda"
    dtype = torch.float32
    scale = 1.0 / math.sqrt(D)
    # 关键点 1: 确保 block_M <= block_size 以对齐因果边界逻辑
    block_M = min(64, block_size) 
    mask_last_token = True
    torch.manual_seed(42)

    num_kv_blocks = SEQ_LEN // block_size

    # 1. 生成原始数据
    Q_raw = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    K_raw = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device)
    V_raw = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device)
    
    # [Modified] W shape uses H (KV heads), not HQ
    logits_raw = torch.randn((B, SEQ_LEN, H, num_kv_blocks), dtype=dtype, device=device)
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    # 2. 构造因果掩码并生成权重 W
    # 注意：Kernel 的因果逻辑是 k_blk < (q_token_idx // block_size)
    q_indices = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)
    
    # 模拟 Kernel 的分块因果边界：每个 block_M 窗口共享起始 token 的 block_id
    q_blk_indices = (q_indices // block_M * block_M) // block_size
    weight_mask = k_blk_indices < q_blk_indices
    
    logits_raw.masked_fill_(~weight_mask, float('-inf'))
    W_raw = torch.softmax(logits_raw, dim=-1)
    W_raw = torch.nan_to_num(W_raw, 0.0)

    # 3. 准备两组完全独立的输入 (隔离 HSA 和 Ref)
    Q_hsa = Q_raw.clone().detach().requires_grad_(True)
    K_hsa = K_raw.clone().detach().requires_grad_(True)
    V_hsa = V_raw.clone().detach().requires_grad_(True)
    W_hsa = W_raw.clone().detach().requires_grad_(True)

    Q_ref = Q_raw.clone().detach().requires_grad_(True)
    K_ref = K_raw.clone().detach().requires_grad_(True)
    V_ref = V_raw.clone().detach().requires_grad_(True)
    W_ref = W_raw.clone().detach().requires_grad_(True)

    # 4. 构造 Ref 所需的 block_indices
    block_indices_dense = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    block_indices_dense = block_indices_dense.view(1, 1, 1, num_kv_blocks).expand(B, SEQ_LEN, H, num_kv_blocks)

    # 5. 前向计算
    O_hsa = HSA_block_M_group_dense(
        Q_hsa, K_hsa, V_hsa, W_hsa, 
        block_size=block_size, sm_scale=scale, block_M=block_M, 
        mask_last_token=mask_last_token, dtype="float32", accum_dtype="float"
    )

    # 修改 Ref 的因果逻辑以匹配 Kernel 的分块行为
    O_ref = hsa_torch_ref(
        Q_ref, K_ref, V_ref, W_ref, block_indices_dense,
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=mask_last_token
    )

    # 6. 辅助校验函数
    def get_abs_err(x, y):
        return (x - y).flatten().abs().max().item()
    def get_err_ratio(x, y):
        err = (x - y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base + 1e-12)
    def assert_close(prefix, ref, tri, ratio=0.005):
        abs_err = get_abs_err(ref, tri)
        rel_ratio = get_err_ratio(ref, tri)
        msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio, msg

    # 7. 验证前向
    assert_close("FWD", O_ref, O_hsa, 0.005)

    # 8. 反向计算
    O_hsa.backward(grad_output)
    O_ref.backward(grad_output)

    # 9. 验证梯度
    assert_close("DQ", Q_ref.grad, Q_hsa.grad, 0.005)
    assert_close("DK", K_ref.grad, K_hsa.grad, 0.005)
    assert_close("DV", V_ref.grad, V_hsa.grad, 0.005)
    assert_close("DW", W_ref.grad, W_hsa.grad, 0.005)
    
    print(f"Test Passed: B={B}, L={SEQ_LEN}, D={D}, BS={block_size}")

if __name__ == "__main__":
    main_block_M_correctness()
    main_block_M_latency()
    # main_rope_correctness_dense()
    
    params_list = [
        (1, 1024, 1, 8, 64,32),
        (2, 1024, 1, 8, 64,32),
        (3, 512, 1, 8, 64,32),
        (4, 512, 1, 8, 64, 64),
        (5, 256, 1, 8, 64,64),
    ]
    for p in params_list:
        test_correctness_fp32(*p)
    
