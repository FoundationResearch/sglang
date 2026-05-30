import torch
import math
import logging
logging.getLogger("tilelang.jit.kernel").setLevel(logging.WARNING)
logging.getLogger("tilelang").setLevel(logging.WARNING)
import tilelang
from tilelang import language as T




from einops import rearrange
def hsa_torch_ref(q, k, v, weights, indices, *, chunk_size: int, sm_scale: float, block_q: int, mask_last_token: bool = False, window_size: int = -1):
    """
    参考实现（与 test_group_qa 一致的数学公式）:
    ...
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
    valid_L = N * chunk_size

    # 截断 k 和 v
    k_truncated = k[:, :valid_L, :, :]
    v_truncated = v[:, :valid_L, :, :]
    
    k_chunks = rearrange(k_truncated, 'B (N S) h d -> B N S h d', S=chunk_size)
    v_chunks = rearrange(v_truncated, 'B (N S) h d -> B N S h d', S=chunk_size)

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
    
    # [Modified] Causal Mask Logic for Reference
    # q_indices: (1, q_blocks, block_q, 1, 1, 1)
    q_indices = torch.arange(L, device=device).view(1, q_blocks, block_q, 1, 1, 1)
    q_real_blk_ids = q_indices // chunk_size
    
    # k_blk_ids: (B, q_blocks, 1, 1, K, 1)
    k_blk_ids = rearrange(indices_q, 'b q h k -> b q 1 1 k h')
    k_blk_ids = torch.repeat_interleave(k_blk_ids, repeats=G, dim=-1) # (B, q_blocks, 1, 1, K, HQ)
    
    
    # Mask: Past only (k < q)
    mask = k_blk_ids < q_real_blk_ids
    
    # Apply mask to qk
    qk = qk.masked_fill(~mask, float("-inf"))
    
    
    if window_size > 0:
        # Window Mask: Mask out chunk if it is within window_size relative to q
        # Valid condition: k_blk_ids < (q_indices - window_size + 1) // chunk_size
        threshold_blk_ids = (q_indices - window_size + 1).div(chunk_size, rounding_mode='floor')
        window_mask = k_blk_ids < threshold_blk_ids
        qk = qk.masked_fill(~window_mask, float("-inf"))
    
    if mask_last_token:
        qk[:, :, :, -1, :, :] = float("-inf")

    p = torch.softmax(qk, dim=3)
    p = torch.nan_to_num(p, nan=0.0)

    # o_k: (B, Bq, X, K, HQ, D)
    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    w_masked = weights.clone()
    valid_mask_expanded = torch.repeat_interleave(valid_mask, dim=-2, repeats=G)
    w_masked = w_masked.masked_fill(~valid_mask_expanded, 0)
    w_exp = w_masked.float() # (B, Bq, HQ, K)
    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'b q x h d -> b (q x) h d')
    return o_ref.to(torch.float32)



def make_dq_layout_hsa(DQ):
    return T.Layout(DQ.shape,
        lambda b, l, h, d: [
            b,
            h,
            l // 8,
            d // 8,
            d % 2,
            4 * (l % 8) + (d % 8) // 2,
        ])

@tilelang.jit(
    out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def hsa_bwd_postprocess(batch, q_len, q_len_padded, heads, head_dim):
    swizzled_shape = [batch, q_len_padded, heads, head_dim]
    output_shape = [batch, q_len, heads, head_dim]
    accum_dtype = "float"
    dtype = "bfloat16"
    blk = 64

    @T.prim_func
    def hsa_post(
            dQ_swizzled: T.Tensor(swizzled_shape, accum_dtype),
            dQ_out: T.Tensor(output_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(q_len, blk), heads, batch, threads=32) as (bx, by, bz):
            i_b = bz
            T.annotate_layout({dQ_swizzled: make_dq_layout_hsa(dQ_swizzled)})

            for r, d in T.Parallel(blk, head_dim):
                tq = bx * blk + r
                if tq < q_len:
                    dQ_out[i_b, tq, by, d] = dQ_swizzled[i_b, tq, by, d]
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
                                          selected_blocks=16, num_weights=None,  block_M = None, mask_last_token=True, 
                                          window_size=-1, dtype = "bfloat16", accum_dtype = "float", num_threads = None):
   
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
    
    num_kv_blocks = kv_len // block_size
    
    if num_weights is None:
        num_weights = num_kv_blocks
    # weight_shape = [batch, q_len, heads, num_kv_blocks]
    weight_shape = [batch, q_len, heads, num_weights]
    scores_lse_shape = [batch, q_len, heads, num_weights]
    
    # dtype = "bfloat16"
    # accum_dtype = "float"
    
    # [Modified] block_M is now independent of groups
    if block_M is None or block_M <= 0:
        block_M = 32
    # block_M = 64
    print("Using block_M =", block_M, "for fwd_block_M kernel (Head-Parallel)")

    BS = block_size
    BK = BV = min(128, tilelang.math.next_power_of_2(head_dim))
    
    num_stages = 2
    # threads = 128
    if num_threads is None:
        num_threads = 128

    @T.prim_func
    def hsa_block_M(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
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
            
            # Optimization for Window Mask: Reduce loop range
            pipeline_limit = T.alloc_var("int32")
            pipeline_limit = q_blk_idx
            if window_size > 0:
                # Max possible valid k block for any t in this block
                max_t = base_t + block_M - 1
                pipeline_limit  = T.floordiv(max_t - window_size + 1, BS)
                pipeline_limit = T.max(pipeline_limit, 0)

            # 1. Load Q (for current Query Head)
            for r, d in T.Parallel(block_M, BK):
                tq = base_t + r
                Q_shared[r, d] = T.if_then_else(tq < q_len, Q[i_b, tq, i_h, d], 0.0)

            T.fill(acc_o, 0)
            T.sync_threads()

            # Loop over KV blocks
            for k_blk_idx in T.Pipelined(pipeline_limit, num_stages=num_stages):
                
                i_s = k_blk_idx * BS
                
                # 2. Load K, V (from corresponding KV Head)
                T.copy(K[i_b, i_s:i_s + BS, i_h_kv, :], K_shared)
                T.copy(V[i_b, i_s:i_s + BS, i_h_kv, :], V_shared)
                
                for r in T.Parallel(block_M):
                    tq = base_t + r
                    W_curr_shared[r] = T.if_then_else(tq < q_len, W[i_b, tq, i_h, k_blk_idx], 0.0)

                # 4. GEMM QK
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                # 5. Softmax & Apply Weights
                for r, c in T.Parallel(block_M, BS):
                    acc_s[r, c] = T.if_then_else(c == BS - 1 and enable_last_token_mask, -T.infinity(accum_dtype), acc_s[r, c])

                if window_size > 0:  
                    for r, c in T.Parallel(block_M, BS):
                        acc_s[r, c] = T.if_then_else(
                            k_blk_idx >= T.floordiv(base_t + r - window_size + 1, BS),
                            -T.infinity(accum_dtype),
                            acc_s[r, c]
                        )
                
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                
                for r, c in T.Parallel(block_M, BS):
                    # acc_s[r, c] = T.exp2(acc_s[r, c] * scale - scores_max[r] * scale)
                    acc_s[r, c] = T.if_then_else(
                        scores_max[r] == -T.infinity(accum_dtype),
                        0.0,
                        T.exp2(acc_s[r, c] * scale - scores_max[r] * scale)
                    )
                T.fill(scores_sum, 0.0)
                T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)

                for r in T.Parallel(block_M):
                    tq = base_t + r
                    if tq < q_len:
                        ScoresLSE[i_b, tq, i_h, k_blk_idx] = T.if_then_else(
                            scores_sum[r] > 0,
                            scores_max[r] * scale + T.log(scores_sum[r]) * 1.44269504,
                            -T.infinity(accum_dtype),
                        )
                
                for r, c in T.Parallel(block_M, BS):
                    # Apply HSA Weight
                    # acc_s[r, c] = acc_s[r, c] * W_curr_shared[r] / scores_sum[r]
                    acc_s[r, c] = T.if_then_else(
                        scores_sum[r] > 0,
                        acc_s[r, c] * W_curr_shared[r] / scores_sum[r],
                        0.0
                    )

                # 6. GEMM PV
                T.copy(acc_s, P_shared)
                T.gemm(P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)
            
            for r, d in T.Parallel(block_M, BV):
                tq = base_t + r
                if tq < q_len:
                    Output[i_b, tq, i_h, d] = O_shared[r, d]

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
    block_M = None, mask_last_token=True, window_size = -1, dtype="bfloat16", accum_dtype="float", num_threads = None
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
    NS_kv = kv_len // BS 
    
    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    do_shape = [batch, q_len, heads, head_dim]

    q_len_padded_for_dq = tilelang.cdiv(q_len, 8) * 8
    dq_shape = [batch, q_len_padded_for_dq, heads, head_dim]
    
    # [Modified] DK/DV shape: Add groups dimension for split accumulation
    # Shape: [NV, groups, batch, kv_len, heads_kv, head_dim]
    dk_shape = [groups, batch, kv_len, heads_kv, head_dim]
    dv_shape = [groups, batch, kv_len, heads_kv, head_dim]
    
    # weight_shape = [batch, q_len, heads, NS_kv]
    # dw_shape = [batch, q_len, heads, NS_kv]
    if num_weights is None:
        num_weights = NS_kv

    weight_shape = [batch, q_len, heads, num_weights]
    dw_shape = [batch, q_len, heads, num_weights]
    scores_lse_shape = [batch, q_len, heads, num_weights]
    
    # [Modified] block_M is independent of groups
    if block_M is None or block_M <= 0:
        block_M = 32
    # block_M = 64
    print("Using block_M =", block_M, "for bwd_block_M kernel (Head-Parallel)")
    
    # [Modified] M_G is just block_M now
    M = block_M
    NP = tilelang.cdiv(q_len, M)

    # num_threads = 128
    if num_threads is None:
        num_threads = 128
    num_stages = 0

    @T.prim_func
    def hsa_bwd_dqkv_block_M(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        W: T.Tensor(weight_shape, dtype),
        DO: T.Tensor(do_shape, dtype),
        ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
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
            saved_lse = T.alloc_fragment([M], accum_dtype)
            
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
            if window_size > 0:
                first_valid_tq = (i_s + 1) * BS + window_size - 1
            else:
                first_valid_tq = (i_s + 1) * BS
            ip_start = T.floordiv(first_valid_tq, M)
            for ip in T.Pipelined(ip_start, NP, num_stages=num_stages):
                base_t = ip * M
                tile_last_tq = base_t + M - 1
                
                if tile_last_tq >= first_valid_tq:
                    # [Modified] Load Q, DO using i_h (Slicing)
                    for r, d in T.Parallel(M, BK):
                        tq = base_t + r
                        Q_shared[r, d] = T.if_then_else(tq < q_len, Q[i_b, tq, i_h, d], 0.0)
                    for r, d in T.Parallel(M, BV):
                        tq = base_t + r
                        dO_shared[r, d] = T.if_then_else(tq < q_len, DO[i_b, tq, i_h, d], 0.0)
                    
                    T.clear(acc_s_tmp)
                    T.gemm(Q_shared, K_shared, acc_s_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # 2. Apply Masking
                    for i, s in T.Parallel(M, BS):
                        tq = base_t + i
                        acc_s_tmp[i, s] = T.if_then_else(  
                        (tq >= q_len) or (enable_last_token_mask & (s == BS - 1)) or (window_size > 0 and i_s >= T.floordiv(tq - window_size + 1, BS)) or (window_size <= 0 and i_s >= T.floordiv(tq, BS)),  
                        -T.infinity(accum_dtype),  
                        acc_s_tmp[i, s]  
                    )

                    for i in T.Parallel(M):
                        tq = base_t + i
                        saved_lse[i] = T.if_then_else(
                            (tq < q_len) & ((window_size > 0 and i_s < T.floordiv(tq - window_size + 1, BS)) or (window_size <= 0 and i_s < T.floordiv(tq, BS))),
                            ScoresLSE[i_b, tq, i_h, i_s],
                            -T.infinity(accum_dtype),
                        )

                    for i, s in T.Parallel(M, BS):
                        tq = base_t + i
                        p_val = T.if_then_else(
                            (tq < q_len) & (saved_lse[i] != -T.infinity(accum_dtype)),
                            T.exp2(acc_s_tmp[i, s] * scale_log2 - saved_lse[i]),
                            0.0,
                        )
                        acc_s_tmp[i, s] = p_val
                        P_shared[i, s] = p_val

                    # === dV & dW Optimization ===
                    
                    # [Step 1] Load W.
                    for r in T.Parallel(M):
                        tq = base_t + r
                        W_local[r] = T.if_then_else(
                            (tq < q_len) & ((window_size > 0 and i_s < T.floordiv(tq - window_size + 1, BS)) or (window_size <= 0 and i_s < T.floordiv(tq, BS))),
                            W[i_b, tq, i_h, i_s],
                            0.0,
                        )

                    # [Step 2] Compute Z = dO @ V.T.
                    T.clear(dV_PdO_frag)
                    T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # [Step 3] Compute dW = Sum(P * Z).
                    for r, s in T.Parallel(M, BS):
                        acc_s_tmp[r, s] = acc_s_tmp[r, s] * dV_PdO_frag[r, s]
                    
                    T.reduce_sum(acc_s_tmp, dw_row_sum_frag, dim=1, clear=True)
                    T.copy(dw_row_sum_frag, dw_row_sum_shared)
                    for r in T.Parallel(M):
                        tq = base_t + r
                        if (tq < q_len) & ((window_size > 0 and i_s < T.floordiv(tq - window_size + 1, BS)) or (window_size <= 0 and i_s < T.floordiv(tq, BS))):
                            DW[i_b, tq, i_h, i_s] = dw_row_sum_shared[r]

                    # [Step 4] Prepare dV and dS inputs.
                    for row_idx, v in T.Parallel(M, BV):
                        dO_weighted_shared[row_idx, v] = W_local[row_idx] * dO_shared[row_idx, v]
                    
                    # Update Z (dV_PdO_frag) to W * Z.
                    for g_row, s in T.Parallel(M, BS):
                        dV_PdO_frag[g_row, s] = dV_PdO_frag[g_row, s] * W_local[g_row]

                    # [Step 5] Compute dV.
                    # dV += P.T @ dO_weighted
                    # T.copy(acc_s_tmp, P_shared)
                    T.gemm(P_shared, dO_weighted_shared, dV_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    # [Step 6] Compute dS.
                    # dS = P * (W*Z - delta)
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
                        if (tq < q_len) & ((window_size > 0 and i_s < T.floordiv(tq - window_size + 1, BS)) or (window_size <= 0 and i_s < T.floordiv(tq, BS))):
                            T.atomic_add(DQ[i_b, tq, i_h, k], dQ_local[i, k])

            T.copy(dK_accum, dK_shared)
            T.copy(dV_accum, dV_shared)
            
            # [Modified] Store DK/DV with group index offset
            # DK shape: [NV, groups, batch, kv_len, heads_kv, head_dim]
            T.copy(dK_shared, DK[g_idx, i_b, i_s_global:i_s_global + BS, i_h_kv, :])
            T.copy(dV_shared, DV[g_idx, i_b, i_s_global:i_s_global + BS, i_h_kv, :])

    return hsa_bwd_dqkv_block_M




class _hsa_block_M_attention_dense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, W, block_size, sm_scale, block_M, mask_last_token, window_size,
                dtype, accum_dtype, num_threads, block_M_bwd, num_threads_bwd):
        # Q: [B, L, HQ, D]
        # K, V: [B, L, H, D]
        # W: [B, L, HQ, S]
        # block_M / num_threads     : used by the FORWARD kernel.
        # block_M_bwd / num_threads_bwd : used by the BACKWARD kernel; if
        #   None they fall back to the forward values (legacy behavior).
        B, SEQ_LEN, HQ, D = Q.shape
        H = K.shape[2]
        groups = HQ // H
        num_weights = W.shape[-1]

        if block_M is None or block_M <= 0:
            block_M = 64
        # Decouple bwd config from fwd config; default to mirroring fwd.
        if block_M_bwd is None or block_M_bwd <= 0:
            block_M_bwd = block_M
        if num_threads_bwd is None or num_threads_bwd <= 0:
            num_threads_bwd = num_threads

        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.block_M = block_M                 # fwd block_M (kept name for compat)
        ctx.block_M_bwd = block_M_bwd         # bwd block_M (independent)
        ctx.mask_last_token = mask_last_token
        ctx.window_size = window_size
        ctx.groups = groups
        ctx.dtype = dtype
        ctx.accum_dtype = accum_dtype
        ctx.num_threads = num_threads             # fwd num_threads
        ctx.num_threads_bwd = num_threads_bwd     # bwd num_threads (independent)
        ctx.num_weights = num_weights

        fwd_kernel = hierarchical_sparse_attention_block_M(
            batch=B, heads=HQ, q_len=SEQ_LEN, kv_len=SEQ_LEN, head_dim=D,
            scale=sm_scale, block_size=block_size, groups=groups,
            selected_blocks=0, num_weights=num_weights,
            block_M=block_M, mask_last_token=mask_last_token, window_size=window_size,
            dtype=dtype, accum_dtype=accum_dtype, num_threads=num_threads,
        )

        scores_lse = torch.full(
            (B, SEQ_LEN, HQ, num_weights),
            float("-inf"),
            dtype=torch.float32,
            device=Q.device,
        )
        O = fwd_kernel(Q, K, V, W, scores_lse)

        ctx.save_for_backward(Q, K, V, W, scores_lse)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, W, scores_lse = ctx.saved_tensors
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        # Backward uses its own (independent) block_M / num_threads so that
        # the bwd kernel can be tuned separately from the fwd kernel.
        block_M = ctx.block_M_bwd
        mask_last_token = ctx.mask_last_token
        window_size = ctx.window_size
        dtype = ctx.dtype
        accum_dtype = ctx.accum_dtype
        num_threads = ctx.num_threads_bwd
        groups = ctx.groups
        num_weights = ctx.num_weights

        B, SEQ_LEN, HQ, D = Q.shape
        H = K.shape[2]
        dq_len_padded = ((SEQ_LEN + 7) // 8) * 8
        dq_shape = [B, dq_len_padded, HQ, D]
        dk_shape = [groups, B, SEQ_LEN, H, D]
        dv_shape = [groups, B, SEQ_LEN, H, D]
        dw_shape = [B, SEQ_LEN, HQ, num_weights]

        DQ = torch.zeros(dq_shape, dtype=torch.float32, device=Q.device)
        DW = torch.zeros(dw_shape, dtype=W.dtype, device=Q.device)
        DK = torch.zeros(dk_shape, dtype=K.dtype, device=Q.device)
        DV = torch.zeros(dv_shape, dtype=V.dtype, device=Q.device)

        hierarchical_sparse_attention_bwd_dqkv_block_M(
            batch=B, heads=HQ, q_len=SEQ_LEN, kv_len=SEQ_LEN, head_dim=D,
            scale=sm_scale, block_size=block_size, groups=groups,
            selected_blocks=0, num_weights=num_weights,
            block_M=block_M, mask_last_token=mask_last_token, window_size=window_size,
            dtype=dtype, accum_dtype=accum_dtype, num_threads=num_threads,
        )(Q, K, V, W, grad_output, scores_lse, DQ, DK, DV, DW)

        post_kernel = hsa_bwd_postprocess(B, SEQ_LEN, dq_len_padded, HQ, D)
        DQ = post_kernel(DQ).to(Q.dtype)
        DK = DK.sum(dim=0).to(K.dtype)
        DV = DV.sum(dim=0).to(V.dtype)

        # Return one gradient slot per forward input (Q, K, V, W) plus a
        # None for every non-tensor configuration arg. Forward now takes
        # 14 inputs (Q,K,V,W + 10 config args) -- we therefore append 10 Nones.
        return DQ, DK, DV, DW, None, None, None, None, None, None, None, None, None, None

def HSA_block_M_head_dense(Q, K, V, W, block_size=64, sm_scale=None, block_M=64, mask_last_token=True,
                           window_size=-1,
                           dtype="bfloat16", accum_dtype="float", num_threads=128,
                           block_M_fwd=None, block_M_bwd=None,
                           num_threads_fwd=None, num_threads_bwd=None):
    """Dense HSA wrapper.

    Backwards-compatible interface:
      * ``block_M`` / ``num_threads`` set the SHARED default for both fwd and bwd.
      * ``block_M_fwd`` / ``num_threads_fwd`` (if not None) override the fwd
        kernel config only.
      * ``block_M_bwd`` / ``num_threads_bwd`` (if not None) override the bwd
        kernel config only.

    Existing callers that only pass ``block_M`` / ``num_threads`` keep the
    legacy behavior (fwd and bwd use the same values).
    """
    bm_fwd = block_M_fwd if block_M_fwd is not None else block_M
    nt_fwd = num_threads_fwd if num_threads_fwd is not None else num_threads
    bm_bwd = block_M_bwd if block_M_bwd is not None else block_M
    nt_bwd = num_threads_bwd if num_threads_bwd is not None else num_threads
    return _hsa_block_M_attention_dense.apply(
        Q, K, V, W, block_size, sm_scale, bm_fwd, mask_last_token, window_size,
        dtype, accum_dtype, nt_fwd, bm_bwd, nt_bwd,
    )




def _create_chunk_window_mask(
    L: int,
    num_chunks: int,
    chunk_size: int,
    window_size: int,
    device,
    q_offset: int = 0,
    is_causal: bool = True,
):
    if not is_causal:
        return torch.ones((L, num_chunks), dtype=torch.bool, device=device)
    q_idx = torch.arange(L, device=device) + q_offset
    chunk_idx = torch.arange(num_chunks, device=device)
    if window_size > 0:
        threshold = (q_idx - window_size + 1).div(chunk_size, rounding_mode="floor")
    else:
        threshold = q_idx.div(chunk_size, rounding_mode="floor")
    return chunk_idx[None, :] < threshold[:, None]


def _dense_head_layout(q: torch.Tensor, k: torch.Tensor, lmk: torch.Tensor, G: int = None):
    B, L, HQ, D = q.shape
    H = k.shape[2]
    lmks_h = lmk.shape[2]
    D_lmk = lmk.shape[3]
    if D_lmk != D:
        assert D_lmk % D == 0, f"lmk D dim ({D_lmk}) must be divisible by q D dim ({D})"
        d_ratio = D_lmk // D
        lmk = lmk.reshape(lmk.shape[0], lmk.shape[1], lmks_h * d_ratio, D)
        lmks_h *= d_ratio
    if G is None:
        assert HQ % H == 0, f"HQ ({HQ}) must be divisible by H ({H})"
        G_eff = HQ // H
        if lmks_h == H:
            per_qhead_lmks = False
        elif lmks_h == HQ:
            per_qhead_lmks = True
        else:
            raise AssertionError(f"lmk heads ({lmks_h}) must be H ({H}) or HQ ({HQ})")
    else:
        assert HQ % G == 0, f"HQ ({HQ}) must be divisible by G ({G})"
        H_from_G = HQ // G
        assert H_from_G == H, f"G ({G}) implies H ({H_from_G}), but K has H ({H})"
        assert lmks_h == HQ, f"when G is given, lmk must have HQ ({HQ}) heads, got {lmks_h}"
        G_eff = G
        per_qhead_lmks = True
    return lmk, H, G_eff, per_qhead_lmks


def _reshape_lse_swa_head(lse_swa: torch.Tensor, B: int, L: int, H: int, G: int):
    HQ = H * G
    if lse_swa.dim() == 3:
        assert lse_swa.shape == (B, L, HQ), f"lse_swa shape {tuple(lse_swa.shape)} != {(B, L, HQ)}"
        return lse_swa
    assert lse_swa.shape == (B, L, H, G), f"lse_swa shape {tuple(lse_swa.shape)} != {(B, L, H, G)}"
    return lse_swa.reshape(B, L, HQ)


def _reshape_bias_head(bias: torch.Tensor, B: int, S: int, H: int, G: int):
    if bias is None:
        return None
    HQ = H * G
    if bias.dim() == 3:
        assert bias.shape == (B, S, HQ), f"bias shape {tuple(bias.shape)} != {(B, S, HQ)}"
        return bias.reshape(B, S, H, G).permute(0, 2, 3, 1).reshape(B, 1, HQ, S)
    assert bias.dim() == 4, f"bias must be [B, S, HQ] or [B, S, H, G], got {tuple(bias.shape)}"
    assert bias.shape == (B, S, H, G), f"bias shape {tuple(bias.shape)} != {(B, S, H, G)}"
    return bias.permute(0, 2, 3, 1).reshape(B, 1, HQ, S)


def _tl_dtype_name(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    raise AssertionError(f"unsupported TileLang dtype: {dtype}")


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def dense_chunk_scores_head_shared_lmk_kernel(
    batch, q_len, heads, kv_heads, head_dim, num_chunks,
    block_size=64, window_size=-1, scale=None,
    block_l=None, block_s=None, dtype="bfloat16", accum_dtype="float", num_threads=128,
):
    if scale is None:
        scale = (1.0 / head_dim) ** 0.5
    groups = heads // kv_heads
    if block_l is None or block_l <= 0:
        block_l = max(1, (16 + groups - 1) // groups)
    if block_s is None or block_s <= 0:
        block_s = 32

    q_shape = [batch, q_len, heads, head_dim]
    lmk_shape = [batch, num_chunks, kv_heads, head_dim]
    scores_shape = [batch, q_len, heads, num_chunks]
    gemm_m = block_l * groups
    gemm_n = block_s
    gemm_k = min(128, tilelang.next_power_of_2(head_dim))

    @T.prim_func
    def dense_chunk_scores(
        Q: T.Tensor(q_shape, dtype),
        LMK: T.Tensor(lmk_shape, dtype),
        Scores: T.Tensor(scores_shape, dtype),
    ):
        with T.Kernel(tilelang.cdiv(q_len, block_l), kv_heads, batch, threads=num_threads) as (bx, by, bz):
            i_b = bz
            i_h_kv = by
            base_l = bx * block_l

            Q_shared = T.alloc_shared([gemm_m, gemm_k], dtype)
            K_shared = T.alloc_shared([gemm_n, gemm_k], dtype)
            acc_s = T.alloc_fragment([gemm_m, gemm_n], accum_dtype)

            for i, d in T.Parallel(gemm_m, gemm_k):
                tq = base_l + i // groups
                g = i % groups
                hq = i_h_kv * groups + g
                Q_shared[i, d] = T.if_then_else((tq < q_len) and (d < head_dim), Q[i_b, tq, hq, d], 0.0)

            tile_last_tq = T.min(base_l + block_l - 1, q_len - 1)
            valid_chunks = T.alloc_var("int32")
            if window_size > 0:
                valid_chunks = T.floordiv(tile_last_tq - window_size + 1, block_size)
            else:
                valid_chunks = T.floordiv(tile_last_tq, block_size)
            valid_chunks = T.max(valid_chunks, 0)
            valid_chunks = T.min(valid_chunks, num_chunks)

            loop_blocks = tilelang.cdiv(valid_chunks, block_s)
            for s_block in T.Pipelined(loop_blocks, num_stages=0):
                base_s = s_block * block_s
                for s_idx, d in T.Parallel(gemm_n, gemm_k):
                    ts = base_s + s_idx
                    K_shared[s_idx, d] = T.if_then_else((ts < num_chunks) and (d < head_dim), LMK[i_b, ts, i_h_kv, d], 0.0)
                T.sync_threads()

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i, s_idx in T.Parallel(gemm_m, gemm_n):
                    tq = base_l + i // groups
                    g = i % groups
                    hq = i_h_kv * groups + g
                    ts = base_s + s_idx
                    if window_size > 0:
                        if (tq < q_len) and (ts < T.floordiv(tq - window_size + 1, block_size)) and (ts < num_chunks):
                            Scores[i_b, tq, hq, ts] = acc_s[i, s_idx] * scale
                    else:
                        if (tq < q_len) and (ts < T.floordiv(tq, block_size)) and (ts < num_chunks):
                            Scores[i_b, tq, hq, ts] = acc_s[i, s_idx] * scale
                T.sync_threads()

    return dense_chunk_scores


class _DenseChunkScoresHeadSharedLMK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score_q, lmk, block_size, window_size, sm_scale):
        B, L, HQ, D = score_q.shape
        S = lmk.shape[1]
        H = lmk.shape[2]
        scores = torch.full((B, L, HQ, S), float("-inf"), device=score_q.device, dtype=score_q.dtype)
        dense_chunk_scores_head_shared_lmk_kernel(
            batch=B, q_len=L, heads=HQ, kv_heads=H, head_dim=D, num_chunks=S,
            block_size=block_size, window_size=window_size, scale=float(sm_scale),
            dtype=_tl_dtype_name(score_q.dtype), accum_dtype="float", num_threads=128,
        )(score_q, lmk, scores)
        ctx.save_for_backward(score_q, lmk)
        ctx.block_size = block_size
        ctx.window_size = window_size
        ctx.sm_scale = float(sm_scale)
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        score_q, lmk = ctx.saved_tensors
        B, L, HQ, D = score_q.shape
        S = lmk.shape[1]
        H = lmk.shape[2]
        G = HQ // H
        grad_view = grad_scores.reshape(B, L, H, G, S)
        allow = _create_chunk_window_mask(L, S, ctx.block_size, ctx.window_size, grad_scores.device)
        grad_view = grad_view.masked_fill(~allow.view(1, L, 1, 1, S), 0.0)
        grad_flat = grad_view.permute(0, 2, 3, 1, 4).contiguous()
        grad_flat.mul_(ctx.sm_scale)
        grad_flat = grad_flat.reshape(B * H * G, L, S)

        q_view = score_q.reshape(B, L, H, G, D)
        q_flat = q_view.permute(0, 2, 3, 1, 4).reshape(B * H * G, L, D)
        k_flat = lmk.unsqueeze(3).expand(B, S, H, G, D).permute(0, 2, 3, 1, 4).reshape(B * H * G, S, D)

        dq_flat = torch.bmm(grad_flat, k_flat)
        dq = dq_flat.view(B, H, G, L, D).permute(0, 3, 1, 2, 4).reshape(B, L, HQ, D)

        dlmk_flat = torch.bmm(grad_flat.transpose(1, 2), q_flat)
        dlmk = dlmk_flat.view(B, H, G, S, D).sum(dim=2).permute(0, 2, 1, 3)
        return dq.to(score_q.dtype), dlmk.to(lmk.dtype), None, None, None


def _compute_dense_chunk_scores_head(
    q: torch.Tensor,
    k: torch.Tensor,
    lmk: torch.Tensor,
    block_size: int,
    window_size: int,
    *,
    lmk_q: torch.Tensor = None,
    bias: torch.Tensor = None,
    q_offset: int = 0,
    is_causal: bool = True,
    drop_mask: torch.Tensor = None,
    sm_scale: float = None,
    G: int = None,
):
    score_q = lmk_q if lmk_q is not None else q
    B, L, HQ, D = score_q.shape
    S = lmk.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    lmk, H, G_eff, per_qhead_lmks = _dense_head_layout(score_q, k, lmk, G=G)
    if (not per_qhead_lmks) and bias is None and drop_mask is None and q_offset == 0 and is_causal:
        scores = _DenseChunkScoresHeadSharedLMK.apply(score_q, lmk, block_size, window_size, float(sm_scale))
        return scores, H, G_eff

    q_view = score_q.reshape(B, L, H, G_eff, D)
    if per_qhead_lmks:
        lmk_view = lmk.reshape(B, S, H, G_eff, D)
        scores = torch.einsum("blhgd,bshgd->blhgs", q_view, lmk_view)
    else:
        scores = torch.einsum("blhgd,bshd->blhgs", q_view, lmk)
    scores = scores.float() * float(sm_scale)
    allow = _create_chunk_window_mask(L, S, block_size, window_size, q.device, q_offset=q_offset, is_causal=is_causal)
    scores = scores.masked_fill(~allow.view(1, L, 1, 1, S), float("-inf"))
    if drop_mask is not None:
        assert drop_mask.shape == (B, L, S), f"drop_mask shape {tuple(drop_mask.shape)} != {(B, L, S)}"
        scores = scores.masked_fill(drop_mask.bool().view(B, L, 1, 1, S), float("-inf"))
    scores = scores.reshape(B, L, HQ, S)
    bias_view = _reshape_bias_head(bias, B, S, H, G_eff)
    if bias_view is not None:
        scores = scores + bias_view.to(device=scores.device, dtype=scores.dtype)
    return scores.to(q.dtype), H, G_eff


def HSA_dense_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lmk: torch.Tensor,
    lse_swa: torch.Tensor,
    block_size: int,
    window_size: int,
    enable_softmax1: bool = True,
    mask_last_token: bool = True,
    lmk_q: torch.Tensor = None,
    bias: torch.Tensor = None,
    q_offset: int = 0,
    is_causal: bool = True,
    drop_mask: torch.Tensor = None,
    sm_scale: float = None,
    G: int = None,
    block_M_fwd: int = None,
    num_threads_fwd: int = None,
    block_M_bwd: int = None,
    num_threads_bwd: int = None,
):
    B, L, HQ, D = q.shape
    scores, H, G_eff = _compute_dense_chunk_scores_head(
        q, k, lmk, block_size, window_size,
        lmk_q=lmk_q, bias=bias, q_offset=q_offset, is_causal=is_causal,
        drop_mask=drop_mask, sm_scale=sm_scale, G=G,
    )
    lse_last = _reshape_lse_swa_head(lse_swa, B, L, H, G_eff).to(scores.dtype).unsqueeze(-1)
    if not enable_softmax1:
        cat_scores = torch.cat([scores, lse_last], dim=-1)
        lse_idx = -1
    else:
        ones = torch.zeros((B, L, HQ, 1), device=q.device, dtype=scores.dtype)
        cat_scores = torch.cat([scores, lse_last, ones], dim=-1)
        lse_idx = -2
    chunk_weights_all = torch.softmax(cat_scores, dim=-1)

    out = HSA_block_M_head_dense(
        Q=q,
        K=k,
        V=v,
        W=chunk_weights_all.to(q.dtype).contiguous(),
        block_size=block_size,
        sm_scale=sm_scale,
        mask_last_token=mask_last_token,
        window_size=window_size,
        block_M_fwd=block_M_fwd,
        num_threads_fwd=num_threads_fwd,
        block_M_bwd=block_M_bwd,
        num_threads_bwd=num_threads_bwd,
    )
    return out, chunk_weights_all, lse_idx

import torch.nn.functional as F
from ops.topk_head_softmax import online_softmax_topk_head
from ops.hsa_fwd_bwd_head import HSA_block_M_head


def _add_gathered_bias_to_topk_scores(scores: torch.Tensor, indices: torch.Tensor, bias: torch.Tensor, H: int, G: int):
    if bias is None:
        return scores
    B, L, HQ, K_topk = scores.shape
    S = bias.shape[1]
    bias_hq = _reshape_bias_head(bias, B, S, H, G).reshape(B, HQ, S).permute(0, 2, 1)
    indices_hq = indices.repeat_interleave(G, dim=2) if indices.shape[2] != HQ else indices
    safe_idx = indices_hq.clamp_min(0).long()
    src = bias_hq.unsqueeze(-1).expand(-1, -1, -1, K_topk)
    gathered = torch.gather(src, dim=1, index=safe_idx)
    return scores + gathered.to(scores.dtype)


def HSA_dense_interface_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lmk: torch.Tensor,
    lse_swa: torch.Tensor,
    block_size: int,
    window_size: int,
    is_causal: bool = True,
    enable_softmax1: bool = True,
    mask_last_token: bool = True,
    lmk_q: torch.Tensor = None,
    bias: torch.Tensor = None,
    q_offset: int = 0,
    drop_mask: torch.Tensor = None,
    sm_scale: float = None,
    G: int = None,
    topk: int = None,
):
    B, L, HQ, D = q.shape
    S = lmk.shape[1]
    lmk, H, G_eff, per_qhead_lmks = _dense_head_layout(q, k, lmk, G=G)
    topk = S if topk is None else topk
    topk_power_of_2 = 1 << (topk - 1).bit_length() if topk > 0 else 1
    G_arg = G_eff if per_qhead_lmks else None
    score_q = lmk_q if lmk_q is not None else q

    indices, scores = online_softmax_topk_head(
        q=score_q,
        lmks=lmk,
        lse_swa=lse_swa,
        topk=topk_power_of_2,
        block_size=block_size,
        window_size=window_size,
        is_causal=is_causal,
        q_offset=q_offset,
        drop_mask=drop_mask,
        sm_scale=sm_scale,
        bias=bias,
        G=G_arg,
    )
    scores = _add_gathered_bias_to_topk_scores(scores, indices, bias, H, G_eff)

    lse_last = _reshape_lse_swa_head(lse_swa, B, L, H, G_eff).to(scores.dtype).unsqueeze(-1)
    if not enable_softmax1:
        cat_scores = torch.cat([scores, lse_last], dim=-1)
        lse_idx = -1
    else:
        ones = torch.zeros((B, L, HQ, 1), device=q.device, dtype=scores.dtype)
        cat_scores = torch.cat([scores, lse_last, ones], dim=-1)
        lse_idx = -2
    chunk_weights_all = F.softmax(cat_scores, dim=-1)

    out = HSA_block_M_head(
        q.contiguous(), k.contiguous(), v.contiguous(),
        weights=chunk_weights_all.to(q.dtype).contiguous(),
        indices=indices.contiguous(),
        block_size=block_size,
        sm_scale=sm_scale,
        mask_last_token=mask_last_token,
    )
    return out, chunk_weights_all, lse_idx



import pytest

@pytest.mark.parametrize("B,L,H,HQ,D,block_size,window_size,is_causal,enable_softmax1,use_topk_kernel,per_qhead_lmks,use_bias,bias_dim,use_drop_mask", [
    (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, False, 3, False),
    (1, 512, 1, 8,  64, 64, 128, True,  False, True,  True,  False, 3, False),
    (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, True,  3, False),
    (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, True,  3, True),
    (2, 1024,2, 16, 128,64, 256, True,  False, False, True,  True,  3, False),
    (1, 512, 1, 8,  64, 64, 64,  True,  True,  False, True,  True,  4, False),
    (1, 512, 1, 8,  64, 64, 128, True,  False, False, True,  True,  3, True),
])
def test_dense_interface_vs_ref(
    B, L, H, HQ, D, block_size, window_size, is_causal, enable_softmax1, use_topk_kernel,
    per_qhead_lmks, use_bias, bias_dim, use_drop_mask
):
    device = "cuda"
    torch.manual_seed(0)

    block_M_fwd = 128
    num_threads_fwd = 128
    block_M_bwd = 128
    num_threads_bwd = 128

    num_chunks = L // block_size
    assert HQ % H == 0
    print(
        f"\n[test_dense_interface_vs_ref] B={B} L={L} H={H} HQ={HQ} D={D} "
        f"block_size={block_size} window_size={window_size} is_causal={is_causal} "
        f"enable_softmax1={enable_softmax1} use_topk_kernel={use_topk_kernel} "
        f"per_qhead_lmks={per_qhead_lmks} use_bias={use_bias} bias_dim={bias_dim} "
        f"use_drop_mask={use_drop_mask} num_chunks={num_chunks} "
        f"block_M_fwd={block_M_fwd} num_threads_fwd={num_threads_fwd} "
        f"block_M_bwd={block_M_bwd} num_threads_bwd={num_threads_bwd}"
    )

    q = torch.randn((B, L, HQ, D), device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((B, L, H, D),  device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((B, L, H, D),  device=device, dtype=torch.bfloat16, requires_grad=True)

    # 构造独立的 lmk_q，避免退化成 q==lmk_q 时“测不出”接口分离逻辑
    lmk_q = (q.detach() + 0.1 * torch.randn_like(q)).requires_grad_(True)

    lmk_heads = HQ if per_qhead_lmks else H
    lmk = torch.randn((B, num_chunks, lmk_heads, D), device=device, dtype=torch.bfloat16)
    lse_swa = torch.randn((B, L, HQ), device=device, dtype=torch.bfloat16)
    if use_bias:
        if bias_dim == 4:
            bias = torch.randn((B, num_chunks, H, HQ // H), device=device, dtype=torch.float32)
        else:
            bias = torch.randn((B, num_chunks, HQ), device=device, dtype=torch.float32)
    else:
        bias = None
    G_arg = HQ // H if per_qhead_lmks else None
    if use_drop_mask:
        allow = _create_chunk_window_mask(L, num_chunks, block_size, window_size, q.device, is_causal=is_causal)
        drop_mask = (torch.rand((B, L, num_chunks), device=device) < 0.2) & allow.view(1, L, num_chunks)
        has_visible = allow.any(dim=-1).view(1, L, 1)
        visible_keep = allow.view(1, L, num_chunks) & (~drop_mask)
        bad = has_visible & (~visible_keep.any(dim=-1, keepdim=True))
        drop_mask = torch.where(bad, torch.zeros_like(drop_mask), drop_mask).to(torch.int32)
    else:
        drop_mask = None

    out_hsa, w_hsa, lse_idx_hsa = HSA_dense_interface(
        q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
        block_size=block_size, window_size=window_size,
        enable_softmax1=enable_softmax1,
        lmk_q=lmk_q,
        bias=bias,
        drop_mask=drop_mask,
        G=G_arg,
        block_M_fwd=block_M_fwd,
        num_threads_fwd=num_threads_fwd,
        block_M_bwd=block_M_bwd,
        num_threads_bwd=num_threads_bwd,
    )

    out_ref, w_ref, lse_idx_ref = HSA_dense_interface_ref(
        q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
        block_size=block_size, window_size=window_size, is_causal=is_causal,
        enable_softmax1=enable_softmax1,
        lmk_q=lmk_q,
        bias=bias,
        drop_mask=drop_mask,
        G=G_arg,
    )

    def rms(x):
        return x.float().flatten().square().mean().sqrt()

    def check(name, a, b, thr):
        diff = (a.float() - b.float())
        ratio = (diff.flatten().square().mean().sqrt() / (rms(b) + 1e-12)).item()
        mx = diff.abs().max().item()
        print(f"[{name}] ratio={ratio:.6f} max_abs={mx:.6e}")
        assert ratio < thr, f"{name} mismatch ratio={ratio}, max_abs={mx}"

    S = num_chunks
    if use_drop_mask:
        check("ChunkWeightMass(einsum vs topk)", w_hsa[:, :, :, :S].sum(dim=-1), w_ref[:, :, :, :lse_idx_ref].sum(dim=-1), thr=5e-3)
    else:
        check("Weights(einsum vs topk)", w_hsa[:, :, :, :S], w_ref[:, :, :, :S], thr=5e-3)

    diff = (out_hsa.float() - out_ref.float())
    ratio = (diff.flatten().square().mean().sqrt() / (rms(out_ref) + 1e-12)).item()
    max_abs = diff.abs().max().item()
    print(f"[FWD output] ratio={ratio:.6f} max_abs={max_abs:.6e}")
    assert ratio < 5e-2, f"FWD mismatch ratio={ratio}, max_abs={max_abs}"

    grad = torch.randn_like(out_hsa)
    (out_hsa * grad).sum().backward()
    dq_hsa, dk_hsa, dv_hsa = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q.grad = k.grad = v.grad = None
    (out_ref * grad).sum().backward()
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

    check("DQ", dq_hsa, dq_ref, thr=5e-3)
    check("DK", dk_hsa, dk_ref, thr=5e-3)
    check("DV", dv_hsa, dv_ref, thr=5e-3)


@pytest.mark.parametrize("B,L,H,HQ,D,block_size,window_size", [
    (1, 512, 1, 8, 64, 64, 128),
    (1, 500, 1, 8, 64, 64, 128),
    (1, 512, 1, 8, 64, 64, -1),
])
def test_dense_chunk_scores_tilelang_fast_path_vs_torch_fallback(B, L, H, HQ, D, block_size, window_size):
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(123)

    num_chunks = (L + block_size - 1) // block_size
    q = torch.randn((B, L, HQ, D), device=device, dtype=dtype)
    k = torch.randn((B, L, H, D), device=device, dtype=dtype)
    lmk_q_base = torch.randn((B, L, HQ, D), device=device, dtype=dtype)
    lmk_base = torch.randn((B, num_chunks, H, D), device=device, dtype=dtype)

    lmk_q_fast = lmk_q_base.detach().clone().requires_grad_(True)
    lmk_fast = lmk_base.detach().clone().requires_grad_(True)
    scores_fast, H_fast, G_fast = _compute_dense_chunk_scores_head(
        q, k, lmk_fast, block_size, window_size,
        lmk_q=lmk_q_fast, is_causal=True,
    )

    lmk_q_ref = lmk_q_base.detach().clone().requires_grad_(True)
    lmk_ref = lmk_base.detach().clone().requires_grad_(True)
    zero_bias = torch.zeros((B, num_chunks, HQ), device=device, dtype=torch.float32)
    scores_ref, H_ref, G_ref = _compute_dense_chunk_scores_head(
        q, k, lmk_ref, block_size, window_size,
        lmk_q=lmk_q_ref, bias=zero_bias, is_causal=True,
    )

    assert H_fast == H_ref
    assert G_fast == G_ref
    allow = _create_chunk_window_mask(L, num_chunks, block_size, window_size, device, is_causal=True)
    allow_hq = allow.view(1, L, 1, num_chunks).expand(B, L, HQ, num_chunks)
    assert torch.isneginf(scores_fast[~allow_hq]).all()
    assert torch.isneginf(scores_ref[~allow_hq]).all()

    diff = scores_fast[allow_hq].float() - scores_ref[allow_hq].float()
    ratio = diff.flatten().square().mean().sqrt() / (scores_ref[allow_hq].float().flatten().square().mean().sqrt() + 1e-12)
    max_abs = diff.abs().max().item() if diff.numel() > 0 else 0.0
    print(f"[DenseChunkScores fast vs fallback] ratio={ratio.item():.6f} max_abs={max_abs:.6e}")
    assert ratio.item() < 5e-3

    grad = torch.randn_like(scores_fast).masked_fill(~allow_hq, 0)
    (scores_fast * grad).sum().backward()
    (scores_ref * grad).sum().backward()

    def _check_grad(name, fast, ref):
        grad_diff = fast.float() - ref.float()
        grad_ratio = grad_diff.flatten().square().mean().sqrt() / (ref.float().flatten().square().mean().sqrt() + 1e-12)
        grad_max_abs = grad_diff.abs().max().item()
        print(f"[{name}] ratio={grad_ratio.item():.6f} max_abs={grad_max_abs:.6e}")
        assert grad_ratio.item() < 5e-3

    _check_grad("DLMK_Q", lmk_q_fast.grad, lmk_q_ref.grad)
    _check_grad("DLMK", lmk_fast.grad, lmk_ref.grad)



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
    # Independent block_M / num_threads for fwd and bwd kernels (sweep winners).
    block_M_fwd = 16
    num_threads_fwd = 128
    block_M_bwd = 32
    num_threads_bwd = 128
    mask_last_token=True
    window_size = 128
    G = HQ // H
    scale = 1.0 / math.sqrt(D)
    
    num_kv_blocks = SEQ_LEN // block_size
    
    print(f"Correctness Config (Dense): Batch={B}, SeqLen={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, G={G}, NumKVBlocks={num_kv_blocks}, BlockSize={block_size}, WindowSize={window_size}, "
          f"block_M_fwd={block_M_fwd}, num_threads_fwd={num_threads_fwd}, block_M_bwd={block_M_bwd}, num_threads_bwd={num_threads_bwd}")
    
    # ---------- 生成测试数据 ----------
    torch.manual_seed(42)
    
    # 创建 requires_grad=True 的输入（这些是 leaf tensors）
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成 Dense 权重: [B, SEQ_LEN, HQ, num_kv_blocks]
    logits = torch.randn((B, SEQ_LEN, HQ, num_kv_blocks), dtype=dtype, device=device)
    
    # Causal Mask for Weights with Window Size Logic
    q_indices = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)
    
    # Logic: k_blk_idx < (q_idx - window_size + 1) // block_size
    if window_size > 0:
        threshold_blk = (q_indices - window_size + 1).div(block_size, rounding_mode='floor')
    else:
        threshold_blk = q_indices // block_size
        
    weight_mask = k_blk_indices < threshold_blk
    
    logits = logits.masked_fill(~weight_mask, -1e4)  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1).requires_grad_(True)  # leaf tensor
    
    # 构造全量的 block_indices 用于 torch ref
    block_indices_dense = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    block_indices_dense = block_indices_dense.view(1, 1, 1, num_kv_blocks).expand(B, SEQ_LEN, H, num_kv_blocks)
    
    # 用于反向传播的梯度
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ========== 测试前向传播 ==========
    
    # HSA_pair 前向 (Dense) -- use independent fwd/bwd block_M & num_threads.
    O_hsa = HSA_block_M_head_dense(
        Q, K, V, W,
        block_size=block_size, sm_scale=scale,
        mask_last_token=mask_last_token, window_size=window_size,
        block_M_fwd=block_M_fwd, num_threads_fwd=num_threads_fwd,
        block_M_bwd=block_M_bwd, num_threads_bwd=num_threads_bwd,
    )
    
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
        mask_last_token=mask_last_token,
        window_size=window_size
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
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=mask_last_token,
        window_size=window_size
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
    O_hsa_bwd = HSA_block_M_head_dense(
        Q, K, V, W,
        block_size=block_size, sm_scale=scale,
        mask_last_token=mask_last_token, window_size=window_size,
        block_M_fwd=block_M_fwd, num_threads_fwd=num_threads_fwd,
        block_M_bwd=block_M_bwd, num_threads_bwd=num_threads_bwd,
    )
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
    window_size = 128
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    scale = 1.0 / math.sqrt(D)
    
    num_kv_blocks = SEQ_LEN // block_size

    print(f"Latency Config (Dense): B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, NumKVBlocks={num_kv_blocks}, block={block_size}, G={G}, block_M={block_M}, mask_last_token={mask_last_token}, window_size={window_size}")

    # 创建 requires_grad=True 的输入（这些是 leaf tensors）
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成 Dense 权重: [B, SEQ_LEN, HQ, num_kv_blocks]
    logits = torch.randn((B, SEQ_LEN, HQ, num_kv_blocks), dtype=torch.bfloat16, device=device)
    
    # Causal Mask for Weights
    q_indices = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)
    
    if window_size > 0:
        threshold_blk = (q_indices - window_size + 1).div(block_size, rounding_mode='floor')
    else:
        threshold_blk = q_indices // block_size

    weight_mask = k_blk_indices < threshold_blk
    
    logits = logits.masked_fill(~weight_mask, -1e4)
    W = F.softmax(logits, dim=-1).requires_grad_(True)
    

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
        O = HSA_block_M_head_dense(Q_tile, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, window_size=window_size)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_head_dense(Q_tile, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, window_size=window_size)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_head_dense(Q_tile, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, window_size=window_size)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # 输出结果
    # =========================================================
    print(f"[TileLang Dense] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")
    print()



# ...existing code...

import pytest
import torch
import math
import torch.nn.functional as F

@pytest.mark.parametrize("B, SEQ_LEN, H, HQ, D, block_size, window_size", [
    (1, 512, 1, 8, 64, 64, 128),
    (2, 1024, 2, 16, 128, 64, 256),
    (1, 512, 1, 8, 64, 64, -1), # Disable window size
])
def test_correctness_fp32(B, SEQ_LEN, H, HQ, D, block_size, window_size):
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
    logits_raw = torch.randn((B, SEQ_LEN, HQ, num_kv_blocks), dtype=dtype, device=device)
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    # 2. 构造因果掩码并生成权重 W
    # 注意：Kernel 的因果逻辑是 k_blk < (q_token_idx // block_size)
    q_indices = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)
    
    # 模拟 Kernel 的分块因果边界：每个 block_M 窗口共享起始 token 的 block_id
    # 对齐逻辑 (Simulation of Alignment) - 保留原有测试逻辑中的对齐，如果有
    # 这里的 q_blk_indices 模拟的是 coarse-grained 掩码，但 window-mask 是 fine-grained
    # 为了测试严谨性，我们使用精确掩码初始化权重
    
    # q_blk_indices = (q_indices // block_M * block_M) // block_size # Old logic
    
    if window_size > 0:
        # 窗口大小掩码：k_blk 必须在 window 之外（更早的时间步）
        # 局部注意力负责 [q - w + 1, q]，HSA 负责 < q - w + 1
        threshold_blk = (q_indices - window_size + 1).div(block_size, rounding_mode='floor')
    else:
        # 纯 Causal Mask
        threshold_blk = q_indices // block_size
        
    weight_mask = k_blk_indices < threshold_blk
    
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
    O_hsa = HSA_block_M_head_dense(
        Q_hsa, K_hsa, V_hsa, W_hsa, 
        block_size=block_size, sm_scale=scale, block_M=block_M, 
        mask_last_token=mask_last_token, dtype="float32", accum_dtype="float", window_size=window_size
    )

    # 修改 Ref 的因果逻辑以匹配 Kernel 的分块行为
    O_ref = hsa_torch_ref(
        Q_ref, K_ref, V_ref, W_ref, block_indices_dense,
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=mask_last_token,
        window_size=window_size
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
    assert_close("FWD", O_ref, O_hsa)

    # 8. 反向计算
    O_hsa.backward(grad_output)
    O_ref.backward(grad_output)

    # 9. 验证梯度
    assert_close("DQ", Q_ref.grad, Q_hsa.grad)
    assert_close("DK", K_ref.grad, K_hsa.grad)
    assert_close("DV", V_ref.grad, V_hsa.grad)
    assert_close("DW", W_ref.grad, W_hsa.grad)
    
    print(f"FP32 Dense Correctness Test Passed for B={B}, SEQ_LEN={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, block_size={block_size}, window_size={window_size}")




import time
def benchmark_dense_interface_weight_methods(
    B: int = 2,
    L: int = 8192,
    H: int = 2,
    HQ: int = 16,
    D: int = 128,
    block_size: int = 64,
    window_size: int = 512,
    is_causal: bool = True,
    enable_softmax1: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    sparse_topk: int = 32,
    num_warmup: int = 20,
    num_iters: int = 50,
    block_M_fwd: int = None,
    num_threads_fwd: int = None,
    block_M_bwd: int = None,
    num_threads_bwd: int = None,
):

    device = "cuda"
    torch.manual_seed(0)

    num_chunks = L // block_size

    # inputs
    q = torch.randn((B, L, HQ, D), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((B, L, H, D), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((B, L, H, D), device=device, dtype=dtype, requires_grad=True)
    lmk = torch.randn((B, num_chunks, H, D), device=device, dtype=dtype)
    lse_swa = torch.randn((B, L, HQ), device=device, dtype=dtype)

    grad_out = torch.randn((B, L, HQ, D), device=device, dtype=dtype)

    def _warmup(fn):
        for _ in range(num_warmup):
            fn()
        torch.cuda.synchronize()

    def _time_ms(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / num_iters

    def _compute_dense_chunk_weights_for_stage():
        scores, H_eff, G_eff = _compute_dense_chunk_scores_head(
            q, k, lmk, block_size, window_size,
            is_causal=is_causal,
        )
        lse_last = _reshape_lse_swa_head(lse_swa, B, L, H_eff, G_eff).to(scores.dtype).unsqueeze(-1)
        if not enable_softmax1:
            cat_scores = torch.cat([scores, lse_last], dim=-1)
        else:
            ones = torch.zeros((B, L, HQ, 1), device=q.device, dtype=scores.dtype)
            cat_scores = torch.cat([scores, lse_last, ones], dim=-1)
        return torch.softmax(cat_scores, dim=-1)

    # ---------------------------
    # 1) dense score + dense HSA path
    # ---------------------------
    def fwd_a():
        _, _, _ = HSA_dense_interface(
            q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
            block_size=block_size, window_size=window_size,
            enable_softmax1=enable_softmax1,
            block_M_fwd=block_M_fwd, num_threads_fwd=num_threads_fwd,
            block_M_bwd=block_M_bwd, num_threads_bwd=num_threads_bwd,
        )

    def fwd_bwd_a():
        q.grad = k.grad = v.grad = None
        out, _, _ = HSA_dense_interface(
            q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
            block_size=block_size, window_size=window_size,
            enable_softmax1=enable_softmax1,
            block_M_fwd=block_M_fwd, num_threads_fwd=num_threads_fwd,
            block_M_bwd=block_M_bwd, num_threads_bwd=num_threads_bwd,
        )
        out.backward(grad_out)

    _warmup(fwd_a)
    _warmup(fwd_bwd_a)
    a_fwd = _time_ms(fwd_a)
    a_total = _time_ms(fwd_bwd_a)
    a_bwd = a_total - a_fwd

    def fwd_dense_chunk_scores_stage():
        _ = _compute_dense_chunk_weights_for_stage()

    W_stage = _compute_dense_chunk_weights_for_stage().detach().to(q.dtype).contiguous()
    grad_w_stage = torch.randn_like(W_stage)
    torch.cuda.synchronize()

    def fwd_bwd_dense_chunk_scores_stage():
        q.grad = k.grad = None
        W = _compute_dense_chunk_weights_for_stage()
        W.backward(grad_w_stage)

    def fwd_dense_hsa_stage():
        with torch.no_grad():
            _ = HSA_block_M_head_dense(
                Q=q, K=k, V=v, W=W_stage,
                block_size=block_size, sm_scale=None,
                mask_last_token=True, window_size=window_size,
                block_M_fwd=block_M_fwd, num_threads_fwd=num_threads_fwd,
                block_M_bwd=block_M_bwd, num_threads_bwd=num_threads_bwd,
            )

    def fwd_bwd_dense_hsa_stage():
        q.grad = k.grad = v.grad = None
        W_leaf = W_stage.detach().clone().requires_grad_(True)
        out = HSA_block_M_head_dense(
            Q=q, K=k, V=v, W=W_leaf,
            block_size=block_size, sm_scale=None,
            mask_last_token=True, window_size=window_size,
            block_M_fwd=block_M_fwd, num_threads_fwd=num_threads_fwd,
            block_M_bwd=block_M_bwd, num_threads_bwd=num_threads_bwd,
        )
        out.backward(grad_out)

    _warmup(fwd_dense_chunk_scores_stage)
    _warmup(fwd_bwd_dense_chunk_scores_stage)
    _warmup(fwd_dense_hsa_stage)
    _warmup(fwd_bwd_dense_hsa_stage)
    dense_chunk_fwd = _time_ms(fwd_dense_chunk_scores_stage)
    dense_chunk_total = _time_ms(fwd_bwd_dense_chunk_scores_stage)
    dense_chunk_bwd = dense_chunk_total - dense_chunk_fwd
    dense_hsa_fwd = _time_ms(fwd_dense_hsa_stage)
    dense_hsa_total = _time_ms(fwd_bwd_dense_hsa_stage)
    dense_hsa_bwd = dense_hsa_total - dense_hsa_fwd

    # ---------------------------
    # 2) sparse topk + sparse HSA path
    # ---------------------------
    q.grad = k.grad = v.grad = None

    def fwd_b():
        _, _, _ = HSA_dense_interface_ref(
            q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
            block_size=block_size, window_size=window_size,
            is_causal=is_causal, enable_softmax1=enable_softmax1,
            topk=sparse_topk,
        )

    def fwd_bwd_b():
        q.grad = k.grad = v.grad = None
        out, _, _ = HSA_dense_interface_ref(
            q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
            block_size=block_size, window_size=window_size,
            is_causal=is_causal, enable_softmax1=enable_softmax1,
            topk=sparse_topk,
        )
        out.backward(grad_out)

    _warmup(fwd_b)
    _warmup(fwd_bwd_b)
    b_fwd = _time_ms(fwd_b)
    b_total = _time_ms(fwd_bwd_b)
    b_bwd = b_total - b_fwd

    print("==== Benchmark: dense vs sparse HSA paths ====")
    print(f"Config: B={B} L={L} H={H} HQ={HQ} D={D} block={block_size} chunks={num_chunks} sparse_topk={sparse_topk} window={window_size} causal={is_causal} softmax1={enable_softmax1} dtype={dtype} "
          f"block_M_fwd={block_M_fwd} num_threads_fwd={num_threads_fwd} block_M_bwd={block_M_bwd} num_threads_bwd={num_threads_bwd}")
    print(f"[dense scores + dense HSA]       FWD {a_fwd:.3f} ms | BWD {a_bwd:.3f} ms | Total {a_total:.3f} ms")
    print(f"[topk kernel + sparse HSA]       FWD {b_fwd:.3f} ms | BWD {b_bwd:.3f} ms | Total {b_total:.3f} ms")
    print(f"[dense stage: chunk weights]     FWD {dense_chunk_fwd:.3f} ms | BWD {dense_chunk_bwd:.3f} ms | Total {dense_chunk_total:.3f} ms")
    print(f"[dense stage: HSA only]          FWD {dense_hsa_fwd:.3f} ms | BWD {dense_hsa_bwd:.3f} ms | Total {dense_hsa_total:.3f} ms")


def benchmark_block_M_num_threads_sweep(
    B: int = 2,
    L: int = 8192,
    H: int = 2,
    HQ: int = 16,
    D: int = 128,
    block_size: int = 64,
    window_size: int = 512,
    enable_softmax1: bool = False,
    mask_last_token: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    block_M_fwd_candidates=(16, 32, 64, 128),
    block_M_bwd_candidates=(16, 32, 64, 128),
    num_threads_fwd_candidates=(64, 128, 256),
    num_threads_bwd_candidates=(64, 128, 256),
    num_warmup: int = 10,
    num_iters: int = 30,
):
    """Independently sweep four knobs of ``HSA_block_M_head_dense``:
    ``block_M_fwd``, ``block_M_bwd``, ``num_threads_fwd``, ``num_threads_bwd``.

    Each knob is searched in a separate 1-D pass. While one knob varies,
    the other three are held at their candidate list's first element
    (which is also the search seed). After each pass we update that knob
    to its winning value, so subsequent passes use already-tuned settings
    for the held-fixed dimensions.

    Stage A (chunk-scoring + softmax) is run once up-front; the resulting
    ``W_fixed`` is reused for every kernel call so the sweep only measures
    the Stage-B kernel.

    Returns a dict with one list per axis: each entry is
    ``(value, time_ms, status)``.
    """
    import time as _time
    device = "cuda"
    torch.manual_seed(0)
    num_chunks = L // block_size

    # ---- inputs (built once, reused across all configs) ----
    q = torch.randn((B, L, HQ, D), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((B, L, H,  D), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((B, L, H,  D), device=device, dtype=dtype, requires_grad=True)
    lmk = torch.randn((B, num_chunks, H, D), device=device, dtype=dtype)
    lse_swa = torch.randn((B, L, HQ), device=device, dtype=dtype)
    grad_out = torch.randn((B, L, HQ, D), device=device, dtype=dtype)

    # ---- Stage A once: produce fixed W for the kernel sweep ----
    with torch.no_grad():
        scores, H_eff, G_eff = _compute_dense_chunk_scores_head(
            q, k, lmk, block_size, window_size,
            is_causal=True,
        )
        lse_last = _reshape_lse_swa_head(lse_swa, B, L, H_eff, G_eff).to(scores.dtype).unsqueeze(-1)
        if enable_softmax1:
            ones = torch.zeros((B, L, HQ, 1), device=q.device, dtype=scores.dtype)
            cat_scores = torch.cat([scores, lse_last, ones], dim=-1)
        else:
            cat_scores = torch.cat([scores, lse_last], dim=-1)
        W_fixed = torch.softmax(cat_scores, dim=-1).to(q.dtype).contiguous()

    def _warmup(fn, n):
        for _ in range(n):
            fn()
        torch.cuda.synchronize()

    def _time_ms(fn, n):
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (_time.perf_counter() - t0) * 1e3 / n

    # ---- candidate validators ----
    def _valid_block_M(bm):
        if bm <= 0 or L % bm != 0:
            return False
        return True

    def _valid_num_threads(nt):
        return nt > 0 and (nt % 32) == 0

    # Filter and pick the seed (first valid candidate) for each axis.
    bm_fwd_list = [x for x in block_M_fwd_candidates if _valid_block_M(x)]
    bm_bwd_list = [x for x in block_M_bwd_candidates if _valid_block_M(x)]
    nt_fwd_list = [x for x in num_threads_fwd_candidates if _valid_num_threads(x)]
    nt_bwd_list = [x for x in num_threads_bwd_candidates if _valid_num_threads(x)]

    if not (bm_fwd_list and bm_bwd_list and nt_fwd_list and nt_bwd_list):
        print("[fatal] one or more candidate lists is empty after filtering; abort.")
        return {}

    # Current "best so far" per axis (seeded with first valid candidate).
    cur_bm_fwd = bm_fwd_list[0]
    cur_bm_bwd = bm_bwd_list[0]
    cur_nt_fwd = nt_fwd_list[0]
    cur_nt_bwd = nt_bwd_list[0]

    print("==== Sweep: per-axis block_M / num_threads for HSA_block_M_head_dense ====")
    print(f"Config: B={B} L={L} H={H} HQ={HQ} D={D} block={block_size} window={window_size} softmax1={enable_softmax1} dtype={dtype}")
    print(f"block_M_fwd     candidates: {bm_fwd_list}")
    print(f"block_M_bwd     candidates: {bm_bwd_list}")
    print(f"num_threads_fwd candidates: {nt_fwd_list}")
    print(f"num_threads_bwd candidates: {nt_bwd_list}")

    def _run_fwd_only(bm_f, nt_f, bm_b, nt_b):
        """Time forward only (no_grad)."""
        def f():
            with torch.no_grad():
                _ = HSA_block_M_head_dense(
                    Q=q, K=k, V=v, W=W_fixed,
                    block_size=block_size, sm_scale=None,
                    mask_last_token=mask_last_token, window_size=window_size,
                    block_M_fwd=bm_f, num_threads_fwd=nt_f,
                    block_M_bwd=bm_b, num_threads_bwd=nt_b,
                )
        _warmup(f, num_warmup)
        return _time_ms(f, num_iters)

    def _run_fwd_bwd(bm_f, nt_f, bm_b, nt_b):
        """Time forward + backward (returns total ms)."""
        def f():
            q.grad = k.grad = v.grad = None
            W_leaf = W_fixed.detach().clone().requires_grad_(True)
            out = HSA_block_M_head_dense(
                Q=q, K=k, V=v, W=W_leaf,
                block_size=block_size, sm_scale=None,
                mask_last_token=mask_last_token, window_size=window_size,
                block_M_fwd=bm_f, num_threads_fwd=nt_f,
                block_M_bwd=bm_b, num_threads_bwd=nt_b,
            )
            out.backward(grad_out)
        _warmup(f, num_warmup)
        return _time_ms(f, num_iters)

    def _safe(fn, *args):
        """Run a timing closure and return (time_ms, status)."""
        try:
            return fn(*args), "ok"
        except Exception as e:                                              # noqa: BLE001
            try:
                torch.cuda.synchronize()
            except Exception:                                               # noqa: BLE001
                pass
            return float("nan"), f"fail: {str(e).splitlines()[0][:160]}"

    results = {
        "block_M_fwd":     [],   # list of (value, fwd_ms, status)
        "num_threads_fwd": [],   # list of (value, fwd_ms, status)
        "block_M_bwd":     [],   # list of (value, bwd_ms, status)
        "num_threads_bwd": [],   # list of (value, bwd_ms, status)
    }

    # ---------------- 1) sweep block_M_fwd (measure FWD only) ----------------
    print(f"---- Sweep block_M_fwd (other axes fixed: bm_bwd={cur_bm_bwd}, nt_fwd={cur_nt_fwd}, nt_bwd={cur_nt_bwd}) ----")
    for bm in bm_fwd_list:
        ms, status = _safe(_run_fwd_only, bm, cur_nt_fwd, cur_bm_bwd, cur_nt_bwd)
        results["block_M_fwd"].append((bm, ms, status))
        if status == "ok":
            print(f"  block_M_fwd={bm:>3}: FWD {ms:7.3f} ms")
        else:
            print(f"  block_M_fwd={bm:>3}: FAILED -> {status[6:]}")
    ok = [r for r in results["block_M_fwd"] if r[2] == "ok"]
    if ok:
        cur_bm_fwd = min(ok, key=lambda r: r[1])[0]
        print(f"  -> best block_M_fwd = {cur_bm_fwd}")

    # ---------------- 2) sweep num_threads_fwd (measure FWD only) ----------------
    print(f"---- Sweep num_threads_fwd (other axes fixed: bm_fwd={cur_bm_fwd}, bm_bwd={cur_bm_bwd}, nt_bwd={cur_nt_bwd}) ----")
    for nt in nt_fwd_list:
        ms, status = _safe(_run_fwd_only, cur_bm_fwd, nt, cur_bm_bwd, cur_nt_bwd)
        results["num_threads_fwd"].append((nt, ms, status))
        if status == "ok":
            print(f"  num_threads_fwd={nt:>3}: FWD {ms:7.3f} ms")
        else:
            print(f"  num_threads_fwd={nt:>3}: FAILED -> {status[6:]}")
    ok = [r for r in results["num_threads_fwd"] if r[2] == "ok"]
    if ok:
        cur_nt_fwd = min(ok, key=lambda r: r[1])[0]
        print(f"  -> best num_threads_fwd = {cur_nt_fwd}")

    # Measure a stable reference FWD time once with the now-fixed fwd config.
    # We subtract this from Total to get a clean BWD-only number for the
    # remaining two sweeps.
    ref_fwd_ms, ref_status = _safe(_run_fwd_only, cur_bm_fwd, cur_nt_fwd, cur_bm_bwd, cur_nt_bwd)
    if ref_status != "ok":
        print(f"[fatal] reference FWD timing failed: {ref_status}; abort BWD sweeps.")
        return results
    print(f"---- Reference FWD time @ (bm_fwd={cur_bm_fwd}, nt_fwd={cur_nt_fwd}) = {ref_fwd_ms:.3f} ms (used as subtractor for BWD sweeps) ----")

    # ---------------- 3) sweep block_M_bwd (measure BWD = Total - ref_FWD) ----------------
    print(f"---- Sweep block_M_bwd (other axes fixed: bm_fwd={cur_bm_fwd}, nt_fwd={cur_nt_fwd}, nt_bwd={cur_nt_bwd}) ----")
    for bm in bm_bwd_list:
        total_ms, status = _safe(_run_fwd_bwd, cur_bm_fwd, cur_nt_fwd, bm, cur_nt_bwd)
        bwd_ms = total_ms - ref_fwd_ms if status == "ok" else float("nan")
        results["block_M_bwd"].append((bm, bwd_ms, status))
        if status == "ok":
            print(f"  block_M_bwd={bm:>3}: BWD {bwd_ms:7.3f} ms (Total {total_ms:7.3f} ms)")
        else:
            print(f"  block_M_bwd={bm:>3}: FAILED -> {status[6:]}")
    ok = [r for r in results["block_M_bwd"] if r[2] == "ok"]
    if ok:
        cur_bm_bwd = min(ok, key=lambda r: r[1])[0]
        print(f"  -> best block_M_bwd = {cur_bm_bwd}")

    # ---------------- 4) sweep num_threads_bwd (measure BWD = Total - ref_FWD) ----------------
    print(f"---- Sweep num_threads_bwd (other axes fixed: bm_fwd={cur_bm_fwd}, nt_fwd={cur_nt_fwd}, bm_bwd={cur_bm_bwd}) ----")
    for nt in nt_bwd_list:
        total_ms, status = _safe(_run_fwd_bwd, cur_bm_fwd, cur_nt_fwd, cur_bm_bwd, nt)
        bwd_ms = total_ms - ref_fwd_ms if status == "ok" else float("nan")
        results["num_threads_bwd"].append((nt, bwd_ms, status))
        if status == "ok":
            print(f"  num_threads_bwd={nt:>3}: BWD {bwd_ms:7.3f} ms (Total {total_ms:7.3f} ms)")
        else:
            print(f"  num_threads_bwd={nt:>3}: FAILED -> {status[6:]}")
    ok = [r for r in results["num_threads_bwd"] if r[2] == "ok"]
    if ok:
        cur_nt_bwd = min(ok, key=lambda r: r[1])[0]
        print(f"  -> best num_threads_bwd = {cur_nt_bwd}")

    # ---------------- final: report combined best & verify end-to-end ----------------
    print("---- Best per-axis ----")
    print(f"  block_M_fwd     = {cur_bm_fwd}")
    print(f"  num_threads_fwd = {cur_nt_fwd}")
    print(f"  block_M_bwd     = {cur_bm_bwd}")
    print(f"  num_threads_bwd = {cur_nt_bwd}")

    final_total_ms, status = _safe(_run_fwd_bwd, cur_bm_fwd, cur_nt_fwd, cur_bm_bwd, cur_nt_bwd)
    if status == "ok":
        print(f"  Combined end-to-end: Total {final_total_ms:.3f} ms "
              f"(ref FWD {ref_fwd_ms:.3f} ms => implied BWD {final_total_ms - ref_fwd_ms:.3f} ms)")
    else:
        print(f"  Combined end-to-end: FAILED -> {status[6:]}")

    return results


if __name__ == "__main__":
    # main_block_M_correctness()
    # main_block_M_latency()
    
    # params_list = [
    #     (1, 1000, 1, 8, 64, 32, 128),
    #     (2, 1024, 1, 8, 64, 32, 256),
    #     (3, 512, 1, 8, 64, 32, 128),
    #     (4, 512, 1, 8, 64, 64, -1),
    #     (5, 256, 1, 8, 64, 64, 64),
    # ]
    # for p in params_list:
    #     test_correctness_fp32(*p)
    
    params_list = [
        (1, 500, 1, 8,  64, 64, 128, True,  False, False, False, False, 3, False),
        (1, 512, 1, 8,  64, 64, 128, True,  False, True,  True,  False, 3, False),
        (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, True,  3, False),
        (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, True,  3, True),
        (1, 512, 1, 8,  64, 64, 64,  True,  True,  False, True,  True,  4, False),
        (1, 512, 1, 8,  64, 64, 128, True,  False, False, True,  True,  3, True),
    ]
    for p in params_list:
        test_dense_interface_vs_ref(*p)
    
    benchmark_dense_interface_weight_methods(
        B = 4,
        L = 8192,
        H= 2,
        HQ= 32,
        D= 128,
        block_size = 64,
        window_size = 512,
        block_M_fwd=64,
        num_threads_fwd=128,
        block_M_bwd=64,
        num_threads_bwd=128,
    )

    # Sweep block_M x num_threads for the Stage-B kernel
    # (HSA_block_M_head_dense). Comment this out if you only want the
    # default benchmark above. Each axis is searched independently:
    # block_M_fwd / block_M_bwd in {16,32,64,128}, num_threads_fwd /
    # num_threads_bwd in {64,128,256}.
    # benchmark_block_M_num_threads_sweep(
    #     block_M_fwd_candidates=(16, 32, 64, 128),
    #     block_M_bwd_candidates=(16, 32, 64, 128),
    #     num_threads_fwd_candidates=(64, 128, 256, 512),
    #     num_threads_bwd_candidates=(64, 128, 256, 512),
    # )

    # pkill -f "burner.*--gpu 7"; export CUDA_VISIBLE_DEVICES=7; python ops/hsa_fwd_bwd_head_dense_forbench.py
