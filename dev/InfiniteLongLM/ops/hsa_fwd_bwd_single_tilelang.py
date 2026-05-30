# ruff: noqa
import torch

import tilelang
from tilelang import language as T
import tilelang.testing

tilelang.testing.set_random_seed(0)


from einops import rearrange
def hsa_torch_ref(q, k, v, weights, indices, *, chunk_size: int, sm_scale: float, block_q: int, mask_last_token: bool = False):
    """
    Head-wise reference impl (matches test_group_qa math):
    - For each query block and each selected K chunk:
      p = softmax_s(q @ k^T * sm_scale)
      o_k = p @ v
    - Final: o = sum_k (weights[:, :, hq, k] * o_k[..., hq])    # per q-head weight

    Shape conventions (head-wise):
    - q: (B, L, HQ, D)
    - k, v: (B, L, H, D)
    - weights: (B, q_blocks, HQ, K)              # per q-head, NOT shared across G
    - indices: (B, q_blocks, H, K) or (B, L, H, K) with L == q_blocks * block_q
    - returns: o_ref: (B, L, HQ, D) float32

    mask_last_token: zero out the last token of each chunk in the attention score
                     (matches the kernel's enable_last_token_mask semantics).
    """
    B, L, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    q_blocks = L // block_q
    device = q.device

    # 若 indices 是按每个 token 给的 (B, L, H, K)，在 BLOCK_Q==1 时 q_blocks == L，可直接使用；
    # 一般情形下，按 block 起始 token 聚合到 (B, q_blocks, H, K)（这里假设传入已是对齐的）
    if indices.shape[1] != q_blocks:
        # 聚合成以每个 q_block 第一个 token 的 indices 为代表
        # 警告：这仅在每个 block 内 indices 保持一致时有效
        idx_view = indices.view(B, q_blocks, block_q, H, -1)
        indices_q = idx_view[:, :, 0, :, :].contiguous()
    else:
        indices_q = indices

    # mask 无效块（-1）
    valid_mask = (indices_q >= 0)  # (B, q_blocks, H, K)
    safe_indices = indices_q.clamp_min(0)

    # 重排 K/V 为 (B, N, S, H, D)
    N = L // chunk_size
    k_chunks = rearrange(k, 'B (N S) h d -> B N S h d', S=chunk_size)
    v_chunks = rearrange(v, 'B (N S) h d -> B N S h d', S=chunk_size)

    # 根据 indices 在 N 维 gather：先展平 (q_blocks, K) 方便一次 gather
    idx_flat = rearrange(safe_indices, 'B Bq h K -> B (Bq K) h').unsqueeze(2).unsqueeze(-1)  # (B, BqK, 1, h, 1)
    idx_flat = idx_flat.expand(-1, -1, chunk_size, -1, D)                                   # (B, BqK, S, h, D)
    idx_flat = idx_flat.long()  
    gather_k = k_chunks.gather(dim=1, index=idx_flat)  # (B, BqK, S, h, D)
    gather_v = v_chunks.gather(dim=1, index=idx_flat)

    gather_k = rearrange(gather_k, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)
    gather_v = rearrange(gather_v, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)

    # 扩到 HQ 维
    k_ = torch.repeat_interleave(gather_k, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)
    v_ = torch.repeat_interleave(gather_v, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)

    # q 分块: (B, L, HQ, D) -> (B, q_blocks, block_q, HQ, D)
    q_chunked = rearrange(q, 'B (Bq X) hq d -> B Bq X hq d', X=block_q)

    # qk: (B, Bq, X, S, K, HQ)
    qk = torch.einsum('b q x h d, b q s k h d -> b q x s k h', q_chunked.float(), k_.float())
    qk = qk * float(sm_scale)

    # mask_last_token：将每个 chunk 内最后一个 token 的 attention score 置为 -inf
    # qk 的维度 S（dim=3）就是 chunk_size 维度
    if mask_last_token:
        qk[:, :, :, -1, :, :] = float('-inf')

    # 在 S 维做 softmax
    p = torch.softmax(qk, dim=3)  # S 维
    # 如果整个 chunk 全为 -inf（不应该发生），softmax 会产生 NaN，这里把 NaN 置为 0
    if mask_last_token:
        p = torch.nan_to_num(p, nan=0.0)

    # o_k: (B, Bq, X, K, HQ, D)
    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    # Weights are per q-head: (B, Bq, HQ, K). Mask invalid blocks per h_kv (broadcast over G).
    w_masked = weights.clone()
    valid_mask_expanded = torch.repeat_interleave(valid_mask, dim=-2, repeats=G)  # (B, Bq, HQ, K)
    w_masked = w_masked.masked_fill(~valid_mask_expanded, 0)
    w_exp = w_masked.float()  # (B, Bq, HQ, K)

    # Aggregate over K
    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'B Bq X hq d -> B (Bq X) hq d')  # back to (B, L, HQ, D)
    return o_ref.to(torch.float32)



def make_dq_layout_hsa(dQ):

    B, L, HQ, D = dQ.shape
    return T.Layout(dQ.shape,
    lambda b, l, h, d:   [b,l, h//8, d//16, (d%16)//2, (h%8), (d%2)]
 )

@tilelang.jit(
    out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def hsa_bwd_postprocess(batch, q_len, heads, head_dim):
    shape = [batch, q_len, heads, head_dim]
    accum_dtype = "float"
    dtype = "bfloat16"
    blk = 64 # 可以调整的块大小

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




@tilelang.jit(
    out_idx=[-1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def hierarchical_sparse_attention(batch,
                                  heads,
                                  q_len,
                                  kv_len,
                                  head_dim,
                                  scale=None,
                                  block_size=64,
                                  groups=16,
                                  selected_blocks=16,
                                  num_weights=None,
                                  mask_last_token=True,
                                  num_threads=None):
    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True

    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    # 允许 num_weights (输入的 weights 张量的最后一维) 大于 selected_blocks (S)
    if num_weights is None:
        num_weights = selected_blocks

    head_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    kv_shape = [batch, kv_len, head_kv, head_dim]
    # head-wise: per q-head weight, NOT shared across the G group.
    weight_shape = [batch, q_len, heads, num_weights]
    block_indices_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(head_dim))

    NK = tilelang.cdiv(head_dim, block_T)
    # NV removed (always 1)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T
    num_stages = 1
    if num_threads is None:
        num_threads = 32
    threads = num_threads

    scores_lse_shape = [batch, q_len, heads, selected_blocks]

    @T.prim_func
    def hsa(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(q_len, batch * head_kv, threads=threads) as (bx, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            # head-wise: per-q-head weight vector of length G for each selected block.
            chunk_weight = T.alloc_fragment([G], dtype)

            i_t, i_bh = bx, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, i_t, i_h * G:(i_h + 1) * G, :], Q_shared)


            T.fill(acc_o, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i_1 in T.Pipelined(NS, num_stages=num_stages):
                blk_idx = BlockIndices[i_b, i_t, i_h, i_1]
                i_s = blk_idx * BS
                if i_s >= 0:
                    T.copy(K[i_b, i_s:i_s + BS, i_h, :], K_shared)
                    # Load per-q-head weights (G values) for this selected block.
                    for g in T.Parallel(G):
                        chunk_weight[g] = W[i_b, i_t, i_h * G + g, i_1]

                    T.clear(acc_s)

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

                    # Masking (last token mask)
                    for g, v in T.Parallel(G, BS):
                        acc_s[g, v] = T.if_then_else(
                            (v == BS - 1) and enable_last_token_mask,
                            -T.infinity(accum_dtype),
                            acc_s[g, v]
                        )

                    # Softmax
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for g, v in T.Parallel(G, BS):
                        acc_s[g, v] = T.exp2(acc_s[g, v] * scale - scores_max[g] * scale)
                    T.fill(scores_sum, 0.0)
                    T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)

                    # 保存 LSE = scores_max * scale + log2(scores_sum)
                    for g in T.Parallel(G):
                        ScoresLSE[i_b, i_t, i_h * G + g, i_1] = T.if_then_else(
                            scores_sum[g] > 0,
                            scores_max[g] * scale + T.log(scores_sum[g]) * 1.44269504,
                            -T.infinity(accum_dtype)
                        )

                    for g, v in T.Parallel(G, BS):
                        acc_s[g, v] = chunk_weight[g] * acc_s[g, v] / scores_sum[g]
                    T.copy(acc_s, acc_s_cast)

                    # V * softmax(Q * K)
                    T.copy(V[i_b, i_s:i_s + BS, i_h, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, i_t, i_h * G:(i_h + 1) * G, :])

    return hsa

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_bwd_dqkv(
    batch,
    heads,
    q_len,
    kv_len,
    head_dim,
    scale=None,
    block_size=64,
    groups=16,
    selected_blocks=16,
    num_weights=None,
    mask_last_token=True,
    dtype="bfloat16",
    accum_dtype="float",
    num_threads=None,
):
    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True

    if scale is None:
        sm_scale = (1.0 / head_dim)**0.5
    else:
        sm_scale = scale

    scale_log2 = sm_scale * 1.44269504  # log2(e)

    # 允许 num_weights (输入的 weights 张量的最后一维) 大于 selected_blocks (S)
    if num_weights is None:
        num_weights = selected_blocks

    from tilelang import language as T

    B = batch
    BS = block_size
    G = groups
    V = head_dim
    K = head_dim
    BK = tilelang.next_power_of_2(K)
    BV = min(128, tilelang.next_power_of_2(head_dim))
    NS_kv = kv_len // BS
    # NV removed (always 1)
    S = selected_blocks
    # NV=1

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    o_shape = [batch, q_len, heads, head_dim]
    do_shape = [batch, q_len, heads, head_dim]
    dq_shape = [batch, q_len, heads, head_dim]
    dk_shape = [batch, kv_len, heads_kv, head_dim]
    dv_shape = [batch, kv_len, heads_kv, head_dim]
    
    # head-wise: per-q-head weights and dW.
    weight_shape = [batch, q_len, heads, num_weights]
    dw_shape = [batch, q_len, heads, num_weights]
    block_mask_shape = [batch, q_len, heads_kv, NS_kv]
    scores_lse_shape = [batch, q_len, heads, selected_blocks]
    
    if num_threads is None:
        num_threads = 128
    num_stages = 0

    @T.prim_func
    def hsa_bwd_dqkv(
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
            ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
    ):
        with T.Kernel(NS_kv, B * heads_kv, threads=num_threads) as (i_s, i_bh):
            i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
            i_s_global = i_s * BS

            # === Shared Memory（独立分配，避免 aliasing，使 num_stages>0 可用） ===
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            dO_shared = T.alloc_shared([G, BV], dtype)
            dO_weighted_shared = T.alloc_shared([G, BV], dtype)
            P_shared = T.alloc_shared([G, BS], dtype)
            dS_shared = T.alloc_shared([G, BS], dtype)

            # === Fragment ===
            dV_PdO_frag = T.alloc_fragment([G, BS], accum_dtype)
            dS_frag = T.alloc_fragment([G, BS], accum_dtype)
            dV_accum = T.alloc_fragment([BS, BV], accum_dtype)
            dK_accum = T.alloc_fragment([BS, BK], accum_dtype)
            dQ_local = T.alloc_fragment([G, BK], accum_dtype)

            acc_s_tmp = T.alloc_fragment([G, BS], accum_dtype)

            dw_row_sum_frag = T.alloc_fragment([G], accum_dtype)
            dw_sum_frag = T.alloc_fragment([1], accum_dtype)

            # LSE fragment（用于复用前向保存的 LSE）
            lse_frag = T.alloc_fragment([G], accum_dtype)
            # head-wise: per-q-head weight vector of length G.
            chunk_weight = T.alloc_fragment([G], dtype)
            found_pos = T.alloc_var("int32")

            T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :BK], K_shared)
            T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, :], V_shared)

            T.fill(dK_accum, 0)
            T.fill(dV_accum, 0)

            T.annotate_layout({
                DQ: make_dq_layout_hsa(DQ),
            })

            for i_q in T.Pipelined(q_len, num_stages=num_stages):
                found_pos = BlockMask[i_b, i_q, i_h, i_s]

                if found_pos != -1:
                    T.copy(Q[i_b, i_q, i_h * G:(i_h + 1) * G, :BK], Q_shared)
                    T.copy(DO[i_b, i_q, i_h * G:(i_h + 1) * G, :], dO_shared)

                    # head-wise: load per-q-head weights for this selected block.
                    for g in T.Parallel(G):
                        chunk_weight[g] = W[i_b, i_q, i_h * G + g, found_pos]

                    # === QK GEMM（直接输出到 acc_s_tmp，省掉 qk_frag）===
                    T.clear(acc_s_tmp)
                    T.gemm(Q_shared, K_shared, acc_s_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # === 加载 LSE 并用 LSE 恢复 P（原地计算）===
                    for g in T.Parallel(G):
                        lse_frag[g] = ScoresLSE[i_b, i_q, i_h * G + g, found_pos]

                    for g, s in T.Parallel(G, BS):
                        acc_s_tmp[g, s] = T.if_then_else(
                            lse_frag[g] > -T.infinity(accum_dtype),
                            T.exp2(acc_s_tmp[g, s] * scale_log2 - lse_frag[g]),
                            0.0
                        )
                    if enable_last_token_mask:
                        for g, s in T.Parallel(G, BS):
                            acc_s_tmp[g, s] = T.if_then_else(
                                s == BS - 1,
                                0.0,
                                acc_s_tmp[g, s]
                            )

                    # === [Step 2] 计算 Z = dO @ V.T ===
                    T.clear(dV_PdO_frag)
                    T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # === [Step 3] 计算 dW = Sum(P * Z)（复用 dS_frag 作为临时变量）===
                    for g, s in T.Parallel(G, BS):
                        dS_frag[g, s] = acc_s_tmp[g, s] * dV_PdO_frag[g, s]

                    T.reduce_sum(dS_frag, dw_row_sum_frag, dim=1, clear=True)
                    # head-wise: dW is per q-head, write G values directly without further reducing.
                    for g in T.Parallel(G):
                        DW[i_b, i_q, i_h * G + g, found_pos] = dw_row_sum_frag[g]

                    # === [Step 4] 准备 dV 的输入 ===
                    # 更新 dO_shared 为 dO_weighted (per-q-head weight)
                    for g, v in T.Parallel(G, BV):
                        dO_weighted_shared[g, v] = chunk_weight[g] * dO_shared[g, v]

                    # === [Step 5] 计算 dV: dV += P.T @ dO_weighted ===
                    T.copy(acc_s_tmp, P_shared)
                    T.gemm(P_shared, dO_weighted_shared, dV_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    # === [Step 6] 计算 dS（复用 dW 计算中的 Di = dw_row_sum_frag）===
                    # dS = sm_scale * W * P * (dp - Di), W is per-q-head
                    for g, s in T.Parallel(G, BS):
                        dS_frag[g, s] = sm_scale * chunk_weight[g] * acc_s_tmp[g, s] * (dV_PdO_frag[g, s] - dw_row_sum_frag[g])

                    T.copy(dS_frag, dS_shared)

                    # === [Step 7] 计算 dK: dK += dS.T @ Q ===
                    T.gemm(dS_shared, Q_shared, dK_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    # === [Step 8] 计算 dQ: dQ += dS @ K ===
                    T.clear(dQ_local)
                    T.gemm(dS_shared, K_shared, dQ_local, policy=T.GemmWarpPolicy.FullRow)

                    for g, k in T.Parallel(G, BK):
                        T.atomic_add(DQ[i_b, i_q, i_h * G + g, k], dQ_local[g, k])

            # 写回 dK 和 dV（复用 K_shared 和 V_shared）
            T.copy(dK_accum, K_shared)
            T.copy(dV_accum, V_shared)
            T.copy(K_shared, DK[i_b, i_s_global:i_s_global + BS, i_h, :BK])
            T.copy(V_shared, DV[i_b, i_s_global:i_s_global + BS, i_h, :])

    return hsa_bwd_dqkv




@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def hsa_kernel_block_mask(
    batch,
    heads,
    q_len,
    kv_len,
    selected_blocks,
    block_size,
    dtype="int32",
):
    """
    优化版：按 (q_len, batch, heads * S) 并行，避免内层循环
    BlockMask[b, t, h, kv_blk] = s (该 kv_blk 在 S 个选中块中的位置)
                               = -1 (未被选中)
    """
    from tilelang import language as T

    S = selected_blocks
    NS_kv = kv_len // block_size

    block_indices_shape = [batch, q_len, heads, selected_blocks]
    block_mask_shape = [batch, q_len, heads, NS_kv]

    @T.prim_func
    def build_block_mask(
        BlockIndices: T.Tensor(block_indices_shape, dtype),
        BlockMask: T.Tensor(block_mask_shape, dtype),
    ):
        with T.Kernel(q_len, batch, heads * S) as (i_t, i_b, i_hs):
            i_h = i_hs // S
            i_s = i_hs % S
            
            block_idx = BlockIndices[i_b, i_t, i_h, i_s]
            
            if block_idx >= 0 and block_idx < NS_kv:
                BlockMask[i_b, i_t, i_h, block_idx] = i_s

    return build_block_mask


# ============================================================
# Phase 1: bwd_dq_dw — 按 Query token 并行，计算 dQ + dW（无 atomic_add）
# ============================================================
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_bwd_dq_dw(
    batch,
    heads,
    q_len,
    kv_len,
    head_dim,
    scale=None,
    block_size=64,
    groups=16,
    selected_blocks=16,
    num_weights=None,
    mask_last_token=True,
    dtype="bfloat16",
    accum_dtype="float",
    num_threads=None,
):
    """
    Phase 1: 按 query token 并行，遍历该 token 的 S 个 selected chunks。
    计算 dQ（累加到 fragment，最后直接写回，无 atomic_add）和 dW。
    完全仿照 Triton 版本的 kernel_attn_bwd_dq 逻辑。
    """
    enable_last_token_mask = mask_last_token

    if scale is None:
        sm_scale = (1.0 / head_dim)**0.5
    else:
        sm_scale = scale

    scale_log2 = sm_scale * 1.44269504  # log2(e)

    if num_weights is None:
        num_weights = selected_blocks

    B = batch
    BS = block_size
    G = groups
    BK = tilelang.next_power_of_2(head_dim)
    BV = min(128, tilelang.next_power_of_2(head_dim))
    # NV removed (always 1)
    S = selected_blocks

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    do_shape = [batch, q_len, heads, head_dim]
    dq_shape = [batch, q_len, heads, head_dim]
    # head-wise: per-q-head weights and dW.
    weight_shape = [batch, q_len, heads, num_weights]
    dw_shape = [batch, q_len, heads, num_weights]
    di_shape = [batch, q_len, heads, selected_blocks]
    block_indices_shape = [batch, q_len, heads_kv, selected_blocks]

    if num_threads is None:
        num_threads = 128
    num_stages = 1

    scores_lse_shape = [batch, q_len, heads, selected_blocks]

    @T.prim_func
    def bwd_dq_dw(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(k_shape, dtype),
            V: T.Tensor(v_shape, dtype),
            W: T.Tensor(weight_shape, dtype),
            DO: T.Tensor(do_shape, dtype),
            BlockIndices: T.Tensor(block_indices_shape, "int32"),
            ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
            DQ: T.Tensor(dq_shape, accum_dtype),
            DW: T.Tensor(dw_shape, accum_dtype),
            DiOut: T.Tensor(di_shape, accum_dtype),
    ):
        with T.Kernel(q_len, B * heads_kv, threads=num_threads) as (i_t, i_bh):
            i_b, i_h = i_bh // heads_kv, i_bh % heads_kv

            # === Shared Memory ===
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            dO_shared = T.alloc_shared([G, BV], dtype)
            # 用于 dS @ K 的 GEMM 输入
            dS_shared = T.alloc_shared([G, BS], dtype)

            # === Fragment ===
            acc_s = T.alloc_fragment([G, BS], accum_dtype)

            # dQ 累加器（跨所有 selected chunks 累加）
            dQ_accum = T.alloc_fragment([G, BK], accum_dtype)
            # 临时 fragment
            dV_PdO_frag = T.alloc_fragment([G, BS], accum_dtype)
            dS_frag = T.alloc_fragment([G, BS], accum_dtype)
            delta_rows = T.alloc_fragment([G], accum_dtype)
            qk_frag = T.alloc_fragment([G, BS], accum_dtype)

            # dW 相关
            dw_row_sum_frag = T.alloc_fragment([G], accum_dtype)
            dw_sum_frag = T.alloc_fragment([1], accum_dtype)

            # LSE fragment
            lse_shared = T.alloc_fragment([G], accum_dtype)
            # head-wise: per-q-head weight vector of length G.
            chunk_weight = T.alloc_fragment([G], dtype)

            # 加载 Q 和 dO（整个 kernel 生命周期内不变）
            T.copy(Q[i_b, i_t, i_h * G:(i_h + 1) * G, :BK], Q_shared)
            T.copy(DO[i_b, i_t, i_h * G:(i_h + 1) * G, :], dO_shared)

            T.fill(dQ_accum, 0)

            for i_s in T.Pipelined(S, num_stages=num_stages):
                blk_idx = BlockIndices[i_b, i_t, i_h, i_s]
                i_s_global = blk_idx * BS

                if blk_idx >= 0:
                    # 加载 K, V
                    T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :BK], K_shared)
                    T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, :], V_shared)

                    # head-wise: load per-q-head weights for this selected block.
                    for g in T.Parallel(G):
                        chunk_weight[g] = W[i_b, i_t, i_h * G + g, i_s]

                    # === QK GEMM ===
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # === 加载 LSE 并用 LSE 恢复 P（省掉 reduce_max + reduce_sum + normalize）===
                    for g in T.Parallel(G):
                        lse_shared[g] = ScoresLSE[i_b, i_t, i_h * G + g, i_s]
                    # T.copy(lse_shared, lse_frag)

                    # Masking + 恢复 P
                    for g, s in T.Parallel(G, BS):
                        acc_s[g, s] = T.if_then_else(
                            lse_shared[g] > -T.infinity(accum_dtype),
                            T.exp2(acc_s[g, s] * scale_log2 - lse_shared[g]),
                            0.0
                        )
                    if enable_last_token_mask:
                        for g, s in T.Parallel(G, BS):
                            acc_s[g, s] = T.if_then_else(
                                s == BS - 1,
                                0.0,
                                acc_s[g, s]
                            )
                    # 现在 acc_s 是归一化的 P（未乘 weight）

                    # === 计算 Z = dO @ V^T ===
                    T.clear(dV_PdO_frag)
                    T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # === 计算 dW = sum(P * Z) （在乘 W 之前）===
                    for g, s in T.Parallel(G, BS):
                        qk_frag[g, s] = acc_s[g, s] * dV_PdO_frag[g, s]
                    T.reduce_sum(qk_frag, dw_row_sum_frag, dim=1, clear=True)
                    # head-wise: write per-q-head dW values directly (no reduction over G).
                    for g in T.Parallel(G):
                        DW[i_b, i_t, i_h * G + g, i_s] = dw_row_sum_frag[g]
                    # Save Di per q-head (zero cost; dw_row_sum_frag still holds these values).
                    for g in T.Parallel(G):
                        DiOut[i_b, i_t, i_h * G + g, i_s] = dw_row_sum_frag[g]

                    # === 乘以 W 得到 W*Z，用于 dS 计算 (per-q-head)===
                    for g, s in T.Parallel(G, BS):
                        dV_PdO_frag[g, s] = dV_PdO_frag[g, s] * chunk_weight[g]

                    # === 计算 dS ===
                    # delta = sum(P * W*Z, dim=1)
                    for g, s in T.Parallel(G, BS):
                        qk_frag[g, s] = acc_s[g, s] * dV_PdO_frag[g, s]
                    T.reduce_sum(qk_frag, delta_rows, dim=1, clear=True)

                    # dS = sm_scale * (P*W*Z - P*delta)
                    for g, s in T.Parallel(G, BS):
                        dS_frag[g, s] = sm_scale * (qk_frag[g, s] - acc_s[g, s] * delta_rows[g])

                    # === 计算 dQ += dS @ K ===
                    T.copy(dS_frag, dS_shared)
                    T.gemm(dS_shared, K_shared, dQ_accum, policy=T.GemmWarpPolicy.FullRow)

            # === 写回 dQ（直接写的 float32，无 swizzle layout）===
            dQ_shared = T.alloc_shared([G, BK], accum_dtype)
            T.copy(dQ_accum, dQ_shared)
            T.copy(dQ_shared, DQ[i_b, i_t, i_h * G:(i_h + 1) * G, :BK])

    return bwd_dq_dw


# ============================================================
# Phase 2: bwd_dkdv — 按 KV chunk 并行，计算 dK + dV（无 atomic_add）
# ============================================================
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_bwd_dkdv(
    batch,
    heads,
    q_len,
    kv_len,
    head_dim,
    scale=None,
    block_size=64,
    groups=16,
    selected_blocks=16,
    num_weights=None,
    mask_last_token=True,
    dtype="bfloat16",
    accum_dtype="float",
    num_threads=None,
):
    """
    Phase 2: 按 KV chunk 并行，遍历所有 query tokens。
    通过 BlockMask 跳过无效 query。
    计算 dK 和 dV（累加到 fragment，最后直接写回，无 atomic_add）。
    完全仿照 Triton 版本的 _attn_bwd_dkdv 逻辑。
    """
    enable_last_token_mask = mask_last_token

    if scale is None:
        sm_scale = (1.0 / head_dim)**0.5
    else:
        sm_scale = scale

    scale_log2 = sm_scale * 1.44269504  # log2(e)

    if num_weights is None:
        num_weights = selected_blocks

    B = batch
    BS = block_size
    G = groups
    BK = tilelang.next_power_of_2(head_dim)
    BV = min(128, tilelang.next_power_of_2(head_dim))
    NS_kv = kv_len // BS
    # NV removed (always 1)
    S = selected_blocks

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    do_shape = [batch, q_len, heads, head_dim]
    dk_shape = [batch, kv_len, heads_kv, head_dim]
    dv_shape = [batch, kv_len, heads_kv, head_dim]
    # head-wise: per-q-head weights.
    weight_shape = [batch, q_len, heads, num_weights]
    block_mask_shape = [batch, q_len, heads_kv, NS_kv]

    if num_threads is None:
        num_threads = 128
    num_stages = 0

    scores_lse_shape = [batch, q_len, heads, selected_blocks]
    di_shape = [batch, q_len, heads, selected_blocks]

    @T.prim_func
    def bwd_dkdv(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(k_shape, dtype),
            V: T.Tensor(v_shape, dtype),
            W: T.Tensor(weight_shape, dtype),
            DO: T.Tensor(do_shape, dtype),
            DK: T.Tensor(dk_shape, dtype),
            DV: T.Tensor(dv_shape, dtype),
            BlockMask: T.Tensor(block_mask_shape, "int32"),
            ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
            DiIn: T.Tensor(di_shape, accum_dtype),
    ):
        with T.Kernel(NS_kv, B * heads_kv, threads=num_threads) as (i_s, i_bh):
            i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
            i_s_global = i_s * BS

            # === Shared Memory ===
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            dO_shared = T.alloc_shared([G, BV], dtype)
            dO_weighted_shared = T.alloc_shared([G, BV], dtype)
            # 复用 buffer 存放 P 和 dS
            P_shared = T.alloc_shared([G, BS], dtype)
            dS_shared = T.alloc_shared([G, BS], dtype)

            # === Fragment ===
            acc_s = T.alloc_fragment([G, BS], accum_dtype)

            dV_accum = T.alloc_fragment([BS, BV], accum_dtype)
            dK_accum = T.alloc_fragment([BS, BK], accum_dtype)
            dV_PdO_frag = T.alloc_fragment([G, BS], accum_dtype)
            dS_frag = T.alloc_fragment([G, BS], accum_dtype)

            # Di fragment（用 fragment 而非 shared，避免 shared memory 开销）
            di_frag = T.alloc_fragment([G], accum_dtype)

            # LSE fragment
            lse_shared = T.alloc_fragment([G], accum_dtype)
            # head-wise: per-q-head weight vector of length G.
            chunk_weight = T.alloc_fragment([G], dtype)

            found_pos = T.alloc_var("int32")

            # 加载 K, V（整个 kernel 生命周期内不变）
            T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :BK], K_shared)
            T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, :], V_shared)

            T.fill(dK_accum, 0)
            T.fill(dV_accum, 0)

            for i_q in T.Pipelined(q_len, num_stages=num_stages):
                found_pos = BlockMask[i_b, i_q, i_h, i_s]

                if found_pos != -1:
                    # 加载 Q 和 dO
                    T.copy(Q[i_b, i_q, i_h * G:(i_h + 1) * G, :BK], Q_shared)
                    T.copy(DO[i_b, i_q, i_h * G:(i_h + 1) * G, :], dO_shared)

                    # head-wise: load per-q-head weights for this selected block.
                    for g in T.Parallel(G):
                        chunk_weight[g] = W[i_b, i_q, i_h * G + g, found_pos]

                    # === QK GEMM ===
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # === 加载 LSE 并用 LSE 恢复 P ===
                    for g in T.Parallel(G):
                        lse_shared[g] = ScoresLSE[i_b, i_q, i_h * G + g, found_pos]
                    # T.copy(lse_shared, lse_frag)

                    for g, s in T.Parallel(G, BS):
                        acc_s[g, s] = T.if_then_else(
                            lse_shared[g] > -T.infinity(accum_dtype),
                            T.exp2(acc_s[g, s] * scale_log2 - lse_shared[g]),
                            0.0
                        )
                    if enable_last_token_mask:
                        for g, s in T.Parallel(G, BS):
                            acc_s[g, s] = T.if_then_else(
                                s == BS - 1,
                                0.0,
                                acc_s[g, s]
                            )
                    # 现在 acc_s 是归一化的 P（未乘 weight）

                    # === 计算 dO_weighted = W * dO (per q-head) ===
                    for g, v in T.Parallel(G, BV):
                        dO_weighted_shared[g, v] = chunk_weight[g] * dO_shared[g, v]

                    # === 计算 dV += P^T @ dO_weighted ===
                    T.copy(acc_s, P_shared)
                    T.gemm(P_shared, dO_weighted_shared, dV_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    # === 加载 Phase 1 保存的 Di（用 fragment，非 shared）===
                    for g in T.Parallel(G):
                        di_frag[g] = DiIn[i_b, i_q, i_h * G + g, found_pos]

                    # === 计算 dp = dO @ V^T ===
                    T.clear(dV_PdO_frag)
                    T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # === 计算 dS = sm_scale * W * P * (dp - Di), W is per-q-head ===
                    for g, s in T.Parallel(G, BS):
                        dS_frag[g, s] = sm_scale * chunk_weight[g] * P_shared[g, s] * (dV_PdO_frag[g, s] - di_frag[g])

                    # === 计算 dK += dS^T @ Q ===
                    T.copy(dS_frag, dS_shared)
                    T.gemm(dS_shared, Q_shared, dK_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

            # === 写回 dK 和 dV ===
            T.copy(dK_accum, K_shared)
            T.copy(dV_accum, V_shared)
            T.copy(K_shared, DK[i_b, i_s_global:i_s_global + BS, i_h, :BK])
            T.copy(V_shared, DV[i_b, i_s_global:i_s_global + BS, i_h, :])

    return bwd_dkdv


class _HSA_single(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, w, block_indices,
                block_size: int, sm_scale: float | None,
                mask_last_token: bool = True,
                num_threads_fwd: int | None = None,
                num_threads_bwd: int | None = None):
        """
        Head-wise HSA single (1 token per kernel-block).
        q: (B, L, HQ, D)
        k,v: (B, L, H, D)
        w: (B, L, HQ, S)              # per q-head weights
        block_indices: (B, L, H, S)    # int32, KV block indices in units of block_size
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and w.is_cuda
        assert block_indices.is_cuda

        B, L, HQ, D = q.shape
        H = k.shape[2]
        S = block_indices.shape[-1]
        G = HQ // H
        # 捕获 weights 的真实 shape，num_weights 可能大于 S
        num_weights = w.shape[-1]
        assert HQ % H == 0
        assert L == k.shape[1] == v.shape[1]
        assert w.shape[2] == HQ, \
            f"head-wise W must have HQ heads dim, got w.shape={tuple(w.shape)} expected (B,L,HQ={HQ},*)"

        if sm_scale is None:
            import math
            sm_scale = 1.0 / math.sqrt(D)

        fwd_kernel = hierarchical_sparse_attention(
            batch=B,
            heads=HQ,
            q_len=L,
            kv_len=L,
            head_dim=D,
            block_size=block_size,
            groups=G,
            selected_blocks=S,
            num_weights=num_weights,  # 传入真实的权重维度
            scale=sm_scale,
            mask_last_token=mask_last_token,
            num_threads=num_threads_fwd,
        )

        # 分配 scores_lse（前向 kernel 写入，反向 kernel 复用）
        scores_lse = torch.full((B, L, HQ, S), float('-inf'), dtype=torch.float32, device=q.device)
        O = fwd_kernel(q, k, v, w.to(torch.bfloat16), block_indices, scores_lse)

        ctx.save_for_backward(q, k, v, w, block_indices, scores_lse)
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.G = G
        ctx.mask_last_token = mask_last_token
        ctx.num_threads_bwd = num_threads_bwd
        ctx.num_weights = num_weights  # 保存 num_weights

        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, w, block_indices, scores_lse = ctx.saved_tensors
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        G = ctx.G
        mask_last_token = ctx.mask_last_token
        num_threads_bwd = ctx.num_threads_bwd
        num_weights = ctx.num_weights

        B, L, HQ, D = q.shape
        H = k.shape[2]
        S = block_indices.shape[-1]
        device = q.device

        build_mask = hsa_kernel_block_mask(
            batch=B,
            heads=H,
            q_len=L,
            kv_len=L,
            selected_blocks=S,
            block_size=block_size,
        )
        NS_kv = L // block_size
        block_mask = torch.full(
            (B, L, H, NS_kv),
            -1,
            dtype=torch.int32,
            device=device,
        )
        build_mask(block_indices, block_mask)

        bwd_kernel = hierarchical_sparse_attention_bwd_dqkv(
            batch=B,
            heads=HQ,
            q_len=L,
            kv_len=L,
            head_dim=D,
            block_size=block_size,
            groups=G,
            selected_blocks=S,
            num_weights=num_weights,  # 传入真实的权重维度
            scale=sm_scale,
            mask_last_token=mask_last_token,
            num_threads=num_threads_bwd,
        )
        # NV removed (always 1)
        
        DQ = torch.zeros(
            (B, L, HQ, D),
            dtype=torch.float32,
            device=device,
        )
        DK = torch.zeros(
            (B, L, H, D),
            dtype=torch.bfloat16,
            device=device,
        )
        DV = torch.zeros(
            (B, L, H, D),
            dtype=torch.bfloat16,
            device=device,
        )
        # head-wise: DW is per q-head.
        DW = torch.zeros(
            (B, L, HQ, num_weights),
            dtype=torch.bfloat16,
            device=device,
        )

        bwd_kernel(q, k, v, w.to(torch.bfloat16), dO, DQ, DK, DV, DW, block_mask, scores_lse)

        # DQ 使用了 swizzle layout (annotate_layout)，需要 post_kernel 反 swizzle
        post_kernel = hsa_bwd_postprocess(B, L, HQ, D)
        DQ = post_kernel(DQ)
        DQ = DQ.to(torch.bfloat16)

        return DQ, DK, DV, DW.float(), None, None, None, None, None, None


def HSA_single(q, k, v, w, block_indices,
               block_size: int = 64,
               sm_scale: float | None = None,
               mask_last_token: bool = True,
               num_threads_fwd: int | None = None,
               num_threads_bwd: int | None = None):
    return _HSA_single.apply(q, k, v, w, block_indices, block_size, sm_scale,
                             mask_last_token, num_threads_fwd, num_threads_bwd)


# ============================================================
# 两阶段反向版本的 Autograd 类
# ============================================================
class _HSA_single_two_phase(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, w, block_indices,
                block_size: int, sm_scale: float | None,
                mask_last_token: bool = True,
                num_threads_fwd: int | None = None,
                num_threads_bwd: int | None = None):
        """Forward identical to _HSA_single (head-wise, w: (B, L, HQ, S))"""
        assert q.is_cuda and k.is_cuda and v.is_cuda and w.is_cuda
        assert block_indices.is_cuda

        B, L, HQ, D = q.shape
        H = k.shape[2]
        S = block_indices.shape[-1]
        G = HQ // H
        num_weights = w.shape[-1]
        assert HQ % H == 0
        assert L == k.shape[1] == v.shape[1]
        assert w.shape[2] == HQ, \
            f"head-wise W must have HQ heads dim, got w.shape={tuple(w.shape)} expected (B,L,HQ={HQ},*)"

        if sm_scale is None:
            import math
            sm_scale = 1.0 / math.sqrt(D)

        fwd_kernel = hierarchical_sparse_attention(
            batch=B, heads=HQ, q_len=L, kv_len=L, head_dim=D,
            block_size=block_size, groups=G, selected_blocks=S,
            num_weights=num_weights, scale=sm_scale,
            mask_last_token=mask_last_token, num_threads=num_threads_fwd,
        )

        # 分配 scores_lse 并传给前向 kernel（in-place 写入）
        scores_lse = torch.full((B, L, HQ, S), float('-inf'), dtype=torch.float32, device=q.device)
        O = fwd_kernel(q, k, v, w.to(torch.bfloat16), block_indices, scores_lse)

        ctx.save_for_backward(q, k, v, w, block_indices, scores_lse)
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.G = G
        ctx.mask_last_token = mask_last_token
        ctx.num_threads_bwd = num_threads_bwd
        ctx.num_weights = num_weights

        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, w, block_indices, scores_lse = ctx.saved_tensors
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        G = ctx.G
        mask_last_token = ctx.mask_last_token
        num_threads_bwd = ctx.num_threads_bwd
        num_weights = ctx.num_weights

        B, L, HQ, D = q.shape
        H = k.shape[2]
        S = block_indices.shape[-1]
        device = q.device

        # NV removed (always 1)

        # ========== Phase 1: 计算 dQ + dW ==========
        bwd_dq_dw_kernel = hierarchical_sparse_attention_bwd_dq_dw(
            batch=B, heads=HQ, q_len=L, kv_len=L, head_dim=D,
            block_size=block_size, groups=G, selected_blocks=S,
            num_weights=num_weights, scale=sm_scale,
            mask_last_token=mask_last_token, num_threads=num_threads_bwd,
        )

        DQ = torch.zeros((B, L, HQ, D), dtype=torch.float32, device=device)
        # head-wise: DW is per q-head.
        DW = torch.zeros((B, L, HQ, num_weights), dtype=torch.float32, device=device)
        DiOut = torch.zeros((B, L, HQ, S), dtype=torch.float32, device=device)

        bwd_dq_dw_kernel(q, k, v, w.to(torch.bfloat16), dO, block_indices, scores_lse, DQ, DW, DiOut)

        # ========== Phase 2: 计算 dK + dV ==========
        # 构建 BlockMask
        build_mask = hsa_kernel_block_mask(
            batch=B, heads=H, q_len=L, kv_len=L,
            selected_blocks=S, block_size=block_size,
        )
        NS_kv = L // block_size
        block_mask = torch.full((B, L, H, NS_kv), -1, dtype=torch.int32, device=device)
        build_mask(block_indices, block_mask)

        bwd_dkdv_kernel = hierarchical_sparse_attention_bwd_dkdv(
            batch=B, heads=HQ, q_len=L, kv_len=L, head_dim=D,
            block_size=block_size, groups=G, selected_blocks=S,
            num_weights=num_weights, scale=sm_scale,
            mask_last_token=mask_last_token, num_threads=num_threads_bwd,
        )

        DK = torch.zeros((B, L, H, D), dtype=torch.bfloat16, device=device)
        DV = torch.zeros((B, L, H, D), dtype=torch.bfloat16, device=device)

        bwd_dkdv_kernel(q, k, v, w.to(torch.bfloat16), dO, DK, DV, block_mask, scores_lse, DiOut)

        # ========== 后处理 ==========
        # NV removed: DQ/DK are already 4D, no need for sum(0)

        return DQ, DK, DV, DW, None, None, None, None, None, None


def HSA_single_two_phase(q, k, v, w, block_indices,
                         block_size: int = 64,
                         sm_scale: float | None = None,
                         mask_last_token: bool = True,
                         num_threads_fwd: int | None = None,
                         num_threads_bwd: int | None = None):
    return _HSA_single_two_phase.apply(q, k, v, w, block_indices, block_size, sm_scale,
                                       mask_last_token, num_threads_fwd, num_threads_bwd)


import math
import torch
import torch.nn.functional as F
from einops import rearrange

def main_block_M_correctness():

    import math
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    # ---------- 配置参数（模拟 bench 场景：H=2, HQ=32, mask_last_token=True）----------
    # 原始正确性测试参数：B=1, SEQ_LEN=1024, H=1, HQ=16, mask_last_token=False
    # bench 测速报错场景：B=4, SEQ_LEN=8322, H=2, HQ=32, mask_last_token=True
    # 这里只改关键参数 H, HQ, mask_last_token，保持 B, SEQ_LEN 较小以加速 Python 循环构造 indices
    B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 2, 32, 128, 16, 64
    MASK_LAST_TOKEN = True  # 新增：与 bench 一致
    dtype = torch.bfloat16
    device = "cuda"
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
    
    # Generate weights: only valid blocks get non-zero probability; invalid logits are
    # masked with -inf so softmax outputs ~0 for them. W remains a leaf tensor (requires_grad=True).
    # head-wise: W is per q-head, shape (B, L, HQ, S).
    logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=torch.float32, device=device)
    valid_mask_h = (block_indices != -1)  # (B, L, H, S)
    valid_mask_hq = torch.repeat_interleave(valid_mask_h, dim=-2, repeats=G)  # (B, L, HQ, S)
    logits = logits.masked_fill(~valid_mask_hq, float('-inf'))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, nan=0.0).detach().requires_grad_(True)  # invalid slots exactly 0
    
    
    # 用于反向传播的梯度
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ========== 测试前向传播 ==========
    
    
    # Torch reference 前向（与 tilelang kernel 对齐，传入 mask_last_token）
    O_ref = hsa_torch_ref(
        Q.float().detach(), 
        K.float().detach(), 
        V.float().detach(), 
        W.detach(), 
        block_indices,
        chunk_size=block_size, 
        sm_scale=scale, 
        block_q=1,
        mask_last_token=MASK_LAST_TOKEN,
    )


    # ========== 测试反向传播 ==========
    
    # ====== 先计算 Torch reference 反向 ======
    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None
    
    O_ref_bwd = hsa_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices,
        chunk_size=block_size, sm_scale=scale, block_q=1,
        mask_last_token=MASK_LAST_TOKEN,
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
    
    
    # 比较梯度
    def compute_grad_diff(grad_hsa, grad_ref, name):
        if grad_hsa is None or grad_ref is None:
            print(f"{name}: grad is None")
            return None
        diff = (grad_hsa.float() - grad_ref.float()).abs()
        max_diff = diff.max().item()
        print(f"{name} max error: {max_diff:.6e}")
        return max_diff
    
    
    BLOCK_Q = 1  # keep consistent with tilelang/test setup (block_q>1 needs matching aggregation)
    q_blocks = SEQ_LEN // BLOCK_Q

    # Aggregate: take the first token of each q_block as the representative.
    # head-wise: weights are (B, q_blocks, HQ, S); indices remain per h_kv (B, q_blocks, H, S).
    indices_blocks = block_indices.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :].contiguous()
    weights_blocks = W.view(B, q_blocks, BLOCK_Q, HQ, S)[:, :, 0, :, :].contiguous()

    # Use original block_indices (with -1 sentinels); the kernel skips invalid blocks via
    # `if i_s >= 0` and build_block_mask skips via `if block_idx >= 0`.
    indices_blocks_hsa = indices_blocks.clone()
    invalid_mask_blocks_h = (indices_blocks_hsa < 0)  # (B, q_blocks, H, S)

    weights_blocks_hsa = weights_blocks.detach().clone()
    weights_blocks_hsa.requires_grad_(True)
    indices_blocks_hsa = indices_blocks_hsa.contiguous()
    weights_blocks_hsa = weights_blocks_hsa.contiguous()

    # head-wise single kernel does NOT require HQ padding (the kernel parallelizes over
    # head_kv at the grid level and over G inside a thread block via T.Parallel(G, BS)).
    Q_hsa = Q.detach().clone().requires_grad_(True)
    grad_output_hsa = grad_output

    # Clear any leftover grads.
    K.grad = None
    V.grad = None

    # Call HSA_single (head-wise).
    O_triton_hsa = HSA_single(Q_hsa, K, V, weights_blocks_hsa, indices_blocks_hsa, block_size=block_size, sm_scale=scale, mask_last_token=MASK_LAST_TOKEN)
    print()
    print("[Tilelang HSA head-wise single] vs [Torch Reference]:")
    fwd_err = (O_triton_hsa.float() - O_ref.float()).abs().max().item()
    print(f"FWD max error: {fwd_err:.6e}")

    # backward
    O_triton_hsa.backward(grad_output_hsa)

    DQ_triton = Q_hsa.grad.clone() if Q_hsa.grad is not None else None
    DK_triton = K.grad.clone() if K.grad is not None else None
    DV_triton = V.grad.clone() if V.grad is not None else None
    DW_triton_blocks = weights_blocks_hsa.grad.clone() if weights_blocks_hsa.grad is not None else None

    # head-wise: DW_ref is per q-head (B, L, HQ, S) -> aggregate per block.
    DW_ref_blocks = DW_ref.view(B, q_blocks, BLOCK_Q, HQ, S)[:, :, 0, :, :]
    # Mask invalid block slots (broadcast h->HQ) before comparison.
    invalid_mask_blocks_hq = torch.repeat_interleave(invalid_mask_blocks_h, dim=-2, repeats=G)

    compute_grad_diff(DQ_triton, DQ_ref, "DQ")
    compute_grad_diff(DK_triton, DK_ref, "DK")
    compute_grad_diff(DV_triton, DV_ref, "DV")
    DW_triton_blocks = weights_blocks_hsa.grad.clone()
    DW_triton_blocks[invalid_mask_blocks_hq] = 0  # mask invalid slots
    compute_grad_diff(DW_triton_blocks, DW_ref_blocks, "DW")

    # ========== Test the two-phase backward variant ==========
    print()
    print("=" * 60)
    print("[Two-Phase BWD] head-wise two-phase backward")
    print("=" * 60)

    # Clear grads
    Q.grad = None
    K.grad = None
    V.grad = None

    Q_hsa2 = Q.detach().clone().requires_grad_(True)
    K2 = K.detach().clone().requires_grad_(True)
    V2 = V.detach().clone().requires_grad_(True)
    weights_blocks_hsa2 = weights_blocks.detach().clone().requires_grad_(True)

    O_two_phase = HSA_single_two_phase(
        Q_hsa2, K2, V2, weights_blocks_hsa2, indices_blocks_hsa,
        block_size=block_size, sm_scale=scale, mask_last_token=MASK_LAST_TOKEN
    )

    fwd_err2 = (O_two_phase.float() - O_ref.float()).abs().max().item()
    print(f"FWD max error: {fwd_err2:.6e}")

    O_two_phase.backward(grad_output_hsa)

    DQ_two = Q_hsa2.grad.clone() if Q_hsa2.grad is not None else None
    DK_two = K2.grad.clone() if K2.grad is not None else None
    DV_two = V2.grad.clone() if V2.grad is not None else None
    DW_two_blocks = weights_blocks_hsa2.grad.clone() if weights_blocks_hsa2.grad is not None else None

    compute_grad_diff(DQ_two, DQ_ref, "DQ")
    compute_grad_diff(DK_two, DK_ref, "DK")
    compute_grad_diff(DV_two, DV_ref, "DV")
    if DW_two_blocks is not None:
        DW_two_blocks[invalid_mask_blocks_hq] = 0
    compute_grad_diff(DW_two_blocks, DW_ref_blocks, "DW")


if __name__ == "__main__":
    main_block_M_correctness()


# python ops/hsa_fwd_bwd_single_tilelang.py
