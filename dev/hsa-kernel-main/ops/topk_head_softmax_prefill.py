import torch
import tilelang
import tilelang.language as T
import math

def ref_prefill_softmax_topk_max_pooling(q, k_lmks, lse_swa, seq_lens, topk, block_size, window_size, is_causal=False):
    """
    Reference implementation for Softmax-then-Max Top-K strategy.
    
    Args:
        q: [B, L, h_kv, G, D]
        k_lmks: [B, S, h_kv, D]
        lse_swa: [B, L, h_q] or [B, L, h_kv, G]
        topk: int
        block_size: int
        window_size: int
        is_causal: bool
        seq_lens: [B] Tensor containing valid sequence lengths.
        
    Returns:
        indices_sorted: [B, L, h_kv, topk] (按分数排序，无效位置为-1)
        scores_sorted: [B, L, h_kv, G, topk] (Raw Logits，无效位置为-inf)
    """
    B, L, h_kv, G, D = q.shape
    S = k_lmks.shape[1]
    
    logits_hsa = torch.einsum("blhgd,bshd->blhgs", q.float(), k_lmks.float())
    
    sm_scale = 1.0 / math.sqrt(D)
    logits_hsa_scaled = logits_hsa * sm_scale

    if seq_lens is not None:
        seq_idx = torch.arange(L, device=q.device).unsqueeze(0) # [1, L]
        len_mask = seq_idx >= seq_lens.unsqueeze(1) # [B, L]
        len_mask_expanded = len_mask.view(B, L, 1, 1, 1)
        logits_hsa_scaled = logits_hsa_scaled.masked_fill(len_mask_expanded, float('-inf'))

    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1) # [L, 1]
        j_idx = torch.arange(S, device=q.device).unsqueeze(0) # [1, S]
        
        # Calculate effective number of landmarks for each batch (only complete blocks)
        # s_len_valid[b] = floor(seq_lens[b] / block_size)
        s_len_valid = seq_lens.div(block_size, rounding_mode='floor') # [B]
        
        valid_lmks_mask = (j_idx < s_len_valid.unsqueeze(1)).float() # [B, 1, S]
        
        threshold_idx = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        threshold_idx = torch.max(threshold_idx, torch.tensor(0, device=q.device))
        
        causal_mask = j_idx >= threshold_idx # [L, S]
        
        combined_mask = causal_mask.unsqueeze(0)  # [1, L, S]
        combined_mask = combined_mask | (valid_lmks_mask == 0).unsqueeze(1) # [B, L, S]
        
        combined_mask_expanded = combined_mask.view(B, L, 1, 1, S)
        logits_hsa_scaled = logits_hsa_scaled.masked_fill(combined_mask_expanded, float('-inf'))

    lse_hsa = torch.logsumexp(logits_hsa_scaled, dim=-1)
    
    if lse_swa.dim() == 3:
        lse_swa_view = lse_swa.view(B, L, h_kv, G)
    else:
        lse_swa_view = lse_swa
        
    lse_total = torch.logaddexp(lse_swa_view, lse_hsa)
    
    log_probs = logits_hsa_scaled - lse_total.unsqueeze(-1)
    
    scores_max_pooling = log_probs.max(dim=3).values
    
    topk_scores, topk_indices = torch.topk(scores_max_pooling, k=topk, dim=-1, sorted=True)
    
    invalid_mask = topk_scores == float('-inf')
    
    huge_idx = S + 10000
    topk_indices = topk_indices.masked_fill(invalid_mask, huge_idx)
    topk_indices = topk_indices.masked_fill(topk_indices == huge_idx, -1)
    
    indices_expanded = topk_indices.unsqueeze(3).expand(-1, -1, -1, G, -1)
    
    safe_indices = indices_expanded.clamp(min=0)
    scores_sorted = torch.gather(logits_hsa_scaled, -1, safe_indices)
    
    scores_sorted = scores_sorted.masked_fill(indices_expanded == -1, float('-inf'))

    return topk_indices, scores_sorted


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hsa_prefill_lse_kernel(
    batch, h_kv, groups, head_dim, block_size, window_size, is_causal,
    BLOCK_L=None, BLOCK_S=None, threads=None
):
    dtype = "bfloat16"
    accum_dtype = "float"
    
    # Dynamic dimension declaration
    max_seq_len = T.dynamic("max_seq_len")
    max_s_len = T.dynamic("max_s_len")
    
    q_shape = [batch, max_seq_len, h_kv, groups, head_dim]
    k_shape = [batch, max_s_len, h_kv, head_dim]
    lse_shape = [batch, max_seq_len, h_kv, groups]
    len_shape = [batch]

    if BLOCK_L is None: BLOCK_L = 16
    if BLOCK_S is None: BLOCK_S = 64
    if threads is None: threads = 128
    
    GEMM_M = BLOCK_L * groups
    GEMM_N = BLOCK_S
    GEMM_K = head_dim

    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        SeqLens: T.Tensor(len_shape, "int32"),
        LSE_Out: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(max_seq_len, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            i_b, i_h = bz, by
            base_l = bx * BLOCK_L
            
            cur_seq_len = SeqLens[i_b]
            s_len_req = cur_seq_len // block_size
            
            Q_shared = T.alloc_shared([GEMM_M, GEMM_K], dtype)
            K_shared = T.alloc_shared([GEMM_N, GEMM_K], dtype)
            score_shared = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)
            
            m_curr = T.alloc_fragment([GEMM_M], accum_dtype)
            m_prev = T.alloc_fragment([GEMM_M], accum_dtype)
            l_prev = T.alloc_fragment([GEMM_M], accum_dtype)
            
            scores_max = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_sum = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_scale = T.alloc_fragment([GEMM_M], accum_dtype)

            T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})

            T.fill(m_prev, -T.infinity(accum_dtype))
            T.fill(l_prev, 0.0)

            # Load Q
            for i, j in T.Parallel(GEMM_M, GEMM_K):
                l_idx = i // groups
                g = i % groups
                tq = base_l + l_idx
                if tq < cur_seq_len:
                    Q_shared[i, j] = Q[i_b, tq, i_h, g, j]
                else:
                    Q_shared[i, j] = 0.0

            # Calculate loop cycles
            loop_limit = T.alloc_var("int32")
            loop_limit = T.ceildiv(s_len_req, BLOCK_S)
            
            if is_causal:
                max_tq_in_block = base_l + BLOCK_L - 1
                max_perm_threshold = (max_tq_in_block - window_size + 1) // block_size
                
                causal_limit = T.alloc_var("int32")
                if max_perm_threshold < 0:
                    causal_limit = 0
                else:
                    causal_limit = T.ceildiv(max_perm_threshold, BLOCK_S)
                
                loop_limit = T.min(loop_limit, causal_limit)

            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S
                
                # Load K
                for i, j in T.Parallel(GEMM_N, GEMM_K):
                    ts = base_s + i
                    if ts < s_len_req:
                        K_shared[i, j] = K[i_b, ts, i_h, j]
                    else:
                        K_shared[i, j] = 0.0
                
                T.sync_threads()
                
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_s, score_shared)
                
                # Apply Mask
                if is_causal:
                    for i, j in T.Parallel(GEMM_M, GEMM_N):
                        l_idx = i // groups
                        tq = base_l + l_idx
                        ts = base_s + j
                        # Validity checks plus HSA Window Mask logic
                        if tq < cur_seq_len and ts < s_len_req:
                            threshold = (tq - window_size + 1) // block_size
                            if ts >= threshold:
                                score_shared[i, j] = -T.infinity(accum_dtype)
                        else:
                            score_shared[i, j] = -T.infinity(accum_dtype)
                
                T.sync_threads()
                T.copy(score_shared, acc_s)
                
                # Online Softmax updates
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
                    val = acc_s[i, j] * sm_scale
                    if val == -T.infinity(accum_dtype) and m_prev[i] == -T.infinity(accum_dtype):
                        acc_s[i, j] = 0.0
                    else:
                        acc_s[i, j] = T.exp(val - m_prev[i])
                
                T.reduce_sum(acc_s, scores_sum, dim=1)
                
                for i in T.Parallel(GEMM_M):
                    l_prev[i] = l_prev[i] * scores_scale[i] + scores_sum[i]
                
                T.sync_threads()
            
            # Write Output
            for i in T.Parallel(GEMM_M):
                l_idx = i // groups
                g = i % groups
                tq = base_l + l_idx
                if tq < cur_seq_len:
                    if l_prev[i] == 0:
                        LSE_Out[i_b, tq, i_h, g] = -T.infinity(accum_dtype)
                    else:
                        LSE_Out[i_b, tq, i_h, g] = m_prev[i] + T.log(l_prev[i])
                else:
                    LSE_Out[i_b, tq, i_h, g] = -T.infinity(accum_dtype)

    return kernel


@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hsa_prefill_weighted_select_kernel(
    batch, h_kv, groups, head_dim, topk, block_size, window_size, is_causal,
    BLOCK_L=None, BLOCK_S=None, threads=None
):
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"
    
    # Dynamic dimension declaration
    max_seq_len = T.dynamic("max_seq_len")
    max_s_len = T.dynamic("max_s_len")
    
    # Shape definitions
    q_shape = [batch, max_seq_len, h_kv, groups, head_dim]
    k_shape = [batch, max_s_len, h_kv, head_dim]
    lse_shape = [batch, max_seq_len, h_kv, groups]
    len_shape = [batch]
    out_indices_shape = [batch, max_seq_len, h_kv, topk]

    if BLOCK_L is None: BLOCK_L = 16
    if BLOCK_S is None: BLOCK_S = 16
    if threads is None: threads = 128
    
    GEMM_M = BLOCK_L * groups
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        LSE_Total: T.Tensor(lse_shape, accum_dtype),
        SeqLens: T.Tensor(len_shape, idx_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(max_seq_len, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            i_b, i_h = bz, by
            base_l = bx * BLOCK_L
            
            cur_seq_len = SeqLens[i_b]
            s_len_req = cur_seq_len // block_size

            
            Q_shared = T.alloc_shared([GEMM_M, head_dim], dtype)
            K_shared = T.alloc_shared([BLOCK_S, head_dim], dtype)
            score_shared = T.alloc_shared([GEMM_M, BLOCK_S], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, BLOCK_S], accum_dtype)
            
            topk_max_scores = T.alloc_local([topk], accum_dtype)
            topk_indices = T.alloc_local([topk], idx_dtype)
            lse_local = T.alloc_local([groups], accum_dtype)
            
            T.fill(topk_max_scores, -T.infinity(accum_dtype))
            T.fill(topk_indices, -1)
            
            tx = T.get_thread_binding()
            
            # Load LSE - only load valid positions
            if tx < BLOCK_L and (base_l + tx) < cur_seq_len:
                for g in T.serial(groups):
                    lse_local[g] = LSE_Total[i_b, base_l + tx, i_h, g]
            
            # Load Q
            for l_idx, g, d in T.Parallel(BLOCK_L, groups, head_dim):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < cur_seq_len:
                    Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                else:
                    Q_shared[flat_m, d] = 0
            
            # [FIX 2] Correct Loop Limit logic for Causal Masking
            loop_limit = T.alloc_var("int32")
            loop_limit = T.ceildiv(s_len_req, BLOCK_S)
            
            if is_causal:
                max_tq = base_l + BLOCK_L - 1
                threshold = (max_tq - window_size + 1) // block_size
                causal_limit = T.alloc_var("int32")
                if threshold <= 0:
                    causal_limit = 0
                else:
                    causal_limit = T.ceildiv(threshold, BLOCK_S)
                
                loop_limit = T.min(loop_limit, causal_limit)
            
            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S
                
                # Load K - check ts < s_len_req
                for s_idx, d in T.Parallel(BLOCK_S, head_dim):
                    ts = base_s + s_idx
                    if ts < s_len_req:
                        K_shared[s_idx, d] = K[i_b, ts, i_h, d]
                    else:
                        K_shared[s_idx, d] = 0
                T.sync_threads()
                
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_s, score_shared)
                
                # Masking Shared - use dynamic values
                if is_causal:
                    for i, j in T.Parallel(GEMM_M, BLOCK_S):
                        l_idx = i // groups
                        tq = base_l + l_idx
                        ts = base_s + j
                        # Validity checks: sequence bounds + causal window
                        if tq < cur_seq_len and ts < s_len_req:
                            if ts >= (tq - window_size + 1) // block_size:
                                score_shared[i, j] = -T.infinity(accum_dtype)
                        else:
                            score_shared[i, j] = -T.infinity(accum_dtype)
                T.sync_threads()
                
                # TopK Selection
                if tx < BLOCK_L and (base_l + tx) < cur_seq_len:
                    my_l_idx = tx
                    val = T.alloc_var(accum_dtype)
                    norm_score = T.alloc_var(accum_dtype)
                    cur_max_norm_score = T.alloc_var(accum_dtype)
                    
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        if ts < s_len_req:
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
            
            # Write Output - handle padding positions
            if tx < BLOCK_L:
                tq = base_l + tx
                if tq < cur_seq_len:
                    for k in T.serial(topk):
                        OutIndices[i_b, tq, i_h, k] = topk_indices[k]
                else:
                    # Fill padding positions with -1
                    for k in T.serial(topk):
                        OutIndices[i_b, tq, i_h, k] = -1

    return kernel

@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hsa_prefill_recompute_scores_kernel(
    batch, h_kv, groups, head_dim, topk,
    BLOCK_L=None, BLOCK_TK=None, threads=None
):
    """
    Prefill version of recompute scores kernel.
    Accepts dynamic shape inputs.
    
    Q:        [B, max_seq_len, h_kv, G, D]
    K:        [B, max_s_len, h_kv, D]
    Indices:  [B, max_seq_len, h_kv, topk]
    OutScores:[B, max_seq_len, h_kv, G, topk]
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    # Dynamic dimensions
    max_seq_len = T.dynamic("max_seq_len")
    max_s_len = T.dynamic("max_s_len")

    q_shape = [batch, max_seq_len, h_kv, groups, head_dim]
    k_shape = [batch, max_s_len, h_kv, head_dim]
    indices_shape = [batch, max_seq_len, h_kv, topk]
    out_scores_shape = [batch, max_seq_len, h_kv, groups, topk]

    if BLOCK_L is None:
        BLOCK_L = 2
    if BLOCK_TK is None:
        BLOCK_TK = 32
    BLOCK_D = head_dim
    if threads is None:
        threads = 64
    MIN_GEMM_M = 16
    GEMM_M = BLOCK_L * groups
    if GEMM_M < MIN_GEMM_M:
        BLOCK_L = T.ceildiv(MIN_GEMM_M, groups)
        GEMM_M = BLOCK_L * groups
    GEMM_N = BLOCK_L * BLOCK_TK
    tk_blocks = T.ceildiv(topk, BLOCK_TK)

    sm_scale = 1.0 / math.sqrt(head_dim)
    
    @T.prim_func
    def fwd_recompute(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        Indices: T.Tensor(indices_shape, idx_dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(max_seq_len, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L
            
            Q_shared = T.alloc_shared([GEMM_M, BLOCK_D], dtype)
            K_shared = T.alloc_shared([BLOCK_L * BLOCK_TK, BLOCK_D], dtype)
            score_shared = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)

            # Load Q
            for l_idx, g, d in T.Parallel(BLOCK_L, groups, BLOCK_D):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < max_seq_len:
                    Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                else:
                    Q_shared[flat_m, d] = T.Cast(dtype, 0.0)

            # Iterate over TopK blocks
            for tk_block in T.serial(tk_blocks):
                tk_base = tk_block * BLOCK_TK
                tk_size = T.min(BLOCK_TK, topk - tk_base)

                # Gather K based on Indices
                for l_idx, tk_idx, d in T.Parallel(BLOCK_L, BLOCK_TK, BLOCK_D):
                    tq = base_l + l_idx
                    off = l_idx * BLOCK_TK + tk_idx
                    if (tq < max_seq_len) and (tk_idx < tk_size):
                        k_id = tk_base + tk_idx
                        idx = Indices[i_b, tq, i_h, k_id]
                        
                        # Boundary check: Valid index and within physical storage
                        if (idx >= 0) and (idx < max_s_len):
                            K_shared[off, d] = K[i_b, idx, i_h, d]
                        else:
                            # Pad or Invalid index -> 0 vector (score will be masked later or small)
                            K_shared[off, d] = T.Cast(dtype, 0.0)
                    else:
                        if off < BLOCK_L * BLOCK_TK:
                            K_shared[off, d] = T.Cast(dtype, 0.0)
                T.sync_threads()

                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow
                )
                T.copy(acc_s, score_shared)
                T.sync_threads()

                # Write out scores
                for l_idx, g, tk_idx in T.Parallel(BLOCK_L, groups, BLOCK_TK):
                    tq = base_l + l_idx
                    if (tq < max_seq_len) and (tk_idx < tk_size):
                        k_id = tk_base + tk_idx
                        idx = Indices[i_b, tq, i_h, k_id]
                        
                        # If index is invalid (-1), write -inf
                        if idx < 0:
                            OutScores[i_b, tq, i_h, g, k_id] = -T.infinity(accum_dtype)
                        else:
                            row = l_idx * groups + g
                            col = l_idx * BLOCK_TK + tk_idx
                            val = score_shared[row, col]
                            OutScores[i_b, tq, i_h, g, k_id] = val * sm_scale

    return fwd_recompute





class SoftmaxTopKMaxPooling_Prefill_Fused(torch.nn.Module):
    def __init__(self, topk, block_size=32, window_size=512, is_causal=False):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        
        self._lse_kernel = None
        self._select_kernel = None
        self._recompute_kernel = None

    def forward(self, q, lmks, lse_swa, seq_lens):
        B, L, h_kv, G, D = q.shape
        topk = self.topk
        
        self._lse_kernel = hsa_prefill_lse_kernel(
            B, h_kv, G, D,
            block_size=self.block_size, window_size=self.window_size, is_causal=self.is_causal,
        )
        self._select_kernel = hsa_prefill_weighted_select_kernel(
            B, h_kv, G, D, topk,
            block_size=self.block_size, window_size=self.window_size, is_causal=self.is_causal,
        )
        self._recompute_kernel = hsa_prefill_recompute_scores_kernel(
            B, h_kv, G, D, topk,
        )
        
        q = q.contiguous()
        lmks = lmks.contiguous()
        seq_lens = seq_lens.to(torch.int32).contiguous()
        
        # 1. LSE_hsa
        lse_hsa = self._lse_kernel(q, lmks, seq_lens)
        
        # 2. Combine LSE
        lse_swa_view = lse_swa.view(B, L, h_kv, G).float()
        lse_total = torch.logaddexp(lse_swa_view, lse_hsa)
        
        # 3. Select Weighted Indices
        indices_raw = self._select_kernel(q, lmks, lse_total, seq_lens)
        
        # 5. Recompute Scores
        scores_grouped = self._recompute_kernel(q, lmks, indices_raw)
        
        scores = scores_grouped.view(B, L, h_kv * G, topk).to(q.dtype)
        
        return indices_raw, scores

_SOFTMAX_MODULE_CACHE = {}

def online_softmax_topk_head(q: torch.Tensor, lmks: torch.Tensor, lse_swa: torch.Tensor, seq_lens: torch.Tensor,
                             topk: int, block_size: int, window_size: int, is_causal: bool = False):
    """
    Functional API for SoftmaxTopKMaxPooling_Prefill_Fused with Variable Length Support
    """
    if q.dim() == 4:
        B, L, h_q, D = q.shape
        h_kv = lmks.shape[2]
        assert h_q % h_kv == 0
        G = h_q // h_kv
        q = q.view(B, L, h_kv, G, D)
    
    cache_key = (topk, block_size, window_size, is_causal)
    
    if cache_key not in _SOFTMAX_MODULE_CACHE:
        _SOFTMAX_MODULE_CACHE[cache_key] = SoftmaxTopKMaxPooling_Prefill_Fused(topk, block_size, window_size, is_causal)

    return _SOFTMAX_MODULE_CACHE[cache_key](q, lmks, lse_swa, seq_lens)


def test_fused_topk_softmax_max_pooling_correctness():
    print("\n" + "=" * 70)
    print("=== Testing Fused Softmax TopK Max-Pooling Kernel Correctness (VarLen) ===")
    print("=" * 70)
    
    B, L, D = 8, 1024, 64 
    h_kv = 2
    G = 8
    S = 128
    topk = 16
    is_causal = True
    block_size = 32
    window_size = 64
    
    dtype = torch.bfloat16
    device = "cuda"
    
    seq_lens = torch.randint(low=1, high=L+1, size=(B,), device=device, dtype=torch.int32)
    seq_lens[0] = -1
    
    print(f"Config: B={B}, MaxL={L}, S={S}, h_kv={h_kv}, G={G}, topk={topk}")
    print(f"SeqLens sample: {seq_lens[:4].tolist()}")

    torch.manual_seed(42)
    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)
    lse_swa = torch.randn(B, L, h_kv, G, dtype=dtype, device=device)*5
    
    # 1. Forward Reference
    ref_indices, ref_scores = ref_prefill_softmax_topk_max_pooling(
        q.detach(), lmks.detach(), lse_swa.detach().float(), seq_lens, topk, block_size, window_size, is_causal
    )
    
    # 2. Forward Fused Kernel
    fused_indices, fused_scores = online_softmax_topk_head(
        q, lmks, lse_swa, seq_lens, topk, block_size, window_size, is_causal
    )
    # Reshape fused scores for alignment
    fused_scores_reshaped = fused_scores.view(B, L, h_kv, G, topk)
    
    # Check Score Difference - Only check valid positions where ref_scores > -inf
    mask_valid = ref_scores > float('-inf')
    if mask_valid.sum() > 0:
        score_diff = (fused_scores_reshaped[mask_valid].float() - ref_scores[mask_valid].float()).abs().max()
    else:
        score_diff = torch.tensor(0.0)
    
    indices_match = (fused_indices == ref_indices).float().mean()
    
    print(f"\nForward Max Score Diff: {score_diff.item():.6f}")
    print(f"Forward Index Match Rate: {indices_match.item()*100:.2f}%")
    
    if indices_match.item() > 0.99 and score_diff.item() < 5e-2:
        print("✅ Forward Validated")
    else:
        print("❌ Forward Failed")
        # Debug info
        print("\nDebug - Ref[0,0,0]:", ref_indices[0,0,0])
        print("Debug - Fused[0,0,0]:", fused_indices[0,0,0])

if __name__ == "__main__":
    test_fused_topk_softmax_max_pooling_correctness()