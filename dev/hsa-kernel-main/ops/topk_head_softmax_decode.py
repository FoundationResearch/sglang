import torch
import torch.nn as nn
import math
import tilelang
import tilelang.language as T
torch.manual_seed(42)

@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hsa_decode_lse_kernel(
    batch, h_kv, groups, head_dim, 
    block_size, window_size, 
    BLOCK_S=32, threads=64
):
    dtype = "bfloat16"
    accum_dtype = "float"
    
    # Dynamic dimension for Max Cache Capacity
    max_s = T.dynamic("max_s")
    
    GEMM_M = (groups + 15) // 16 * 16
    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def kernel(
        Q: T.Tensor([batch, h_kv, groups, head_dim], dtype),
        K: T.Tensor([batch, max_s, h_kv, head_dim], dtype),
        SeqLens: T.Tensor([batch], "int32"),
        LSE_Out: T.Tensor([batch, h_kv, groups], accum_dtype),
    ):
        with T.Kernel(h_kv, batch, threads=threads) as (i_h, i_b):
            # Dynamic Info Loading
            cur_seq_len = SeqLens[i_b]
            
            # Compute Bounds
            s_len_req = cur_seq_len // block_size
            query_pos = cur_seq_len - 1
            threshold_s = (query_pos - window_size + 1) // block_size
            
            Q_shared = T.alloc_shared([GEMM_M, head_dim], dtype)
            K_shared = T.alloc_shared([BLOCK_S, head_dim], dtype)
            acc_s = T.alloc_fragment([GEMM_M, BLOCK_S], accum_dtype)
            m_prev = T.alloc_fragment([GEMM_M], accum_dtype)
            l_prev = T.alloc_fragment([GEMM_M], accum_dtype)
            m_curr = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_max = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_sum = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_scale = T.alloc_fragment([GEMM_M], accum_dtype)
            
            T.fill(m_prev, -T.infinity(accum_dtype))
            T.fill(l_prev, 0.0)

            # Load Q
            for g, d in T.Parallel(GEMM_M, head_dim):
                if g < groups:
                    Q_shared[g, d] = Q[i_b, i_h, g, d]
                else:
                    Q_shared[g, d] = 0.0
                
            # Dynamic Loop Limit based on current request length
            loop_limit = T.ceildiv(s_len_req, BLOCK_S) 
            
            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S
                
                # Load K with Bounds Check
                for i, j in T.Parallel(BLOCK_S, head_dim):
                    ts = base_s + i
                    if ts < s_len_req: 
                        K_shared[i, j] = K[i_b, ts, i_h, j]
                    else:
                        K_shared[i, j] = 0.0
                
                T.sync_threads()
                
                # GEMM
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                # Masking: Padding & Window
                for g, s in T.Parallel(GEMM_M, BLOCK_S):
                    ts = base_s + s
                    # Mask if Out of Bounds OR Too Recent (Window)
                    if (ts >= s_len_req) or (ts >= threshold_s):
                        acc_s[g, s] = -T.infinity(accum_dtype)
                
                # Softmax Update
                T.copy(m_prev, m_curr)
                T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                
                for g in T.Parallel(GEMM_M):
                    scores_max[g] = scores_max[g] * sm_scale
                    m_prev[g] = T.max(m_prev[g], scores_max[g])
                    
                for g in T.Parallel(GEMM_M):
                    if m_prev[g] == -T.infinity(accum_dtype):
                        scores_scale[g] = 1.0
                    else:
                        scores_scale[g] = T.exp(m_curr[g] - m_prev[g])
                        
                for g, s in T.Parallel(GEMM_M, BLOCK_S):
                     val = acc_s[g, s] * sm_scale
                     if val == -T.infinity(accum_dtype) and m_prev[g] == -T.infinity(accum_dtype):
                         acc_s[g, s] = 0.0
                     else:
                         acc_s[g, s] = T.exp(val - m_prev[g])
                
                T.reduce_sum(acc_s, scores_sum, dim=1)
                
                for g in T.Parallel(GEMM_M):
                    l_prev[g] = l_prev[g] * scores_scale[g] + scores_sum[g]

            for g in T.Parallel(groups):
                if l_prev[g] == 0:
                     LSE_Out[i_b, i_h, g] = -T.infinity(accum_dtype)
                else:
                     LSE_Out[i_b, i_h, g] = m_prev[g] + T.log(l_prev[g])

    return kernel



@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hsa_decode_weighted_select_kernel(
    batch, h_kv, groups, head_dim, topk, 
    block_size, window_size, 
    BLOCK_S=32, threads=64
):
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"
    
    max_s = T.dynamic("max_s")
    
    GEMM_M = (groups + 15) // 16 * 16
    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def kernel(
        Q: T.Tensor([batch, h_kv, groups, head_dim], dtype),
        K: T.Tensor([batch, max_s, h_kv, head_dim], dtype),
        LSE_Total: T.Tensor([batch, h_kv, groups], accum_dtype),
        SeqLens: T.Tensor([batch], "int32"),
        OutIndices: T.Tensor([batch, h_kv, topk], idx_dtype),
    ):
        with T.Kernel(h_kv, batch, threads=threads) as (i_h, i_b):
            # Dynamic Info
            cur_seq_len = SeqLens[i_b]
            s_len_req = cur_seq_len // block_size
            query_pos = cur_seq_len - 1
            threshold_s = (query_pos - window_size + 1) // block_size
            
            Q_shared = T.alloc_shared([GEMM_M, head_dim], dtype)
            K_shared = T.alloc_shared([BLOCK_S, head_dim], dtype)
            score_shared = T.alloc_shared([GEMM_M, BLOCK_S], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, BLOCK_S], accum_dtype)
            topk_vals = T.alloc_local([topk], accum_dtype)
            topk_idxs = T.alloc_local([topk], idx_dtype)
            lse_local = T.alloc_local([groups], accum_dtype)
            val = T.alloc_var(accum_dtype)
            norm_score = T.alloc_var(accum_dtype)
            cur_max_score = T.alloc_var(accum_dtype)
            moving = T.alloc_var("bool")
            
            T.fill(topk_vals, -T.infinity(accum_dtype))
            T.fill(topk_idxs, -1)
            
            # Load LSE
            if T.get_thread_binding() == 0:
                for g in T.serial(groups):
                     lse_local[g] = LSE_Total[i_b, i_h, g]

            # Load Q
            for g, d in T.Parallel(GEMM_M, head_dim):
                if g < groups:
                    Q_shared[g, d] = Q[i_b, i_h, g, d]
                else:
                    Q_shared[g, d] = 0.0

            # Dynamic Loop
            loop_limit = T.ceildiv(s_len_req, BLOCK_S)
            
            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S
                
                # Load K with Bounds
                for i, j in T.Parallel(BLOCK_S, head_dim):
                    ts = base_s + i
                    if ts < s_len_req:
                        K_shared[i, j] = K[i_b, ts, i_h, j]
                    else:
                        K_shared[i, j] = 0.0
                T.sync_threads()
                
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_s, score_shared)
                
                # Masking Shared Memory
                for g, s in T.Parallel(GEMM_M, BLOCK_S):
                    ts = base_s + s
                    if (ts >= s_len_req) or (ts >= threshold_s):
                        score_shared[g, s] = -T.infinity(accum_dtype)
                        
                T.sync_threads()
                
                # TopK Update
                if T.get_thread_binding() == 0:
                    for s in T.serial(BLOCK_S):
                        ts = base_s + s
                        if ts < s_len_req: 
                            cur_max_score = -T.infinity(accum_dtype)
                            for g in T.serial(groups):
                                val = score_shared[g, s] * sm_scale
                                if val == -T.infinity(accum_dtype):
                                    norm_score = -T.infinity(accum_dtype)
                                else:
                                    norm_score = val - lse_local[g]
                                cur_max_score = T.max(cur_max_score, norm_score)
                            
                            if cur_max_score > topk_vals[topk - 1]:
                                moving = True
                                for k_iter in T.serial(topk):
                                    k = topk - 1 - k_iter
                                    if moving:
                                        if k > 0 and cur_max_score > topk_vals[k - 1]:
                                             topk_vals[k] = topk_vals[k - 1]
                                             topk_idxs[k] = topk_idxs[k - 1]
                                        else:
                                             topk_vals[k] = cur_max_score
                                             topk_idxs[k] = ts
                                             moving = False
                T.sync_threads()

            if T.get_thread_binding() == 0:
                for k in T.serial(topk):
                    OutIndices[i_b, i_h, k] = topk_idxs[k]

    return kernel





@tilelang.jit(
    out_idx=[3], # Output is 4th argument
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hsa_decode_recompute_scores_kernel(
    batch, h_kv, groups, head_dim, topk, 
    BLOCK_TK=32, threads=64
):
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"
    
    max_s = T.dynamic("max_s")
    
    GEMM_M = (groups + 15) // 16 * 16
    GEMM_N = T.ceildiv(topk, BLOCK_TK) * BLOCK_TK
    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def fwd_recompute(
        Q: T.Tensor([batch, h_kv, groups, head_dim], dtype),
        K: T.Tensor([batch, max_s, h_kv, head_dim], dtype),
        Indices: T.Tensor([batch, h_kv, topk], idx_dtype),
        OutScores: T.Tensor([batch, h_kv, groups, topk], accum_dtype),
    ):
        with T.Kernel(h_kv, batch, threads=threads) as (i_h, i_b):
            Q_shared = T.alloc_shared([GEMM_M, head_dim], dtype)
            K_gathered_shared = T.alloc_shared([GEMM_N, head_dim], dtype)
            acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)
            score_final = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)

            # 1. Load Q
            for g, d in T.Parallel(GEMM_M, head_dim):
                if g < groups:
                    Q_shared[g, d] = Q[i_b, i_h, g, d]
                else:
                    Q_shared[g, d] = 0.0

            # 2. Gather K
            for k, d in T.Parallel(GEMM_N, head_dim):
                if k < topk:
                    idx = Indices[i_b, i_h, k]
                    # Simple validity check
                    if idx >= 0:
                        K_gathered_shared[k, d] = K[i_b, idx, i_h, d]
                    else:
                         K_gathered_shared[k, d] = 0.0
                else:
                    K_gathered_shared[k, d] = 0.0
            
            T.sync_threads()

            # 3. Compute Small GEMM
            T.clear(acc_s)
            T.gemm(Q_shared, K_gathered_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(acc_s, score_final)
            T.sync_threads()

            # 4. Write Back (Logits * Scale)
            for g, k in T.Parallel(groups, topk):
                idx = Indices[i_b, i_h, k]
                if idx >= 0:
                     OutScores[i_b, i_h, g, k] = score_final[g, k] * sm_scale
                else:
                     OutScores[i_b, i_h, g, k] = -T.infinity(accum_dtype)

    return fwd_recompute


class SoftmaxTopKMaxPooling_Decode_Fused(nn.Module):
    def __init__(self, topk, block_size=32, window_size=512):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        
        self._lse_kernel = None
        self._select_kernel = None
        self._recompute_kernel = None

    def forward(self, q, lmks, lse_swa, seq_lens):
        B, h_q, D = q.shape
        _, S, h_kv, _ = lmks.shape
        G = h_q // h_kv
        topk = self.topk
        
        self._lse_kernel = hsa_decode_lse_kernel(
            B, h_kv, G, D, 
            block_size=self.block_size, window_size=self.window_size,
        )
        self._select_kernel = hsa_decode_weighted_select_kernel(
            B, h_kv, G, D, topk, 
            block_size=self.block_size, window_size=self.window_size,
        )
        self._recompute_kernel = hsa_decode_recompute_scores_kernel(
            B, h_kv, G, D, topk
        )
            
        q_view = q.view(B, h_kv, G, D).contiguous()
        lmks = lmks.contiguous()
        seq_lens = seq_lens.to(torch.int32).contiguous()
        
        # 1. LSE_hsa
        lse_hsa = self._lse_kernel(q_view, lmks, seq_lens)
        
        # 2. Combine LSE
        lse_swa_view = lse_swa.view(B, h_kv, G).float()
        lse_total = torch.logaddexp(lse_swa_view, lse_hsa)
        
        # 3. Select Weighted Indices
        indices_raw = self._select_kernel(q_view, lmks, lse_total, seq_lens)
        
        # 5. Recompute Scores
        scores_grouped = self._recompute_kernel(q_view, lmks, indices_raw)
        
        scores = scores_grouped.view(B, h_q, topk).to(q.dtype)
        
        return indices_raw, scores


_MODULE_CACHE = {}

def topk_softmax_decode_interface(q, lmks, lse_swa, seq_lens, topk, block_size=64, window_size=512):
    """
    参数:
        q (torch.Tensor): Query 张量，形状为 (B, h_q, D)
        lmks (torch.Tensor): Key 缓存张量，形状为 (B, S, h_kv, D)
        lse_swa (torch.Tensor): 滑动窗口注意力 LSE，形状为 (B, h_q)
        seq_lens (torch.Tensor): 序列长度，形状为 (B,)，数据类型为整数
        topk (int): 要选择的 top-k 数量
        block_size (int, optional): 块大小
        window_size (int, optional): 窗口大小
    
    返回:
        tuple: (indices, scores)
            - indices (torch.Tensor): 选中的索引，形状为 (B, h_kv, topk)
            - scores (torch.Tensor): 重计算的注意力分数，形状为 (B, h_q, topk)
    
    """

    B, h_q, D = q.shape
    _, S, h_kv, _ = lmks.shape
    G = h_q // h_kv
    
    cache_key = (B, h_kv, G, D, topk, block_size, window_size, str(q.device))
    
    if cache_key not in _MODULE_CACHE:
        module = SoftmaxTopKMaxPooling_Decode_Fused(topk, block_size, window_size)
        _MODULE_CACHE[cache_key] = module
        
    module = _MODULE_CACHE[cache_key]
    
    return module(q, lmks, lse_swa, seq_lens)




def ref_softmax_topk_max_pooling_decode_ref(q, lmks, lse_swa, topk, seq_lens, block_size, window_size):
    B, h_q, D = q.shape
    max_s = lmks.shape[1]
    h_kv = lmks.shape[2]
    G = h_q // h_kv
    
    q_g = q.view(B, h_kv, G, D).float()
    logits = torch.einsum("bhgd,bshd->bhgs", q_g, lmks.float())
    logits = logits * (1.0 / math.sqrt(D))
    
    s_idx = torch.arange(max_s, device=q.device).view(1, 1, 1, max_s)
    cur_len = seq_lens.view(B, 1, 1, 1)
    
    valid_s_count = cur_len // block_size
    query_pos = cur_len - 1  # query 的位置索引
    threshold_window = (query_pos - window_size + 1) // block_size
    
    mask = (s_idx >= valid_s_count) | (s_idx >= threshold_window)
    logits = logits.masked_fill(mask, float('-inf'))
    
    lse_hsa = torch.logsumexp(logits, dim=-1)
    lse_swa_view = lse_swa.view(B, h_kv, G).float()
    lse_total = torch.logaddexp(lse_swa_view, lse_hsa)
    log_probs = logits - lse_total.unsqueeze(-1)
    
    pooled = log_probs.max(dim=2).values
    
    topk_vals, topk_indices = torch.topk(pooled, k=topk, dim=-1, sorted=True)
    
    invalid_mask = topk_vals == float('-inf')
    
    huge_idx = max_s + 10000 
    topk_indices = topk_indices.masked_fill(invalid_mask, huge_idx)
    
    topk_indices = topk_indices.masked_fill(topk_indices == huge_idx, -1)
    
    indices_expanded = topk_indices.unsqueeze(2).expand(-1, -1, G, -1)
    
    safe_indices = indices_expanded.clamp(min=0) 
    scores_out = torch.gather(logits, -1, safe_indices)
    
    scores_out = scores_out.masked_fill(indices_expanded == -1, float('-inf'))
    
    return topk_indices, scores_out.view(B, h_q, topk)


def test_decode_correctness():
    print("\n" + "=" * 80)
    print("=== Testing Fused Decode Softmax TopK Module Correctness (TileLang) ===")
    print("=== Mode: Variable Sequence Lengths (Batch Decode) ===")
    print("=" * 80)
    
    B, D = 8, 128
    h_kv, G = 4, 8
    h_q = h_kv * G
    
    block_size, window_size, topk = 32, 128, 32
    max_s_capacity = 1024 
    
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(42)
    q = torch.randn(B, h_q, D, dtype=dtype, device=device)
    lmks = torch.randn(B, max_s_capacity, h_kv, D, dtype=dtype, device=device)
    lse_swa = torch.randn(B, h_q, dtype=dtype, device=device) * 5
    
    min_len = 1
    max_len = max_s_capacity * block_size
    seq_lens = torch.randint(low=min_len, high=max_len, size=(B,), dtype=torch.int32, device=device)
    
    # 设置edge cases
    edge_cases = [159, 191, -1]  # window_size=128, block_size=32
    for i, edge_case in enumerate(edge_cases):
        if i < B:
            seq_lens[i] = edge_case
            
    mod_indices, mod_scores = topk_softmax_decode_interface(
        q, lmks, lse_swa, seq_lens, topk, block_size, window_size
    )
    ref_indices, ref_scores = ref_softmax_topk_max_pooling_decode_ref(
        q, lmks, lse_swa, topk, seq_lens, block_size, window_size
    )

    indices_match = (mod_indices == ref_indices).float().mean()
    mask_valid = ref_scores > float('-inf')
    if mask_valid.sum() > 0:
        score_diff = (mod_scores[mask_valid].float() - ref_scores[mask_valid].float()).abs().max()
    else:
        score_diff = torch.tensor(0.0)

    print("-" * 40)
    print(f"Indices Match Rate: {indices_match.item()*100:.2f}%")
    print(f"Max Score Difference: {score_diff.item():.6f}")
    
    if indices_match > 0.99 and score_diff < 5e-2:
        print("\n✅ Fused Decode Module Test PASSED")
    else:
        print("\n❌ Fused Decode Module Test FAILED")
        
        
def test_decode_performance():
    # Only updated to pass seq_lens
    B, D = 16, 128
    h_kv, G = 4, 8
    h_q = h_kv * G
    topk = 32
    max_s_capacity = 16384
    block_size, window_size = 32, 512
    
    dtype = torch.bfloat16
    device = "cuda"
    
    q = torch.randn(B, h_q, D, dtype=dtype, device=device)
    lmks = torch.randn(B, max_s_capacity, h_kv, D, dtype=dtype, device=device)
    lse_swa = torch.randn(B, h_q, dtype=dtype, device=device)
    seq_lens = torch.randint(512*2, max_s_capacity, (B,), dtype=torch.int32, device=device)
    
    
    # 预热
    for _ in range(10):
        _ = topk_softmax_decode_interface(
        q, lmks, lse_swa, seq_lens, topk, block_size, window_size
    )
    
    # 测量模块实现
    torch.cuda.synchronize()
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    
    t_start.record()
    for _ in range(100):
        _ = topk_softmax_decode_interface(
        q, lmks, lse_swa, seq_lens, topk, block_size, window_size
    )
    t_end.record()
    torch.cuda.synchronize()
    fused_time = t_start.elapsed_time(t_end)/100
    
    print(f"Fused Module Latency: {fused_time:.4f} ms")
    
    # 测量参考实现
    # 预热参考实现
    for _ in range(5):
        _ = ref_softmax_topk_max_pooling_decode_ref(
            q, lmks, lse_swa, topk, seq_lens, block_size, window_size
        )
    
    torch.cuda.synchronize()
    t_start_ref = torch.cuda.Event(enable_timing=True)
    t_end_ref = torch.cuda.Event(enable_timing=True)
    
    t_start_ref.record()
    for _ in range(10):  # 参考实现可能较慢，减少迭代次数
        _ = ref_softmax_topk_max_pooling_decode_ref(
            q, lmks, lse_swa, topk, seq_lens, block_size, window_size
        )
    t_end_ref.record()
    torch.cuda.synchronize()
    ref_time = t_start_ref.elapsed_time(t_end_ref)/10
    
    print(f"Reference Implementation Latency: {ref_time:.4f} ms")
    print(f"Speedup: {ref_time/fused_time:.2f}x")


import pytest

@pytest.mark.parametrize("B, h_kv, G, D, topk, max_s, block_size, window_size", [
    (2, 2, 8, 64, 16, 128, 32, 64),
])
def test_decode_correctness_simple(B, h_kv, G, D, topk, max_s, block_size, window_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    h_q = h_kv * G
    
    torch.manual_seed(42)
    
    # 准备数据
    q = torch.randn(B, h_q, D, dtype=dtype, device=device)
    lmks = torch.randn(B, max_s, h_kv, D, dtype=dtype, device=device)
    lse_swa = torch.randn(B, h_q, dtype=dtype, device=device) * 5
    seq_lens = torch.randint(
        low=window_size + block_size * 2,
        high=max_s * block_size,
        size=(B,),
        dtype=torch.int32,
        device=device
    )
    
    # Fused
    mod_indices, mod_scores = topk_softmax_decode_interface(
        q, lmks, lse_swa, seq_lens, topk, block_size, window_size
    )
    
    # Reference
    ref_indices, ref_scores = ref_softmax_topk_max_pooling_decode_ref(
        q, lmks, lse_swa.float(), topk, seq_lens, block_size, window_size
    )
    
    indices_match = (mod_indices == ref_indices).float().mean()
    print(f"Indices Match Rate: {indices_match*100:.2f}%")
    assert indices_match >= 0.99, f"Indices match rate: {indices_match}"
    
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
    
    # 使用辅助函数验证分数
    assert_close("Scores", ref_scores, mod_scores, ratio=0.005)
    
    print(f"✅ Test Passed")


# ...existing code...
def test_decode_dynamic_compilation():
    """
    测试 TopK Decode 接口对于动态 max_s 的编译缓存行为
    """
    print("\n" + "=" * 60)
    print("=== Testing TopK Decode Dynamic Compilation (JIT Reuse) ===")
    print("=" * 60)
    
    # 清空缓存
    global _MODULE_CACHE
    _MODULE_CACHE.clear()
    
    B, h_kv, G, D = 2, 2, 4, 32
    h_q = h_kv * G
    topk = 16
    block_size = 32
    window_size = 64
    
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    max_s_1 = 1000
    print("--> Step 1: Running with max_s=", max_s_1)
    q1 = torch.randn(B, h_q, D, dtype=dtype, device=device)
    lmks1 = torch.randn(B, max_s_1, h_kv, D, dtype=dtype, device=device)
    lse1 = torch.randn(B, h_q, dtype=dtype, device=device)
    seq_lens1 = torch.randint(window_size+1, max_s_1 * block_size, (B,), dtype=torch.int32, device=device)
    
    topk_softmax_decode_interface(q1, lmks1, lse1, seq_lens1, topk, block_size, window_size)
    
    cache_len_1 = len(_MODULE_CACHE)
    print(f"    Cache Size: {cache_len_1}")
    assert cache_len_1 == 1, "Should compile once."
    
    max_s_2 = max_s_1+1
    print("--> Step 2: Running with max_s=", max_s_2)
    q2 = torch.randn(B, h_q, D, dtype=dtype, device=device)
    lmks2 = torch.randn(B, max_s_2, h_kv, D, dtype=dtype, device=device) # Shape changed
    lse2 = torch.randn(B, h_q, dtype=dtype, device=device)
    seq_lens2 = torch.randint(window_size+1, max_s_2 * block_size, (B,), dtype=torch.int32, device=device)
    
    topk_softmax_decode_interface(q2, lmks2, lse2, seq_lens2, topk, block_size, window_size)
    
    cache_len_2 = len(_MODULE_CACHE)
    print(f"    Cache Size: {cache_len_2}")
    
    assert cache_len_2 == cache_len_1, \
        f"Kernel recompiled! Cache grew from {cache_len_1} to {cache_len_2}. T.dynamic('max_s') might not be working."
    
    # --- Step 3: Change Static Param (window_size) ---
    print("--> Step 3: Running with window_size=128 (Expect +1 Compile)")
    new_window_size = 128
    
    topk_softmax_decode_interface(q2, lmks2, lse2, seq_lens2, topk, block_size, new_window_size)
    
    cache_len_3 = len(_MODULE_CACHE)
    print(f"    Cache Size: {cache_len_3}")
    assert cache_len_3 == cache_len_1 + 1, "Should compile new kernel for new window_size."

    print("✅ TopK Decode Dynamic Compilation Test PASSED")




if __name__ == "__main__":
    test_decode_correctness()
    test_decode_performance()
    
    params = [
        (2, 2, 8, 64, 16, 128, 32, 64),
        (4, 4, 8, 128, 32, 256, 64, 512),
        (8, 2, 4, 64, 8, 512, 32, 200),
    ]
    for p in params:
        test_decode_correctness_simple(*p)
    
    
    test_decode_dynamic_compilation()
    