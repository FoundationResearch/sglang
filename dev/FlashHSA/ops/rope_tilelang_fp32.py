import torch
import tilelang
from tilelang import language as T

@tilelang.jit(
    out_idx=[4, 5],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def rope_fwd_kernel(batch, seq_len, heads_q, heads_k, dim, threads=64):
    
    dtype = "bfloat16"
    compute_dtype = "float32"
    
    assert dim % 2 == 0
    dim_half = dim // 2

    @T.prim_func
    def main(
        Q: T.Buffer((batch, seq_len, heads_q, dim), dtype),
        K: T.Buffer((batch, seq_len, heads_k, dim), dtype),
        Cos: T.Buffer((batch, seq_len, dim), dtype),
        Sin: T.Buffer((batch, seq_len, dim), dtype),
        Q_out: T.Buffer((batch, seq_len, heads_q, dim), dtype),
        K_out: T.Buffer((batch, seq_len, heads_k, dim), dtype),
    ):
        
        total_tokens = batch * seq_len
        
        with T.Kernel(total_tokens, threads=threads) as bx:
            b_idx = bx // seq_len
            l_idx = bx % seq_len
            cos_frag = T.alloc_fragment((dim,), compute_dtype)
            sin_frag = T.alloc_fragment((dim,), compute_dtype)
            
            for d in T.Parallel(dim):
                cos_frag[d] = T.cast(Cos[b_idx, l_idx, d], compute_dtype)
                sin_frag[d] = T.cast(Sin[b_idx, l_idx, d], compute_dtype)
            
            for h in T.unroll(heads_q):
                for d in T.Parallel(dim // 2):
                    idx_1 = d
                    idx_2 = d + dim_half
                    
                    q1 = T.cast(Q[b_idx, l_idx, h, idx_1], compute_dtype)
                    q2 = T.cast(Q[b_idx, l_idx, h, idx_2], compute_dtype)
                    
                    c_val = cos_frag[idx_1] 
                    s_val = sin_frag[idx_1]
                    
                    o1 = q1 * c_val - q2 * s_val
                    o2 = q2 * c_val + q1 * s_val
                    
                    Q_out[b_idx, l_idx, h, idx_1] = T.cast(o1, dtype)
                    Q_out[b_idx, l_idx, h, idx_2] = T.cast(o2, dtype)

            for h in T.unroll(heads_k):
                for d in T.Parallel(dim // 2):
                    idx_1 = d
                    idx_2 = d + dim_half
                    
                    k1 = T.cast(K[b_idx, l_idx, h, idx_1], compute_dtype)
                    k2 = T.cast(K[b_idx, l_idx, h, idx_2], compute_dtype)
                    
                    c_val = cos_frag[idx_1]
                    s_val = sin_frag[idx_1]
                    
                    o1 = k1 * c_val - k2 * s_val
                    o2 = k2 * c_val + k1 * s_val
                    
                    K_out[b_idx, l_idx, h, idx_1] = T.cast(o1, dtype)
                    K_out[b_idx, l_idx, h, idx_2] = T.cast(o2, dtype)

    return main



def rope_rotary_pos_emb(q, k, cos, sin):
    """
    Apply Rotary Positional Embedding using TileLang (Fused & FP32 Compute).
    
    Args:
        q: [batch, seq_len, heads_q, dim] (BF16)
        k: [batch, seq_len, heads_k, dim] (BF16)
        cos: [batch, seq_len, dim] (BF16)
        sin: [batch, seq_len, dim] (BF16)
        
    Returns:
        q_out, k_out: [batch, seq_len, heads, dim] (BF16)
    """
    B, L, HQ, D = q.shape
    H = k.shape[2]
    
    kernel_func = rope_fwd_kernel(B, L, HQ, H, D)
    
    return kernel_func(q, k, cos, sin)




import torch.nn.functional as F
from liger_kernel.transformers.rope import liger_rotary_pos_emb
    
def test_rope_correctness():
    torch.manual_seed(42)
    device = "cuda"
    
    # Config
    B, L, H, D = 2, 128, 4, 64
    test_inverse = True 
    
    # Inputs (BF16)
    q = torch.randn((B, L, H, D), dtype=torch.bfloat16, device=device)
    k = torch.randn((B, L, H, D), dtype=torch.bfloat16, device=device)
    
    # Random Cos/Sin for testing
    cos = torch.randn((B, L, D), dtype=torch.bfloat16, device=device)
    sin = torch.randn((B, L, D), dtype=torch.bfloat16, device=device)

    print(f"Testing Shape: Q{q.shape}, Cos{cos.shape}, Test Inverse={test_inverse}")

    # -------------------------------------------------------
    # 1. Baseline: Liger Kernel (High Precision Reference)
    # -------------------------------------------------------
    q_ref_in = q.transpose(1, 2).contiguous().float() 
    k_ref_in = k.transpose(1, 2).contiguous().float()
    cos_ref = cos.float()
    sin_ref = sin.float()
    
    sin_input_liger = -sin_ref if test_inverse else sin_ref
    
    q_out_liger, k_out_liger = liger_rotary_pos_emb(q_ref_in, k_ref_in, cos_ref, sin_input_liger)
    
    q_out_ref = q_out_liger.transpose(1, 2).contiguous().to(torch.bfloat16)
    k_out_ref = k_out_liger.transpose(1, 2).contiguous().to(torch.bfloat16)

    # -------------------------------------------------------
    # 2. TileLang Kernel (Using Wrapper)
    # -------------------------------------------------------
    sin_input_tl = -sin if test_inverse else sin
    
    # 直接调用封装好的函数
    q_out_tl, k_out_tl = rope_rotary_pos_emb(q, k, cos, sin_input_tl)

    # -------------------------------------------------------
    # 3. Validation
    # -------------------------------------------------------
    diff_q = (q_out_ref.float() - q_out_tl.float()).abs()
    max_diff_q = diff_q.max().item()
    
    diff_k = (k_out_ref.float() - k_out_tl.float()).abs()
    max_diff_k = diff_k.max().item()
    
    print(f"Max Diff Q: {max_diff_q:.6f}")
    print(f"Max Diff K: {max_diff_k:.6f}")

    if max_diff_q < 1e-2:
        print("✅ Test Passed! TileLang kernel matches Liger (FP32) behavior.")
    else:
        print("❌ Test Failed! Large discrepancy detected.")




def test_rope_performance():
    import time
    
    print("\n" + "=" * 60)
    print("🚀 RoPE Kernel Raw Latency Benchmark (Kernel Only)")
    print("=" * 60)
    
    torch.manual_seed(42)
    device = "cuda"
    
    # B=Batch, L=SeqLen, H=Heads, D=HeadDim
    B, L, HQ, H, D = 128, 4096, 32, 4, 128 
    dtype = torch.bfloat16
    
    print(f"Config: Batch={B}, SeqLen={L}, Heads={H}, Dim={D}")
    
    # ------------------------------------------------------------------
    # 准备数据
    # ------------------------------------------------------------------
    
    q_tl = torch.randn((B, L, HQ, D), dtype=dtype, device=device)
    k_tl = torch.randn((B, L, H, D), dtype=dtype, device=device)
    cos_tl = torch.randn((B, L, D), dtype=dtype, device=device)
    sin_tl = torch.randn((B, L, D), dtype=dtype, device=device)
    
    q_liger = q_tl.transpose(1, 2).contiguous()
    k_liger = k_tl.transpose(1, 2).contiguous()
    cos_liger = cos_tl
    sin_liger = sin_tl

    # ------------------------------------------------------------------
    # 定义测试步骤
    # ------------------------------------------------------------------
    
    def baseline_kernel_only():
        # 纯 Liger Kernel 调用
        liger_rotary_pos_emb(q_liger, k_liger, cos_liger, sin_liger)

    def tilelang_kernel_only():
        # 纯 TileLang Kernel 调用
        rope_rotary_pos_emb(q_tl, k_tl, cos_tl, sin_tl)

    # ------------------------------------------------------------------
    # 测量工具函数
    # ------------------------------------------------------------------
    def benchmark(func, name, num_iters=100, num_warmup=20):
        # Warmup
        for _ in range(num_warmup):
            func()
        torch.cuda.synchronize()
        
        # Timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iters):
            func()
        end_event.record()
        torch.cuda.synchronize()
        
        avg_time = start_event.elapsed_time(end_event) / num_iters
        print(f"[{name}] Average Latency: {avg_time:.3f} ms")
        return avg_time

    # ------------------------------------------------------------------
    # 执行测试
    # ------------------------------------------------------------------
    
    print("-" * 30)
    time_liger = benchmark(baseline_kernel_only, "Liger Kernel")
    time_tilelang = benchmark(tilelang_kernel_only, "TileLang Kernel")
    

if __name__ == "__main__":
    # 运行正确性测试
    test_rope_correctness()
    
    # 运行性能测试
    test_rope_performance()
    