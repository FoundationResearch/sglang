import torch
import tilelang
from tilelang import language as T

@tilelang.jit(
    out_idx=[4, 5],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def rope_fwd_kernel(batch, seq_len, heads_q, heads_k, dim, threads=64, dtype="bfloat16"):
    
    # dtype = "bfloat16"
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



def rope_rotary_pos_emb(q, k, cos, sin, dtype="bfloat16"):
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
    
    kernel_func = rope_fwd_kernel(B, L, HQ, H, D, dtype=dtype)
    
    return kernel_func(q, k, cos, sin)


@tilelang.jit(
    out_idx=[4, 5],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def rope_fwd_kernel_bhld(batch, seq_len, heads_q, heads_k, dim, threads=64, dtype="bfloat16"):
    """
    RoPE kernel optimized for (Batch, Heads, SeqLen, Dim) layout.
    """
    # dtype = "bfloat16"
    compute_dtype = "float32"
    
    assert dim % 2 == 0
    dim_half = dim // 2

    @T.prim_func
    def main(
        Q: T.Buffer((batch, heads_q, seq_len, dim), dtype),
        K: T.Buffer((batch, heads_k, seq_len, dim), dtype),
        Cos: T.Buffer((batch, seq_len, dim), dtype),
        Sin: T.Buffer((batch, seq_len, dim), dtype),
        Q_out: T.Buffer((batch, heads_q, seq_len, dim), dtype),
        K_out: T.Buffer((batch, heads_k, seq_len, dim), dtype),
    ):
        
        # Parallelize over Batch * SeqLen
        # Each thread block handles one token (across all heads) or a chunk of tokens
        total_tokens = batch * seq_len
        
        with T.Kernel(total_tokens, threads=threads) as bx:
            b_idx = bx // seq_len
            l_idx = bx % seq_len
            
            # Load Cos/Sin for this token (shared across all heads)
            cos_frag = T.alloc_fragment((dim,), compute_dtype)
            sin_frag = T.alloc_fragment((dim,), compute_dtype)
            
            for d in T.Parallel(dim):
                cos_frag[d] = T.cast(Cos[b_idx, l_idx, d], compute_dtype)
                sin_frag[d] = T.cast(Sin[b_idx, l_idx, d], compute_dtype)
            
            # Process Query Heads
            for h in T.unroll(heads_q):
                for d in T.Parallel(dim // 2):
                    idx_1 = d
                    idx_2 = d + dim_half
                    
                    # Access pattern: [b, h, l, d]
                    q1 = T.cast(Q[b_idx, h, l_idx, idx_1], compute_dtype)
                    q2 = T.cast(Q[b_idx, h, l_idx, idx_2], compute_dtype)
                    
                    c_val = cos_frag[idx_1] 
                    s_val = sin_frag[idx_1]
                    
                    o1 = q1 * c_val - q2 * s_val
                    o2 = q2 * c_val + q1 * s_val
                    
                    Q_out[b_idx, h, l_idx, idx_1] = T.cast(o1, dtype)
                    Q_out[b_idx, h, l_idx, idx_2] = T.cast(o2, dtype)

            # Process Key Heads
            for h in T.unroll(heads_k):
                for d in T.Parallel(dim // 2):
                    idx_1 = d
                    idx_2 = d + dim_half
                    
                    # Access pattern: [b, h, l, d]
                    k1 = T.cast(K[b_idx, h, l_idx, idx_1], compute_dtype)
                    k2 = T.cast(K[b_idx, h, l_idx, idx_2], compute_dtype)
                    
                    c_val = cos_frag[idx_1]
                    s_val = sin_frag[idx_1]
                    
                    o1 = k1 * c_val - k2 * s_val
                    o2 = k2 * c_val + k1 * s_val
                    
                    K_out[b_idx, h, l_idx, idx_1] = T.cast(o1, dtype)
                    K_out[b_idx, h, l_idx, idx_2] = T.cast(o2, dtype)

    return main

def rope_rotary_pos_emb_bhld(q, k, cos, sin, dtype="bfloat16"):
    """
    Apply Rotary Positional Embedding using TileLang (Fused & FP32 Compute).
    Optimized for BHLD layout.
    
    Args:
        q: [batch, heads_q, seq_len, dim] (BF16)
        k: [batch, heads_k, seq_len, dim] (BF16)
        cos: [batch, seq_len, dim] (BF16)
        sin: [batch, seq_len, dim] (BF16)
        
    Returns:
        q_out, k_out: [batch, heads, seq_len, dim] (BF16)
    """
    B, HQ, L, D = q.shape
    HK = k.shape[1]
    
    kernel_func = rope_fwd_kernel_bhld(B, L, HQ, HK, D, dtype=dtype)
    
    return kernel_func(q, k, cos, sin)






# ...existing code...

@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def rope_lmk_kernel(batch, seq_len, num_chunks, heads_k, dim, chunk_size, threads=64, dtype="bfloat16"):
    
    # dtype = "bfloat16"
    compute_dtype = "float32"
    
    assert dim % 2 == 0
    dim_half = dim // 2

    @T.prim_func
    def main(
        LMK: T.Buffer((batch, num_chunks, heads_k, dim), dtype),
        Cos: T.Buffer((batch, seq_len, dim), dtype),
        Sin: T.Buffer((batch, seq_len, dim), dtype),
        LMK_out: T.Buffer((batch, num_chunks, heads_k, dim), dtype),
    ):
        
        total_tokens = batch * num_chunks
        
        with T.Kernel(total_tokens, threads=threads) as bx:
            b_idx = bx // num_chunks
            chunk_idx = bx % num_chunks 
            
            phy_pos_idx = (chunk_idx + 1) * chunk_size - 1
            valid_idx = T.min(phy_pos_idx, seq_len - 1)

            cos_frag = T.alloc_fragment((dim,), compute_dtype)
            sin_frag = T.alloc_fragment((dim,), compute_dtype)
            
            for d in T.Parallel(dim):
                cos_frag[d] = T.cast(Cos[b_idx, valid_idx, d], compute_dtype)
                sin_frag[d] = T.cast(Sin[b_idx, valid_idx, d], compute_dtype)
            
            for h in T.unroll(heads_k):
                for d in T.Parallel(dim // 2):
                    idx_1 = d
                    idx_2 = d + dim_half
                    
                    k1 = T.cast(LMK[b_idx, chunk_idx, h, idx_1], compute_dtype)
                    k2 = T.cast(LMK[b_idx, chunk_idx, h, idx_2], compute_dtype)
                    
                    c_val = cos_frag[idx_1]
                    s_val = sin_frag[idx_1]
                    
                    o1 = k1 * c_val - k2 * s_val
                    o2 = k2 * c_val + k1 * s_val
                    
                    LMK_out[b_idx, chunk_idx, h, idx_1] = T.cast(o1, dtype)
                    LMK_out[b_idx, chunk_idx, h, idx_2] = T.cast(o2, dtype)

    return main





def apply_rope_to_lmks(lmk_k, cos, sin, chunk_size, dtype="bfloat16"):
    """
    Helper function to apply RoPE to Landmark Embeddings.
    
    Args:
        lmk_k: [Batch, Num_Chunks, Heads_K, Dim] - The sparse LMKs
        cos:   [Batch, Seq_Len, Dim] - Dense Cosine table
        sin:   [Batch, Seq_Len, Dim] - Dense Sine table
        chunk_size: int - stride for LMK placement
    """
    B, Num_Chunks, H, D = lmk_k.shape
    Seq_Len = cos.shape[1]
    
    kernel = rope_lmk_kernel(B, Seq_Len, Num_Chunks, H, D, chunk_size, dtype=dtype)
    return kernel(lmk_k, cos, sin)





def assert_close(prefix, ref, tri, atol=5e-3, rtol=5e-3):
    def get_abs_err(x, y):
        return (x - y).abs().max().item()
    
    def get_err_ratio(x, y):
        err = (x - y).square().mean().sqrt().item()
        base = (x).square().mean().sqrt().item()
        return err / (base + 1e-12)

    abs_err = get_abs_err(ref, tri)
    rel_ratio = get_err_ratio(ref, tri)
    msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
    print(msg)
    
    if abs_err > atol and rel_ratio > rtol:
        raise AssertionError(f"❌ {msg}")


import torch.nn.functional as F
from liger_kernel.transformers.rope import liger_rotary_pos_emb
def run_rope_correctness_test(B, L, HQ, HK, D):
    """
    Robust correctness test for standard RoPE kernel.
    Compares TileLang (BLHD) against Liger (BHLD).
    """
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    print(f"\n🚀 Testing Standard RoPE: B={B}, L={L}, HQ={HQ}, HK={HK}, D={D}")

    # 1. Prepare Inputs (BLHD for TileLang)
    q = torch.randn((B, L, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, L, HK, D), dtype=dtype, device=device)
    
    # 2. Construct Symmetric Cos/Sin
    cos_half = torch.randn((B, L, D//2), dtype=dtype, device=device)
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin_half = torch.randn((B, L, D//2), dtype=dtype, device=device)
    sin = torch.cat([sin_half, sin_half], dim=-1)

    # -------------------------------------------------------
    # 3. Baseline: Liger (Requires BHLD and Float)
    # -------------------------------------------------------
    # Transpose BLHD -> BHLD for Liger
    q_liger = q.transpose(1, 2).contiguous().float()
    k_liger = k.transpose(1, 2).contiguous().float()
    cos_liger = cos.float()
    sin_liger = sin.float()

    # Liger computation
    q_out_liger, k_out_liger = liger_rotary_pos_emb(q_liger, k_liger, cos_liger, sin_liger)
    
    # Transpose back to BLHD and cast to BF16 for comparison
    q_ref = q_out_liger.transpose(1, 2).contiguous().to(dtype)
    k_ref = k_out_liger.transpose(1, 2).contiguous().to(dtype)

    # -------------------------------------------------------
    # 4. Kernel: TileLang (BLHD)
    # -------------------------------------------------------
    q_tl, k_tl = rope_rotary_pos_emb(q, k, cos, sin)

    # -------------------------------------------------------
    # 5. Validation using assert_close
    # -------------------------------------------------------
    assert_close("Standard RoPE Q", q_ref.float(), q_tl.float(), atol=1e-2, rtol=1e-2)
    assert_close("Standard RoPE K", k_ref.float(), k_tl.float(), atol=1e-2, rtol=1e-2)
    print("✅ Passed")



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
    

def run_test_case(B, L, S, H, D):
    device = "cuda"
    dtype = torch.bfloat16
    
    num_chunks = L // S
    if num_chunks == 0: return

    print(f"\n🚀 Testing Case: B={B}, L={L}, S={S}, H={H}, D={D}, NumChunks={num_chunks}")
    torch.manual_seed(42)

    # 1. Inputs
    lmk_k = torch.randn((B, num_chunks, H, D), dtype=dtype, device=device)
    
    # 2. Symmetric Cos/Sin
    cos_half = torch.randn((B, L, D//2), dtype=dtype, device=device)
    cos_full = torch.cat([cos_half, cos_half], dim=-1)
    sin_half = torch.randn((B, L, D//2), dtype=dtype, device=device)
    sin_full = torch.cat([sin_half, sin_half], dim=-1)
    
    # 3. Reference
    indices = torch.arange(1, num_chunks + 1, device=device) * S - 1
    indices = torch.clamp(indices, max=L-1).long()
    
    cos_ref = cos_full[:, indices, :].unsqueeze(2).float()
    sin_ref = sin_full[:, indices, :].unsqueeze(2).float()
    x = lmk_k.float()
    
    x1 = x[..., :D//2]
    x2 = x[..., D//2:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    k_ref = (x * cos_ref) + (rotated_x * sin_ref)
    
    # 4. Kernel
    k_tl = apply_rope_to_lmks(lmk_k, cos_full, sin_full, chunk_size=S)
    
    assert_close("LMK RoPE", k_ref, k_tl.float())
    print("✅ Passed")

if __name__ == "__main__":
    standard_rope_cases = [
        (2, 128, 4, 4, 64),    # Base
        (1, 1024, 32, 8, 128), # Large (GQA)
        (4, 512, 16, 1, 64),   # MQA
        (2, 100, 8, 8, 32),    # Odd length
    ]
    
    for params in standard_rope_cases:
        try:
            run_rope_correctness_test(*params)
        except AssertionError as e:
            print(e)
            exit(1)
    # test_rope_performance()
    
    test_cases = [
        (2, 128, 8, 4, 64),
        (1, 2048, 64, 8, 128),
        (4, 512, 16, 2, 64),
        (2, 100, 10, 4, 32),
        (2, 105, 10, 4, 32)
    ]
    
    for params in test_cases:
        try:
            run_test_case(*params)
        except AssertionError as e:
            print(e)
            exit(1)