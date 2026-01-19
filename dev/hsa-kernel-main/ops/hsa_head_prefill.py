# ruff: noqa
import torch

import tilelang
from tilelang import language as T
import tilelang.testing

tilelang.testing.set_random_seed(0)


from einops import rearrange
def hsa_torch_ref(q, k, v, weights, indices, *, chunk_size: int, sm_scale: float, block_q: int):
    """
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
    orig_l = L  
    device = q.device

    pad_len = (chunk_size - (orig_l % chunk_size)) % chunk_size
    if pad_len > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
    
    new_l = k.shape[1]
    if indices.shape[1] != q_blocks:
        idx_view = indices.view(B, q_blocks, block_q, H, -1)
        indices_q = idx_view[:, :, 0, :, :].contiguous()
    else:
        indices_q = indices

    valid_mask = (indices_q >= 0)  # (B, q_blocks, H, K)
    safe_indices = indices_q.clamp_min(0)

    N = new_l // chunk_size
    k_chunks = rearrange(k, 'B (N S) h d -> B N S h d', S=chunk_size)
    v_chunks = rearrange(v, 'B (N S) h d -> B N S h d', S=chunk_size)

    idx_flat = rearrange(safe_indices, 'B Bq h K -> B (Bq K) h').unsqueeze(2).unsqueeze(-1)  # (B, BqK, 1, h, 1)
    idx_flat = idx_flat.expand(-1, -1, chunk_size, -1, D)                                   # (B, BqK, S, h, D)
    idx_flat = idx_flat.long()  
    gather_k = k_chunks.gather(dim=1, index=idx_flat)  # (B, BqK, S, h, D)
    gather_v = v_chunks.gather(dim=1, index=idx_flat)

    valid_chunks = orig_l // chunk_size
    padding_mask = (safe_indices < valid_chunks)  # (B, q_blocks, H, K)
    padding_mask_expanded = rearrange(padding_mask, 'B Bq h K -> B (Bq K) h').unsqueeze(2).unsqueeze(-1)  # (B, BqK, 1, h, 1)
    padding_mask_expanded = padding_mask_expanded.expand(-1, -1, chunk_size, -1, D)  # (B, BqK, S, h, D)
    
    gather_k = gather_k * padding_mask_expanded.float()
    gather_v = gather_v * padding_mask_expanded.float()

    gather_k = rearrange(gather_k, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)
    gather_v = rearrange(gather_v, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)

    k_ = torch.repeat_interleave(gather_k, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)
    v_ = torch.repeat_interleave(gather_v, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)

    # q 分块: (B, L, HQ, D) -> (B, q_blocks, block_q, HQ, D)
    q_chunked = rearrange(q, 'B (Bq X) hq d -> B Bq X hq d', X=block_q)

    # qk: (B, Bq, X, S, K, HQ)
    qk = torch.einsum('b q x h d, b q s k h d -> b q x s k h', q_chunked.float(), k_.float())
    qk = qk * float(sm_scale)

    p = torch.softmax(qk, dim=3)  # S 维

    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    w_masked = weights.clone()
    valid_mask_expanded = torch.repeat_interleave(valid_mask, dim=-2, repeats=G)  # (B, Bq, HQ, K)
    w_masked = w_masked.masked_fill(~valid_mask_expanded, 0)

    padding_mask_for_w = padding_mask.float()  # (B, q_blocks, H, K)
    padding_mask_for_w_expanded = torch.repeat_interleave(padding_mask_for_w, dim=-2, repeats=G)  # (B, Bq, HQ, K)
    w_masked = w_masked * padding_mask_for_w_expanded

    w_exp = w_masked.float()  # (B, Bq, HQ, K)

    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'B Bq X hq d -> B (Bq X) hq d')  # 回到 (B, L, HQ, D)
    
    return o_ref.to(torch.float32)



@tilelang.jit(
    out_idx=[-1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def hierarchical_sparse_attention(batch,
                              heads,
                              head_dim,
                              scale=None,
                              block_size=64,
                              groups=16,
                              selected_blocks=16,
                              num_weights=None):
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    q_len_dynamic = T.dynamic("q_len")
    kv_len_dynamic = T.dynamic("kv_len")
    
    head_kv = heads // groups
    q_shape = [batch, q_len_dynamic, heads, head_dim]
    kv_shape = [batch, kv_len_dynamic, head_kv, head_dim]

    # 允许 num_weights (输入的 weights 张量的最后一维) 大于 selected_blocks (K)
    if num_weights is None:
        num_weights = selected_blocks

    weight_shape = [batch, q_len_dynamic, heads, num_weights]
    block_indices_shape = [batch, q_len_dynamic, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(head_dim))

    NK = tilelang.cdiv(head_dim, block_T)
    NV = tilelang.cdiv(head_dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    BS = block_S
    BK = BV = block_T
    num_stages = 2
    threads = 32  # 大于32后就需要把acc_s_cast改成shared了(多了向共享内存的拷贝),但是改成128后延迟没变

    kv_group_num = heads // head_kv

    # GEMM 的 M 维度：如果不满 16，强制 Pad 到 16
    GEMM_M = T.max(kv_group_num, 16)


    @T.prim_func
    def hsa(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(batch, q_len_dynamic, head_kv, threads=threads) as (bx, by, bz):

            Q_shared = T.alloc_shared([GEMM_M, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([GEMM_M, BV], dtype)

            acc_s = T.alloc_fragment([GEMM_M, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([GEMM_M, BS], dtype)
            acc_o = T.alloc_fragment([GEMM_M, BV], accum_dtype)
            
            scores_max = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_sum = T.alloc_fragment([GEMM_M], accum_dtype)

            batch_id = bx
            seq_id = by
            kv_head_id = bz

            q_head_global_start = kv_head_id * kv_group_num

            for i, j in T.Parallel(kv_group_num, head_dim):
                Q_shared[i, j] = Q[batch_id, seq_id, q_head_global_start + i, j]

            T.fill(acc_o, 0)
            
            W_local_shared = T.alloc_shared([kv_group_num, selected_blocks], dtype)
            T.fill(W_local_shared, 0.0)
            for g_idx, k_idx in T.Parallel(kv_group_num, selected_blocks):
                W_local_shared[g_idx, k_idx] = W[batch_id, seq_id, q_head_global_start + g_idx, k_idx]
            
            for k in T.Pipelined(S, num_stages=num_stages):
                blk_idx = BlockIndices[batch_id, seq_id, kv_head_id, k]
                
                if blk_idx >= 0:
                    i_s = blk_idx * BS
                    
                    T.copy(K[batch_id, i_s:i_s + BS, kv_head_id, :], K_shared)

                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for g, v in T.Parallel(GEMM_M, BS):
                        acc_s[g, v] = T.if_then_else(
                            scores_max[g] == -T.infinity(accum_dtype),
                            0.0,
                            T.exp2(acc_s[g, v] * scale - scores_max[g] * scale)
                        )
                    
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    
                    for g, v in T.Parallel(GEMM_M, BS):
                        head_weight = W_local_shared[g, k]  # 从 shared memory 读取
                        acc_s[g, v] = T.if_then_else(
                            scores_sum[g] > 0,
                            head_weight * acc_s[g, v] / scores_sum[g],
                            0.0
                        )
                    
                    T.copy(acc_s, acc_s_cast)

                    T.copy(V[batch_id, i_s:i_s + BS, kv_head_id, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)
            for i, j in T.Parallel(kv_group_num, head_dim):
                Output[batch_id, seq_id, q_head_global_start + i, j] = O_shared[i, j]

    return hsa



class _HSA_prefill(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._fwd_kernel = None

    def forward(self, q, k, v, w, block_indices, block_size, sm_scale=None):
        """
        q: (B, L, HQ, D)
        k,v: (B, L, H, D)
        w: (B, L, HQ, num_weights) 或 (B, L, HQ, S)，其中 num_weights >= S
        block_indices: (B, L, H, S), int32
        """

        B, L, HQ, D = q.shape
        H = k.shape[2]
        S = block_indices.shape[-1]
        # 捕获 weights 的真实 shape，num_weights 可能大于 S
        num_weights = w.shape[-1]
        G = HQ // H
        
        assert HQ % H == 0, f"Heads {HQ} must be divisible by KV Heads {H}"
        
        if self._fwd_kernel is None:
            self._fwd_kernel = hierarchical_sparse_attention(
                batch=B,
                heads=HQ,
                head_dim=D,
                block_size=block_size,
                groups=G,
                selected_blocks=S,
                num_weights=num_weights,
                scale=sm_scale,
            )

        O = self._fwd_kernel(q, k, v, w, block_indices)
        return O


_HSA_PREFILL_MODULE_CACHE = {}

def HSA_prefill_interface(q, k, v, w, block_indices,
                block_size: int,
                sm_scale: float | None = None):
    """
    参数:
        q (torch.Tensor): Query 张量，形状为 (B, L, HQ, D)
        k (torch.Tensor): Key 张量，形状为 (B, L, H, D)
        v (torch.Tensor): Value 张量，形状为 (B, L, H, D)
        w (torch.Tensor): 权重张量，形状为 (B, L, HQ, num_weights) 或 (B, L, HQ, S)，其中 num_weights >= S
        block_indices (torch.Tensor): Block 索引，形状为 (B, L, H, S), int32
        block_size (int): Block 大小
        sm_scale (float, optional): Softmax 缩放因子
    
    返回:
        torch.Tensor: 输出张量，形状为 (B, L, HQ, D)
    """
    B, L, HQ, D = q.shape
    H = k.shape[2]
    S = block_indices.shape[-1]
    num_weights = w.shape[-1] 
    device = q.device
    
    # 使用动态长度标记，确保不同长度的输入可以复用同一个 kernel
    cache_key = (B, HQ, H, D, block_size, S, num_weights, sm_scale, str(device))
    
    if cache_key not in _HSA_PREFILL_MODULE_CACHE:
        module = _HSA_prefill()
        _HSA_PREFILL_MODULE_CACHE[cache_key] = module
    
    module = _HSA_PREFILL_MODULE_CACHE[cache_key]
    
    return module(q, k, v, w, block_indices, block_size, sm_scale)





import math
import torch
import torch.nn.functional as F
from einops import rearrange
def test_hsa_prefill_correctness():
    """
    测试 HSA prefill 阶段的正确性
    """
    import math
    import torch
    
    B = 2
    L = 500  # prefill 长度
    HQ = 8
    H = 1
    D = 128
    S = 8
    block_size = 32
    
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sm_scale = 1.0 / math.sqrt(D)
    
    torch.manual_seed(10)
    
    # 创建输入张量（prefill: q/k/v 都有序列维度 L）
    q = torch.randn((B, L, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, L, H, D), dtype=dtype, device=device)
    v = torch.randn((B, L, H, D), dtype=dtype, device=device)
    
    # 创建 block_indices 和 weights（prefill: 形状为 (B, L, H, S)）
    num_blocks = L // block_size
    block_indices = torch.randint(-1, num_blocks, (B, L, H, S), dtype=torch.int32, device=device)
    weights = torch.rand((B, L, HQ, S), dtype=dtype, device=device)
    
    # TileLang Call
    O_tl = HSA_prefill_interface(q, k, v, weights, block_indices, 
                      block_size=block_size, sm_scale=sm_scale)
    
    # Ref Call
    O_ref = hsa_torch_ref(q, k, v, weights, block_indices,
                         chunk_size=block_size, sm_scale=sm_scale, block_q=1)
    
    def get_abs_err(x, y):
        return (x - y).abs().max().item()
    
    def get_err_ratio(x, y):
        err = (x - y).flatten().square().mean().sqrt().item()
        base = x.flatten().square().mean().sqrt().item()
        return err / (base + 1e-12)
    
    abs_err = get_abs_err(O_ref, O_tl.float())
    err_ratio = get_err_ratio(O_ref, O_tl.float())
    
    print(f"Max Absolute Error: {abs_err:.6f}")
    print(f"Error Ratio: {err_ratio:.6f}")
    
    assert abs_err < 5e-2, f"Error {abs_err} too high"
    print("\n✅ HSA Prefill Test PASSED")


def benchmark_hsa_prefill():
    """
    HSA Prefill 性能基准测试 (vs PyTorch Ref)
    """
    print("\n" + "=" * 80)
    print("=== Benchmarking HSA Prefill Performance ===")
    print("=" * 80)

    B = 2      # Batch size
    L = 512    # Prefill 序列长度
    HQ = 8    # Total Query Heads per GPU (assuming TP)
    H = 1      # Total KV Heads
    D = 128    # Head dimension
    S = 8     # Selected blocks (Sparse)
    block_size = 32
    
    dtype = torch.bfloat16
    device = "cuda"
    
    # Init Tensors（prefill: q/k/v 都有序列维度 L）
    q = torch.randn((B, L, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, L, H, D), dtype=dtype, device=device)
    v = torch.randn((B, L, H, D), dtype=dtype, device=device)
    w = torch.rand((B, L, HQ, S), dtype=dtype, device=device)
    
    num_blocks = L // block_size
    block_indices = torch.randint(0, num_blocks, (B, L, H, S), dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(D)
    
    # 1. Warmup & Correctness Check (Light)
    print("Warming up...")
    for _ in range(5):
        HSA_prefill_interface(q, k, v, w, block_indices, block_size, sm_scale)
        hsa_torch_ref(q, k, v, w, block_indices, chunk_size=block_size, sm_scale=sm_scale, block_q=1)
    torch.cuda.synchronize()
    
    # 2. Benchmark TileLang Kernel
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    iters = 100
    start_event.record()
    for _ in range(iters):
        HSA_prefill_interface(q, k, v, w, block_indices, block_size, sm_scale)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_lat = start_event.elapsed_time(end_event) / iters
    
    # 3. Benchmark PyTorch Ref
    ref_iters = 20  
    start_event_ref = torch.cuda.Event(enable_timing=True)
    end_event_ref = torch.cuda.Event(enable_timing=True)
    
    start_event_ref.record()
    for _ in range(ref_iters):
        hsa_torch_ref(q, k, v, w, block_indices, chunk_size=block_size, sm_scale=sm_scale, block_q=1)
    end_event_ref.record()
    torch.cuda.synchronize()
    
    avg_lat_ref = start_event_ref.elapsed_time(end_event_ref) / ref_iters

    print(f"Config: B={B}, L={L}, HQ={HQ}, H={H}, D={D}, S={S}, BlockSize={block_size}")
    print("-" * 50)
    print(f"PyTorch Ref Avg Latency : {avg_lat_ref:.3f} ms")
    print(f"TileLang Kernel Avg Latency: {avg_lat:.3f} ms")
    print(f"Speedup                    : {avg_lat_ref / avg_lat:.2f}x")
    print("-" * 50)

import pytest

@pytest.mark.parametrize("B, L, HQ, H, D, S, block_size", [
    (1, 64, 8, 1, 64, 8, 32),       # Smallest case
])
def test_hsa_prefill_pytest(B, L, HQ, H, D, S, block_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)
    
    torch.manual_seed(42)
    
    # 准备数据
    q = torch.randn((B, L, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, L, H, D), dtype=dtype, device=device)
    v = torch.randn((B, L, H, D), dtype=dtype, device=device)
    
    num_blocks = L // block_size
    block_indices = torch.randint(0, num_blocks, (B, L, H, S), dtype=torch.int32, device=device)
    weights = torch.rand((B, L, HQ, S), dtype=dtype, device=device)
    
    # 设置一些无效块（-1）
    if L > 1:
        block_indices[:, L//2:, :, :] = -1
    
    # TileLang Output
    out_tl = HSA_prefill_interface(q, k, v, weights, block_indices, 
                      block_size=block_size, sm_scale=sm_scale)
    
    # Reference Output
    out_ref = hsa_torch_ref(q, k, v, weights, block_indices,
                         chunk_size=block_size, sm_scale=sm_scale, block_q=1)
    
    # 验证逻辑
    def get_abs_err(x, y):
        return (x - y).flatten().abs().max().item()

    def get_err_ratio(x, y):
        err = (x - y).flatten().square().mean().sqrt().item()
        base = x.flatten().square().mean().sqrt().item()
        # 避免 base 为 0 时除以 0
        return err / (base + 1e-12)

    def assert_close(prefix, ref, tri, ratio):
        msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
        print(msg)
        assert get_err_ratio(ref, tri) < ratio, msg

    print(f"\nConfig: [B={B}, L={L}, HQ={HQ}, H={H}, D={D}, S={S}]")
    assert_close("Output", out_ref, out_tl.float(), ratio=0.02)
    
    print(f"✅ Test Passed")

def test_hsa_prefill_dynamic_compilation():
    """
    测试对于动态长度 q_len 和 kv_len，是否能够复用已编译的 Kernel
    """
    print("\n" + "=" * 60)
    print("=== Testing HSA Prefill Dynamic Compilation (JIT Reuse) ===")
    print("=" * 60)
    
    # 清空缓存以确保测试环境干净
    global _HSA_PREFILL_MODULE_CACHE
    _HSA_PREFILL_MODULE_CACHE.clear()
    
    B, HQ, H, D = 2, 16, 2, 64
    S, block_size = 8, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    sm_scale = 1.0
    
    # --- 第一次调用：初始长度 ---
    L1 = 500
    print(f"--> Step 1: Running with q_len=kv_len={L1}")
    q1 = torch.randn((B, L1, HQ, D), dtype=dtype, device=device)
    k1 = torch.randn((B, L1, H, D), dtype=dtype, device=device)
    v1 = torch.randn((B, L1, H, D), dtype=dtype, device=device)
    w1 = torch.rand((B, L1, HQ, S), dtype=dtype, device=device)
    idx1 = torch.randint(0, L1 // block_size, (B, L1, H, S), dtype=torch.int32, device=device)
    
    HSA_prefill_interface(q1, k1, v1, w1, idx1, block_size, sm_scale)
    
    cache_len_1 = len(_HSA_PREFILL_MODULE_CACHE)
    print(f"    Cache Size: {cache_len_1}")
    assert cache_len_1 == 1, "Should have compiled once."
    
    # --- 第二次调用：长度变化 ---
    L2 = L1 + 1
    print(f"--> Step 2: Running with q_len=kv_len={L2} (Shape changed)")
    q2 = torch.randn((B, L2, HQ, D), dtype=dtype, device=device)
    k2 = torch.randn((B, L2, H, D), dtype=dtype, device=device)
    v2 = torch.randn((B, L2, H, D), dtype=dtype, device=device)
    w2 = torch.rand((B, L2, HQ, S), dtype=dtype, device=device)
    idx2 = torch.randint(0, L2 // block_size, (B, L2, H, S), dtype=torch.int32, device=device)
    
    HSA_prefill_interface(q2, k2, v2, w2, idx2, block_size, sm_scale)
    
    cache_len_2 = len(_HSA_PREFILL_MODULE_CACHE)
    print(f"    Cache Size: {cache_len_2}")
    
    # 核心检查点：缓存大小不应该增加（应该复用 kernel）
    assert cache_len_2 == cache_len_1, \
        f"Kernel recompiled! Cache grew from {cache_len_1} to {cache_len_2}. T.dynamic() might not be working as expected."
    
    # --- 第三次调用：修改 Block Size (静态参数变化) ---
    print(f"--> Step 3: Running with block_size=32 (Changes expected: +1 Compile)")
    new_block_size = 32
    # 对应的 block index 重建
    idx3 = torch.randint(0, L2 // new_block_size, (B, L2, H, S), dtype=torch.int32, device=device)
    
    HSA_prefill_interface(q2, k2, v2, w2, idx3, new_block_size, sm_scale)
    
    cache_len_3 = len(_HSA_PREFILL_MODULE_CACHE)
    print(f"    Cache Size: {cache_len_3}")
    assert cache_len_3 == cache_len_1 + 1, "Should compile a new kernel for different block_size."
    
    print("✅ Dynamic Compilation Test PASSED")




if __name__ == "__main__":
    test_hsa_prefill_correctness()
    benchmark_hsa_prefill() 
    
    params = [
    # "B, L, HQ, H, D, S, block_size"
    (1, 60, 8, 1, 64, 8, 32),       # Smallest case
    (2, 250, 8, 1, 128, 8, 64),    # Standard case
    (4, 500, 16, 2, 128, 16, 32),  # Large GQA case (G=8)
    (2, 100, 4, 4, 128, 8, 64),    # MHA case (G=1)
    ]
    for p in params:
        test_hsa_prefill_pytest(*p)
    
    test_hsa_prefill_dynamic_compilation()