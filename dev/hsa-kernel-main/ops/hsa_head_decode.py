# ruff: noqa
import tilelang
from tilelang import language as T
import tilelang.testing
import torch
import torch.nn.functional as F
from einops import rearrange
tilelang.testing.set_random_seed(0)
import math

from einops import rearrange
def hsa_torch_ref(q, k, v, weights, indices, *, chunk_size: int, sm_scale: float, block_q: int):
    """
    - q: (B, L, HQ, D) 或 (B, HQ, D)（decode 阶段）
    - k, v: (B, L, H, D)
    - weights: (B, q_blocks, HQ, K) 或 (B, HQ, K)（decode 阶段）
    - indices: (B, q_blocks, H, K) 或 (B, L, H, K) 或 (B, H, K)（decode 阶段）
    - 返回: o_ref: (B, L, HQ, D) 或 (B, HQ, D) float32
    """
    # 保存原始维度信息
    orig_q_dim = q.dim()
    if orig_q_dim == 3:
        # decode 阶段：q 形状为 (B, HQ, D)，添加 L=1 维度
        q = q.unsqueeze(1)  # (B, 1, HQ, D)
        # 如果 weights 和 indices 也是三维的，也添加维度
        if weights.dim() == 3:
            weights = weights.unsqueeze(1)  # (B, 1, HQ, K)
        if indices.dim() == 3:
            indices = indices.unsqueeze(1)  # (B, 1, H, K)
    
    B, L, HQ, D = q.shape
    H = k.shape[2]
    orig_kv_l = k.shape[1]
    G = HQ // H
    q_blocks = L // block_q
    device = q.device
    
    # --- 处理非整数倍长度：进行 Padding ---
    pad_len = (chunk_size - (orig_kv_l % chunk_size)) % chunk_size
    if pad_len > 0:
        # 在序列维度 (dim=1) 填充 0
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
    
    new_kv_l = k.shape[1]

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
    N = new_kv_l // chunk_size
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

    # 在 S 维做 softmax
    p = torch.softmax(qk, dim=3)  # S 维

    # o_k: (B, Bq, X, K, HQ, D)
    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    # 权重：将 valid_mask 扩到 HQ 维，然后无效块置 0
    w_masked = weights.clone()
    valid_mask_expanded = torch.repeat_interleave(valid_mask, dim=-2, repeats=G)  # (B, Bq, HQ, K)
    w_masked = w_masked.masked_fill(~valid_mask_expanded, 0)
    w_exp = w_masked.float()  # (B, Bq, HQ, K)

    # 按 K 聚合
    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'B Bq X hq d -> B (Bq X) hq d')  # 回到 (B, L, HQ, D)
    
    # 如果是 decode 阶段，移除 L 维度
    if orig_q_dim == 3:
        o_ref = o_ref.squeeze(1)  # (B, HQ, D)
    
    return o_ref.to(torch.float32)

@tilelang.jit(
    out_idx=[-1], 
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def hierarchical_sparse_attention_decode(batch,
                                  heads,
                                  head_kv,
                                  head_dim,
                                  block_size,
                                  selected_blocks,
                                  num_weights=None,
                                  scale=None):
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    max_s = T.dynamic("max_s")

    q_shape = [batch, heads, head_dim]
    kv_shape = [batch, max_s, head_kv, head_dim]
    
    # 允许 num_weights (输入的 weights 张量的最后一维) 大于 selected_blocks (K)
    if num_weights is None:
        num_weights = selected_blocks
    
    weight_shape = [batch, heads, num_weights]
    block_indices_shape = [batch, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    
    dtype = "bfloat16"
    accum_dtype = "float"

    BS = block_size
    BK = BV = min(128, tilelang.math.next_power_of_2(head_dim))
    
    kv_group_num = heads // head_kv
    
    # GEMM 的 M 维度：如果不满 16，强制 Pad 到 16
    GEMM_M = T.max(kv_group_num, 16)

    @T.prim_func
    def hsa_decode(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor(weight_shape, dtype),
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(batch, head_kv, threads=32) as (bx, by):

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
            kv_head_id = by
            
            q_head_global_start = kv_head_id * kv_group_num
            
            for i, j in T.Parallel(kv_group_num, head_dim):
                Q_shared[i, j] = Q[batch_id, q_head_global_start + i, j]

            T.fill(acc_o, 0)
            
            W_local_shared = T.alloc_shared([kv_group_num, selected_blocks], dtype)
            T.fill(W_local_shared, 0.0)
            for g_idx, k_idx in T.Parallel(kv_group_num, selected_blocks):
                W_local_shared[g_idx, k_idx] = W[batch_id, q_head_global_start + g_idx, k_idx]
            
            for k in T.Pipelined(selected_blocks, num_stages=2):
                blk_idx = BlockIndices[batch_id, kv_head_id, k]
                
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
                    
                    # [修改] 从 shared memory 中读取每个 query head 对应的权重
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
                Output[batch_id, q_head_global_start + i, j] = O_shared[i, j]

    return hsa_decode




class _HSA_decode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._fwd_kernel = None

    def forward(self, q, k, v, w, block_indices, block_size, sm_scale=None):
        """
        q: (B, HQ, D)
        k,v: (B, max_s, H, D)
        w: (B, HQ, num_weights) 或 (B, HQ, S)，其中 num_weights >= S
        block_indices: (B, H, S), int32
        """
        B, HQ, D = q.shape
        _, max_s, H, _ = k.shape
        S = block_indices.shape[-1]
        
        # 捕获 weights 的真实 shape，num_weights 可能大于 S
        num_weights = w.shape[-1]
        
        assert HQ % H == 0, f"Heads {HQ} must be divisible by KV Heads {H}"
        
        if self._fwd_kernel is None:
            self._fwd_kernel = hierarchical_sparse_attention_decode(
                batch=B,
                heads=HQ,
                head_kv=H,
                head_dim=D,
                block_size=block_size,
                selected_blocks=S,
                num_weights=num_weights,  # 传入真实的权重维度
                scale=sm_scale,
            )

        O = self._fwd_kernel(q, k, v, w, block_indices)
        return O


_HSA_MODULE_CACHE = {}

def HSA_decode_interface(q, k, v, w, block_indices,
               block_size,
               sm_scale=None):
    """
    参数:
        q (torch.Tensor): Query 张量，形状为 (B, HQ, D)
        k (torch.Tensor): Key 缓存张量，形状为 (B, max_s, H, D)
        v (torch.Tensor): Value 缓存张量，形状为 (B, max_s, H, D)
        w (torch.Tensor): 权重张量，形状为 (B, HQ, num_weights) 或 (B, HQ, S)，其中 num_weights >= S
        block_indices (torch.Tensor): Block 索引，形状为 (B, H, S), int32
        block_size (int): Block 大小
        sm_scale (float, optional): Softmax 缩放因子
    
    返回:
        torch.Tensor: 输出张量，形状为 (B, HQ, D)
    """
    B, HQ, D = q.shape
    _, max_s, H, _ = k.shape
    S = block_indices.shape[-1]
    num_weights = w.shape[-1]  # 捕获真实的权重维度
    device = q.device
    
    cache_key = (B, HQ, H, D, block_size, S, num_weights, sm_scale, str(device))
    
    if cache_key not in _HSA_MODULE_CACHE:
        module = _HSA_decode()
        _HSA_MODULE_CACHE[cache_key] = module
    
    module = _HSA_MODULE_CACHE[cache_key]
    
    return module(q, k, v, w, block_indices, block_size, sm_scale)


def test_hsa_decode_correctness():
    """
    测试 HSA decode 阶段的正确性
    """
    import math
    import torch
    
    B = 4
    HQ = 16
    H = 2
    D = 128
    S = 8
    block_size = 64
    max_cache_len = 1000
    
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sm_scale = 1.0 / math.sqrt(D)
    
    torch.manual_seed(10)
    
    q = torch.randn((B, HQ, D), dtype=dtype, device=device)
    k_cache = torch.randn((B, max_cache_len, H, D), dtype=dtype, device=device)
    v_cache = torch.randn((B, max_cache_len, H, D), dtype=dtype, device=device)
    
    block_indices = torch.randint(0, max_cache_len // block_size, (B, H, S), dtype=torch.int32, device=device)
    weights = torch.rand((B, HQ, S), dtype=dtype, device=device)
    
    # TileLang Call
    O_tl = HSA_decode_interface(q, k_cache, v_cache, weights, block_indices, 
                      block_size=block_size, sm_scale=sm_scale)
    
    # Ref Call
    O_ref = hsa_torch_ref(q, k_cache, v_cache, weights, block_indices,
                         chunk_size=block_size, sm_scale=sm_scale, block_q=1)
    
    def get_abs_err(x, y):
        return (x - y).abs().max().item()

    abs_err = get_abs_err(O_ref, O_tl.float())
    print(f"Max Absolute Error: {abs_err:.6f}")
    
    assert abs_err < 1e-2, f"Error {abs_err} too high"
    print("\n✅ HSA Decode Test PASSED")




def benchmark_hsa_decode():
    """
    HSA Decode 性能基准测试 (vs PyTorch Ref)
    """
    print("\n" + "=" * 80)
    print("=== Benchmarking HSA Decode Performance ===")
    print("=" * 80)

    # 模拟 Llama-2-70B GQA 设置 (Group=8)
    B = 32
    HQ = 64  # Total Query Heads per GPU (assuming TP)
    H = 8    # Total KV Heads
    D = 128
    S = 64   # Selected blocks (Sparse)
    block_size = 64
    max_cache_len = 4096
    
    dtype = torch.bfloat16
    device = "cuda"
    
    # Init Tensors
    q = torch.randn((B, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, max_cache_len, H, D), dtype=dtype, device=device)
    v = torch.randn((B, max_cache_len, H, D), dtype=dtype, device=device)
    w = torch.rand((B, HQ, S), dtype=dtype, device=device)
    block_indices = torch.randint(0, max_cache_len // block_size, (B, H, S), dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(D)
    
    # 1. Warmup & Correctness Check (Light)
    print("Warming up...")
    for _ in range(5):
        HSA_decode_interface(q, k, v, w, block_indices, block_size, sm_scale)
        hsa_torch_ref(q, k, v, w, block_indices, chunk_size=block_size, sm_scale=sm_scale, block_q=1, )
    torch.cuda.synchronize()
    
    # 2. Benchmark TileLang Kernel
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    iters = 100
    start_event.record()
    for _ in range(iters):
        HSA_decode_interface(q, k, v, w, block_indices, block_size, sm_scale)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_lat = start_event.elapsed_time(end_event) / iters
    
    # 3. Benchmark Matrix (PyTorch Ref)
    # Ref 实现通常包含大量显存搬运，次数少一点避免太慢
    ref_iters = 20  
    start_event_ref = torch.cuda.Event(enable_timing=True)
    end_event_ref = torch.cuda.Event(enable_timing=True)
    
    start_event_ref.record()
    for _ in range(ref_iters):
        hsa_torch_ref(q, k, v, w, block_indices, chunk_size=block_size, sm_scale=sm_scale, block_q=1, )
    end_event_ref.record()
    torch.cuda.synchronize()
    
    avg_lat_ref = start_event_ref.elapsed_time(end_event_ref) / ref_iters

    print(f"Config: B={B}, HQ={HQ}, H={H}, D={D}, S={S}, BlockSize={block_size}")
    print("-" * 50)
    print(f"PyTorch Ref Avg Latency : {avg_lat_ref:.3f} ms")
    print(f"TileLang Kernel Avg Latency: {avg_lat:.3f} ms")
    print(f"Speedup                    : {avg_lat_ref / avg_lat:.2f}x")
    print("-" * 50)


import pytest

@pytest.mark.parametrize("B, HQ, H, D, S, block_size, max_cache_len", [
    (1, 16, 2, 64, 8, 64, 512),
])
def test_hsa_decode_pytest(B, HQ, H, D, S, block_size, max_cache_len):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)
    
    torch.manual_seed(42)
    
    # 准备数据
    q = torch.randn((B, HQ, D), dtype=dtype, device=device)
    k_cache = torch.randn((B, max_cache_len, H, D), dtype=dtype, device=device)
    v_cache = torch.randn((B, max_cache_len, H, D), dtype=dtype, device=device)
    
    block_indices = torch.randint(0, max_cache_len // block_size, (B, H, S), dtype=torch.int32, device=device)
    weights = torch.rand((B, HQ, S), dtype=dtype, device=device)
    
    if B > 1:
        block_indices[-1, :, :] = -1
    
    # TileLang Output
    out_tl = HSA_decode_interface(q, k_cache, v_cache, weights, block_indices, 
                      block_size=block_size,  sm_scale=sm_scale)
    
    # Reference Output
    out_ref = hsa_torch_ref(q, k_cache, v_cache, weights, block_indices,
                         chunk_size=block_size, sm_scale=sm_scale, block_q=1, )
    

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

    print(f"\nConfig: [B={B}, HQ={HQ}, H={H}, D={D}, S={S}]")
    assert_close("Output", out_ref, out_tl.float(), ratio=0.02)
    
    print(f"✅ Test Passed")


def test_hsa_decode_dynamic_compilation():
    """
    测试对于动态长度 max_s，是否能够复用已编译的 Kernel
    """
    print("\n" + "=" * 60)
    print("=== Testing HSA Decode Dynamic Compilation (JIT Reuse) ===")
    print("=" * 60)
    
    # 清空缓存以确保测试环境干净
    global _HSA_MODULE_CACHE
    _HSA_MODULE_CACHE.clear()
    
    B, HQ, H, D = 2, 16, 2, 64
    S, block_size = 8, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    sm_scale = 1.0
    
    max_s_1 = 1000
    print("--> Step 1: Running with max_s=", max_s_1)
    q1 = torch.randn((B, HQ, D), dtype=dtype, device=device)
    k1 = torch.randn((B, max_s_1, H, D), dtype=dtype, device=device)
    v1 = torch.randn((B, max_s_1, H, D), dtype=dtype, device=device)
    w1 = torch.rand((B, HQ, S), dtype=dtype, device=device)
    idx1 = torch.randint(0, max_s_1 // block_size, (B, H, S), dtype=torch.int32, device=device)
    
    HSA_decode_interface(q1, k1, v1, w1, idx1, block_size, sm_scale)
    
    cache_len_1 = len(_HSA_MODULE_CACHE)
    print(f"    Cache Size: {cache_len_1}")
    assert cache_len_1 == 1, "Should have compiled once."
    
    max_s_2 = max_s_1+1
    print("--> Step 2: Running with max_s=", max_s_2)
    q2 = torch.randn((B, HQ, D), dtype=dtype, device=device)
    k2 = torch.randn((B, max_s_2, H, D), dtype=dtype, device=device) # Shape changed
    v2 = torch.randn((B, max_s_2, H, D), dtype=dtype, device=device)
    w2 = torch.rand((B, HQ, S), dtype=dtype, device=device)
    idx2 = torch.randint(0, max_s_2 // block_size, (B, H, S), dtype=torch.int32, device=device)
    
    HSA_decode_interface(q2, k2, v2, w2, idx2, block_size, sm_scale)
    
    cache_len_2 = len(_HSA_MODULE_CACHE)
    print(f"    Cache Size: {cache_len_2}")
    
    # 核心检查点：缓存大小不应该增加
    assert cache_len_2 == cache_len_1, \
        f"Kernel recompiled! Cache grew from {cache_len_1} to {cache_len_2}. T.dynamic() might not be working as expected."
    
    # --- 第三次调用：修改 Block Size (静态参数变化) ---
    print("--> Step 3: Running with block_size=32 (Changes expected: +1 Compile)")
    new_block_size = 32
    # 对应的 block index 重建
    idx3 = torch.randint(0, max_s_2 // new_block_size, (B, H, S), dtype=torch.int32, device=device)
    
    HSA_decode_interface(q2, k2, v2, w2, idx3, new_block_size, sm_scale)
    
    cache_len_3 = len(_HSA_MODULE_CACHE)
    print(f"    Cache Size: {cache_len_3}")
    assert cache_len_3 == cache_len_1 + 1, "Should compile a new kernel for different block_size."

    print("✅ Dynamic Compilation Test PASSED")


if __name__ == "__main__":
    test_hsa_decode_correctness()
    
    benchmark_hsa_decode()
    param_list=[  
                # "B, HQ, H, D, S, block_size, max_cache_len"
                (1, 16, 1, 64, 8, 64, 512),       # Smallest case
                (4, 32, 4, 128, 16, 64, 1024),    # Standard case
                (8, 64, 8, 128, 32, 32, 2000),    # Large GQA case (G=8)
                (2, 4, 4, 128, 8, 64, 100),       # MHA case (G=1), check padding logic
                
        
    ]
    for p in param_list:
        test_hsa_decode_pytest(*p)
        
        
    test_hsa_decode_dynamic_compilation()