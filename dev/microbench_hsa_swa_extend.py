"""Microbench hsa_swa_extend_kernel BLOCK_M/BLOCK_N/num_warps/num_stages sweep."""
import os
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import torch
import triton

# Import the kernel + monkey-patch the wrapper to expose tuning knobs.
from sglang.srt.layers.attention.hsa.kernels import hsa_swa_extend
from sglang.srt.layers.attention.hsa.kernels.hsa_swa_extend import hsa_swa_extend_kernel

device = "cuda"
dtype = torch.bfloat16

# Match real prefill @16K HSA-345M:
T = 16384
HQ = 16
H = 2
D = 64
hsa_window = 512
PAGE_SIZE = 64
prefix_len = 0
extend_len = T
total_kv = prefix_len + extend_len

q = torch.randn(T, HQ, D, device=device, dtype=dtype)
k_cache = torch.randn(total_kv, H, D, device=device, dtype=dtype)
v_cache = torch.randn(total_kv, H, D, device=device, dtype=dtype)
kv_indices = torch.arange(total_kv, device=device, dtype=torch.int64)


def run_kernel(BLOCK_M, BLOCK_N, num_warps, num_stages):
    out = torch.empty((T, HQ, D), device=device, dtype=torch.bfloat16)
    lse = torch.full((T, HQ), float("-inf"), device=device, dtype=torch.float32)
    grid = (triton.cdiv(T, BLOCK_M), H)
    hsa_swa_extend_kernel[grid](
        q, k_cache, v_cache, kv_indices, out, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        sm_scale=float(D ** -0.5),
        T=T, HQ=HQ, H=H, D=D,
        TOTAL_KV=int(total_kv),
        PREFIX_LEN=int(prefix_len),
        SW=int(hsa_window),
        PAGE_SIZE=int(PAGE_SIZE),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, lse


def time_kernel(BLOCK_M, BLOCK_N, num_warps, num_stages, iters=40, warmup=8):
    for _ in range(warmup):
        run_kernel(BLOCK_M, BLOCK_N, num_warps, num_stages)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_kernel(BLOCK_M, BLOCK_N, num_warps, num_stages)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


configs = []
for bm in [8, 16, 32, 64]:
    for bn in [32, 64, 128]:
        for nw in [2, 4, 8]:
            for ns in [1, 2, 3]:
                try:
                    ms = time_kernel(bm, bn, nw, ns)
                    configs.append((bm, bn, nw, ns, ms))
                    print(f"BM={bm:>2} BN={bn:>3} NW={nw} NS={ns}: {ms:.3f} ms", flush=True)
                except Exception as e:
                    print(f"BM={bm:>2} BN={bn:>3} NW={nw} NS={ns}: FAIL {str(e)[:60]}", flush=True)

print("\n=== TOP 10 ===")
configs.sort(key=lambda x: x[4])
for bm, bn, nw, ns, ms in configs[:10]:
    print(f"  BM={bm:>2} BN={bn:>3} NW={nw} NS={ns}: {ms:.3f} ms")

print("\n=== CURRENT DEFAULT (BM=16 BN=64 NW=4 NS=1) vs BEST ===")
current = next((c for c in configs if c[:4] == (16, 64, 4, 1)), None)
if current:
    best = configs[0]
    print(f"  current: {current[4]:.3f} ms  config={current[:4]}")
    print(f"  best:    {best[4]:.3f} ms  config={best[:4]}")
    print(f"  speedup: {current[4] / best[4]:.2f}x")
