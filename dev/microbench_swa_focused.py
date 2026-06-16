"""Focused SWA kernel sweep — smaller search space to fit in budget."""
import os
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import torch
import triton
import sys
import time

from sglang.srt.layers.attention.hsa.kernels.hsa_swa_extend import hsa_swa_extend_kernel

device = "cuda"
dtype = torch.bfloat16

# Real shape @16K
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


def time_config(BLOCK_M, BLOCK_N, num_warps, num_stages, iters=20, warmup=4):
    try:
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
    except Exception as e:
        return None


# Smaller focused sweep — based on common Blackwell sweet spots.
# Default is BLOCK_M=16, BLOCK_N=64, NW=4, NS=1.
CONFIGS = [
    # (BM, BN, NW, NS)
    (16, 64, 4, 1),   # current default
    (16, 64, 4, 2),
    (16, 64, 4, 3),
    (16, 64, 8, 1),
    (16, 64, 8, 2),
    (16, 128, 4, 1),
    (16, 128, 4, 2),
    (16, 128, 8, 1),
    (16, 128, 8, 2),
    (32, 64, 4, 1),
    (32, 64, 4, 2),
    (32, 64, 8, 1),
    (32, 64, 8, 2),
    (32, 128, 4, 1),
    (32, 128, 4, 2),
    (32, 128, 8, 1),
    (32, 128, 8, 2),
    (8, 64, 4, 1),
    (8, 128, 4, 1),
    (8, 128, 8, 1),
    (8, 64, 2, 1),
    (8, 64, 4, 2),
]

results = []
t0 = time.time()
for cfg in CONFIGS:
    bm, bn, nw, ns = cfg
    if time.time() - t0 > 400:
        print("Time budget exceeded; stopping.", flush=True)
        break
    ms = time_config(bm, bn, nw, ns)
    if ms is None:
        print(f"  BM={bm:>2} BN={bn:>3} NW={nw} NS={ns}: FAIL", flush=True)
    else:
        results.append((bm, bn, nw, ns, ms))
        print(f"  BM={bm:>2} BN={bn:>3} NW={nw} NS={ns}: {ms:.3f} ms", flush=True)

print("\n=== SORTED RESULTS ===")
results.sort(key=lambda x: x[4])
for r in results:
    print(f"  BM={r[0]:>2} BN={r[1]:>3} NW={r[2]} NS={r[3]}: {r[4]:.3f} ms")

if results:
    default = next((r for r in results if r[:4] == (16, 64, 4, 1)), None)
    best = results[0]
    print("\n=== DEFAULT vs BEST ===")
    if default:
        print(f"  default: {default[4]:.3f} ms ({default[:4]})")
    print(f"  best:    {best[4]:.3f} ms ({best[:4]})")
    if default:
        print(f"  speedup: {default[4] / best[4]:.2f}x")
