"""Microbenchmark `hsa_extend_paged_fwd_kernel` block_m/num_warps/num_stages sweep.

Builds synthetic inputs matching the real prefill @16K shape and times just
the kernel — no end-to-end overhead. ~1s per config vs ~60s for a full bench.
"""
import os
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import torch
from sglang.srt.layers.attention.hsa.kernels.hsa_extend import hsa_extend_paged_fwd

device = "cuda"
dtype = torch.bfloat16

# Match real prefill @16K of HSA-345M:
T = 16384         # extend tokens
HQ = 16           # q-heads (HSA branch)
H_kv = 2          # kv-heads
D = 64            # head_dim
TOPK = 32         # selected pages per query/kv-head
PAGE_SIZE = 64
B_seq = 1
MAX_T = 16384 + 200

# Storage pool size (must hold all extend tokens, padded to page boundaries).
Nloc = MAX_T

q = torch.randn(T, HQ, D, device=device, dtype=dtype)
k_cache = torch.randn(Nloc, H_kv, D, device=device, dtype=dtype)
v_cache = torch.randn(Nloc, H_kv, D, device=device, dtype=dtype)
# page_table_1: [B, MAX_T] mapping engine_pos -> kv_loc
page_table_1 = torch.arange(MAX_T, device=device, dtype=torch.int32).unsqueeze(0)
# selected_page_ids: [T, H_kv, TOPK] page IDs (valid range 0..MAX_T/PAGE_SIZE-1)
n_pages = MAX_T // PAGE_SIZE
selected_page_ids = torch.randint(
    0, n_pages, (T, H_kv, TOPK), device=device, dtype=torch.int32
)
hsa_weights = torch.softmax(
    torch.randn(T, HQ, TOPK, device=device, dtype=torch.float32), dim=-1
).to(dtype)
token_to_seq_id = torch.zeros(T, device=device, dtype=torch.int32)

torch.cuda.synchronize()


def time_kernel(block_m, num_warps, num_stages, iters=50, warmup=10):
    # Warmup (also primes JIT cache).
    for _ in range(warmup):
        hsa_extend_paged_fwd(
            q=q, k_cache=k_cache, v_cache=v_cache, page_table_1=page_table_1,
            selected_page_ids=selected_page_ids, hsa_weights=hsa_weights,
            page_size=PAGE_SIZE, sm_scale=D ** -0.5, mask_last_token=True,
            token_to_seq_id=token_to_seq_id,
            block_m=block_m, num_warps=num_warps, num_stages=num_stages,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        hsa_extend_paged_fwd(
            q=q, k_cache=k_cache, v_cache=v_cache, page_table_1=page_table_1,
            selected_page_ids=selected_page_ids, hsa_weights=hsa_weights,
            page_size=PAGE_SIZE, sm_scale=D ** -0.5, mask_last_token=True,
            token_to_seq_id=token_to_seq_id,
            block_m=block_m, num_warps=num_warps, num_stages=num_stages,
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


configs = []
for bm in [1, 2, 4, 8, 16]:
    for nw in [2, 4, 8]:
        for ns in [2, 3, 4]:
            try:
                ms = time_kernel(bm, nw, ns)
                configs.append((bm, nw, ns, ms))
                print(f"BM={bm:>2} NW={nw} NS={ns}: {ms:.3f} ms", flush=True)
            except Exception as e:
                print(f"BM={bm:>2} NW={nw} NS={ns}: FAIL {str(e)[:80]}", flush=True)

print("\n=== TOP 10 ===")
configs.sort(key=lambda x: x[3])
for bm, nw, ns, ms in configs[:10]:
    print(f"  BM={bm:>2} NW={nw} NS={ns}: {ms:.3f} ms")
print("\n=== CURRENT DEFAULT (BM=1 NW=2 NS=2) vs BEST ===")
current = next((c for c in configs if c[:3] == (1, 2, 2)), None)
if current:
    best = configs[0]
    print(f"  current: {current[3]:.3f} ms")
    print(f"  best:    {best[3]:.3f} ms  ({best[:3]})")
    print(f"  speedup: {current[3] / best[3]:.2f}x")
