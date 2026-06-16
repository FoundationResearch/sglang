"""Direct microbenchmark of the topk_head_maxpool selector kernel.

Bypasses sglang entirely — calls _online_topk_head_maxpool with
realistic HSA prefill shapes (per_qhead path, is_training=False).

Usage:
  python dev/bench_selector_micro.py                       # default L=32768
  LEN=131072 python dev/bench_selector_micro.py            # 128K
  LEN=32768 ITERS=200 python dev/bench_selector_micro.py
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsa-kernel-main"))
from ops.topk_head_maxpool import online_topk_head, TopKMaxPooling_Fused  # noqa: E402

LEN = int(os.environ.get("LEN", "32768"))
ITERS = int(os.environ.get("ITERS", "200"))
WARMUP = int(os.environ.get("WARMUP", "20"))
H_KV = int(os.environ.get("H_KV", "2"))
G = int(os.environ.get("G", "8"))
D = int(os.environ.get("D", "64"))
TOPK = int(os.environ.get("TOPK", "32"))
BLOCK = int(os.environ.get("BLOCK", "64"))
WINDOW = int(os.environ.get("WINDOW", "512"))


def make_inputs(L):
    S = L // BLOCK
    torch.manual_seed(0)
    q = torch.randn(1, L, H_KV, G, D, dtype=torch.bfloat16, device="cuda") * 0.1
    lmks = torch.randn(1, S, H_KV * G, D, dtype=torch.bfloat16, device="cuda") * 0.1
    lse_swa = torch.randn(1, L, H_KV * G, dtype=torch.float32, device="cuda")
    return q, lmks, lse_swa


def run_once(q, lmks, lse_swa, L):
    out_idx, out_scores = online_topk_head(
        q, lmks,
        topk=TOPK,
        block_size=BLOCK,
        window_size=WINDOW,
        is_causal=True,
        G=G,
        lse_swa=lse_swa,
        q_offset=0,
        is_training=False,
    )
    return out_idx, out_scores


def main():
    print(f"==== selector microbench  L={LEN}  H_KV={H_KV} G={G} D={D} TOPK={TOPK} ====")
    q, lmks, lse_swa = make_inputs(LEN)
    # Warmup (JIT compile)
    for _ in range(WARMUP):
        run_once(q, lmks, lse_swa, LEN)
    torch.cuda.synchronize()

    # Time using cuda events
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(ITERS):
        start.record()
        run_once(q, lmks, lse_swa, LEN)
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))  # ms
    times.sort()
    n = len(times)
    median = times[n // 2]
    p10 = times[n // 10]
    mean = sum(times) / n
    print(f"  median:  {median:8.3f} ms")
    print(f"  p10   :  {p10:8.3f} ms")
    print(f"  mean  :  {mean:8.3f} ms")


if __name__ == "__main__":
    main()
