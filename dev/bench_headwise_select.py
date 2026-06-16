"""P2 microbench: headwise prefill chunk-selection, fused softmax kernel vs the
torch einsum reference. Measures per-call latency and peak memory at growing
context to show the fused path is faster and avoids the [T, h_q, S] fp32
materialization that OOMs the einsum path on long-context downstream eval.

Run: CUDA_VISIBLE_DEVICES=5 python dev/bench_headwise_select.py
"""
import math
import os
import sys
import time

import torch

OPS = os.path.join(os.path.dirname(__file__), "hsa-kernel-main", "ops")
if OPS not in sys.path:
    sys.path.insert(0, OPS)
from topk_head_softmax_official import online_softmax_topk_head  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16
# hsa345m_real topology: 16 hsa q-heads, 2 kv-heads (G=8), head_dim 64.
H_Q, H_KV, D = 16, 2, 64
G = H_Q // H_KV
TOPK, CHUNK, WINDOW = 32, 64, 512


def einsum_select(q, lmk, lse_swa, prior_b, prefix_len):
    """Mirror hsa_backend.py headwise einsum path (no prior_b, raw max-over-G)."""
    T = q.shape[0]
    S = lmk.shape[0]
    sm = 1.0 / math.sqrt(D)
    scores_pqh = torch.einsum("thd,shd->ths", q.float(), lmk.float()) * sm  # [T,h_q,S]
    q_pos = torch.arange(prefix_len, prefix_len + T, device=DEV)
    chunk_end = torch.arange(1, S + 1, device=DEV) * CHUNK - 1
    chunk_start = torch.arange(0, S, device=DEV) * CHUNK
    vis = (chunk_end.unsqueeze(0) < q_pos.unsqueeze(1)) & (
        chunk_start.unsqueeze(0) < (q_pos.unsqueeze(1) - WINDOW + 1)
    )
    scores_kv = scores_pqh.view(T, H_KV, G, S).max(dim=2).values
    scores_kv = scores_kv.masked_fill(~vis.unsqueeze(1), float("-inf"))
    eff = min(TOPK, S)
    _, idx = scores_kv.topk(eff, dim=-1)
    return idx


def fused_select(q, lmk, lse_swa, prior_b, prefix_len):
    idx, scr = online_softmax_topk_head(
        q.unsqueeze(0).contiguous(), lmk.unsqueeze(0).contiguous(),
        lse_swa.unsqueeze(0).contiguous(), TOPK, CHUNK, WINDOW,
        is_causal=True, q_offset=prefix_len, is_training=False,
        bias=prior_b.unsqueeze(0), G=G,
    )
    return idx


def bench(fn, *args, iters=10):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for _ in range(3):  # warmup (kernel compile / cache)
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e9
    return ms, peak


def main():
    print(f"{'T':>8} {'S':>6} | {'einsum ms':>10} {'einsum GB':>10} | "
          f"{'fused ms':>9} {'fused GB':>9} | {'speedup':>7}")
    for T in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        S = T // CHUNK
        torch.manual_seed(0)
        q = torch.randn(T, H_Q, D, device=DEV, dtype=DT)
        lmk = torch.randn(S, H_Q, D, device=DEV, dtype=DT)
        lse_swa = torch.randn(T, H_Q, device=DEV, dtype=torch.float32)
        prior_b = torch.randn(S, H_Q, device=DEV, dtype=torch.float32)
        prefix_len = 0
        try:
            e_ms, e_gb = bench(einsum_select, q, lmk, lse_swa, prior_b, prefix_len)
        except torch.cuda.OutOfMemoryError:
            e_ms, e_gb = float("nan"), float("nan")
            torch.cuda.empty_cache()
        f_ms, f_gb = bench(fused_select, q, lmk, lse_swa, prior_b, prefix_len)
        sp = e_ms / f_ms if e_ms == e_ms else float("nan")
        print(f"{T:>8} {S:>6} | {e_ms:>10.3f} {e_gb:>10.3f} | "
              f"{f_ms:>9.3f} {f_gb:>9.3f} | {sp:>6.2f}x")


if __name__ == "__main__":
    main()
