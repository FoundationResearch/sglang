"""
Benchmark: Dense extend on ALL heads vs SWA-only heads.

Tests at realistic model scales to determine whether head subsetting saves time.
Uses direct kernel call (no Python proxy overhead).

Usage:
  CUDA_VISIBLE_DEVICES=0 python dev/bench_head_subset.py
"""
import torch
import time


def bench_extend_kernel_direct():
    """Benchmark the triton extend kernel directly with different head counts."""
    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        extend_attention_fwd_unified,
    )

    device = "cuda"
    dtype = torch.bfloat16
    warmup = 5
    iters = 20

    print("=" * 80)
    print("DIRECT KERNEL BENCHMARK: extend_attention_fwd_unified")
    print("=" * 80)
    print()

    configs = [
        # (label, seq_len, HQ_total, HQ_swa, H_total, H_swa, D)
        # 7B-class: 32 q-heads, 8 kv-heads, 75/25 split
        ("7B-like 1K",   1024, 32, 24, 8, 6, 128),
        ("7B-like 4K",   4096, 32, 24, 8, 6, 128),
        ("7B-like 8K",   8192, 32, 24, 8, 6, 128),
        ("7B-like 16K", 16384, 32, 24, 8, 6, 128),
        # 7B-class: 50/50 split
        ("7B 50/50 1K",  1024, 32, 16, 8, 4, 128),
        ("7B 50/50 4K",  4096, 32, 16, 8, 4, 128),
        ("7B 50/50 8K",  8192, 32, 16, 8, 4, 128),
        ("7B 50/50 16K",16384, 32, 16, 8, 4, 128),
        # 70B-class: 64 q-heads, 8 kv-heads, 75/25 split
        ("70B-like 1K",  1024, 64, 48, 8, 6, 128),
        ("70B-like 4K",  4096, 64, 48, 8, 6, 128),
        ("70B-like 8K",  8192, 64, 48, 8, 6, 128),
    ]

    header = f"{'Config':<20} {'SeqLen':>7} {'All(ms)':>9} {'SWA(ms)':>9} {'Speedup':>8} {'HQ_all':>6} {'HQ_swa':>6}"
    print(header)
    print("-" * len(header))

    for label, seq_len, HQ_total, HQ_swa, H_total, H_swa, D in configs:
        B = 1  # batch size = 1
        T = seq_len  # all tokens in extend

        # Build fake inputs
        q_all = torch.randn(T, HQ_total, D, device=device, dtype=dtype)
        o_all = torch.empty_like(q_all)
        k_buf_all = torch.randn(T + 256, H_total, D, device=device, dtype=dtype)
        v_buf_all = torch.randn(T + 256, H_total, D, device=device, dtype=dtype)

        # SWA-only slices (strided views)
        q_swa = q_all[:, :HQ_swa, :]
        o_swa = o_all[:, :HQ_swa, :]  # strided output
        k_buf_swa = k_buf_all[:, :H_swa, :]
        v_buf_swa = v_buf_all[:, :H_swa, :]

        # Metadata: single sequence, no prefix
        qo_indptr = torch.tensor([0, T], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, T], dtype=torch.int32, device=device)
        kv_indices = torch.arange(T, dtype=torch.int64, device=device)
        prefix_lens = torch.tensor([0], dtype=torch.int32, device=device)

        sm_scale = D ** -0.5

        # Warmup ALL heads
        for _ in range(warmup):
            o_tmp = torch.empty_like(q_all)
            extend_attention_fwd_unified(
                q_all, o_tmp, k_buf_all, v_buf_all,
                qo_indptr, kv_indptr, kv_indices, prefix_lens,
                T, sm_scale=sm_scale,
            )

        # Bench ALL heads
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            o_tmp = torch.empty_like(q_all)
            extend_attention_fwd_unified(
                q_all, o_tmp, k_buf_all, v_buf_all,
                qo_indptr, kv_indptr, kv_indices, prefix_lens,
                T, sm_scale=sm_scale,
            )
        torch.cuda.synchronize()
        ms_all = (time.perf_counter() - t0) / iters * 1000

        # Warmup SWA heads
        for _ in range(warmup):
            o_tmp_swa = torch.empty(T, HQ_swa, D, device=device, dtype=dtype)
            extend_attention_fwd_unified(
                q_swa, o_tmp_swa, k_buf_swa, v_buf_swa,
                qo_indptr, kv_indptr, kv_indices, prefix_lens,
                T, sm_scale=sm_scale,
            )

        # Bench SWA heads
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            o_tmp_swa = torch.empty(T, HQ_swa, D, device=device, dtype=dtype)
            extend_attention_fwd_unified(
                q_swa, o_tmp_swa, k_buf_swa, v_buf_swa,
                qo_indptr, kv_indptr, kv_indices, prefix_lens,
                T, sm_scale=sm_scale,
            )
        torch.cuda.synchronize()
        ms_swa = (time.perf_counter() - t0) / iters * 1000

        speedup = ms_all / ms_swa
        print(f"{label:<20} {seq_len:>7} {ms_all:>9.2f} {ms_swa:>9.2f} {speedup:>7.2f}x {HQ_total:>6} {HQ_swa:>6}")


if __name__ == "__main__":
    bench_extend_kernel_direct()
