"""
Benchmark: FA3 (flash_attn_func) vs tilelang flex_attn_tl (mask_lmk=True).

Fixed shape:
    batch = 4, seq_len = 8192, h_q = h_kv = 32, head_dim = 128, dtype = bf16

Measures forward / backward / total time (median over multiple runs) for
both kernels under training setting (Lq == Lk, requires_grad=True).

Run:
    cd /data/workspace/wxy_local/InfiniteLongLM
    python unittests/bench_fa3_vs_flex_attn_tl.py
"""

import os
import sys
import statistics

import torch

# Make repo root importable when invoked as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ops.flex_attn_tilelang import flex_attn_tl, flex_attn_tl_two_phase  # noqa: E402
from ops.example_gqa_bwd import attention as gqa_attention  # noqa: E402
from flash_attn_interface import flash_attn_func  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH = 4
SEQ_LEN = 8192
H_Q = 32
H_KV = 32
HEAD_DIM = 128
DTYPE = torch.bfloat16
DEVICE = "cuda"

# flex_attn_tl-specific knobs (typical training setting in modeling_qwen_lhsa).
WINDOW_SIZE = SEQ_LEN
CHUNK_SIZE = 64

WARMUP = 10
ITERS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_inputs_bhld():
    """Inputs in (B, H, L, D) layout for flex_attn_tl."""
    torch.manual_seed(0)
    q = torch.randn(BATCH, H_Q, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE,
                    requires_grad=True)
    k = torch.randn(BATCH, H_KV, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE,
                    requires_grad=True)
    v = torch.randn(BATCH, H_KV, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE,
                    requires_grad=True)
    return q, k, v


def _make_inputs_blhd():
    """Inputs in (B, L, H, D) layout for FA3."""
    torch.manual_seed(0)
    q = torch.randn(BATCH, SEQ_LEN, H_Q, HEAD_DIM, device=DEVICE, dtype=DTYPE,
                    requires_grad=True)
    k = torch.randn(BATCH, SEQ_LEN, H_KV, HEAD_DIM, device=DEVICE, dtype=DTYPE,
                    requires_grad=True)
    v = torch.randn(BATCH, SEQ_LEN, H_KV, HEAD_DIM, device=DEVICE, dtype=DTYPE,
                    requires_grad=True)
    return q, k, v


def _bench(fwd_fn, bwd_fn, warmup=WARMUP, iters=ITERS):
    """Return (fwd_ms, bwd_ms, total_ms) medians across iters runs.

    fwd_fn(): runs forward, returns the loss tensor.
    bwd_fn(loss): runs backward on loss.
    """
    # Warmup.
    for _ in range(warmup):
        loss = fwd_fn()
        bwd_fn(loss)
    torch.cuda.synchronize()

    fwd_times, bwd_times, total_times = [], [], []
    for _ in range(iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev2 = torch.cuda.Event(enable_timing=True)

        ev0.record()
        loss = fwd_fn()
        ev1.record()
        bwd_fn(loss)
        ev2.record()
        torch.cuda.synchronize()

        fwd_times.append(ev0.elapsed_time(ev1))
        bwd_times.append(ev1.elapsed_time(ev2))
        total_times.append(ev0.elapsed_time(ev2))

    return (
        statistics.median(fwd_times),
        statistics.median(bwd_times),
        statistics.median(total_times),
    )


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
def bench_fa3():
    q, k, v = _make_inputs_blhd()

    def fwd():
        # FA3: causal=True; output shape (B, L, H, D)
        out = flash_attn_func(q, k, v, causal=True)
        # flash_attn_interface.flash_attn_func may return (out, lse) or just out.
        if isinstance(out, tuple):
            out = out[0]
        # Use sum() as a cheap scalar loss to drive backward.
        return out.float().sum()

    def bwd(loss):
        for t in (q, k, v):
            if t.grad is not None:
                t.grad = None
        loss.backward()

    return _bench(fwd, bwd)


def bench_gqa_baseline(use_atomic=True):
    """Tilelang causal GQA baseline from ops/example_gqa_bwd.py.

    Same (B, L, H, D) layout as FA3. `use_atomic=True` uses the atomic-add
    backward; `use_atomic=False` uses the split (per-group) backward.

    NOTE: this kernel is hard-coded to float16, so we feed fp16 inputs here
    (only this baseline uses fp16; FA3 / Flex-TL still use bf16).
    """
    torch.manual_seed(0)
    q = torch.randn(BATCH, SEQ_LEN, H_Q, HEAD_DIM, device=DEVICE, dtype=torch.float16,
                    requires_grad=True)
    k = torch.randn(BATCH, SEQ_LEN, H_KV, HEAD_DIM, device=DEVICE, dtype=torch.float16,
                    requires_grad=True)
    v = torch.randn(BATCH, SEQ_LEN, H_KV, HEAD_DIM, device=DEVICE, dtype=torch.float16,
                    requires_grad=True)
    groups = H_Q // H_KV

    def fwd():
        out = gqa_attention(q, k, v, True, groups, use_atomic)
        return out.float().sum()

    def bwd(loss):
        for t in (q, k, v):
            if t.grad is not None:
                t.grad = None
        loss.backward()

    return _bench(fwd, bwd)


def _bench_flex_attn_tl(impl_fn):
    q, k, v = _make_inputs_bhld()

    def fwd():
        out, _ = impl_fn(
            q, k, v,
            window_size=WINDOW_SIZE,
            chunk_size=CHUNK_SIZE,
            training=True,
            mask_lmk=True,
            expand_to_chunk=False,
        )
        return out.float().sum()

    def bwd(loss):
        for t in (q, k, v):
            if t.grad is not None:
                t.grad = None
        loss.backward()

    return _bench(fwd, bwd)


def bench_flex_attn_tl():
    return _bench_flex_attn_tl(flex_attn_tl)


def bench_flex_attn_tl_two_phase():
    return _bench_flex_attn_tl(flex_attn_tl_two_phase)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    assert torch.cuda.is_available(), "CUDA is required for this benchmark."
    torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 78)
    print(
        f"Shape: B={BATCH}, L={SEQ_LEN}, Hq={H_Q}, Hkv={H_KV}, D={HEAD_DIM}, "
        f"dtype={DTYPE}"
    )
    print(
        f"flex_attn_tl: window_size={WINDOW_SIZE}, chunk_size={CHUNK_SIZE}, "
        f"mask_lmk=True, expand_to_chunk=False"
    )
    print(f"warmup={WARMUP}, iters={ITERS}  (all times are medians, in ms)")
    print("=" * 78)

    print("\n[FA3] flash_attn_func(causal=True) ...")
    fa3_fwd, fa3_bwd, fa3_total = bench_fa3()
    print(f"  fwd  : {fa3_fwd:8.3f} ms")
    print(f"  bwd  : {fa3_bwd:8.3f} ms")
    print(f"  total: {fa3_total:8.3f} ms")

    print("\n[GQA-base atomic] example_gqa_bwd.attention(causal=True, atomic) ...")
    gqa_a_fwd, gqa_a_bwd, gqa_a_total = bench_gqa_baseline(use_atomic=True)
    print(f"  fwd  : {gqa_a_fwd:8.3f} ms")
    print(f"  bwd  : {gqa_a_bwd:8.3f} ms")
    print(f"  total: {gqa_a_total:8.3f} ms")

    print("\n[GQA-base split] example_gqa_bwd.attention(causal=True, split) ...")
    gqa_s_fwd, gqa_s_bwd, gqa_s_total = bench_gqa_baseline(use_atomic=False)
    print(f"  fwd  : {gqa_s_fwd:8.3f} ms")
    print(f"  bwd  : {gqa_s_bwd:8.3f} ms")
    print(f"  total: {gqa_s_total:8.3f} ms")

    print("\n[Flex-TL] flex_attn_tl(mask_lmk=True) [atomic bwd] ...")
    tl_fwd, tl_bwd, tl_total = bench_flex_attn_tl()
    print(f"  fwd  : {tl_fwd:8.3f} ms")
    print(f"  bwd  : {tl_bwd:8.3f} ms")
    print(f"  total: {tl_total:8.3f} ms")

    print("\n[Flex-TL-2P] flex_attn_tl_two_phase(mask_lmk=True) [two-phase bwd] ...")
    tp_fwd, tp_bwd, tp_total = bench_flex_attn_tl_two_phase()
    print(f"  fwd  : {tp_fwd:8.3f} ms")
    print(f"  bwd  : {tp_bwd:8.3f} ms")
    print(f"  total: {tp_total:8.3f} ms")

    print("\n" + "-" * 78)
    print("Slowdown ratio vs FA3 (>1 means slower than FA3):")
    print(f"  fwd   GQA-A: {gqa_a_fwd / fa3_fwd:6.2f} x   GQA-S: {gqa_s_fwd / fa3_fwd:6.2f} x   "
          f"TL: {tl_fwd / fa3_fwd:6.2f} x   TL-2P: {tp_fwd / fa3_fwd:6.2f} x")
    print(f"  bwd   GQA-A: {gqa_a_bwd / fa3_bwd:6.2f} x   GQA-S: {gqa_s_bwd / fa3_bwd:6.2f} x   "
          f"TL: {tl_bwd / fa3_bwd:6.2f} x   TL-2P: {tp_bwd / fa3_bwd:6.2f} x")
    print(f"  total GQA-A: {gqa_a_total / fa3_total:6.2f} x   GQA-S: {gqa_s_total / fa3_total:6.2f} x   "
          f"TL: {tl_total / fa3_total:6.2f} x   TL-2P: {tp_total / fa3_total:6.2f} x")
    print("\nTwo-phase speedup over original Flex-TL (>1 means TL-2P faster):")
    print(f"  fwd  : {tl_fwd / tp_fwd:6.2f} x")
    print(f"  bwd  : {tl_bwd / tp_bwd:6.2f} x")
    print(f"  total: {tl_total / tp_total:6.2f} x")
    print("=" * 78)


if __name__ == "__main__":
    main()


# pkill -f "burner.*--gpu 7"; python unittests/bench_fa3_vs_flex_attn_tl.py