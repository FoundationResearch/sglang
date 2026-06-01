# Long-context latency sweep — HSA vs Dense (sglang, GB200, bs=1)

Latency comparison across context-length buckets, single GB200, batch_size=1,
decode_len=128 (matching the H20 paper table's settings).  Each cell shows the
median benchmark latency from `sglang.bench_one_batch`.

## Setup

- Single GB200 (188 GB HBM)
- sglang HSA branch (`hsa_dev` @ commit `c8914f1a4`, after rounds R0–R5 of
  decode/prefill speed optimizations)
- `--cuda-graph-max-bs 1 --mem-fraction-static 0.85` (single-bs cuda graph
  capture, full memory pool)
- `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` so Qwen-style configs can be
  benched beyond their native context

Scripts: `dev/bench_long_context_345m_sweep.sh` (345M) and
`dev/bench_long_context_sweep.sh` (7B).

## 345M apples-to-apples: HSA vs Dense-Fair (same arch)

Both models: 16 hidden layers, hidden_size=1024, head_dim=64, num_q_heads=16,
num_kv_heads=2 (8:1 GQA).  Only the attention path differs — HSA uses sparse
chunk selection (`qwen_lhsa`), Dense uses full attention (`qwen3`).  Built
`/home/hal-alex/workspace/dense345m_fair` by stripping HSA-specific fields
from the HSA config.

```
Length   HSA Pf(ms)  Dense Pf(ms)  Pf ratio   HSA Dc(ms)  Dense Dc(ms)  Dc ratio
                                  (Dense/HSA; <1 means HSA slower)
-----------------------------------------------------------------------------
  8K        16.75        14.48      0.86x       1.76         1.58       0.90x
 16K        34.83        24.48      0.70x       1.85         1.57       0.85x
 32K        84.68        64.09      0.76x       1.91         1.59       0.83x
 64K       239.35       198.60      0.83x       1.93         1.63       0.84x
128K       773.87       692.50      0.89x       1.99         1.76       0.88x
256K      2743.44      2582.81      0.94x       2.15         1.90       0.88x
512K       (HSA bench hangs in warmup; needs investigation — see below)
```

**Headline**: in apples-to-apples on GB200, sglang HSA-345M is currently
**slower than same-arch dense across 8K-256K** by 6-30%.  The Pf gap is
narrowing with length (0.70× → 0.94×), suggesting HSA could win at >512K
once the constant overhead is amortized.  Dc gap is steadier (~0.85-0.90×).

The paper's table (H20, naive Full Attention baseline) shows HSA winning Pf
crossover at 32K and Dc crossover at 128K.  We don't reproduce those wins on
GB200 vs sglang Dense, likely because:
1. sglang Dense uses FlashInfer (very optimized), the paper's "Full Attention"
   baseline is a less-tuned reference implementation
2. 345M with 8:1 GQA has a tiny KV cache (~8KB/tok), so Dense isn't
   bandwidth-bound until VERY long context

## 345M misleading prior comparison (DO NOT cite)

The first sweep used `dev/InfiniteLongLM/configs/full_attn_tiny/config.json` as
the "baseline", which had 24L / 4KV — i.e. **50% more layers and 2× more KV
heads** than HSA.  That comparison made HSA look 1.66× faster prefill / 1.73×
faster decode at 256K, but that's mostly architecture mismatch, not the HSA
sparse attention.  Kept here only as a record of the bug.

```
Length    HSA Pf(ms)  24L-Base Pf(ms)  pseudo-ratio   HSA Dc(ms)  24L-Base Dc(ms)  pseudo-ratio
   8K         16.92          18.73        "1.11x"        1.82          1.98          "1.09x"
  16K         34.78          37.65        "1.08x"        1.80          2.09          "1.16x"
  32K         84.84          98.56        "1.16x"        1.95          2.25          "1.15x"
  64K        239.39         322.39        "1.35x"        1.95          2.43          "1.25x"
 128K        773.56        1178.31        "1.52x"        1.97          2.84          "1.44x"
 256K       2742.99        4546.09        "1.66x"        2.14          3.70          "1.73x"
```

## 7B (dummy weights) — earlier sweep, kept for record

7B HSA dummy at GB200 hits OOM beyond 128K (the dummy config uses MHA dense
layers — KV cache scales 32 layers × 32 kv heads).  Reported numbers up to
128K with full cuda-graph; 256K used `--disable-cuda-graph` so the decode
numbers are dominated by Python overhead.

```
Length   HSA-7B Pf(ms)  Qwen2.5-7B Pf(ms)  Pf ratio   HSA-7B Dc(ms)  Qwen-7B Dc(ms)  Dc ratio
   8K          88.88              90.60       1.02x         4.40            4.18      0.95x
  16K         183.68             196.49       1.07x         4.53            4.35      0.96x
  32K         394.44             453.32       1.15x         4.87            4.48      0.92x
  64K         927.48            1183.65       1.28x         5.40            4.68      0.87x
 128K        2495.91            3380.08       1.35x         6.48            5.19      0.80x
 256K        7403.89†          11434.95†        —*           *               *         —
 512K         OOM/hung           OOM/hung        —           —               —         —
```
* 256K used `--disable-cuda-graph` so decode latency includes large Python
overhead per step and is not directly comparable to the cuda-graph buckets.
The Qwen2.5-7B baseline also has a different architecture (different layer
count, GQA ratio); cf. 345M section for the apples-to-apples picture.

## 512K HSA: hangs in warmup

The 345M HSA-345M bench at 512K consistently sits in the bench `Warmup ...`
phase for >2 min with GPU at 0% utilization.  The main Python thread is in
the `do_wait` syscall (waiting for a child) while CPU sits at 20-80%.  The
KV pool allocates 20M token slots (155 GB) before warmup begins — that part
is fine.  Possible causes (not yet narrowed down):
- tilelang HSA forward kernel JIT compilation for the new seq_len=524288
  shape (the kernel is shape-specialised)
- O(N) Python loop in the HSA selector / metadata builder at 8192 chunks

Next step: re-run with `TILELANG_*` env vars to disable autotune, or trace
with py-spy.
