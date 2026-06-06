# HSA Speed Test — B200 Baseline + How to Reproduce

This doc captures the canonical HSA vs Dense bench setup, the B200 baseline
numbers measured on `hpc-rack-2-*`, and the exact commands to reproduce on a
different host (e.g. H200, H100, H20).

Use it as a recipe — every command is meant to be runnable as-is.

---

## 0. Environment

| | Value |
|---|---|
| Branch | `hsa_dev` |
| Python | `/home/hal-alex/miniconda3/envs/alexsg/bin/python` (env `alexsg`) |
| Working dir | `/home/hal-alex/workspace/sglang` |
| JIT toolchain | tilelang + triton 3.5.1; activate hook + bin symlinks already wired up |

Run all commands from the repo root.

---

## 1. Models

Both are 345M-class fair architectures (same `hidden=1024`, `layers=16`, `heads=16q+2kv`, `head_dim=64`, `mlp=4096`). HSA adds `~2.1M` parameters from `lmk_q_proj` LoRA + extra norms (architectural cost of sparse attention).

| Model | Path | Weights | Notes |
|---|---|---|---|
| HSA-345M | `/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real` | Real (trained) | `model_type: qwen_lhsa`, `hsa_topk=32`, `chunk_size=64`, `hsa_sliding_window=512`, `enable_lmk_q_proj=true`, `enable_prior_query=true` |
| Dense-Fair-345M | `/home/hal-alex/workspace/sglang/dev/bench_models/dense345m_fair` | Dummy (perf only) | `model_type: qwen3`, no sliding window |

**Note**: HSA safetensors on disk is 446 MB (stores `lm_head` separately even though `tie_word_embeddings=true`). sglang ties at load time → effective 344M params on GPU, same as Dense.

For perf bench, both use `--load-format dummy` so weight values don't matter.

### 1.1 Repo layout & weights

Configs and tokenizer files for both models live in `dev/bench_models/` and are tracked by git. **Weight files are gitignored** (`*.safetensors`, `*.bin`, `*.pth`, `*.pt` under `dev/bench_models/`) so the 853 MB HSA checkpoint doesn't bloat the repo.

When cloning fresh on a new host:

* `dev/bench_models/dense345m_fair/` — no weights needed; bench uses `--load-format dummy`. Ready to use as-is.
* `dev/bench_models/hsa345m_real/` — bench also runs `--load-format dummy` (random weights), so for perf bench alone you don't need the real checkpoint either. Numerical correctness work (alignment KL via `dev/align/compare.py`) uses a separate harness at `dev/align/weights_345m/`, not this dir.

So for **perf bench on a new host**, no weight fetching needed.

---

## 2. Canonical bench setup

* `--batch-size 1 --tp 1` (single-stream, single-rank — matches the paper's micro-benchmark setup)
* `--cuda-graph-max-bs 1` (production decode path)
* `--mem-fraction-static 0.50` (leaves room for activations at long context; do NOT raise to 0.80+ on H20/H100 — KV pool over-allocation can starve activations and trigger OOM that looks like a slowdown)
* `--input-len $L --output-len 32` (32 decode steps after prefill — enough to get a stable decode median)
* `--context-length $((L + 200))` (200-token safety pad)
* `--load-format dummy` (no weight loading — pure perf)
* `--trust-remote-code` (required by the HSA model class)

**HSA-specific**: `--attention-backend hsa`. Optionally `--no-hsa-headwise-topk-softmax` to enable the max-pooling prefill topk (see §6).

**Dense baseline (apples-to-apples)**: `--attention-backend triton`. Without this, Dense defaults to `sgl-kernel`'s native C++ paged_attn which beats anything triton-based — not a fair comparison vs HSA's triton kernels.

---

## 3. Reading bench output

`sglang.bench_one_batch` runs TWO measurement rounds and prints `Prefill. latency:` and `Decode.  median latency:` lines twice:

```
Prefill. latency: 1.95143 s, ...    ← FIRST: cold, includes JIT compile (ignore)
Decode 0...
Decode.  median latency: 0.03439 s  ← FIRST: cold decode median (ignore)
Prefill. latency: 0.08863 s, ...    ← SECOND: warm prefill (this is the number)
Decode 0...
Decode.  median latency: 0.01732 s  ← SECOND: warm decode median (this is the number)
```

**Always take the SECOND occurrence.** The first is cold + JIT and can be 10-20x slower; treating it as the "real" number is the #1 cause of bogus "HSA is 10x slower" reports.

The scripted sweeps in this dir use `awk 'NR==2{print}'` to skip the cold run.

---

## 4. Quick start (one-liner)

CG path, default mode, full sweep, 3 runs each:

```bash
LENGTHS="8192 16384 32768" N_RUNS=3 ./dev/bench_regression_check_cg.sh
```

Output is one row per (L, backend, run). Pipe to the aggregator:

```bash
LENGTHS="8192 16384 32768" N_RUNS=3 ./dev/bench_regression_check_cg.sh \
  | python dev/_parse_bench_sweep.py
```

Prints median/best/worst per (L, backend), plus HSA/Dense ratios. Takes about 15-20 min for the 3-length × 2-backend × 3-run config.

---

## 5. B200 baseline (this branch, commit `e8b96139e`)

Measured on `hpc-rack-2-12/-14` (GB200), 3 runs each, CG enabled.

### 5.1 Decode (production path)

| L | HSA median | HSA best | Dense median | Dense best | HSA/Dense |
|---|---|---|---|---|---|
| 8K | 2.54 ms | 2.50 ms | 2.15 ms | 2.11 ms | 1.18x |
| 16K | **2.90 ms** | 2.84 ms | **2.86 ms** | 2.80 ms | **1.01x (crossover)** |
| 32K | **2.52 ms** | 2.52 ms | 4.34 ms | 4.32 ms | **0.58x (HSA wins 1.72x)** |

### 5.2 Prefill

| L | HSA median | Dense median | HSA/Dense |
|---|---|---|---|
| 8K | 57.0 ms | 22.8 ms | 2.50x |
| 16K | 88.2 ms | 64.7 ms | 1.36x |
| 32K | **166.8 ms** | **215.8 ms** | **0.77x (HSA wins)** |

Pattern: HSA loses on short context (selector + sparse-attention fixed overhead dominates), wins on long context (sparse attention scales sub-linearly). Crossover is `~16K decode` / `~32K prefill`.

### 5.3 Longer context (disable-CG sweep, prefill numbers — CG only affects decode)

| L | HSA prefill | Dense prefill | HSA/Dense |
|---|---|---|---|
| 32K | 167 ms | 216 ms | 0.78x |
| 64K | 402 ms | 788 ms | **0.51x (HSA 1.96x faster)** |

Beyond 32K Dense gets crushed (full attention is O(L^2)) while HSA stays near-linear in selector cost. The R21-R35 changelog has decode numbers out to 512K showing the same pattern.

---

## 6. The `--no-hsa-headwise-topk-softmax` flag

`commit e8b96139e` added a max-pooling prefill topk path that skips the expensive `hsa_lse` reduction and uses `swa_lse` directly as the per-query normalizer.

* **Default (no flag)**: slow PyTorch `fp32` einsum + `max(dim=G)` + `topk` chain in `_run_selection_extend_batched`. On B200/H100 with full fp32 tensor-core utilization, this runs OK; on H20/weak-fp32 hardware it's the prefill bottleneck.
* **`--no-hsa-headwise-topk-softmax`** (or set `headwise_topk_softmax: false` in model config): routes the per-q-head path through the fused tilelang kernel `online_topk_head` from `dev/hsa-kernel-main/ops/topk_head_maxpool.py`. Upstream measurements showed `~2x faster prefill topk forward` on H100.

### 6.1 Correctness already verified

On B200 at L=8K/16K/32K, output is **bit-identical** between default and maxpool modes (`max_abs_diff=0, KL=0, top-5 5/5 match`). See `dev/test_headwise_topk_softmax_perf.py` and the commit message.

### 6.2 Perf spot-check on B200 (default vs maxpool, 3 runs)

| L | default prefill | maxpool prefill | Delta | default decode | maxpool decode | Delta |
|---|---|---|---|---|---|---|
| 8K | 55.3 ms | 54.4 ms | -1.6% | 2.46 ms | 2.42 ms | -1.6% |
| 16K | 86.4 ms | 85.4 ms | -1.2% | 2.82 ms | 2.81 ms | -0.4% |
| 32K | 166.4 ms | 165.1 ms | -0.8% | 2.53 ms | 2.48 ms | -2.0% |

6/6 metrics are slightly faster with maxpool, but all within run-to-run noise. On B200, fp32 SIMT isn't the bottleneck so the gain is small.

**Expected on H20**: maxpool prefill should be noticeably faster than default. Run `./dev/bench_maxpool_spotcheck.sh` and report numbers.

---

## 7. Full reproducible scripts

| Script | What it does |
|---|---|
| `dev/bench_regression_check_cg.sh` | CG path (production). Sweeps `LENGTHS` × `{hsa, triton}` × `N_RUNS`. Use this to compare with B200 §5 numbers. |
| `dev/bench_regression_check.sh` | Same but with `--disable-cuda-graph`. Useful for breaking down kernel vs Python overhead. |
| `dev/bench_maxpool_spotcheck.sh` | HSA only, default vs `--no-hsa-headwise-topk-softmax`. Tests §6 on the local hardware. |
| `dev/_parse_bench_sweep.py` | Aggregates any of the above into median/best/worst tables. |

All accept `LENGTHS`, `N_RUNS`, `DECODE_LEN` env vars.

Example: full apples-to-apples on a new host:

```bash
LENGTHS="8192 16384 32768" N_RUNS=5 ./dev/bench_regression_check_cg.sh \
  | tee /tmp/sweep_$(hostname).txt \
  | python dev/_parse_bench_sweep.py
```

---

## 8. Comparing across hardware (B200 -> H200 / H100 / H20)

Different hardware will land at different absolute numbers but the **HSA/Dense ratio at each L should be roughly preserved** because both backends use the same triton kernel infrastructure (just different attention algorithms). If the ratio diverges far from B200's, it's a software issue, not a hardware one.

### 8.1 Expected hardware scaling (rough, bf16 tensor core)

| GPU | bf16 TC (TFLOPS) | Relative to B200 | Expected HSA 8K prefill |
|---|---|---|---|
| B200 | ~2200 | 1.0x | 57 ms (measured) |
| H100 | ~989 | 0.45x | ~100-150 ms (estimate) |
| H200 | ~989 | 0.45x | ~100-150 ms (estimate) |
| H20 | ~148 | 0.067x | ~700-900 ms (estimate) |

H100/H200/H20 absolute numbers will all be slower than B200, but **the HSA/Dense ratio** (e.g. `HSA 2.5x slower than Dense at 8K`, `crossover at 32K`) should hold.

### 8.2 Reporting back

When running on a new host, capture:

1. `nvidia-smi --query-gpu=name,memory.total --format=csv | head -2` (hardware ID)
2. `git rev-parse HEAD` (which commit was used)
3. The full sweep output (so we can re-aggregate if needed)
4. The aggregated table from `_parse_bench_sweep.py`
5. If running maxpool: the spot-check table too

---

## 9. Debug checklist if numbers look way off

In rough order of frequency:

1. **Are you reading the SECOND `Prefill. latency:` line?** If you see HSA 10x slower than Dense at 8K, you're probably reading the cold (JIT-included) prefill. The 1.95s vs 0.09s gap is 22x.

2. **Did you pass `--attention-backend triton` for Dense?** Without it, Dense defaults to `sgl-kernel` (C++ native) and crushes HSA — but that's not the comparison we want.

3. **Is `--cuda-graph-max-bs 1` set?** CG enables the production decode path. Without it, all measurements include ~12-14ms Python orchestration overhead per decode step.

4. **Mem-fraction sanity**: at 16K with `--mem-fraction-static 0.80`, KV pool sizing can be too aggressive and force kernels into a slow workspace-starved path. Keep it at 0.50 for these benches.

5. **Is the right backend running?** HSA on a CG-enabled run used to silently fall back to dense (R15.2 silent fallback bug). Verify with `dev/probe_hsa_cg_kernels.py` — should see `hsa_decode_paged_fwd_kernel` calls and ZERO `_fwd_grouped_kernel_stage1` calls.

6. **At long context (>=128K), `--mem-fraction-static 0.50` might not give enough KV pool**. Add `--max-total-tokens 2000000` to explicitly cap pool size and free up activation memory. (See `dev/hsa_optimization_log.md` for the 512K OOM history.)

7. **If `headwise_topk_softmax=False` (maxpool) and you see NaN**: probably the SWA branch's `_last_swa_lse_hq_extend` wasn't populated before selection ran. Should not happen in normal forward order, but worth checking — fall back to default (drop the `--no-hsa-headwise-topk-softmax` flag) to verify.

---

## 10. Background context (for human readers)

For the long story of how we got here:

* `dev/hsa_optimization_log.md` — R1 through R35 changelog. Every kernel fusion, every measured improvement.
* `dev/InfiniteLongLM/` — the upstream training repo + transformers eager model (used as alignment ground truth via `dev/align/compare.py`).
* `dev/hsa-kernel-main/ops/` — production triton/tilelang kernels.

For correctness validation:

* `dev/align/compare.py` — runs sglang HSA + official transformers HSA on the same input, reports KL per layer + final logit. Use this to confirm a code change didn't break numerics. R35 baseline: prefill KL ~3e-5, decode KL ~4e-3.
* `dev/test_long_ctx_cg_correctness.py` — CG vs eager bit-comparison at long context.
* `dev/test_decode_greedy_vs_official.py` — token-by-token greedy decode comparison.
