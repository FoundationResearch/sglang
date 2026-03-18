# HSA Test & Benchmark Instructions

## Prerequisites

Activate the sglang dev environment (with CUDA, Triton, and all sglang deps installed):

```bash
conda activate alexsg  # or your env name
```

All commands below assume you are at the repo root (`sglang/`).

---

## Running Tests

### All HSA tests at once

```bash
python -m pytest python/sglang/test/attention/hsa/ -v
```

### By category

**Triton kernel (decode paged attention):**
```bash
python -m pytest python/sglang/test/attention/hsa/test_flashhsa_triton_hsa_decode_paged_kernel_gpu.py -v
```

**HSA backend — decode (InnerX split-head):**
```bash
python -m pytest python/sglang/test/attention/hsa/test_hsa_backend_innerx_split_head_decode_gpu.py -v
```

**HSA backend — extend/prefill (InnerX split-head):**
```bash
python -m pytest python/sglang/test/attention/hsa/test_hsa_backend_innerx_split_head_extend_gpu.py -v
```

**Optimized extend vs reference (per-component + end-to-end):**
```bash
python -m pytest python/sglang/test/attention/hsa/test_hsa_extend_optimized_vs_reference_gpu.py -v
```

**Selector (top-k page selection):**
```bash
python -m pytest python/sglang/test/attention/hsa/test_hsa_selector_decode_gpu.py -v
```

**End-to-end decode (model-runner-like):**
```bash
python -m pytest python/sglang/test/attention/hsa/test_innerx_model_runner_like_end_to_end_decode_gpu.py -v
```

**Radix cache / KV reuse:**
```bash
python -m pytest python/sglang/test/attention/hsa/test_innerx_radix_kv_reuse_end_to_end_gpu.py -v
python -m pytest python/sglang/test/attention/hsa/test_innerx_radix_branching_prefix_reuse_end_to_end_gpu.py -v
python -m pytest python/sglang/test/attention/hsa/test_hsa_radix_cache_lmk_gpu.py -v
```

**Scheduler / continuous batching:**
```bash
python -m pytest python/sglang/test/attention/hsa/test_innerx_scheduler_continuous_batching_e2e_gpu.py -v
```

**LMK injection / position IDs:**
```bash
python -m pytest python/sglang/test/attention/hsa/test_hsa_lmk_runtime_injection_gpu.py -v
python -m pytest python/sglang/test/attention/hsa/test_flashhsa_positions_gpu.py -v
```

### Using the reference (Python-loop) fallback

Set the env var `SGLANG_HSA_EXTEND_REFERENCE=1` to force all extend operations to use the original per-token Python-loop implementations instead of the optimized batched/Triton paths. Useful for debugging:

```bash
SGLANG_HSA_EXTEND_REFERENCE=1 python -m pytest python/sglang/test/attention/hsa/test_hsa_backend_innerx_split_head_extend_gpu.py -v
```

### Verbose output

Some tests support `SGLANG_HSA_TEST_VERBOSE=1` to print per-token error breakdowns:

```bash
SGLANG_HSA_TEST_VERBOSE=1 python -m pytest python/sglang/test/attention/hsa/test_hsa_backend_innerx_split_head_extend_gpu.py -v -s
```

---

## Running Benchmarks

The benchmark script compares HSA extend/decode against vanilla dense (TritonAttnBackend) attention.

### Quick smoke test (small dims, fast)

```bash
python python/sglang/test/attention/hsa/bench_hsa_vs_dense_extend_decode.py \
  --B 2 --prefix 64 --extend 16 --page-size 4 \
  --hq 4 --h 2 --d 16 --topk 2 --window 4
```

### Realistic model-like config

```bash
python python/sglang/test/attention/hsa/bench_hsa_vs_dense_extend_decode.py \
  --B 4 --prefix 256 --extend 64 --page-size 64 \
  --hq 32 --h 8 --d 128 --topk 4 --window 64
```

### Long context

```bash
python python/sglang/test/attention/hsa/bench_hsa_vs_dense_extend_decode.py \
  --B 4 --prefix 4096 --extend 128 --page-size 64 \
  --hq 32 --h 8 --d 128 --topk 8 --window 128
```

### Larger batch

```bash
python python/sglang/test/attention/hsa/bench_hsa_vs_dense_extend_decode.py \
  --B 8 --prefix 1024 --extend 64 --page-size 64 \
  --hq 32 --h 8 --d 128 --topk 4 --window 64
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--B` | 4 | Batch size |
| `--prefix` | 256 | Cached prefix length per sequence |
| `--extend` | 64 | New tokens per sequence (prefill) |
| `--page-size` | 64 | Page/chunk size |
| `--hq` | 32 | Number of query heads |
| `--h` | 8 | Number of KV heads |
| `--d` | 128 | Head dimension |
| `--topk` | 4 | HSA top-k pages to select |
| `--window` | 64 | SWA window size |
| `--warmup` | 5 | Warmup iterations |
| `--iters` | 20 | Timed iterations |
| `--extend-only` | — | Only run extend benchmark |
| `--decode-only` | — | Only run decode benchmark |

### What the benchmark measures

**Extend:**
- Dense (vanilla SWA) `forward_extend` — the baseline
- HSA `forward_extend` (optimized) — production batched/Triton path
- HSA `forward_extend` (reference) — old Python-loop path
- Per-component breakdown: internal SWA, top-k selection, sparse attention (each optimized vs reference)

**Decode:**
- Dense (vanilla SWA) `forward_decode` — the baseline
- HSA `forward_decode` — Triton kernel path

---

## File Map

| File | Purpose |
|------|---------|
| `python/sglang/srt/layers/attention/hsa_backend.py` | Main HSA backend (extend + decode) |
| `python/sglang/srt/layers/attention/hsa/kernels/hsa_decode.py` | Triton kernel for paged sparse attention (decode + extend) |
| `python/sglang/srt/layers/attention/hsa/selector.py` | Top-k page selection logic |
| `python/sglang/srt/layers/attention/hsa/metadata.py` | HSAMetadata dataclass |
| `python/sglang/srt/layers/attention/hsa/utils.py` | Page table utilities |
| `python/sglang/test/attention/hsa/` | All tests and benchmarks |
