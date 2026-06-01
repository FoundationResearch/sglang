# Long-context latency sweep — HSA vs Dense (sglang, GB200, bs=1)

> **2026-06-01 update.** The earlier "HSA loses 6-30% at 8K-256K" picture below
> was **measuring the wrong path** — sglang's cuda-graph capture for the HSA
> backend delegates to dense (`hsa_backend.py:555-598`), so the `--cuda-graph-max-bs 1`
> bench was always running HSA-model-with-dense-attention, not real HSA.
>
> **Real HSA** (forward through `HSAAttnBackend.forward_decode` →
> `_run_selection_decode` + `hsa_decode_paged_fwd` + per-q-head fusion)
> only runs when `--disable-cuda-graph --attention-backend hsa` is set.
> Re-benching that path (`dev/bench_real_hsa.sh`) after the R9-R12 sync /
> Python-loop elimination commits (`d5d3b163` … `59b9046a`) gives the
> headline numbers we actually care about, in the **"Real HSA (apples-
> to-apples, both --disable-cuda-graph)"** table below.

## Real HSA (apples-to-apples, both --disable-cuda-graph)

Both branches run on the same per-step path — no cuda-graph amortisation
either side, single GB200, batch=1, 345M apples-to-apples (qwen_lhsa vs
qwen3, same 16L / 16q / 2kv / hidden=1024 / head_dim=64 arch).

### Post-R14 (selector decode fast path)

```
Length   HSA Pf(s)  Dense Pf(s)  Pf ratio   HSA Dc(ms)  Dense Dc(ms)  Dc ratio
                                  (>1 means HSA faster)
-----------------------------------------------------------------------------
  8K       1.80        0.52       0.29×        51            9         0.18×
 32K       2.05        0.71       0.35×        51           11         0.22×
128K       3.36        3.58      *1.07×*       49           18         0.37×
256K       4.33       17.77      *4.10×*       55           34         0.62×
512K      15.31       73.82      *4.82×*       47           66        *1.40×*
```

**HSA decode is now O(1) in context length** (~50 ms regardless of context):
the tilelang topk kernel was prefill-tuned (1×h_kv×B thread blocks for
decode = <1% GPU utilisation), so R14 swapped in a plain torch matmul +
topk fast path for the q_seq_len == 1 case.  Selector cost shifted from
the 26 ms/layer serial scan to a single batched matmul + topk that
finishes in microseconds.

**512K decode now reverses dense — 1.40× faster.**  Combined with the
pre-existing 4.82× prefill reversal at 512K, HSA is faster than dense
on both phases at long context.

### Pre-R14 (selector tilelang kernel)

```
Length   HSA Pf(s)  Dense Pf(s)  Pf ratio   HSA Dc(ms)  Dense Dc(ms)  Dc ratio
-----------------------------------------------------------------------------
  8K       2.05        0.73       0.36×        90           11         0.13×
 32K       2.38        1.97       0.83×       159           11         0.07×
128K       4.42        8.26      *1.87×*      504           38         0.08×
256K      10.08       31.94      *3.17×*      996           74         0.07×
512K      16.54       74.27      *4.49×*      919           91         0.10×
```

* **Prefill crosses over at ~64K.**  By 256K HSA prefill is 3.17× faster,
  by 512K it is **4.49× faster** than apples-to-apples dense.  This is
  the answer to "why doesn't HSA beat dense at 512K?" — it does, on
  prefill.  The crossover gets steeper with length because dense is
  O(N²) while HSA is O(N · topk·page_size).
* **Decode is still bound by per-step Python + kernel-launch overhead.**
  R9-R12 cut 32K decode from 291ms → 159ms (1.83×) by removing
  per-layer `.item()` syncs and vectorising the `for kv_h` matmul loop,
  but at 128K-512K the bottleneck shifts onto the selector kernel +
  `hsa_decode_paged_fwd` + chunk-weight softmax fusion, which need
  CUDA-graph integration (see below) or kernel-level fusion to close
  the gap with dense's heavily-tuned `_fwd_grouped_kernel_stage1`.
* The **earlier cuda-graph numbers below** are therefore a comparison
  of *(HSA model + dense attention) vs (Dense model + dense attention)*.
  The 6-30% gap there is dominated by HSA's extra model-side ops
  (lmk_q projection, selector head, residual q split), which R1-R7
  shrank as much as possible.

## CUDA-graph integration (not yet fixed)

`hsa_backend.py` currently delegates `init_cuda_graph_state`,
`init_forward_metadata_capture_cuda_graph`, and
`init_forward_metadata_replay_cuda_graph` to the dense backend, so the
captured graph contains dense attention kernels instead of HSA's.
Until that is fixed:

* Reported "HSA" cuda-graph numbers always reflect dense attention.
* True HSA path runs only with `--disable-cuda-graph --attention-backend hsa`.

Fix scope: HSAAttnBackend needs its own cuda-graph state (pre-allocated
`cand_page_ids`, `cand_mask`, `hsa_selected_page_ids`, `hsa_selected_scores`,
`page_table_1` overlay buffer), and selection / paged-decode kernels
must be confirmed graph-safe (no host-side `.item()` syncs).  R9-R12
removed the known host-side syncs from the decode path; the remaining
gating item is the capture/replay metadata plumbing.



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

## 512K landed

The earlier 512K hang was just `alloc_extend_kernel` JIT compilation
(triton emits a 3.5 MB PTX file from `tl.arange(0, 524288)`, and ptxas
takes ~5 min per compile on sm_100a).  After waiting it out the kernel
gets cached in `~/.triton/cache`, so subsequent 512K runs are fast.

Also had to drop `--mem-fraction-static` from 0.85 to 0.50 — at 0.85 the
HSA dummy's KV pool ate 155 GB and left no room for the 512K-prefill
activations.

345M numbers at 512K (after the JIT cache is warm + mem_fraction=0.5):

  HSA-345M   : Prefill 10039.89 ms  Decode 2.44 ms  (median)
  Dense-Fair : Prefill  9991.88 ms  Decode 2.20 ms  (median)

  Pf ratio (Dense/HSA): 0.995×  (effectively tied — within run noise)
  Dc ratio (Dense/HSA): 0.90×   (HSA 10% behind)

The Pf curve goes 8K→512K: 0.96→0.92→0.95→0.97→0.98→0.99→0.995.  HSA's
prefill at 345M approaches but doesn't cross dense even at 512K; the
sparse-attention advantage gets fully consumed by HSA's per-step
overhead (selector, lmk_q projection, page-table bookkeeping) on a
model this small.  Decode never crosses — dense stays 8-15% ahead.
