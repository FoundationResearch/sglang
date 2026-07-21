# batch_size > 1 support for the HSA backend — verification record

Base commit: `ad573d9da` (hsa_h200). GPU: H200 (shared with an external job, so
absolute latencies are noisy; all speed numbers below are interleaved A/B on the
same GPU against a `git worktree` of the base commit).
Checkpoint: `dev/align/ckpt_345m_bench_hf` (16 heads / 2 kv → G=8, all-HSA,
prior_query, chunk 64, window 512).

## What was broken

1. **Prefill (extend) asserted `B == 1`** — two hard asserts in
   `hsa_backend.py`; a second concurrent request crashed the scheduler.
2. **Decode ran at B>1 but was silently wrong.**
   `_maybe_write_decode_chunk_lmk_k` early-returned on `batch_size != 1`, so
   chunks completed *during decode* got no landmark. A chunk only becomes
   selectable once it leaves the sliding window, so the corruption first shows
   up ~512 generated tokens later — invisible to every short-generation test.
3. **Landmarks were keyed by `(req_pool_idx, chunk_idx)`**, which is wrong in
   three separate ways: a radix prefix-cache hit found no landmark for the
   reused chunks, a recycled `req_pool_idx` inherited the previous request's
   rows, and `ReqToChunkPool.free` was never called anywhere in
   `python/sglang/srt/` so slots leaked for the process lifetime.

## What changed

* Landmarks are keyed by **KV page id** (`KvPageChunkSlots`). HSA requires
  `page_size == chunk_size`, so chunk *c* of a sequence is exactly one KV page;
  the page id names the chunk's *content*, so prefix-sharing requests share
  landmarks for free, a landmark lives exactly as long as its page, and the
  allocator / free list / 512-chunk decode pre-allocation all disappear.
* The decode-time landmark writer is `[B]`-shaped, and its layer-invariant part
  (boundary test, slot, token locations) is computed once per step instead of
  once per layer.
* Extend steps 3-7 (internal SWA → chunk selection → merged softmax → sparse
  attention) run once per sequence via `_extend_hsa_heads`; steps 1-2 (KV write,
  dense SWA heads) and step 8 stay batched. At B==1 the call is the original
  code path unchanged, which is why alignment is bit-identical.

## Results

### Alignment vs the HF reference (friend's verify trio, CUDA graph ON)

| prompt tokens | token match | top-2 | KL mean | KL max | vs recorded baseline |
|---|---|---|---|---|---|
| 512  | 12/12 (100%) | 12/12 | 4.22e-4 | 7.4e-4 | identical |
| 2048 | 12/12 (100%) | 12/12 | 1.11e-4 | 2.2e-4 | identical |

Reproduce with `dev/reverify_align.sh <PT> <GPU>` (reuses a cached HF pickle;
prime it once with `dev/run_cg_vs_hf_verify.sh`).

### Batched vs serial equivalence (`dev/test_hsa_batch_equivalence.py`)

6 prompts of very different lengths (including one under a single page), B=6
concurrent vs one at a time. All PASS:

| config | short (48 tok) | long (900 tok) |
|---|---|---|
| default | exact | exact |
| `--no-cuda-graph` | exact | agrees 724 tok |
| `--disable-radix-cache` | exact | exact / agrees 745 tok |
| `--prefill-max-requests 1` | exact | exact |
| `--disable-radix-cache --no-cuda-graph` | exact | agrees 724 tok |

**On the long-generation criterion:** batching changes GEMM shapes and hence
floating-point reduction order, so exact agreement across batch compositions is
not achievable. The `--fp-noise-control` run measures the floor: two *equally
correct* batch compositions of the same prompt agree for **731 tokens**. The
observed 724/745 sit at that floor. The bug this test exists to catch diverges
at ~512 (the sliding-window boundary) with structurally broken output, an order
of magnitude earlier and qualitatively different.

`dev/test_hsa_prefix_cache_landmark.py` (cold vs warm run of the same prompt)
passes with the radix cache both on and off.

### Speed (**batch 1 on both sides** — no regression, decode improved)

`bench_one_batch --batch-size 1 --cuda-graph-max-bs 1`, so none of this comes
from batching; it is the same single-sequence workload the published numbers
were measured on.

| | base `ad573d9da` | now | |
|---|---|---|---|
| decode 32K (CG) | 3.85 ms | **3.34 ms** | 1.15× faster |
| prefill 8K (eager) | 48.3 ms | 48.5 ms | within noise |
| prefill 32K (eager) | 166.2 ms | 165.0 ms | within noise |

The decode gain is attributable to one change, confirmed by ablation (same
binary, the per-step cache switched off by env var):

| | decode 32K |
|---|---|
| landmark-writer scratch hoisted to once per step | 3.38 / 3.37 ms |
| same code, hoist disabled | 3.79 / 3.81 ms |
| base commit `ad573d9da` | 3.83 / 3.85 / 3.91 ms |

The writer's boundary test, slot lookup and 63-entry page-table gather depend
only on batch state, so they used to run once per HSA layer (16× for this
all-HSA 345M) and are now computed once per step. Those ops are captured into
the CUDA graph and replayed every step, and decode here is bound by kernel
count rather than FLOPs — the same effect the R75-R77 fusions exploited.
Hoist-disabled matching the base commit also shows the rest of the work
(page-keyed slots, per-sequence extend) is latency-neutral at B=1.

### Speed (what batched prefill buys — `dev/bench_hsa_batched_prefill.py`)

| batch | `prefill_max_requests=1` | batched | speedup |
|---|---|---|---|
| 16 × 128 tok | 0.645 s | 0.408 s | 1.58× |
| 8 × 512 tok | 0.315 s | 0.220 s | 1.43× |
| 4 × 2048 tok | 0.165 s | 0.140 s | 1.18× |

Per-sequence work inside an HSA layer is identical either way, so the win is
amortising the rest of the forward and the scheduler step — largest for short
requests, tapering as a single request gets long enough to saturate the GPU.

## Still limited to B == 1 / TP == 1

The landmark writers still return early under `get_attention_tp_size() > 1`
(unchanged, pre-existing). The chunk-selection tilelang kernels still take a
scalar `q_offset`, which is why extend loops per sequence rather than running
one varlen launch; making `topk_head_maxpool` varlen would remove that loop.
