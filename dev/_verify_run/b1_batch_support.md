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

## Round 2 — batched decode optimisation

Profiling decode at B=16 / 64K (eager, `bench_one_batch --profile`) against B=1
showed what actually scales with batch. Two fixes, both alignment-preserving:

1. **Pool-direct candidate scoring.** The selector pre-gathered every
   candidate's landmark into `[num_layers, B, C_max, h_q, D]` — **1.08 GB per
   step** at B=16 / 64K — then the score kernel read it back. `_pqh_score_maxpool`
   now indexes the landmark pool directly at `hsa_per_step_slots`, so the copy
   is gone; the same bytes are read once instead of written and read.
   (`hsa_per_step_all_lmk_k` / `all_prior_b` are removed.)
2. **Split-K heuristic.** `hsa_decode_paged_fwd` holds a `[PAGE_SIZE, D]` K and
   V tile per program, making it latency-bound rather than occupancy-bound. The
   old rule bailed to `SPLIT_K=1` as soon as `B*HQ >= num_sms`, which is exactly
   the batched case. Splitting to one page per program wins at every (B, L)
   measured, B=1..128 — see the table in `_pick_split_k`.

Decode GPU time at B=16 / 64K: **7.13 ms → 5.42 → 3.87 ms**;
`hsa_decode_paged_fwd_kernel` 2781 µs → 1152 µs.

### HSA vs dense decode across batch (ms/step, idle H200)

| L | bs=1 | bs=4 | bs=16 | bs=32 |
|---|---|---|---|---|
| 8K  | 3.18 / 2.73 → 0.86× | 3.84 / 2.75 → 0.72× | 5.41 / 2.83 → 0.52× | 7.08 / 3.10 → 0.44× |
| 16K | 3.17 / 4.03 → 1.27× | 3.84 / 4.02 → 1.05× | 5.63 / 4.16 → 0.74× | 7.22 / 4.53 → 0.63× |
| 32K | 3.27 / 6.82 → 2.09× | 4.03 / 6.57 → 1.63× | 5.77 / 6.76 → 1.17× | 7.56 / 7.57 → 1.00× |
| 64K | 3.31 / 11.81 → 3.57× | 4.24 / 11.57 → 2.73× | 5.98 / 12.14 → 2.03× | harness OOM |

(HSA / dense → speedup. Round 1 gave 1.32× at 64K/bs=16 and *lost* at
32K/bs=16; the crossover at bs=16 moved from ~45K down to ~28K.)

Dense decode barely moves with batch (11.8 → 12.1 ms at 64K) because it is
badly under-occupied at bs=1 and absorbs batch almost for free, so HSA's
advantage still erodes with batch — just far less than before. Note this 345M
pair has only 2 KV heads, which understates dense's KV traffic relative to a
real MHA model; the 7B (32 layers, 32 KV heads) should hold up better, but that
is reasoning, not measurement.

Alignment after both changes: 12/12 tokens at 512 and 2048 prompt tokens,
KL 4.22e-4 (unchanged) and 1.06e-4 (was 1.11e-4 — the split-K change reorders
the partial-sum reduction). Batch-vs-serial equivalence still passes.

## Round 3 — decode kernel launch shape

With SPLIT_K at its maximum a `hsa_decode_paged_fwd` program handles a single
page, so the block wants to be as small as possible: more resident blocks beats
wider blocks for work this plentiful and independent. `num_warps` 4 → 1 is a
one-line change worth 1-17%, growing with batch size (measurements in the
`_DECODE_NUM_WARPS` comment). `num_stages` made no difference.

**Tried and rejected: G-fusion of the decode kernel.** The Gh q-heads sharing a
KV head select the same pages, so the kernel reads each page's K/V Gh times —
an 8× redundancy at G=8, and exactly the trick the extend kernel uses (R38/R39).
Implemented it (grid over KV heads, two-phase K-then-V to hold register
pressure, per-g reductions kept bit-identical) and it is a **±2% wash**:
B=1 −2%, B=16 −1%, B=64 +2%, B=128 +2%. The redundant reads were already
being served by L2, and fusing costs exactly as much parallelism as it saves
bandwidth in a kernel that is latency-bound, not bandwidth-bound. Reverted;
not worth a hand-written kernel variant.

### HSA vs dense decode after three rounds (ms/step, idle H200)

| L | bs=1 | bs=4 | bs=16 | bs=32 |
|---|---|---|---|---|
| 8K  | 3.14 / 2.74 → 0.87× | 3.77 / 2.83 → 0.75× | 5.00 / 2.87 → 0.57× | 6.26 / 3.23 → 0.52× |
| 16K | 3.18 / 4.09 → 1.29× | 3.80 / 4.08 → 1.07× | 5.19 / 4.19 → 0.81× | 6.34 / 4.54 → 0.72× |
| 32K | 3.22 / 6.69 → 2.08× | 3.98 / 6.49 → 1.63× | 5.25 / 6.88 → 1.31× | 6.48 / 7.52 → 1.16× |
| 64K | 3.29 / 11.71 → 3.56× | 4.20 / 11.56 → 2.75× | 5.46 / 12.02 → 2.20× | harness OOM |

Batched decode over the three rounds, B=16 / 64K: **9.14 → 5.98 → 5.46 ms**
(1.67× total), i.e. 1.32× → 2.20× against dense. The bs=16 crossover moved from
~45K to ~24K, and bs=32 now wins at 32K where it used to lose (0.75× → 1.16×).

Alignment after round 3: 12/12 tokens at both prompt lengths, KL 4.38e-4
(was 4.22e-4 — the warp count changes the in-block reduction order) and
1.06e-4. Batch-vs-serial equivalence still passes.

## Methodology notes

**Which runs use which weights.** Every latency number here comes from
`bench_one_batch --load-format dummy` on `dev/bench_models/*` (random weights),
matching the methodology of the published batch-1 table. Every correctness
number — alignment, batch-vs-serial equivalence, prefix-cache reuse — uses the
real trained checkpoint `dev/align/ckpt_345m_bench_hf`.

Random weights are a fair proxy for dense attention, where the access pattern
does not depend on the weights, but HSA *selects* which pages to read, so it
deserved a check. `dev/bench_models/hsa345m_real` and
`dev/align/ckpt_345m_bench_hf` have identical geometry (16 layers, 16 q-heads,
2 KV heads, head_dim 64), so they are directly comparable:

| L | bs | dummy | trained |
|---|---|---|---|
| 8K  | 1  | 2.98 ms | 2.90 ms |
| 8K  | 16 | 4.22 ms | 4.21 ms |
| 32K | 1  | 3.01 ms | 2.96 ms |
| 32K | 16 | 4.52 ms | 4.51 ms |

Trained weights are marginally *faster* (≈2% at bs=1, indistinguishable at
bs=16), so the dummy-weight numbers are if anything conservative. Top-k selects
a fixed 32 pages either way, so only the addresses differ, and at batch the
concurrency hides that.

**The 345M pair is HSA's worst case.** Both models are GQA with h_q=16,
h_kv=2 (G=8) over 16 layers, so dense decode reads only 16 KB of KV per token.
The released 7B is MHA — 32 layers × 32 KV heads × head_dim 128 = 512 KB per
token, **32× more**. At 64K that is 1 GB/step for the 345M against ~32 GB/step
for the 7B: the 345M's dense baseline sits ~33× off its bandwidth roofline at
bs=1 and therefore absorbs batch almost for free, which is precisely why HSA's
advantage erodes with batch here.

## MHA vs GQA at the same model size

To isolate the head-count factor, `dev/bench_models/{hsa,dense}345m_mha_hd128`
are the same 345M configs with `num_key_value_heads` 2 → 16 (G=1, the layout
the released 7B uses) and nothing else changed. Decode ms, HSA / dense →
speedup, dummy weights, idle H200:

| L | bs=1 | bs=4 | bs=16 |
|---|---|---|---|
| 8K  | 3.08 / 3.65 → 1.19× | 3.93 / 3.84 → 0.98× | 5.61 / 10.18 → 1.81× |
| 16K | 3.17 / 5.61 → 1.77× | 3.98 / 5.92 → 1.49× | 5.68 / 15.82 → 2.79× |
| 32K | 3.27 / 9.46 → 2.89× | 4.12 / 10.08 → 2.45× | 5.86 / 30.50 → 5.20× |
| 64K | 3.34 / 17.12 → 5.13× | 4.29 / 18.37 → 4.28× | KV capacity, see below |

Against the GQA pair at bs=16 (0.57× / 0.81× / 1.31× / 2.20×) the picture
inverts. HSA's own latency barely moves — 5.00 → 5.61 ms at 8K, 5.25 → 5.86 at
32K, about +10% — because it reads `topk*64 + window` tokens and had ~10× of
bandwidth headroom to absorb the extra heads. Dense is what changes: 2.87 →
10.18 ms at 8K/bs=16, 6.88 → 30.50 at 32K/bs=16. Crucially it stops absorbing
batch for free (32K dense, bs 1/4/16: GQA 6.69/6.49/6.88 flat, MHA
9.46/10.08/30.50), so under MHA HSA's advantage *grows* with batch instead of
shrinking, and it wins at every length tested including 8K.

This isolates only the head count (8×); the released 7B adds 2× the layers on
top, so its curve should be better still. Synthetic config with dummy weights —
no trained MHA 345M exists.

## HSA saves bandwidth, not KV capacity

HSA stores the full KV cache exactly like dense; it only reads less of it per
step. So KV *capacity* limits hit both identically, and MHA hits them 8× sooner
(128 KB/token vs 16 KB/token). At 64K × bs=16 the MHA pair needs 1,048,576
token slots and cannot get them on one H200:

| mem-fraction | token slots | needed |
|---|---|---|
| 0.50 | 552,896 | 1,048,576 |
| 0.85 | 952,320 | 1,048,576 |
| 0.92 | 1,032,192 | 1,048,576 |
| GQA @ 0.50 | 4,430,592 | 1,048,576 |

It misses by 1.6% at mem-fraction 0.92, with nothing left for weights,
activations and the CUDA graph — a hard wall, not a tuning failure (bs=15 fits,
bs=16 does not). Extrapolated to the released 7B at 512 KB/token, 64K × bs=16
would need 512 GB of KV.

So the honest reading of the MHA table is that long context and large batch
cannot both be had on one GPU: the reachable regimes there are long context
with small batch (64K, bs=1/4 → 5.13× / 4.28×) and medium context with large
batch (32K, bs=16 → 5.20×). The 64K × bs=16 cell would need multiple GPUs, KV
offload or KV quantisation, and that comparison would have to be redone.

## Still limited to B == 1 / TP == 1

The landmark writers still return early under `get_attention_tp_size() > 1`
(unchanged, pre-existing). The chunk-selection tilelang kernels still take a
scalar `q_offset`, which is why extend loops per sequence rather than running
one varlen launch; making `topk_head_maxpool` varlen would remove that loop.
