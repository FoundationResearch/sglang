# HiLS-Attention (HSA) backend for SGLang

HiLS-Attention — a hierarchical sparse attention (HSA) mechanism — is a
landmark-based sparse-attention backend for long-context inference, invoked as
`--attention-backend hsa`. Each key/value sequence is split into fixed-size
**chunks** (pages); a small set of per-chunk **landmark** keys lets every query
select only the top-`k` most relevant chunks and attend to those, plus a local
**sliding window**. Cost grows as `O(N · topk · page_size)` instead of dense
`O(N²)`, so it crosses over dense attention at long context while staying closely
aligned with the reference model.

This is the SGLang serving backend for **HiLS-Attention**. For the method,
training code, and reference implementation see the paper
[arXiv:2607.02980](https://arxiv.org/pdf/2607.02980) and the main repository
[Tencent-Hunyuan/HiLS-Attention](https://github.com/Tencent-Hunyuan/HiLS-Attention).

## Enabling it

### 1. Download the released weights

```bash
huggingface-cli download tencent/HiLS-Attention-7B --local-dir ./HiLS-Attention-7B
```

### 2. Transform the checkpoint for SGLang

The released weights use the upstream `HiLS*` / `hils_*` naming, while this
backend expects the `HSAForCausalLM` / `FlashHSAConfig` schema (`hsa_*` keys).
**The weight tensors are already compatible — only `config.json` needs
translating.** The converter rewrites the config and symlinks the weights
(add `--copy` to make a standalone copy):

```bash
python scripts/convert_hils_checkpoint.py \
    --src ./HiLS-Attention-7B --dst ./HiLS-Attention-7B-sglang
```

It derives the `hsa_*` geometry from the source config
(`hsa_heads`, `hsa_qk_ratio`, `hsa_topk`, `hsa_sliding_window`, `apply_hsa_rope`,
`head_dim`), sets `architectures=["HSAForCausalLM"]` and the matching
`model_type` (`olmo_lhsa` post-norm / `qwen_lhsa` pre-norm), audits that every
weight tensor matches the HiLS-Attention schema, and validates the result through
`FlashHSAConfig`.

### 3. Serve

Launch with the `hsa` attention backend and a page size equal to the model's
chunk size (default **64**):

```bash
python -m sglang.launch_server \
    --model-path ./HiLS-Attention-7B-sglang \
    --attention-backend hsa \
    --page-size 64 \
    --trust-remote-code
```

> **`--page-size` must equal `chunk_size`** (64 for the released checkpoints).
> The default page size of 1 runs but is numerically wrong for HiLS-Attention — the
> landmark chunking assumes one page == one chunk.

### Runtime behavior specific to the HiLS-Attention backend

The backend auto-configures two things at launch (each logs a warning); no flags
are needed:

- **`prefill_max_requests=1`** — the HiLS-Attention prefill (extend) kernels handle
  a single sequence at a time, so prefill batches are capped to one request. **Decode
  stays fully batched**, so concurrent-generation throughput is unaffected.
- **`disable_overlap_schedule=True`** — HiLS-Attention decode interleaves virtual landmark
  (LMK) tokens whose next input is chosen synchronously from the current sequence
  length, which the overlap scheduler cannot support (it launches the next forward
  before that decision). Single-batch latency (the benchmark table above) is
  unaffected; only online-serving decode throughput may drop slightly, since
  per-step CPU overhead is no longer hidden behind GPU compute.

The model config (`config.json`) drives the HiLS-Attention geometry:

| field | meaning |
|---|---|
| `hsa_heads` | number of attention heads routed through HiLS-Attention (== `num_attention_heads` for an all-sparse / `hsa_denom=1` model) |
| `hsa_topk` | chunks selected per query (e.g. 32) |
| `chunk_size` | tokens per chunk/page (64) |
| `hsa_sliding_window` / `sliding_window` | local window width (e.g. 512) |
| `enable_prior_query` | use per-chunk landmark queries + entropy-bias (`prior_b`) selection |
| `enable_lmk_q_proj`, `lmk_q_lora_dim` | landmark-query projection |
| `layerwise_lmkq_norm` | RMSNorm the landmark query over the full projected width instead of per-head |
| `apply_hsa_rope`, `enable_inrange_rope`, `use_hope` | RoPE variants applied to the HiLS-Attention branch |

Both GQA (`G = h_q / h_kv > 1`) and MHA (`G = 1`) prior-query checkpoints are
supported.

## Selection path: fast vs exact

Chunk selection has two implementations, chosen by
`SGLANG_HSA_HEADWISE_TOPK_SOFTMAX` (env) / `headwise_topk_softmax` (config) /
`--hsa-headwise-topk-softmax` (server arg), in that priority order:

- **`0` (default) — max-pool fast path.** Fused tilelang max-over-group top-k.
  Fastest; use for serving. (Note: the max-pool kernel requires `G > 1`; `G = 1`
  models are routed to the softmax-top-k kernel automatically.)
- **`1` — exact softmax-top-k path.** Matches the training/reference selection
  bit-for-bit. Use for consistency/alignment testing.

## Kernels

The tilelang selection/attention kernels live in `dev/hsa-kernel-main/ops/` and
are imported at runtime by `selector.py`. `tilelang` must be installed and its
bundled TVM must import cleanly in your environment.

## Performance (345M, H200, bf16, chunk 64, top-k 32, 512-window, batch 1)

Prefill is warm full-input latency; decode is median per-token with CUDA graphs.
Speedup = full-attention / HiLS-Attention.

| Context | HiLS prefill | Full prefill | Pf speedup | HiLS decode | Full decode | Dc speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8K   | 47 ms   | 29 ms   | 0.62× | 3.87 ms | 2.83 ms | 0.73× |
| 16K  | 81 ms   | 85 ms   | 1.05× | 3.87 ms | 4.21 ms | 1.09× |
| 32K  | 154 ms  | 290 ms  | 1.88× | 3.93 ms | 6.74 ms | 1.72× |
| 64K  | 317 ms  | 1058 ms | 3.34× | 4.08 ms | 12.0 ms | 2.94× |
| 128K | 715 ms  | 4162 ms | 5.82× | 4.35 ms | 22.5 ms | 5.16× |
| 256K | 1796 ms | 16751 ms| 9.33× | 4.71 ms | 43.6 ms | 9.26× |
| 512K | 4972 ms | 66993 ms| 13.5× | 5.47 ms | 85.9 ms | 15.7× |

Parity at ~16K; 13.5× / 15.7× (prefill / decode) faster at 512K. H100 reproduces
these within a few percent. Re-run with `dev/sweep_8k_512k.sh`.
