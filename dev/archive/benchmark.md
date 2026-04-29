# HSA Extend/Decode Performance

Benchmarks comparing HSA (Hierarchical Sparse Attention) against vanilla dense attention (TritonAttnBackend / SWA) in sglang.

All numbers measured on a single GPU with `torch.cuda.synchronize()` around timed iterations. The "reference" row is the old Python-loop implementation that was replaced by the optimized batched/Triton path.

---

## Extend (Prefill)

### Config: B=4, prefix=256, extend=64, page_size=64, HQ=32, H=8, D=128, topk=4, window=64

| Method | Time (ms) | vs Dense |
|--------|-----------|----------|
| Dense (vanilla SWA) extend | 0.30 | 1.0x |
| **HSA extend (optimized)** | **2.93** | **0.10x** |
| HSA extend (reference loops) | 779.0 | 0.0004x |

**Sub-component breakdown (optimized vs reference):**

| Component | Optimized (ms) | Reference (ms) | Speedup |
|-----------|---------------|----------------|---------|
| Internal SWA (batched bmm) | 1.03 | 195.9 | 190x |
| TopK Selection (batched) | 0.61 | 121.3 | 198x |
| Sparse Attention (Triton kernel) | 0.16 | 459.1 | 2876x |

### Config: B=4, prefix=4096, extend=128, page_size=64, HQ=32, H=8, D=128, topk=8, window=128

| Method | Time (ms) | vs Dense |
|--------|-----------|----------|
| Dense (vanilla SWA) extend | 0.32 | 1.0x |
| **HSA extend (optimized)** | **3.89** | **0.08x** |
| HSA extend (reference loops) | 2951.2 | 0.0001x |

**Sub-component breakdown:**

| Component | Optimized (ms) | Reference (ms) | Speedup |
|-----------|---------------|----------------|---------|
| Internal SWA | 1.53 | 397.3 | 260x |
| TopK Selection | 0.68 | 220.5 | 323x |
| Sparse Attention | 0.58 | 2330.0 | 4028x |

### Config: B=8, prefix=1024, extend=64, page_size=64, HQ=32, H=8, D=128, topk=4, window=64

| Method | Time (ms) | vs Dense |
|--------|-----------|----------|
| Dense (vanilla SWA) extend | 0.29 | 1.0x |
| **HSA extend (optimized)** | **3.21** | **0.09x** |
| HSA extend (reference loops) | 1799.9 | 0.0002x |

**Sub-component breakdown:**

| Component | Optimized (ms) | Reference (ms) | Speedup |
|-----------|---------------|----------------|---------|
| Internal SWA | 1.16 | 389.3 | 336x |
| TopK Selection | 0.61 | 219.0 | 358x |
| Sparse Attention | 0.31 | 1187.0 | 3818x |

---

## Decode

### Config: B=4, seq_len=320 (prefix=256+extend=64), page_size=64, HQ=32, H=8, D=128, topk=4, window=64

| Method | Time (ms) | vs Dense |
|--------|-----------|----------|
| Dense (vanilla SWA) decode | 0.33 | 1.0x |
| HSA decode | 7.42 | 0.04x |

### Config: B=4, seq_len=4224 (prefix=4096+extend=128), page_size=64, HQ=32, H=8, D=128, topk=8, window=128

| Method | Time (ms) | vs Dense |
|--------|-----------|----------|
| Dense (vanilla SWA) decode | 0.34 | 1.0x |
| HSA decode | 14.04 | 0.02x |

### Config: B=8, seq_len=1088 (prefix=1024+extend=64), page_size=64, HQ=32, H=8, D=128, topk=4, window=64

| Method | Time (ms) | vs Dense |
|--------|-----------|----------|
| Dense (vanilla SWA) decode | 0.33 | 1.0x |
| HSA decode | 26.92 | 0.01x |

---

## Analysis

### Optimized vs reference extend

The optimized extend path is **200-750x faster** than the old Python-loop reference, depending on configuration:

| Optimization | Technique | Typical speedup |
|-------------|-----------|-----------------|
| Sparse attention | Reuse decode Triton kernel with `token_to_seq_id` indirection | 2000-4000x |
| TopK selection | Vectorized candidate building + single batched `select_topk_pages_decode` call | 200-360x |
| Internal SWA | Padded window positions + batched `torch.bmm` over all tokens | 190-340x |

The sparse attention Triton kernel shows the largest speedup because it replaces a triple-nested Python loop (tokens x pages x kv_heads) with a single fused GPU kernel.

### HSA vs dense attention

HSA extend is currently ~10-12x slower than vanilla dense (SWA) attention at these context lengths. This is expected because HSA does strictly more work per layer:

1. Dense extend on all heads (same cost as vanilla SWA)
2. Internal SWA on HSA heads (chunk-aligned windowed attention)
3. TopK page selection (LMK key scoring + top-k)
4. Paged sparse attention over selected pages
5. Merged softmax + weighted fusion

The trade-off is that HSA's sparse attention scales **sub-linearly** with context length (only attending to top-k pages regardless of total length), while dense SWA scales linearly. At sufficiently long contexts (tens of thousands of tokens), HSA should become faster than dense for the KV-retrieval portion.

### Decode bottleneck

HSA decode is 20-80x slower than dense decode. The overhead comes from:
- Per-request page selection (candidate building + scoring + top-k)
- flex_attention for SWA heads (not yet optimized with a fused kernel)
- Merged softmax computation

The decode path has more room for optimization (e.g., fused selection kernel, CUDA-graph support for the full HSA pipeline).

---

## How to reproduce

```bash
# Short context
python python/sglang/test/attention/hsa/bench_hsa_vs_dense_extend_decode.py \
  --B 4 --prefix 256 --extend 64 --page-size 64 \
  --hq 32 --h 8 --d 128 --topk 4 --window 64

# Long context
python python/sglang/test/attention/hsa/bench_hsa_vs_dense_extend_decode.py \
  --B 4 --prefix 4096 --extend 128 --page-size 64 \
  --hq 32 --h 8 --d 128 --topk 8 --window 128

# Large batch
python python/sglang/test/attention/hsa/bench_hsa_vs_dense_extend_decode.py \
  --B 8 --prefix 1024 --extend 64 --page-size 64 \
  --hq 32 --h 8 --d 128 --topk 4 --window 64
```

See `dev/run_instruction.md` for full benchmark CLI options.
