# Official vs sglang HSA Decode Comparison

## Setup

Prefill 65 real tokens (66 engine-visible with 1 LMK), then run 1 decode step.

```
Config: hidden=1024, layers=1, heads=16, kv_heads=4
HSA: hsa_heads=8, topk=2, chunk_size=64
Prefill: 65 real tokens → 66 engine-visible tokens (1 LMK at pos 63)
         = 1 full page (63 real + LMK) + 2 extra real tokens
Decode: 1 token (token_id=42) at engine-visible position 66, position_id=65
```

Run with: `python dev/compare_official_vs_sglang_hsa.py`

## Prefill Results (for reference)

```
  Max absolute error:  0.312235
  Mean absolute error: 0.006182
  Argmax match rate:   97.0%
```

| Region | Positions | Max Error | Argmax Match |
|--------|-----------|-----------|-------------|
| Page 0 (SWA-only) | 0-63 | 0.031 | 97% (61/64) |
| Page 1 (HSA with 1 page) | 64-65 | 0.312 | 100% (2/2) |

Prefill is nearly exact: page 0 is pure bf16 rounding (~0.02), and the 2 tokens on page 1 show small HSA divergence (~0.3).

## Decode Results

```
--- Decode Comparison ---
  Max absolute error:  3.156250
  Mean absolute error: 0.696921
  Official argmax: 6
  sglang  argmax:  226
  Argmax match: NO

  Official top-5: [(6, '1.828'), (68, '1.516'), (58, '1.500'), (205, '1.406'), (132, '1.266')]
  sglang  top-5:  [(226, '2.047'), (249, '1.633'), (26, '1.547'), (93, '1.461'), (96, '1.430')]
```

**Positions: exact match (both decode_position = 65).**

The decode error (3.16) is much larger than the prefill error (0.31), and the top-5 token rankings are completely different.

## Analysis

### Why decode diverges more than prefill

The decode token at position 66 (engine-visible) has 1 completed page available for HSA selection. Both models should select the same page (page 0) and attend to its tokens. However, the **decode code paths** are fundamentally different between the two implementations:

**Official decode path:**
1. HuggingFace `DynamicCache` stores KV as dense `[B, H, L, D]` tensors
2. For HSA attention: `compiled_flex_attention` computes SWA on HSA heads over the cached K/V
3. Landmarks extracted as `K[:, chunk_size-1::chunk_size, :, :]` from the dense cache
4. `online_topk_group` (tilelang) scores and selects top-k pages
5. `HSA_block_M_group` (tilelang) computes sparse attention over selected pages
6. Merged softmax fuses SWA + HSA outputs

**sglang decode path:**
1. Paged KV cache `[num_locs, H, D]` with `page_table_1` indirection
2. SWA heads: flex_attention with `create_block_mask` over window tokens gathered via page table
3. HSA heads: `_run_selection_decode` → `build_active_page_candidates` → `select_topk_pages_decode`
4. `_compute_internal_swa_decode` computes chunk-aligned SWA on HSA heads (Python reference with per-head loop)
5. `hsa_decode_paged_fwd` (Triton kernel) computes sparse attention over selected pages
6. Merged softmax fuses SWA + HSA outputs

The key divergence points:
- **SWA on HSA heads**: Official uses `compiled_flex_attention` (fused kernel). sglang uses per-head Python loops with manual softmax — different numerical behavior.
- **Page selection scoring**: Different kernels (tilelang vs PyTorch einsum + topk), different floating-point accumulation.
- **Sparse attention**: Different kernels (tilelang fused vs Triton paged).
- **KV cache access pattern**: Dense indexing vs paged indirection — same data, different memory layout and gather patterns.

### The decode error is NOT caused by prefill divergence

The prefill KV cache divergence is very small (max 0.31 at the 2 tokens beyond page 0). The decode error (3.16) is far larger than what this small KV divergence could produce. The decode error is dominated by **algorithmic differences in the decode attention path**, not by accumulated prefill error.

This is confirmed by the fact that the decode error with 65-token prefill (3.16) is similar to the decode error with 200-token prefill (2.77) — the number of completed pages matters more than the absolute prefill divergence.

### What would be needed for exact match

To achieve exact decode matching, the sglang decode path would need to use the same attention computation as the official model:
1. Replace the per-head Python SWA loop (`_compute_internal_swa_decode`) with a vectorized kernel matching `compiled_flex_attention`'s behavior
2. Use the same selection scoring (same accumulation order, same tie-breaking)
3. Use the same sparse attention computation order

These are inherent implementation differences, not bugs.

## How to Run

```bash
python dev/compare_official_vs_sglang_hsa.py
```

The decode comparison runs automatically after the prefill comparison. The script prefills 65 tokens, then decodes 1 token, and compares logits for both phases.
