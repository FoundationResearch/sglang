# HSA Full Support Roadmap

## Current Status Summary

### What's Working (with test coverage)
- **InnerX ultra split-head decode** — SWA + HSA branches, selection, paged kernel, o_proj stitching
- **HSA backend** (`--attention-backend hsa`) — registered, metadata init, selection dispatch
- **LMK runtime injection** — every `page_size-1` tokens, LMK inserted, KV cached, output discarded
- **Paged HSA decode kernel** (Triton) — per-page block attention with LMK masking
- **Top-K page selection** — head/group/softmax_head strategies, LMK-K as page repr
- **Radix cache with LMK** — page-aligned prefix reuse, partial page protection
- **Continuous batching** — scheduler-level E2E verified

### What's Missing / Not Verified
1. **Prefill/extend uses dense fallback** — not true HSA semantics
2. **No SWA↔HSA merged softmax** during prefill (the key innovation in reference)
3. **SWA decode branch** uses Python per-request loops, not batched kernels
4. **No chunk_align semantics** for SWA window
5. **No CUDA graph** native HSA support (delegates to dense)
6. **No speculative decoding / overlap scheduling** verification
7. **Multi-layer / large-config** edge cases not systematically tested

---

## Gap Analysis: Reference Model vs SGLang Implementation

### Reference `HierarchicalSparseAttention.forward()` (prefill path)

The reference model's HSA layer during **prefill** does this:

```
1. SWA branch: q/k/v → qk_norm → RoPE → sliding_window_attention → o_upper
2. HSA branch: hsa_q/k/v → qk_norm (NO RoPE)
3. SWA-within-HSA: flex_attention(hsa_q, hsa_k, hsa_v, window, chunk_size) → swa_o, lse_sum
4. LMK-K extraction: lmk_k = hsa_k_norm[:, chunk_size-1::chunk_size, :, :]
5. Selection: topk_func(lmk_q, lmk_k, topk, ...) → indices, scores
6. Merged softmax: cat_scores = [chunk_scores, lse_sum] → softmax → chunk_weights
7. Block-sparse HSA: HSA_func(q, k, v, weights=chunk_weights, indices=indices, mask_last_token=True) → hsa_o
8. SWA weight extraction: swa_weight = chunk_weights[:, :, :, -1]
9. Fusion: o_lower = hsa_o + swa_o * swa_weight
10. Concat: o = cat([o_upper, o_lower], dim=head)
```

**Key insight**: Steps 3+6+7+9 form the "LHSA" pattern from `current_design.md`:
- `exp_sum` = the logsumexp from the SWA-within-HSA branch
- HSA fuses with SWA via globally-normalized softmax over `[chunk_scores | exp_sum]`
- This is NOT just "run SWA and HSA separately then concat" — the HSA branch *internally* runs SWA and merges its weight with chunk selection weights

### What SGLang decode already does correctly

The decode path correctly implements steps 1-10 in spirit:
- SWA branch: flex_attention or torch fallback with window
- HSA branch: selection → paged kernel → weighted combination
- But the SWA↔HSA weight fusion uses `softmax(selected_scores)` only — it does NOT incorporate SWA's logsumexp into the HSA branch weights

### `current_design.md` LHSA Pseudocode

```
head lhsa(swa_len, need_retrieval=False, pos_emb=rope, chunk_align_=False):
    exp_sum = SWA(swa_len, pos_emb, chunk_align)
    HSA(effective_chunk(skip swa chunks), exp_sum → weight fusion)

every layer:
    a * lhsa(2048, False, rope, False)   # SWA branch
  + b * lhsa(2048, True, nope, True)     # HSA branch (retrieval, nope, chunk-aligned)
```

This means each LHSA call internally does:
1. Run SWA (sliding window attention) → get output + logsumexp
2. Run HSA with chunk selection → but chunks covered by SWA are excluded, and SWA's exp_sum participates in the softmax normalization

The `chunk_align=True` means: if `swa_len=16, chunk_size=64, current_len=70`, then `effective_chunks = ceil(70/64) = 2`, so `effective_swa_len = 128` (aligned to chunk boundary).

---

## Phase 1: Prefill/Extend HSA Semantics (Critical Path)

**Goal**: Replace dense fallback with true HSA semantics during prefill/extend. This closes the full lifecycle (prefill → decode).

### P1.1: HSA Prefill Kernel (Torch Reference First)

Implement `forward_extend` in `hsa_backend.py` with true HSA semantics:

```
For each HSA layer during extend:
  1. Save KV as usual
  2. SWA-within-HSA: compute attention within sliding window, return output + logsumexp
  3. Extract LMK-K from completed pages
  4. Selection: topk over LMK-K representations
  5. Merged softmax: combine [chunk_scores | swa_logsumexp]
  6. Block-sparse attention over selected chunks
  7. Weighted fusion: o = hsa_output + swa_output * swa_weight
```

Start with a **torch reference** implementation (no Triton kernel needed yet), like we did for decode.

**Key decisions**:
- For extend, the query length > 1. Selection must be done per-query-position (or per-chunk-of-queries)
- The reference uses `compiled_flex_attention` with a block mask that: (a) is causal, (b) applies sliding window, (c) excludes LMK positions from attention
- Selection uses `online_topk_group` which handles multi-query prefill natively

### P1.2: Extend Selection for Multi-Token Queries

During prefill, selection is per-query-token (or per-query-chunk):
- Each query token at position `t` can only see completed chunks before `t`
- Chunk visibility is causal: query at position 70 with chunk_size=64 can see chunks 0 (positions 0-62) but NOT chunk 1 (which includes position 70 itself)
- This is different from decode where selection is just per-request

Implementation approach:
- For correctness-first: loop over query positions, call decode-style selection per position
- For performance: batch all queries and use a causal mask on the chunk dimension

### P1.3: SWA↔HSA Merged Softmax (LHSA Fusion)

The reference model's critical semantic: SWA's logsumexp and HSA's chunk scores are combined before softmax:

```python
cat_scores = torch.cat([chunk_scores, lse_sum.unsqueeze(-1)], dim=-1)
chunk_weights = F.softmax(cat_scores, dim=-1)
```

This means:
- Higher SWA logsumexp → more weight on local context → less weight on retrieved chunks
- This is a **globally normalized** weighting, not independent softmax on each part

For decode, this means we should also get the SWA branch's logsumexp and fold it into the HSA selection weights. Current decode does NOT do this — it runs `softmax(selected_scores)` independently.

**TODO**: Decide if decode should also adopt merged softmax, or if current decode semantics are acceptable (they are mathematically different but may produce similar results in practice).

### P1.4: Tests

- `test_hsa_backend_extend_innerx_gpu.py`: torch reference comparison for extend path
- Verify: prefill with 1 chunk (< chunk_size) → pure SWA, no HSA selection
- Verify: prefill with 3 chunks → selection over completed chunks, causal
- Verify: decode after prefill → weights are consistent

### Estimated Files to Modify
- `python/sglang/srt/layers/attention/hsa_backend.py` — `forward_extend` rewrite
- `python/sglang/srt/layers/attention/hsa/kernels/` — possible new extend kernel
- `python/sglang/srt/layers/attention/hsa/selector.py` — multi-token selection support

---

## Phase 2: Decode Path Improvements

### P2.1: Merged Softmax in Decode

Align decode with reference semantics:
- After SWA branch computation, extract logsumexp
- Combine with HSA chunk selection scores before softmax
- This requires `flex_attention` to return lse, or compute it manually

Currently decode's SWA branch returns `out_swa` but not `lse_sum`. Need to propagate this.

### P2.2: chunk_align Semantics

Implement `chunk_align=True` logic:
- When enabled, SWA window snaps to chunk boundaries
- `effective_swa_len = ceil(current_pos / chunk_size) * chunk_size` if that covers ≤ allowed chunks
- Affects which chunks are excluded from HSA selection candidates

### P2.3: Batched SWA Kernel for Decode

Current SWA decode branch uses Python loops per request:
```python
for b in range(B):
    # per-request flex_attention or torch fallback
```

Replace with a batched kernel (batched flex_attention or custom Triton sliding-window decode):
- This is a **performance-only** change, no semantic difference
- Can reuse existing Triton `decode_attention` with window constraints

### P2.4: enable_softmax1 Support

The reference model's "softmax off by one" adds a zero logit:
```python
cat_scores = torch.cat([scores, lse_sum, zeros], dim=-1)
```
This biases the model toward lower total attention weight when context is uninformative. Low priority but needed for full compatibility.

---

## Phase 3: System-Level Integration

### P3.1: CUDA Graph Support

Current: all CUDA graph ops delegate to dense backend.

Plan:
- HSA selection + paged kernel need to be CUDA-graph-compatible
- Main challenge: dynamic `selected_page_ids` shape varies per request
- Solution: pre-allocate max-size buffers (topk * max_batch), mask unused entries
- Follow NSA's pattern for CUDA graph capture/replay

### P3.2: Speculative Decoding Verification

- Verify that spec draft + verify paths handle LMK injection correctly
- Spec tokens may cross page boundaries → need to trigger LMK injection during spec decode
- The verify step must account for LMK positions in the acceptance check

### P3.3: Overlap Scheduling

- Verify that prefill/decode overlap doesn't corrupt HSA metadata
- Selection metadata is per-forward-pass, should be isolated per forward call
- KV cache writes from overlapping prefill must not invalidate in-flight decode's selected pages

### P3.4: Multi-Round Prefilling (from design doc)

"考虑多轮prefilling 观察是否支持lhsa的prefilling"

Multi-round prefill = chunked prefill where a long prompt is split into multiple extend calls:
- Each extend call processes a chunk of the prompt
- LMK injection must be consistent across chunks (partial pages carry over)
- Radix cache must correctly handle prefix reuse across chunked prefill rounds
- Selection during later chunks must see completed pages from earlier chunks

---

## Phase 4: Performance Optimization

### P4.1: Fused Selection + Transform

Merge top-k selection + page_id transform into a single fused op:
- Currently: compute scores → topk → transform page_ids → gather KV
- Fused: single Triton kernel for score computation + topk + page_id output

### P4.2: HSA Prefill Kernel (Triton)

Replace torch reference prefill with a dedicated Triton kernel:
- Block-sparse attention over selected chunks
- Fused with per-chunk weighting
- Similar to the reference's `HSA_block_M_group` but paged

### P4.3: Profiling & Benchmarks

- Selection overhead profiling (q·E_i + topk)
- Random page access bandwidth analysis
- Long-context benchmarks: dense vs HSA throughput/latency
- Targets: 128K+ context summarization, RAG workloads

---

## Phase 5: Feature Completeness

### P5.1: num_swa_layers

Reference model supports a prefix of pure-SWA layers (`num_swa_layers`) before HSA layers begin. Currently `full_attn_interleave` handles the interleaving pattern, but doesn't support a pure-SWA prefix.

### P5.2: hsa_visible_window (Training Parity)

Limit how far back HSA can select chunks. Training-time feature that constrains the selection window. Not critical for inference but needed for training/fine-tuning support.

### P5.3: enable_stick_breaking

Alternative to softmax for chunk weight computation. Low priority.

### P5.4: Gate Attention Verification

`enable_gate` is implemented but needs more test coverage with real weights.

---

## Priority Order

| Priority | Phase | Task | Effort | Impact |
|----------|-------|------|--------|--------|
| **P0** | 1.1 | Prefill/extend HSA torch reference | Large | Critical — closes lifecycle |
| **P0** | 1.4 | Extend path tests | Medium | Validates P0 |
| **P1** | 1.2 | Multi-token selection for extend | Medium | Required for P0 |
| **P1** | 1.3 | SWA↔HSA merged softmax | Medium | Semantic correctness |
| **P2** | 2.1 | Merged softmax in decode | Small | Alignment with reference |
| **P2** | 2.3 | Batched SWA decode kernel | Medium | Performance |
| **P2** | 3.4 | Multi-round prefilling | Medium | Production readiness |
| **P3** | 3.1 | CUDA graph support | Large | Production readiness |
| **P3** | 2.2 | chunk_align semantics | Small | Feature completeness |
| **P4** | 3.2 | Spec decoding verification | Medium | Advanced feature |
| **P4** | 4.1-4.3 | Performance optimization | Large | Throughput |
| **P5** | 5.x | Feature completeness (swa_layers, stick_breaking, etc.) | Small each | Parity |

---

## Open Questions for Discussion

1. **Merged softmax in decode**: Should we align decode to also use SWA↔HSA merged softmax, or is the current independent-softmax decode acceptable? The mathematical difference is subtle but real — merged softmax gives a globally normalized weighting.

2. **Prefill strategy**: Torch reference first vs. directly write Triton kernel? Torch reference is safer for correctness but adds a performance gap to close later.

3. **chunk_align for SWA branch**: The design doc mentions `chunk_align=True` for the HSA (nope) branch. Should the SWA (rope) branch also support chunk alignment, or is that only for HSA?

4. **Multi-round prefilling**: Any specific constraints from the scheduler/engine that would make this harder? Does the current chunked prefill already handle page boundary crossings with LMK?

5. **hsa_sliding_window vs sliding_window**: The reference model has separate `hsa_sliding_window` and `sliding_window` configs. Currently sglang uses `sliding_window_merging_size` and `sliding_window_attention_size`. Are these 1:1 mappings?
