# HSA KV Cache Design: SWA Head Eviction

## Problem

In HSA with InnerX split-head architecture, each attention layer has two head groups:

- **SWA heads** (e.g., 3/4 of KV heads): sliding window attention, only need the last `window_size` tokens
- **HSA heads** (e.g., 1/4 of KV heads): sparse page-based attention, need all pages for landmark retrieval

Currently, sglang stores KV for all heads at every token location in a single pool `[num_locations, H_total, D]`. SWA head KV outside the window is never read but never freed — wasting `(H_swa / H_total) * (context_len - window_size) * head_dim * 2` bytes per request.

For a typical config (H_swa=3, H_hsa=1, window=512, context=32K, D=128, bf16), this waste is ~36MB per request — significant at scale.

## Existing SWA Infrastructure in sglang

sglang already has a complete dual-pool system for models with per-layer SWA/full-attention separation (e.g., Gemma3):

### Architecture

```
SWAKVPool (swa_memory_pool.py)
├── full_kv_pool: MHATokenToKVPool    — stores KV for full-attention layers
├── swa_kv_pool: MHATokenToKVPool     — stores KV for sliding-window layers (smaller)
└── layers_mapping: {layer_id → (pool_index, is_swa_layer)}

SWATokenToKVPoolAllocator (swa_memory_pool.py)
├── full_attn_allocator               — allocates token slots in full pool
├── swa_attn_allocator                — allocates token slots in SWA pool (bounded)
└── full_to_swa_index_mapping         — maps full pool indices to SWA pool indices

SWARadixCache (swa_radix_cache.py)
├── full_lru_list                     — LRU for full-attention eviction
├── swa_lru_list                      — LRU for SWA-only eviction
└── TreeNode with dual lock refs:
    ├── full_lock_ref                 — locked while any request uses this node
    └── swa_lock_ref                  — locked only for last window_size tokens
```

### How eviction works

**Locking (when a request starts using cached KV):**

`inc_lock_ref(node)` walks from the leaf (most recent tokens) toward the root (oldest):
- Increments `full_lock_ref` for **every** node along the path
- Increments `swa_lock_ref` only until accumulated size reaches `sliding_window_size`, then stops

This means tokens older than the window have `swa_lock_ref == 0` immediately — they are evictable in the SWA pool without any explicit cleanup step.

**Eviction (when scheduler needs memory):**

`evict(full_num_tokens, swa_num_tokens)` runs in two phases:

1. **Phase 1 (full LRU):** Find nodes with `full_lock_ref == 0`, free from BOTH pools, delete node from tree
2. **Phase 2 (SWA-only):** Find nodes with `swa_lock_ref == 0`:
   - Internal nodes: call `free_swa()` to reclaim SWA pool slots, mark `swa_tombstone = True` (full-attention KV survives)
   - Leaf nodes: free from both pools

This is **lazy eviction under memory pressure** — old SWA KV stays in memory until the scheduler needs the slots, then they're the first to be reclaimed.

**Runtime buffer extraction:**

During the forward pass, `update_sliding_window_buffer()` in `triton_backend.py` builds `window_kv_indices` containing only the last `window_size` tokens. For SWA layers, indices are translated to SWA pool locations via `translate_loc_from_full_to_swa()`.

## Incompatibility with HSA

The existing system splits by **layer**: each layer is either full-attention or SWA, and reads from one pool or the other.

HSA splits by **head within a layer**: heads 0..H_swa-1 are SWA, heads H_swa..H_total-1 are HSA. Both head groups exist in the same layer and currently share the same KV buffer `[num_locs, H_total, D]`.

```
Existing (per-layer):                    HSA (per-head within layer):
┌─────────────────────────┐              ┌─────────────────────────┐
│ Layer 0: Full Attn      │ → full_pool  │ Layer 0: [SWA│HSA] heads│ → both pools
│ Layer 1: SWA            │ → swa_pool   │ Layer 1: [SWA│HSA] heads│ → both pools
│ Layer 2: Full Attn      │ → full_pool  │ ...                     │
└─────────────────────────┘              └─────────────────────────┘
```

The key API mismatch: `set_kv_buffer(layer, loc, k, v)` writes all heads to one pool based on layer_id. HSA needs to split K/V by head dimension and write SWA heads to the SWA pool and HSA heads to the full pool.

## Proposed Design: HSASplitHeadKVPool

A new KV pool variant that splits by head within each layer, reusing the existing dual-pool and radix cache infrastructure.

### KV Pool

```python
class HSASplitHeadKVPool(KVCache):
    """KV cache that stores SWA and HSA heads in separate pools per layer."""

    def __init__(self, size, size_swa, page_size, dtype,
                 h_swa, h_hsa, head_dim, layer_num, device, ...):
        # SWA pool: bounded, only stores SWA KV heads
        self.swa_kv_pool = MHATokenToKVPool(
            size=size_swa, head_num=h_swa, head_dim=head_dim,
            layer_num=layer_num, ...
        )
        # HSA pool: full context, only stores HSA KV heads
        self.hsa_kv_pool = MHATokenToKVPool(
            size=size, head_num=h_hsa, head_dim=head_dim,
            layer_num=layer_num, ...
        )
        self.h_swa = h_swa
        self.h_hsa = h_hsa
        self.full_to_swa_index_mapping = None  # set by allocator

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, ...):
        """Split K/V by head dim and write to respective pools."""
        swa_k = cache_k[:, :self.h_swa, :]
        hsa_k = cache_k[:, self.h_swa:, :]
        swa_v = cache_v[:, :self.h_swa, :]
        hsa_v = cache_v[:, self.h_swa:, :]

        # HSA heads → full pool (all tokens kept)
        self.hsa_kv_pool.set_kv_buffer(None, loc, hsa_k, hsa_v, ...,
                                        layer_id_override=layer.layer_id)

        # SWA heads → SWA pool (bounded, uses translated locations)
        swa_loc = self.translate_loc_from_full_to_swa(loc)
        self.swa_kv_pool.set_kv_buffer(None, swa_loc, swa_k, swa_v, ...,
                                        layer_id_override=layer.layer_id)

    def get_key_buffer(self, layer_id):
        """Return full key buffer (HSA heads only) for paged sparse attention."""
        return self.hsa_kv_pool.get_key_buffer(layer_id)

    def get_swa_key_buffer(self, layer_id):
        """Return SWA key buffer for sliding window attention."""
        return self.swa_kv_pool.get_key_buffer(layer_id)

    # Similar for get_value_buffer / get_swa_value_buffer
```

### Allocator

Reuse `SWATokenToKVPoolAllocator` directly:
- `full_attn_allocator` manages HSA pool slots (all pages)
- `swa_attn_allocator` manages SWA pool slots (bounded by `B * window_size`)
- `full_to_swa_index_mapping` maps between them

### Radix Cache

Reuse `SWARadixCache` directly:
- Dual lock refs work the same way — `swa_lock_ref` only locks last `window_size` tokens
- SWA tombstone eviction frees SWA pool slots while keeping HSA pool slots alive
- No changes needed to the eviction algorithm

### Attention Backend Changes

The HSA attention backend (`hsa_backend.py`) needs to read from the correct pool:

```python
def forward_decode(self, q, k, v, layer, forward_batch, ...):
    pool = forward_batch.token_to_kv_pool

    # SWA heads: read from SWA pool (bounded window)
    k_cache_swa = pool.get_swa_key_buffer(layer.layer_id)  # [num_swa_locs, H_swa, D]
    v_cache_swa = pool.get_swa_value_buffer(layer.layer_id)

    # HSA heads: read from HSA pool (all pages)
    k_cache_hsa = pool.get_key_buffer(layer.layer_id)      # [num_full_locs, H_hsa, D]
    v_cache_hsa = pool.get_value_buffer(layer.layer_id)

    # SWA branch: use window_kv_indices (translated to SWA pool locs)
    # HSA branch: use page_table_1 (full pool locs, all pages available)
```

For extend, the dense backend also needs awareness of the split pool for the SWA portion.

### Memory Savings

| Component | Current | With split pool |
|-----------|---------|-----------------|
| SWA heads KV | `context_len * H_swa * D * 2` | `window_size * H_swa * D * 2` |
| HSA heads KV | `context_len * H_hsa * D * 2` | `context_len * H_hsa * D * 2` |
| **Total per request** | `context_len * H_total * D * 2` | `window_size * H_swa * D * 2 + context_len * H_hsa * D * 2` |

For a 3/4 SWA + 1/4 HSA split with window=512, context=32K:
- Current: `32K * 4 * D * 2` = 32K entries per head group
- Proposed: `512 * 3 * D * 2 + 32K * 1 * D * 2` = 1.5K + 8K = 9.5K entries per head group
- **~3.4x memory reduction**

## Implementation Plan

### Phase 1: HSASplitHeadKVPool (core data structure)
- New file `python/sglang/srt/mem_cache/hsa_memory_pool.py`
- Implements `HSASplitHeadKVPool` with `set_kv_buffer` / `get_key_buffer` / `get_swa_key_buffer`
- Unit tests comparing split vs unified pool correctness

### Phase 2: Allocator + cache integration
- Wire `SWATokenToKVPoolAllocator` for HSA models
- Wire `SWARadixCache` for HSA models (set `is_hybrid_swa = True` in scheduler)
- Compute `size_swa` from `B * window_size` budget

### Phase 3: Backend integration
- Modify `HSAAttnBackend` to read SWA/HSA heads from different pools
- Modify dense backend extend to use SWA pool for SWA heads
- Update `init_forward_metadata` to build separate KV index arrays

### Phase 4: Testing
- Correctness: verify outputs match single-pool implementation
- Memory: measure actual pool sizes vs single-pool baseline
- Eviction: stress test with many concurrent long-context requests

## Open Questions

1. **Non-HSA layers in HSA models**: If `full_attn_interleave > 1`, some layers are regular full attention (not split-head). These layers need all heads in the full pool. The `layers_mapping` approach from `SWAKVPool` could handle this: non-HSA layers map entirely to the full pool.

2. **Page table mapping**: The current `page_table_1` maps token positions to full-pool locations. SWA reads need translation via `full_to_swa_index_mapping`. The `update_sliding_window_buffer` already does this for per-layer SWA — same mechanism works here.

3. **Prefix cache with split pools**: When reusing a cached prefix, both SWA and HSA pool slots must be valid. If SWA slots were evicted (tombstoned), the SWA KV must be recomputed during extend. The existing `swa_tombstone` mechanism handles this — extend must check and re-fill SWA KV for tombstoned nodes.
