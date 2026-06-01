"""Landmark Lmk-K Pool for the per-q-head HSA selection path.

The high-GQA HSA variants (qwen_lhsa with hsa_qk_ratio > 1) want a
per-q-head landmark key ``lmk_k`` for each completed chunk — softmax-weighted
aggregation of the chunk's K values, queried by the chunk-end ``lmk_q``,
shape ``(h_q, head_dim)``. sglang's default selector reads the
last-token-K from the paged KV cache at shape ``(h_kv, head_dim)``, which
matches the official ``compact_lmk_k=True`` mode but NOT the default
``chunk_attn_pool`` MHA-style mode.

This module provides the storage that lets sglang match the default mode:

    * ``LandmarkLmkKPool``: per-layer, per-chunk tensor ``[num_chunk_slots,
      h_q, head_dim]`` holding the pre-aggregated ``lmk_k``.
    * ``ReqToChunkPool``: maps ``(req_idx, chunk_idx)`` -> ``chunk_slot``,
      mirroring the existing ``ReqToTokenPool`` layout.

The pool is "A" in the design discussion: at chunk completion (during
prefill or a decode step that finishes a chunk), the HSA layer computes
``chunk_attn_pool(mu_q, K_chunk)`` once and writes the result. The
selector at decode just gathers — no inline compute, no recompute across
decode steps. Storage is identical to caching the raw ``lmk_q``, the
trade is one-time work at chunk completion vs N×decode-step compute.

Lifecycle is tied to the KV cache: slots are alloc'd as chunks complete
and freed when the request is freed. The pool is sized for the worst-case
``max_chunks = max_total_num_tokens // chunk_size``.
"""
from __future__ import annotations

from typing import Optional

import torch


class LandmarkLmkKPool:
    """Per-layer per-chunk landmark-key storage.

    Layout: ``[num_layers, num_chunk_slots, h_q, head_dim]`` (contiguous on
    the layer/slot axes for cheap per-layer gather).

    Slot allocator is a free-list. Allocations come in batches of
    contiguous-but-not-required slots; ``alloc(n)`` returns a 1D int32
    tensor of slot ids.
    """

    def __init__(
        self,
        num_chunk_slots: int,
        num_layers: int,
        h_q: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_chunk_slots = int(num_chunk_slots)
        self.num_layers = int(num_layers)
        self.h_q = int(h_q)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = device

        # Storage. The +1 slot is a "padding" slot (id 0 reserved for "no chunk")
        # so callers can safely gather from indices that may be -1; we map -1 -> 0
        # and zero out the result via mask. We use 1..N for real slots.
        self.pool = torch.zeros(
            (self.num_layers, self.num_chunk_slots + 1, self.h_q, self.head_dim),
            dtype=dtype,
            device=device,
        )

        # Free list: real slot ids 1..num_chunk_slots
        self._free = list(range(1, self.num_chunk_slots + 1))

    # ---- allocation ----

    def available_size(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> Optional[torch.Tensor]:
        """Allocate ``n`` slots. Returns int32 tensor on self.device, or None
        if exhausted. Order of returned ids is allocation order from the free
        list head."""
        if n <= 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        if len(self._free) < n:
            return None
        out = self._free[:n]
        self._free = self._free[n:]
        return torch.tensor(out, dtype=torch.int32, device=self.device)

    def free(self, slots) -> None:
        """Release slots. Accepts a list[int] or 1D tensor of ids. Zero ids
        (the padding slot) are silently ignored."""
        if isinstance(slots, torch.Tensor):
            slots = slots.detach().to("cpu", torch.int64).tolist()
        for s in slots:
            s = int(s)
            if s > 0:
                self._free.append(s)

    # ---- I/O ----

    def set(self, layer_id: int, slots: torch.Tensor, values: torch.Tensor) -> None:
        """Write ``values`` shape ``[n, h_q, head_dim]`` at ``slots`` (int).

        Caller guarantees slots are real (>0) and distinct.
        """
        if slots.numel() == 0:
            return
        idx = slots.to(self.pool.device, torch.int64)
        v = values.to(dtype=self.pool.dtype, device=self.pool.device)
        # pool: [L, N+1, H, D]
        self.pool[int(layer_id)].index_copy_(0, idx, v)

    def get(self, layer_id: int, slots: torch.Tensor) -> torch.Tensor:
        """Gather ``[..., h_q, head_dim]`` from ``slots`` shape ``[...]`` int.

        Slot id 0 returns the zero padding row, which is the right neutral
        element for masked positions.  Negative values (-1 sentinel) are
        clamped to 0.
        """
        idx = slots.clamp(min=0).to(self.pool.device, torch.int64)
        flat = idx.reshape(-1)
        gathered = self.pool[int(layer_id)].index_select(0, flat)  # [n, H, D]
        return gathered.view(*idx.shape, self.h_q, self.head_dim)


class ReqToChunkPool:
    """Maps ``(req_pool_idx, chunk_idx)`` -> ``LandmarkLmkKPool`` slot id.

    Mirrors ``ReqToTokenPool``'s layout: a single ``[num_reqs, max_chunks]``
    int32 table, initialised to 0 (= padding slot / "not yet allocated").

    Allocation happens per-chunk in the HSA layer's prefill path: when chunk
    ``c`` completes for request ``r``, the layer asks ``LandmarkLmkKPool``
    for a slot, writes the aggregated ``lmk_k``, and records the slot here.
    Free happens when the request is released by the scheduler.
    """

    def __init__(self, num_reqs: int, max_chunks_per_req: int, device: torch.device) -> None:
        self.num_reqs = int(num_reqs)
        self.max_chunks = int(max_chunks_per_req)
        self.device = device
        self.req_to_chunk_slot = torch.zeros(
            (self.num_reqs, self.max_chunks), dtype=torch.int32, device=device
        )

    def assign(self, req_idx: torch.Tensor, chunk_idx: torch.Tensor, slots: torch.Tensor) -> None:
        """Record ``slots[i]`` at ``(req_idx[i], chunk_idx[i])``."""
        if slots.numel() == 0:
            return
        r = req_idx.to(self.device, torch.int64).view(-1)
        c = chunk_idx.to(self.device, torch.int64).view(-1)
        s = slots.to(self.device, torch.int32).view(-1)
        self.req_to_chunk_slot[r, c] = s

    def gather_slots(self, req_idx: torch.Tensor, chunk_ids: torch.Tensor) -> torch.Tensor:
        """For batch ``req_idx`` shape ``[B]`` and ``chunk_ids`` shape ``[B, C]``
        (-1 sentinel for padded entries), return slot ids shape ``[B, C]``.
        Out-of-range chunks return 0 (padding slot)."""
        r = req_idx.to(self.device, torch.int64).view(-1, 1).expand(-1, chunk_ids.shape[1])
        c = chunk_ids.to(self.device, torch.int64).clamp(min=0)
        slots = self.req_to_chunk_slot[r, c]
        # Mask out padded chunks (chunk_id < 0) -> 0
        valid = (chunk_ids >= 0).to(slots.dtype)
        return slots * valid

    def free(self, req_idx, lmk_k_pool: "LandmarkLmkKPool") -> None:
        """Release all chunk slots held by request(s) ``req_idx``."""
        if isinstance(req_idx, torch.Tensor):
            req_list = req_idx.detach().to("cpu", torch.int64).tolist()
        else:
            req_list = [int(req_idx)]
        for r in req_list:
            row = self.req_to_chunk_slot[r]
            nonzero = row[row != 0]
            if nonzero.numel() > 0:
                lmk_k_pool.free(nonzero)
            row.zero_()
