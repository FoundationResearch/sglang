from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HSAMetadata:
    """
    Minimal HSA metadata scaffold.

    This is intentionally small and aligned with SGLang's paged KV semantics.
    As HSA implementation progresses, we can extend this dataclass similarly to
    `NSAMetadata` (see `python/sglang/srt/layers/attention/nsa_backend.py`).
    """

    page_size: int

    # token-level KV length per sequence (int32)
    cache_seqlens_int32: torch.Tensor
    # max KV length in this batch (token-level)
    max_seqlen_k: int

    # token->slot mapping table (page_size=1 semantics; token_loc in global pool)
    page_table_1: torch.Tensor
    # page_id table (page_size>1); for page_size==1, this may alias page_table_1
    real_page_table: torch.Tensor

    # Paged attention indices used by dense backends (decode/extend)
    kv_indptr: Optional[torch.Tensor] = None
    kv_indices: Optional[torch.Tensor] = None

    # Sliding-window variants (optional; only when enabled)
    window_kv_indptr: Optional[torch.Tensor] = None
    window_kv_indices: Optional[torch.Tensor] = None

    # ---- Step 4+ (selection) optional debug fields (decode path) ----
    # These are populated by HSAAttnBackend for observability and unit tests.
    hsa_cand_page_ids: Optional[torch.Tensor] = None  # [B, C] int32 padded -1
    hsa_cand_mask: Optional[torch.Tensor] = None  # [B, C] bool
    hsa_selected_page_ids: Optional[torch.Tensor] = None  # [B, H, K] int32 padded -1
    hsa_selected_scores: Optional[torch.Tensor] = None  # [B, H, K] float32


