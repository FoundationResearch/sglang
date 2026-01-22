from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class HSASelectionResult:
    """Selection output for decode (single token per sequence).

    Shapes follow kv-head semantics (H = num_kv_heads, G = num_q_heads / num_kv_heads):
    - cand_page_ids: [B, C] int32, padded with -1
    - cand_mask: [B, C] bool
    - selected_page_ids: [B, H, K] int32, padded with -1
    - selected_scores: [B, H, K] float32 (max-pooled scores)
    """

    cand_page_ids: torch.Tensor
    cand_mask: torch.Tensor
    selected_page_ids: torch.Tensor
    selected_scores: torch.Tensor


def _unique_sorted_int32(x: torch.Tensor) -> torch.Tensor:
    """Unique + sorted for int tensors; returns int32."""
    if x.numel() == 0:
        return x.to(torch.int32)
    x = x.to(torch.int64)
    x, _ = torch.sort(x)
    x = torch.unique_consecutive(x)
    return x.to(torch.int32)


def build_active_page_candidates(
    *,
    page_table_1: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    window_size: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-request active page candidates (padded).

    Candidate pages are derived from token locations in `page_table_1[:, :seq_len]`
    and de-duplicated at page granularity. If window_size is provided (>0), pages
    touched by the last `window_size` tokens are excluded (SWA→HSA mode).

    Returns:
      cand_page_ids: [B, Cmax] int32 padded with -1
      cand_mask: [B, Cmax] bool (True for valid entries)
    """
    if page_table_1.dim() != 2:
        raise ValueError(f"page_table_1 must be 2D, got {page_table_1.shape}")
    B = int(page_table_1.shape[0])
    seq_lens_i64 = seq_lens.to(torch.int64)

    per_req_pages = []
    max_c = 0
    for b in range(B):
        seqlen = int(seq_lens_i64[b].item())
        if seqlen <= 0:
            pages = page_table_1.new_empty((0,), dtype=torch.int32)
        else:
            locs = page_table_1[b, :seqlen].to(torch.int64)
            pages = _unique_sorted_int32(locs // int(page_size))
            if window_size is not None and int(window_size) > 0:
                w = min(seqlen, int(window_size))
                w_locs = page_table_1[b, seqlen - w : seqlen].to(torch.int64)
                w_pages = _unique_sorted_int32(w_locs // int(page_size))
                if w_pages.numel() > 0 and pages.numel() > 0:
                    pages = pages[~torch.isin(pages, w_pages)]

        per_req_pages.append(pages)
        max_c = max(max_c, int(pages.numel()))

    cand_page_ids = page_table_1.new_full((B, max_c), -1, dtype=torch.int32)
    cand_mask = page_table_1.new_zeros((B, max_c), dtype=torch.bool)
    for b, pages in enumerate(per_req_pages):
        if pages.numel() == 0:
            continue
        n = int(pages.numel())
        cand_page_ids[b, :n] = pages
        cand_mask[b, :n] = True

    return cand_page_ids, cand_mask


def select_topk_pages_decode(
    *,
    q: torch.Tensor,
    cand_page_ids: torch.Tensor,
    cand_mask: torch.Tensor,
    cand_chunk_repr: torch.Tensor,
    cand_chunk_repr_valid: torch.Tensor,
    topk: int,
    selection_strategy: str,
    sm_scale: Optional[float] = None,
) -> HSASelectionResult:
    """Compute top-k page selection for decode step.

    Args:
      q: [B, HQ*D] or [B, HQ, D]
      cand_page_ids: [B, C] int32 padded with -1
      cand_mask: [B, C] bool
      cand_chunk_repr: [B, C, H, D] (kv heads)
      cand_chunk_repr_valid: [B, C] bool, validity under page_version guard
      topk: fixed K
      selection_strategy: "group" or "head" (mirrors hsa-kernel reference naming)
      sm_scale: optional scaling; default 1/sqrt(D)
    """
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if cand_page_ids.shape[:2] != cand_mask.shape[:2]:
        raise ValueError("cand_page_ids and cand_mask must have same [B,C] shape")
    if cand_page_ids.shape[:2] != cand_chunk_repr.shape[:2]:
        raise ValueError("cand_page_ids and cand_chunk_repr must align on [B,C]")

    B, C = cand_page_ids.shape
    _, _, H, D = cand_chunk_repr.shape

    if q.dim() == 2:
        HQD = int(q.shape[1])
        if HQD % D != 0:
            raise ValueError(f"q last dim {HQD} must be multiple of D={D}")
        HQ = HQD // D
        q_ = q.view(B, HQ, D)
    elif q.dim() == 3:
        q_ = q
        HQ = int(q_.shape[1])
    else:
        raise ValueError(f"q must be [B, HQ*D] or [B, HQ, D], got {q.shape}")

    if HQ % H != 0:
        raise ValueError(f"HQ={HQ} must be divisible by H={H}")
    G = HQ // H
    q_hgd = q_.view(B, H, G, D)

    if sm_scale is None:
        sm_scale = float(D) ** -0.5

    # Mask invalid candidates.
    valid_cand = cand_mask & cand_chunk_repr_valid
    # For scoring, treat invalid as -inf so they never win topk.
    neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=torch.float32)

    # scores_max: [B, H, C] float32
    if selection_strategy == "group":
        q_group_sum = q_hgd.sum(dim=2)  # [B, H, D]
        # [B,H,C]
        scores_max = torch.einsum("bhd,bchd->bhc", q_group_sum.float(), cand_chunk_repr.float())
        scores_max = scores_max * float(sm_scale)
    elif selection_strategy == "head":
        # scores_all: [B,H,G,C]
        scores_all = torch.einsum("bhgd,bchd->bhgc", q_hgd.float(), cand_chunk_repr.float())
        scores_max = scores_all.max(dim=2).values * float(sm_scale)
    else:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

    if C == 0:
        selected_page_ids = cand_page_ids.new_full((B, H, topk), -1, dtype=torch.int32)
        selected_scores = q.new_full((B, H, topk), float("-inf"), dtype=torch.float32)
        return HSASelectionResult(
            cand_page_ids=cand_page_ids,
            cand_mask=cand_mask,
            selected_page_ids=selected_page_ids,
            selected_scores=selected_scores,
        )

    scores_max = scores_max.masked_fill(~valid_cand[:, None, :], neg_inf)

    # torch.topk requires k <= C
    k_eff = min(int(topk), int(C))
    top_scores, top_idx = torch.topk(scores_max, k=k_eff, dim=-1, sorted=False)  # [B,H,k_eff]

    # Sort indices for determinism (matches hsa-kernel ref).
    top_idx_sorted, order = torch.sort(top_idx, dim=-1)
    top_scores_sorted = torch.gather(top_scores, -1, order)

    # Map from candidate-index space to page_ids.
    gathered_page_ids = torch.gather(
        cand_page_ids[:, None, :].expand(B, H, C),
        dim=-1,
        index=top_idx_sorted.to(torch.int64),
    ).to(torch.int32)

    # Any entry with -inf score becomes invalid (-1 page id).
    gathered_page_ids = gathered_page_ids.masked_fill(
        top_scores_sorted == neg_inf, -1
    )

    # Pad to fixed K if needed.
    if k_eff < topk:
        pad_n = topk - k_eff
        pad_pages = cand_page_ids.new_full((B, H, pad_n), -1, dtype=torch.int32)
        pad_scores = q.new_full((B, H, pad_n), float("-inf"), dtype=torch.float32)
        gathered_page_ids = torch.cat([gathered_page_ids, pad_pages], dim=-1)
        top_scores_sorted = torch.cat([top_scores_sorted, pad_scores], dim=-1)

    return HSASelectionResult(
        cand_page_ids=cand_page_ids,
        cand_mask=cand_mask,
        selected_page_ids=gathered_page_ids,
        selected_scores=top_scores_sorted.to(torch.float32),
    )


