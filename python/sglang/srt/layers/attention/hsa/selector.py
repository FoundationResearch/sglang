from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# 尝试导入 fused online topk kernel（优先路径），失败则 fallback 到 torch topk
_online_topk_group = None
_online_topk_head_maxpool = None
try:
    _hsa_kernel_ops_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "dev", "hsa-kernel-main", "ops")
    )
    if _hsa_kernel_ops_dir not in sys.path:
        sys.path.insert(0, _hsa_kernel_ops_dir)
    from topk_group import online_topk_group as _online_topk_group
    print("[HSA] Fused online_topk_group kernel loaded successfully.")
except Exception as _e:
    _online_topk_group = None
    print(f"[HSA] Failed to import fused online_topk_group, will fallback to torch topk: {_e}")

# Max-pooling top-k (no hsa_lse, uses swa_lse directly as per-query offset).
# Used by the prefill path when headwise_topk_softmax=False — ~2x faster than the
# softmax-then-max-pool training kernel, with negligible quality impact upstream.
try:
    if _hsa_kernel_ops_dir not in sys.path:
        sys.path.insert(0, _hsa_kernel_ops_dir)
    from topk_head_maxpool import online_topk_head as _online_topk_head_maxpool
    print("[HSA] Fused online_topk_head (max-pooling) kernel loaded successfully.")
except Exception as _e:
    _online_topk_head_maxpool = None
    print(f"[HSA] Failed to import fused online_topk_head_maxpool: {_e}")


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
      cand_chunk_repr_valid: [B, C] bool, caller-provided validity mask
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


def select_topk_pages_decode_fused(
    *,
    q: torch.Tensor,
    cand_page_ids: torch.Tensor,
    cand_mask: torch.Tensor,
    cand_repr: torch.Tensor,
    topk: int,
    page_size: int,
    sm_scale: Optional[float] = None,
    selection_strategy: str = "group",
    G: Optional[int] = None,
    per_qhead_prior_b: Optional[torch.Tensor] = None,
    _decode_q_offset: Optional[int] = None,
    _decode_hsa_window: Optional[int] = None,
) -> Optional[HSASelectionResult]:
    """使用 fused online_topk_group kernel 进行 top-k page selection。

    接口与 select_topk_pages_decode 对齐，返回 HSASelectionResult。
    如果 fused kernel 不可用则返回 None（调用方应 fallback 到 torch 路径）。

    Args:
      q: [B, HQ*D] or [B, HQ, D]  — query tensor
      cand_page_ids: [B, C] int32 padded with -1
      cand_mask: [B, C] bool
      cand_repr: [B, C, H_sel, D] — 候选 chunk 的 landmark 表示
      topk: 选取的 top-k 数量
      page_size: chunk/page 大小
      sm_scale: 可选的 attention scale（fused kernel 内部处理）
      selection_strategy: "group" or "head"（fused kernel 仅支持 group 模式）
      G: optional GQA group multiplier for the landmark heads.  ``None`` (default)
         delegates to the underlying kernel's shared-K path (lmks at h_kv
         granularity).  ``G > 1`` activates the per-q-head path: requires
         ``cand_repr.shape[2] == q_heads == h_kv * G``.  Used when sglang's
         layer has pre-computed per-q-head landmark Ks via chunk_attn_pool.
    """
    if _online_topk_group is None:
        return None

    B = q.shape[0]
    D = cand_repr.shape[-1]

    # q: [B, HQ*D] or [B, HQ, D] → [B, 1, HQ, D]
    if q.dim() == 2:
        HQ = q.shape[1] // D
        q_4d = q.view(B, 1, HQ, D)
    elif q.dim() == 3:
        q_4d = q.unsqueeze(1)  # [B, HQ, D] → [B, 1, HQ, D]
    else:
        return None  # 不支持的 shape，fallback

    # cand_repr: [B, C_max, H_sel, D].
    #
    # Per-q-head path (G > 1): cand_repr arrives with H_sel == h_q.  sglang's
    # fused online_topk_group does ``q.sum(over G)`` when h_q > h_kv (kernel
    # semantics), which is NOT what the official wants for per-q-head lmks.
    # The official's online_softmax_topk_head with explicit G uses
    # ``max(over G) of per-q-head scores`` then topk.  We replicate that in
    # pure PyTorch — slower but algorithmically faithful, which is what
    # alignment needs.  Fused kernel still handles the H_sel == h_kv (shared K)
    # case, which is the existing default.
    if G is not None and G > 1:
        # NOTE on R54b experiment: tried calling _online_topk_head_maxpool here
        # for the L=1 decode case — verified topk indices match the pure-pytorch
        # path (100% set overlap), but the tilelang kernel under-utilises the GPU
        # at L=1 (block-wise scheduling assumes multi-token training): CUDA time
        # rose from 13.3ms→19ms and decode latency went 32ms→38ms in profile.
        # Kept the pure-pytorch path; the kernel may revisit later if we add an
        # L=1 specialisation.
        # Pure-PyTorch per-q-head topk + max-over-G selection, matching the
        # official online_softmax_topk_head with explicit G:
        #   1. per-q-head scores   = q · lmk_k  (per q-head)
        #   2. selection scores    = max over G  → h_kv
        #   3. topk on selection   → indices at h_kv
        #   4. output scores       = gather per-q-head scores at those indices
        #                            (broadcast h_kv indices to h_q)  → h_q-shape
        # Indices stay h_kv-shaped (one selection per kv group).  Per-q-head
        # scores are returned separately via _per_qhead_scores so the
        # downstream chunk-weight fusion can use them.
        import math as _m
        B_c, C_c, h_q_c, D_c = cand_repr.shape
        h_kv_c = h_q_c // G
        sm_scale_ref = float(sm_scale) if sm_scale is not None else (1.0 / _m.sqrt(D_c))
        # R54d: drop redundant ops. Key invariants for the hot path (large C):
        #   * torch.topk indices are always in [0, C) — no need for clamp_min(0)
        #     when gathering with them (verified: topk on -inf-masked never
        #     returns negative indices unless K > num_finite, handled below).
        #   * torch.sort(topk_idx_kv) is NOT needed — chunk_weight kernel doesn't
        #     require ascending chunk ids (it iterates by k_offs, not chunk_id).
        #   * Final selected_page_ids.masked_fill(<0, -1) is a no-op when topk
        #     returns no negative indices.
        #   * scores_hq's masked_fill(invalid_hq, -inf) is redundant because the
        #     downstream chunk_weight kernel reads selected_page_ids and masks
        #     scores where sel_pages < 0.
        # Net: ~10 fewer launches per layer × 16 layers ≈ 160 launches/step.
        eff_topk = min(int(topk), int(C_c))
        q_3 = q_4d.view(B_c, h_q_c, D_c).float()                              # [B, h_q, D]
        cand_f = cand_repr.float()                                            # [B, C, h_q, D]
        scores_pqh = torch.einsum("bhd,bchd->bhc", q_3, cand_f) * sm_scale_ref  # [B, h_q, C]
        # R48 (Item #6): add entropy bias to per-q-head scores. With R47
        # (proper multi-layer slot reuse) the bias is consistent with the
        # official's chunk_attn_pool output.  prior_b_pool is fp32, matches.
        if per_qhead_prior_b is not None:
            scores_pqh.add_(per_qhead_prior_b.permute(0, 2, 1))
        # Selection scores: max over G  → [B, h_kv, C]
        scores_kv_sel = scores_pqh.view(B_c, h_kv_c, G, C_c).max(dim=2).values
        # In-place mask via cand_mask (broadcast cheaply, no expand+reshape needed).
        scores_kv_sel.masked_fill_(
            ~cand_mask.unsqueeze(1),  # [B, 1, C] — broadcast against [B, h_kv, C]
            float("-inf"),
        )
        _, topk_idx_kv = scores_kv_sel.topk(eff_topk, dim=-1)                # [B, h_kv, K] int64
        # No torch.sort — chunk_weight handles arbitrary order.
        # Per-q-head scores: gather with the h_kv→h_q broadcast.
        topk_idx_hq = topk_idx_kv.unsqueeze(2).expand(B_c, h_kv_c, G, eff_topk).reshape(
            B_c, h_q_c, eff_topk
        )
        scores_hq = torch.gather(scores_pqh, dim=-1, index=topk_idx_hq)
        # h_kv selection scores at the topk indices.
        scores_kv_out = torch.gather(scores_kv_sel, dim=-1, index=topk_idx_kv)
        if eff_topk < int(topk):
            # Cold path: very short context. Pad to topk size and mask invalids.
            pad = int(topk) - eff_topk
            topk_idx_kv = torch.cat(
                [topk_idx_kv,
                 topk_idx_kv.new_full((B_c, h_kv_c, pad), -1)],
                dim=-1,
            )
            scores_kv_out = torch.cat(
                [scores_kv_out,
                 scores_kv_out.new_full((B_c, h_kv_c, pad), float("-inf"))],
                dim=-1,
            )
            scores_hq = torch.cat(
                [scores_hq,
                 scores_hq.new_full((B_c, h_q_c, pad), float("-inf"))],
                dim=-1,
            )
        selected_page_ids = topk_idx_kv.to(torch.int32)
        # Attach per-q-head scores onto the result via a side-channel attribute
        # (the dataclass HSASelectionResult doesn't have a field for it).
        result = HSASelectionResult(
            cand_page_ids=cand_page_ids,
            cand_mask=cand_mask,
            selected_page_ids=selected_page_ids,
            selected_scores=scores_kv_out.to(torch.float32),
        )
        result._per_qhead_scores = scores_hq.to(torch.float32)  # [B, h_q, K]
        return result

    # R14: decode fast path — when q has a single query token (q_4d.shape[1]==1),
    # the tilelang `_online_topk_group` kernel uses (1, h_kv, B) thread blocks
    # × BLOCK_L=32 threads each, leaving the GB200 ~99% idle.  Plain torch
    # matmul + topk runs in <1 ms vs the tilelang kernel's 26 ms+ at long
    # context, and the math is identical (same q.sum-over-G semantics, same
    # softmax-scale, same ascending-sorted topk).
    if q_4d.shape[1] == 1 and G is None:  # G>1 path was already handled above
        import math as _m
        B_d = q_4d.shape[0]
        HQ_d = q_4d.shape[2]
        D_d = q_4d.shape[3]
        C_d = cand_repr.shape[1]
        H_sel_d = cand_repr.shape[2]

        # Match `OnlineTopKUnifiedFn.forward` GQA semantics:
        #   h_q >= h_kv -> sum q over G
        #   h_q <  h_kv -> sum lmks over G
        if HQ_d >= H_sel_d:
            assert HQ_d % H_sel_d == 0
            G_grp = HQ_d // H_sel_d
            q_grouped = (
                q_4d.view(B_d, 1, H_sel_d, G_grp, D_d).sum(dim=3).squeeze(1)
            )  # [B, h_kv, D]
            lmks_for_score = cand_repr  # [B, C, h_kv, D]
        else:
            assert H_sel_d % HQ_d == 0
            G_grp = H_sel_d // HQ_d
            lmks_for_score = cand_repr.view(B_d, C_d, HQ_d, G_grp, D_d).sum(dim=3)
            q_grouped = q_4d.squeeze(1)  # [B, HQ, D]

        h_shared = q_grouped.shape[1]
        sm_scale_d = float(sm_scale) if sm_scale is not None else (1.0 / _m.sqrt(D_d))
        # scores: [B, h_shared, C]
        scores_d = torch.einsum("bhd,bchd->bhc", q_grouped, lmks_for_score) * sm_scale_d
        scores_d = scores_d.masked_fill(~cand_mask.unsqueeze(1), float("-inf"))

        eff_topk = min(int(topk), int(C_d))
        # R17: skip the trailing sort + gather (selected pages feed into
        # softmax-aggregated downstream kernels — permutation-invariant).
        # R22: sorted=False — sbtopk skips its internal bitonic value-sort.
        top_scores_d, top_idx_d = scores_d.topk(eff_topk, dim=-1, sorted=False)
        idx_sorted_d = top_idx_d
        scores_sorted_d = top_scores_d

        # Pad to topk if fewer candidates than topk.
        if eff_topk < int(topk):
            pad_n = int(topk) - eff_topk
            idx_sorted_d = torch.cat(
                [idx_sorted_d,
                 idx_sorted_d.new_full((B_d, h_shared, pad_n), -1)],
                dim=-1,
            )
            scores_sorted_d = torch.cat(
                [scores_sorted_d,
                 scores_sorted_d.new_full((B_d, h_shared, pad_n), float("-inf"))],
                dim=-1,
            )

        selected_page_ids_d = idx_sorted_d.to(torch.int32)
        selected_page_ids_d = selected_page_ids_d.masked_fill(
            scores_sorted_d == float("-inf"), -1
        )

        return HSASelectionResult(
            cand_page_ids=cand_page_ids,
            cand_mask=cand_mask,
            selected_page_ids=selected_page_ids_d,
            selected_scores=scores_sorted_d.to(torch.float32),
        )

    fused_indices, fused_scores = _online_topk_group(
        q=q_4d,
        lmks=cand_repr,
        topk=topk,
        block_size=page_size,
        window_size=0,
        is_causal=False,
        q_offset=0,
        is_training=False,
    )

    # fused_indices: [B, 1, h_shared, topk] → [B, h_shared, topk]
    fused_indices = fused_indices.squeeze(1)
    fused_scores = fused_scores.squeeze(1)

    # 将 chunk index 映射回 page_id（候选集是 [0, C_max)，index == page_id）
    selected_page_ids = fused_indices.to(torch.int32)
    selected_page_ids = selected_page_ids.masked_fill(fused_indices < 0, -1)

    return HSASelectionResult(
        cand_page_ids=cand_page_ids,
        cand_mask=cand_mask,
        selected_page_ids=selected_page_ids,
        selected_scores=fused_scores.to(torch.float32),
    )
