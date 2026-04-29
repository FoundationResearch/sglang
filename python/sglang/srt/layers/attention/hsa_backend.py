from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Optional, Set

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hsa.kernels import hsa_decode_paged_fwd
from sglang.srt.layers.attention.hsa.metadata import HSAMetadata
from sglang.srt.layers.attention.hsa.selector import (
    _online_topk_group,
    build_active_page_candidates,
    select_topk_pages_decode,
    select_topk_pages_decode_fused,
)
from sglang.srt.layers.attention.hsa.utils import transform_page_table_1_to_real
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.distributed import tensor_model_parallel_all_gather
from sglang.srt.layers.dp_attention import get_attention_tp_size

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


logger = logging.getLogger(__name__)

_COMPILED_FLEX_ATTN = None
_FLEX_FALLBACK_WARNED = False


def _get_compiled_flex_attention():
    """Return a cached torch.compile'd flex_attention callable (best-effort)."""
    global _COMPILED_FLEX_ATTN
    if _COMPILED_FLEX_ATTN is not None:
        return _COMPILED_FLEX_ATTN
    from torch.nn.attention.flex_attention import flex_attention

    # Compiling the base entrypoint (as recommended by PyTorch) avoids the unfused slow path.
    _COMPILED_FLEX_ATTN = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTN


class HSAAttnBackend(AttentionBackend):
    """
    HSA (Hierarchical Sparse Attention) backend entrypoint.

    Phase 1 (this file): implement the "core scheduling point" so we can select the backend
    via CLI and have it run end-to-end. We build HSA-style metadata (page_table_1/real_page_table)
    and delegate dense attention compute + CUDA-graph plumbing to `TritonAttnBackend`.

    Phase 2+: replace delegation with real HSA selection + paged HSA kernels.

    Docs:
    - dev/hsa_dev_roadmap.md
    - dev/hsa_sglang_impl_todo.md
    """

    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__()
        self.model_runner = model_runner
        self.device = model_runner.device
        self.page_size = model_runner.page_size

        # HSA config resolution (override-only server args; model config is source-of-truth).
        server_args = getattr(model_runner, "server_args", None)
        cfg = getattr(getattr(model_runner, "model", None), "config", None)

        default_topk = getattr(cfg, "hsa_topk", 64)
        override_topk = getattr(server_args, "hsa_topk", None)
        self.hsa_topk = int(override_topk) if override_topk is not None else int(default_topk)

        default_sel = getattr(cfg, "hsa_selection_strategy", "head")
        override_sel = getattr(server_args, "hsa_selection_strategy", None)
        self.hsa_selection_strategy = (
            str(override_sel) if override_sel is not None else str(default_sel)
        )

        self.hsa_layers = getattr(server_args, "hsa_layers", None)
        self._hsa_layer_ids: Optional[Set[int]] = self._resolve_hsa_layer_ids()

        # LHSA merged softmax: enable_softmax1 adds a zero logit to the denominator.
        self.enable_softmax1 = bool(getattr(cfg, "enable_softmax1", False))

        # InnerX ultra only: we always run split-head stitch for HSA layers.
        # SWA window and LMK visibility are controlled by per-layer kwargs.
        self.hsa_window_size = None
        self.hsa_enable_swa_merging = False

        # Delegate dense attention implementation for now.
        # NOTE: For encoder-decoder models, triton backend is not supported.
        if model_runner.model_config.is_encoder_decoder:
            raise ValueError(
                "HSAAttnBackend currently delegates to TritonAttnBackend, which does not support "
                "encoder-decoder (cross attention). Please use --attention-backend flashinfer for "
                "encoder-decoder models, or wait for HSA cross-attention support."
            )

        self._dense_backend = TritonAttnBackend(model_runner, **kwargs)
        self.forward_metadata: Optional[HSAMetadata] = None

    def _resolve_hsa_layer_ids(self) -> Optional[Set[int]]:
        """Resolve per-layer HSA enable set.

        Priority:
        1) CLI: --hsa-layers (string/list)
        2) Model-provided default (FlashHSA): model.get_flashhsa_hsa_layer_ids()
        3) None => apply HSA selection to all layers (current default behavior)
        """

        layers = self.hsa_layers
        if layers is not None:
            try:
                if isinstance(layers, (list, tuple, set)):
                    return set(int(x) for x in layers)
                if isinstance(layers, str):
                    parts = [p for p in re.split(r"[,\s]+", layers.strip()) if p]
                    return set(int(p) for p in parts)
            except Exception:
                return None

        get_default = getattr(getattr(self.model_runner, "model", None), "get_flashhsa_hsa_layer_ids", None)
        if callable(get_default):
            try:
                default_layers = get_default()
                if default_layers is not None:
                    return set(int(x) for x in default_layers)
            except Exception:
                return None

        return None

    def _is_hsa_layer(self, layer_id: int) -> bool:
        if self._hsa_layer_ids is None:
            return True
        return int(layer_id) in self._hsa_layer_ids

    # (removed) _get_effective_window_size: InnerX ultra passes window via kwargs

    def _run_selection_decode(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        selection_q: Optional[torch.Tensor] = None,
        window_size_override: Optional[int] = None,
        kv_head_offset: int = 0,
        kv_head_count: Optional[int] = None,
        split_info: Optional[dict] = None,
    ) -> None:
        """Top-k page selection for decode.

        Uses logical token positions (not KV slot indices) for candidate
        building and window exclusion, matching the official FlashHSA semantics:
          limit_chunk = (query_pos - window_size + 1) // page_size
          valid candidates: completed pages with page_id < limit_chunk
        """
        md = self.forward_metadata
        if md is None:
            return
        if not self._is_hsa_layer(layer.layer_id):
            return

        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if pool is None or not hasattr(pool, "get_key_buffer"):
            return

        hsa_window = 0
        if window_size_override is not None:
            hsa_window = int(window_size_override)

        page_size = int(self.page_size)
        B = int(forward_batch.batch_size)
        H_sel = int(kv_head_count) if kv_head_count is not None else int(layer.tp_k_head_num)
        device = q.device

        # Overlay out_cache_loc into page_table_1 at position (seq_len - 1).
        page_table_1 = md.page_table_1
        if getattr(forward_batch, "out_cache_loc", None) is not None:
            seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
            if B > 0:
                page_table_1 = page_table_1.clone()
                out_locs = forward_batch.out_cache_loc[:B].to(torch.int32)
                for b in range(B):
                    seqlen = int(seq_lens_i64[b].item())
                    if seqlen > 0 and seqlen <= page_table_1.shape[1]:
                        page_table_1[b, seqlen - 1] = out_locs[b]

        seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)

        # Compute per-request candidate pages using logical positions.
        # completed_pages = seq_len // page_size
        # limit_chunk = (query_pos - window_size + 1) // page_size
        #   where query_pos = seq_len - 1
        # Valid candidates: page_id in [0, min(completed_pages, limit_chunk))
        completed_pages = seq_lens_i64 // page_size  # [B]

        if hsa_window > 0:
            query_pos = seq_lens_i64 - 1  # [B]
            limit_chunk = (query_pos - hsa_window + 1) // page_size  # [B]
            limit_chunk = limit_chunk.clamp(min=0)
            effective_cands = torch.min(completed_pages, limit_chunk)  # [B]
        else:
            effective_cands = completed_pages  # [B]

        effective_cands = effective_cands.clamp(min=0)
        C_max = int(effective_cands.max().item())

        if C_max == 0:
            md.hsa_cand_page_ids = page_table_1.new_full((B, 0), -1, dtype=torch.int32)
            md.hsa_cand_mask = page_table_1.new_zeros((B, 0), dtype=torch.bool)
            md.hsa_selected_page_ids = page_table_1.new_full(
                (B, H_sel, int(self.hsa_topk)), -1, dtype=torch.int32
            )
            md.hsa_selected_scores = q.new_full(
                (B, H_sel, int(self.hsa_topk)), float("-inf"), dtype=torch.float32
            )
            return

        # Build padded candidate page_ids [B, C_max] using logical page indices.
        cand_range = torch.arange(C_max, device=device, dtype=torch.int32)
        cand_page_ids = cand_range.unsqueeze(0).expand(B, C_max).contiguous()
        cand_mask = cand_page_ids < effective_cands.unsqueeze(1).to(torch.int32)
        cand_page_ids = cand_page_ids.masked_fill(~cand_mask, -1)

        # Load LMK keys via page_table_1 (logical page → KV slot).
        safe_page_ids = cand_page_ids.clamp(min=0).to(torch.int64)
        lmk_token_pos = safe_page_ids * page_size + (page_size - 1)  # [B, C_max]
        lmk_token_pos_safe = lmk_token_pos.clamp(max=page_table_1.shape[1] - 1)
        lmk_locs = torch.gather(
            page_table_1.to(torch.int64), 1, lmk_token_pos_safe
        )  # [B, C_max]

        k_cache = pool.get_key_buffer(layer.layer_id)  # [num_locs, H_total, D]
        D = int(k_cache.shape[2])
        flat_lmk_locs = lmk_locs.reshape(-1)
        flat_repr = k_cache[flat_lmk_locs]  # [B*C_max, H_total, D]
        if kv_head_count is not None:
            flat_repr = flat_repr[:, int(kv_head_offset):int(kv_head_offset) + int(kv_head_count), :]
        cand_repr = flat_repr.view(B, C_max, H_sel, D)

        # unified_retrieval: group+sum KV heads to match retrieval_dim.
        unified = split_info is not None and split_info.get("unified_retrieval", False)
        retrieval_dim = split_info.get("retrieval_dim", None) if split_info is not None else None
        if unified and retrieval_dim is not None:
            tp_size = get_attention_tp_size()
            if tp_size > 1:
                # All-gather KV head landmark keys across TP ranks so every rank
                # sees the full head dimension, matching training semantics where
                # a single retrieval head operates on all KV heads.
                cand_repr = tensor_model_parallel_all_gather(cand_repr.contiguous(), dim=2)
                H_sel = cand_repr.shape[2]  # updated to full H_hsa
            flat_dim = H_sel * D
            if flat_dim != retrieval_dim:
                G_groups = flat_dim // retrieval_dim
                cand_repr = cand_repr.reshape(B, C_max, 1, G_groups, retrieval_dim)
                cand_repr = cand_repr.sum(dim=3)
            else:
                cand_repr = cand_repr.reshape(B, C_max, 1, flat_dim)

        q_sel = selection_q if selection_q is not None else q
        if unified and retrieval_dim is not None:
            tp_size = get_attention_tp_size()
            if tp_size > 1:
                # All-gather sel_q across TP ranks to reconstruct the full
                # retrieval_dim vector (ColumnParallelLinear splits along last dim).
                q_sel = tensor_model_parallel_all_gather(q_sel.contiguous(), dim=-1)
        sm_scale_val = getattr(layer, "scaling", None)

        sel = select_topk_pages_decode_fused(
            q=q_sel,
            cand_page_ids=cand_page_ids,
            cand_mask=cand_mask,
            cand_repr=cand_repr,
            topk=int(self.hsa_topk),
            page_size=int(self.page_size),
            sm_scale=sm_scale_val,
            selection_strategy=str(self.hsa_selection_strategy),
        )
        if sel is None:
            sel = select_topk_pages_decode(
                q=q_sel,
                cand_page_ids=cand_page_ids,
                cand_mask=cand_mask,
                cand_chunk_repr=cand_repr,
                cand_chunk_repr_valid=cand_mask,
                topk=int(self.hsa_topk),
                selection_strategy=str(self.hsa_selection_strategy),
                sm_scale=sm_scale_val,
            )

        md.hsa_cand_page_ids = sel.cand_page_ids
        md.hsa_cand_mask = sel.cand_mask
        md.hsa_selected_page_ids = sel.selected_page_ids
        md.hsa_selected_scores = sel.selected_scores

    def _compute_internal_swa_decode(
        self,
        *,
        q_hsa: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        page_table_1: torch.Tensor,
        H_swa: int,
        H_hsa: int,
        HQ_hsa: int,
        hsa_window: int,
    ) -> tuple:
        """Internal SWA on HSA heads (LHSA semantics).

        Computes chunk-aligned sliding window attention on the HSA heads,
        excluding LMK positions.  Returns output and per-kv-head logsumexp
        for merged softmax.

        Returns
        -------
        swa_o : [B, HQ_hsa, D] float32
        lse_kv : [B, H_hsa] float32  (GQA-aggregated logsumexp)
        """
        B, _, D = q_hsa.shape
        device = q_hsa.device

        swa_o = torch.zeros((B, HQ_hsa, D), device=device, dtype=torch.float32)
        lse_kv = torch.full((B, H_hsa), float("-inf"), device=device, dtype=torch.float32)

        if hsa_window <= 0:
            return swa_o, lse_kv

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None:
            return swa_o, lse_kv

        k_cache_all = pool.get_key_buffer(layer.layer_id)
        v_cache_all = pool.get_value_buffer(layer.layer_id)
        seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        page_size = int(self.page_size)
        sm_scale = float(getattr(layer, "scaling", 1.0))

        for b in range(B):
            seqlen = int(seq_lens_i64[b].item())
            if seqlen <= 0:
                continue
            q_pos = seqlen - 1
            # Chunk-aligned window start (reference: block_causal_mask).
            raw_start = q_pos - hsa_window + 1
            chunk_start = max(0, (raw_start // page_size) * page_size) if raw_start >= 0 else 0

            # Positions [chunk_start, q_pos], excluding LMK slots.
            tok_pos = torch.arange(chunk_start, seqlen, device=device, dtype=torch.int64)
            keep = (tok_pos % page_size) != (page_size - 1)
            tok_pos = tok_pos[keep]
            if tok_pos.numel() == 0:
                continue

            token_locs = page_table_1[b, tok_pos].to(torch.int64)
            q_hgd = q_hsa[b].view(H_hsa, Gh, D).to(torch.float32)

            lse_per_q = torch.full((H_hsa, Gh), float("-inf"), device=device, dtype=torch.float32)
            for kv_h in range(H_hsa):
                kv_h_global = H_swa + kv_h
                k_win = k_cache_all[token_locs, kv_h_global, :].to(torch.float32)
                v_win = v_cache_all[token_locs, kv_h_global, :].to(torch.float32)
                logits = (q_hgd[kv_h] @ k_win.transpose(0, 1)) * sm_scale
                lse_per_q[kv_h] = torch.logsumexp(logits, dim=-1)
                p = torch.softmax(logits, dim=-1)
                o = p @ v_win
                hq_start = kv_h * Gh
                swa_o[b, hq_start : hq_start + Gh, :] = o

            # Aggregate logsumexp across GQA groups → per-kv-head.
            lse_kv[b] = torch.logsumexp(lse_per_q, dim=-1)

        return swa_o, lse_kv

    def _build_hsa_lmk_excluded_indices(self, forward_batch: ForwardBatch):
        """Build KV indices with LMK positions excluded (for HSA layers only).

        FlashHSA mask semantics: LMK tokens are written to KV cache but never
        attended as KV.  We build a separate kv_indptr/kv_indices pair that
        skips LMK slots so HSA layers see only real tokens.
        """
        bs = forward_batch.batch_size
        device = self.device
        page_size = int(self.page_size)

        max_seqlen = int(forward_batch.seq_lens.max().item())
        pos = torch.arange(max_seqlen, device=device, dtype=torch.int64)
        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen
        ]
        valid = pos[None, :] < forward_batch.seq_lens.to(torch.int64)[:, None]
        keep = valid & ((pos[None, :] + 1) % page_size != 0)
        kv_lens = keep.sum(dim=1, dtype=torch.int32)

        kv_indptr = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)
        kv_indices = page_table[keep].to(torch.int64)

        dense_backend = self._dense_backend
        num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=device)
        dense_backend.get_num_kv_splits(num_kv_splits, kv_lens)

        return kv_indptr, kv_indices, kv_lens, num_kv_splits

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # First, initialize dense backend metadata (standard indices, no LMK exclusion).
        # SWA layers will use these indices directly.
        self._dense_backend.init_forward_metadata(forward_batch)

        # Build HSA metadata scaffold (paged-KV-first).
        if forward_batch.seq_lens_cpu is not None:
            max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
        else:
            max_seqlen_k = int(forward_batch.seq_lens.max().item())

        page_table_1 = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]
        real_page_table = transform_page_table_1_to_real(page_table_1, self.page_size)

        dense_md = getattr(self._dense_backend, "forward_metadata", None)
        kv_indptr = getattr(dense_md, "kv_indptr", None)
        kv_indices = getattr(dense_md, "kv_indices", None)
        window_kv_indptr = getattr(dense_md, "window_kv_indptr", None)
        window_kv_indices = getattr(dense_md, "window_kv_indices", None)

        # Build HSA-specific LMK-excluded indices for decode.
        hsa_kv_indptr = None
        hsa_kv_indices = None
        hsa_kv_lens = None
        hsa_num_kv_splits = None
        if forward_batch.forward_mode.is_decode_or_idle():
            hsa_kv_indptr, hsa_kv_indices, hsa_kv_lens, hsa_num_kv_splits = (
                self._build_hsa_lmk_excluded_indices(forward_batch)
            )

        # Build extend-specific fields when in extend mode.
        token_positions = None
        token_to_seq_id = None
        extend_seq_lens = None
        extend_prefix_lens = None
        engine_indices = None
        if forward_batch.forward_mode.is_extend():
            token_positions = forward_batch.positions
            extend_seq_lens = forward_batch.extend_seq_lens
            extend_prefix_lens = forward_batch.extend_prefix_lens
            if extend_seq_lens is not None:
                token_to_seq_id = torch.repeat_interleave(
                    torch.arange(len(extend_seq_lens), device=self.device, dtype=torch.int32),
                    extend_seq_lens.to(torch.int64),
                )
                engine_indices = torch.cat([
                    torch.arange(int(plen), int(plen) + int(elen),
                                 device=self.device, dtype=torch.int64)
                    for plen, elen in zip(extend_prefix_lens, extend_seq_lens)
                ])

        self.forward_metadata = HSAMetadata(
            page_size=self.page_size,
            cache_seqlens_int32=forward_batch.seq_lens.to(torch.int32),
            max_seqlen_k=max_seqlen_k,
            page_table_1=page_table_1,
            real_page_table=real_page_table,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            window_kv_indptr=window_kv_indptr,
            window_kv_indices=window_kv_indices,
            hsa_kv_indptr=hsa_kv_indptr,
            hsa_kv_indices=hsa_kv_indices,
            hsa_kv_lens=hsa_kv_lens,
            hsa_num_kv_splits=hsa_num_kv_splits,
            token_positions=token_positions,
            token_to_seq_id=token_to_seq_id,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            engine_indices=engine_indices,
        )

    # ---- CUDA graph plumbing: delegate to dense backend for now ----

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        return self._dense_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        return self._dense_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        return self._dense_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return self._dense_backend.get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        return self._dense_backend.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        return self._dense_backend.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Optional: InnerX-style split-head HSA passes extra tensors via kwargs.
        split_info = kwargs.get("hsa_split_head_info", None)
        selection_q = kwargs.get("hsa_selection_q", None)

        # Step 4: run selection to populate metadata.
        if split_info is not None:
            h_swa = int(split_info.get("h_swa", 0))
            h_hsa = int(split_info.get("h_hsa", 0))
            hsa_window = int(split_info.get("swa_window_size", 0) or 0)
            self._run_selection_decode(
                q=q,
                layer=layer,
                forward_batch=forward_batch,
                selection_q=selection_q,
                window_size_override=hsa_window,
                kv_head_offset=h_swa,
                kv_head_count=h_hsa,
                split_info=split_info,
            )
        else:
            self._run_selection_decode(q=q, layer=layer, forward_batch=forward_batch)

        # Non-HSA layers: delegate to dense backend.
        if not self._is_hsa_layer(layer.layer_id):
            return self._dense_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None:
            return self._dense_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )

        # Save current KV into the paged KV cache (same as TritonAttnBackend).
        if save_kv_cache:
            pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        # Prepare q: [B, HQ, D]
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError(
                "HSAAttnBackend.hsa_decode_paged_fwd currently assumes qk_head_dim == v_head_dim."
            )

        # Robustness: overlay out_cache_loc into a temporary token->slot table at position (seq_len - 1).
        page_table_1 = md.page_table_1
        if getattr(forward_batch, "out_cache_loc", None) is not None:
            seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
            B = int(forward_batch.batch_size)
            if B > 0:
                page_table_1 = page_table_1.clone()
                out_locs = forward_batch.out_cache_loc[:B].to(torch.int32)
                for b in range(B):
                    seqlen = int(seq_lens_i64[b].item())
                    if seqlen > 0 and seqlen <= page_table_1.shape[1]:
                        page_table_1[b, seqlen - 1] = out_locs[b]

        # ---- InnerX split-head HSA (ultra reference) ----
        # In this mode, the layer output is a head-wise concatenation:
        #   out = cat([out_swa_heads, out_hsa_heads], dim=head)
        # where:
        #   - SWA heads run local sliding-window attention
        #   - HSA heads run sparse paged attention with per-page weights
        if split_info is not None:
            selected_page_ids = md.hsa_selected_page_ids
            selected_scores = md.hsa_selected_scores
            if selected_page_ids is None or selected_scores is None:
                raise RuntimeError("HSA selection did not populate metadata")

            HQ_total = int(layer.tp_q_head_num)
            H_total = int(layer.tp_k_head_num)
            HQ_swa = int(split_info.get("hq_swa", 0))
            HQ_hsa = int(split_info.get("hq_hsa", 0))
            H_swa = int(split_info.get("h_swa", 0))
            H_hsa = int(split_info.get("h_hsa", 0))
            window_size = split_info.get("swa_window_size", None)
            swa_exclude_lmk = bool(split_info.get("swa_exclude_lmk", False))

            if HQ_swa + HQ_hsa != HQ_total:
                raise ValueError(
                    f"hsa_split_head_info mismatch: HQ_swa+HQ_hsa={HQ_swa+HQ_hsa} != HQ_total={HQ_total}"
                )
            if H_swa + H_hsa != H_total:
                raise ValueError(
                    f"hsa_split_head_info mismatch: H_swa+H_hsa={H_swa+H_hsa} != H_total={H_total}"
                )
            if HQ_hsa <= 0 or H_hsa <= 0:
                raise ValueError("InnerX split-head requires non-empty HSA head partitions")
            # unified_retrieval: selection returns [B, 1, K]; expand to [B, H_hsa, K]
            if int(selected_page_ids.shape[1]) == 1 and H_hsa > 1:
                selected_page_ids = selected_page_ids.expand(B, H_hsa, -1).contiguous()
                selected_scores = selected_scores.expand(B, H_hsa, -1).contiguous()
                md.hsa_selected_page_ids = selected_page_ids
                md.hsa_selected_scores = selected_scores
            elif int(selected_page_ids.shape[1]) != H_hsa:
                raise ValueError(
                    f"Selection H mismatch: got {int(selected_page_ids.shape[1])}, expected H_hsa={H_hsa}"
                )

            B = int(q3.shape[0])
            D = int(q3.shape[2])
            q_swa = q3[:, :HQ_swa, :]
            q_hsa = q3[:, HQ_swa:, :]

            # SWA branch: sliding window attention on upper (SWA) heads.
            upper_swa_window = int(split_info.get("upper_swa_window_size", 0) or -1)
            out_swa = torch.zeros((B, HQ_swa, D), device=q3.device, dtype=torch.float32)
            if HQ_swa > 0 and H_swa > 0:
                import torch.nn.functional as F

                seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
                k_cache_all = pool.get_key_buffer(layer.layer_id)
                v_cache_all = pool.get_value_buffer(layer.layer_id)
                swa_sliding_window = upper_swa_window if upper_swa_window > 0 else None
                sm_scale = float(getattr(layer, "scaling", 1.0))
                assert HQ_swa % H_swa == 0
                Gs = HQ_swa // H_swa

                for b in range(B):
                    seqlen = int(seq_lens_i64[b].item())
                    if seqlen <= 0:
                        continue
                    if swa_sliding_window is not None and int(swa_sliding_window) > 0:
                        w = min(seqlen, int(swa_sliding_window))
                    else:
                        w = seqlen
                    start = seqlen - w
                    tok_pos = torch.arange(start, seqlen, device=q3.device, dtype=torch.int64)
                    if swa_exclude_lmk:
                        keep = (tok_pos % int(self.page_size)) != (int(self.page_size) - 1)
                        tok_pos = tok_pos[keep]
                    if tok_pos.numel() == 0:
                        continue
                    token_locs = page_table_1[b, tok_pos].to(torch.int64)

                    # Gather K/V: [H_swa, S, D]
                    k_win = k_cache_all[token_locs, :H_swa, :].transpose(0, 1)
                    v_win = v_cache_all[token_locs, :H_swa, :].transpose(0, 1)
                    # Expand for GQA: [HQ_swa, S, D]
                    k_win = k_win.repeat_interleave(Gs, dim=0)
                    v_win = v_win.repeat_interleave(Gs, dim=0)
                    # Q: [HQ_swa, 1, D]
                    q_b = q_swa[b].unsqueeze(1)

                    # SDPA: no mask needed for decode (Q_LEN=1, attend to all)
                    o = F.scaled_dot_product_attention(
                        q_b.to(torch.float32), k_win.to(torch.float32),
                        v_win.to(torch.float32), scale=sm_scale,
                    )  # [HQ_swa, 1, D]
                    out_swa[b] = o.squeeze(1)

            # Internal SWA on HSA heads (LHSA semantics).
            swa_o_inner, lse_kv = self._compute_internal_swa_decode(
                q_hsa=q_hsa,
                layer=layer,
                forward_batch=forward_batch,
                page_table_1=page_table_1,
                H_swa=H_swa,
                H_hsa=H_hsa,
                HQ_hsa=HQ_hsa,
                hsa_window=hsa_window,
            )

            # HSA branch weights via merged softmax (LHSA reference lines 371-384).
            assert HQ_hsa % H_hsa == 0
            Gh = HQ_hsa // H_hsa
            TOPK = int(selected_page_ids.shape[2])
            valid = selected_page_ids >= 0
            scores = selected_scores.masked_fill(~valid, float("-inf"))

            if hsa_window > 0:
                # Merged softmax: cat([chunk_scores, swa_lse]) → globally normalized.
                if not self.enable_softmax1:
                    cat_scores = torch.cat(
                        [scores, lse_kv.unsqueeze(-1)], dim=-1
                    )  # [B, H_hsa, K+1]
                    swa_weight_idx = -1
                else:
                    cat_scores = torch.cat(
                        [
                            scores,
                            lse_kv.unsqueeze(-1),
                            torch.zeros(B, H_hsa, 1, device=scores.device, dtype=scores.dtype),
                        ],
                        dim=-1,
                    )  # [B, H_hsa, K+2]
                    swa_weight_idx = -2
                merged_w = torch.softmax(cat_scores, dim=-1)
                merged_w = torch.nan_to_num(merged_w, nan=0.0)
                w_kv = merged_w[:, :, :TOPK].to(q_hsa.dtype)
                swa_w_kv = merged_w[:, :, swa_weight_idx]  # [B, H_hsa]
            else:
                # Backward compat: independent softmax (no internal SWA).
                w_kv = torch.softmax(scores, dim=-1)
                w_kv = torch.nan_to_num(w_kv, nan=0.0).to(q_hsa.dtype)
                swa_w_kv = None

            w_q = (
                w_kv[:, :, None, :]
                .expand(B, H_hsa, Gh, TOPK)
                .reshape(B, HQ_hsa, TOPK)
                .contiguous()
            )

            k_cache_hsa = pool.get_key_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :]
            v_cache_hsa = pool.get_value_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :]
            out_hsa = hsa_decode_paged_fwd(
                q=q_hsa,
                k_cache=k_cache_hsa,
                v_cache=v_cache_hsa,
                page_table_1=page_table_1,
                selected_page_ids=selected_page_ids,
                hsa_weights=w_q,
                page_size=int(self.page_size),
                sm_scale=getattr(layer, "scaling", None),
                mask_last_token=True,
            )  # [B, HQ_hsa, D]

            # Weighted SWA fusion (LHSA: o_lower = hsa_o + swa_o * swa_weight).
            if swa_w_kv is not None:
                swa_w_q = (
                    swa_w_kv[:, :, None]
                    .expand(B, H_hsa, Gh)
                    .reshape(B, HQ_hsa)
                )
                out_hsa = out_hsa.to(torch.float32) + swa_o_inner * swa_w_q[:, :, None]

            out_all = torch.empty((B, HQ_total, D), device=q3.device, dtype=torch.bfloat16)
            if HQ_swa > 0:
                out_all[:, :HQ_swa, :] = out_swa.to(torch.bfloat16)
            out_all[:, HQ_swa:, :] = out_hsa.to(torch.bfloat16)
            return out_all.reshape(q.shape[0], HQ_total * layer.v_head_dim)
        raise RuntimeError(
            "InnerX ultra only: HSA layers must pass `hsa_split_head_info` kwargs."
        )

    # ---- Extend helpers (production + reference) ----

    _USE_EXTEND_REFERENCE = os.getenv("SGLANG_HSA_EXTEND_REFERENCE", "0") == "1"

    def _compute_internal_swa_extend(
        self,
        *,
        q_hsa: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        page_table_1: torch.Tensor,
        H_swa: int,
        H_hsa: int,
        HQ_hsa: int,
        hsa_window: int,
    ) -> tuple:
        """Dispatch to batched (production) or reference implementation."""
        if self._USE_EXTEND_REFERENCE:
            return self._compute_internal_swa_extend_reference(
                q_hsa=q_hsa, layer=layer, forward_batch=forward_batch,
                page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
                HQ_hsa=HQ_hsa, hsa_window=hsa_window,
            )
        return self._compute_internal_swa_extend_batched(
            q_hsa=q_hsa, layer=layer, forward_batch=forward_batch,
            page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
            HQ_hsa=HQ_hsa, hsa_window=hsa_window,
        )

    def _compute_internal_swa_extend_batched(
        self,
        *,
        q_hsa: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        page_table_1: torch.Tensor,
        H_swa: int,
        H_hsa: int,
        HQ_hsa: int,
        hsa_window: int,
    ) -> tuple:
        """Internal SWA on HSA heads for extend via triton extend kernel.

        Uses the triton extend_attention_fwd_unified kernel with custom
        kv_indices that implement chunk-aligned sliding window + LMK exclusion,
        matching the training block_causal_mask semantics exactly.

        Memory: O(max_kv_per_token) for indices, no element-level mask needed.
        Only supports batch=1 (SGLang continuous batching guarantee).

        Returns
        -------
        swa_o : [T, HQ_hsa, D] float32
        lse_kv : [T, H_hsa] float32  (GQA-aggregated logsumexp)
        """
        T, _, D = q_hsa.shape
        device = q_hsa.device

        swa_o = torch.zeros((T, HQ_hsa, D), device=device, dtype=torch.float32)
        lse_kv = torch.full((T, H_hsa), float("-inf"), device=device, dtype=torch.float32)

        if hsa_window <= 0:
            return swa_o, lse_kv

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None or md.token_positions is None or md.token_to_seq_id is None:
            return swa_o, lse_kv

        # --- 只支持 batch=1 ---
        extend_seq_lens = md.extend_seq_lens
        extend_prefix_lens = md.extend_prefix_lens
        assert extend_seq_lens is not None and extend_prefix_lens is not None
        B = int(extend_seq_lens.shape[0])
        assert B == 1, (
            f"_compute_internal_swa_extend_batched 当前只支持 batch=1，"
            f"但收到 B={B}。"
        )

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        page_size = int(self.page_size)
        sm_scale = float(getattr(layer, "scaling", 1.0))

        prefix_len = int(extend_prefix_lens[0].item())
        extend_len = int(extend_seq_lens[0].item())

        # --- 构建 chunk-aligned window + LMK-excluded 的 kv_indices ---
        # 对于每个 Q token t (engine_idx = prefix_len + t):
        #   chunk_start = max(0, ((engine_idx - hsa_window + 1) // page_size) * page_size)
        #   valid KV: positions in [chunk_start, engine_idx] where (pos+1) % page_size != 0
        # 由于 batch=1，我们构建一个统一的 kv_indptr/kv_indices

        # 计算所有 Q token 的 engine indices
        engine_indices = torch.arange(prefix_len, prefix_len + extend_len,
                                      device=device, dtype=torch.int64)  # [T]

        # 计算每个 Q token 的 chunk-aligned window start
        raw_starts = engine_indices - hsa_window + 1  # [T]
        chunk_starts = (raw_starts // page_size) * page_size  # [T], chunk-aligned
        chunk_starts = chunk_starts.clamp(min=0)

        # 每个 Q token 的 KV 范围: [chunk_start, engine_idx]
        # 排除 LMK: (pos + 1) % page_size != 0
        # 最大 KV 长度 per token: hsa_window + page_size (chunk alignment 可能多一个 page)
        # 减去 LMK tokens: 每 page_size 个 token 有 1 个 LMK
        max_kv_per_token = hsa_window + page_size  # 上界

        # 构建 kv_indptr 和 kv_indices
        # 为了效率，使用向量化操作
        kv_counts = torch.zeros(T, device=device, dtype=torch.int32)
        # 实际 KV 数 = (engine_idx - chunk_start + 1) - num_lmk_in_range
        range_lens = engine_indices - chunk_starts + 1  # [T]
        # LMK 数量: 在 [chunk_start, engine_idx] 范围内 (pos+1) % page_size == 0 的数量
        # = (engine_idx // page_size) - (chunk_start // page_size) + (1 if (engine_idx+1)%page_size==0 else 0)
        # 简化: floor(engine_idx / page_size) - floor((chunk_start-1) / page_size)
        # 但更准确的是: 在 [chunk_start, engine_idx] 中，page_size-1, 2*page_size-1, ... 的数量
        first_lmk = (chunk_starts // page_size) * page_size + (page_size - 1)  # 第一个可能的 LMK 位置
        # 如果 first_lmk < chunk_start，则从下一个 LMK 开始
        first_lmk = torch.where(first_lmk >= chunk_starts, first_lmk,
                                first_lmk + page_size)
        # LMK 数量 = max(0, (engine_idx - first_lmk) // page_size + 1) if first_lmk <= engine_idx
        num_lmk = torch.where(
            first_lmk <= engine_indices,
            (engine_indices - first_lmk) // page_size + 1,
            torch.zeros_like(engine_indices),
        )
        kv_counts = (range_lens - num_lmk).to(torch.int32).clamp(min=0)

        kv_indptr = torch.zeros(T + 1, device=device, dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(kv_counts, dim=0)
        total_kv = int(kv_indptr[-1].item())

        if total_kv == 0:
            return swa_o, lse_kv

        # 构建 kv_indices: 对于每个 Q token，列出其 KV slot indices
        # 使用 page_table_1 将 engine position → KV cache slot
        kv_indices = torch.empty(total_kv, device=device, dtype=torch.int64)

        # 向量化构建: 展开所有 Q token 的 KV 范围
        # 方法: 对每个 Q token t，生成 [chunk_start[t], engine_idx[t]] 中非 LMK 的位置
        # 然后通过 page_table_1 映射到 KV cache slot
        #
        # 为了避免 Python 循环，使用 padded 矩阵 + mask 的方式
        max_range = int(range_lens.max().item())
        if max_range <= 0:
            return swa_o, lse_kv

        # offsets: [T, max_range], 每行是 [0, 1, ..., max_range-1]
        offsets = torch.arange(max_range, device=device, dtype=torch.int64).unsqueeze(0)  # [1, max_range]
        # positions: [T, max_range], 每行是 [chunk_start[t], chunk_start[t]+1, ...]
        positions = chunk_starts.unsqueeze(1) + offsets  # [T, max_range]

        # valid mask: position <= engine_idx AND (position+1) % page_size != 0
        valid = (positions <= engine_indices.unsqueeze(1)) & \
                ((positions + 1) % page_size != 0)

        # 通过 page_table_1 映射到 KV cache slot
        positions_safe = positions.clamp(min=0, max=page_table_1.shape[1] - 1)
        kv_slots = page_table_1[0, positions_safe].to(torch.int64)  # [T, max_range]

        # 使用 valid mask 提取有效的 kv_slots
        kv_indices = kv_slots[valid]  # [total_kv]

        # --- 调用 triton extend kernel ---
        # 把每个 Q token 当作一个独立的 "sequence"，这样每个 Q token
        # 有自己的 KV indices（chunk-aligned window + LMK excluded）
        # batch_size = T, 每个 "sequence" 有 1 个 Q token
        qo_indptr = torch.arange(T + 1, device=device, dtype=torch.int32)  # [0, 1, 2, ..., T]
        prefix_lens_tensor = kv_counts  # 每个 "sequence" 的所有 KV 都是 "prefix"

        # 获取 HSA heads 的 KV cache buffer
        pool_k = pool.get_key_buffer(layer.layer_id)[:, H_swa:H_swa + H_hsa, :]
        pool_v = pool.get_value_buffer(layer.layer_id)[:, H_swa:H_swa + H_hsa, :]

        # 输出 tensor
        swa_o_3 = torch.zeros((T, HQ_hsa, D), device=device, dtype=q_hsa.dtype)

        # LSE 输出 tensor: [T, HQ_hsa]，由 triton kernel 直接写入
        lse_raw = torch.full((T, HQ_hsa), float("-inf"), device=device, dtype=torch.float32)

        # 使用 unified kernel（1-stage），因为所有 KV 都在 cache 中（没有 extend KV）
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd_unified,
        )
        extend_attention_fwd_unified(
            q_hsa,           # [T, HQ_hsa, D]
            swa_o_3,         # [T, HQ_hsa, D] output
            pool_k,          # KV cache key buffer (HSA heads only)
            pool_v,          # KV cache value buffer (HSA heads only)
            qo_indptr,       # [T+1]: [0, 1, 2, ..., T]
            kv_indptr,       # [T+1]: per-token KV offsets
            kv_indices,      # [total_kv]: KV cache slot indices
            prefix_lens_tensor,  # [T]: 每个 "sequence" 的所有 KV 都是 prefix
            max_len_extend=1,    # 每个 "sequence" 只有 1 个 extend token
            sm_scale=sm_scale,
            is_causal=False,  # causal mask 已通过 kv_indices 实现
            lse_output=lse_raw,  # [T, HQ_hsa] kernel 直接写入 LSE
        )

        swa_o = swa_o_3.to(torch.float32)

        # --- 从 kernel 返回的 per-query-head LSE 聚合到 per-kv-head ---
        # lse_raw: [T, HQ_hsa] → [T, H_hsa, Gh] → logsumexp over Gh → [T, H_hsa]
        lse_per_group = lse_raw.view(T, H_hsa, Gh)  # [T, H_hsa, Gh]
        lse_kv = torch.logsumexp(lse_per_group, dim=-1)  # [T, H_hsa]

        return swa_o, lse_kv

    def _compute_internal_swa_extend_reference(
        self,
        *,
        q_hsa: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        page_table_1: torch.Tensor,
        H_swa: int,
        H_hsa: int,
        HQ_hsa: int,
        hsa_window: int,
    ) -> tuple:
        """Internal SWA on HSA heads for extend (per-token).

        For each query token t at position pos_t:
          1. Chunk-aligned window: start = max(0, ((pos_t - window + 1) // PS) * PS)
          2. Range [chunk_start, pos_t], excluding LMK positions
          3. Compute softmax(q @ K^T * sm_scale) @ V → output + LSE
          4. Aggregate LSE across GQA groups → per-kv-head LSE

        Returns
        -------
        swa_o : [T, HQ_hsa, D] float32
        lse_kv : [T, H_hsa] float32
        """
        T, _, D = q_hsa.shape
        device = q_hsa.device

        swa_o = torch.zeros((T, HQ_hsa, D), device=device, dtype=torch.float32)
        lse_kv = torch.full((T, H_hsa), float("-inf"), device=device, dtype=torch.float32)

        if hsa_window <= 0:
            return swa_o, lse_kv

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None or md.token_positions is None or md.token_to_seq_id is None:
            return swa_o, lse_kv

        k_cache_all = pool.get_key_buffer(layer.layer_id)
        v_cache_all = pool.get_value_buffer(layer.layer_id)
        token_to_seq_id = md.token_to_seq_id

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        page_size = int(self.page_size)
        sm_scale = float(getattr(layer, "scaling", 1.0))

        engine_indices = md.engine_indices
        if engine_indices is None:
            return swa_o, lse_kv

        for t in range(T):
            eng_idx = int(engine_indices[t].item())
            b = int(token_to_seq_id[t].item())

            # Chunk-aligned window start using ENGINE index.
            raw_start = eng_idx - hsa_window + 1
            chunk_start = max(0, (raw_start // page_size) * page_size) if raw_start >= 0 else 0

            # Engine indices [chunk_start, eng_idx], excluding LMK slots.
            # LMK exclusion: (engine_idx + 1) % page_size == 0
            tok_pos = torch.arange(chunk_start, eng_idx + 1, device=device, dtype=torch.int64)
            keep = ((tok_pos + 1) % page_size) != 0
            tok_pos = tok_pos[keep]
            if tok_pos.numel() == 0:
                continue

            token_locs = page_table_1[b, tok_pos].to(torch.int64)
            q_hgd = q_hsa[t].view(H_hsa, Gh, D).to(torch.float32)

            lse_per_q = torch.full((H_hsa, Gh), float("-inf"), device=device, dtype=torch.float32)
            for kv_h in range(H_hsa):
                kv_h_global = H_swa + kv_h
                k_win = k_cache_all[token_locs, kv_h_global, :].to(torch.float32)
                v_win = v_cache_all[token_locs, kv_h_global, :].to(torch.float32)
                logits = (q_hgd[kv_h] @ k_win.transpose(0, 1)) * sm_scale
                lse_per_q[kv_h] = torch.logsumexp(logits, dim=-1)
                p = torch.softmax(logits, dim=-1)
                o = p @ v_win
                hq_start = kv_h * Gh
                swa_o[t, hq_start : hq_start + Gh, :] = o

            # Aggregate logsumexp across GQA groups → per-kv-head.
            lse_kv[t] = torch.logsumexp(lse_per_q, dim=-1)

        return swa_o, lse_kv

    def _run_selection_extend(
        self,
        *,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        selection_q: Optional[torch.Tensor] = None,
        page_table_1: torch.Tensor,
        kv_head_offset: int = 0,
        kv_head_count: Optional[int] = None,
        hsa_window: int = 0,
        split_info: Optional[dict] = None,
    ) -> None:
        """Dispatch to batched (production) or reference implementation."""
        if self._USE_EXTEND_REFERENCE:
            return self._run_selection_extend_reference(
                q=q, layer=layer, forward_batch=forward_batch,
                selection_q=selection_q, page_table_1=page_table_1,
                kv_head_offset=kv_head_offset, kv_head_count=kv_head_count,
                hsa_window=hsa_window,
            )
        return self._run_selection_extend_batched(
            q=q, layer=layer, forward_batch=forward_batch,
            selection_q=selection_q, page_table_1=page_table_1,
            kv_head_offset=kv_head_offset, kv_head_count=kv_head_count,
            hsa_window=hsa_window,
            split_info=split_info,
        )

    def _run_selection_extend_batched(
        self,
        *,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        selection_q: Optional[torch.Tensor] = None,
        split_info: Optional[dict] = None,
        page_table_1: torch.Tensor,
        kv_head_offset: int = 0,
        kv_head_count: Optional[int] = None,
        hsa_window: int = 0,
    ) -> None:
        """Memory-efficient extend selection via shared LMK + online_topk_group.

        复用 training 的 online_topk_group kernel 调用方式：
        只 gather 一次共享的 LMK [1, S, H, D]，然后调用
        online_topk_group(B=1, L=T, is_causal=True, q_offset=prefix_len)，
        kernel 内部自动处理 causal mask + head group sum。
        内存从 O(T × C_max × H × D) 降到 O(S × H × D)。
        当前只支持 batch=1（SGLang continuous batching 保证）。
        """
        md = self.forward_metadata
        if md is None:
            return
        if not self._is_hsa_layer(layer.layer_id):
            return

        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if pool is None or not hasattr(pool, "get_key_buffer"):
            return
        if md.token_positions is None or md.token_to_seq_id is None:
            return

        # --- 只支持 batch=1，多序列直接报错暴露问题 ---
        extend_seq_lens = md.extend_seq_lens
        extend_prefix_lens = md.extend_prefix_lens
        assert extend_seq_lens is not None and extend_prefix_lens is not None, \
            "extend_seq_lens and extend_prefix_lens must be set for extend mode"
        B = int(extend_seq_lens.shape[0])
        assert B == 1, (
            f"_run_selection_extend_batched 当前只支持 batch=1，"
            f"但收到 B={B}。SGLang continuous batching 应保证 extend 阶段 B=1。"
        )

        T = int(md.token_positions.shape[0])
        H_sel = int(kv_head_count) if kv_head_count is not None else int(layer.tp_k_head_num)
        page_size = int(self.page_size)
        device = q.device

        k_cache = pool.get_key_buffer(layer.layer_id)  # [num_locs, H_total, D]
        D = int(k_cache.shape[2])

        q_sel = selection_q if selection_q is not None else q
        if q_sel.dim() == 2:
            HQ_sel = q_sel.shape[1] // D
            q_sel_3 = q_sel.view(T, HQ_sel, D)
        else:
            q_sel_3 = q_sel
            HQ_sel = int(q_sel_3.shape[1])

        # --- 计算 prefix_len 和总序列长度 ---
        prefix_len = int(extend_prefix_lens[0].item())
        total_seq_len = prefix_len + int(extend_seq_lens[0].item())  # prefix + extend
        S = total_seq_len // page_size  # 总 LMK chunk 数

        if S == 0:
            md.hsa_ext_selected_page_ids = torch.full(
                (T, H_sel, int(self.hsa_topk)), -1, device=device, dtype=torch.int32
            )
            md.hsa_ext_selected_scores = torch.full(
                (T, H_sel, int(self.hsa_topk)), float("-inf"), device=device, dtype=torch.float32
            )
            return

        # --- 1. 只 gather 一次共享的 LMK [S, H, D] ---
        lmk_positions = torch.arange(S, device=device, dtype=torch.int64) * page_size + (page_size - 1)  # [S]
        lmk_positions = lmk_positions.clamp(max=page_table_1.shape[1] - 1)
        lmk_locs = page_table_1[0, lmk_positions].to(torch.int64)  # [S]
        lmk_keys = k_cache[lmk_locs]  # [S, H_total, D]
        if kv_head_count is not None:
            lmk_keys = lmk_keys[:, int(kv_head_offset):int(kv_head_offset) + int(kv_head_count), :]
        # lmk_keys: [S, H_sel, D]

        # --- 2. unified_retrieval 降维（跟 training 一致）---
        unified = split_info is not None and split_info.get("unified_retrieval", False)
        retrieval_dim = split_info.get("retrieval_dim", None) if split_info is not None else None
        if unified and retrieval_dim is not None:
            tp_size = get_attention_tp_size()
            if tp_size > 1:
                # All-gather KV head landmark keys across TP ranks
                lmk_keys = tensor_model_parallel_all_gather(lmk_keys.contiguous(), dim=1)
                H_sel = lmk_keys.shape[1]  # updated to full H_hsa
                # All-gather sel_q across TP ranks
                q_sel_3 = tensor_model_parallel_all_gather(q_sel_3.contiguous(), dim=-1)
            # 跟 training 一致: rearrange('S H D -> S 1 (H D)')
            lmk_keys = lmk_keys.reshape(S, 1, H_sel * D)  # [S, 1, H_sel*D]
            H_sel_kernel = 1
            D_kernel = H_sel * D  # = retrieval_dim (e.g. 1024)
        else:
            H_sel_kernel = H_sel
            D_kernel = D

        # --- 3. 调用 online_topk_group（跟 training 一致）---
        assert _online_topk_group is not None, (
            "Fused online_topk_group kernel 未加载，无法执行 extend selection。"
            "请确保 hsa-kernel-main/ops/topk_group.py 可正常导入。"
        )

        # Q: [T, HQ_sel, D] → [1, T, HQ_sel, D]
        # 对于 unified_retrieval，HQ_sel=1, D=retrieval_dim
        q_4d = q_sel_3.unsqueeze(0)  # [1, T, HQ_sel, D_sel]
        lmk_4d = lmk_keys.unsqueeze(0)  # [1, S, H_sel_kernel, D_kernel]

        # q_offset = prefix_len，表示 extend 的第一个 token 在全局序列中的位置
        fused_indices, fused_scores = _online_topk_group(
            q=q_4d,
            lmks=lmk_4d,
            topk=int(self.hsa_topk),
            block_size=page_size,
            window_size=hsa_window if hsa_window > 0 else 0,
            is_causal=True,
            q_offset=prefix_len,
            is_training=False,
        )
        # fused_indices: [1, T, h_shared, topk], fused_scores: [1, T, h_shared, topk]

        # --- 4. 转换输出格式 ---
        # squeeze batch dim: [T, h_shared, topk]
        selected_page_ids = fused_indices.squeeze(0).to(torch.int32)
        selected_scores = fused_scores.squeeze(0).to(torch.float32)

        # kernel 返回的 index 是 chunk index（0-based），即 page_id
        # 无效位（-1）保持不变
        selected_page_ids = selected_page_ids.masked_fill(selected_page_ids < 0, -1)

        # 如果 unified_retrieval，h_shared=1，需要 repeat 到 H_sel（原始 kv head 数）
        if unified and retrieval_dim is not None:
            orig_H_sel = int(kv_head_count) if kv_head_count is not None else int(layer.tp_k_head_num)
            if selected_page_ids.shape[1] != orig_H_sel:
                selected_page_ids = selected_page_ids.repeat_interleave(
                    orig_H_sel // selected_page_ids.shape[1], dim=1
                )
                selected_scores = selected_scores.repeat_interleave(
                    orig_H_sel // selected_scores.shape[1], dim=1
                )

        md.hsa_ext_selected_page_ids = selected_page_ids  # [T, H_sel, K]
        md.hsa_ext_selected_scores = selected_scores  # [T, H_sel, K]

    def _run_selection_extend_reference(
        self,
        *,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        selection_q: Optional[torch.Tensor] = None,
        page_table_1: torch.Tensor,
        kv_head_offset: int = 0,
        kv_head_count: Optional[int] = None,
        hsa_window: int = 0,
    ) -> None:
        """Per-token top-k page selection for extend (reference implementation).

        For each query token t at position pos_t in sequence b:
          1. Completed pages before pos_t: page_ids 0..(pos_t // page_size - 1)
          2. Exclude pages within the SWA window (chunk-aligned)
          3. Load LMK keys and score
          4. TopK → selected_page_ids[t, H, K], scores[t, H, K]
        """
        md = self.forward_metadata
        if md is None:
            return
        if not self._is_hsa_layer(layer.layer_id):
            return

        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if pool is None or not hasattr(pool, "get_key_buffer"):
            return
        if md.token_positions is None or md.token_to_seq_id is None:
            return

        T = int(md.token_positions.shape[0])
        H_sel = int(kv_head_count) if kv_head_count is not None else int(layer.tp_k_head_num)
        page_size = int(self.page_size)
        token_positions = md.token_positions
        token_to_seq_id = md.token_to_seq_id
        device = q.device

        k_cache = pool.get_key_buffer(layer.layer_id)  # [num_locs, H_total, D]
        D = int(k_cache.shape[2])
        sm_scale = getattr(layer, "scaling", None)

        q_sel = selection_q if selection_q is not None else q
        if q_sel.dim() == 2:
            HQ_sel = q_sel.shape[1] // D
            q_sel_3 = q_sel.view(T, HQ_sel, D)
        else:
            q_sel_3 = q_sel
            HQ_sel = int(q_sel_3.shape[1])

        all_page_ids = []
        all_scores = []

        _engine_indices = md.engine_indices
        if _engine_indices is None:
            return

        for t in range(T):
            eng_idx = int(_engine_indices[t].item())
            b = int(token_to_seq_id[t].item())

            # Completed pages using engine index.
            completed = eng_idx // page_size
            if completed <= 0:
                all_page_ids.append(
                    torch.full((1, H_sel, self.hsa_topk), -1, device=device, dtype=torch.int32)
                )
                all_scores.append(
                    torch.full((1, H_sel, self.hsa_topk), float("-inf"), device=device, dtype=torch.float32)
                )
                continue

            cand_pages = torch.arange(completed, device=device, dtype=torch.int32)

            # Exclude pages within the SWA window using engine index.
            # Official: limit_chunk = (engine_idx - window_size + 1) // block_size
            if hsa_window > 0:
                raw_start = eng_idx - hsa_window + 1
                limit_chunk = max(0, raw_start // page_size) if raw_start >= 0 else 0
                cand_pages = cand_pages[cand_pages < limit_chunk]

            C = int(cand_pages.numel())
            if C == 0:
                all_page_ids.append(
                    torch.full((1, H_sel, self.hsa_topk), -1, device=device, dtype=torch.int32)
                )
                all_scores.append(
                    torch.full((1, H_sel, self.hsa_topk), float("-inf"), device=device, dtype=torch.float32)
                )
                continue

            # Load LMK keys: K[page_id * page_size + (page_size - 1)].
            lmk_token_pos = cand_pages.to(torch.int64) * page_size + (page_size - 1)
            lmk_locs = page_table_1[b, lmk_token_pos].to(torch.int64)
            lmk_repr = k_cache[lmk_locs]  # [C, H_total, D]
            if kv_head_count is not None:
                lmk_repr = lmk_repr[:, int(kv_head_offset):int(kv_head_offset) + int(kv_head_count), :]
            # lmk_repr: [C, H_sel, D]

            # Use select_topk_pages_decode with batch_size=1.
            cand_page_ids_1 = cand_pages.unsqueeze(0)  # [1, C]
            cand_mask_1 = torch.ones((1, C), device=device, dtype=torch.bool)
            cand_repr_1 = lmk_repr.unsqueeze(0)  # [1, C, H_sel, D]

            sel = select_topk_pages_decode(
                q=q_sel_3[t:t+1],  # [1, HQ_sel, D]
                cand_page_ids=cand_page_ids_1,
                cand_mask=cand_mask_1,
                cand_chunk_repr=cand_repr_1,
                cand_chunk_repr_valid=cand_mask_1,
                topk=int(self.hsa_topk),
                selection_strategy=str(self.hsa_selection_strategy),
                sm_scale=sm_scale,
            )
            all_page_ids.append(sel.selected_page_ids)  # [1, H_sel, K]
            all_scores.append(sel.selected_scores)  # [1, H_sel, K]

        md.hsa_ext_selected_page_ids = torch.cat(all_page_ids, dim=0)  # [T, H_sel, K]
        md.hsa_ext_selected_scores = torch.cat(all_scores, dim=0)  # [T, H_sel, K]

    def _hsa_sparse_attn_extend(
        self,
        *,
        q_hsa: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        selected_page_ids: torch.Tensor,
        hsa_weights: torch.Tensor,
        H_hsa: int,
        HQ_hsa: int,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Dispatch to Triton kernel (production) or reference implementation."""
        if self._USE_EXTEND_REFERENCE:
            return self._hsa_sparse_attn_extend_reference(
                q_hsa=q_hsa, k_cache=k_cache, v_cache=v_cache,
                page_table_1=page_table_1, selected_page_ids=selected_page_ids,
                hsa_weights=hsa_weights, H_hsa=H_hsa, HQ_hsa=HQ_hsa,
                sm_scale=sm_scale,
            )
        # Production path: reuse the decode Triton kernel with token_to_seq_id.
        md = self.forward_metadata
        return hsa_decode_paged_fwd(
            q=q_hsa,                              # [T, HQ_hsa, D]
            k_cache=k_cache,                       # [Nloc, H_hsa, D]
            v_cache=v_cache,                       # [Nloc, H_hsa, D]
            page_table_1=page_table_1,             # [B, MAX_T]
            selected_page_ids=selected_page_ids,   # [T, H_hsa, K]
            hsa_weights=hsa_weights,               # [T, HQ_hsa, K]
            page_size=int(self.page_size),
            sm_scale=sm_scale,
            mask_last_token=True,
            token_to_seq_id=md.token_to_seq_id,    # [T] int32
        )

    def _hsa_sparse_attn_extend_reference(
        self,
        *,
        q_hsa: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        selected_page_ids: torch.Tensor,
        hsa_weights: torch.Tensor,
        H_hsa: int,
        HQ_hsa: int,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Paged HSA sparse attention for extend (per-token, PyTorch reference).

        For each query token t, for each selected page k:
          1. Load PAGE_SIZE tokens from paged cache via page_table_1
          2. softmax(q @ K^T * sm_scale) @ V, with mask_last_token
          3. Multiply by hsa_weights[t, hq, k] and accumulate

        Returns: out [T, HQ_hsa, D] bf16
        """
        T, _, D = q_hsa.shape
        device = q_hsa.device
        TOPK = int(selected_page_ids.shape[2])
        page_size = int(self.page_size)

        md = self.forward_metadata
        token_to_seq_id = md.token_to_seq_id

        if sm_scale is None:
            sm_scale = float(D) ** -0.5

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa

        out = torch.zeros((T, HQ_hsa, D), device=device, dtype=torch.float32)

        for t in range(T):
            b = int(token_to_seq_id[t].item())
            q_t = q_hsa[t].view(H_hsa, Gh, D).to(torch.float32)  # [H_hsa, Gh, D]

            for k_i in range(TOPK):
                page_id = int(selected_page_ids[t, :, k_i].max().item())  # same across H
                if page_id < 0:
                    continue

                # Token indices for this page.
                tok_pos = torch.arange(page_size, device=device, dtype=torch.int64) + page_id * page_size
                token_locs = page_table_1[b, tok_pos].to(torch.int64)

                for kv_h in range(H_hsa):
                    pid = int(selected_page_ids[t, kv_h, k_i].item())
                    if pid < 0:
                        continue
                    w = hsa_weights[t, kv_h * Gh : (kv_h + 1) * Gh, k_i].to(torch.float32)  # [Gh]

                    k_page = k_cache[token_locs, kv_h, :].to(torch.float32)  # [PS, D]
                    v_page = v_cache[token_locs, kv_h, :].to(torch.float32)  # [PS, D]

                    logits = (q_t[kv_h] @ k_page.transpose(0, 1)) * sm_scale  # [Gh, PS]
                    # Mask last token (LMK).
                    logits[:, page_size - 1] = float("-inf")
                    p = torch.softmax(logits, dim=-1)  # [Gh, PS]
                    o_page = p @ v_page  # [Gh, D]

                    hq_start = kv_h * Gh
                    out[t, hq_start : hq_start + Gh, :] += w[:, None] * o_page

        return out.to(torch.bfloat16)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        split_info = kwargs.get("hsa_split_head_info", None)
        selection_q = kwargs.get("hsa_selection_q", None)

        # Non-HSA layers or no split_info: delegate to dense.
        if not self._is_hsa_layer(layer.layer_id) or split_info is None:
            kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
            return self._dense_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs_clean
            )

        md = self.forward_metadata
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if md is None or pool is None:
            kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
            return self._dense_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs_clean
            )

        # Step 1: Save full KV to paged cache (all heads).
        if save_kv_cache:
            pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        # Extract head split info.
        HQ_swa = int(split_info.get("hq_swa", 0))
        HQ_hsa = int(split_info.get("hq_hsa", 0))
        H_swa = int(split_info.get("h_swa", 0))
        H_hsa = int(split_info.get("h_hsa", 0))
        hsa_window = int(split_info.get("swa_window_size", 0) or 0)
        upper_swa_window = int(split_info.get("upper_swa_window_size", 0) or -1)

        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError(
                "HSA extend currently assumes qk_head_dim == v_head_dim."
            )

        # Reshape q to [T, HQ, D].
        HQ_total = int(layer.tp_q_head_num)
        D = int(layer.qk_head_dim)
        T = q.shape[0]
        q3 = q.view(T, HQ_total, D)
        q_hsa = q3[:, HQ_swa:, :]   # [T, HQ_hsa, D]

        # Step 2: Run dense extend on SWA heads ONLY.
        # When HQ_swa == 0 (all heads are HSA, e.g. hsa_denom=1), skip this step.
        H_total = int(layer.tp_k_head_num)

        dense_out_3 = torch.empty((T, HQ_total, D), device=q.device, dtype=q.dtype)

        if HQ_swa > 0 and H_swa > 0:
            # We call the triton kernel directly with head-sliced tensor views,
            # avoiding the `forward_extend` Python-level overhead.
            # The kernel computes `kv_group_num = q.shape[1] // k_buffer.shape[1]`, so
            # passing sliced Q (HQ_swa) and sliced KV buffer (H_swa) gives the correct GQA ratio.
            pool_k = pool.get_key_buffer(layer.layer_id)[:, :H_swa, :]
            pool_v = pool.get_value_buffer(layer.layer_id)[:, :H_swa, :]

            db = self._dense_backend
            dense_md = db.forward_metadata

            k3 = k.view(-1, H_total, D)
            v3 = v.view(-1, H_total, D)

            if db.enable_deterministic:
                # 1-stage unified kernel: uses kv_indices for both prefix + extend
                db.extend_attention_fwd_unified(
                    q3[:, :HQ_swa, :],
                    dense_out_3[:, :HQ_swa, :],
                    pool_k, pool_v,
                    dense_md.qo_indptr,
                    dense_md.kv_indptr,
                    dense_md.kv_indices,
                    forward_batch.extend_prefix_lens.to(torch.int32),
                    dense_md.max_extend_len,
                    sm_scale=layer.scaling,
                    sliding_window_size=upper_swa_window,
                )
            else:
                # 2-stage kernel: uses k_extend/v_extend + k_buffer/v_buffer separately
                kv_indptr = dense_md.kv_indptr
                kv_indices = dense_md.kv_indices
                db.extend_attention_fwd(
                    q3[:, :HQ_swa, :],
                    k3[:, :H_swa, :].contiguous(),
                    v3[:, :H_swa, :].contiguous(),
                    dense_out_3[:, :HQ_swa, :],
                    pool_k, pool_v,
                    dense_md.qo_indptr,
                    kv_indptr, kv_indices,
                    dense_md.custom_mask,
                    True,  # is_causal
                    dense_md.mask_indptr,
                    dense_md.max_extend_len,
                    sm_scale=layer.scaling,
                    sliding_window_size=upper_swa_window,
                )

        page_table_1 = md.page_table_1

        # Step 3: Internal SWA on HSA heads.
        swa_o_inner, lse_kv = self._compute_internal_swa_extend(
            q_hsa=q_hsa,
            layer=layer,
            forward_batch=forward_batch,
            page_table_1=page_table_1,
            H_swa=H_swa,
            H_hsa=H_hsa,
            HQ_hsa=HQ_hsa,
            hsa_window=hsa_window,
        )

        # Step 4: TopK selection per token.
        self._run_selection_extend(
            q=q_hsa,
            layer=layer,
            forward_batch=forward_batch,
            selection_q=selection_q,
            page_table_1=page_table_1,
            kv_head_offset=H_swa,
            kv_head_count=H_hsa,
            hsa_window=hsa_window,
            split_info=split_info,
        )

        selected_page_ids = md.hsa_ext_selected_page_ids  # [T, H_hsa, K]
        selected_scores = md.hsa_ext_selected_scores  # [T, H_hsa, K]

        if selected_page_ids is None or selected_scores is None:
            # No selection results (e.g. all tokens at early positions).
            # HSA heads uninitialized; zero them.
            dense_out_3[:, HQ_swa:, :] = 0
            return dense_out_3.reshape(T, HQ_total * D)

        # Step 5: Merged softmax.
        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        TOPK = int(selected_page_ids.shape[2])

        # unified_retrieval: selection returns [T, 1, K]; expand to [T, H_hsa, K]
        if selected_page_ids.shape[1] == 1 and H_hsa > 1:
            selected_page_ids = selected_page_ids.expand(T, H_hsa, TOPK).contiguous()
            selected_scores = selected_scores.expand(T, H_hsa, TOPK).contiguous()

        valid = selected_page_ids >= 0
        scores = selected_scores.masked_fill(~valid, float("-inf"))

        if hsa_window > 0:
            if not self.enable_softmax1:
                cat_scores = torch.cat(
                    [scores, lse_kv.unsqueeze(-1)], dim=-1
                )  # [T, H_hsa, K+1]
                swa_weight_idx = -1
            else:
                cat_scores = torch.cat(
                    [
                        scores,
                        lse_kv.unsqueeze(-1),
                        torch.zeros(T, H_hsa, 1, device=scores.device, dtype=scores.dtype),
                    ],
                    dim=-1,
                )  # [T, H_hsa, K+2]
                swa_weight_idx = -2
            merged_w = torch.softmax(cat_scores, dim=-1)
            merged_w = torch.nan_to_num(merged_w, nan=0.0)
            w_kv = merged_w[:, :, :TOPK].to(q_hsa.dtype)
            swa_w_kv = merged_w[:, :, swa_weight_idx]  # [T, H_hsa]
        else:
            w_kv = torch.softmax(scores, dim=-1)
            w_kv = torch.nan_to_num(w_kv, nan=0.0).to(q_hsa.dtype)
            swa_w_kv = None

        # Expand weights to q-head space: [T, H_hsa, K] → [T, HQ_hsa, K].
        w_q = (
            w_kv[:, :, None, :]
            .expand(T, H_hsa, Gh, TOPK)
            .reshape(T, HQ_hsa, TOPK)
            .contiguous()
        )

        # Step 6: Paged HSA sparse attention.
        k_cache_hsa = pool.get_key_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :]
        v_cache_hsa = pool.get_value_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :]
        out_hsa = self._hsa_sparse_attn_extend(
            q_hsa=q_hsa,
            k_cache=k_cache_hsa,
            v_cache=v_cache_hsa,
            page_table_1=page_table_1,
            selected_page_ids=selected_page_ids,
            hsa_weights=w_q,
            H_hsa=H_hsa,
            HQ_hsa=HQ_hsa,
            sm_scale=getattr(layer, "scaling", None),
        )  # [T, HQ_hsa, D] bf16

        # Step 7: Weighted SWA fusion (LHSA: o_lower = hsa_o + swa_o * swa_weight).
        if swa_w_kv is not None:
            swa_w_q = (
                swa_w_kv[:, :, None]
                .expand(T, H_hsa, Gh)
                .reshape(T, HQ_hsa)
            )
            out_hsa = out_hsa.to(torch.float32) + swa_o_inner * swa_w_q[:, :, None]

        # Step 8: Write HSA heads into the pre-allocated output tensor.
        dense_out_3[:, HQ_swa:, :] = out_hsa.to(dense_out_3.dtype)
        return dense_out_3.reshape(T, HQ_total * D)
