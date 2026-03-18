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
    build_active_page_candidates,
    select_topk_pages_decode,
)
from sglang.srt.layers.attention.hsa.utils import transform_page_table_1_to_real
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

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
    _COMPILED_FLEX_ATTN = torch.compile(flex_attention, dynamic=False)
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
                if default_layers:
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
    ) -> None:
        """Step 4 (Torch reference): SWA→HSA selection for decode.

        For now this populates debug fields on HSAMetadata; compute is still delegated
        to dense backend. This keeps future kernel integration straightforward.
        """
        md = self.forward_metadata
        if md is None:
            return
        if not self._is_hsa_layer(layer.layer_id):
            return

        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if pool is None or not hasattr(pool, "get_key_buffer"):
            return

        window = 0
        if window_size_override is not None:
            window = int(window_size_override)
        # InnerX ultra: selection candidates are built without window exclusion (window=0).
        # Robustness: in decode, some codepaths/tests may not have updated req_to_token yet.
        # Overlay out_cache_loc into a temporary token->slot table at position (seq_len - 1).
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

        cand_page_ids, cand_mask = build_active_page_candidates(
            page_table_1=page_table_1,
            seq_lens=md.cache_seqlens_int32,
            page_size=md.page_size,
            window_size=window,
        )

        # Flatten candidates for repr lookup.
        B, C = cand_page_ids.shape
        if C == 0:
            # Still populate for observability.
            md.hsa_cand_page_ids = cand_page_ids
            md.hsa_cand_mask = cand_mask
            H_sel = int(kv_head_count) if kv_head_count is not None else int(layer.tp_k_head_num)
            md.hsa_selected_page_ids = cand_page_ids.new_full(
                (B, H_sel, self.hsa_topk), -1, dtype=torch.int32
            )
            md.hsa_selected_scores = q.new_full(
                (B, H_sel, self.hsa_topk), float("-inf"), dtype=torch.float32
            )
            return

        # FlashHSA semantics:
        # E_i is K(LMK) for each page, where LMK is the last slot in a page:
        #   lmk_loc = page_id * page_size + (page_size - 1)
        #
        # Only *completed* pages (LMK already written into KV) may participate.
        seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
        completed_pages = torch.div(
            seq_lens_i64, int(self.page_size), rounding_mode="floor"
        )  # [B]

        cand_completed = cand_mask & (cand_page_ids.to(torch.int64) < completed_pages[:, None])

        safe_page_ids = cand_page_ids.clamp_min(0).to(torch.int64)
        lmk_locs = safe_page_ids * int(self.page_size) + (int(self.page_size) - 1)  # [B,C]
        flat_lmk_locs = lmk_locs.reshape(-1)

        k_cache = pool.get_key_buffer(layer.layer_id)  # [num_locs, H_total, D]
        flat_repr = k_cache[flat_lmk_locs]  # [B*C, H_total, D]
        if kv_head_count is not None:
            flat_repr = flat_repr[:, int(kv_head_offset) : int(kv_head_offset) + int(kv_head_count), :]
        cand_repr = flat_repr.view(B, C, flat_repr.shape[1], flat_repr.shape[2])
        cand_repr_valid = cand_completed

        q_sel = selection_q if selection_q is not None else q
        sel = select_topk_pages_decode(
            q=q_sel,
            cand_page_ids=cand_page_ids,
            cand_mask=cand_mask,
            cand_chunk_repr=cand_repr,
            cand_chunk_repr_valid=cand_repr_valid,
            topk=int(self.hsa_topk),
            selection_strategy=str(self.hsa_selection_strategy),
            sm_scale=getattr(layer, "scaling", None),
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

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # First, initialize dense backend metadata so decode/extend remains runnable.
        self._dense_backend.init_forward_metadata(forward_batch)

        # Build HSA metadata scaffold (paged-KV-first).
        # We intentionally mirror the "page_table_1 -> real_page_table" idea used by NSA.
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

        # Build extend-specific fields when in extend mode.
        token_positions = None
        token_to_seq_id = None
        extend_seq_lens = None
        extend_prefix_lens = None
        if forward_batch.forward_mode.is_extend():
            token_positions = forward_batch.positions  # [total_extend_tokens]
            extend_seq_lens = forward_batch.extend_seq_lens  # [B] int32
            extend_prefix_lens = forward_batch.extend_prefix_lens  # [B] int32
            if extend_seq_lens is not None:
                token_to_seq_id = torch.repeat_interleave(
                    torch.arange(len(extend_seq_lens), device=self.device, dtype=torch.int32),
                    extend_seq_lens.to(torch.int64),
                )

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
            token_positions=token_positions,
            token_to_seq_id=token_to_seq_id,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
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
            if int(selected_page_ids.shape[1]) != H_hsa:
                raise ValueError(
                    f"Selection H mismatch: got {int(selected_page_ids.shape[1])}, expected H_hsa={H_hsa}"
                )

            B = int(q3.shape[0])
            D = int(q3.shape[2])
            q_swa = q3[:, :HQ_swa, :]
            q_hsa = q3[:, HQ_swa:, :]

            # SWA branch (best-effort flex_attention, optional LMK exclusion).
            out_swa = torch.zeros((B, HQ_swa, D), device=q3.device, dtype=torch.float32)
            if window_size is not None and int(window_size) > 0 and HQ_swa > 0 and H_swa > 0:
                k_cache_all = pool.get_key_buffer(layer.layer_id)
                v_cache_all = pool.get_value_buffer(layer.layer_id)
                try:
                    from torch.nn.attention.flex_attention import create_block_mask

                    flex_attn = _get_compiled_flex_attention()
                    seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
                    for b in range(B):
                        seqlen = int(seq_lens_i64[b].item())
                        if seqlen <= 0:
                            continue
                        q_idx_global = seqlen - 1
                        start_global = max(0, q_idx_global - int(window_size) + 1)
                        kv_pos_global = start_global + torch.arange(
                            int(window_size), device=q3.device, dtype=torch.int64
                        )
                        kv_pos_clamped = kv_pos_global.clamp_min(0).clamp_max(seqlen - 1)
                        token_locs = page_table_1[b, kv_pos_clamped].to(torch.int64)
                        k_win = (
                            k_cache_all[token_locs, :H_swa, :].transpose(0, 1).unsqueeze(0)
                        )  # [1,H_swa,W,D]
                        v_win = (
                            v_cache_all[token_locs, :H_swa, :].transpose(0, 1).unsqueeze(0)
                        )  # [1,H_swa,W,D]
                        q_b = q_swa[b].unsqueeze(0).unsqueeze(2)  # [1,HQ_swa,1,D]

                        def block_causal_mask(_bb, _hh, _q_idx, kv_idx):
                            kv_global = kv_idx + start_global
                            ok = (kv_global >= start_global) & (kv_global <= q_idx_global)
                            if swa_exclude_lmk:
                                ok = ok & (((kv_global + 1) % int(self.page_size)) != 0)
                            return ok

                        block_mask = create_block_mask(
                            block_causal_mask, B=None, H=None, Q_LEN=1, KV_LEN=int(window_size)
                        )
                        o = flex_attn(
                            q_b,
                            k_win,
                            v_win,
                            block_mask=block_mask,
                            enable_gqa=True,
                        )[0]
                        out_swa[b] = o.squeeze(0).squeeze(1).to(torch.float32)
                except Exception:
                    # Slow torch fallback (kept small; correctness-first).
                    seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
                    assert HQ_swa % H_swa == 0
                    Gs = HQ_swa // H_swa
                    k_cache_all = pool.get_key_buffer(layer.layer_id)
                    v_cache_all = pool.get_value_buffer(layer.layer_id)
                    for b in range(B):
                        seqlen = int(seq_lens_i64[b].item())
                        if seqlen <= 0:
                            continue
                        w = min(seqlen, int(window_size))
                        start = seqlen - w
                        tok_pos = torch.arange(start, seqlen, device=q3.device, dtype=torch.int64)
                        if swa_exclude_lmk:
                            keep = (tok_pos % int(self.page_size)) != (int(self.page_size) - 1)
                            tok_pos = tok_pos[keep]
                        if tok_pos.numel() == 0:
                            continue
                        token_locs = page_table_1[b, tok_pos].to(torch.int64)
                        q_hgd = q_swa[b].view(H_swa, Gs, D).to(torch.float32)
                        for kv_h in range(H_swa):
                            k_win = k_cache_all[token_locs, kv_h, :].to(torch.float32)
                            v_win = v_cache_all[token_locs, kv_h, :].to(torch.float32)
                            logits = (q_hgd[kv_h] @ k_win.transpose(0, 1)) * float(
                                getattr(layer, "scaling", 1.0)
                            )
                            p = torch.softmax(logits, dim=-1)
                            o = p @ v_win
                            hq_start = kv_h * Gs
                            out_swa[b, hq_start : hq_start + Gs, :] = o

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
        """Batched internal SWA on HSA heads for extend (vectorized over tokens).

        Eliminates the per-token Python loop by padding windows to a uniform
        size and using batched matmuls.

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
        token_positions = md.token_positions  # [T] int64
        token_to_seq_id = md.token_to_seq_id  # [T] int32

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        page_size = int(self.page_size)
        sm_scale = float(getattr(layer, "scaling", 1.0))

        # Compute per-token chunk-aligned window start.
        raw_starts = token_positions - hsa_window + 1  # [T]
        chunk_starts = torch.where(
            raw_starts >= 0,
            (raw_starts // page_size) * page_size,
            torch.zeros_like(raw_starts),
        ).clamp(min=0)  # [T]

        # Window length (before LMK exclusion).
        win_lens = token_positions - chunk_starts + 1  # [T]
        max_win = int(win_lens.max().item())
        if max_win <= 0:
            return swa_o, lse_kv

        # Build padded window positions [T, max_win].
        offsets = torch.arange(max_win, device=device, dtype=torch.int64)  # [max_win]
        win_positions = chunk_starts.unsqueeze(1) + offsets.unsqueeze(0)  # [T, max_win]

        # Validity: position <= pos_t AND not LMK position.
        valid = (win_positions <= token_positions.unsqueeze(1))
        valid = valid & ((win_positions % page_size) != (page_size - 1))

        # Gather token_locs via page_table_1[seq_id, position].
        safe_positions = win_positions.clamp(min=0, max=page_table_1.shape[1] - 1)
        pt_per_token = page_table_1[token_to_seq_id.long()]  # [T, MAX_T]
        token_locs = torch.gather(
            pt_per_token, 1, safe_positions
        ).to(torch.int64)  # [T, max_win]

        # Expand valid mask for GQA groups: [T, Gh, max_win].
        valid_expanded = valid.unsqueeze(1).expand(T, Gh, max_win)

        lse_per_q = torch.full((T, H_hsa, Gh), float("-inf"), device=device, dtype=torch.float32)

        flat_locs = token_locs.reshape(-1)  # [T * max_win]

        for kv_h in range(H_hsa):
            kv_h_global = H_swa + kv_h
            # Gather K, V: [T * max_win, D] -> [T, max_win, D]
            k_win = k_cache_all[flat_locs, kv_h_global, :].view(T, max_win, D).to(torch.float32)
            v_win = v_cache_all[flat_locs, kv_h_global, :].view(T, max_win, D).to(torch.float32)

            # q for this kv_head group: [T, Gh, D]
            q_group = q_hsa[:, kv_h * Gh : (kv_h + 1) * Gh, :].to(torch.float32)

            # Batched attention: logits = [T, Gh, max_win]
            logits = torch.bmm(q_group, k_win.transpose(1, 2)) * sm_scale

            # Mask invalid positions.
            logits = logits.masked_fill(~valid_expanded, float("-inf"))

            # LSE per query head group.
            lse_per_q[:, kv_h, :] = torch.logsumexp(logits, dim=-1)  # [T, Gh]

            # Softmax + weighted V.
            p = torch.softmax(logits, dim=-1)  # [T, Gh, max_win]
            o = torch.bmm(p, v_win)  # [T, Gh, D]
            swa_o[:, kv_h * Gh : (kv_h + 1) * Gh, :] = o

        # Aggregate logsumexp across GQA groups -> per-kv-head.
        lse_kv = torch.logsumexp(lse_per_q, dim=-1)  # [T, H_hsa]

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
        token_positions = md.token_positions
        token_to_seq_id = md.token_to_seq_id

        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        page_size = int(self.page_size)
        sm_scale = float(getattr(layer, "scaling", 1.0))

        for t in range(T):
            pos_t = int(token_positions[t].item())
            b = int(token_to_seq_id[t].item())

            # Chunk-aligned window start (same as decode _compute_internal_swa_decode).
            raw_start = pos_t - hsa_window + 1
            chunk_start = max(0, (raw_start // page_size) * page_size) if raw_start >= 0 else 0

            # Positions [chunk_start, pos_t], excluding LMK slots.
            tok_pos = torch.arange(chunk_start, pos_t + 1, device=device, dtype=torch.int64)
            keep = (tok_pos % page_size) != (page_size - 1)
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
        )

    def _run_selection_extend_batched(
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
        """Batched per-token top-k page selection for extend.

        Vectorizes candidate building, LMK key gathering, and scoring
        across all T tokens in a single call to select_topk_pages_decode.
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
        token_positions = md.token_positions  # [T] int64
        token_to_seq_id = md.token_to_seq_id  # [T] int32
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

        # 1. Compute per-token completed page count and exclusion boundary.
        completed_pages = token_positions // page_size  # [T]

        if hsa_window > 0:
            raw_starts = token_positions - hsa_window + 1
            chunk_starts = torch.where(
                raw_starts >= 0,
                (raw_starts // page_size) * page_size,
                torch.zeros_like(raw_starts),
            ).clamp(min=0)
            window_start_pages = chunk_starts // page_size  # [T]
            # Effective candidate count: pages before the SWA window.
            effective_cands = torch.min(completed_pages, window_start_pages)
        else:
            effective_cands = completed_pages

        effective_cands = effective_cands.clamp(min=0)
        C_max = int(effective_cands.max().item())

        if C_max == 0:
            md.hsa_ext_selected_page_ids = torch.full(
                (T, H_sel, int(self.hsa_topk)), -1, device=device, dtype=torch.int32
            )
            md.hsa_ext_selected_scores = torch.full(
                (T, H_sel, int(self.hsa_topk)), float("-inf"), device=device, dtype=torch.float32
            )
            return

        # 2. Build padded candidate page_ids [T, C_max].
        cand_range = torch.arange(C_max, device=device, dtype=torch.int32)
        cand_page_ids = cand_range.unsqueeze(0).expand(T, C_max).contiguous()  # [T, C_max]
        cand_mask = cand_page_ids < effective_cands.unsqueeze(1).to(torch.int32)  # [T, C_max]
        cand_page_ids = cand_page_ids.masked_fill(~cand_mask, -1)

        # 3. Gather LMK keys for all candidates.
        safe_page_ids = cand_page_ids.clamp(min=0).to(torch.int64)
        lmk_token_pos = safe_page_ids * page_size + (page_size - 1)  # [T, C_max]

        # Map to storage locations via page_table_1[seq_id, lmk_pos].
        pt_per_token = page_table_1[token_to_seq_id.long()]  # [T, MAX_T]
        lmk_token_pos_safe = lmk_token_pos.clamp(max=pt_per_token.shape[1] - 1)
        lmk_locs = torch.gather(
            pt_per_token, 1, lmk_token_pos_safe
        ).to(torch.int64)  # [T, C_max]

        flat_locs = lmk_locs.reshape(-1)
        cand_repr = k_cache[flat_locs]  # [T*C_max, H_total, D]
        if kv_head_count is not None:
            cand_repr = cand_repr[:, int(kv_head_offset):int(kv_head_offset) + int(kv_head_count), :]
        cand_repr = cand_repr.view(T, C_max, H_sel, D)  # [T, C_max, H_sel, D]

        # 4. Single batched selection call with B=T.
        sel = select_topk_pages_decode(
            q=q_sel_3,
            cand_page_ids=cand_page_ids,
            cand_mask=cand_mask,
            cand_chunk_repr=cand_repr,
            cand_chunk_repr_valid=cand_mask,
            topk=int(self.hsa_topk),
            selection_strategy=str(self.hsa_selection_strategy),
            sm_scale=sm_scale,
        )

        md.hsa_ext_selected_page_ids = sel.selected_page_ids  # [T, H_sel, K]
        md.hsa_ext_selected_scores = sel.selected_scores  # [T, H_sel, K]

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

        for t in range(T):
            pos_t = int(token_positions[t].item())
            b = int(token_to_seq_id[t].item())

            # Completed pages: page ids 0 .. (pos_t // page_size - 1).
            completed = pos_t // page_size
            if completed <= 0:
                all_page_ids.append(
                    torch.full((1, H_sel, self.hsa_topk), -1, device=device, dtype=torch.int32)
                )
                all_scores.append(
                    torch.full((1, H_sel, self.hsa_topk), float("-inf"), device=device, dtype=torch.float32)
                )
                continue

            cand_pages = torch.arange(completed, device=device, dtype=torch.int32)

            # Exclude pages within the SWA window (chunk-aligned).
            if hsa_window > 0:
                raw_start = pos_t - hsa_window + 1
                chunk_start = max(0, (raw_start // page_size) * page_size) if raw_start >= 0 else 0
                window_start_page = chunk_start // page_size
                cand_pages = cand_pages[cand_pages < window_start_page]

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

        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError(
                "HSA extend currently assumes qk_head_dim == v_head_dim."
            )

        # Reshape q to [T, HQ, D].
        HQ_total = int(layer.tp_q_head_num)
        D = int(layer.qk_head_dim)
        T = q.shape[0]
        q3 = q.view(T, HQ_total, D)
        q_hsa = q3[:, HQ_swa:, :]  # [T, HQ_hsa, D]

        # Step 2: Run dense extend on ALL heads (save_kv_cache=False since already saved).
        # SWA head outputs will be correct from this; HSA head outputs will be overwritten.
        kwargs_clean = {kk: vv for kk, vv in kwargs.items() if not kk.startswith("hsa_")}
        dense_out = self._dense_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=False, **kwargs_clean
        )
        dense_out_3 = dense_out.view(T, HQ_total, D)

        # page_table_1 already covers all positions (seq_lens = prefix + extend).
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
        )

        selected_page_ids = md.hsa_ext_selected_page_ids  # [T, H_hsa, K]
        selected_scores = md.hsa_ext_selected_scores  # [T, H_hsa, K]

        if selected_page_ids is None or selected_scores is None:
            # No selection results (e.g. all tokens at early positions) → return dense output.
            return dense_out

        # Step 5: Merged softmax.
        assert HQ_hsa % H_hsa == 0
        Gh = HQ_hsa // H_hsa
        TOPK = int(selected_page_ids.shape[2])

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

        # Step 8: Assemble final output — keep SWA heads from dense, overwrite HSA heads.
        out_all = dense_out_3.clone()
        out_all[:, HQ_swa:, :] = out_hsa.to(torch.bfloat16)
        return out_all.reshape(T, HQ_total * D)


