from __future__ import annotations

import logging
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

        self.hsa_window_size = getattr(server_args, "hsa_window_size", None)
        # NOTE (terminology): we refer to SWA+HSA combination as "merged" (not "fused") to avoid
        # confusion with a future single-kernel fused implementation.
        default_merging = bool(getattr(cfg, "enable_swa_hsa_merging", False)) or bool(
            getattr(cfg, "use_sliding_window_merging", False)
        )
        override_merging = getattr(server_args, "hsa_enable_swa_merging", None)
        self.hsa_enable_swa_merging = (
            bool(override_merging) if override_merging is not None else bool(default_merging)
        )

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

    def _get_effective_window_size(self) -> Optional[int]:
        if self.hsa_window_size is not None:
            return int(self.hsa_window_size)
        # FlashHSA: allow separate merging window size (distinct from standalone SWA layers).
        get_merging_window = getattr(
            getattr(self.model_runner, "model", None),
            "get_flashhsa_merging_sliding_window_size",
            None,
        )
        if callable(get_merging_window):
            try:
                w = get_merging_window()
                if w is not None:
                    return int(w)
            except Exception:
                pass
        if getattr(self.model_runner, "sliding_window_size", None) is not None:
            return int(self.model_runner.sliding_window_size)
        return None

    def _run_selection_decode(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
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

        window = self._get_effective_window_size()
        # SWA→HSA mode: exclude SWA window pages from candidate set.
        # (If fusion is enabled later, we'd switch to softmax_head logic.)
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
            md.hsa_selected_page_ids = cand_page_ids.new_full(
                (B, layer.tp_k_head_num, self.hsa_topk), -1, dtype=torch.int32
            )
            md.hsa_selected_scores = q.new_full(
                (B, layer.tp_k_head_num, self.hsa_topk), float("-inf"), dtype=torch.float32
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

        k_cache = pool.get_key_buffer(layer.layer_id)  # [num_locs, H, D]
        flat_repr = k_cache[flat_lmk_locs]  # [B*C, H, D]
        cand_repr = flat_repr.view(B, C, flat_repr.shape[1], flat_repr.shape[2])
        cand_repr_valid = cand_completed

        sel = select_topk_pages_decode(
            q=q,
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
        # Step 4: run selection (SWA→HSA) to populate metadata.
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

        selected_page_ids = md.hsa_selected_page_ids  # [B, H, K]
        selected_scores = md.hsa_selected_scores  # [B, H, K]
        H = int(layer.tp_k_head_num)
        HQ = int(layer.tp_q_head_num)
        assert HQ % H == 0
        G = HQ // H
        TOPK = int(selected_page_ids.shape[2])

        # HSA weights / merged weights:
        # - HSA-only: softmax over selected_scores => hsa_weights
        # - SWA→HSA merged: softmax over cat([selected_scores, lse_swa]) => [hsa_weights, swa_weight]
        if not self.hsa_enable_swa_merging:
            valid = selected_page_ids >= 0
            scores = selected_scores.masked_fill(~valid, float("-inf"))
            w_kv = torch.softmax(scores, dim=-1)
            w_kv = torch.nan_to_num(w_kv, nan=0.0).to(q3.dtype)
            w_q = (
                w_kv[:, :, None, :]
                .expand(w_kv.shape[0], H, G, TOPK)
                .reshape(w_kv.shape[0], HQ, TOPK)
                .contiguous()
            )
            swa_weight_q = None
            out_swa = None
        else:
            # NOTE (terminology): we call this "merged" (not "fused") to avoid confusion with a future
            # single-kernel fused implementation.
            window_size = self._get_effective_window_size()
            if window_size is None or int(window_size) <= 0:
                # No SWA branch; fall back to HSA-only.
                valid = selected_page_ids >= 0
                scores = selected_scores.masked_fill(~valid, float("-inf"))
                w_kv = torch.softmax(scores, dim=-1)
                w_kv = torch.nan_to_num(w_kv, nan=0.0).to(q3.dtype)
                w_q = (
                    w_kv[:, :, None, :]
                    .expand(w_kv.shape[0], H, G, TOPK)
                    .reshape(w_kv.shape[0], HQ, TOPK)
                    .contiguous()
                )
                swa_weight_q = None
                out_swa = None
            else:
                # SWA branch for SWA→HSA merged path.
                #
                # Official FlashHSA uses `torch.nn.attention.flex_attention` with a custom block mask:
                #   - local windowed causal
                #   - excludes LMK slots ((kv_idx + 1) % chunk_size != 0)
                #
                # We mirror that here for speed & semantic alignment.
                B = int(q3.shape[0])
                D = int(q3.shape[2])
                k_cache_all = pool.get_key_buffer(layer.layer_id)
                v_cache_all = pool.get_value_buffer(layer.layer_id)

                out_swa = torch.zeros((B, HQ, D), device=q3.device, dtype=torch.float32)
                lse_swa = torch.full((B, HQ), float("-inf"), device=q3.device, dtype=torch.float32)

                try:
                    from torch.nn.attention.flex_attention import create_block_mask
                    flex_attn = _get_compiled_flex_attention()

                    seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
                    for b in range(B):
                        seqlen = int(seq_lens_i64[b].item())
                        if seqlen <= 0:
                            continue

                        q_idx_global = seqlen - 1
                        start_global = q_idx_global - int(window_size) + 1
                        if start_global < 0:
                            start_global = 0
                        chunk_start = (start_global // int(self.page_size)) * int(self.page_size)

                        # Build a fixed-length KV window [window_size] by clamping positions into [0, seqlen-1].
                        kv_pos_global = start_global + torch.arange(
                            int(window_size), device=q3.device, dtype=torch.int64
                        )  # [W]
                        kv_pos_clamped = kv_pos_global.clamp_min(0).clamp_max(seqlen - 1)
                        token_locs = page_table_1[b, kv_pos_clamped].to(torch.int64)  # [W]

                        # Gather KV: [W, H, D] -> [1, H, W, D]
                        k_win = k_cache_all[token_locs].transpose(0, 1).unsqueeze(0)  # [1,H,W,D]
                        v_win = v_cache_all[token_locs].transpose(0, 1).unsqueeze(0)  # [1,H,W,D]
                        q_b = q3[b].unsqueeze(0).unsqueeze(2)  # [1,HQ,1,D]

                        def block_causal_mask(_bb, _hh, _q_idx, kv_idx):
                            # Map kv_idx (0..W-1) into global token index.
                            kv_global = kv_idx + start_global
                            # windowed + chunk aligned + exclude LMK slots
                            return (
                                (kv_global >= chunk_start)
                                & (kv_global <= q_idx_global)
                                & (((kv_global + 1) % int(self.page_size)) != 0)
                            )

                        block_mask = create_block_mask(
                            block_causal_mask, B=None, H=None, Q_LEN=1, KV_LEN=int(window_size)
                        )
                        # Newer PyTorch uses return_aux; keep a backward-compatible fallback.
                        try:
                            from torch.nn.attention.flex_attention import AuxRequest

                            o, aux = flex_attn(
                                q_b,
                                k_win,
                                v_win,
                                block_mask=block_mask,
                                enable_gqa=True,
                                return_aux=AuxRequest(lse=True),
                            )
                            lse = aux.lse
                        except Exception:
                            o, lse = flex_attn(
                                q_b,
                                k_win,
                                v_win,
                                block_mask=block_mask,
                                enable_gqa=True,
                                return_lse=True,
                            )  # o: [1,HQ,1,D], lse: [1,HQ,1]

                        out_swa[b] = o.squeeze(0).squeeze(1).to(torch.float32)
                        lse_swa[b] = lse.squeeze(0).squeeze(1).to(torch.float32)
                except Exception:
                    # Fallback to a simple torch implementation (kept for robustness across torch versions).
                    global _FLEX_FALLBACK_WARNED
                    if not _FLEX_FALLBACK_WARNED:
                        _FLEX_FALLBACK_WARNED = True
                        logger.warning(
                            "FlashHSA SWA branch: flex_attention failed; falling back to slow torch impl. "
                            "This may be much slower. Set SGLANG_HSA_TEST_VERBOSE=1 for more context."
                        )
                    seq_lens_i64 = md.cache_seqlens_int32.to(torch.int64)
                    for b in range(B):
                        seqlen = int(seq_lens_i64[b].item())
                        if seqlen <= 0:
                            continue
                        w = min(seqlen, int(window_size))
                        start = seqlen - w
                        tok_pos = torch.arange(start, seqlen, device=q3.device, dtype=torch.int64)
                        keep = (tok_pos % int(self.page_size)) != (int(self.page_size) - 1)
                        tok_pos = tok_pos[keep]
                        if tok_pos.numel() == 0:
                            continue
                        token_locs = page_table_1[b, tok_pos].to(torch.int64)

                        q_hgd = q3[b].view(H, G, D).to(torch.float32)
                        for kv_h in range(H):
                            k_win = k_cache_all[token_locs, kv_h, :].to(torch.float32)
                            v_win = v_cache_all[token_locs, kv_h, :].to(torch.float32)
                            logits = (q_hgd[kv_h] @ k_win.transpose(0, 1)) * float(
                                getattr(layer, "scaling", 1.0)
                            )
                            lse = torch.logsumexp(logits, dim=-1)
                            p = torch.softmax(logits, dim=-1)
                            o = p @ v_win
                            hq_start = kv_h * G
                            out_swa[b, hq_start : hq_start + G, :] = o
                            lse_swa[b, hq_start : hq_start + G] = lse

                out_swa = out_swa.to(torch.bfloat16)

                # Expand selected_scores from kv-head space [B,H,K] to q-head space [B,HQ,K].
                scores_kv = selected_scores.to(torch.float32)
                valid = (selected_page_ids >= 0)
                scores_kv = scores_kv.masked_fill(~valid, float("-inf"))
                scores_q = (
                    scores_kv[:, :, None, :]
                    .expand(scores_kv.shape[0], H, G, TOPK)
                    .reshape(scores_kv.shape[0], HQ, TOPK)
                    .contiguous()
                )

                cat = torch.cat([scores_q, lse_swa.unsqueeze(-1)], dim=-1)  # [B,HQ,K+1]
                w_all = torch.softmax(cat, dim=-1)
                w_all = torch.nan_to_num(w_all, nan=0.0)
                w_q = w_all[:, :, :TOPK].to(q3.dtype).contiguous()
                swa_weight_q = w_all[:, :, TOPK].to(q3.dtype).contiguous()

        out3 = hsa_decode_paged_fwd(
            q=q3,
            k_cache=pool.get_key_buffer(layer.layer_id),
            v_cache=pool.get_value_buffer(layer.layer_id),
            page_table_1=page_table_1,
            selected_page_ids=selected_page_ids,
            hsa_weights=w_q,
            page_size=int(self.page_size),
            sm_scale=getattr(layer, "scaling", None),
            mask_last_token=True,
        )
        if swa_weight_q is not None and out_swa is not None:
            out3 = out3 + (swa_weight_q[:, :, None] * out_swa.to(out3.dtype))
        return out3.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

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
        out = self._dense_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )
        return out


