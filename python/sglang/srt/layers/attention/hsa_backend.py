from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
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

        # Minimal config from CLI (stored for future HSA logic)
        self.hsa_topk = getattr(model_runner.server_args, "hsa_topk", 64)
        self.hsa_selection_strategy = getattr(
            model_runner.server_args, "hsa_selection_strategy", "head"
        )
        self.hsa_layers = getattr(model_runner.server_args, "hsa_layers", None)
        self.hsa_window_size = getattr(model_runner.server_args, "hsa_window_size", None)
        self.hsa_enable_swa_fusion = getattr(
            model_runner.server_args, "hsa_enable_swa_fusion", False
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

    def _is_hsa_layer(self, layer_id: int) -> bool:
        layers = self.hsa_layers
        if layers is None:
            return True
        try:
            return int(layer_id) in set(int(x) for x in layers)
        except Exception:
            return True

    def _get_effective_window_size(self) -> Optional[int]:
        if self.hsa_window_size is not None:
            return int(self.hsa_window_size)
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
        # Step 4: run selection (SWA→HSA) to populate metadata; compute still uses dense backend.
        self._run_selection_decode(q=q, layer=layer, forward_batch=forward_batch)
        # Phase 1: delegate to dense attention backend for correctness/runability.
        # Phase 2+: replace with HSA selection + paged HSA kernel.
        out = self._dense_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )
        return out

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


