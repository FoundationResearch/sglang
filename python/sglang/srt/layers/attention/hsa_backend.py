from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hsa.metadata import HSAMetadata
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

    def _maybe_save_chunk_repr_for_completed_pages(
        self,
        *,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ) -> None:
        """Phase-1 hook: write a placeholder per-page chunk repr on page completion.

        Contract: only write repr for pages that are fully completed, i.e.
        a token that makes (seq_len % page_size == 0).

        Implementation (placeholder): use the K vector of the boundary token
        as the chunk representation. This will be replaced by the real HSA
        landmark/chunk repr computation later.
        """
        if not save_kv_cache:
            return

        pool = getattr(forward_batch, "token_to_kv_pool", None)
        if pool is None or not hasattr(pool, "save_chunk_repr"):
            return

        if getattr(pool, "page_size", self.page_size) != self.page_size:
            raise ValueError(
                f"HSA contract violated: token_to_kv_pool.page_size != model_runner.page_size "
                f"({getattr(pool, 'page_size', None)} vs {self.page_size})"
            )

        # Decode: one new token per sequence.
        if forward_batch.extend_seq_lens is None or forward_batch.extend_start_loc is None:
            seq_lens = forward_batch.seq_lens[: forward_batch.batch_size].to(torch.int64)
            completed = (seq_lens % self.page_size) == 0
            if not bool(completed.any()):
                return

            token_locs = forward_batch.out_cache_loc[: forward_batch.batch_size].to(
                torch.int64
            )
            token_locs = token_locs[completed]
            page_ids = token_locs // self.page_size

            k_cache = pool.get_key_buffer(layer.layer_id)
            repr = k_cache[token_locs]
            pool.save_chunk_repr(layer.layer_id, page_ids=page_ids, repr=repr)
            return

        # Extend: potentially multiple new tokens per sequence.
        # Identify boundary tokens inside the extend segment: positions where (prefix_len + t) % page_size == 0.
        extend_prefix_lens = forward_batch.extend_prefix_lens
        extend_seq_lens = forward_batch.extend_seq_lens
        extend_start_loc = forward_batch.extend_start_loc
        if extend_prefix_lens is None or extend_seq_lens is None or extend_start_loc is None:
            return

        token_locs_list = []
        for i in range(int(forward_batch.batch_size)):
            pre = int(extend_prefix_lens[i].item())
            ext = int(extend_seq_lens[i].item())
            if ext <= 0:
                continue
            # t in [1..ext] where pre+t hits a multiple of page_size.
            first_t = self.page_size - (pre % self.page_size)
            if first_t == 0:
                first_t = self.page_size
            if first_t > ext:
                continue
            for t in range(first_t, ext + 1, self.page_size):
                idx = int(extend_start_loc[i].item()) + (t - 1)
                token_locs_list.append(forward_batch.out_cache_loc[idx].to(torch.int64))

        if not token_locs_list:
            return

        token_locs = torch.stack(token_locs_list, dim=0).to(torch.int64)
        page_ids = token_locs // self.page_size

        k_cache = pool.get_key_buffer(layer.layer_id)
        repr = k_cache[token_locs]
        pool.save_chunk_repr(layer.layer_id, page_ids=page_ids, repr=repr)

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
        # Phase 1: delegate to dense attention backend for correctness/runability.
        # Phase 2+: replace with HSA selection + paged HSA kernel.
        out = self._dense_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )
        self._maybe_save_chunk_repr_for_completed_pages(
            layer=layer, forward_batch=forward_batch, save_kv_cache=save_kv_cache
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
        self._maybe_save_chunk_repr_for_completed_pages(
            layer=layer, forward_batch=forward_batch, save_kv_cache=save_kv_cache
        )
        return out


