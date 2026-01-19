from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class HSAAttnBackend(AttentionBackend):
    """
    HSA (Hierarchical Sparse Attention) backend entrypoint.

    This is currently a stub to enable CLI/registry wiring:
      --attention-backend hsa

    The actual implementation will follow `dev/hsa_sglang_impl_todo.md` and
    `dev/hsa_dev_roadmap.md` (paged-KV-first, AttentionBackend + metadata).
    """

    def __init__(self, model_runner: ModelRunner, **_kwargs):
        super().__init__()
        self.model_runner = model_runner

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        raise NotImplementedError(
            "HSAAttnBackend is not implemented yet. "
            "See dev/hsa_sglang_impl_todo.md and dev/hsa_dev_roadmap.md."
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **_kwargs,
    ):
        raise NotImplementedError(
            "HSAAttnBackend.forward_decode is not implemented yet. "
            "See dev/hsa_sglang_impl_todo.md."
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **_kwargs,
    ):
        raise NotImplementedError(
            "HSAAttnBackend.forward_extend is not implemented yet. "
            "See dev/hsa_sglang_impl_todo.md."
        )


