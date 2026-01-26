"""
FlashHSA model definition for SGLang.

This provides a native SGLang model entry class named `HSAForCausalLM`, matching
FlashHSA config.json:
  - "architectures": ["HSAForCausalLM"]
  - "model_type": "flash_hsa"

Key FlashHSA semantic differences we must respect at load time:
  - Landmark token (LMK) is a *real* token id. FlashHSA uses `lmk_id == vocab_size`
    and allocates embeddings with `next_of_y(vocab_size + 1, 32)` rows.
  - We keep `config.vocab_size` as the *base* vocab size (so logits/sampling and
    stop conditions behave as usual), but allocate embed/lm_head weights using the
    padded vocab size so LMK has a valid embedding row and weights can be loaded.
  - Sliding-window size used for the fused SWA/HSA path is renamed to
    `sliding_window_fusion_size` (fallback to `sliding_window` for compatibility).

NOTE: The actual fused SWA/HSA kernels are implemented by attention backends.
This file focuses on model construction + weight loading compatibility.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models.qwen3 import Qwen3ForCausalLM as _Qwen3ForCausalLM
from sglang.srt.models.qwen3 import Qwen3Model as _Qwen3Model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def next_of_y(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _get_sliding_window_fusion_size(config) -> Optional[int]:
    # New name (requested): sliding_window_fusion_size
    if hasattr(config, "sliding_window_fusion_size") and getattr(
        config, "sliding_window_fusion_size"
    ) is not None:
        return int(getattr(config, "sliding_window_fusion_size"))
    return None


def _get_flashhsa_padded_vocab_size(config) -> int:
    base_vocab_size = int(getattr(config, "vocab_size"))
    # FlashHSA: lmk_id == base_vocab_size, so embeddings must have >= base_vocab_size + 1 rows.
    return int(next_of_y(base_vocab_size + 1, 32))


class FlashHSAModel(_Qwen3Model):
    """Qwen3-like model, but with FlashHSA padded vocab embeddings."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        # Build the standard Qwen3 model first, then replace embeddings with the padded size.
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

        padded_vocab_size = _get_flashhsa_padded_vocab_size(config)

        # Replace embedding layer on the first PP rank only, mirroring Qwen2Model/Qwen3Model behavior.
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                padded_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
                params_dtype=(
                    torch.float32
                    if get_global_server_args().rl_on_policy_target is not None
                    else None
                ),
            )


class HSAForCausalLM(_Qwen3ForCausalLM):
    """
    FlashHSA entry class for SGLang.

    We inherit Qwen3ForCausalLM to reuse:
      - logits processor / pooler plumbing
      - Qwen3 weight loading remaps
      - PP weight tying logic (we override sizes where necessary)
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        # We cannot call super().__init__ directly because it will build lm_head with base vocab size.
        nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        # Build model with padded vocab embeddings (LMK row exists).
        self.model = FlashHSAModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        padded_vocab_size = _get_flashhsa_padded_vocab_size(config)

        # handle the lm head on different pp ranks (same as Qwen3ForCausalLM, but padded vocab size)
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    padded_vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP (same as Qwen3ForCausalLM, but padded vocab size)
        if self.pp_group.world_size > 1 and getattr(config, "tie_word_embeddings", False):
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.world_size - 1
                )
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=self.lm_head.weight.shape,
                    dtype=next(self.model.parameters()).dtype,
                    src=0,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        # Reuse Qwen3ForCausalLM components.
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.pooler import Pooler, PoolingType

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support (kept for parity)
        self.capture_aux_hidden_states = False

    # ---- FlashHSA-specific helpers for engine ----

    def get_attention_sliding_window_size(self) -> Optional[int]:
        # This value is used by ModelRunner to configure SWA window indices for backends.
        use_sw = bool(getattr(self.config, "use_sliding_window", False))
        if not use_sw:
            return None
        return _get_sliding_window_fusion_size(self.config)

    def get_flashhsa_hsa_layer_ids(self) -> List[int]:
        """Default per-layer HSA pattern from FlashHSA: every `full_attn_interleave`-th layer."""
        interleave = int(getattr(self.config, "full_attn_interleave", 0) or 0)
        num_layers = int(getattr(self.config, "num_hidden_layers", 0) or 0)
        if interleave <= 0 or num_layers <= 0:
            return []
        return [i for i in range(num_layers) if (i % interleave) == (interleave - 1)]

    def get_flashhsa_chunk_size(self) -> Optional[int]:
        if hasattr(self.config, "chunk_size") and getattr(self.config, "chunk_size") is not None:
            return int(getattr(self.config, "chunk_size"))
        return None

    # Weight loading is inherited from Qwen3ForCausalLM via _Qwen3ForCausalLM.


# Model registry entrypoint
EntryClass = HSAForCausalLM


