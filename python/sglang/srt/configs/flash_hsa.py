"""
FlashHSAConfig for SGLang.

We implement a dedicated HF config class (instead of aliasing Qwen3Config) so:
- `model_type="flash_hsa"` is first-class and self-describing
- FlashHSA-specific knobs (e.g. `chunk_size`, `full_attn_interleave`) live in a
  proper config type
- We can evolve validation/migrations without coupling to upstream Qwen3Config

We also register this config with Transformers AutoConfig mapping so that
`AutoConfig.from_pretrained(<flashhsa checkpoint>)` works out-of-the-box.
"""

from __future__ import annotations

from typing import Literal, Optional

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation


class FlashHSAConfig(PretrainedConfig):
    model_type = "flash_hsa"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # ---- Official FlashHSA-format knobs (compat input) ----
        # Upstream uses a single window size for both:
        # - HSA-layer SWA (for SWA→HSA merging)
        # - standalone sliding window attention layers
        #
        # We accept these keys and map them into the split knobs below.
        use_sliding_window: Optional[bool] = None,
        sliding_window: Optional[int] = None,
        # Sliding window for HSA-layer SWA used in SWA→HSA merging (inside HierarchicalSparseAttention).
        use_sliding_window_merging: bool = False,
        sliding_window_merging_size: Optional[int] = None,
        # Sliding window for standalone sliding-window attention layers (layer_types == "sliding_attention").
        use_sliding_window_attention: bool = False,
        sliding_window_attention_size: Optional[int] = None,
        max_window_layers: int = 28,
        layer_types=None,
        # HSA specifics
        chunk_size: int = 64,
        enable_softmax1: bool = True,
        hsa_mode: Literal["dense", "sparse"] = "sparse",
        hsa_topk: int = 16,
        full_attn_interleave: int = 4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # ---- Sliding window config semantics ----
        # Two supported input formats:
        #   A) Official FlashHSA: (use_sliding_window, sliding_window)
        #      -> mapped to BOTH merging + attention windows
        #   B) Split SGLang: (use_sliding_window_merging, sliding_window_merging_size) and
        #                   (use_sliding_window_attention, sliding_window_attention_size)
        #
        # We do NOT allow mixing formats to avoid ambiguous intent.
        if use_sliding_window is not None or sliding_window is not None:
            if (
                use_sliding_window_merging
                or use_sliding_window_attention
                or sliding_window_merging_size is not None
                or sliding_window_attention_size is not None
            ):
                raise ValueError(
                    "FlashHSAConfig: do not mix official keys "
                    "(`use_sliding_window`, `sliding_window`) with split keys "
                    "(`use_sliding_window_merging`, `sliding_window_merging_size`, "
                    "`use_sliding_window_attention`, `sliding_window_attention_size`)."
                )
            if use_sliding_window is None:
                raise ValueError(
                    "FlashHSAConfig: official format requires `use_sliding_window` "
                    "when `sliding_window` is provided."
                )
            if bool(use_sliding_window):
                if sliding_window is None:
                    raise ValueError(
                        "FlashHSAConfig: official format requires `sliding_window` "
                        "when `use_sliding_window=True`."
                    )
                w = int(sliding_window)
                self.use_sliding_window_merging = True
                self.sliding_window_merging_size = w
                self.use_sliding_window_attention = True
                self.sliding_window_attention_size = w
            else:
                self.use_sliding_window_merging = False
                self.sliding_window_merging_size = None
                self.use_sliding_window_attention = False
                self.sliding_window_attention_size = None
        else:
            # Split format (no implicit fallback between the two sizes)
            self.use_sliding_window_merging = bool(use_sliding_window_merging)
            self.use_sliding_window_attention = bool(use_sliding_window_attention)
            self.sliding_window_merging_size = (
                int(sliding_window_merging_size)
                if sliding_window_merging_size is not None
                else None
            )
            self.sliding_window_attention_size = (
                int(sliding_window_attention_size)
                if sliding_window_attention_size is not None
                else None
            )
            if self.use_sliding_window_merging and self.sliding_window_merging_size is None:
                raise ValueError(
                    "FlashHSAConfig: `use_sliding_window_merging=True` requires `sliding_window_merging_size`."
                )
            if self.use_sliding_window_attention and self.sliding_window_attention_size is None:
                raise ValueError(
                    "FlashHSAConfig: `use_sliding_window_attention=True` requires `sliding_window_attention_size`."
                )

        self.max_window_layers = max_window_layers

        self.chunk_size = chunk_size
        self.enable_softmax1 = enable_softmax1
        self.hsa_mode = hsa_mode
        self.hsa_topk = hsa_topk
        self.full_attn_interleave = full_attn_interleave

        # RoPE validation (mirrors HF style)
        if self.rope_scaling is not None and isinstance(self.rope_scaling, dict):
            if "type" in self.rope_scaling and "rope_type" not in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # Layer types default pattern (mirrors FlashHSA upstream)
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if full_attn_interleave > 0 and (i + 1) % full_attn_interleave == 0
                else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


def _register_flash_hsa_autoconfig() -> None:
    """Register FlashHSAConfig into Transformers AutoConfig mapping."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "flash_hsa" not in CONFIG_MAPPING:
            CONFIG_MAPPING.register("flash_hsa", FlashHSAConfig)
    except Exception:
        # Be conservative: any import/version mismatch should not break SGLang.
        print("[ERROR] Failed to register FlashHSAConfig into Transformers AutoConfig mapping")
        return


_register_flash_hsa_autoconfig()



