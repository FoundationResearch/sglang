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
        # Sliding window (Fusion SWA/HSA)
        use_sliding_window_fusion: bool = False,
        sliding_window_fusion_size: Optional[int] = None,
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

        self.use_sliding_window_fusion = bool(use_sliding_window_fusion)
        self.sliding_window_fusion_size = (
            int(sliding_window_fusion_size)
            if sliding_window_fusion_size is not None
            else None
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



