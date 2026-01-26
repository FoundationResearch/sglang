"""
FlashHSA config shim for SGLang.

SGLang loads HuggingFace configs via `transformers.AutoConfig.from_pretrained`.
FlashHSA checkpoints typically use:
  - model_type = "flash_hsa"
  - architectures = ["HSAForCausalLM"]

Transformers may not know the "flash_hsa" model_type out-of-the-box, so we
register it as an alias of Qwen3Config for config parsing.
"""

from __future__ import annotations


def _register_flash_hsa_autoconfig() -> None:
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        # Register "flash_hsa" -> Qwen3Config (alias) for AutoConfig parsing.
        # If already registered, ignore.
        if "flash_hsa" not in CONFIG_MAPPING:
            CONFIG_MAPPING.register("flash_hsa", Qwen3Config)
    except Exception:
        # Be conservative: any import/version mismatch should not break SGLang.
        return


_register_flash_hsa_autoconfig()


