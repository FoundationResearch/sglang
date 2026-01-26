import json
import os
from pathlib import Path

import pytest


def _vprint(msg: str):
    if os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") not in ("", "0", "false", "False"):
        print(msg, flush=True)


def test_flash_hsa_autoconfig_and_registry(tmp_path: Path):
    """
    Validate two things:
      1) Transformers AutoConfig can parse model_type="flash_hsa" after SGLang's shim registration.
      2) SGLang ModelRegistry can resolve the architecture name "HSAForCausalLM".

    This test does NOT instantiate the full model (would require distributed init + GPU memory).
    """

    # Importing sglang.srt.configs triggers the side-effect AutoConfig registration shim.
    import sglang.srt.configs  # noqa: F401

    from transformers import AutoConfig

    # Create a minimal HF-like folder with config.json.
    cfg_path = tmp_path / "config.json"
    cfg = {
        "model_type": "flash_hsa",
        "architectures": ["HSAForCausalLM"],
        "vocab_size": 100278,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "use_sliding_window_fusion": True,
        # New name (requested); old name "sliding_window" should not be required.
        "sliding_window_fusion_size": 64,
        # FlashHSA scheduling pattern
        "full_attn_interleave": 4,
        "chunk_size": 64,
        "tie_word_embeddings": True,
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    c = AutoConfig.from_pretrained(str(tmp_path), trust_remote_code=True)
    _vprint(
        f"loaded AutoConfig: model_type={c.model_type} arch={getattr(c,'architectures',None)}"
    )

    from sglang.srt.configs.flash_hsa import FlashHSAConfig

    assert c.model_type == "flash_hsa"
    assert isinstance(c, FlashHSAConfig)
    assert getattr(c, "architectures", None) == ["HSAForCausalLM"]
    assert getattr(c, "sliding_window_fusion_size", None) == 64

    # Registry resolution
    from sglang.srt.models.registry import ModelRegistry

    supported = set(ModelRegistry.get_supported_archs())
    _vprint(f"supported archs contains HSAForCausalLM? {'HSAForCausalLM' in supported}")
    assert "HSAForCausalLM" in supported


