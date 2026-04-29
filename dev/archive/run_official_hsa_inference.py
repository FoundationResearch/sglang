"""
Run official FlashHSA (modeling_hsa_lmk.py) inference standalone.

This script mocks the `veomni` dependency (only used for distributed training)
and loads the official HSA model for single-GPU inference.

Usage:
  # With a trained checkpoint:
  python dev/run_official_hsa_inference.py \
    --checkpoint /path/to/hsa/checkpoint \
    --config dev/InfiniteLongLM/configs/flash_hsa/config_hsa_ultra_win512_1per2_stb.json \
    --prompt "The capital of France is"

  # Random weights (for testing the pipeline):
  python dev/run_official_hsa_inference.py \
    --config dev/InfiniteLongLM/configs/flash_hsa/config_hsa_ultra_win512_1per2_stb.json \
    --random-weights \
    --prompt "Hello world"

  # Custom tokenizer:
  python dev/run_official_hsa_inference.py \
    --checkpoint /path/to/hsa/checkpoint \
    --config dev/InfiniteLongLM/configs/flash_hsa/config_hsa_ultra_win512_1per2_stb.json \
    --tokenizer Qwen/Qwen3-0.6B \
    --prompt "Once upon a time"

Requirements:
  pip install transformers torch einops tilelang liger-kernel
"""

import argparse
import json
import sys
import os
import types
import logging

# ============================================================
# 1. Mock `veomni` before importing the official model.
#    veomni is only used for distributed training features;
#    for single-GPU inference we can safely stub it out.
# ============================================================

def _install_veomni_mock():
    """Install a minimal veomni mock into sys.modules."""

    # -- veomni (top-level) --
    veomni = types.ModuleType("veomni")
    sys.modules["veomni"] = veomni

    # -- veomni.distributed --
    veomni_dist = types.ModuleType("veomni.distributed")
    sys.modules["veomni.distributed"] = veomni_dist
    veomni.distributed = veomni_dist

    # -- veomni.distributed.parallel_state --
    veomni_ps = types.ModuleType("veomni.distributed.parallel_state")

    class _FakeParallelState:
        sp_enabled = False
        tp_enabled = False
        pp_enabled = False

    veomni_ps.get_parallel_state = lambda: _FakeParallelState()
    sys.modules["veomni.distributed.parallel_state"] = veomni_ps
    veomni_dist.parallel_state = veomni_ps

    # -- veomni.distributed.sequence_parallel --
    veomni_sp = types.ModuleType("veomni.distributed.sequence_parallel")
    veomni_sp.slice_position_embedding = lambda x, **kwargs: x  # identity
    sys.modules["veomni.distributed.sequence_parallel"] = veomni_sp
    veomni_dist.sequence_parallel = veomni_sp

    # -- veomni.utils --
    veomni_utils = types.ModuleType("veomni.utils")
    sys.modules["veomni.utils"] = veomni_utils
    veomni.utils = veomni_utils

    # -- veomni.utils.logging --
    veomni_logging = types.ModuleType("veomni.utils.logging")

    def _get_logger(name=None):
        logger = logging.getLogger(name)
        # Add rank-aware convenience methods (no-op in single-GPU).
        if not hasattr(logger, "info_rank0"):
            logger.info_rank0 = logger.info
        if not hasattr(logger, "warning_rank0"):
            logger.warning_rank0 = logger.warning
        return logger

    veomni_logging.get_logger = _get_logger
    sys.modules["veomni.utils.logging"] = veomni_logging
    veomni_utils.logging = veomni_logging

    # -- veomni.utils.import_utils --
    veomni_import = types.ModuleType("veomni.utils.import_utils")
    veomni_import.is_liger_kernel_available = lambda: True
    veomni_import.is_torch_npu_available = lambda: False
    veomni_import.is_transformers_version_greater_or_equal_to = lambda v: True
    sys.modules["veomni.utils.import_utils"] = veomni_import
    veomni_utils.import_utils = veomni_import

    # -- veomni.models --
    veomni_models = types.ModuleType("veomni.models")
    sys.modules["veomni.models"] = veomni_models
    veomni.models = veomni_models

    # -- veomni.models.module_utils --
    import torch.nn as nn
    veomni_mu = types.ModuleType("veomni.models.module_utils")
    veomni_mu.GradientCheckpointingLayer = nn.Module  # stub base class
    sys.modules["veomni.models.module_utils"] = veomni_mu
    veomni_models.module_utils = veomni_mu

    # -- veomni.models.loader (used by FlashHSA/__init__.py) --
    veomni_loader = types.ModuleType("veomni.models.loader")

    class _FakeRegistry:
        def register(self, name):
            def decorator(fn):
                return fn
            return decorator

    veomni_loader.MODELING_REGISTRY = _FakeRegistry()
    veomni_loader.MODEL_CONFIG_REGISTRY = _FakeRegistry()
    sys.modules["veomni.models.loader"] = veomni_loader
    veomni_models.loader = veomni_loader


_install_veomni_mock()

# ============================================================
# 2. Add InfiniteLongLM to sys.path so `ops.*` and `utils.*`
#    can be imported by the official model code.
# ============================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_INFINITELONGLM_DIR = os.path.join(_SCRIPT_DIR, "InfiniteLongLM")
if _INFINITELONGLM_DIR not in sys.path:
    sys.path.insert(0, _INFINITELONGLM_DIR)

# Pre-register stub modules for other model families that models/__init__.py
# tries to import (we only need FlashHSA).
for _stub_name in ["models.SWANGPT", "models.DRT", "models.SWANNSA"]:
    sys.modules[_stub_name] = types.ModuleType(_stub_name)

# ============================================================
# 3. Import official HSA model.
# ============================================================

import torch
from transformers import AutoTokenizer

# Register the HSA config type with HuggingFace.
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


def _register_hsa_config():
    """Register HSAConfig so AutoConfig can parse model_type='flash_hsa_*'."""
    from models.FlashHSA.configuration_hsa import HSAConfig
    for name in ["flash_hsa_innerx", "flash_hsa_innerx_ultra", "flash_hsa_lmk"]:
        if name not in CONFIG_MAPPING:
            try:
                CONFIG_MAPPING.register(name, HSAConfig)
            except Exception:
                CONFIG_MAPPING._extra_content[name] = HSAConfig


_register_hsa_config()

from models.FlashHSA.modeling_hsa_lmk import HSAForCausalLM, HierarchicalSparseAttention
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

# Patch 1: HierarchicalSparseAttention doesn't set num_key_value_groups,
# but eager_attention_forward needs it for repeat_kv.
_orig_hsa_init = HierarchicalSparseAttention.__init__

def _patched_hsa_init(self, config, layer_idx):
    _orig_hsa_init(self, config, layer_idx)
    if not hasattr(self, "num_key_value_groups"):
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

HierarchicalSparseAttention.__init__ = _patched_hsa_init

# Patch 2: tilelang kernels require contiguous tensors, but
# lmk_k = hsa_k_norm[:, chunk_size-1::chunk_size, :, :] is a non-contiguous slice.
# Wrap topk_func and hsa_func to make inputs contiguous.
import ops.topk_group as _topk_mod
import ops.hsa_fwd_bwd_group as _hsa_mod

_orig_topk = _topk_mod.online_topk_group

def _fix_strides(t):
    return t.reshape(t.shape).contiguous()

def _contiguous_topk(q, k, *args, **kwargs):
    return _orig_topk(_fix_strides(q), _fix_strides(k), *args, **kwargs)

_topk_mod.online_topk_group = _contiguous_topk

_orig_hsa_fwd = _hsa_mod.HSA_block_M_group

def _contiguous_hsa_fwd(q, k, v, *args, **kwargs):
    return _orig_hsa_fwd(_fix_strides(q), _fix_strides(k), _fix_strides(v), *args, **kwargs)

_hsa_mod.HSA_block_M_group = _contiguous_hsa_fwd


# ============================================================
# 4. Inference helpers.
# ============================================================

def prepare_input_with_landmarks(
    input_ids: torch.Tensor,
    chunk_size: int,
    lmk_id: int,
    device: torch.device,
) -> tuple:
    """Insert LMK tokens and build landmark-aware position IDs.

    Args:
        input_ids: [1, L] original token IDs
        chunk_size: page/chunk size (LMK inserted every chunk_size-1 tokens)
        lmk_id: landmark token ID (usually vocab_size)
        device: target device

    Returns:
        (input_ids_with_lmk, position_ids, attention_mask)
    """
    # Insert landmark tokens.
    input_ids_lmk = insert_special_tokens(input_ids, lmk_id, chunk_size)
    L_new = input_ids_lmk.shape[1]

    # Build position IDs (LMK shares position with previous real token).
    position_ids = create_position_ids_with_landmarks(
        input_ids.shape[1], chunk_size, device
    )

    # Attention mask: all ones.
    attention_mask = torch.ones((1, L_new), dtype=torch.long, device=device)

    return input_ids_lmk.to(device), position_ids.to(device), attention_mask


@torch.no_grad()
def generate_greedy(
    model: HSAForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Simple greedy/sampling generation with LMK injection.

    This manually handles LMK insertion at each decode step,
    mirroring sglang's hsa_decode_postprocess_sampled_token() logic.
    """
    device = input_ids.device
    chunk_size = model.chunk_size
    lmk_id = model.lmk_id
    vocab_size = model.config.vocab_size  # original vocab (LMK = vocab_size)

    # Step 1: Prefill with landmark-augmented input.
    input_ids_lmk, position_ids, attention_mask = prepare_input_with_landmarks(
        input_ids, chunk_size, lmk_id, device
    )

    # Run prefill.
    outputs = model(
        input_ids=input_ids_lmk,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :vocab_size]  # exclude LMK from sampling

    generated_ids = []
    # Track engine-visible length (including LMK tokens).
    engine_len = int(input_ids_lmk.shape[1])

    for step in range(max_new_tokens):
        # Sample next token.
        if temperature <= 0 or temperature == 1.0 and top_p == 1.0:
            next_token = logits.argmax(dim=-1)  # [1]
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        token_id = int(next_token.item())
        generated_ids.append(token_id)

        # Check if next engine position is LMK slot.
        next_engine_pos = engine_len  # 0-based index of next token
        is_lmk_slot = ((next_engine_pos + 1) % chunk_size) == 0

        if is_lmk_slot:
            # Insert LMK token first, then the real token.

            # LMK step: feed LMK token.
            lmk_input = torch.tensor([[lmk_id]], device=device)
            lmk_pos_id = next_engine_pos - ((next_engine_pos + 1) // chunk_size)
            lmk_position_ids = torch.tensor([[lmk_pos_id]], device=device, dtype=torch.long)
            lmk_mask = torch.ones((1, engine_len + 1), dtype=torch.long, device=device)

            lmk_out = model(
                input_ids=lmk_input,
                position_ids=lmk_position_ids,
                attention_mask=lmk_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = lmk_out.past_key_values
            engine_len += 1
            # Discard LMK logits — it's an internal step.

            # Now feed the real token.
            real_input = next_token.unsqueeze(0)  # [1, 1]
            real_engine_pos = engine_len
            real_pos_id = real_engine_pos - ((real_engine_pos + 1) // chunk_size)
            real_position_ids = torch.tensor([[real_pos_id]], device=device, dtype=torch.long)
            real_mask = torch.ones((1, engine_len + 1), dtype=torch.long, device=device)

            real_out = model(
                input_ids=real_input,
                position_ids=real_position_ids,
                attention_mask=real_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = real_out.past_key_values
            logits = real_out.logits[:, -1, :vocab_size]
            engine_len += 1
        else:
            # Normal step: feed the sampled token directly.
            real_input = next_token.unsqueeze(0)  # [1, 1]
            real_pos_id = next_engine_pos - ((next_engine_pos + 1) // chunk_size)
            real_position_ids = torch.tensor([[real_pos_id]], device=device, dtype=torch.long)
            real_mask = torch.ones((1, engine_len + 1), dtype=torch.long, device=device)

            real_out = model(
                input_ids=real_input,
                position_ids=real_position_ids,
                attention_mask=real_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = real_out.past_key_values
            logits = real_out.logits[:, -1, :vocab_size]
            engine_len += 1

    return torch.tensor(generated_ids, device=device)


# ============================================================
# 5. Main.
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run official FlashHSA inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to HSA model checkpoint (HF format or safetensors dir)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to HSA config.json")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name or path (default: use checkpoint path, or Qwen/Qwen3-0.6B)")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--random-weights", action="store_true",
                        help="Use random weights (skip checkpoint loading)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda, cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # Load config.
    print(f"Loading config from {args.config}")
    from models.FlashHSA.configuration_hsa import HSAConfig
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    config = HSAConfig(**config_dict)

    print(f"  model_type={config.model_type}")
    print(f"  hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
    print(f"  heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")
    print(f"  chunk_size={config.chunk_size}, hsa_topk={config.hsa_topk}")
    print(f"  hsa_heads={config.hsa_heads}, hsa_qk_ratio={config.hsa_qk_ratio}")

    # Load or initialize model.
    if args.random_weights:
        print("Initializing model with random weights...")
        config._attn_implementation = "eager"
        model = HSAForCausalLM(config)
        model = model.to(dtype=dtype, device=device)
    elif args.checkpoint:
        print(f"Loading model from {args.checkpoint}")
        if os.path.isdir(args.checkpoint):
            # Try loading as HF checkpoint directory.
            model = HSAForCausalLM.from_pretrained(
                args.checkpoint,
                config=config,
                torch_dtype=dtype,
                device_map=args.device,
            )
        else:
            # Single file: load state dict.
            config._attn_implementation = "eager"
            model = HSAForCausalLM(config)
            state_dict = torch.load(args.checkpoint, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
            model = model.to(dtype=dtype, device=device)
    else:
        parser.error("Must specify --checkpoint or --random-weights")

    model.eval()

    # Patch model instances: replace topk_func and hsa_func to ensure dense strides.
    # tilelang kernels check strides strictly; .contiguous() is a no-op for size-1 dims
    # (PyTorch considers them already contiguous), so we use .clone() to force a copy.
    def _densify(t):
        """Force a dense-stride tensor (reshape forces compact strides)."""
        return t.reshape(t.shape).contiguous()

    for module in model.modules():
        if isinstance(module, HierarchicalSparseAttention):
            _orig_tf = module.topk_func
            _orig_hf = module.hsa_func
            module.topk_func = lambda q, k, *a, _f=_orig_tf, **kw: _f(_densify(q), _densify(k), *a, **kw)
            module.hsa_func = lambda q, k, v, *a, _f=_orig_hf, **kw: _f(_densify(q), _densify(k), _densify(v), *a, **kw)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"  vocab_size={config.vocab_size}, lmk_id={model.lmk_id}")
    print(f"  padded_vocab_size={model.vocab_size}")

    # Load tokenizer.
    tokenizer_path = args.tokenizer or args.checkpoint or "Qwen/Qwen3-0.6B"
    print(f"Loading tokenizer from {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {tokenizer_path}: {e}")
        print("Falling back to Qwen/Qwen3-0.6B tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    # Tokenize prompt.
    print(f"\nPrompt: {args.prompt!r}")
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"Input tokens: {input_ids.shape[1]}")

    # Generate.
    print(f"Generating {args.max_new_tokens} tokens...")
    output_ids = generate_greedy(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Decode.
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    full_text = args.prompt + output_text
    print(f"\n{'='*60}")
    print(f"Output: {full_text}")
    print(f"{'='*60}")
    print(f"Generated {len(output_ids)} tokens")


if __name__ == "__main__":
    main()
