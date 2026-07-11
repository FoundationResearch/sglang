#!/usr/bin/env python3
"""
Convert a public HiLS-Attention checkpoint to an SGLang-HSA-loadable checkpoint.

The released weights (e.g. https://huggingface.co/tencent/HiLS-Attention-7B) use
the upstream `HiLS*` / `hils_*` naming. The SGLang HSA backend expects the
`HSAForCausalLM` / `FlashHSAConfig` schema with `hsa_*` keys. The **weight tensors
are already compatible** — only `config.json` needs translating — so this tool
just rewrites the config and (by default) symlinks the weight/tokenizer files into
a new directory. No tensor is renamed or copied unless you pass `--copy`.

Usage:
    python scripts/convert_hils_checkpoint.py \
        --src /path/to/HiLS-Attention-7B \
        --dst /path/to/HiLS-Attention-7B-sglang

Then serve/infer with the HSA backend:
    python -m sglang.launch_server --model-path /path/to/HiLS-Attention-7B-sglang \
        --attention-backend hsa --page-size 64 --trust-remote-code
"""

import argparse
import json
import os
import shutil
import sys

# Weight-tensor name suffixes we expect in an HSA checkpoint. Used only for an
# informational audit — anything outside these patterns is reported, not blocked.
_KNOWN_SUFFIXES = (
    "embed_tokens.weight", "norm.weight", "lm_head.weight", "lmk_embed",
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "q_norm.weight", "k_norm.weight",
    "lmk_q_proj.0.weight", "lmk_q_proj.1.weight", "lmk_q_norm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "post_feedforward_layernorm.weight",
)

_WEIGHT_GLOBS = (".safetensors", ".bin", ".index.json", ".pt")
_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "generation_config.json", "tokenizer.model",
    "added_tokens.json", "chat_template.jinja",
)


def _first(cfg, *keys, default=None):
    """Return the first present key from `keys` (supports hils_/hsa_ aliases)."""
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def translate_config(src_cfg: dict) -> dict:
    """Return an SGLang-HSA config.json derived from an upstream HiLS config."""
    c = dict(src_cfg)  # keep all original keys (harmless extras land in kwargs)

    h_q = int(_first(c, "num_attention_heads"))
    h_kv = int(_first(c, "num_key_value_heads", default=h_q))
    hidden = int(_first(c, "hidden_size"))

    # --- route to our SGLang model + config classes ---
    c["architectures"] = ["HSAForCausalLM"]
    src_mt = str(_first(c, "model_type", default="")).lower()
    # decoder topology: OLMo3 post-norm vs Qwen3 pre-norm. Detect from the source
    # model_type; `olmo_lhsa` -> "olmo" post-norm, `qwen_lhsa` -> "qwen" pre-norm.
    c["model_type"] = "qwen_lhsa" if "qwen" in src_mt else "olmo_lhsa"

    # --- HSA geometry (derive from the source; add `hsa_*` aliases) ---
    c.setdefault("head_dim", hidden // h_q)
    # All released HiLS heads run through HSA (no SWA/HSA head split).
    c["hsa_heads"] = int(_first(c, "hsa_heads", default=h_q))
    c["hsa_qk_ratio"] = int(_first(c, "hsa_qk_ratio", default=max(1, h_q // h_kv)))
    c["hsa_topk"] = int(_first(c, "hsa_topk", "hils_topk", default=32))
    c["hsa_sliding_window"] = int(
        _first(c, "hsa_sliding_window", "hils_sliding_window", "sliding_window", default=512)
    )
    c["apply_hsa_rope"] = bool(_first(c, "apply_hsa_rope", "apply_hils_rope", default=False))
    c["retrieval_head_num"] = int(_first(c, "retrieval_head_num", default=h_q))

    # `use_sliding_window` + `sliding_window` (official format) are left as-is;
    # FlashHSAConfig maps them to the SWA attention/merging windows for the
    # non-HSA layers. Do NOT also emit the split keys, or FlashHSAConfig rejects
    # the mixed format.
    return c


def audit_weight_keys(src: str) -> None:
    """Print any weight tensor names that fall outside the known HSA schema."""
    idx = os.path.join(src, "model.safetensors.index.json")
    keys = []
    if os.path.exists(idx):
        keys = list(json.load(open(idx))["weight_map"].keys())
    else:
        try:
            from safetensors import safe_open  # noqa
            import glob
            for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
                with safe_open(f, "pt") as sf:
                    keys.extend(sf.keys())
        except Exception:
            print("  [audit] no index / safetensors found; skipping key audit")
            return
    unknown = [k for k in keys if not k.endswith(_KNOWN_SUFFIXES)]
    print(f"  [audit] {len(keys)} weight tensors; "
          f"{len(keys) - len(unknown)} match the HSA schema, {len(unknown)} unrecognized")
    for k in unknown[:20]:
        print(f"          UNRECOGNIZED (loads verbatim, may be unused): {k}")


def link_or_copy(src: str, dst: str, name: str, copy: bool) -> None:
    s, d = os.path.join(src, name), os.path.join(dst, name)
    if os.path.lexists(d):
        os.remove(d)
    if copy:
        shutil.copy2(s, d)
    else:
        os.symlink(os.path.realpath(s), d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Source HiLS-Attention checkpoint dir")
    ap.add_argument("--dst", required=True, help="Output SGLang-HSA checkpoint dir")
    ap.add_argument("--copy", action="store_true",
                    help="Copy weight/tokenizer files instead of symlinking")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing --dst")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if not os.path.isdir(src):
        print(f"[error] --src is not a directory: {src}", file=sys.stderr)
        return 1
    src_cfg_path = os.path.join(src, "config.json")
    if not os.path.exists(src_cfg_path):
        print(f"[error] no config.json in {src}", file=sys.stderr)
        return 1
    if os.path.exists(dst) and os.listdir(dst) and not args.force:
        print(f"[error] --dst exists and is non-empty (use --force): {dst}", file=sys.stderr)
        return 1
    os.makedirs(dst, exist_ok=True)

    # 1) translate config
    src_cfg = json.load(open(src_cfg_path))
    new_cfg = translate_config(src_cfg)
    json.dump(new_cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"[ok] wrote translated config.json  ({src_cfg.get('model_type')} "
          f"-> {new_cfg['model_type']}, arch {new_cfg['architectures'][0]})")
    print("     hsa_heads=%d hsa_qk_ratio=%d hsa_topk=%d hsa_sliding_window=%d "
          "apply_hsa_rope=%s head_dim=%d" % (
              new_cfg["hsa_heads"], new_cfg["hsa_qk_ratio"], new_cfg["hsa_topk"],
              new_cfg["hsa_sliding_window"], new_cfg["apply_hsa_rope"], new_cfg["head_dim"]))

    # 2) audit weight keys (no remapping needed — informational)
    audit_weight_keys(src)

    # 3) link/copy weights + tokenizer
    linked = 0
    for name in sorted(os.listdir(src)):
        if name == "config.json":
            continue
        if name.endswith(_WEIGHT_GLOBS) or name in _TOKENIZER_FILES:
            link_or_copy(src, dst, name, args.copy)
            linked += 1
    verb = "copied" if args.copy else "symlinked"
    print(f"[ok] {verb} {linked} weight/tokenizer files into {dst}")

    # 4) validate the translated config parses through FlashHSAConfig
    try:
        import sglang.srt.configs.flash_hsa  # noqa: registers AutoConfig
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(dst, trust_remote_code=True)
        assert type(cfg).__name__ == "FlashHSAConfig", type(cfg).__name__
        print(f"[ok] config validates: FlashHSAConfig decoder_variant={cfg.decoder_variant} "
              f"SWA={cfg.use_sliding_window_attention}/{cfg.sliding_window_attention_size}")
    except Exception as e:  # pragma: no cover - validation is best-effort
        print(f"[warn] could not validate config through FlashHSAConfig: {e}")

    print(f"\nDone. Serve with:\n"
          f"  python -m sglang.launch_server --model-path {dst} \\\n"
          f"      --attention-backend hsa --page-size 64 --trust-remote-code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
