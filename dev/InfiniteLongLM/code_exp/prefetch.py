"""Quick model-build / forward-prefetch sanity check.

Builds one or more foundation models from JSON config(s), runs a tiny forward
pass, prints param counts, and reports any exception.

Usage
-----
    python code_exp/prefetch.py <config_path> [<config_path> ...]

Examples
--------
    python code_exp/prefetch.py configs/flash_hsa/config_hsa_pope_halfdim_nope_chunkattn_345M.json

    python code_exp/prefetch.py \
        configs/flash_hsa/config_hsa_pope_halfdim_nope_chunkattn_345M.json \
        configs/flash_hsa/config_hsa_pope_halfdim_nope_chunkattn_345M_layerbias.json
"""

import argparse
import sys

import models  # noqa: F401  (registers model classes via side effects)
from veomni.models import build_foundation_model
import torch

torch.manual_seed(42)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build foundation model(s) from JSON config and run a "
                    "tiny forward pass for sanity check.",
    )
    p.add_argument(
        "configs", nargs="+",
        help="One or more paths to model config JSON files. "
             "Each will be loaded and forwarded sequentially.",
    )
    p.add_argument(
        "--seq-len", type=int, default=64*8,
        help="Length of the random input sequence used for forward (default: 512).",
    )
    p.add_argument(
        "--vocab-cap", type=int, default=1000,
        help="Upper bound (exclusive) for the random input_ids (default: 1000).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        for config_path in args.configs:
            print(config_path)
            model = build_foundation_model(config_path=config_path)

            model = model.cuda()
            print(f"path: {config_path} param cnt: {count_parameters(model)}")
            # print(model)
            model.to(torch.bfloat16)
            model.train()
            input_ids = torch.randint(0, args.vocab_cap, (1, args.seq_len)).cuda()
            _ = model(input_ids, use_cache=False)

        print("[INFO] prefetch.py completed successfully.")
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
