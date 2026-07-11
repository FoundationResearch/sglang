#!/usr/bin/env python3
"""
Run offline inference with the HSA backend on one or more prompts.

A thin convenience wrapper around sglang's offline Engine that sets the HSA
serving knobs for you (attention-backend=hsa, page-size=64). Prompts can be
passed as positional args, repeated --prompt flags, a --prompt-file (one prompt
per line), or piped on stdin.

Examples:
    # positional prompts
    python scripts/run_hsa_infer.py --model-path ./HiLS-Attention-7B-sglang \
        "The capital of France is" "Water is made of hydrogen and"

    # from a file, longer generations
    python scripts/run_hsa_infer.py --model-path ./HiLS-Attention-7B-sglang \
        --prompt-file prompts.txt --max-new-tokens 128

    # from stdin
    echo "Once upon a time" | python scripts/run_hsa_infer.py \
        --model-path ./HiLS-Attention-7B-sglang
"""

import argparse
import sys


def collect_prompts(args) -> list:
    prompts = list(args.prompts)
    for p in args.prompt or []:
        prompts.append(p)
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts.extend(line.rstrip("\n") for line in f if line.strip())
    if not prompts and not sys.stdin.isatty():
        text = sys.stdin.read()
        if text.strip():
            prompts.append(text)
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="Converted HSA checkpoint dir")
    ap.add_argument("prompts", nargs="*", help="Prompt(s) as positional args")
    ap.add_argument("--prompt", action="append", help="A prompt (repeatable)")
    ap.add_argument("--prompt-file", help="File with one prompt per line")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--page-size", type=int, default=64, help="Must equal chunk_size (64)")
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--context-length", type=int, default=None,
                    help="Raise for long prompts (HSA inserts an LMK every 63 tokens, "
                         "so allow ~1.6%% headroom over your longest prompt + output).")
    args = ap.parse_args()

    prompts = collect_prompts(args)
    if not prompts:
        ap.error("no prompts given (positional, --prompt, --prompt-file, or stdin)")

    import sglang as sgl

    # Size context_length to the actual prompts unless the user set it. HSA's
    # landmark pool scales with context_length, so defaulting to the model's full
    # max (e.g. 131072) reserves tens of GiB for tiny prompts and OOMs. HSA also
    # inserts one LMK token every (page_size-1) real tokens, so add ~1.6% headroom.
    if args.context_length is not None:
        context_length = args.context_length
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        max_prompt_tok = max(len(tok(p)["input_ids"]) for p in prompts)
        expanded = max_prompt_tok + max_prompt_tok // (args.page_size - 1)
        context_length = max(2048, expanded + args.max_new_tokens + 256)

    kw = dict(
        model_path=args.model_path,
        attention_backend="hsa",
        page_size=args.page_size,
        trust_remote_code=True,
        mem_fraction_static=args.mem_fraction_static,
        context_length=context_length,
    )
    # NOTE: the HSA backend auto-sets disable_overlap_schedule=True and
    # prefill_max_requests=1 (single-sequence prefill; decode stays batched).
    llm = sgl.Engine(**kw)

    sp = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    print("=" * 72)
    for p in prompts:
        out = llm.generate(p, sp)
        print(f"PROMPT: {p!r}")
        print(f"OUTPUT: {out['text']!r}")
        print("-" * 72)
    llm.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
