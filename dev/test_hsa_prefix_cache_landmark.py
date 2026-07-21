#!/usr/bin/env python3
"""Is the HSA landmark pool correct across radix prefix-cache reuse?

Prefix caching must be output-invariant: generating from a prompt a second time
(when its KV is already in the radix tree) has to produce exactly the same
tokens as the first, cold run.

The landmark pool is keyed by (req_pool_idx, chunk_idx), but reused KV pages
carry the landmarks of the request that *originally* produced them.  A request
served from the prefix cache gets a fresh req_pool_idx with no slots recorded
for the cached chunks, so the selector falls back to the padding row.  Like the
B>1 decode-writer bug, this only becomes visible once a cached chunk ages out of
the sliding window, i.e. after > hsa_sliding_window generated tokens.

Run with --disable-radix-cache to confirm the same prompts agree when no reuse
can happen.
"""

import argparse
import sys

DEFAULT_MODEL = "dev/align/ckpt_345m_bench_hf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--disable-radix-cache", action="store_true")
    ap.add_argument("--no-cuda-graph", action="store_true")
    args = ap.parse_args()

    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = (
        "The history of computing began when "
        + " ".join(["machines"] * 300)
        + ". In summary,"
    )

    llm = sgl.Engine(
        model_path=args.model,
        attention_backend="hsa",
        page_size=64,
        trust_remote_code=True,
        mem_fraction_static=0.25,
        context_length=4096,
        disable_overlap_schedule=True,
        disable_cuda_graph=args.no_cuda_graph,
        disable_radix_cache=args.disable_radix_cache,
        prefill_max_requests=1,
        max_running_requests=1,
        log_level="warning",
    )
    sp = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": args.max_new_tokens}

    cold = llm.generate(prompt, sp)["text"]
    warm = llm.generate(prompt, sp)["text"]   # identical prompt -> full cache hit
    llm.shutdown()

    tc, tw = tok(cold)["input_ids"], tok(warm)["input_ids"]
    print("=" * 78)
    print(f"radix cache {'DISABLED' if args.disable_radix_cache else 'ENABLED'}, "
          f"max_new_tokens={args.max_new_tokens}")
    if tc == tw:
        print(f"RESULT: PASS - cold and warm runs agree ({len(tc)} tokens)")
        return 0
    div = next((j for j in range(min(len(tc), len(tw))) if tc[j] != tw[j]),
               min(len(tc), len(tw)))
    print(f"RESULT: FAIL - diverges at generated token #{div}")
    print(f"    cold: {tok.decode(tc[div:div + 12])!r}")
    print(f"    warm: {tok.decode(tw[div:div + 12])!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
