#!/usr/bin/env python3
"""HSA batch_size > 1 equivalence regression test.

Running N prompts concurrently must produce the same greedy output as running
them one at a time.  "The same" needs one qualification: batching changes GEMM
shapes, hence floating-point reduction order, so two *equally correct* batch
compositions already drift apart after enough greedy steps.  This test
therefore checks two things:

  * SHORT generation must match EXACTLY.  Drift needs hundreds of steps to
    flip a token, so any short-horizon mismatch is a real bug.

  * LONG generation must agree for at least ``--min-agree`` tokens, chosen to
    be past ``hsa_sliding_window`` (512).  That threshold is what separates a
    genuine landmark bug from fp drift: a chunk completed during decode only
    becomes *selectable* once it falls out of the sliding window, so a broken
    decode-time landmark writer diverges at ~512 tokens, while fp drift shows
    up much later and moves around with batch composition.

``--fp-noise-control`` measures where two equally-correct batch compositions
diverge from each other, i.e. the floor this test cannot go below.

Usage:
    python dev/test_hsa_batch_equivalence.py [--model DIR] [--no-cuda-graph]
"""

import argparse
import sys

DEFAULT_MODEL = "dev/align/ckpt_345m_bench_hf"
# Deliberately staggered so the batch always holds mixed sequence lengths:
# requests cross chunk boundaries on different steps, hit different prefix
# lengths, and the 1-word prompt is under one page (the S == 0 pure-SWA path).
PROMPT_WORDS = (1, 5, 40, 120, 300, 700)


def build_prompts():
    base = "The history of computing began when "
    return [base + " ".join(["machines"] * n) + ". In summary," for n in PROMPT_WORDS]


def agree_len(a, b):
    """Number of leading tokens two token lists share."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--context-length", type=int, default=6144)
    ap.add_argument("--mem-fraction-static", type=float, default=0.25)
    ap.add_argument("--short-tokens", type=int, default=48)
    ap.add_argument("--long-tokens", type=int, default=900)
    ap.add_argument("--min-agree", type=int, default=600,
                    help="Long-generation tokens that must match; must exceed "
                         "hsa_sliding_window (512) to be meaningful")
    ap.add_argument("--prefill-max-requests", type=int, default=None,
                    help="Leave unset to let the backend batch prefills too")
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--disable-radix-cache", action="store_true")
    ap.add_argument("--max-running-requests", type=int, default=16,
                    help="Set to 1 to force serial execution of a concurrent submit")
    ap.add_argument("--only", default=None,
                    help="Comma-separated prompt indices, e.g. 3,5 — for shrinking "
                         "a failing case down to a minimal reproducer")
    ap.add_argument("--fp-noise-control", action="store_true",
                    help="Also report where two equally-correct batch "
                         "compositions diverge from each other")
    args = ap.parse_args()

    import sglang as sgl
    from transformers import AutoTokenizer

    prompts = build_prompts()
    if args.only:
        keep = [int(i) for i in args.only.split(",")]
        prompts = [prompts[i] for i in keep]
        print(f"prompt subset: {keep} (word counts {[PROMPT_WORDS[i] for i in keep]})")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    kw = dict(
        model_path=args.model,
        attention_backend="hsa",
        page_size=args.page_size,
        trust_remote_code=True,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        disable_overlap_schedule=True,
        disable_cuda_graph=args.no_cuda_graph,
        disable_radix_cache=args.disable_radix_cache,
        max_running_requests=args.max_running_requests,
        log_level="warning",
    )
    if args.prefill_max_requests is not None:
        kw["prefill_max_requests"] = args.prefill_max_requests
    llm = sgl.Engine(**kw)

    all_ok = True
    for tag, max_new, exact in (
        ("short", args.short_tokens, True),
        ("long", args.long_tokens, False),
    ):
        sp = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": max_new}
        print("=" * 78)
        print(f"{tag} generation: max_new_tokens={max_new}, B={len(prompts)} vs B=1"
              f"{'  (exact match required)' if exact else f'  (>= {args.min_agree} tokens must match)'}")
        out_batch = llm.generate(prompts, sp)
        out_serial = [llm.generate(p, sp) for p in prompts]

        for i, (a, b) in enumerate(zip(out_batch, out_serial)):
            ta = tok(a["text"])["input_ids"]
            tb = tok(b["text"])["input_ids"]
            n = agree_len(ta, tb)
            same_len = len(ta) == len(tb)
            if exact:
                ok = (ta == tb)
                note = "MATCH" if ok else f"MISMATCH at token #{n}"
            else:
                # Full agreement, or agreement well past the sliding window.
                ok = (ta == tb) or n >= args.min_agree
                note = (
                    "MATCH" if (ta == tb and same_len)
                    else f"agrees for {n} tokens (fp drift after)" if ok
                    else f"DIVERGES at token #{n} — too early, expected >= {args.min_agree}"
                )
            all_ok &= ok
            print(f"    [{tag}] prompt[{i}] ({len(ta)} tok): {note}")
            if not ok:
                print(f"        batched: {tok.decode(ta[n:n + 12])!r}")
                print(f"        serial : {tok.decode(tb[n:n + 12])!r}")

    if args.fp_noise_control and len(prompts) > 2:
        sp = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": args.long_tokens}
        target = len(prompts) - 3
        sub = list(range(1, len(prompts) - 1))
        a = llm.generate(prompts, sp)[target]["text"]
        b = llm.generate([prompts[i] for i in sub], sp)[sub.index(target)]["text"]
        n = agree_len(tok(a)["input_ids"], tok(b)["input_ids"])
        print("-" * 78)
        print(f"fp-noise control: two equally-correct batch compositions agree "
              f"for {n} tokens")

    print("=" * 78)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    llm.shutdown()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
