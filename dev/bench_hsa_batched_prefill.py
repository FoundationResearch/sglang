#!/usr/bin/env python3
"""What does batched prefill actually buy the HSA backend?

Runs the same set of concurrent requests twice through the real serving path:

  * ``prefill_max_requests=1`` — one sequence per extend forward (what the
    backend was limited to before batch_size>1 support).
  * unset — the scheduler packs several sequences into one extend forward.

Per-sequence work inside an HSA layer is identical either way (the extend
helpers run once per sequence), so the win comes from amortising everything
else: one forward instead of N over the non-HSA layers, projections and MLPs,
and one scheduler step instead of N.  That makes short requests the interesting
case; a single long-context prefill already saturates the GPU on its own.

Usage: python dev/bench_hsa_batched_prefill.py [--model DIR]
"""

import argparse
import time

DEFAULT_MODEL = "dev/align/ckpt_345m_bench_hf"
CASES = ((16, 128), (8, 512), (4, 2048))  # (num requests, prompt tokens each)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def make(n_tok, salt):
        s = f"Document {salt}. " + "machines " * (n_tok * 2)
        return tok.decode(tok(s)["input_ids"][:n_tok])

    sp = {"temperature": 0.0, "max_new_tokens": 1}  # isolate prefill
    results = {}

    for label, pmr in (("prefill B=1", 1), ("prefill B>1", None)):
        kw = dict(
            model_path=args.model, attention_backend="hsa", page_size=64,
            trust_remote_code=True, mem_fraction_static=0.35,
            context_length=16384, disable_overlap_schedule=True,
            max_running_requests=64, disable_radix_cache=True,
            log_level="warning",
        )
        if pmr is not None:
            kw["prefill_max_requests"] = pmr
        llm = sgl.Engine(**kw)
        llm.generate(make(256, "warm"), sp)  # JIT / autotune
        for n_req, L in CASES:
            prompts = [make(L, f"{label}-{L}-{i}") for i in range(n_req)]
            best = float("inf")
            for _ in range(args.repeat):
                t0 = time.time()
                llm.generate(prompts, sp)
                best = min(best, time.time() - t0)
            results[(label, n_req, L)] = best
        llm.shutdown()

    print("=" * 74)
    print(f"{'case':<26}{'prefill B=1':>14}{'prefill B>1':>14}{'speedup':>12}")
    print("-" * 74)
    for n_req, L in CASES:
        a = results[("prefill B=1", n_req, L)]
        b = results[("prefill B>1", n_req, L)]
        print(f"{f'{n_req} x {L} tok':<26}{a:>13.3f}s{b:>13.3f}s{a / b:>11.2f}x")
    print("=" * 74)


if __name__ == "__main__":
    main()
