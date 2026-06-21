"""P5 repro: cross-request lmk cache under radix prefix reuse.

Same engine (so weights are identical), greedy decode, compared by output token ids:
  fresh : flush cache, run B alone  -> prefix fully prefilled, lmk written for all chunks
  reuse : flush cache, run A (caches shared prefix), then run B -> B hits radix prefix,
          prefix prefill SKIPPED, so lmk for prefix chunks is never written for B.

If fresh != reuse, the lmk pool is not restored across the radix prefix reuse (P5).

Run: CUDA_VISIBLE_DEVICES=5 python dev/repro_p5_cache_reuse.py
"""
import os
import sys

import sglang as sgl

MODEL = "dev/bench_models/hsa345m_real"
PREFIX_LEN = 3200          # >3000 (friend's condition), 50 chunks at chunk_size=64
SUF = 64
GEN = 24
VOCAB = 50000              # safe id range for dummy weights

# Deterministic pseudo-random token ids (no RNG dependency across runs).
def ids(seed, n):
    out, x = [], (seed * 2654435761) & 0xFFFFFFFF
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x % VOCAB)
    return out

prefix = ids(1, PREFIX_LEN)
A = prefix + ids(2, SUF)
B = prefix + ids(3, SUF)

SP = {"temperature": 0.0, "max_new_tokens": GEN}


def gen(engine, input_ids):
    out = engine.generate(input_ids=input_ids, sampling_params=SP)
    o = out[0] if isinstance(out, list) else out
    return o["output_ids"] if "output_ids" in o else o["token_ids"]


def main():
    engine = sgl.Engine(
        model_path=MODEL, load_format="dummy", attention_backend="hsa",
        page_size=64, mem_fraction_static=0.5, trust_remote_code=True,
        disable_cuda_graph=True, disable_radix_cache=False,
        context_length=PREFIX_LEN + SUF + GEN + 64, skip_tokenizer_init=True,
    )
    try:
        engine.flush_cache()
        b_fresh = gen(engine, B)

        engine.flush_cache()
        _ = gen(engine, A)          # cache the shared prefix
        b_reuse = gen(engine, B)    # B should hit the radix prefix

        same = b_fresh == b_reuse
        nmatch = sum(int(x == y) for x, y in zip(b_fresh, b_reuse))
        print("\n================ P5 RESULT ================")
        print("B fresh :", b_fresh)
        print("B reuse :", b_reuse)
        print(f"match   : {nmatch}/{len(b_fresh)}  ->  {'IDENTICAL (no bug?)' if same else 'DIVERGES => P5 reproduced'}")
        print("===========================================")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
