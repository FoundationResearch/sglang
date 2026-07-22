#!/usr/bin/env python3
"""Does the HSA backend do real continuous batching?

batch_size > 1 alone only proves a *static* batch works. Continuous batching
means requests arrive at different steps and join a batch that is already
decoding, and finished requests leave it mid-flight. This fires requests at
staggered wall-clock times (via a thread pool) so a new prefill lands while
earlier requests are several decode steps in, then checks each request's output
against the same request run alone. If admission-into-a-live-decode-batch were
mishandled, the staggered outputs would diverge from the solo outputs.
"""
import sys, time, threading

MODEL = "dev/align/ckpt_345m_bench_hf"


def main() -> int:
    import sglang as sgl
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    base = "The history of computing began when "
    # Different lengths so they cross chunk boundaries at different steps.
    prompts = [base + " ".join(["machines"] * n) + ". In summary," for n in (30, 90, 200, 400)]
    sp = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 700}

    llm = sgl.Engine(
        model_path=MODEL, attention_backend="hsa", page_size=64,
        trust_remote_code=True, mem_fraction_static=0.30, context_length=4096,
        disable_overlap_schedule=True, prefill_max_requests=1,
        max_running_requests=8, log_level="warning",
    )

    # Reference: each prompt generated alone.
    solo = [llm.generate(p, sp)["text"] for p in prompts]

    # Staggered arrival: submit each prompt from its own thread, spaced out so a
    # new request is admitted while the earlier ones are mid-decode.
    results = [None] * len(prompts)
    def fire(i):
        results[i] = llm.generate(prompts[i], sp)["text"]
    threads = []
    for i, p in enumerate(prompts):
        t = threading.Thread(target=fire, args=(i,))
        t.start(); threads.append(t)
        time.sleep(0.25)  # ~tens of decode steps between arrivals
    for t in threads:
        t.join()

    ok = True
    for i in range(len(prompts)):
        ta, tb = tok(results[i])["input_ids"], tok(solo[i])["input_ids"]
        n = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]), min(len(ta), len(tb)))
        same = ta == tb
        # Same fp-drift tolerance as the batch test: must agree past the window.
        good = same or n >= 600
        ok &= good
        print(f"  req[{i}] ({len(ta)} tok): "
              + ("MATCH" if same else f"agrees {n} tok" if good else f"DIVERGES at {n} — too early"))
    print("RESULT:", "PASS" if ok else "FAIL")
    llm.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
