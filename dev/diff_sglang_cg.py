"""Compare two SGLang verify-worker pickles (e.g. CG-on vs CG-off) on output
token ids + per-step top-k logprob sets. Used to validate the P7 owner-gating
CG fix end-to-end through the real sgl.Engine serving path.

Usage: python dev/diff_sglang_cg.py <ref.pkl> <test.pkl>
"""
import pickle, sys

ref = pickle.load(open(sys.argv[1], "rb"))
tst = pickle.load(open(sys.argv[2], "rb"))

ri = ref["decode_token_ids"]; ti = tst["decode_token_ids"]
n = min(len(ri), len(ti))
tok_match = sum(1 for i in range(n) if ri[i] == ti[i])
print(f"decode steps compared: {n}")
print(f"  ref tokens : {ri[:n]}")
print(f"  test tokens: {ti[:n]}")
print(f"  token match: {tok_match}/{n} ({tok_match/max(1,n)*100:.1f}%)")
varied = len(set(ri[:n])) > 1
print(f"  ref output varied (not collapsed): {varied}  (distinct={len(set(ri[:n]))})")

# top-k set overlap per step (from meta_info.output_top_logprobs: list of [ (lp, id, ...), ...])
def tops(res):
    m = res.get("meta_info", {})
    raw = m.get("output_top_logprobs")
    out = []
    if raw:
        for entry in raw:
            out.append(set(int(t[1]) for t in entry) if entry else set())
    return out

rt, tt = tops(ref), tops(tst)
if rt and tt:
    k = min(len(rt), len(tt))
    jacc = [len(rt[i] & tt[i]) / max(1, len(rt[i] | tt[i])) for i in range(k)]
    full = sum(1 for i in range(k) if rt[i] == tt[i])
    print(f"  top-k set identical: {full}/{k} ({full/max(1,k)*100:.1f}%)  mean Jaccard={sum(jacc)/max(1,k):.3f}")

verdict = "PASS (CG==eager)" if tok_match == n and varied else \
          ("TRIVIAL (collapsed output)" if not varied else "MISMATCH")
print(f"VERDICT: {verdict}")
