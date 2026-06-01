"""Capture the kernel sequence executing during HSA + cuda graph bench.

Goal: prove the captured graph contains HSA's selector + hsa_decode_paged_fwd
+ chunk_weight kernels — NOT dense's _fwd_grouped_kernel_stage1.

If only dense kernels show up, we have a silent fallback (the earlier
ultrareview lesson).  If HSA kernels show up, R15 works as intended.
"""
import os, sys
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

LEN = int(os.environ.get("LEN", 32768))
USE_CG = os.environ.get("CG", "1") == "1"

sys.argv = [
    "bench",
    "--model-path", "/home/hal-alex/workspace/hsa345m_real",
    "--tp", "1", "--batch-size", "1",
    "--input-len", str(LEN), "--output-len", "8",
    "--context-length", str(LEN + 200),
    "--attention-backend", "hsa",
    "--mem-fraction-static", "0.40",
    "--trust-remote-code",
]
sys.argv += ["--cuda-graph-max-bs", "1"] if USE_CG else ["--disable-cuda-graph"]

import torch
from torch.profiler import profile, ProfilerActivity
import argparse
import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs

orig_decode = bob.decode
decode_calls = [0]
prof_holder = {"prof": None}


def wrapped(*a, **kw):
    decode_calls[0] += 1
    if decode_calls[0] == 4:
        print(f"[PROFILER] starting on decode #{decode_calls[0]}", flush=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False,
        ) as prof:
            for _ in range(3):
                ret = orig_decode(*a, **kw)
            torch.cuda.synchronize()
        prof_holder["prof"] = prof
        return ret
    return orig_decode(*a, **kw)


bob.decode = wrapped
parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
bob.BenchArgs.add_cli_args(parser)
args = parser.parse_args()
bob.main(ServerArgs.from_cli_args(args), bob.BenchArgs.from_cli_args(args))

prof = prof_holder["prof"]
print(f"\n=== Profiler ran (CG={USE_CG} LEN={LEN}) ===", flush=True)
if prof is None:
    print("ERROR: profiler did not run")
    sys.exit(1)

# Aggregate kernel names from the events list to catch ones that fired
# inside the captured graph (which the keyAverages table may collapse).
hsa_kernel_names = [
    "hsa_decode_paged_fwd",
    "fused_chunk_weight_per_qhead",
    "hsa_build_page_table_1",
    "online_softmax_topk",  # selector helper
]
dense_kernel_names = [
    "_fwd_grouped_kernel_stage1",
    "_fwd_kernel_stage1",
    "_fwd_grouped_kernel_stage2",
]

found_hsa = set()
found_dense = set()
all_kernel_names = set()
events = prof.events()
for ev in events:
    n = ev.name or ""
    if "kernel" in n.lower() or n.endswith("Fn") or "_kernel" in n:
        all_kernel_names.add(n)
    for hn in hsa_kernel_names:
        if hn in n:
            found_hsa.add(n)
    for dn in dense_kernel_names:
        if dn in n:
            found_dense.add(n)

print(f"\nHSA-specific kernels found ({len(found_hsa)}):")
for k in sorted(found_hsa):
    print(f"  {k}")
print(f"\nDense-only kernels found ({len(found_dense)}):")
for k in sorted(found_dense):
    print(f"  {k}")

if USE_CG:
    if found_hsa:
        print("\n[PASS] HSA kernels ran under cuda graph — NOT a silent fallback.")
    else:
        print("\n[FAIL] No HSA kernels detected — silent fallback to dense!")
else:
    print("\n(reference run for comparison)")

# Top GPU kernels
print("\nTop 20 kernels by GPU time:")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
