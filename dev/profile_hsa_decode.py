"""Profile a single HSA decode step to identify the actual per-layer bottleneck.

Wraps sglang.bench_one_batch with a torch profiler hook around the decode phase
so we can see per-op CPU/GPU time and pinpoint which kernels are dominating.
"""
import os
import sys

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

# We patch torch profiler around the decode of bench_one_batch.
import torch
from torch.profiler import profile, ProfilerActivity, record_function

import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs
import argparse

# Parse args.
_argv = [
    "profile",
    "--model-path", "/home/hal-alex/workspace/hsa345m_real",
    "--load-format", "dummy",
    "--tp", "1", "--batch-size", "1",
    "--input-len", str(int(os.environ.get("LEN", 32768))),
    "--output-len", "8",
    "--context-length", str(int(os.environ.get("LEN", 32768)) + 200),
    "--attention-backend", "hsa",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.50",
    "--trust-remote-code",
]
if "HSA_TOPK_OVERRIDE" in os.environ:
    _argv.extend(["--hsa-topk", os.environ["HSA_TOPK_OVERRIDE"]])
sys.argv = _argv

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
bob.BenchArgs.add_cli_args(parser)
args = parser.parse_args()
server_args = ServerArgs.from_cli_args(args)
bench_args = bob.BenchArgs.from_cli_args(args)

# Patch latency_tester to wrap decode in profiler.
orig_decode = bob.decode

decode_calls = [0]
prof_path = "/tmp/hsa_decode_profile"


def wrapped_decode(*a, **kw):
    decode_calls[0] += 1
    if decode_calls[0] == 4:  # warm + 3 measured, then start profiler
        print(f"\n=== STARTING PROFILER ON DECODE #{decode_calls[0]} ===", flush=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(3):
                ret = orig_decode(*a, **kw)
            torch.cuda.synchronize()
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=25,
        ), flush=True)
        # also dump top CPU
        print("\n--- TOP CPU TIME ---", flush=True)
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=20,
        ), flush=True)
        prof.export_chrome_trace(prof_path + ".json")
        print(f"\nTrace written to {prof_path}.json", flush=True)
        return ret
    return orig_decode(*a, **kw)


bob.decode = wrapped_decode

bob.main(server_args, bench_args)
