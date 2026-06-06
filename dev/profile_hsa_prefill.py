"""Profile a single HSA prefill (extend) call to identify per-layer bottleneck.

Hooks bench_one_batch.extend so we profile the SECOND call (warm-up=#1, profiled=#2).
"""
import os
import sys

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import torch
from torch.profiler import profile, ProfilerActivity

import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs
import argparse

BACKEND = os.environ.get("BACKEND", "hsa")  # 'hsa' or 'triton'
MODEL = os.environ.get(
    "MODEL",
    "/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real"
    if BACKEND == "hsa"
    else "/home/hal-alex/workspace/sglang/dev/bench_models/dense345m_fair",
)

_argv = [
    "profile_prefill",
    "--model-path", MODEL,
    "--load-format", "dummy",
    "--tp", "1", "--batch-size", "1",
    "--input-len", str(int(os.environ.get("LEN", 16384))),
    "--output-len", "4",
    "--context-length", str(int(os.environ.get("LEN", 16384)) + 200),
    "--attention-backend", BACKEND,
    "--page-size", "64",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.50",
    "--trust-remote-code",
]
sys.argv = _argv

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
bob.BenchArgs.add_cli_args(parser)
args = parser.parse_args()
server_args = ServerArgs.from_cli_args(args)
bench_args = bob.BenchArgs.from_cli_args(args)

orig_extend = bob.extend
extend_calls = [0]
prof_path = f"/tmp/{BACKEND}_prefill_profile"


def wrapped_extend(*a, **kw):
    extend_calls[0] += 1
    if extend_calls[0] == 2:  # #1 = JIT/warm-up, #2 = measured
        print(f"\n=== STARTING {BACKEND.upper()} PREFILL PROFILER (call #{extend_calls[0]}) ===", flush=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            ret = orig_extend(*a, **kw)
            torch.cuda.synchronize()
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=30,
        ), flush=True)
        print("\n--- TOP CPU TIME ---", flush=True)
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=25,
        ), flush=True)
        prof.export_chrome_trace(prof_path + ".json")
        print(f"\nTrace written to {prof_path}.json", flush=True)
        return ret
    return orig_extend(*a, **kw)


bob.extend = wrapped_extend
bob.main(server_args, bench_args)
