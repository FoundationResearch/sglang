"""Profile dense(triton) decode for comparison vs HSA."""
import os
import sys

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import torch
from torch.profiler import profile, ProfilerActivity

import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs
import argparse

sys.argv = [
    "profile_dense",
    "--model-path", "/home/hal-alex/workspace/dense345m_fair",
    "--load-format", "dummy",
    "--tp", "1", "--batch-size", "1",
    "--input-len", str(int(os.environ.get("LEN", 16384))),
    "--output-len", "8",
    "--context-length", str(int(os.environ.get("LEN", 16384)) + 200),
    "--attention-backend", "triton",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.50",
    "--trust-remote-code",
]

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
bob.BenchArgs.add_cli_args(parser)
args = parser.parse_args()
server_args = ServerArgs.from_cli_args(args)
bench_args = bob.BenchArgs.from_cli_args(args)

orig_decode = bob.decode
decode_calls = [0]


def wrapped_decode(*a, **kw):
    decode_calls[0] += 1
    if decode_calls[0] == 4:
        print(f"\n=== STARTING DENSE PROFILER ===", flush=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        ) as prof:
            for _ in range(3):
                ret = orig_decode(*a, **kw)
            torch.cuda.synchronize()
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=20,
        ), flush=True)
        return ret
    return orig_decode(*a, **kw)


bob.decode = wrapped_decode
bob.main(server_args, bench_args)
