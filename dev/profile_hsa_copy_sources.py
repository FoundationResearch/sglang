"""Find where the 625 aten::copy_ calls come from during HSA prefill @16K.

Replaces torch.Tensor.copy_, .to, and .contiguous to log Python stack so
we can attribute the copies to specific call sites.
"""
import os
import sys
import collections
import traceback

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import torch
from torch.profiler import profile, ProfilerActivity

import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs
import argparse

sys.argv = [
    "profile_copy",
    "--model-path", "/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real",
    "--load-format", "dummy",
    "--tp", "1", "--batch-size", "1",
    "--input-len", "16384",
    "--output-len", "4",
    "--context-length", "16584",
    "--attention-backend", "hsa",
    "--page-size", "64",
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

# We'll patch ONLY during the second extend call to avoid the warmup noise.
extend_calls = [0]
orig_extend = bob.extend
counters = collections.Counter()

# Patch torch.Tensor.copy_ via wrap to capture stack
def make_logging_wrapper(name, real):
    def wrapped(self, *args, **kwargs):
        # Skip if not on CUDA tensor (don't care about CPU copies)
        if self.is_cuda:
            stack = traceback.extract_stack(limit=10)
            # Pick the frame closest to user code (skip torch internal frames)
            sglang_frames = [f for f in stack if "/sglang/srt/" in f.filename or "/sglang/models" in f.filename]
            if sglang_frames:
                f = sglang_frames[-1]
                key = f"{os.path.basename(f.filename)}:{f.lineno}"
                counters[key] += 1
        return real(self, *args, **kwargs)
    return wrapped


_orig_to = torch.Tensor.to


def to_wrapper(self, *args, **kwargs):
    if self.is_cuda and args and isinstance(args[0], torch.dtype):
        target_dtype = args[0]
        if target_dtype != self.dtype:
            stack = traceback.extract_stack(limit=10)
            sglang_frames = [f for f in stack if "/sglang/srt/" in f.filename or "/sglang/models" in f.filename]
            if sglang_frames:
                f = sglang_frames[-1]
                key = f"to({target_dtype}): {os.path.basename(f.filename)}:{f.lineno}"
                counters[key] += 1
    return _orig_to(self, *args, **kwargs)


_orig_contig = torch.Tensor.contiguous


def contig_wrapper(self, *args, **kwargs):
    if self.is_cuda:
        if not self.is_contiguous():
            stack = traceback.extract_stack(limit=10)
            sglang_frames = [f for f in stack if "/sglang/srt/" in f.filename or "/sglang/models" in f.filename]
            if sglang_frames:
                f = sglang_frames[-1]
                key = f"contig: {os.path.basename(f.filename)}:{f.lineno}"
                counters[key] += 1
    return _orig_contig(self, *args, **kwargs)


def wrapped_extend(*a, **kw):
    extend_calls[0] += 1
    if extend_calls[0] == 2:
        # Install patches
        torch.Tensor.to = to_wrapper
        torch.Tensor.contiguous = contig_wrapper
        ret = orig_extend(*a, **kw)
        torch.cuda.synchronize()
        # Uninstall
        torch.Tensor.to = _orig_to
        torch.Tensor.contiguous = _orig_contig

        print("\n=== TOP COPY/TO/CONTIG SOURCES (filter cuda only) ===\n", flush=True)
        for site, count in counters.most_common(40):
            print(f"  {count:5d}  {site}")
        return ret
    return orig_extend(*a, **kw)


bob.extend = wrapped_extend
bob.main(server_args, bench_args)
