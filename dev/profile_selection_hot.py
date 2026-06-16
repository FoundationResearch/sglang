"""Trace selection_extend_batched's hot region to find launches that matter."""
import os
import sys
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
import torch
from torch.profiler import profile, ProfilerActivity, record_function

import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs
import argparse

sys.argv = [
    "p",
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

# Wrap key functions in record_function spans for visibility in the profile
import sglang.srt.layers.attention.hsa_backend as hb
orig_run_sel = hb.HSAAttnBackend._run_selection_extend_batched
orig_compute_swa = hb.HSAAttnBackend._compute_internal_swa_extend
orig_hsa_sparse = hb.HSAAttnBackend._hsa_sparse_attn_extend

def wrap_sel(self, *a, **kw):
    with record_function("RUN_SELECTION_EXTEND"):
        return orig_run_sel(self, *a, **kw)

def wrap_swa(self, *a, **kw):
    with record_function("COMPUTE_INTERNAL_SWA"):
        return orig_compute_swa(self, *a, **kw)

def wrap_sparse(self, *a, **kw):
    with record_function("HSA_SPARSE_ATTN"):
        return orig_hsa_sparse(self, *a, **kw)

hb.HSAAttnBackend._run_selection_extend_batched = wrap_sel
hb.HSAAttnBackend._compute_internal_swa_extend = wrap_swa
hb.HSAAttnBackend._hsa_sparse_attn_extend = wrap_sparse

orig_extend = bob.extend
extend_calls = [0]
def wrapped_extend(*a, **kw):
    extend_calls[0] += 1
    if extend_calls[0] == 2:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False,
        ) as prof:
            ret = orig_extend(*a, **kw)
            torch.cuda.synchronize()
        print("\n=== SECTION TOTALS ===\n", flush=True)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
            top_level_events_only=False,
        ), flush=True)
        return ret
    return orig_extend(*a, **kw)

bob.extend = wrapped_extend
bob.main(server_args, bench_args)
