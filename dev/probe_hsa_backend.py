"""Probe whether HSAAttnBackend.forward_decode is actually being called."""
import os
import sys

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
sys.argv = [
    "bench",
    "--model-path", "/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real",
    "--load-format", "dummy",
    "--tp", "1",
    "--batch-size", "1",
    "--input-len", "8192",
    "--output-len", "3",
    "--context-length", "8392",
    "--cuda-graph-max-bs", "1",
    "--mem-fraction-static", "0.5",
    "--trust-remote-code",
]

from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend

_orig_init = HSAAttnBackend.__init__
def _logged_init(self, *a, **kw):
    print("[PROBE] HSAAttnBackend.__init__ called", flush=True)
    return _orig_init(self, *a, **kw)
HSAAttnBackend.__init__ = _logged_init

_orig_fd = HSAAttnBackend.forward_decode
_fd_counter = [0]
def _logged_fd(self, *a, **kw):
    _fd_counter[0] += 1
    if _fd_counter[0] <= 4:
        layer_id = a[3].layer_id if len(a) > 3 else "?"
        print(f"[PROBE] HSAAttnBackend.forward_decode #{_fd_counter[0]} layer_id={layer_id}", flush=True)
    return _orig_fd(self, *a, **kw)
HSAAttnBackend.forward_decode = _logged_fd

# Also probe hsa_decode_paged_fwd
from sglang.srt.layers.attention.hsa.kernels import hsa_decode as _hd
_orig_pf = _hd.hsa_decode_paged_fwd
_pf_counter = [0]
def _logged_pf(*a, **kw):
    _pf_counter[0] += 1
    if _pf_counter[0] <= 4:
        print(f"[PROBE] hsa_decode_paged_fwd #{_pf_counter[0]}", flush=True)
    return _orig_pf(*a, **kw)
_hd.hsa_decode_paged_fwd = _logged_pf
# also patch the module's symbol in hsa_backend
import sglang.srt.layers.attention.hsa_backend as _hb
_hb.hsa_decode_paged_fwd = _logged_pf

import argparse
import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs
parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
bob.BenchArgs.add_cli_args(parser)
args = parser.parse_args()
server_args = ServerArgs.from_cli_args(args)
bench_args = bob.BenchArgs.from_cli_args(args)
bob.main(server_args, bench_args)
print(f"[PROBE] TOTAL HSAAttnBackend.forward_decode calls: {_fd_counter[0]}", flush=True)
print(f"[PROBE] TOTAL hsa_decode_paged_fwd calls:         {_pf_counter[0]}", flush=True)
