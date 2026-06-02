"""Trace page_table_1 + seq_lens at each HSA forward call to see if cuda
graph captures stale data, or if sglang somehow keeps them fresh."""
import os, sys

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
sys.argv = [
    "bench",
    "--model-path", "/home/hal-alex/workspace/hsa345m_real",
    "--load-format", "dummy", "--tp", "1", "--batch-size", "1",
    "--input-len", "8192", "--output-len", "6",
    "--context-length", "8392",
    "--attention-backend", "hsa",
    "--cuda-graph-max-bs", "1",
    "--mem-fraction-static", "0.40",
    "--trust-remote-code",
]

import torch
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend

_orig_init = HSAAttnBackend.init_forward_metadata
_init_calls = [0]
def _logged_init(self, forward_batch):
    _init_calls[0] += 1
    print(
        f"[INIT #{_init_calls[0]}] mode={forward_batch.forward_mode.name} "
        f"bs={forward_batch.batch_size} seq_lens={forward_batch.seq_lens.tolist()} "
        f"out_cache_loc.head={forward_batch.out_cache_loc[:2].tolist() if forward_batch.out_cache_loc is not None else None}",
        flush=True,
    )
    ret = _orig_init(self, forward_batch)
    md = self.forward_metadata
    if md is not None:
        pt = md.page_table_1
        cs = md.cache_seqlens_int32
        print(
            f"[INIT #{_init_calls[0]}] page_table_1 ptr=0x{pt.data_ptr():x} "
            f"shape={tuple(pt.shape)} content[0,:5]={pt[0, :5].tolist()} "
            f"content[0,seqlen-1]={pt[0, int(cs[0])-1].item() if cs[0] > 0 else None}",
            flush=True,
        )
        print(
            f"[INIT #{_init_calls[0]}] cache_seqlens_int32 ptr=0x{cs.data_ptr():x} content={cs[:2].tolist()}",
            flush=True,
        )
    return ret
HSAAttnBackend.init_forward_metadata = _logged_init

_orig_fd = HSAAttnBackend.forward_decode
_fd_calls = [0]
def _logged_fd(self, *a, **kw):
    _fd_calls[0] += 1
    layer = a[3] if len(a) > 3 else kw.get("layer")
    if _fd_calls[0] <= 32 or _fd_calls[0] % 16 == 0:
        md = self.forward_metadata
        layer_id = getattr(layer, "layer_id", "?") if layer is not None else "?"
        if md is not None and md.page_table_1 is not None:
            pt = md.page_table_1
            cs = md.cache_seqlens_int32
            print(
                f"[FD #{_fd_calls[0]} L{layer_id}] page_table_1 ptr=0x{pt.data_ptr():x} "
                f"cs={cs[:2].tolist()} pt[0,cs-1]={pt[0, int(cs[0])-1].item() if cs[0]>0 else None}",
                flush=True,
            )
    return _orig_fd(self, *a, **kw)
HSAAttnBackend.forward_decode = _logged_fd

import argparse, sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs
p = argparse.ArgumentParser()
ServerArgs.add_cli_args(p)
bob.BenchArgs.add_cli_args(p)
args = p.parse_args()
bob.main(ServerArgs.from_cli_args(args), bob.BenchArgs.from_cli_args(args))

print(f"\n[SUMMARY] init_forward_metadata calls: {_init_calls[0]}")
print(f"[SUMMARY] forward_decode calls: {_fd_calls[0]}")
