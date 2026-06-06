"""Compare HSA attention output (NOT model logits) with vs without cuda
graph.  We capture the output of HSAAttnBackend.forward_decode for each
layer at the second decode step.  These tensors are the direct output of
the HSA path and bypass any downstream NaN amplification.

If with-CG output matches without-CG output, HSA cuda graph integration
is numerically correct.

Run twice as separate processes (one CG off, one CG on) with the SAME
np.random seed so synthetic inputs are identical.
"""
import os, sys, argparse
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import numpy as np
import torch

mode = sys.argv[1] if len(sys.argv) > 1 else "off"
out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/hsa_cg_attn_{mode}.pt"

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend

cg_on = (mode == "on")
INPUT_LEN = int(os.environ.get("INPUT_LEN", 512))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", 4))

sys.argv = [
    "bench",
    "--model-path", "/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real",
    "--tp", "1", "--batch-size", "1",
    "--input-len", str(INPUT_LEN),
    "--output-len", str(OUTPUT_LEN),
    "--context-length", str(INPUT_LEN + OUTPUT_LEN + 100),
    "--attention-backend", "hsa",
    "--mem-fraction-static", "0.40",
    "--trust-remote-code",
]
if cg_on:
    sys.argv.extend(["--cuda-graph-max-bs", "1"])
else:
    sys.argv.append("--disable-cuda-graph")

# Re-seed bench's input synthesis to be deterministic.
_orig_prep = bob.prepare_synthetic_inputs_for_latency_test
def _seeded_prep(*a, **kw):
    np.random.seed(42)
    return _orig_prep(*a, **kw)
bob.prepare_synthetic_inputs_for_latency_test = _seeded_prep

# ----- Capture: hook forward_decode to record per-layer outputs ------
# We record at the 2nd decode step (after warmup) for every HSA layer.
captured = {"per_layer_out": {}, "decode_step": [0]}
orig_fd = HSAAttnBackend.forward_decode


def patched_fd(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
    out = orig_fd(self, q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)
    step = captured["decode_step"][0]
    if step == 2:  # 2nd actual decode step
        captured["per_layer_out"][int(layer.layer_id)] = out.detach().clone().cpu()
    return out


HSAAttnBackend.forward_decode = patched_fd

# Track decode step boundaries via the outer decode() wrapper.
orig_decode = bob.decode
def patched_decode(input_ids, batch, model_runner):
    captured["decode_step"][0] += 1
    return orig_decode(input_ids, batch, model_runner)
bob.decode = patched_decode

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
bob.BenchArgs.add_cli_args(parser)
args = parser.parse_args()
bob.main(ServerArgs.from_cli_args(args), bob.BenchArgs.from_cli_args(args))

torch.save(captured["per_layer_out"], out_path)
print(f"\nSaved {len(captured['per_layer_out'])} layers to {out_path}")
for lid in sorted(captured["per_layer_out"].keys()):
    t = captured["per_layer_out"][lid]
    finite = torch.isfinite(t).all().item()
    print(
        f"  L{lid}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"finite={finite} abs_max={t.abs().max().item():.3e}"
    )
