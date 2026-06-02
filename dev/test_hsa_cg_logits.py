"""Determinism-based correctness test for HSA + cuda graph.

Run twice — once with CG off, once with CG on — with the SAME random seed
so the synthetic input is identical.  Real HSA-345M weights are used so
the model produces meaningful (non-NaN) output.

Then compare:
  * next_token_ids at each decode step
  * logits at one decode step (max abs diff, max rel diff)

If correctness is preserved, with-CG matches without-CG.

Usage: python dev/test_hsa_cg_logits.py {off|on} <out_path>
"""
import os, sys, argparse
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import numpy as np
import torch

mode = sys.argv[1] if len(sys.argv) > 1 else "off"
out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/hsa_cg_{mode}.pt"

# Set seed BEFORE any sglang import so the input synthesis is deterministic.
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs

cg_on = (mode == "on")
INPUT_LEN = int(os.environ.get("INPUT_LEN", 512))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", 4))

sys.argv = [
    "bench",
    "--model-path", "/home/hal-alex/workspace/aligned345m_bench",
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

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
bob.BenchArgs.add_cli_args(parser)
args = parser.parse_args()

# Re-seed RIGHT before bench's input synthesis (np.random.randint).
np.random.seed(42)

server_args = ServerArgs.from_cli_args(args)
bench_args = bob.BenchArgs.from_cli_args(args)

# Force np.random seed again right before prepare_synthetic_inputs is called.
_orig_prep = bob.prepare_synthetic_inputs_for_latency_test
def _seeded_prep(*a, **kw):
    np.random.seed(42)
    return _orig_prep(*a, **kw)
bob.prepare_synthetic_inputs_for_latency_test = _seeded_prep

captured = {"decode_tokens": [], "decode_logits": None}
orig_decode = bob.decode
decode_count = [0]


def patched_decode(input_ids, batch, model_runner):
    decode_count[0] += 1
    ret = orig_decode(input_ids, batch, model_runner)
    next_tokens, logits = ret
    captured["decode_tokens"].append(next_tokens.detach().cpu().tolist())
    if decode_count[0] == 2:
        captured["decode_logits"] = logits.detach().clone().cpu()
    return ret


bob.decode = patched_decode
bob.main(server_args, bench_args)
torch.save(captured, out_path)

print(f"\nSaved to {out_path}")
print(f"decode tokens (steps 1..{decode_count[0]}): {captured['decode_tokens']}")
if captured["decode_logits"] is not None:
    l = captured["decode_logits"]
    print(f"step-2 logits: shape={tuple(l.shape)} dtype={l.dtype}")
    print(f"  finite: {torch.isfinite(l).all().item()}")
    print(f"  NaN count: {torch.isnan(l).sum().item()}")
    print(f"  argmax: {l.argmax(dim=-1).tolist()}")
    print(f"  max: {l.max().item():.4f}  min: {l.min().item():.4f}")
