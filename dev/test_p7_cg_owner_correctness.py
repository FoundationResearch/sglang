"""P7 owner-gating correctness: HSA decode CG-on vs CG-off (eager).

Dummy weights are deterministic (initialize_dummy_weights seed=1234), so a
CG-on run and a CG-off run in separate processes share identical weights and
identical synthetic input (np seed 42).  If the owner-gating CG fix is correct,
CG-on decode tokens + logits match CG-off (eager, never had the P7 bug).

Long INPUT_LEN exercises the per-q-head multi-chunk selection slow path
(cand pages / slots / all-layer LMK gather) — exactly where P7 lived.

Usage: python dev/test_p7_cg_owner_correctness.py {off|on} <out_path>
"""
import os, sys, argparse
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import numpy as np
import torch

mode = sys.argv[1] if len(sys.argv) > 1 else "off"
out_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/p7_cg_{mode}.pt"

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import sglang.bench_one_batch as bob
from sglang.srt.server_args import ServerArgs

cg_on = (mode == "on")
INPUT_LEN = int(os.environ.get("INPUT_LEN", 16384))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", 6))
MODEL = os.environ.get(
    "MODEL", "dev/bench_models/hsa345m_real_hd128"
)

sys.argv = [
    "bench",
    "--model-path", MODEL,
    "--load-format", "dummy",
    "--tp", "1", "--batch-size", "1",
    "--input-len", str(INPUT_LEN),
    "--output-len", str(OUTPUT_LEN),
    "--context-length", str(INPUT_LEN + OUTPUT_LEN + 256),
    "--attention-backend", "hsa",
    "--page-size", "64",
    "--max-running-requests", "1",
    "--mem-fraction-static", "0.50",
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

np.random.seed(42)
server_args = ServerArgs.from_cli_args(args)
bench_args = bob.BenchArgs.from_cli_args(args)

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
        captured["decode_logits"] = logits.detach().clone().cpu().float()
    return ret


bob.decode = patched_decode
bob.main(server_args, bench_args)
torch.save(captured, out_path)

print(f"\nSaved to {out_path}")
print(f"decode tokens (steps 1..{decode_count[0]}): {captured['decode_tokens']}")
if captured["decode_logits"] is not None:
    l = captured["decode_logits"]
    print(f"step-2 logits: shape={tuple(l.shape)} finite={torch.isfinite(l).all().item()} "
          f"argmax={l.argmax(dim=-1).tolist()}")
