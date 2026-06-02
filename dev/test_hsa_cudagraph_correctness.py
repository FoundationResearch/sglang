"""Compare HSA decode output: cuda-graph ON vs cuda-graph OFF.

If outputs match, cuda graph is correct.  If they diverge, the captured
graph is reading stale metadata and we need explicit HSA replay hooks.
"""
import os, sys, subprocess, re

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"


def run_bench(cg_on: bool) -> str:
    """Return stdout of bench_one_batch with deterministic seed."""
    cmd = [
        sys.executable, "-m", "sglang.bench_one_batch",
        "--model-path", "/home/hal-alex/workspace/hsa345m_real",
        "--load-format", "dummy",
        "--tp", "1", "--batch-size", "1",
        "--input-len", "8192", "--output-len", "8",
        "--context-length", "8392",
        "--attention-backend", "hsa",
        "--mem-fraction-static", "0.40",
        "--trust-remote-code",
        "--correctness-test",
    ]
    if not cg_on:
        cmd.append("--disable-cuda-graph")
    else:
        cmd.extend(["--cuda-graph-max-bs", "1"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONHASHSEED"] = "0"

    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    return p.stdout + "\n" + p.stderr


def extract_tokens(out: str):
    """Look for emitted decode token ids in the bench stdout."""
    # bench_one_batch with --correctness-test prints generated tokens.
    # Look for "Output" lines or token-id-looking sequences.
    toks = re.findall(r"output_token_ids: \[(.*?)\]", out)
    if not toks:
        toks = re.findall(r"generated[:\s]+\[(.*?)\]", out, re.IGNORECASE)
    return toks


if __name__ == "__main__":
    print("=== HSA cuda-graph OFF ===", flush=True)
    out_off = run_bench(cg_on=False)
    print(out_off[-3000:])
    print("\n=== HSA cuda-graph ON ===", flush=True)
    out_on = run_bench(cg_on=True)
    print(out_on[-3000:])

    # Token comparison
    t_off = extract_tokens(out_off)
    t_on = extract_tokens(out_on)
    print(f"\nTOKENS_OFF: {t_off}")
    print(f"TOKENS_ON:  {t_on}")
    if t_off and t_on and t_off == t_on:
        print("CORRECTNESS PASS — cuda-graph ON matches OFF")
    else:
        print("CORRECTNESS DIVERGED — cuda graph reads stale HSA metadata")
