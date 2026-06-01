"""Parse the long-context sweep launcher capture into a clean per-bucket comparison table.

Usage:
  python dev/parse_long_ctx_sweep.py /tmp/.../b0b8u4y3g.output
"""
import re
import sys
from collections import defaultdict


def main(path):
    with open(path) as f:
        lines = f.readlines()

    # Group by (bucket, model)
    # State machine: track current bucket and current model header.
    results = defaultdict(dict)  # results[bucket][model] = {"prefill_ms": ..., "decode_ms": ...}
    bucket = None
    model = None
    bench_count = 0  # bench_one_batch prints warmup + benchmark; we want the SECOND (benchmark) values.
    cur_prefill = None
    cur_decode = None

    for raw in lines:
        s = raw.strip()
        # Bucket header
        m = re.match(r"CONTEXT BUCKET: (\d+)", s)
        if m:
            bucket = int(m.group(1))
            continue
        # Run header
        m = re.match(r"==== (\S+)\s+input=(\d+)", s)
        if m:
            model = m.group(1)
            bench_count = 0
            cur_prefill = None
            cur_decode = None
            continue

        m = re.match(r"Prefill\. latency: ([\d.]+) s, throughput:\s+([\d.]+) token/s", s)
        if m:
            cur_prefill = (float(m.group(1)) * 1000.0, float(m.group(2)))
            continue
        m = re.match(r"Decode\.  median latency: ([\d.]+) s, median throughput:\s+([\d.]+) token/s", s)
        if m:
            cur_decode = (float(m.group(1)) * 1000.0, float(m.group(2)))
            continue
        m = re.match(r"Total\. latency:\s+([\d.]+) s", s)
        if m:
            # End of one bench (warmup or benchmark).  We want the SECOND occurrence.
            bench_count += 1
            if bench_count == 2 and cur_prefill and cur_decode and bucket and model:
                results[bucket][model] = {
                    "prefill_ms": cur_prefill[0],
                    "prefill_tps": cur_prefill[1],
                    "decode_ms": cur_decode[0],
                    "decode_tps": cur_decode[1],
                }

    print(f"\n{'Length':>8} {'Method':<15} {'Prefill (ms)':>14} {'vs Qwen':>8} {'Decode (ms/tok)':>16} {'vs Qwen':>8}")
    print("-" * 80)

    methods_order = ["HSA-7B", "Qwen2.5-7B"]
    for bucket in sorted(results.keys()):
        row = results[bucket]
        if "Qwen2.5-7B" not in row:
            continue
        q_pf = row["Qwen2.5-7B"]["prefill_ms"]
        q_dc = row["Qwen2.5-7B"]["decode_ms"]
        for m in methods_order:
            if m not in row:
                continue
            pf = row[m]["prefill_ms"]
            dc = row[m]["decode_ms"]
            pf_ratio = q_pf / pf if pf > 0 else float("nan")
            dc_ratio = q_dc / dc if dc > 0 else float("nan")
            print(
                f"{bucket:>8} {m:<15} {pf:>14.2f} {pf_ratio:>7.2f}x "
                f"{dc:>16.3f} {dc_ratio:>7.2f}x"
            )
        print()


if __name__ == "__main__":
    main(sys.argv[1])
