"""Profile HSA decode kernel time at varying --hsa-topk.

Uses torch profiler to extract exact CUDA kernel times — bypasses the
~14ms Python overhead noise that drowns the topk signal in wall-time bench.

Per topk in {8,16,32,64,128}: profile 3 decode steps, sum self-CUDA time
for the topk-sensitive kernels:
  - aten::topk (sbtopk::gatherTopK) — selector top-K
  - hsa_decode_paged_fwd_kernel — sparse attention over selected pages
  - fused_selector_score_kernel
  - fused_chunk_weight_h_kv_kernel
"""
import os
import sys
import re
import subprocess

LEN = int(os.environ.get("LEN", "16384"))
TOPKS = [8, 16, 32, 64, 128]
TARGET_KERNELS = [
    "aten::topk",
    "hsa_decode_paged_fwd_kernel",
    "fused_selector_score_kernel",
    "fused_chunk_weight_h_kv_kernel",
    "fused_internal_swa_decode_kernel",
]
ALL_HSA_ATTN_KERNELS = TARGET_KERNELS + [
    "sgl_kernel::fused_inplace_qknorm",
    "sgl_kernel::store_cache",
    "sgl_kernel::apply_rope_pos_ids_cos_sin_cache",
]


def run_profile(topk):
    """Run profile_hsa_decode.py with HSA_TOPK and return per-kernel self-CUDA us dict."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")
    env["LEN"] = str(LEN)
    env["HSA_TOPK_OVERRIDE"] = str(topk)
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run(
        ["/home/hal-alex/miniconda3/envs/alexsg/bin/python",
         os.path.join(here, "profile_hsa_decode.py")],
        env=env, capture_output=True, text=True, timeout=600,
    )
    lines = out.stdout.splitlines()
    # Parse the cuda-time table; columns are space-padded.
    kernel_us = {}
    in_table = False
    for line in lines:
        if "STARTING PROFILER" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if "Self CUDA time total" in line:
            m = re.search(r"Self CUDA time total:\s*([\d.]+)\s*ms", line)
            if m:
                kernel_us["__total__"] = float(m.group(1)) * 1000  # us
            break
        # The torch profiler table columns are (in order):
        #   Name, Self CPU %, Self CPU, CPU total %, CPU total, CPU time avg,
        #   Self CUDA, Self CUDA %, CUDA total, CUDA time avg, # of Calls
        # Self CUDA is the 7th DATA column (index 6 when splitting by 2+ whitespace).
        cols = re.split(r"\s{2,}", line.strip())
        if len(cols) < 11:
            continue
        name = cols[0]
        self_cuda_str = cols[6]  # 7th column (0-indexed: 6)
        m = re.match(r"([\d.]+)\s*([um]?s)", self_cuda_str)
        if not m:
            continue
        val, unit = float(m.group(1)), m.group(2)
        us = val * (1000.0 if unit == "ms" else (1.0 if unit == "us" else 1e6))
        for target in ALL_HSA_ATTN_KERNELS:
            if target in name:
                kernel_us[target] = us
                break
    return kernel_us, out.stdout


def main():
    print(f"==== HSA topk profile sweep at L={LEN} ====\n")
    cols = ["topk"] + TARGET_KERNELS + ["sum_topk_sensitive", "total_self_cuda"]
    print("  " + "  ".join(f"{c:<32}" if c in TARGET_KERNELS else f"{c:<16}" for c in cols))
    print("  " + "-" * 220)
    results = []
    for topk in TOPKS:
        print(f"\n  >>> profiling topk={topk} ...", flush=True)
        kernel_us, raw = run_profile(topk)
        sum_sensitive = sum(kernel_us.get(k, 0) for k in TARGET_KERNELS)
        total = kernel_us.get("__total__", 0)
        row = [topk] + [kernel_us.get(k, 0) for k in TARGET_KERNELS] + [sum_sensitive, total]
        results.append(row)
        cells = [f"{topk:<6}"]
        for k in TARGET_KERNELS:
            cells.append(f"{kernel_us.get(k, 0):>10.1f}us".ljust(32))
        cells.append(f"{sum_sensitive:>10.1f}us".ljust(16))
        cells.append(f"{total:>10.1f}us".ljust(16))
        print("  " + "  ".join(cells))

    print("\n\n==== SUMMARY ====")
    print(f"{'topk':<6} {'aten::topk':<14} {'hsa_decode_fwd':<16} {'selector_score':<16} "
          f"{'chunk_weight':<14} {'sum':<12} {'total':<12}")
    for row in results:
        topk = row[0]
        topk_us = row[1] / 3  # 3 decode steps profiled
        hsa_fwd_us = row[2] / 3
        sel_score = row[3] / 3
        chunk_w = row[4] / 3
        sum_us = row[6] / 3
        total = row[7] / 3
        print(f"{topk:<6} {topk_us:<14.1f} {hsa_fwd_us:<16.1f} {sel_score:<16.1f} "
              f"{chunk_w:<14.1f} {sum_us:<12.1f} {total:<12.1f}")


if __name__ == "__main__":
    main()
