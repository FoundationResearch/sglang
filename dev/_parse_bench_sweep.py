"""Aggregate bench_regression_check.sh output into per-(L, backend) median table."""
import sys, re, statistics
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "-"
text = sys.stdin.read() if path == "-" else open(path).read()

rows = defaultdict(lambda: {"prefill": [], "decode": []})
header_seen = False
for line in text.splitlines():
    if line.startswith("L      ") and "backend" in line:
        header_seen = True
        continue
    if not header_seen:
        continue
    m = re.match(r"^\s*(\d+)\s+(\w+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s*$", line)
    if not m:
        continue
    L, backend, prefill_ms, decode_ms = int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
    rows[(L, backend)]["prefill"].append(prefill_ms)
    rows[(L, backend)]["decode"].append(decode_ms)

if not rows:
    print("(no rows parsed)"); sys.exit(1)

print(f"{'L':>7} {'backend':>9}   {'prefill_med':>12} {'prefill_best':>12} {'prefill_worst':>13}   {'decode_med':>10} {'decode_best':>11} {'decode_worst':>12}  N")
print("-" * 110)
all_L = sorted(set(L for (L, _) in rows))
for L in all_L:
    for backend in ("hsa", "triton"):
        key = (L, backend)
        if key not in rows:
            continue
        p = rows[key]["prefill"]; d = rows[key]["decode"]
        if not p:
            continue
        print(f"{L:>7} {backend:>9}   "
              f"{statistics.median(p):>12.2f} {min(p):>12.2f} {max(p):>13.2f}   "
              f"{statistics.median(d):>10.3f} {min(d):>11.3f} {max(d):>12.3f}  "
              f"{len(p)}")

print()
print("=== HSA vs Dense ratio (median) ===")
for L in all_L:
    h = rows.get((L, "hsa"))
    t = rows.get((L, "triton"))
    if not (h and t and h["prefill"] and t["prefill"]):
        continue
    hp = statistics.median(h["prefill"]); tp = statistics.median(t["prefill"])
    hd = statistics.median(h["decode"]); td = statistics.median(t["decode"])
    print(f"  L={L:>6}  prefill HSA/Dense = {hp/tp:.2f}x  ({hp:.1f} / {tp:.1f} ms)   "
          f"decode HSA/Dense = {hd/td:.2f}x  ({hd:.2f} / {td:.2f} ms)")
