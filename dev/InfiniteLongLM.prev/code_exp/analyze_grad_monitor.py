"""Post-hoc analysis of HSA_GRAD_MONITOR logs.

Targets two failure modes:
1. Hard blow-up (NaN/Inf) -- section [1].
2. Soft drift: grad norm gradually climbs (e.g. from ~0.1 to ~10) without any
   single step crossing a hard alert threshold -- sections [5] & [6].

Usage
-----
1. Run training with:

       export HSA_GRAD_MONITOR=1
       export HSA_GRAD_MONITOR_DIR=./grad_monitor_logs
       export HSA_GRAD_MONITOR_EVERY=1          # or e.g. 10 to subsample
       export HSA_GRAD_MONITOR_LAYERS=0,1,2     # optional: only these layers

   In your training loop, once per optimizer step, call:

       from models.FlashHSA.grad_monitor import set_step
       set_step(global_step)

2. After PPL drifts (or the run crashes):

       python code_exp/analyze_grad_monitor.py ./grad_monitor_logs

   Optional flags:
       --baseline-steps N  : use first N steps as baseline (default 50).
       --spike-mult K      : a tag is "spiked" when its current
                             grad_norm > K * baseline_median (default 5.0).
       --metric grad_norm  : or grad_absmax; controls which stat drives
                             spike detection (default grad_norm).
       --top-n N           : trim Top-N lists (default 20).
       --focus-step S      : drill down into step S (print ordered tag list).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(paths: List[str]):
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print("ERROR: pandas is required. pip install pandas", file=sys.stderr)
        sys.exit(1)
    import pandas as pd

    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.jsonl")))
        else:
            files.append(p)
    if not files:
        print(f"No .jsonl files found under: {paths}", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_json(f, lines=True))
        except ValueError as e:
            print(f"WARN: failed to read {f}: {e}", file=sys.stderr)
    if not dfs:
        print("All files failed to parse.", file=sys.stderr)
        sys.exit(1)
    df = pd.concat(dfs, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Section [0] global scalars trajectory (loss / total_grad_norm / lr)
# ---------------------------------------------------------------------------

def _scalars_trajectory(df, baseline_steps: int):
    s = df[df["event"] == "scalar"].copy()
    if s.empty:
        print("  (no global scalars logged; call log_loss / log_total_grad_norm)")
        return
    names = sorted(s["name"].dropna().unique().tolist())
    print(f"  Scalars logged: {names}")
    for name in names:
        sub = s[s["name"] == name].sort_values("step")
        if sub.empty or "value" not in sub.columns:
            continue
        values = sub["value"].astype(float)
        steps = sub["step"].astype(int)
        base = values[steps < steps.min() + baseline_steps]
        base_med = float(base.median()) if len(base) else float("nan")
        peak_idx = values.idxmax()
        peak_step = int(steps.loc[peak_idx])
        peak_val = float(values.loc[peak_idx])
        first_step = int(steps.min())
        last_val = float(values.iloc[-1])
        print(f"    [{name}] baseline_median={base_med:.4g}  "
              f"last_value={last_val:.4g}  "
              f"peak={peak_val:.4g} @ step {peak_step}  "
              f"first_step={first_step}")


# ---------------------------------------------------------------------------
# Section [1] hard non-finite
# ---------------------------------------------------------------------------

def _first_nonfinite(df):
    if "finite" not in df.columns:
        print("  (no 'finite' column)")
        return
    bad = df[(df["event"].isin(["bwd", "param_grad"])) &
             (df["finite"] == False)]  # noqa: E712
    if bad.empty:
        print("  (no non-finite gradients detected)")
        return
    bad = bad.sort_values(["step", "order"])
    first = bad.groupby("step").head(1)
    print(f"  First non-finite gradient per step ({len(first)} step(s)):")
    cols = [c for c in ["step", "order", "event", "name",
                        "nan_frac", "inf_frac"] if c in first.columns]
    print(first[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Section [2] top peaks
# ---------------------------------------------------------------------------

def _top_peaks(df, metric: str, top_n: int):
    bwd = df[df["event"].isin(["bwd", "param_grad"])].copy()
    if bwd.empty or metric not in bwd.columns:
        print(f"  (no data for metric {metric!r})")
        return
    peaks = (bwd.groupby("name")[metric]
                .max()
                .sort_values(ascending=False)
                .head(top_n))
    print(f"\n  Top {top_n} tensors by peak {metric}:")
    for name, val in peaks.items():
        print(f"    {val:>12.4g}   {name}")


# ---------------------------------------------------------------------------
# Section [3] alert timeline (kept for non-finite / hard threshold)
# ---------------------------------------------------------------------------

def _alert_timeline(df):
    if "alert" not in df.columns:
        print("  (no 'alert' column)")
        return
    alerts = df[df["alert"] == True]  # noqa: E712
    if alerts.empty:
        print("  (no alerts triggered)")
        return
    alerts = alerts.sort_values(["step", "order"])
    cols = [c for c in ["step", "order", "event", "name",
                        "grad_absmax", "grad_norm",
                        "nan_frac", "inf_frac"] if c in alerts.columns]
    print(f"  Alert timeline ({len(alerts)} events) -- first 30:")
    print(alerts[cols].head(30).to_string(index=False))


# ---------------------------------------------------------------------------
# Section [4] param grad growth ratio
# ---------------------------------------------------------------------------

def _layerwise_param_trend(df, top_n: int):
    p = df[df["event"] == "param_grad"].copy()
    if p.empty or "grad_norm" not in p.columns:
        return
    agg = (p.groupby(["step", "name"])["grad_norm"]
             .mean()
             .reset_index())
    last = agg.sort_values("step").groupby("name").tail(5)
    first = agg.sort_values("step").groupby("name").head(5)
    mean_last = last.groupby("name")["grad_norm"].mean()
    mean_first = first.groupby("name")["grad_norm"].mean().replace(0, 1e-30)
    ratio = (mean_last / mean_first).sort_values(ascending=False).head(top_n)
    print(f"  Top {top_n} params by grad_norm growth (last 5 / first 5):")
    for name, r in ratio.items():
        print(f"    x{r:>10.3g}   {name}")


# ---------------------------------------------------------------------------
# Section [5] Baseline-relative spike detection  (NEW)
#
# Per tag, compute a robust baseline (median) from the first ``baseline_steps``
# steps.  Then flag each (step, tag) point whose current metric exceeds
# ``spike_mult * baseline_median``.  The "first tag to spike at each step"
# is the prime suspect for a soft drift failure.
# ---------------------------------------------------------------------------

def _spike_detection(df, metric: str, baseline_steps: int, spike_mult: float,
                     top_n: int):
    import pandas as pd
    bwd = df[df["event"].isin(["bwd", "param_grad"])].copy()
    if bwd.empty or metric not in bwd.columns:
        print(f"  (no data for metric {metric!r})")
        return None

    # aggregate within a step (there may be multiple orders of the same
    # forward-tag due to gradient checkpointing etc.)
    g = (bwd.groupby(["step", "order", "name", "event"])[metric]
            .max()
            .reset_index())

    min_step = int(g["step"].min())
    baseline_cutoff = min_step + baseline_steps
    baseline = (g[g["step"] < baseline_cutoff]
                    .groupby("name")[metric]
                    .median()
                    .rename("baseline"))
    if baseline.empty:
        print("  (not enough baseline steps)")
        return None

    g = g.merge(baseline, on="name", how="left")
    g["baseline"] = g["baseline"].fillna(0.0).replace(0.0, 1e-30)
    g["ratio"] = g[metric] / g["baseline"]
    g["spiked"] = g["ratio"] >= spike_mult

    print(f"  baseline = median over steps [{min_step}, {baseline_cutoff}), "
          f"spike_mult = {spike_mult}x, metric = {metric}")

    spikes = g[g["spiked"]].sort_values(["step", "order"])
    if spikes.empty:
        print("  (no tag exceeded baseline * spike_mult)")
        return g

    # First spike per step -- THIS is the drift root-cause candidate.
    first = spikes.groupby("step").head(1)
    print(f"\n  First spiked tag at each step ({len(first)} steps flagged, "
          f"showing first 30):")
    cols = ["step", "order", "event", "name", "baseline", metric, "ratio"]
    fmt = first[cols].head(30).copy()
    for c in ("baseline", metric, "ratio"):
        fmt[c] = fmt[c].map(lambda v: f"{v:.4g}")
    print(fmt.to_string(index=False))

    # Rank tags by how often they are the first spike, and by peak ratio.
    earliest = (first.groupby("name")
                     .agg(first_step=("step", "min"),
                          n_times=("step", "count"),
                          peak_ratio=("ratio", "max"))
                     .sort_values(["first_step", "peak_ratio"],
                                  ascending=[True, False])
                     .head(top_n))
    print(f"\n  Top {top_n} 'earliest spiker' tags (who drifts first, most):")
    print(earliest.to_string())

    return g


# ---------------------------------------------------------------------------
# Section [6] Rolling trend for top-K drifting tags  (NEW)
# ---------------------------------------------------------------------------

def _rolling_trend(g, metric: str, top_n: int = 10):
    """Show the metric trajectory of the most-drifted tags."""
    if g is None or g.empty:
        return
    # Pick tags with the highest final/base ratio (averaged over the last 10%
    # of steps, to be robust to single-step spikes).
    max_step = int(g["step"].max())
    tail_cutoff = max_step - max(1, (max_step - int(g["step"].min())) // 10)
    tail = g[g["step"] >= tail_cutoff]
    tail_ratio = (tail.groupby("name")["ratio"]
                      .mean()
                      .sort_values(ascending=False)
                      .head(top_n))
    if tail_ratio.empty:
        return
    print(f"  Top {top_n} tags by tail/baseline ratio (tail = last 10% of steps):")
    for name, r in tail_ratio.items():
        print(f"    x{r:>10.3g}   {name}")

    # Sparse trajectory sample (10 evenly-spaced steps) for visual inspection.
    print(f"\n  Trajectory of top-5 drifting tags (metric = {metric}):")
    top5 = list(tail_ratio.head(5).index)
    sub = g[g["name"].isin(top5)].copy()
    steps = sorted(sub["step"].unique().tolist())
    if len(steps) > 10:
        idx = [int(i * (len(steps) - 1) / 9) for i in range(10)]
        steps = [steps[i] for i in idx]
    sub = sub[sub["step"].isin(steps)]
    pivot = sub.pivot_table(index="name", columns="step", values=metric,
                            aggfunc="max")
    pivot = pivot.reindex(top5)  # keep rank order
    # Scientific format, 3 sig figs
    print(pivot.applymap(lambda v: f"{v:.3g}" if v == v else "").to_string())


# ---------------------------------------------------------------------------
# Section [7] Focus on a single step (drill-down)
# ---------------------------------------------------------------------------

def _focus_step(df, metric: str, step: int):
    f = df[(df["step"] == step) &
           (df["event"].isin(["bwd", "param_grad"]))].copy()
    if f.empty:
        print(f"  (no bwd/param_grad records at step {step})")
        return
    f = f.sort_values("order")
    cols = [c for c in ["order", "event", "name", "grad_norm",
                        "grad_absmax", "finite"] if c in f.columns]
    print(f"  All gradient events at step {step} ({len(f)} rows, "
          f"sorted by forward order):")
    print(f[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="Directory or .jsonl file(s) from HSA_GRAD_MONITOR")
    ap.add_argument("--metric", default="grad_norm",
                    choices=["grad_norm", "grad_absmax"],
                    help="Which metric drives spike detection.")
    ap.add_argument("--baseline-steps", type=int, default=50,
                    help="Use the first N steps as robust baseline.")
    ap.add_argument("--spike-mult", type=float, default=5.0,
                    help="Flag points where metric > mult * baseline_median.")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--focus-step", type=int, default=None,
                    help="Drill into a specific step and print ordered tags.")
    args = ap.parse_args()

    df = _load(args.paths)
    steps = df["step"].dropna().astype(int)
    print(f"Loaded {len(df):,} records, steps {steps.min()} .. {steps.max()}")
    print(f"Metric = {args.metric} | baseline = first {args.baseline_steps} steps | "
          f"spike_mult = {args.spike_mult}")

    print("\n[0] Global scalars (loss / total_grad_norm / ...)")
    _scalars_trajectory(df, args.baseline_steps)

    print("\n[1] First non-finite gradient (hard blow-up)")
    _first_nonfinite(df)

    print(f"\n[2] Peak {args.metric} per tensor")
    _top_peaks(df, args.metric, args.top_n)

    print("\n[3] Hard-alert timeline (>alert_norm or non-finite)")
    _alert_timeline(df)

    print("\n[4] Parameter grad-norm growth (last 5 steps / first 5 steps)")
    _layerwise_param_trend(df, args.top_n)

    print("\n[5] Baseline-relative SPIKE detection  <-- use this for soft drift")
    g = _spike_detection(df, args.metric, args.baseline_steps,
                         args.spike_mult, args.top_n)

    print("\n[6] Rolling trend for drifting tags")
    _rolling_trend(g, args.metric, top_n=args.top_n)

    if args.focus_step is not None:
        print(f"\n[7] Drill-down: step = {args.focus_step}")
        _focus_step(df, args.metric, args.focus_step)


if __name__ == "__main__":
    main()
