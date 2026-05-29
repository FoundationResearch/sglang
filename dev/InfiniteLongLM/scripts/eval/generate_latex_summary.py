#!/usr/bin/env python3
"""Parse per-model summary logs and produce LaTeX tables for PPL and RULER."""

import argparse
import json
import os
import re
import sys
from collections import defaultdict


# ── human-readable length labels ──────────────────────────────────
def _len_label(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}M"
    if n >= 1024:
        return f"{n // 1024}K"
    return str(n)


# ── parse model name → (base_name, step) ─────────────────────────
_STEP_RE = re.compile(r"^(.+?)-step(\d+)$")


def _step_label(step_str: str) -> str:
    """Convert step number to human-readable: 1000 -> '1k', 14500 -> '14.5k'."""
    if step_str == "-":
        return "-"
    try:
        n = int(step_str)
    except ValueError:
        return step_str
    if n >= 1000:
        val = n / 1000
        if val == int(val):
            return f"{int(val)}k"
        return f"{val:g}k"
    return step_str


def _split_name_step(model_name: str):
    m = _STEP_RE.match(model_name)
    if m:
        return m.group(1), _step_label(m.group(2))
    return model_name, "-"


# ── parse PPL summary log ────────────────────────────────────────
_PPL_RE = re.compile(
    r"Test Length:\s*(\d+),\s*Final Mean Loss:\s*[\d.]+,\s*PPL:\s*([\d.]+)"
)


def parse_ppl_log(path: str) -> dict:
    """Return {seq_len_int: ppl_float}."""
    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _PPL_RE.search(line)
            if m:
                results[int(m.group(1))] = float(m.group(2))
    return results


# ── parse RULER summary log (JSONL) ──────────────────────────────
# task_id -> short name mapping
RULER_TASK_SHORT = {0: "S-N", 1: "MQ-N", 2: "VT"}


def parse_ruler_log(path: str) -> dict:
    """Return {(seq_len_int, task_id_int): exact_match_rate}."""
    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (int(obj["max_seq_len"]), int(obj["task_id"]))
            results[key] = float(obj["exact_match_rate"])
    return results


# ── collect all data from LOG_DIR ─────────────────────────────────
def collect(log_dir: str):
    """Return (ppl_data, ruler_data, model_order).

    ppl_data:   {model_name: {seq_len: ppl}}
    ruler_data: {model_name: {(seq_len, task_id): exact_match_rate}}
    model_order: list of model_name in discovery order
    """
    ppl_data = {}
    ruler_data = {}
    seen_order = []

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".summary.log"):
            continue
        fpath = os.path.join(log_dir, fname)

        if fname.endswith(".ppl.summary.log"):
            model_name = fname.rsplit(".ppl.summary.log", 1)[0]
            if model_name not in seen_order:
                seen_order.append(model_name)
            ppl_data[model_name] = parse_ppl_log(fpath)

        elif ".ruler.task" in fname:
            # e.g. innerx-step1000.ruler.task0.summary.log
            model_name = fname.split(".ruler.task")[0]
            if model_name not in seen_order:
                seen_order.append(model_name)
            if model_name not in ruler_data:
                ruler_data[model_name] = {}
            ruler_data[model_name].update(parse_ruler_log(fpath))

    return ppl_data, ruler_data, seen_order


# ── PPL LaTeX table ───────────────────────────────────────────────
# Fixed column order for PPL table (always present even if no data)
_PPL_SEQ_LENS = [64, 128, 512, 8192, 16384, 65536, 131072, 262144, 524288, 1048576]


def make_ppl_table(ppl_data, model_order):
    if not ppl_data:
        return ""
    # use fixed columns, but only keep those that appear in at least one model
    # always keep 64 in the list
    present_lens = {l for d in ppl_data.values() for l in d}
    all_lens = [l for l in _PPL_SEQ_LENS if l in present_lens or l == 64]
    if not all_lens:
        return ""

    len_labels = [_len_label(l) for l in all_lens]
    # extra empty column between Steps and data columns
    n_data_cols = len(all_lens)

    lines = []
    lines.append("% ── PPL Table ──")
    # l|l|c| + data cols  (the extra 'c' is the empty separator column)
    col_spec = "l|l|c|" + "c" * n_data_cols
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = "Models & Steps & & " + " & ".join(len_labels) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # group models by base name for multirow
    grouped = []  # [(base, [(model_name, step)])]
    prev_base = None
    for model_name in model_order:
        if model_name not in ppl_data:
            continue
        base, step = _split_name_step(model_name)
        if prev_base is None or base != prev_base:
            grouped.append((base, []))
            prev_base = base
        grouped[-1][1].append((model_name, step))

    for base, members in grouped:
        n_members = len(members)
        for i, (model_name, step) in enumerate(members):
            d = ppl_data[model_name]
            vals = []
            for l in all_lens:
                if l in d:
                    vals.append(f"{d[l]:.2f}")
                else:
                    vals.append("-")
            if i == 0 and n_members > 1:
                base_cell = f"\\multirow{{{n_members}}}{{*}}{{{base}}}"
            elif i == 0:
                base_cell = base
            else:
                base_cell = ""
            # empty cell between Steps and data
            line = f"{base_cell} & {step} & & " + " & ".join(vals) + r" \\"
            lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ── RULER LaTeX table ─────────────────────────────────────────────
def make_ruler_table(ruler_data, model_order):
    if not ruler_data:
        return ""

    # collect all seq_lens and task_ids
    all_lens = sorted({k[0] for d in ruler_data.values() for k in d})
    all_tasks = sorted({k[1] for d in ruler_data.values() for k in d})
    if not all_lens or not all_tasks:
        return ""

    n_tasks = len(all_tasks)
    n_lens = len(all_lens)
    total_data_cols = n_lens * n_tasks

    lines = []
    lines.append("% ── RULER Table ──")

    # column spec: l l | ccc | ccc | ...
    col_spec = "l|l|" + "|".join(["c" * n_tasks] * n_lens)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # first header row: Models & Steps & multicolumn for each seq_len
    parts = ["\\multirow{2}{*}{Models}", "\\multirow{2}{*}{Steps}"]
    for l in all_lens:
        label = _len_label(l)
        parts.append(
            f"\\multicolumn{{{n_tasks}}}{{c|}}{{{label}}}"
        )
    lines.append(" & ".join(parts) + r" \\")

    # second header row: empty & empty & task short names repeated
    task_labels = [RULER_TASK_SHORT.get(t, f"T{t}") for t in all_tasks]
    parts = ["", ""]
    for _ in all_lens:
        parts.extend(task_labels)
    lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\midrule")

    # group models by base name for multirow
    grouped = []  # [(base, [(model_name, step)])]
    prev_base = None
    for model_name in model_order:
        if model_name not in ruler_data:
            continue
        base, step = _split_name_step(model_name)
        if prev_base is None or base != prev_base:
            grouped.append((base, []))
            prev_base = base
        grouped[-1][1].append((model_name, step))

    for base, members in grouped:
        n_members = len(members)
        for i, (model_name, step) in enumerate(members):
            d = ruler_data[model_name]
            vals = []
            for l in all_lens:
                for t in all_tasks:
                    key = (l, t)
                    if key in d:
                        vals.append(f"{int(round(d[key] * 100))}")
                    else:
                        vals.append("")
            if i == 0 and n_members > 1:
                base_cell = f"\\multirow{{{n_members}}}{{*}}{{{base}}}"
            elif i == 0:
                base_cell = base
            else:
                base_cell = ""
            line = f"{base_cell} & {step} & " + " & ".join(vals) + r" \\"
            lines.append(line)
        lines.append(r"\midrule")

    # remove last \midrule and replace with \bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX summary tables")
    parser.add_argument("log_dir", help="Directory containing *.summary.log files")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file path (default: LOG_DIR/summary.log)",
    )
    args = parser.parse_args()

    ppl_data, ruler_data, model_order = collect(args.log_dir)

    parts = []
    parts.append(f"% Auto-generated evaluation summary (LaTeX)")
    parts.append(f"% Log dir: {os.path.abspath(args.log_dir)}")
    parts.append("")

    ppl_table = make_ppl_table(ppl_data, model_order)
    if ppl_table:
        parts.append(ppl_table)
        parts.append("")

    ruler_table = make_ruler_table(ruler_data, model_order)
    if ruler_table:
        parts.append(ruler_table)
        parts.append("")

    output_text = "\n".join(parts)

    out_path = args.output or os.path.join(args.log_dir, "summary.log")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(output_text)
    print(f"\nSummary written to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
