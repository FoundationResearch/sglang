#!/usr/bin/env python3
"""Parse per-model LongBench v1 result.json files and produce a LaTeX table."""

import argparse
import json
import os
import re
import sys

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ── Task categories (from eval_longbench_v1.py) ─────────────────
TASK_CATEGORIES = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    "Code": ["lcc", "repobench-p"],
}

CATEGORY_ORDER = ["Single-Doc QA", "Multi-Doc QA", "Summarization", "Few-shot", "Synthetic", "Code"]

ALL_DATASETS = [d for cat in CATEGORY_ORDER for d in TASK_CATEGORIES[cat]]

# Short names for LaTeX column headers
DATASET_SHORT = {
    "narrativeqa": "NQA",
    "qasper": "Qas",
    "multifieldqa_en": "MF-en",
    "multifieldqa_zh": "MF-zh",
    "hotpotqa": "HQA",
    "2wikimqa": "2Wiki",
    "musique": "Mus",
    "dureader": "DuR",
    "gov_report": "Gov",
    "qmsum": "QMS",
    "multi_news": "MN",
    "vcsum": "VCS",
    "trec": "TREC",
    "triviaqa": "TQA",
    "samsum": "Sam",
    "lsht": "LSHT",
    "passage_count": "PC",
    "passage_retrieval_en": "PR-en",
    "passage_retrieval_zh": "PR-zh",
    "lcc": "LCC",
    "repobench-p": "Repo",
}


# ── parse model name -> (base_name, step) ───────────────────────
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


# ── collect all data from LOG_DIR ────────────────────────────────
def collect(log_dir: str):
    """Return (lb_data, model_order).

    lb_data:     {model_name: {dataset: score}}
    model_order: list of model_name in discovery order
    """
    lb_data = {}
    seen_order = []

    for entry in sorted(os.listdir(log_dir)):
        result_path = os.path.join(log_dir, entry, "result.json")
        if not os.path.isfile(result_path):
            continue
        model_name = entry
        if model_name not in seen_order:
            seen_order.append(model_name)
        with open(result_path, "r", encoding="utf-8") as f:
            lb_data[model_name] = json.load(f)

    return lb_data, seen_order


# ── LongBench v1 LaTeX table ────────────────────────────────────
def make_longbench_table(lb_data, model_order):
    if not lb_data:
        return ""

    lines = []
    lines.append("% ── LongBench v1 Table ──")

    # Column layout: Model | Step | per-category avg columns | Overall
    # Then a detailed table with all 21 datasets
    # We do two tables: (1) category-level summary, (2) full per-dataset

    # ── Table 1: Category-level summary ──
    n_cats = len(CATEGORY_ORDER)
    # l|l| + n_cats category cols + 1 overall col
    col_spec = "l|l|" + "c" * n_cats + "|c"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    cat_short = {
        "Single-Doc QA": "S-QA",
        "Multi-Doc QA": "M-QA",
        "Summarization": "Sum",
        "Few-shot": "Few",
        "Synthetic": "Syn",
        "Code": "Code",
    }
    header_parts = ["Models", "Steps"]
    for cat in CATEGORY_ORDER:
        header_parts.append(cat_short.get(cat, cat))
    header_parts.append("Avg")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # group models by base name for multirow
    grouped = []
    prev_base = None
    for model_name in model_order:
        if model_name not in lb_data:
            continue
        base, step = _split_name_step(model_name)
        if prev_base is None or base != prev_base:
            grouped.append((base, []))
            prev_base = base
        grouped[-1][1].append((model_name, step))

    for base, members in grouped:
        n_members = len(members)
        for i, (model_name, step) in enumerate(members):
            d = lb_data[model_name]
            vals = []
            all_scores = []
            for cat in CATEGORY_ORDER:
                datasets = TASK_CATEGORIES[cat]
                cat_scores = [d[ds] for ds in datasets if ds in d]
                if cat_scores:
                    avg = _mean(cat_scores)
                    vals.append(f"{avg:.2f}")
                    all_scores.extend(cat_scores)
                else:
                    vals.append("-")

            # Overall
            if all_scores:
                vals.append(f"{_mean(all_scores):.2f}")
            else:
                vals.append("-")

            if i == 0 and n_members > 1:
                base_cell = f"\\multirow{{{n_members}}}{{*}}{{{base}}}"
            elif i == 0:
                base_cell = base
            else:
                base_cell = ""
            line = f"{base_cell} & {step} & " + " & ".join(vals) + r" \\"
            lines.append(line)
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append("")

    # ── Table 2: Full per-dataset breakdown ──
    lines.append("% ── LongBench v1 Full Breakdown ──")

    n_tasks_per_cat = [len(TASK_CATEGORIES[c]) for c in CATEGORY_ORDER]
    # l|l| + groups separated by |
    col_spec = "l|l|" + "|".join(["c" * n for n in n_tasks_per_cat]) + "|c"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # First header row: multicolumn per category
    parts = [r"\multirow{2}{*}{Models}", r"\multirow{2}{*}{Steps}"]
    for cat in CATEGORY_ORDER:
        n = len(TASK_CATEGORIES[cat])
        parts.append(f"\\multicolumn{{{n}}}{{c|}}{{{cat_short.get(cat, cat)}}}")
    parts.append(r"\multirow{2}{*}{Avg}")
    lines.append(" & ".join(parts) + r" \\")

    # Second header row: dataset short names
    parts = ["", ""]
    for cat in CATEGORY_ORDER:
        for ds in TASK_CATEGORIES[cat]:
            parts.append(DATASET_SHORT.get(ds, ds))
    parts.append("")
    lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\midrule")

    for base, members in grouped:
        n_members = len(members)
        for i, (model_name, step) in enumerate(members):
            d = lb_data[model_name]
            vals = []
            all_scores = []
            for cat in CATEGORY_ORDER:
                for ds in TASK_CATEGORIES[cat]:
                    if ds in d:
                        vals.append(f"{d[ds]:.1f}")
                        all_scores.append(d[ds])
                    else:
                        vals.append("-")
            # Overall
            if all_scores:
                vals.append(f"{_mean(all_scores):.2f}")
            else:
                vals.append("-")

            if i == 0 and n_members > 1:
                base_cell = f"\\multirow{{{n_members}}}{{*}}{{{base}}}"
            elif i == 0:
                base_cell = base
            else:
                base_cell = ""
            line = f"{base_cell} & {step} & " + " & ".join(vals) + r" \\"
            lines.append(line)
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ── plain text summary ───────────────────────────────────────────
def make_plain_summary(lb_data, model_order):
    """Generate a plain text summary table (similar to eval script output)."""
    lines = []
    for model_name in model_order:
        if model_name not in lb_data:
            continue
        d = lb_data[model_name]
        base, step = _split_name_step(model_name)
        lines.append(f"\n{'='*70}")
        lines.append(f"  Model: {model_name}")
        lines.append(f"{'='*70}")
        lines.append(f"  {'Category':<20} {'Datasets':>10} {'Avg Score':>12}")
        lines.append(f"{'='*70}")

        all_scores = []
        for cat in CATEGORY_ORDER:
            datasets = TASK_CATEGORIES[cat]
            cat_scores = [d[ds] for ds in datasets if ds in d]
            if cat_scores:
                avg = round(_mean(cat_scores), 2)
                all_scores.extend(cat_scores)
                detail = ", ".join(f"{ds}={d[ds]}" for ds in datasets if ds in d)
                lines.append(f"  {cat:<20} {len(cat_scores):>10} {avg:>12.2f}")
                lines.append(f"    {detail}")

        if all_scores:
            lines.append(f"{'='*70}")
            lines.append(f"  {'Overall':<20} {len(all_scores):>10} {round(_mean(all_scores), 2):>12.2f}")
        lines.append(f"{'='*70}")

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate LongBench v1 LaTeX summary tables")
    parser.add_argument("log_dir", help="Directory containing model subdirs with result.json")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file path (default: LOG_DIR/longbench_v1_summary.log)",
    )
    args = parser.parse_args()

    lb_data, model_order = collect(args.log_dir)

    if not lb_data:
        print("No result.json files found!", file=sys.stderr)
        sys.exit(1)

    parts = []
    parts.append(f"% Auto-generated LongBench v1 evaluation summary (LaTeX)")
    parts.append(f"% Log dir: {os.path.abspath(args.log_dir)}")
    parts.append(f"% Models: {len(lb_data)}")
    parts.append("")

    latex_table = make_longbench_table(lb_data, model_order)
    if latex_table:
        parts.append(latex_table)
        parts.append("")

    plain_summary = make_plain_summary(lb_data, model_order)
    if plain_summary:
        parts.append("")
        parts.append("% ── Plain Text Summary ──")
        parts.append(plain_summary)

    output_text = "\n".join(parts)

    out_path = args.output or os.path.join(args.log_dir, "longbench_v1_summary.log")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(output_text)
    print(f"\nSummary written to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
