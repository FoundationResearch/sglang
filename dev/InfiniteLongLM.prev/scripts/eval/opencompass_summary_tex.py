import csv, io, os, sys, re

# Accept multiple log dirs as arguments
log_dirs = sys.argv[1:]
if not log_dirs:
    print("Usage: python opencompass_summary_tex.py <log_dir1> [log_dir2] ...")
    sys.exit(1)

DATASET_ORDER = [
    ("cmath_gen",          "CMATH"),
    ("gsm8k_gen",          "GSM8K"),
    ("cruxeval_o_gen",     "CRUXEval"),
    ("humaneval_plus_gen", "HumanEval+"),
    ("mbpp_plus_gen",      "MBPP+"),
]

# For these datasets, extract a specific metric instead of overall_average.
# Key: dataset internal name, Value: (dataset_col, metric_col) to match in CSV.
DATASET_SPECIFIC_METRIC = {
    "humaneval_plus_gen": ("humaneval_plus", "humaneval_plus_plus_pass_1"),
    "mbpp_plus_gen":      ("mbpp_plus",      "mbpp_plus_plus_pass_1"),
}


def extract_score(summary_path, ds_internal):
    """Extract the score from a summary txt file.

    For datasets listed in DATASET_SPECIFIC_METRIC, extract the specific
    metric row (e.g. humaneval_plus_plus_pass_1) instead of overall_average.
    For all other datasets, extract overall_average as before.
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()
    csv_marker = "csv format\n"
    idx = text.find(csv_marker)
    if idx < 0:
        return None
    csv_text = text[idx + len(csv_marker):]
    caret_end = csv_text.find("\n")
    if caret_end < 0:
        return None
    csv_text = csv_text[caret_end + 1:].lstrip("\n")
    lines = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        if "," not in line:
            if lines:
                break
            continue
        lines.append(line)
    if not lines:
        return None
    reader = csv.reader(io.StringIO("\n".join(lines)))
    rows = list(reader)

    specific = DATASET_SPECIFIC_METRIC.get(ds_internal)
    if specific:
        target_dataset, target_metric = specific
        for row in rows:
            if len(row) >= 3 and row[0].strip() == target_dataset and row[2].strip() == target_metric:
                for cell in reversed(row):
                    cell = cell.strip()
                    if not cell or cell == "-":
                        continue
                    try:
                        return float(cell)
                    except ValueError:
                        continue
        return None
    else:
        for row in rows:
            if row and row[0].strip() == "overall_average":
                for cell in reversed(row):
                    cell = cell.strip()
                    if not cell or cell == "-":
                        continue
                    try:
                        return float(cell)
                    except ValueError:
                        continue
        return None


def find_summary_file(dataset_dir):
    """Find the latest summary_*.txt under a dataset output dir."""
    candidates = sorted(
        os.path.join(r, f)
        for r, _, fs in os.walk(dataset_dir)
        for f in fs
        if f.startswith("summary_") and f.endswith(".txt")
    )
    return candidates[-1] if candidates else None


# Discover all models from all log dirs
all_models = set()
for log_dir in log_dirs:
    if not os.path.isdir(log_dir):
        continue
    for model_name in os.listdir(log_dir):
        model_path = os.path.join(log_dir, model_name)
        if os.path.isdir(model_path) and not model_name.startswith(".") and model_name not in ("queue.log", "summary.log"):
            all_models.add(model_name)

# Sort models: extract step number for natural ordering
def model_sort_key(name):
    m = re.search(r"step(\d+)", name)
    return (name.rsplit("-step", 1)[0] if "-step" in name else name,
            int(m.group(1)) if m else 0)

all_models = sorted(all_models, key=model_sort_key)

# Collect scores: model -> dataset_internal -> score
scores = {}
for model_name in all_models:
    scores[model_name] = {}
    for ds_internal, _ in DATASET_ORDER:
        # Search across all log dirs for this model+dataset combo
        for log_dir in log_dirs:
            dataset_dir = os.path.join(log_dir, model_name, ds_internal)
            if not os.path.isdir(dataset_dir):
                continue
            summary_file = find_summary_file(dataset_dir)
            if summary_file:
                val = extract_score(summary_file, ds_internal)
                if val is not None:
                    scores[model_name][ds_internal] = val
                    break  # found score, no need to check other log dirs

# Build LaTeX output
display_names = [dn for _, dn in DATASET_ORDER]
n_datasets = len(display_names)

lines = []
lines.append("% Auto-generated evaluation summary")
for log_dir in log_dirs:
    lines.append(f"% Log dir: {log_dir}")
lines.append("")

# Table header
col_spec = "l" + "c" * (n_datasets + 1)  # +1 for AVG
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\begin{tabular}{" + col_spec + "}")
lines.append(r"\toprule")
header_cols = " & ".join(display_names + ["AVG"])
lines.append(f"Model & {header_cols} \\\\")
lines.append(r"\midrule")

for model_name in all_models:
    row_vals = []
    valid_scores = []
    for ds_internal, _ in DATASET_ORDER:
        val = scores[model_name].get(ds_internal)
        if val is not None:
            row_vals.append(f"{val:.2f}")
            valid_scores.append(val)
        else:
            row_vals.append("-")
    if valid_scores:
        avg = sum(valid_scores) / len(valid_scores)
        row_vals.append(f"{avg:.2f}")
    else:
        row_vals.append("-")
    row_str = " & ".join(row_vals)
    # Escape underscores in model name for LaTeX
    safe_name = model_name.replace("_", r"\_")
    lines.append(f"{safe_name} & {row_str} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{Evaluation results on math and code benchmarks.}")
lines.append(r"\end{table}")

output = "\n".join(lines) + "\n"

# Write summary to the first log dir
summary_path = os.path.join(log_dirs[0], "summary_tex.log")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(output)

print(f"LaTeX summary written to: {summary_path}")
print()
print(output)
