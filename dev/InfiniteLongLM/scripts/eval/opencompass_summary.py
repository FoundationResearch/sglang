import csv, io, os, sys, re

master_log_dir = sys.argv[1]

DATASET_ORDER = [
    ("mmlu_ppl_ac766d",              "MMLU(5-shot)"),
    ("gpqa_few_shot_ppl_4b5a83",     "GPQA(5-shot)"),
    ("hellaswag_10shot_ppl_59c85e",  "Hellaswag(10-shot)"),
    ("ARC_c_few_shot_ppl",           "ARC-c(25-shot)"),
    ("SuperGLUE_BoolQ_few_shot_ppl", "BoolQ(5-shot)"),
    ("race_few_shot_ppl",            "Race(3-shot)"),
]

def sanitize_dataset_tag(name):
    tag = name.replace(",", "_").replace("/", "_")
    tag = re.sub(r"[^A-Za-z0-9_.\-]", "", tag)
    return tag

def extract_overall_average(summary_path):
    """Extract the overall_average score from a summary txt file."""
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

def find_summary_file(model_dir):
    """Find the latest summary_*.txt under a model output dir."""
    candidates = sorted(
        os.path.join(r, f)
        for r, _, fs in os.walk(model_dir)
        for f in fs
        if f.startswith("summary_") and f.endswith(".txt")
    )
    return candidates[-1] if candidates else None

# Discover all models from subdirectories
all_models = set()
for ds_internal, _ in DATASET_ORDER:
    ds_tag = sanitize_dataset_tag(ds_internal)
    oc_dir = os.path.join(master_log_dir, ds_tag, "opencompass_outputs")
    if os.path.isdir(oc_dir):
        for model_name in os.listdir(oc_dir):
            if os.path.isdir(os.path.join(oc_dir, model_name)):
                all_models.add(model_name)
all_models = sorted(all_models)

# Collect scores: model -> dataset_internal -> score
scores = {}
for model_name in all_models:
    scores[model_name] = {}
    for ds_internal, _ in DATASET_ORDER:
        ds_tag = sanitize_dataset_tag(ds_internal)
        model_dir = os.path.join(master_log_dir, ds_tag, "opencompass_outputs", model_name)
        if not os.path.isdir(model_dir):
            continue
        summary_file = find_summary_file(model_dir)
        if summary_file:
            val = extract_overall_average(summary_file)
            if val is not None:
                scores[model_name][ds_internal] = val

# Build LaTeX output
display_names = [dn for _, dn in DATASET_ORDER]
header_cols = " & ".join(display_names + ["AVG"])
header_line = f"& {header_cols}\\\\"

lines = []
lines.append("% Auto-generated evaluation summary")
lines.append(f"% Log dir: {master_log_dir}")
lines.append("")
lines.append(header_line)
lines.append("\\midrule")

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
    lines.append(f"{model_name} & {row_str}\\\\")

output = "\n".join(lines) + "\n"

summary_path = os.path.join(master_log_dir, "summary.log")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(output)

print(f"LaTeX summary written to: {summary_path}")
print()
print(output)