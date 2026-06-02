#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

cd "$REPO_ROOT"
OPENCOMPASS_PATH=${OPENCOMPASS_PATH:-/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/opencompass}
export PYTHONPATH="$REPO_ROOT${OPENCOMPASS_PATH:+:$OPENCOMPASS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "${PYTHON_BIN:-}" ]; then
    true
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "python/python3 都不可用，请先激活运行环境" >&2
    exit 1
fi

# GPQA (5-shot) -> gpqa_few_shot_ppl_4b5a83
# HellaSwag (10-shot) -> hellaswag_10shot_ppl_59c85e
# ARC-c (25-shot) -> ARC_c_few_shot_ppl
# ARC-c (0-shot) -> ARC_c_ppl
# BoolQ (5-shot) -> SuperGLUE_BoolQ_few_shot_ppl

pkill -f "burner"

DATASETS=${DATASETS:-hellaswag_10shot_ppl_59c85e}
DEBUG=${DEBUG:-1}
TOKENIZER_PATH=${TOKENIZER_PATH:-/apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_vocab}
DATASET_TAG=$(printf '%s' "$DATASETS" | tr ',/' '__' | tr -cd '[:alnum:]_.-')

GPU_IDS=(
    0 1 2 3 4 5 6 7
)

MODEL_NAMES=(
    lhsa_dropout_step1000
    lhsa_dropout_step2000
    lhsa_dropout_step3000
    lhsa_dropout_step4000
    lhsa_dropout_step5000
    lhsa_dropout_step6000
    lhsa_dropout_fix_swa_mlp_step1000
    lhsa_dropout_fix_swa_mlp_step2000
    lhsa_dropout_fix_swa_mlp_step3000
    lhsa_dropout_fix_swa_mlp_step4000
)

HSA_CONFIGS=(
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
    /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_7B/olmo3_lhsa_dropout.json
)

HF_PATHS=(
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-trainall-40B/global_step_1000/hf_ckpt/
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-trainall-40B/global_step_2000/hf_ckpt/
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-trainall-40B/global_step_3000/hf_ckpt/
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-trainall-40B/global_step_4000/hf_ckpt/
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-trainall-40B/global_step_5000/hf_ckpt/
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-trainall-40B/global_step_6000/hf_ckpt/
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage1/global_step_1000/hf_ckpt/
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage1/global_step_2000/hf_ckpt/
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage1/global_step_3000/hf_ckpt/
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage1/global_step_4000/hf_ckpt/
)

if [ "${#MODEL_NAMES[@]}" -ne "${#HSA_CONFIGS[@]}" ] || \
   [ "${#MODEL_NAMES[@]}" -ne "${#HF_PATHS[@]}" ]; then
    echo "MODEL_NAMES / HSA_CONFIGS / HF_PATHS 长度必须一致" >&2
    exit 1
fi

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "GPU_IDS 不能为空" >&2
    exit 1
fi

if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "TOKENIZER_PATH 不存在: $TOKENIZER_PATH" >&2
    exit 1
fi

if [ ! -d "$OPENCOMPASS_PATH/opencompass" ]; then
    echo "OPENCOMPASS_PATH 不存在或不是 OpenCompass 仓库: $OPENCOMPASS_PATH" >&2
    exit 1
fi

LOG_DIR="$SCRIPT_DIR/logs/eval_olmo3_opencompass_${DATASET_TAG}_$(date +%Y%m%d_%H%M%S)"
WORK_DIR_ROOT="$LOG_DIR/opencompass_outputs"
mkdir -p "$WORK_DIR_ROOT"
echo "Logs will be saved to: $LOG_DIR"
echo "Datasets: $DATASETS"
echo "Queue size: ${#GPU_IDS[@]} GPU(s), ${#MODEL_NAMES[@]} model(s)"

append_overall_average_to_summary() {
    local summary_file="$1"

    "$PYTHON_BIN" - "$summary_file" <<'PY'
import csv
import io
import sys

summary_file = sys.argv[1]

with open(summary_file, "r", encoding="utf-8") as fin:
    text = fin.read()

if "overall_average" in text:
    raise SystemExit(0)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

csv_marker = "csv format\n"
divider_marker = "$" * 124
table_marker = "tabulate format\n"

csv_start = text.find(csv_marker)
if csv_start < 0:
    raise SystemExit(0)

csv_after_marker = text[csv_start + len(csv_marker):]
csv_caret_end = csv_after_marker.find("\n")
if csv_caret_end < 0:
    raise SystemExit(0)

csv_preamble = text[: csv_start + len(csv_marker) + csv_caret_end + 1]
csv_text = csv_after_marker[csv_caret_end + 1 :].lstrip("\n")
csv_lines = []
for line in csv_text.splitlines():
    if not line.strip():
        continue
    if "," not in line:
        if csv_lines:
            break
        continue
    csv_lines.append(line)

if not csv_lines:
    raise SystemExit(0)

reader = csv.reader(io.StringIO("\n".join(csv_lines)))
rows = list(reader)
if len(rows) < 2:
    raise SystemExit(0)

header = rows[0]
data_rows = rows[1:]
value_start = 4
if len(header) <= value_start:
    raise SystemExit(0)

column_sums = [0.0] * (len(header) - value_start)
column_counts = [0] * (len(header) - value_start)

for row in data_rows:
    if not row or row[0] == "overall_average":
        continue
    for i in range(value_start, min(len(row), len(header))):
        cell = row[i].strip()
        if not cell:
            continue
        try:
            value = float(cell)
        except ValueError:
            continue
        column_sums[i - value_start] += value
        column_counts[i - value_start] += 1

metric = next((row[2] for row in data_rows if len(row) > 2 and row[0] != "overall_average"), "average")
mode = next((row[3] for row in data_rows if len(row) > 3 and row[0] != "overall_average"), "average")
overall_row = ["overall_average", "-", metric, mode]
for total, count in zip(column_sums, column_counts):
    overall_row.append(f"{(total / count):.2f}" if count else "")

csv_rows = rows + [overall_row]
csv_output = io.StringIO()
writer = csv.writer(csv_output, lineterminator="\n")
writer.writerows(csv_rows)
csv_block = csv_output.getvalue().rstrip("\n")

table_start = text.find(table_marker)
if table_start >= 0:
    table_after_marker = text[table_start + len(table_marker):]
    table_caret_end = table_after_marker.find("\n")
    if table_caret_end >= 0:
        table_preamble = text[: table_start + len(table_marker) + table_caret_end + 1]
        divider_start = text.find(divider_marker)
        if divider_start >= 0:
            if tabulate is not None:
                table_block = tabulate(
                    data_rows + [overall_row],
                    headers=header,
                    tablefmt="simple",
                    stralign="left",
                    numalign="right",
                )
            else:
                table_rows = [header] + data_rows + [overall_row]
                widths = [0] * len(header)
                for row in table_rows:
                    for i, cell in enumerate(row):
                        widths[i] = max(widths[i], len(str(cell)))

                def format_table_row(row):
                    parts = []
                    for i, cell in enumerate(row):
                        cell = str(cell)
                        if i >= value_start:
                            parts.append(cell.rjust(widths[i]))
                        else:
                            parts.append(cell.ljust(widths[i]))
                    return "  ".join(parts)

                separator = "  ".join("-" * width for width in widths)
                table_block = "\n".join(
                    [format_table_row(header), separator]
                    + [format_table_row(row) for row in data_rows + [overall_row]]
                )
            text = table_preamble + table_block + "\n" + text[divider_start:]

csv_start = text.find(csv_marker)
csv_after_marker = text[csv_start + len(csv_marker):]
csv_caret_end = csv_after_marker.find("\n")
csv_preamble = text[: csv_start + len(csv_marker) + csv_caret_end + 1]
text = csv_preamble + csv_block + "\n"

with open(summary_file, "w", encoding="utf-8") as fout:
    fout.write(text)
PY
}

run_model_eval() {
    local gpu_id="$1"
    local model_name="$2"
    local hsa_config="$3"
    local hf_path="$4"
    local run_log="$LOG_DIR/${model_name}.run.log"
    local work_dir="$WORK_DIR_ROOT/${model_name}"
    local resolved_hf_path="$work_dir/hf_with_tokenizer"
    local insert_lmk
    local adjust_lmk_pos
    local chunk_size
    local cmd=(
        "$PYTHON_BIN" eval/eval_opencompass.py
        --datasets "$DATASETS"
        --hf-type base
        --hf-path "$resolved_hf_path"
        --hsa-config "$hsa_config"
        -w "$work_dir"
    )

    read -r insert_lmk adjust_lmk_pos chunk_size < <("$PYTHON_BIN" - "$hsa_config" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fin:
    config = json.load(fin)

insert_lmk = bool(config.get("insert_landmarks", False) or config.get("adjust_lmk_pos", False))
adjust_lmk_pos = bool(config.get("adjust_lmk_pos", False))
chunk_size = int(config.get("chunk_size", 64))
print(f'{"1" if insert_lmk else "0"} {"1" if adjust_lmk_pos else "0"} {chunk_size}')
PY
)

    cmd+=(--model-kwargs torch_dtype=torch.bfloat16 attn_implementation=flash_attention_3)
    if [ "$insert_lmk" = "1" ]; then
        cmd+=(auto_insert_lmk=True)
    fi

    if [ "$DEBUG" = "1" ]; then
        cmd+=(--debug)
    fi

    mkdir -p "$work_dir"
    rm -rf "$resolved_hf_path"
    mkdir -p "$resolved_hf_path"

    shopt -s nullglob dotglob
    for src in "$hf_path"/*; do
        ln -sfn "$src" "$resolved_hf_path/$(basename "$src")"
    done
    for src in "$TOKENIZER_PATH"/*; do
        ln -sfn "$src" "$resolved_hf_path/$(basename "$src")"
    done
    shopt -u nullglob dotglob
    "$PYTHON_BIN" - "$hsa_config" "$resolved_hf_path/config.json" "$insert_lmk" "$adjust_lmk_pos" <<'PY'
import json
import sys

src_path, dst_path, insert_lmk, adjust_lmk_pos = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(src_path, "r", encoding="utf-8") as fin:
    config = json.load(fin)

config["insert_landmarks"] = insert_lmk == "1"
config["adjust_lmk_pos"] = adjust_lmk_pos == "1"

with open(dst_path, "w", encoding="utf-8") as fout:
    json.dump(config, fout, ensure_ascii=False, indent=2)
    fout.write("\n")
PY

    : > "$run_log"

    {
        echo "[$(date '+%F %T')] Start model=${model_name}, gpu=${gpu_id}"
        echo "[$(date '+%F %T')] Work dir: ${work_dir}"
        echo "[$(date '+%F %T')] HF path: ${hf_path}"
        echo "[$(date '+%F %T')] Tokenizer path: ${TOKENIZER_PATH}"
        echo "[$(date '+%F %T')] Config path: ${hsa_config}"
        echo "[$(date '+%F %T')] Resolved HF path: ${resolved_hf_path}"
        echo "[$(date '+%F %T')] insert_lmk=${insert_lmk}, adjust_lmk_pos=${adjust_lmk_pos}, chunk_size=${chunk_size}"
        echo "[$(date '+%F %T')] Command: CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
    } | tee -a "$run_log"

    if ! CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" 2>&1 | tee -a "$run_log"; then
        echo "[$(date '+%F %T')] ERROR model=${model_name} failed" | tee -a "$run_log"
        return 1
    fi

    local latest_summary_file=""
    latest_summary_file=$(find "$work_dir" -maxdepth 3 -path '*/summary/summary_*.txt' | sort | tail -n 1 || true)
    if [ -n "$latest_summary_file" ]; then
        append_overall_average_to_summary "$latest_summary_file"
        echo "[$(date '+%F %T')] Added overall average to summary: ${latest_summary_file}" | tee -a "$run_log"
    fi

    echo "[$(date '+%F %T')] Finished model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"
}

failed=0
available_gpus=("${GPU_IDS[@]}")
active_pids=()
active_gpus=()
active_models=()
REAPED_GPU_ID=""

pop_gpu() {
    REAPED_GPU_ID="${available_gpus[0]}"
    available_gpus=("${available_gpus[@]:1}")
}

reap_one_job() {
    local finished_pid
    local wait_status=0
    local active_idx

    wait -n -p finished_pid "${active_pids[@]}" || wait_status=$?
    if [ "$wait_status" -ne 0 ]; then
        failed=1
    fi

    for active_idx in "${!active_pids[@]}"; do
        if [ "${active_pids[$active_idx]}" = "$finished_pid" ]; then
            local finished_gpu="${active_gpus[$active_idx]}"
            local finished_model="${active_models[$active_idx]}"
            available_gpus+=("$finished_gpu")
            echo "[$(date '+%F %T')] Slot released: gpu=${finished_gpu}, model=${finished_model}, status=${wait_status}" | tee -a "$LOG_DIR/queue.log"

            unset 'active_pids[active_idx]'
            unset 'active_gpus[active_idx]'
            unset 'active_models[active_idx]'
            active_pids=("${active_pids[@]}")
            active_gpus=("${active_gpus[@]}")
            active_models=("${active_models[@]}")
            REAPED_GPU_ID="$finished_gpu"
            return
        fi
    done
}

for idx in "${!MODEL_NAMES[@]}"; do
    if [ "${#available_gpus[@]}" -eq 0 ]; then
        reap_one_job
    fi

    pop_gpu
    gpu_id="$REAPED_GPU_ID"
    model_name="${MODEL_NAMES[$idx]}"
    echo "[$(date '+%F %T')] Launch model=${model_name} on gpu=${gpu_id}" | tee -a "$LOG_DIR/queue.log"

    run_model_eval \
        "$gpu_id" \
        "$model_name" \
        "${HSA_CONFIGS[$idx]}" \
        "${HF_PATHS[$idx]}" &

    active_pids+=($!)
    active_gpus+=("$gpu_id")
    active_models+=("$model_name")
done

while [ "${#active_pids[@]}" -gt 0 ]; do
    reap_one_job
done

{
    echo "datasets=${DATASETS}"
    echo "debug=${DEBUG}"
    echo "tokenizer_path=${TOKENIZER_PATH}"
    echo "opencompass_path=${OPENCOMPASS_PATH}"
    echo "log_dir=${LOG_DIR}"
    echo "work_dir_root=${WORK_DIR_ROOT}"
    for model_name in "${MODEL_NAMES[@]}"; do
        echo "${model_name}.run.log"
        echo "opencompass_outputs/${model_name}"
    done
    echo "queue.log"
} > "$LOG_DIR/index.log"

echo "All logs saved to: $LOG_DIR"

if [ "$failed" -ne 0 ]; then
    echo "Some opencompass evaluation jobs failed." >&2
    exit 1
fi
