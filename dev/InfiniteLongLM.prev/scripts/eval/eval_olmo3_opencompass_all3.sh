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

if ! pkill -f "burner" >/dev/null 2>&1; then
    echo "No existing burner process found."
fi

DATASET_LIST=(
    gpqa_few_shot_ppl_4b5a83
    mmlu_ppl_ac766d
    hellaswag_10shot_ppl_59c85e
    ARC_c_few_shot_ppl
    SuperGLUE_BoolQ_few_shot_ppl
    race_few_shot_ppl
)
DEBUG=${DEBUG:-1}
TOKENIZER_PATH=${TOKENIZER_PATH:-/apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_vocab}
RUN_TAG=$(date +%Y%m%d_%H%M%S)
MASTER_LOG_DIR="$SCRIPT_DIR/logs/eval_olmo3_opencompass_all_${RUN_TAG}"

GPU_IDS=(
    1 2 3 4 5 6 7
)
DATASET_PARALLEL=${DATASET_PARALLEL:-6}

MODEL_NAMES=(
    8KA2K-step3000
    8KA2K-no-noise-step2000

)

HSA_CONFIGS=(
    configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk.json
    configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json
)

HF_PATHS=(
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-layerqk-64gpu/global_step_3000/hf_ckpt
    /apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu/global_step_2000/hf_ckpt
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

build_gpu_groups() {
    local num_gpus="${#GPU_IDS[@]}"
    local num_models="${#MODEL_NAMES[@]}"
    local requested_parallel="$DATASET_PARALLEL"
    local start=0
    local remaining="$num_gpus"
    local slot
    local slots_left
    local group_size
    local -a group=()

    if [ "$requested_parallel" -lt 1 ]; then
        echo "DATASET_PARALLEL 必须 >= 1" >&2
        exit 1
    fi

    if [ "$requested_parallel" -gt "$num_gpus" ]; then
        requested_parallel="$num_gpus"
    fi

    DATASET_PARALLEL="$requested_parallel"
    GPU_GROUPS=()

    for ((slot = 0; slot < DATASET_PARALLEL; slot++)); do
        slots_left=$((DATASET_PARALLEL - slot))
        group_size=$(((remaining + slots_left - 1) / slots_left))
        # 每组 GPU 数不超过模型数，避免分配了卡却没有模型跑
        if [ "$group_size" -gt "$num_models" ]; then
            group_size="$num_models"
        fi
        group=("${GPU_IDS[@]:start:group_size}")
        GPU_GROUPS+=("${group[*]}")
        start=$((start + group_size))
        remaining=$((num_gpus - start))
    done
}

build_gpu_groups

mkdir -p "$MASTER_LOG_DIR"
echo "Master logs will be saved to: $MASTER_LOG_DIR"
echo "Datasets: ${DATASET_LIST[*]}"
echo "Queue size: ${#GPU_IDS[@]} GPU(s), ${#MODEL_NAMES[@]} model(s)"
echo "Dataset parallelism: ${DATASET_PARALLEL}"
for slot_idx in "${!GPU_GROUPS[@]}"; do
    echo "Dataset slot ${slot_idx}: GPUs ${GPU_GROUPS[$slot_idx]}"
done

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
    local checkpoint_vocab_size
    local tokenizer_vocab_size
    local cmd=(
        "$PYTHON_BIN" eval/eval_opencompass.py
        --datasets "$CURRENT_DATASET"
        --hf-type base
        --hf-path "$resolved_hf_path"
        --hsa-config "$hsa_config"
        -w "$work_dir"
    )

    read -r insert_lmk adjust_lmk_pos chunk_size checkpoint_vocab_size tokenizer_vocab_size < <("$PYTHON_BIN" - "$hsa_config" "$hf_path" "$TOKENIZER_PATH" <<'PY'
import json
import os
import struct
import sys

config_path, hf_path, tokenizer_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(config_path, "r", encoding="utf-8") as fin:
    config = json.load(fin)

def detect_checkpoint_vocab_size(hf_dir: str, default: int) -> int:
    index_path = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as fin:
            index = json.load(fin)
        for key in ("model.embed_tokens.weight", "lm_head.weight"):
            shard = index.get("weight_map", {}).get(key)
            if not shard:
                continue
            shard_path = os.path.join(hf_dir, shard)
            with open(shard_path, "rb") as fin:
                header_len = struct.unpack("<Q", fin.read(8))[0]
                header = json.loads(fin.read(header_len))
            if key in header and "shape" in header[key] and header[key]["shape"]:
                return int(header[key]["shape"][0])

    checkpoint_config_path = os.path.join(hf_dir, "config.json")
    if os.path.exists(checkpoint_config_path):
        with open(checkpoint_config_path, "r", encoding="utf-8") as fin:
            checkpoint_config = json.load(fin)
        return int(checkpoint_config.get("vocab_size", default))

    return int(default)

def detect_tokenizer_vocab_size(tokenizer_dir: str, default: int) -> int:
    vocab_json_path = os.path.join(tokenizer_dir, "vocab.json")
    if os.path.exists(vocab_json_path):
        with open(vocab_json_path, "r", encoding="utf-8") as fin:
            vocab = json.load(fin)
        return len(vocab)

    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as fin:
            tokenizer = json.load(fin)
        model_vocab = tokenizer.get("model", {}).get("vocab", {})
        if model_vocab:
            return len(model_vocab)

    return int(default)

insert_lmk = bool(config.get("insert_landmarks", False) or config.get("adjust_lmk_pos", False))
adjust_lmk_pos = bool(config.get("adjust_lmk_pos", False))
chunk_size = int(config.get("chunk_size", 64))
checkpoint_vocab_size = detect_checkpoint_vocab_size(hf_path, config.get("vocab_size", 0))
tokenizer_vocab_size = detect_tokenizer_vocab_size(tokenizer_path, config.get("vocab_size", 0))
print(
    f'{"1" if insert_lmk else "0"} '
    f'{"1" if adjust_lmk_pos else "0"} '
    f'{chunk_size} '
    f'{checkpoint_vocab_size} '
    f'{tokenizer_vocab_size}'
)
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
    "$PYTHON_BIN" - "$hsa_config" "$resolved_hf_path/config.json" "$insert_lmk" "$adjust_lmk_pos" "$tokenizer_vocab_size" <<'PY'
import json
import sys

src_path, dst_path, insert_lmk, adjust_lmk_pos, tokenizer_vocab_size = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
with open(src_path, "r", encoding="utf-8") as fin:
    config = json.load(fin)

config["insert_landmarks"] = insert_lmk == "1"
config["adjust_lmk_pos"] = adjust_lmk_pos == "1"
config["vocab_size"] = int(tokenizer_vocab_size)

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
        echo "[$(date '+%F %T')] insert_lmk=${insert_lmk}, adjust_lmk_pos=${adjust_lmk_pos}, chunk_size=${chunk_size}, checkpoint_vocab_size=${checkpoint_vocab_size}, tokenizer_vocab_size=${tokenizer_vocab_size}"
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

run_dataset_eval() {
    local dataset_name="$1"
    local gpu_group="$2"
    local dataset_tag
    local failed=0
    local REAPED_GPU_ID=""
    local -a available_gpus=()
    local -a active_pids=()
    local -a active_gpus=()
    local -a active_models=()

    dataset_tag=$(printf '%s' "$dataset_name" | tr ',/' '__' | tr -cd '[:alnum:]_.-')
    CURRENT_DATASET="$dataset_name"
    LOG_DIR="$MASTER_LOG_DIR/${dataset_tag}"
    WORK_DIR_ROOT="$LOG_DIR/opencompass_outputs"
    IFS=' ' read -r -a available_gpus <<< "$gpu_group"

    mkdir -p "$WORK_DIR_ROOT"

    echo
    echo "==== Dataset: ${CURRENT_DATASET} ===="
    echo "Logs will be saved to: $LOG_DIR"
    echo "Assigned GPUs: ${available_gpus[*]}"

    for idx in "${!MODEL_NAMES[@]}"; do
        if [ "${#available_gpus[@]}" -eq 0 ]; then
            reap_one_job
        fi

        pop_gpu
        gpu_id="$REAPED_GPU_ID"
        model_name="${MODEL_NAMES[$idx]}"
        echo "[$(date '+%F %T')] Launch dataset=${CURRENT_DATASET}, model=${model_name} on gpu=${gpu_id}" | tee -a "$LOG_DIR/queue.log"

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
        echo "datasets=${CURRENT_DATASET}"
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

    echo "Dataset ${CURRENT_DATASET} logs saved to: $LOG_DIR"

    if [ "$failed" -ne 0 ]; then
        echo "Dataset ${CURRENT_DATASET} has failed jobs." >&2
        return 1
    fi
}

overall_failed=0
CURRENT_DATASET=""
LOG_DIR=""
WORK_DIR_ROOT=""
available_dataset_slots=()
dataset_active_pids=()
dataset_active_names=()
dataset_active_slots=()
REAPED_DATASET_SLOT=""

for ((slot_idx = 0; slot_idx < DATASET_PARALLEL; slot_idx++)); do
    available_dataset_slots+=("$slot_idx")
done

pop_dataset_slot() {
    REAPED_DATASET_SLOT="${available_dataset_slots[0]}"
    available_dataset_slots=("${available_dataset_slots[@]:1}")
}

reap_one_dataset_job() {
    local finished_pid
    local wait_status=0
    local active_idx

    wait -n -p finished_pid "${dataset_active_pids[@]}" || wait_status=$?
    if [ "$wait_status" -ne 0 ]; then
        overall_failed=1
    fi

    for active_idx in "${!dataset_active_pids[@]}"; do
        if [ "${dataset_active_pids[$active_idx]}" = "$finished_pid" ]; then
            local finished_dataset="${dataset_active_names[$active_idx]}"
            local finished_slot="${dataset_active_slots[$active_idx]}"
            available_dataset_slots+=("$finished_slot")
            echo "[$(date '+%F %T')] Dataset slot released: slot=${finished_slot}, dataset=${finished_dataset}, status=${wait_status}" | tee -a "$MASTER_LOG_DIR/queue.log"

            unset 'dataset_active_pids[active_idx]'
            unset 'dataset_active_names[active_idx]'
            unset 'dataset_active_slots[active_idx]'
            dataset_active_pids=("${dataset_active_pids[@]}")
            dataset_active_names=("${dataset_active_names[@]}")
            dataset_active_slots=("${dataset_active_slots[@]}")
            REAPED_DATASET_SLOT="$finished_slot"
            return
        fi
    done
}

for dataset_name in "${DATASET_LIST[@]}"; do
    if [ "${#available_dataset_slots[@]}" -eq 0 ]; then
        reap_one_dataset_job
    fi

    pop_dataset_slot
    slot_idx="$REAPED_DATASET_SLOT"
    echo "[$(date '+%F %T')] Launch dataset=${dataset_name} on slot=${slot_idx}, gpus=${GPU_GROUPS[$slot_idx]}" | tee -a "$MASTER_LOG_DIR/queue.log"

    run_dataset_eval "$dataset_name" "${GPU_GROUPS[$slot_idx]}" &
    dataset_active_pids+=($!)
    dataset_active_names+=("$dataset_name")
    dataset_active_slots+=("$slot_idx")
done

while [ "${#dataset_active_pids[@]}" -gt 0 ]; do
    reap_one_dataset_job
done

{
    echo "master_log_dir=${MASTER_LOG_DIR}"
    echo "datasets=${DATASET_LIST[*]}"
    echo "dataset_parallel=${DATASET_PARALLEL}"
    for slot_idx in "${!GPU_GROUPS[@]}"; do
        echo "slot_${slot_idx}=${GPU_GROUPS[$slot_idx]}"
    done
    for dataset_name in "${DATASET_LIST[@]}"; do
        dataset_tag=$(printf '%s' "$dataset_name" | tr ',/' '__' | tr -cd '[:alnum:]_.-')
        echo "${dataset_name} -> ${dataset_tag}"
    done
} > "$MASTER_LOG_DIR/index.log"

echo "All dataset logs saved under: $MASTER_LOG_DIR"

# ── Generate summary.log in LaTeX format ──
"$PYTHON_BIN" - "$MASTER_LOG_DIR" <<'PYEOF'
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
PYEOF

echo "All dataset logs saved under: $MASTER_LOG_DIR"

if [ "$overall_failed" -ne 0 ]; then
    echo "Some opencompass evaluation jobs failed." >&2
    exit 1
fi
