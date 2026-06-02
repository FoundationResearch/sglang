#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

cd "$REPO_ROOT"
export PYTHONPATH=./

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

SEQ_LEN_LIST=(
    $((8 * 1024))
)

DATA_PATH=/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized
VOCAB_DIR=./configs/olmo3_vocab/
MAX_SAMPLES=100
LAST_K_TOKENS=512

GPU_IDS=(
0 1 2 3 4 5 6 7
)

MODEL_NAMES=(
    sanity_check
)

MODEL_CONFIGS=(
    configs/olmo3_7B/olmo3_param_reuse.json
)

CKPT_PATHS=(
    /apdcephfs_sh8/share_300719895/guhao/checkpoints/olmo3_param_reuse_64gpu/
)

if [ "${#MODEL_NAMES[@]}" -ne "${#MODEL_CONFIGS[@]}" ] || \
   [ "${#MODEL_NAMES[@]}" -ne "${#CKPT_PATHS[@]}" ]; then
    echo "MODEL_NAMES / MODEL_CONFIGS / CKPT_PATHS 长度必须一致" >&2
    exit 1
fi

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "GPU_IDS 不能为空" >&2
    exit 1
fi

LOG_DIR="$SCRIPT_DIR/logs/eval_olmo3_ppl_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"
echo "Queue size: ${#GPU_IDS[@]} GPU(s), ${#MODEL_NAMES[@]} model(s)"

run_model_eval() {
    local gpu_id="$1"
    local model_name="$2"
    local model_config="$3"
    local ckpt_path="$4"
    local run_log="$LOG_DIR/${model_name}.run.log"
    local summary_log="$LOG_DIR/${model_name}.summary.log"
    local eval_args=()
    local insert_lmk
    local adjust_lmk_pos
    local chunk_size
    local had_failures=0

    read -r insert_lmk adjust_lmk_pos chunk_size < <("$PYTHON_BIN" - "$model_config" <<'PY'
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

    if [ "$insert_lmk" = "1" ]; then
        eval_args+=(--insert_lmk)
    fi

    if [ "$adjust_lmk_pos" = "1" ]; then
        eval_args+=(--adjust_lmk_pos)
    fi

    : > "$run_log"
    : > "$summary_log"

    {
        echo "[$(date '+%F %T')] Start model=${model_name}, gpu=${gpu_id}"
        echo "[$(date '+%F %T')] Summary log: ${summary_log}"
        echo "[$(date '+%F %T')] Eval args: ${eval_args[*]}"
        echo "[$(date '+%F %T')] Chunk size: ${chunk_size}"
    } | tee -a "$run_log"

    for max_seq_len in "${SEQ_LEN_LIST[@]}"; do
        if [ "$insert_lmk" = "1" ] && [ "$max_seq_len" -le "$chunk_size" ]; then
            echo "[$(date '+%F %T')] Skip ${model_name} max_seq_len=${max_seq_len}: insert_lmk model with chunk_size=${chunk_size} is unsupported at this length" | tee -a "$run_log"
            continue
        fi

        echo "[$(date '+%F %T')] GPU ${gpu_id} | ${model_name} | max_seq_len=${max_seq_len}" | tee -a "$run_log"

        if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" eval/eval_ppl.py \
            --config_path "$model_config" \
            --vocab_dir "$VOCAB_DIR" \
            --checkpoint_path "$ckpt_path" \
            --data_path "$DATA_PATH" \
            --max_seq_len "$max_seq_len" \
            --max_samples "$MAX_SAMPLES" \
            --last_k_tokens "$LAST_K_TOKENS" \
            "${eval_args[@]}" \
            --summary_log "$summary_log" 2>&1 | tee -a "$run_log"; then
            had_failures=1
            echo "[$(date '+%F %T')] ERROR ${model_name} max_seq_len=${max_seq_len} failed, continue to next length" | tee -a "$run_log"
            continue
        fi
    done

    echo "[$(date '+%F %T')] Finished model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"

    if [ "$had_failures" -ne 0 ]; then
        return 1
    fi
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
        "${MODEL_CONFIGS[$idx]}" \
        "${CKPT_PATHS[$idx]}" &

    active_pids+=($!)
    active_gpus+=("$gpu_id")
    active_models+=("$model_name")
done

while [ "${#active_pids[@]}" -gt 0 ]; do
    reap_one_job
done

# if compgen -G "$LOG_DIR/*.summary.log" > /dev/null; then
#     cat "$LOG_DIR"/*.summary.log > "$LOG_DIR/all_models.summary.log"
# fi

# echo "All logs saved to: $LOG_DIR"

if [ "$failed" -ne 0 ]; then
    echo "Some evaluation jobs failed." >&2
    exit 1
fi
