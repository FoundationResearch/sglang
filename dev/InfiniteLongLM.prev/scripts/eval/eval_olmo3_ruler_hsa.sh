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

TASK_IDS=(
0 1 2
)

if [ "$#" -gt 0 ]; then
    TASK_IDS=("$@")
fi

SEQ_LEN_LIST=(
    $((8 * 1024))
    $((16 * 1024))
    $((64 * 1024))
    $((128 * 1024))
    $((256 * 1024))
    # $((512 * 1024))
)

CORPUS_PATH=/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/
VOCAB_DIR=./configs/olmo3_vocab/
MAX_SAMPLES=100
PRINT_EVERY=1
SEGMENT_SIZE=-1
TP_SIZE=1

GPU_IDS=(
3 4 5 6 7
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

MODEL_CONFIGS=(
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

CKPT_PATHS=(
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

if [ "${#MODEL_NAMES[@]}" -ne "${#MODEL_CONFIGS[@]}" ] || \
   [ "${#MODEL_NAMES[@]}" -ne "${#CKPT_PATHS[@]}" ]; then
    echo "MODEL_NAMES / MODEL_CONFIGS / CKPT_PATHS 长度必须一致" >&2
    exit 1
fi

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "GPU_IDS 不能为空" >&2
    exit 1
fi

if [ "${#TASK_IDS[@]}" -eq 0 ]; then
    echo "TASK_IDS 不能为空" >&2
    exit 1
fi

LOG_DIR="$SCRIPT_DIR/logs/eval_olmo3_ruler_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"
echo "Queue size: ${#GPU_IDS[@]} GPU(s), ${#MODEL_NAMES[@]} model(s), ${#TASK_IDS[@]} task(s)"

run_model_eval() {
    local gpu_id="$1"
    local task_id="$2"
    local model_name="$3"
    local model_config="$4"
    local ckpt_path="$5"
    local log_prefix="${model_name}.task${task_id}"
    local run_log="$LOG_DIR/${log_prefix}.run.log"
    local summary_log="$LOG_DIR/${log_prefix}.summary.log"
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
        echo "[$(date '+%F %T')] Start model=${model_name}, gpu=${gpu_id}, task_id=${task_id}"
        echo "[$(date '+%F %T')] Summary log: ${summary_log}"
        echo "[$(date '+%F %T')] Eval args: ${eval_args[*]}"
        echo "[$(date '+%F %T')] Chunk size: ${chunk_size}"
    } | tee -a "$run_log"

    for max_seq_len in "${SEQ_LEN_LIST[@]}"; do
        if [ "$insert_lmk" = "1" ] && [ "$max_seq_len" -le "$chunk_size" ]; then
            echo "[$(date '+%F %T')] Skip ${model_name} task_id=${task_id} max_seq_len=${max_seq_len}: insert_lmk model with chunk_size=${chunk_size} is unsupported at this length" | tee -a "$run_log"
            continue
        fi

        echo "[$(date '+%F %T')] GPU ${gpu_id} | ${model_name} | task_id=${task_id} | max_seq_len=${max_seq_len}" | tee -a "$run_log"

        if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" eval/eval_ruler_hf.py \
            --config_path "$model_config" \
            --vocab_dir "$VOCAB_DIR" \
            --corpus_path "$CORPUS_PATH" \
            --checkpoint_path "$ckpt_path" \
            --task_id "$task_id" \
            --segment_size "$SEGMENT_SIZE" \
            --max_seq_len "$max_seq_len" \
            --max_samples "$MAX_SAMPLES" \
            --print_every "$PRINT_EVERY" \
            --tp_size "$TP_SIZE" \
            "${eval_args[@]}" \
            --summary_log "$summary_log" 2>&1 | tee -a "$run_log"; then
            had_failures=1
            echo "[$(date '+%F %T')] ERROR ${model_name} task_id=${task_id} max_seq_len=${max_seq_len} failed, continue to next length" | tee -a "$run_log"
            continue
        fi
    done

    echo "[$(date '+%F %T')] Finished model=${model_name}, gpu=${gpu_id}, task_id=${task_id}" | tee -a "$run_log"

    if [ "$had_failures" -ne 0 ]; then
        return 1
    fi
}

failed=0
available_gpus=("${GPU_IDS[@]}")
active_pids=()
active_gpus=()
active_jobs=()
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
            local finished_job="${active_jobs[$active_idx]}"
            available_gpus+=("$finished_gpu")
            echo "[$(date '+%F %T')] Slot released: gpu=${finished_gpu}, job=${finished_job}, status=${wait_status}" | tee -a "$LOG_DIR/queue.log"

            unset 'active_pids[active_idx]'
            unset 'active_gpus[active_idx]'
            unset 'active_jobs[active_idx]'
            active_pids=("${active_pids[@]}")
            active_gpus=("${active_gpus[@]}")
            active_jobs=("${active_jobs[@]}")
            REAPED_GPU_ID="$finished_gpu"
            return
        fi
    done
}

for task_id in "${TASK_IDS[@]}"; do
    for idx in "${!MODEL_NAMES[@]}"; do
        if [ "${#available_gpus[@]}" -eq 0 ]; then
            reap_one_job
        fi

        pop_gpu
        gpu_id="$REAPED_GPU_ID"
        model_name="${MODEL_NAMES[$idx]}"
        job_name="${model_name}.task${task_id}"
        echo "[$(date '+%F %T')] Launch job=${job_name} on gpu=${gpu_id}" | tee -a "$LOG_DIR/queue.log"

        run_model_eval \
            "$gpu_id" \
            "$task_id" \
            "$model_name" \
            "${MODEL_CONFIGS[$idx]}" \
            "${CKPT_PATHS[$idx]}" &

        active_pids+=($!)
        active_gpus+=("$gpu_id")
        active_jobs+=("$job_name")
    done
done

while [ "${#active_pids[@]}" -gt 0 ]; do
    reap_one_job
done

if compgen -G "$LOG_DIR/*.summary.log" > /dev/null; then
    cat "$LOG_DIR"/*.summary.log > "$LOG_DIR/all_models.summary.log"
fi

{
    printf 'task_ids='
    printf '%s ' "${TASK_IDS[@]}"
    printf '\n'
    echo "log_dir=${LOG_DIR}"
    for task_id in "${TASK_IDS[@]}"; do
        for model_name in "${MODEL_NAMES[@]}"; do
            echo "${model_name}.task${task_id}.run.log"
            echo "${model_name}.task${task_id}.summary.log"
        done
    done
    echo "all_models.summary.log"
    echo "queue.log"
} > "$LOG_DIR/index.log"

echo "All logs saved to: $LOG_DIR"

if [ "$failed" -ne 0 ]; then
    echo "Some ruler evaluation jobs failed." >&2
    exit 1
fi
