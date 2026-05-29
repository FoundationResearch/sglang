#!/usr/bin/env bash
# ============================================================
# LongBench-v2 evaluation (direct choice / cloze)
#
# EVAL_SCRIPT controls the mode:
#   eval/eval_longbench2_direct.py — instruct models (1 forward/sample)
#   eval/eval_longbench2_cloze.py  — base models (4 forwards/sample)
#
# Usage:
#   bash scripts/eval/eval_olmo3_longbench2.sh
# ============================================================

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

# ============================================================
#  Models: "name|config|ckpt_path"
#  config can be empty for standard HF models (tokenizer is in ckpt)
# ============================================================
MODELS=(
    "innerx-step1000|configs/olmo3_7B/olmo3_lhsa_innerx.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-innerx-lr3e-4-warmup/global_step_1000/hf_ckpt"
    "interleave-step1000|configs/olmo3_7B/olmo3_lhsa_interleave.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave/global_step_1000/hf_ckpt"
)

pkill -f "burner" || true
# ============================================================
#  GPU pool
# ============================================================
GPU_IDS=(
4 5
)
# ============================================================
#  LongBench-v2 settings
# ============================================================
LOCAL_DATASET="${LOCAL_DATASET:-}"          # empty = download from HF
MAX_INPUT_TOKENS=65536                     # 0 = no truncation
SEGMENT_SIZE=4096                            # chunk prefill segment size (<=0 = full forward)
VOCAB_DIR=./configs/olmo3_vocab/           # tokenizer for HSA models


# Eval script: "direct" for instruct models, "cloze" for base models
# direct = eval/eval_longbench2_direct.py  (1 forward, compare A/B/C/D logits)
# cloze  = eval/eval_longbench2_cloze.py   (4 forwards, avg log-prob per option)
# EVAL_SCRIPT=eval/eval_longbench2_direct.py
EVAL_SCRIPT=eval/eval_longbench2_cloze.py

SAVE_ROOT="$SCRIPT_DIR/logs/eval_longbench2_$(date +%Y%m%d_%H%M%S)"

# ============================================================
#  Parse MODELS
# ============================================================
MODEL_NAMES=()
MODEL_CONFIGS=()
CKPT_PATHS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name config path <<< "$entry"
    MODEL_NAMES+=("$name")
    MODEL_CONFIGS+=("${config}")
    CKPT_PATHS+=("$path")
done

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
    echo "MODELS 不能为空" >&2
    exit 1
fi

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "GPU_IDS 不能为空" >&2
    exit 1
fi

# ============================================================
#  Logging
# ============================================================
mkdir -p "$SAVE_ROOT"
echo "Logs / results will be saved to: $SAVE_ROOT"
echo "Queue size: ${#GPU_IDS[@]} GPU(s), ${#MODEL_NAMES[@]} model(s)"

# ============================================================
#  Eval function for one model
# ============================================================
run_longbench_eval() {
    local gpu_id="$1"
    local model_name="$2"
    local model_config="$3"
    local ckpt_path="$4"
    local save_dir="$SAVE_ROOT/${model_name}"
    local run_log="$SAVE_ROOT/${model_name}.run.log"

    mkdir -p "$save_dir"
    : > "$run_log"

    echo "[$(date '+%F %T')] [LongBench2] Start model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"

    local extra_args=()

    # If config is provided → HSA model, pass --config_path and --vocab_dir
    if [ -n "$model_config" ]; then
        extra_args+=(--config_path "$model_config")
        extra_args+=(--vocab_dir "$VOCAB_DIR")
    fi

    if [ -n "$LOCAL_DATASET" ]; then
        extra_args+=(--local_dataset "$LOCAL_DATASET")
    fi

    if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$EVAL_SCRIPT" \
        --checkpoint_path "$ckpt_path" \
        --save_dir "$save_dir" \
        --max_input_tokens "$MAX_INPUT_TOKENS" \
        --segment_size "$SEGMENT_SIZE" \
        --n_proc 1 \
        "${extra_args[@]}" \
        2>&1 | tee -a "$run_log"; then
        echo "[$(date '+%F %T')] [ERROR] ${model_name} failed" | tee -a "$run_log"
        return 1
    fi

    echo "[$(date '+%F %T')] [LongBench2] Finished model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"
    return 0
}

# ============================================================
#  GPU scheduler
# ============================================================
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

    wait -n -p finished_pid "${active_pids[@]}" || wait_status=$?
    if [ "$wait_status" -ne 0 ]; then
        failed=1
    fi

    for active_idx in "${!active_pids[@]}"; do
        if [ "${active_pids[$active_idx]}" = "$finished_pid" ]; then
            local finished_gpu="${active_gpus[$active_idx]}"
            local finished_job="${active_jobs[$active_idx]}"
            available_gpus+=("$finished_gpu")
            echo "[$(date '+%F %T')] Slot released: gpu=${finished_gpu}, job=${finished_job}, status=${wait_status}" | tee -a "$SAVE_ROOT/queue.log"

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

dispatch_job() {
    local job_name="$1"
    shift

    if [ "${#available_gpus[@]}" -eq 0 ]; then
        reap_one_job
    fi

    pop_gpu
    local gpu_id="$REAPED_GPU_ID"
    local func="$1"
    shift

    echo "[$(date '+%F %T')] Launch job=${job_name} on gpu=${gpu_id}" | tee -a "$SAVE_ROOT/queue.log"

    "$func" "$gpu_id" "$@" &

    active_pids+=($!)
    active_gpus+=("$gpu_id")
    active_jobs+=("$job_name")
}

# ============================================================
#  Dispatch
# ============================================================
echo "--- Queueing LongBench-v2 jobs ---"
for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    dispatch_job "${model_name}.longbench2" \
        run_longbench_eval \
        "$model_name" \
        "${MODEL_CONFIGS[$idx]}" \
        "${CKPT_PATHS[$idx]}"
done

# Wait for all
while [ "${#active_pids[@]}" -gt 0 ]; do
    reap_one_job
done

echo ""
echo "All results saved to: $SAVE_ROOT"

if [ "$failed" -ne 0 ]; then
    echo "Some evaluation jobs failed." >&2
    exit 1
fi
