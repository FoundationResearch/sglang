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

# ============================================================
#  Eval mode: ppl / ruler / all
# ============================================================
EVAL_MODE="${EVAL_MODE:-all}"

# ============================================================
#  Models: each entry is  "name|config|ckpt_path|ruler_max_seq_len"
#  ruler_max_seq_len is optional (0 or omitted = no limit)
# ============================================================
MODELS=(
    "olmo3-cpt-step7000|configs/olmo3_7B/olmo3_param_reuse.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/olmo3-param-reuse-lr3e-4-64gpu/global_step_7000/hf_ckpt|$((64 * 1024))"
    "olmo3-cpt-step8000|configs/olmo3_7B/olmo3_param_reuse.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/olmo3-param-reuse-lr3e-4-64gpu/global_step_8000/hf_ckpt|$((64 * 1024))"
    "olmo3-cpt-step9000|configs/olmo3_7B/olmo3_param_reuse.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/olmo3-param-reuse-lr3e-4-64gpu/global_step_9000/hf_ckpt|$((64 * 1024))"
    "olmo3-cpt-step10000|configs/olmo3_7B/olmo3_param_reuse.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/olmo3-param-reuse-lr3e-4-64gpu/global_step_10000/hf_ckpt|$((64 * 1024))"

    "olmo3-interleave-step7000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA512_non_unified.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA512-non-unified-64gpu/global_step_7000/hf_ckpt"
    "olmo3-interleave-step8000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA512_non_unified.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA512-non-unified-64gpu/global_step_8000/hf_ckpt"
    "olmo3-interleave-step9000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA512_non_unified.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA512-non-unified-64gpu/global_step_9000/hf_ckpt"


)

# ============================================================
#  GPU pool
# ============================================================
GPU_IDS=(
1 2 3 4 5 6 7
)

# ============================================================
#  PPL settings
# ============================================================
PPL_SEQ_LEN_LIST=(
    64
    128
    512
    $((8 * 1024))
    $((16 * 1024))
    $((64 * 1024))
    $((128 * 1024))
    $((256 * 1024))
    $((512 * 1024))
    $((1024 * 1024))
)
PPL_DATA_PATH=/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized
PPL_MAX_SAMPLES=100
PPL_LAST_K_TOKENS=512

# ============================================================
#  RULER settings
# ============================================================
RULER_SEQ_LEN_LIST=(
    $((8 * 1024))
    $((16 * 1024))
    $((64 * 1024))
    $((128 * 1024))
    $((256 * 1024))
    # $((512 * 1024))
)
RULER_TASK_IDS=(0 1 2)
RULER_CORPUS_PATH=/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/
RULER_MAX_SAMPLES=100
RULER_PRINT_EVERY=1
RULER_SEGMENT_SIZE=-1
RULER_TP_SIZE=1

# Allow overriding TASK_IDS from command line args
if [ "$#" -gt 0 ]; then
    RULER_TASK_IDS=("$@")
fi

VOCAB_DIR=./configs/olmo3_vocab/

# ============================================================
#  Parse MODELS array into parallel arrays
# ============================================================
MODEL_NAMES=()
MODEL_CONFIGS=()
CKPT_PATHS=()
RULER_MAX_SEQ_LENS=()    # per-model ruler cap (0 = no limit)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name config path ruler_cap <<< "$entry"
    MODEL_NAMES+=("$name")
    MODEL_CONFIGS+=("$config")
    CKPT_PATHS+=("$path")
    RULER_MAX_SEQ_LENS+=("${ruler_cap:-0}")
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
LOG_DIR="$SCRIPT_DIR/logs/eval_olmo3_${EVAL_MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Eval mode: ${EVAL_MODE}"
echo "Logs will be saved to: $LOG_DIR"
echo "Queue size: ${#GPU_IDS[@]} GPU(s), ${#MODEL_NAMES[@]} model(s)"

# ============================================================
#  Helper: read model config flags
# ============================================================
parse_model_config() {
    local model_config="$1"
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
    echo "$insert_lmk $adjust_lmk_pos $chunk_size"
}

build_eval_args() {
    local insert_lmk="$1"
    local adjust_lmk_pos="$2"
    local args=""
    if [ "$insert_lmk" = "1" ]; then
        args+=" --insert_lmk"
    fi
    if [ "$adjust_lmk_pos" = "1" ]; then
        args+=" --adjust_lmk_pos"
    fi
    echo "$args"
}

# ============================================================
#  PPL evaluation for one model
# ============================================================
run_ppl_eval() {
    local gpu_id="$1"
    local model_name="$2"
    local model_config="$3"
    local ckpt_path="$4"
    local run_log="$LOG_DIR/${model_name}.ppl.run.log"
    local summary_log="$LOG_DIR/${model_name}.ppl.summary.log"
    local had_failures=0

    local insert_lmk adjust_lmk_pos chunk_size
    read -r insert_lmk adjust_lmk_pos chunk_size < <(parse_model_config "$model_config")
    local eval_args=()
    [ "$insert_lmk" = "1" ] && eval_args+=(--insert_lmk)
    [ "$adjust_lmk_pos" = "1" ] && eval_args+=(--adjust_lmk_pos)

    : > "$run_log"
    : > "$summary_log"

    {
        echo "[$(date '+%F %T')] [PPL] Start model=${model_name}, gpu=${gpu_id}"
        echo "[$(date '+%F %T')] Eval args: ${eval_args[*]:-<none>}"
        echo "[$(date '+%F %T')] Chunk size: ${chunk_size}"
    } | tee -a "$run_log"

    for max_seq_len in "${PPL_SEQ_LEN_LIST[@]}"; do
        if [ "$insert_lmk" = "1" ] && [ "$max_seq_len" -le "$chunk_size" ]; then
            echo "[$(date '+%F %T')] Skip ${model_name} max_seq_len=${max_seq_len}: chunk_size=${chunk_size}" | tee -a "$run_log"
            continue
        fi

        echo "[$(date '+%F %T')] GPU ${gpu_id} | ${model_name} | PPL | max_seq_len=${max_seq_len}" | tee -a "$run_log"

        if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" eval/eval_ppl.py \
            --config_path "$model_config" \
            --vocab_dir "$VOCAB_DIR" \
            --checkpoint_path "$ckpt_path" \
            --data_path "$PPL_DATA_PATH" \
            --max_seq_len "$max_seq_len" \
            --max_samples "$PPL_MAX_SAMPLES" \
            --last_k_tokens "$PPL_LAST_K_TOKENS" \
            "${eval_args[@]}" \
            --summary_log "$summary_log" 2>&1 | tee -a "$run_log"; then
            had_failures=1
            echo "[$(date '+%F %T')] ERROR ${model_name} PPL max_seq_len=${max_seq_len} failed" | tee -a "$run_log"
            continue
        fi
    done

    echo "[$(date '+%F %T')] [PPL] Finished model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"
    [ "$had_failures" -ne 0 ] && return 1
    return 0
}

# ============================================================
#  RULER evaluation for one (model, task_id) combo
# ============================================================
run_ruler_eval() {
    local gpu_id="$1"
    local task_id="$2"
    local model_name="$3"
    local model_config="$4"
    local ckpt_path="$5"
    local ruler_cap="${6:-0}"
    local log_prefix="${model_name}.ruler.task${task_id}"
    local run_log="$LOG_DIR/${log_prefix}.run.log"
    local summary_log="$LOG_DIR/${log_prefix}.summary.log"
    local had_failures=0

    local insert_lmk adjust_lmk_pos chunk_size
    read -r insert_lmk adjust_lmk_pos chunk_size < <(parse_model_config "$model_config")
    local eval_args=()
    [ "$insert_lmk" = "1" ] && eval_args+=(--insert_lmk)
    [ "$adjust_lmk_pos" = "1" ] && eval_args+=(--adjust_lmk_pos)

    : > "$run_log"
    : > "$summary_log"

    {
        echo "[$(date '+%F %T')] [RULER] Start model=${model_name}, gpu=${gpu_id}, task_id=${task_id}"
        echo "[$(date '+%F %T')] Eval args: ${eval_args[*]:-<none>}"
        echo "[$(date '+%F %T')] Chunk size: ${chunk_size}"
    } | tee -a "$run_log"

    for max_seq_len in "${RULER_SEQ_LEN_LIST[@]}"; do
        if [ "$insert_lmk" = "1" ] && [ "$max_seq_len" -le "$chunk_size" ]; then
            echo "[$(date '+%F %T')] Skip ${model_name} task_id=${task_id} max_seq_len=${max_seq_len}: chunk_size=${chunk_size}" | tee -a "$run_log"
            continue
        fi

        if [ "$ruler_cap" -gt 0 ] && [ "$max_seq_len" -gt "$ruler_cap" ]; then
            echo "[$(date '+%F %T')] Skip ${model_name} task_id=${task_id} max_seq_len=${max_seq_len}: exceeds ruler_max_seq_len=${ruler_cap}" | tee -a "$run_log"
            continue
        fi

        echo "[$(date '+%F %T')] GPU ${gpu_id} | ${model_name} | RULER task=${task_id} | max_seq_len=${max_seq_len}" | tee -a "$run_log"

        if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" eval/eval_ruler_hf.py \
            --config_path "$model_config" \
            --vocab_dir "$VOCAB_DIR" \
            --corpus_path "$RULER_CORPUS_PATH" \
            --checkpoint_path "$ckpt_path" \
            --task_id "$task_id" \
            --segment_size "$RULER_SEGMENT_SIZE" \
            --max_seq_len "$max_seq_len" \
            --max_samples "$RULER_MAX_SAMPLES" \
            --print_every "$RULER_PRINT_EVERY" \
            --tp_size "$RULER_TP_SIZE" \
            "${eval_args[@]}" \
            --summary_log "$summary_log" 2>&1 | tee -a "$run_log"; then
            had_failures=1
            echo "[$(date '+%F %T')] ERROR ${model_name} RULER task=${task_id} max_seq_len=${max_seq_len} failed" | tee -a "$run_log"
            continue
        fi
    done

    echo "[$(date '+%F %T')] [RULER] Finished model=${model_name}, gpu=${gpu_id}, task_id=${task_id}" | tee -a "$run_log"
    [ "$had_failures" -ne 0 ] && return 1
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

dispatch_job() {
    # dispatch_job <job_name> <func> <arg1> <arg2> ...
    # The first arg to <func> will be gpu_id (auto-assigned).
    local job_name="$1"
    shift

    if [ "${#available_gpus[@]}" -eq 0 ]; then
        reap_one_job
    fi

    pop_gpu
    local gpu_id="$REAPED_GPU_ID"
    local func="$1"
    shift

    echo "[$(date '+%F %T')] Launch job=${job_name} on gpu=${gpu_id}" | tee -a "$LOG_DIR/queue.log"

    "$func" "$gpu_id" "$@" &

    active_pids+=($!)
    active_gpus+=("$gpu_id")
    active_jobs+=("$job_name")
}

# ============================================================
#  Dispatch jobs
# ============================================================

# --- PPL ---
if [ "$EVAL_MODE" = "ppl" ] || [ "$EVAL_MODE" = "all" ]; then
    echo "--- Queueing PPL jobs ---"
    for idx in "${!MODEL_NAMES[@]}"; do
        model_name="${MODEL_NAMES[$idx]}"
        dispatch_job "${model_name}.ppl" \
            run_ppl_eval \
            "$model_name" \
            "${MODEL_CONFIGS[$idx]}" \
            "${CKPT_PATHS[$idx]}"
    done
fi

# --- RULER ---
if [ "$EVAL_MODE" = "ruler" ] || [ "$EVAL_MODE" = "all" ]; then
    echo "--- Queueing RULER jobs ---"
    for task_id in "${RULER_TASK_IDS[@]}"; do
        for idx in "${!MODEL_NAMES[@]}"; do
            model_name="${MODEL_NAMES[$idx]}"
            dispatch_job "${model_name}.ruler.task${task_id}" \
                run_ruler_eval \
                "$task_id" \
                "$model_name" \
                "${MODEL_CONFIGS[$idx]}" \
                "${CKPT_PATHS[$idx]}" \
                "${RULER_MAX_SEQ_LENS[$idx]}"
        done
    done
fi

# --- Wait for remaining ---
while [ "${#active_pids[@]}" -gt 0 ]; do
    reap_one_job
done

# ============================================================
#  Aggregate summary logs
# ============================================================
if compgen -G "$LOG_DIR/*.summary.log" > /dev/null; then
    cat "$LOG_DIR"/*.summary.log > "$LOG_DIR/all_models.summary.log"
fi

# Generate LaTeX summary tables (PPL + RULER)
"$PYTHON_BIN" "$SCRIPT_DIR/generate_latex_summary.py" "$LOG_DIR" \
    -o "$LOG_DIR/summary.log" || echo "Warning: LaTeX summary generation failed" >&2

{
    echo "eval_mode=${EVAL_MODE}"
    echo "log_dir=${LOG_DIR}"
    echo "models=${MODEL_NAMES[*]}"
    [ "$EVAL_MODE" = "ruler" ] || [ "$EVAL_MODE" = "all" ] && printf 'ruler_task_ids=%s\n' "${RULER_TASK_IDS[*]}"
} > "$LOG_DIR/index.log"

echo ""
echo "All logs saved to: $LOG_DIR"

if [ "$failed" -ne 0 ]; then
    echo "Some evaluation jobs failed." >&2
    exit 1
fi
