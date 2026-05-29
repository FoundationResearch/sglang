#!/usr/bin/env bash
# ============================================================
# LongBench v1 evaluation (generate + score with official metrics)
#
# Usage:
#   bash scripts/eval/eval_olmo3_longbench_v1.sh
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
#  config can be empty for standard HF models
# ============================================================
MODELS=(
    # "Olmo-3-7B-Instruct-SFT||/apdcephfs_fsgm/share_303843174/user/guhao/Models/Olmo-3-7B-Instruct-SFT"
    
    # "8KA1K-no-noise-warmup1k-step1000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_1000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step2000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_2000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step3000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_3000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step4000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_4000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step5000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_5000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step6000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_6000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step7000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_7000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step8000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_8000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step9000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_9000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step10000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_10000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step11000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_11000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step12000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_12000/hf_ckpt"
    # "8KA1K-no-noise-warmup1k-step13000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_13000/hf_ckpt"
    "olmo3-cpt-step13000|configs/olmo3_7B/olmo3_param_reuse.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/olmo3_cpt_64gpu/global_step_13000/hf_ckpt"



    # "8KA2K-no-noise-warmup1k-step1000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_1000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step2000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_2000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step3000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_3000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step4000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_4000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step5000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_5000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step6000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_6000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step7000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_7000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step8000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_8000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step9000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_9000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step10000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_10000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step11000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_11000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step12000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_12000/hf_ckpt"
    # "8KA2K-no-noise-warmup1k-step13000|configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json|/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_13000/hf_ckpt"
)

# ============================================================
#  GPU pool
# ============================================================
GPU_IDS=(
7
)

# ============================================================
#  LongBench v1 settings
# ============================================================
MAX_LENGTH=65536                           # middle truncation limit
VOCAB_DIR=./configs/olmo3_vocab/           # tokenizer for HSA models
DATASETS=""                                # empty = all 21 tasks; or comma-separated, e.g. "hotpotqa,qasper"
SAVE_ROOT="$SCRIPT_DIR/logs/eval_longbench_v1_$(date +%Y%m%d_%H%M%S)"

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

# ============================================================
#  Logging
# ============================================================
mkdir -p "$SAVE_ROOT"
echo "Logs / results will be saved to: $SAVE_ROOT"
echo "Queue size: ${#GPU_IDS[@]} GPU(s), ${#MODEL_NAMES[@]} model(s)"

# ============================================================
#  Eval function for one model
# ============================================================
run_longbench_v1_eval() {
    local gpu_id="$1"
    local model_name="$2"
    local model_config="$3"
    local ckpt_path="$4"
    local save_dir="$SAVE_ROOT/${model_name}"
    local run_log="$SAVE_ROOT/${model_name}.run.log"

    mkdir -p "$save_dir"
    : > "$run_log"

    echo "[$(date '+%F %T')] [LongBench-v1] Start model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"

    local extra_args=()

    if [ -n "$model_config" ]; then
        extra_args+=(--config_path "$model_config")
        extra_args+=(--vocab_dir "$VOCAB_DIR")
    fi

    if [ -n "$DATASETS" ]; then
        extra_args+=(--datasets "$DATASETS")
    fi

    if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" eval/eval_longbench_v1.py \
        --checkpoint_path "$ckpt_path" \
        --save_dir "$save_dir" \
        --max_length "$MAX_LENGTH" \
        --n_proc 1 \
        "${extra_args[@]}" \
        2>&1 | tee -a "$run_log"; then
        echo "[$(date '+%F %T')] [ERROR] ${model_name} failed" | tee -a "$run_log"
        return 1
    fi

    echo "[$(date '+%F %T')] [LongBench-v1] Finished model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"
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
echo "--- Queueing LongBench v1 jobs ---"
for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    dispatch_job "${model_name}.longbench_v1" \
        run_longbench_v1_eval \
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
