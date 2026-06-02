#!/usr/bin/env bash
# ============================================================
# LongBench v1 evaluation using SGLang Engine (HSA models)
#
# Supports:
#   - Dataset subset presets (--subset)
#   - Multi-GPU dataset-level parallelism: each GPU runs an
#     independent SGLang Engine on a subset of datasets for
#     the SAME model.
#
# Usage:
#   bash scripts/eval/eval_olmo3_longbench_v1_sglang.sh
#   SUBSET=fewshot_code bash scripts/eval/eval_olmo3_longbench_v1_sglang.sh
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
#  SGLang-HSA paths
# ============================================================
TOKENIZER_PATH=${TOKENIZER_PATH:-configs/olmo3_vocab}
SGLANG_HSA_ROOT=${SGLANG_HSA_ROOT:-/apdcephfs_fsgm/share_303843174/user/guhao/SGLang-HSA}
export PYTHONPATH="${REPO_ROOT}:${SGLANG_HSA_ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================
#  Models: "name|config|ckpt_path"
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
#  GPU pool & SGLang settings
# ============================================================
GPU_IDS=(
    0 1 2 3 4 5 6 7
)

MAX_LENGTH=65536
SGLANG_TP=1
SGLANG_PAGE_SIZE=64
SGLANG_MAX_TOTAL_TOKENS=131072
SGLANG_MEM_FRACTION_STATIC=0.90
SGLANG_BATCH_SIZE=1              # HSA backend 当前只支持 bsz=1
SGLANG_MAX_RUNNING_REQUESTS=1    # 限制 scheduler 并发，防止 continuous batching 攒 batch

SAVE_ROOT="$SCRIPT_DIR/logs/eval_longbench_v1_sglang_$(date +%Y%m%d_%H%M%S)"

# ============================================================
#  Dataset subset presets
#
#  Set SUBSET env var before running, e.g.:
#    SUBSET=fewshot_code bash scripts/eval/eval_olmo3_longbench_v1_sglang.sh
#
#  Available presets:
#    all           - all 21 datasets (default)
#    single_doc_qa - narrativeqa,qasper,multifieldqa_en,multifieldqa_zh
#    multi_doc_qa  - hotpotqa,2wikimqa,musique,dureader
#    summarization - gov_report,qmsum,multi_news,vcsum
#    fewshot       - trec,triviaqa,samsum,lsht
#    synthetic     - passage_count,passage_retrieval_en,passage_retrieval_zh
#    code          - lcc,repobench-p
#    fewshot_code  - trec,triviaqa,samsum,lsht,lcc,repobench-p
#    qa            - single_doc_qa + multi_doc_qa
#    Or any comma-separated dataset names directly.
# ============================================================
SUBSET=${SUBSET:-all}

resolve_subset() {
    local s="$1"
    case "$s" in
        all)
            echo ""  # empty = all datasets in Python script
            ;;
        single_doc_qa)
            echo "narrativeqa,qasper,multifieldqa_en,multifieldqa_zh"
            ;;
        multi_doc_qa)
            echo "hotpotqa,2wikimqa,musique,dureader"
            ;;
        summarization)
            echo "gov_report,qmsum,multi_news,vcsum"
            ;;
        fewshot)
            echo "trec,triviaqa,samsum,lsht"
            ;;
        synthetic)
            echo "passage_count,passage_retrieval_en,passage_retrieval_zh"
            ;;
        code)
            echo "lcc,repobench-p"
            ;;
        fewshot_code)
            echo "trec,triviaqa,samsum,lsht,lcc,repobench-p"
            ;;
        qa)
            echo "narrativeqa,qasper,multifieldqa_en,multifieldqa_zh,hotpotqa,2wikimqa,musique,dureader"
            ;;
        *)
            # Treat as raw comma-separated dataset names
            echo "$s"
            ;;
    esac
}

DATASETS=$(resolve_subset "$SUBSET")
echo "Dataset subset: SUBSET=${SUBSET} -> DATASETS='${DATASETS:-all}'"

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
#  Prepare resolved HF path (merge config + tokenizer)
# ============================================================
prepare_resolved_hf_path() {
    local model_path="$1"
    local config_path="$2"
    local resolved_dir="$3"

    mkdir -p "$resolved_dir"

    shopt -s nullglob dotglob
    for src in "$model_path"/*; do
        ln -sfn "$(readlink -f "$src")" "$resolved_dir/$(basename "$src")"
    done
    for src in "$TOKENIZER_PATH"/*; do
        ln -sfn "$(readlink -f "$src")" "$resolved_dir/$(basename "$src")"
    done
    shopt -u nullglob dotglob

    # Build merged config.json (checkpoint config + HSA config)
    rm -f "$resolved_dir/config.json"
    $PYTHON_BIN - "$model_path" "$config_path" "$resolved_dir/config.json" <<'PYEOF'
import json, os, sys
hf_dir, hsa_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
base_cfg_path = os.path.join(hf_dir, "config.json")
with open(hsa_path) as f: hsa = json.load(f)
if os.path.exists(base_cfg_path):
    with open(base_cfg_path) as f: config = json.load(f)
    config.update(hsa)
else:
    config = hsa
config["insert_landmarks"] = bool(config.get("insert_landmarks") or config.get("adjust_lmk_pos"))
config["adjust_lmk_pos"]   = bool(config.get("adjust_lmk_pos", False))
with open(dst_path, "w") as f: json.dump(config, f, indent=2); f.write("\n")
print(f"[config] arch={config.get('architectures')}, vocab_size={config.get('vocab_size')}, "
      f"insert_lmk={config['insert_landmarks']}, adjust_lmk_pos={config['adjust_lmk_pos']}")
PYEOF
}

# ============================================================
#  Split a comma-separated dataset list into N roughly-equal parts.
#  Returns comma-separated dataset string for part index $2 (0-based).
# ============================================================
split_datasets() {
    local datasets_csv="$1"
    local n_parts="$2"
    local part_idx="$3"

    # Convert CSV to array
    IFS=',' read -ra ds_arr <<< "$datasets_csv"
    local total=${#ds_arr[@]}

    # Compute start and count for this part (round-robin is simpler
    # but contiguous chunks are easier to reason about)
    local base_size=$((total / n_parts))
    local remainder=$((total % n_parts))
    local start=0
    for ((i = 0; i < part_idx; i++)); do
        local extra=0
        if [ "$i" -lt "$remainder" ]; then extra=1; fi
        start=$((start + base_size + extra))
    done
    local size=$base_size
    if [ "$part_idx" -lt "$remainder" ]; then size=$((size + 1)); fi

    if [ "$size" -eq 0 ]; then
        echo ""
        return
    fi

    # Slice the array and join with commas
    local result=""
    for ((i = start; i < start + size; i++)); do
        if [ -n "$result" ]; then result="${result},"; fi
        result="${result}${ds_arr[$i]}"
    done
    echo "$result"
}

# ============================================================
#  Generate worker: runs one SGLang Engine on one GPU
#  for a specific subset of datasets.
# ============================================================
run_sglang_generate() {
    local gpu_id="$1"
    local model_name="$2"
    local resolved_hf_path="$3"
    local model_config="$4"
    local ds_subset="$5"
    local save_dir="$6"
    local run_log="$7"

    echo "[$(date '+%F %T')] [GPU ${gpu_id}] Start generate: model=${model_name}, datasets=${ds_subset}" | tee -a "$run_log"

    local extra_args=()
    if [ -n "$ds_subset" ]; then
        extra_args+=(--datasets "$ds_subset")
    fi

    local attn_backend_args=()
    if [ -n "$model_config" ]; then
        attn_backend_args+=(--sglang-attention-backend hsa)
    fi

    local sglang_port=$((31000 + gpu_id * 100))

    if ! SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
         CUDA_VISIBLE_DEVICES="$gpu_id" \
         "$PYTHON_BIN" eval/eval_longbench_v1_sglang.py \
        --hf-path "$resolved_hf_path" \
        --save-dir "$save_dir" \
        --max-length "$MAX_LENGTH" \
        --sglang-tp "$SGLANG_TP" \
        --sglang-page-size "$SGLANG_PAGE_SIZE" \
        --sglang-max-total-tokens "$SGLANG_MAX_TOTAL_TOKENS" \
        --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
        --sglang-batch-size "$SGLANG_BATCH_SIZE" \
        --sglang-max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS" \
        --sglang-port "$sglang_port" \
        --generate-only \
        "${attn_backend_args[@]}" \
        "${extra_args[@]}" \
        2>&1 | tee -a "$run_log"; then
        echo "[$(date '+%F %T')] [ERROR] GPU ${gpu_id}: ${model_name} generate failed" | tee -a "$run_log"
        return 1
    fi

    echo "[$(date '+%F %T')] [GPU ${gpu_id}] Finished generate: model=${model_name}" | tee -a "$run_log"
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
#  Resolve all datasets to run (for splitting purposes)
# ============================================================
if [ -n "$DATASETS" ]; then
    ALL_DS_CSV="$DATASETS"
else
    ALL_DS_CSV="narrativeqa,qasper,multifieldqa_en,multifieldqa_zh,hotpotqa,2wikimqa,musique,dureader,gov_report,qmsum,multi_news,vcsum,trec,triviaqa,samsum,lsht,passage_count,passage_retrieval_en,passage_retrieval_zh,lcc,repobench-p"
fi

IFS=',' read -ra _all_ds_arr <<< "$ALL_DS_CSV"
TOTAL_DATASETS=${#_all_ds_arr[@]}

# ============================================================
#  Dispatch: for each model, split datasets across GPUs
# ============================================================
echo "--- Queueing LongBench v1 SGLang jobs ---"

for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    model_config="${MODEL_CONFIGS[$idx]}"
    ckpt_path="${CKPT_PATHS[$idx]}"
    save_dir="$SAVE_ROOT/${model_name}"
    run_log_base="$SAVE_ROOT/${model_name}"

    mkdir -p "$save_dir"

    # Prepare resolved HF path once per model
    resolved_hf_path="$save_dir/hf_with_tokenizer"
    if [ -n "$model_config" ]; then
        prepare_resolved_hf_path "$ckpt_path" "$model_config" "$resolved_hf_path"
    else
        resolved_hf_path="$ckpt_path"
    fi

    # Determine number of GPU workers for this model
    # (cap at total datasets — no point having more workers than datasets)
    n_workers=${#GPU_IDS[@]}
    if [ "$n_workers" -gt "$TOTAL_DATASETS" ]; then
        n_workers=$TOTAL_DATASETS
    fi

    echo "[${model_name}] Splitting ${TOTAL_DATASETS} datasets across ${n_workers} GPUs"

    for ((part = 0; part < n_workers; part++)); do
        part_datasets=$(split_datasets "$ALL_DS_CSV" "$n_workers" "$part")
        if [ -z "$part_datasets" ]; then
            continue
        fi
        echo "  GPU worker ${part}: ${part_datasets}"

        dispatch_job "${model_name}.part${part}" \
            run_sglang_generate \
            "$model_name" \
            "$resolved_hf_path" \
            "$model_config" \
            "$part_datasets" \
            "$save_dir" \
            "${run_log_base}.part${part}.log"
    done
done

# Wait for all generate jobs
while [ "${#active_pids[@]}" -gt 0 ]; do
    reap_one_job
done

echo ""
echo "=== All generate jobs complete ==="

# ============================================================
#  Final evaluation: score all predictions per model
# ============================================================
echo "--- Running final evaluation ---"

for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    model_config="${MODEL_CONFIGS[$idx]}"
    ckpt_path="${CKPT_PATHS[$idx]}"
    save_dir="$SAVE_ROOT/${model_name}"

    resolved_hf_path="$save_dir/hf_with_tokenizer"
    if [ -z "$model_config" ]; then
        resolved_hf_path="$ckpt_path"
    fi

    local_extra_args=()
    if [ -n "$DATASETS" ]; then
        local_extra_args+=(--datasets "$DATASETS")
    fi

    echo "[${model_name}] Evaluating predictions in ${save_dir}/pred/"

    "$PYTHON_BIN" eval/eval_longbench_v1_sglang.py \
        --hf-path "$resolved_hf_path" \
        --save-dir "$save_dir" \
        --eval-only \
        "${local_extra_args[@]}" \
        2>&1 | tee -a "$SAVE_ROOT/${model_name}.eval.log"
done

echo ""
echo "All results saved to: $SAVE_ROOT"

if [ "$failed" -ne 0 ]; then
    echo "Some evaluation jobs failed." >&2
    exit 1
fi
