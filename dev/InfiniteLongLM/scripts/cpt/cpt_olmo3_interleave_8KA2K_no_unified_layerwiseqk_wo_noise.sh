export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified_layerqk_wo_noise.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-interleave-non-unified-layerwiseqk/"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="lhsa-olmo3-interleave-8KA2K-non-unified-layerqk-wo-noise"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-interleave-8KA2K-non-unified-no-noise-layerqk-64gpu"
export TOKEN_CNT=500_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe_64gpu.yaml"
export MAX_LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=13_000  # 50B tokens
export SAVE_STEPS=1_000
export LR_WARMUP_RATIO=0.0067

export EXTRA_ARGS="--train.lr_decay_style cosine --train.no_decay_params norm bias embed"

if [ "$MAX_STEPS" = "auto" ] || [ "$MAX_STEPS" = "-1" ]; then
    token_cnt_clean=${TOKEN_CNT//_/}
    global_batch_size_clean=${GLOBAL_BATCH_SIZE//_/}
    max_seq_len_clean=${MAX_SEQ_LEN//_/}
    tokens_per_step=$((global_batch_size_clean * max_seq_len_clean))

    if [ "$tokens_per_step" -le 0 ]; then
        echo "ERROR: GLOBAL_BATCH_SIZE * MAX_SEQ_LEN must be > 0, got ${GLOBAL_BATCH_SIZE} * ${MAX_SEQ_LEN}" >&2
        exit 1
    fi

    export MAX_STEPS=$((token_cnt_clean / tokens_per_step))

    if [ "$MAX_STEPS" -le 0 ]; then
        echo "ERROR: computed MAX_STEPS=${MAX_STEPS} from TOKEN_CNT=${TOKEN_CNT}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, MAX_SEQ_LEN=${MAX_SEQ_LEN}" >&2
        exit 1
    fi
fi

echo "Using MAX_STEPS=${MAX_STEPS} (TOKEN_CNT=${TOKEN_CNT}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, MAX_SEQ_LEN=${MAX_SEQ_LEN})"

MAX_PREFETCH_RETRIES=10
for i in $(seq 1 $MAX_PREFETCH_RETRIES); do
    echo "[Prefetch] Attempt $i/$MAX_PREFETCH_RETRIES ..."
    python code_exp/prefetch.py $MODEL_CONFIG
    if [ $? -eq 0 ]; then
        echo "[Prefetch] Success on attempt $i."
        break
    fi
    if [ $i -eq $MAX_PREFETCH_RETRIES ]; then
        echo "[Prefetch] Failed after $MAX_PREFETCH_RETRIES attempts, aborting." >&2
        exit 1
    fi
    echo "[Prefetch] Attempt $i failed, retrying..."
    sleep 2
done

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh 2>&1 | tee "$LOG_FILE"
