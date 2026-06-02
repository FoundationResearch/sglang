export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ "${USE_LOCAL_VEOMNI_SRC:-1}" = "1" ]; then
    export PYTHONPATH="${PROJECT_ROOT}/../veomni_src:${PROJECT_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
fi

export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_innerx.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-innerx/"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="lhsa-olmo3-innerx-lr3e-4-drop0.2-disturb0.2"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-innerx-lr3e-4-warmup"
export LOAD_CHECKPOINT_PATH=""
export TOKEN_CNT=500_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=3e-4
export MIN_LR=3e-4
export MAX_STEPS=${MAX_STEPS:-auto}
export SAVE_STEPS=250
export LR_WARMUP_RATIO=0.0067

export EXTRA_ARGS="--train.lr_decay_style constant --train.no_decay_params norm bias embed"

export USE_LIGER_KERNEL=1

export USE_LIGER_RMSNORM=0
export USE_FLASH_ATTN_RMSNORM=1
export USE_LIGER_ROPE=1
export USE_LIGER_SWIGLU=1
export USE_LIGER_CE=1

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

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh 2>&1 | tee "$LOG_FILE"

