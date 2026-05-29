export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_half_pope.json"


CEPH_MODEL_PATH="/apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-half-pope"
LOCAL_MODEL_PATH="/root/local_ckpt/lhsa-olmo3-half-pope"
if [ ! -f "${LOCAL_MODEL_PATH}/pytorch_model.bin" ]; then
    echo "[ckpt-cache] Copying checkpoint to local disk: ${LOCAL_MODEL_PATH} ..."
    rm -rf "${LOCAL_MODEL_PATH}"
    mkdir -p "${LOCAL_MODEL_PATH}"
    cp "${CEPH_MODEL_PATH}/pytorch_model.bin" "${LOCAL_MODEL_PATH}/pytorch_model.bin"
    echo "[ckpt-cache] Copy done ($(du -sh "${LOCAL_MODEL_PATH}/pytorch_model.bin" | cut -f1))."
else
    echo "[ckpt-cache] Local checkpoint already exists, skipping copy."
fi

export MODEL_PATH="${LOCAL_MODEL_PATH}"
export CORPUS_PATH="/apdcephfs_tj5/share_300719894/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="lhsa-olmo3-half-pope"
export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-half-pope"
export TOKEN_CNT=500_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe_64gpu.yaml"
export MAX_LR=2e-4
export MIN_LR=2e-5
export MAX_STEPS=13_000  # 50B tokens
export SAVE_STEPS=1_000
export LR_WARMUP_RATIO=$(python - << 'PY'
import os
max_steps = int(os.environ["MAX_STEPS"])
print(1000 / max_steps)
PY
)

export EXTRA_ARGS="\
  --train.load_optimizer_state true \
  --train.load_lr_scheduler_state true \
  --train.load_dataloader_state true \
  --train.load_rng_state true \
  --train.include_frozen_params_in_optimizer false \
  --train.lr_decay_style cosine \
  --train.no_decay_params norm bias embed \
"

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
    python code_exp/flash_hsa_run.py
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
