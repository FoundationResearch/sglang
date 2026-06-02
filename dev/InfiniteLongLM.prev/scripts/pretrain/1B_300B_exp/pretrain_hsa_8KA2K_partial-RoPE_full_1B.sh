export MODEL_CONFIG="configs/flash_hsa/config_hsa_8KA2K_partial-RoPE_1B.json"
export CORPUS_PATH="/apdcephfs_tj5/share_300719894/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa_8KA2K_RoPE_partial-full-wonoise-1B-300B"
export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/hsa_8KA2K_RoPE_partial-full-wonoise-1B-300B"
export MAX_STEPS=143000
export SAVE_STEPS=5000
export TRAIN_SIZE=300000000000
export MICRO_BATCH_SIZE=8
export GLOBAL_BATCH_SIZE=256
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



bash scripts/pretrain/pretrain_ruler_task_5per_dist_300B.sh