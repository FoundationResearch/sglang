export PYTHONPATH=./

export MODEL_CONFIG="configs/flash_hsa/config_lsa_yoco.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="config_lsa_yoco_gpu32"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/config_lsa_newyoco-ruler-5per"

export MAX_STEPS=30000
export SAVE_STEPS=10000
export TRAIN_SIZE=10000000000
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=128

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

bash scripts/pretrain/pretrain_ruler_task_5per_dist.sh
