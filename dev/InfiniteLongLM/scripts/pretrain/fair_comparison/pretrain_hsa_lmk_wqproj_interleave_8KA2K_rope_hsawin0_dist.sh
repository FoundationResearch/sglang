export MODEL_CONFIG="configs/flash_hsa/config_lsa_rope_interleave_hsawin0.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-interleave-8KA2K-rope-woswa-hsawin0-dist"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa-interleave-8KA2K-rope-woswa-hsawin0-dist"

export MAX_STEPS=30000
export SAVE_STEPS=10000
export TRAIN_SIZE=10000000000
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=128

bash scripts/pretrain/pretrain_ruler_task_5per_dist.sh
