export MODEL_CONFIG="configs/swan_gpt_tiny/config_rope_full_theta10000_1B.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="rope-full-attn-swangpt-1B-300B"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/rope-full-attn-rope-swangpt-1B-300B"

export MAX_STEPS=143000
export SAVE_STEPS=5000
export TRAIN_SIZE=300000000000
export MICRO_BATCH_SIZE=8
export GLOBAL_BATCH_SIZE=256

bash scripts/pretrain/pretrain_ruler_task_5per_dist_300B.sh
