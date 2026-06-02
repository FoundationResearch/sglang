export MODEL_CONFIG="configs/flash_hsa/config_hsa_full_pope_halfdim_8KA1K_noise_0.5_0.1_1B.json"
export CORPUS_PATH="/apdcephfs_tj5/share_300719894/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-full-pope-halfdim-8KA1K-noise-0.5-0.1-1B-300B"
export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/hsa-full-pope-halfdim-8KA1K-noise-0.5-0.1-1B-300B"
export MAX_STEPS=143000
export SAVE_STEPS=5000
export TRAIN_SIZE=300000000000
export MICRO_BATCH_SIZE=8
export GLOBAL_BATCH_SIZE=256

bash scripts/pretrain/pretrain_ruler_task_5per_dist_300B.sh