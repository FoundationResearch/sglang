export MODEL_CONFIG="configs/flash_hsa/config_lsa_wqproj_interleave_disturb0.5_dropout0.5_8KA1K_unified_1B.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="lsa-wqproj-interleave-disturb0.5-dropout0.5-8KA1K-unified-1B-300B"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/lsa-wqproj-interleave-disturb0.5-dropout0.5-8KA1K-unified-1B-300B"
export MAX_STEPS=143000
export SAVE_STEPS=5000
export TRAIN_SIZE=300000000000
export MICRO_BATCH_SIZE=8
export GLOBAL_BATCH_SIZE=256

bash scripts/pretrain/pretrain_ruler_task_5per_dist_300B.sh
