export MODEL_CONFIG="configs/flash_hsa/config_lsa_unified.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="lsa-interleave-unified"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/lsa_interleave_unified"

bash scripts/pretrain/pretrain_ruler_task_5per.sh
