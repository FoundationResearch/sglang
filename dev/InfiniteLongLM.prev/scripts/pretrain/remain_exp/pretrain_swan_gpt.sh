export MODEL_CONFIG="configs/swan_gpt_tiny/config_std_hd64.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="swangpt_ruler-single"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/swangpt-ruler-5per-single"

bash scripts/pretrain/pretrain_ruler_task_5per.sh
