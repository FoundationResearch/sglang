
export MODEL_CONFIG="configs/swan_nsa/config_swan_nsa.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="nsa-swan-ruler"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/nsa-swan-ruler"

bash scripts/pretrain/pretrain_ruler_task_5per.sh
