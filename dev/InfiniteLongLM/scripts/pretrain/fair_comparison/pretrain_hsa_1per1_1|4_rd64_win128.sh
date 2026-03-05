
export MODEL_CONFIG="configs/flash_hsa/config_hsa_win128_1per1_1|4_rd64.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-ultra-top32-win128-ruler"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-ultra-top32-win128-ruler-5per"

bash scripts/pretrain/pretrain_ruler_task_5per.sh