export MODEL_CONFIG="configs/flash_hsa/config_hsa_ultra_stb_win128.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8064
export WANDB_NAME="hsa-ultra-stb-top32-win128-ruler"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-ultra-stb-top32-win128-ruler"

bash scripts/pretrain/pretrain_ruler_task.sh