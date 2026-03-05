rm -fr /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-ultra-top32-win512-ruler-rd256

export MODEL_CONFIG="configs/flash_hsa/config_hsa_ultra_win512_adjlmk_rd256.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-ultra-top32-win512-ruler-rd256"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-ultra-top32-win512-ruler-rd256"

bash scripts/pretrain/pretrain_ruler_task_100per.sh