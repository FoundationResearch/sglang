
export MODEL_CONFIG="configs/flash_hsa/config_hsa_lmk_win512_w_lmk_q.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-lmk-qproj-win512-ruler"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-lmk-qproj-win512-ruler-5per"

bash scripts/pretrain/pretrain_ruler_task_5per.sh