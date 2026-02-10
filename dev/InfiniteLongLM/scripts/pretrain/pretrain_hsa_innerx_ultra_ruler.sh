export MODEL_CONFIG="configs/flash_hsa/config_swan_nope_sparse_innerx_win512_perlayer.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8064
export WANDB_NAME="hsa-innerx-1/4-top32-win512-ruler"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-innerx-1/4-top32-win512-ruler"

bash scripts/pretrain/pretrain_ruler_task.sh