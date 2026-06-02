export MODEL_CONFIG="configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_interleave_origindrop0.1_8KA1K.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-interleave-win512-origindrop0.1-8KA1K-ruler"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa-interleave-win512-origindrop0.1-8KA1K-ruler-5per"

bash scripts/pretrain/pretrain_ruler_task_5per.sh
