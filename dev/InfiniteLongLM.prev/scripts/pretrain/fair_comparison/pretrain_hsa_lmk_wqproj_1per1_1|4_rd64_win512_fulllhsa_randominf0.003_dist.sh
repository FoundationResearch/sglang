export MODEL_CONFIG="configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_fulllhsa_randominf0.003.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-full-lsa-win512-randrop0.003-ruler"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa-lmk-qproj-fulllhsa-win512-randominf0.003-ruler-5per"

export MAX_STEPS=30000
export SAVE_STEPS=10000
export TRAIN_SIZE=10000000000

bash scripts/pretrain/pretrain_ruler_task_5per_dist.sh
