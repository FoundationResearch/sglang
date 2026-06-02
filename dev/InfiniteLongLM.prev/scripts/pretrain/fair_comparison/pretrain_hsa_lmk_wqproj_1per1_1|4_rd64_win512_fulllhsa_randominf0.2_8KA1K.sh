export MODEL_CONFIG="configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_fulllhsa_randominf0.2_8KA1K.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-full-lsa-win512-randrop0.2-8KA1K-ruler"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa-lmk-qproj-fulllhsa-win512-randominf0.2-8KA1K-ruler-5per"

bash scripts/pretrain/pretrain_ruler_task_5per.sh
