export MODEL_CONFIG="configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_full-lhsa_candgate0.1_candrop0.03.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-full-lsa-win512-candgate0.1-candrop0.03-ruler"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa-full-lsa-win512-candgate0.1-candrop0.03-ruler"
bash scripts/pretrain/pretrain_ruler_task_5per.sh
