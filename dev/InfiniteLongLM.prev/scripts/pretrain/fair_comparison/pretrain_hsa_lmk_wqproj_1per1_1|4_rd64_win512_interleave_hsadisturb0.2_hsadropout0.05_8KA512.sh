export MODEL_CONFIG="configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_wqproj_interleave_hsadisturb0.2_hsadropout0.05_8KA512.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-interleave-win512-hsadisturb0.2-hsadropout0.05-8KA512-ruler"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa-interleave-win512-hsadisturb0.2-hsadropout0.05-8KA512-ruler-5per"
bash scripts/pretrain/pretrain_ruler_task_5per.sh
