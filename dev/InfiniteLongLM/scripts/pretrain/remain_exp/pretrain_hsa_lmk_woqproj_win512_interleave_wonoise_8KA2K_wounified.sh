export MODEL_CONFIG="configs/flash_hsa/config_hsa_lmk_swa512_hsawin512_woqproj_interleave_wonoise_8KA2K.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-woqproj-interleave-win512-8KA2K-wounified-wonoise"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa-woqproj-interleave-win512-8KA2K-wounified-wonoise-ruler-5per"
bash scripts/pretrain/pretrain_ruler_task_5per.sh
