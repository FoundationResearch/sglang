
export MODEL_CONFIG="configs/swan_nsa/config_hybrid_nsa.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="nsa-ultra-1per1-top32-win512-ruler-5per"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/nsa-innerx-ultra-1per1-5per"

bash scripts/pretrain/pretrain_ruler_task_5per.sh
