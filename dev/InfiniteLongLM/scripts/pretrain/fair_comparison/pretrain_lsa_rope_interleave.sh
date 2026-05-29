export MODEL_CONFIG="configs/flash_hsa/config_lsa_rope_wo_lmk_q.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="config_lsa_full_rope-wnoise"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/config_lsa_full-rope-ruler-5per"
export GRADIENT_CKPT=true

bash scripts/pretrain/pretrain_ruler_task_5per.sh
