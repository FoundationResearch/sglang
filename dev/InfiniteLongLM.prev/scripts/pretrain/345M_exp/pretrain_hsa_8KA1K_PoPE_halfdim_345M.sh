export MODEL_CONFIG="configs/flash_hsa/config_hsa_8KA1K_PoPE_halfdim_345M.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa_8KA1K_PoPE_halfdim_345M"
export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/hsa_8KA1K_PoPE_theta10000_halfdim_345M"
export GRADIENT_CKPT=true
bash scripts/pretrain/pretrain_ruler_task_5per_345M.sh