export MODEL_CONFIG="configs/flash_hsa/config_lsa_rope_interleave_hsawin0_recent8_disturb0.2_dropout0.2.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa-interleave-8KA2K-rope-woswa-hsawin0-recent8-disturb0.2-dropout0.2-tiledtoL"
export OUTPUT_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa-interleave-8KA2K-rope-woswa-hsawin0-recent8-disturb0.2-dropout0.2-tiledtoL"

bash scripts/pretrain/pretrain_ruler_task_5per.sh
