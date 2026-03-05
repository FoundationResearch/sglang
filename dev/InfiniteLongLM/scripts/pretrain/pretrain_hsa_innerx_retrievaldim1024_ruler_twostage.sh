set -e
export WANDB_PROJECT="ruler_pretrain_twostage"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"


# stage 1
export WANDB_NAME="hsa-innerx-top32-stage1-win128-retrievaldim1024"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/${WANDB_NAME}"
export MODEL_CONFIG="configs/flash_hsa/config_swan_nope_sparse_innerx_stage1_win128_retrievaldim1024.json"
export MAX_SEQ_LEN=2048
export MAX_STEPS=5000
export SAVE_STEPS=5000

bash scripts/pretrain/pretrain_ruler_task_twostage.sh




# stage 2
STAGE1_WEIGHTS_PATH="${OUTPUT_DIR}/checkpoints/global_step_${MAX_STEPS}/hf_ckpt"

export WANDB_NAME="hsa-innerx-top32-stage2-win512-retrievaldim1024"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/${WANDB_NAME}"
export MODEL_CONFIG="configs/flash_hsa/config_swan_nope_sparse_innerx_stage2_win512_retrievaldim1024.json"
export MAX_SEQ_LEN=8192
export MAX_STEPS=25000
export SAVE_STEPS=5000
export EXTRA_ARGS="--model.model_path ${STAGE1_WEIGHTS_PATH}"

bash scripts/pretrain/pretrain_ruler_task_twostage.sh
