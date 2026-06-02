export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ "${USE_LOCAL_VEOMNI_SRC:-1}" = "1" ]; then
    export PYTHONPATH="${PROJECT_ROOT}/../veomni_src:${PROJECT_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
fi

export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_dropout.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-dropout/"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192

# Stage2: unfreeze all params and keep training without repeating data (resume dataloader state).
export WANDB_NAME="lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage2-3e-4to3e-5"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage2-3e-4to3e-5"
# Stage1 已经训练了 20B，本脚本再训 20B，使总 token 达到 40B。
# 因为 MAX_STEPS 是“总”步数，所以 TOKEN_CNT 需要设为总 token 数。
export TOKEN_CNT=40_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=3e-4
export MIN_LR=3e-5
# Compute total steps needed for TOKEN_CNT tokens in total.
export TOKEN_CNT_INT=${TOKEN_CNT//_/}
export TOKENS_PER_STEP=$((GLOBAL_BATCH_SIZE * MAX_SEQ_LEN))
export MAX_STEPS=$(((TOKEN_CNT_INT + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
export SAVE_STEPS=1000


export STAGE1_CKPT="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage1/checkpoints/global_step_4769"
export STAGE1_STEP=$(basename "${STAGE1_CKPT}" | sed -E 's/.*global_step_([0-9]+)/\1/')
export REMAINING_STEPS=$((MAX_STEPS - STAGE1_STEP))
if [ "${REMAINING_STEPS}" -le 0 ]; then
  echo "ERROR: stage2 remaining steps must be positive, got MAX_STEPS=${MAX_STEPS}, STAGE1_STEP=${STAGE1_STEP}"
  exit 1
fi
export STAGE2_DECAY_RATIO=$(awk "BEGIN { printf \"%.18g\", ${REMAINING_STEPS} / ${MAX_STEPS} }")

export LR_WARMUP_RATIO=0.0

# Resume model + dataloader/RNG to avoid repeating data.
# Rebuild optimizer (all params unfrozen) and start a fresh stage2-only cosine
# schedule over the remaining steps, instead of inheriting stage1's scheduler progress.
export EXTRA_ARGS="\
  --train.load_checkpoint_path ${STAGE1_CKPT} \
  --train.load_optimizer_state false \
  --train.load_lr_scheduler_state false \
  --train.include_frozen_params_in_optimizer false \
  --train.lr_decay_style cosine \
  --train.lr_decay_ratio ${STAGE2_DECAY_RATIO} \
"

export USE_LIGER_KERNEL=1

# Liger RMSNorm backward has unstable grads here, so keep Liger on for the
# other kernels but route RMSNorm to flash-attn instead.
export USE_LIGER_RMSNORM=0
export USE_FLASH_ATTN_RMSNORM=1
export USE_LIGER_ROPE=1
export USE_LIGER_SWIGLU=1
export USE_LIGER_CE=1

if [ -z "${STAGE1_CKPT}" ] || [ ! -d "${STAGE1_CKPT}" ]; then
  echo "ERROR: please set STAGE1_CKPT to an existing stage1 checkpoint dir (global_step_xxx). Got: '${STAGE1_CKPT}'"
  exit 1
fi

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh
