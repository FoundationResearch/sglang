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

# Stage2 resume: continue training from stage2 global_step_7000.
export WANDB_NAME="lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage2-3e-4to3e-5"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage2-3e-4to3e-5"
# 总 token 目标仍然是 40B，MAX_STEPS 保持不变。
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

# ---- 与 stage2 原脚本的关键区别 ----
# 从 stage2 的 global_step_7000 恢复，而非 stage1。
# 加载 optimizer 和 lr_scheduler 状态，使训练无缝继续。
export STAGE2_RESUME_CKPT="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-dropout-fix-swa-mlp-40B-stage2-3e-4to3e-5/checkpoints/global_step_7000"

# stage2 原始的 decay_ratio 仍然需要设置（与原 stage2 脚本保持一致），
# 因为 lr_scheduler 状态会从 checkpoint 恢复，这里只是确保配置一致。
export STAGE1_STEP=4769
export REMAINING_STEPS=$((MAX_STEPS - STAGE1_STEP))
export STAGE2_DECAY_RATIO=$(awk "BEGIN { printf \"%.18g\", ${REMAINING_STEPS} / ${MAX_STEPS} }")

export LR_WARMUP_RATIO=0.0

# Resume model + optimizer + lr_scheduler + dataloader/RNG from stage2 checkpoint.
export EXTRA_ARGS="\
  --train.load_checkpoint_path ${STAGE2_RESUME_CKPT} \
  --train.load_optimizer_state true \
  --train.load_lr_scheduler_state true \
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

if [ -z "${STAGE2_RESUME_CKPT}" ] || [ ! -d "${STAGE2_RESUME_CKPT}" ]; then
  echo "ERROR: please ensure STAGE2_RESUME_CKPT exists. Got: '${STAGE2_RESUME_CKPT}'"
  exit 1
fi

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh
