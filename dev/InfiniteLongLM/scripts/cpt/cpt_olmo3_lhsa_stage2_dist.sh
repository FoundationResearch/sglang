export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_upper_16.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-upper-16/"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192

# Stage2: unfreeze MLP and keep training without repeating data (resume dataloader state).
export WANDB_NAME="olmo3-lhsa-upper-16-40B-stage2"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/olmo3-lhsa-upper-16-40B-stage2"
export TOKEN_CNT=40_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=$(python - <<'PY'
import math
import os

token_cnt = int(os.environ["TOKEN_CNT"].replace("_", ""))
global_bsz = int(os.environ["GLOBAL_BATCH_SIZE"].replace("_", ""))
max_seq_len = int(os.environ["MAX_SEQ_LEN"].replace("_", ""))
print(math.ceil(token_cnt / (global_bsz * max_seq_len)))
PY
)
export SAVE_STEPS=1000
# Stage1 ckpt step should be the 20B boundary (so stage2 adds ~20B and total reaches 40B).
# With packed training, 20B tokens ~= ceil(20B / (GLOBAL_BATCH_SIZE * MAX_SEQ_LEN)) steps.
export STAGE1_CKPT="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/olmo3_lhsa_100B/checkpoints/global_step_4000"
export LR_WARMUP_RATIO=$(python - <<'PY'
import os
max_steps = int(os.environ["MAX_STEPS"])
print(50 / max_steps)
PY
)
if [ -z "${STAGE1_CKPT}" ] || [ ! -d "${STAGE1_CKPT}" ]; then
  echo "ERROR: please set STAGE1_CKPT to an existing stage1 checkpoint dir (global_step_xxx). Got: '${STAGE1_CKPT}'"
  exit 1
fi

export EXTRA_ARGS="\
  --train.load_checkpoint_path ${STAGE1_CKPT} \
  --train.load_optimizer_state false \
  --train.load_lr_scheduler_state true \
  --train.include_frozen_params_in_optimizer false \
  --train.lr_decay_style cosine \
  --train.lr_decay_ratio 1.0 \
"

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh
