export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_full.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/guhao/checkpoints/full-lhsa-olmo3-7B/"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="full-lhsa-olmo3-7B-100B-fixmlp"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/full-lhsa-olmo3-7B-100B-fixmlp"
export TOKEN_CNT=100_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=4768
export SAVE_STEPS=1000
export LR_WARMUP_RATIO=$(python - <<'PY'
import os
max_steps = int(os.environ["MAX_STEPS"])
print(50 / max_steps)
PY
)
export EXTRA_ARGS="--train.freeze_pattern '*.mlp.*' --train.include_frozen_params_in_optimizer true --train.lr_decay_style constant"

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh
