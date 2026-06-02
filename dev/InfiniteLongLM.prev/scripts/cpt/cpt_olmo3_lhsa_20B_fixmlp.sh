export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/lhsa-olmo3-7B/"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="olmo3_lhsa_100B"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/olmo3_lhsa_100B"
export TOKEN_CNT=100_000_000_000
export BATCH_SIZE=8
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=4768
export LR_WARMUP_RATIO=$(python - <<'PY'
import os
max_steps = int(os.environ["MAX_STEPS"])
print(50 / max_steps)
PY
)
export EXTRA_ARGS="--train.freeze_pattern *.mlp.* --train.lr_decay_style constant"

bash scripts/cpt/CPT.sh