export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_upper_16.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-upper-16/"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
# Single-stage training: train all parameters from the start (including MLP).
export WANDB_NAME="olmo3_lhsa_upper_16_40B_trainall"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/olmo3_lhsa_40B_trainall"
export TOKEN_CNT=40_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=3e-4
export MIN_LR=3e-5
# 40B tokens / (GLOBAL_BATCH_SIZE * MAX_SEQ_LEN) =
# 40,000,000,000 / (512 * 8192) ~= 9536.74 steps -> use 9537 to cover all 40B tokens.
export MAX_STEPS=9537

export SAVE_STEPS=1000
export LR_WARMUP_RATIO=$(python - <<'PY'
import os
max_steps = int(os.environ["MAX_STEPS"])
# Scale warmup to keep ~same warmup ratio as OLMo3's 2000-step warmup in the 100B-token run:
# 2000 / 23842 ~= 0.0839 -> 0.0839 * 9537 ~= 800 steps.
print(800 / max_steps)
PY
)
# OLMo3-style cosine decay to MIN_LR at the end of training.
export EXTRA_ARGS="--train.lr_decay_style cosine --train.lr_decay_ratio 1.0"

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh
