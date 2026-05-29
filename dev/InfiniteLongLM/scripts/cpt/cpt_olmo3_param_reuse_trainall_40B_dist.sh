export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ "${USE_LOCAL_VEOMNI_SRC:-1}" = "1" ]; then
    export PYTHONPATH="${PROJECT_ROOT}/../veomni_src:${PROJECT_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
fi

export MODEL_CONFIG="configs/olmo3_7B/olmo3_param_reuse.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/guhao/checkpoints/olmo3_param_reuse"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="olmo3_param_reuse_trainall_40B_disable_liger_rmsnorm"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/olmo3_param_reuse_trainall_40B"
export TOKEN_CNT=40_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=$(python3 - <<'PY'
import math
import os

token_cnt = int(os.environ["TOKEN_CNT"].replace("_", ""))
global_bsz = int(os.environ["GLOBAL_BATCH_SIZE"].replace("_", ""))
max_seq_len = int(os.environ["MAX_SEQ_LEN"].replace("_", ""))
print(math.ceil(token_cnt / (global_bsz * max_seq_len)))
PY
)
export SAVE_STEPS=1000
export LR_WARMUP_RATIO=$(python3 - <<'PY'
import os
max_steps = int(os.environ["MAX_STEPS"])
print(200 / max_steps)
PY
)

export EXTRA_ARGS="--train.lr_decay_style cosine --train.lr_decay_ratio 1.0"
export USE_LIGER_KERNEL=1

export USE_LIGER_RMSNORM=0
export USE_LIGER_ROPE=1
export USE_LIGER_SWIGLU=1
export USE_LIGER_CE=1

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh
