export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ "${USE_LOCAL_VEOMNI_SRC:-0}" = "1" ]; then
    export PYTHONPATH="${PROJECT_ROOT}/../veomni_src:${PROJECT_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
fi
export MODELING_BACKEND=hf

export MODEL_CONFIG="/apdcephfs_sh8/share_300719895/shared/models/OLMO3/OLMo-stage1-step999000/config.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/shared/models/OLMO3/OLMo-stage1-step999000"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=1024
export WANDB_DISABLED=true
export USE_WANDB=false
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/olmo3_param_reuse_trainall_40B_debug"
export TOKEN_CNT=100_000_000
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
export SAVE_STEPS=50
export LR_WARMUP_RATIO=$(python3 - <<'PY'
import os
max_steps = int(os.environ["MAX_STEPS"])
ratio = 800 / max_steps if max_steps > 0 else 0.0
print(min(0.1, ratio))
PY
)

export EXTRA_ARGS="--train.lr_decay_style cosine --train.lr_decay_ratio 1.0 --train.debug_grad_topk 20 --train.debug_grad_steps 3 --train.enable_reentrant true"

pkill -f "burner"

cd "$PROJECT_ROOT"
bash scripts/cpt/CPT.sh
