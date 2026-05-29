export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ "${USE_LOCAL_VEOMNI_SRC:-1}" = "1" ]; then
    export PYTHONPATH="${PROJECT_ROOT}/../veomni_src:${PROJECT_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
fi

export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_innerx.json"
export MODEL_PATH="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-innerx/"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="lhsa-olmo3-innerx-fixmlp-3e-4"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-innerx-fixmlp"
export TOKEN_CNT=40_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=3e-4
export MIN_LR=3e-4
export MAX_STEPS=4768
export SAVE_STEPS=1000
export LR_WARMUP_RATIO=0

export EXTRA_ARGS="--train.freeze_pattern '\.mlp\.' --train.lr_decay_style constant"

export USE_LIGER_KERNEL=1

export USE_LIGER_RMSNORM=0
export USE_FLASH_ATTN_RMSNORM=1
export USE_LIGER_ROPE=1
export USE_LIGER_SWIGLU=1
export USE_LIGER_CE=1

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh
