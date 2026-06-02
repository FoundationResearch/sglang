#!/bin/bash
# ============================================================
# Merge multiple HF checkpoints (average) for a given experiment.
#
# For each experiment, specify:
#   EXP_NAME      — experiment name (used to locate HF ckpts)
#   STEPS         — space-separated list of global steps to merge
#   MEMO          — (optional) alias for the merged model, default: "merged_model"
#
# The script expects HF checkpoints at:
#   {DEST_ROOT}/{EXP_NAME}/global_step_{step}/hf_ckpt
#
# Usage:
#   EXP_NAME="xxx" STEPS="1000 2000 3000" MEMO="avg-1k-3k" \
#       bash scripts/ckpt_transfer/merge_ckpt.sh
# ============================================================

# ---------- Required env vars ----------
if [ -z "${EXP_NAME}" ] || [ -z "${STEPS}" ]; then
    echo "[ERROR] EXP_NAME and STEPS must be set."
    echo "Example:"
    echo "  EXP_NAME=\"xxx\" STEPS=\"1000 2000 3000\" bash $0"
    exit 1
fi
STEPS=(${STEPS})
DEST_ROOT="${DEST_ROOT:-/apdcephfs_fsgm/share_303843174/user/guhao/Models}"
MEMO="${MEMO:-merged_model}"
# ---------------------------------------

DEST_BASE_DIR="${DEST_ROOT}/${EXP_NAME}"

echo "============================================================"
echo "Experiment:  ${EXP_NAME}"
echo "Source root: ${DEST_BASE_DIR}"
echo "Steps:       ${STEPS[*]}"
echo "Memo:        ${MEMO}"
echo "============================================================"

# Build the list of checkpoint paths
CKPT_PATHS=()
for step in "${STEPS[@]}"; do
    ckpt_path="${DEST_BASE_DIR}/global_step_${step}/hf_ckpt"
    if [ ! -d "${ckpt_path}" ]; then
        echo "[ERROR] Checkpoint not found: ${ckpt_path}"
        exit 1
    fi
    CKPT_PATHS+=("${ckpt_path}")
done

OUTPUT_DIR="${DEST_BASE_DIR}/${MEMO}"

echo ""
echo ">>> Merging ${#CKPT_PATHS[@]} checkpoints (average) ..."
echo "    Output: ${OUTPUT_DIR}"

python3 utils/merge_checkpoints.py \
    --memo "${MEMO}" \
    --output "${OUTPUT_DIR}" \
    "${CKPT_PATHS[@]}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Merge failed."
    exit 1
fi

echo ""
echo "============================================================"
echo "[${EXP_NAME}] Merge done! Memo: ${MEMO}"
echo "============================================================"
