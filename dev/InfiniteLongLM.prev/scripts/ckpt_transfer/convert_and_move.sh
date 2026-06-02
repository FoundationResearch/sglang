#!/bin/bash
# ============================================================
# Convert DCP checkpoints to HF format, then move to target dir.
#
# Supports two modes:
#   1. Standalone: edit the variables below and run directly.
#   2. Called by parent script: pass EXP_NAME, CKPT_BASE_DIR,
#      STEPS (space-separated) as env vars.
#
# Example (standalone):
#   bash scripts/ckpt_transfer/convert_and_move.sh
#
# Example (from parent):
#   EXP_NAME="xxx" CKPT_BASE_DIR="/path/to" STEPS="8000 9537" \
#       bash scripts/ckpt_transfer/convert_and_move.sh
# ============================================================

# ---------- Required env vars ----------
# EXP_NAME, STEPS must be set by the caller.
# CKPT_BASE_DIR, CKPT_ROOT, DEST_ROOT are optional (have sensible defaults).
if [ -z "${EXP_NAME}" ] || [ -z "${STEPS}" ]; then
    echo "[ERROR] EXP_NAME and STEPS must be set."
    echo "Example:"
    echo "  EXP_NAME=\"xxx\" STEPS=\"8000 9537\" bash $0"
    exit 1
fi
STEPS=(${STEPS})
CKPT_ROOT="${CKPT_ROOT:-/apdcephfs_tj5/share_300719894/user/guhao/checkpoints}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-${CKPT_ROOT}/${EXP_NAME}/checkpoints}"
DEST_ROOT="${DEST_ROOT:-/apdcephfs_fsgm/share_303843174/user/guhao/Models}"
# ---------------------------------------

DEST_BASE_DIR="${DEST_ROOT}/${EXP_NAME}"

MAX_PARALLEL="${MAX_PARALLEL:-4}"

echo "============================================================"
echo "Experiment:  ${EXP_NAME}"
echo "Source:      ${CKPT_BASE_DIR}"
echo "Destination: ${DEST_BASE_DIR}"
echo "Steps:       ${STEPS[*]}"
echo "Parallel:    ${MAX_PARALLEL}"
echo "============================================================"

# ---------- worker function (runs in background) ----------
convert_one_step() {
    local step="$1"
    local ckpt_path="${CKPT_BASE_DIR}/global_step_${step}"
    local dest_dir="${DEST_BASE_DIR}/global_step_${step}"
    local log_file="${CKPT_BASE_DIR}/.convert_step_${step}.log"

    {
        echo ">>> [Step ${step}] Converting DCP -> HF ..."
        echo "    ckpt_path: ${ckpt_path}"

        python3 utils/convert_dcp_to_hf_ckpt.py \
            --save_checkpoint_path "${ckpt_path}"

        if [ $? -ne 0 ]; then
            echo "    [ERROR] Conversion failed for step ${step}, skipping move."
            return 1
        fi

        echo "    [OK] Conversion done."

        # echo ">>> [Step ${step}] Moving hf_ckpt -> ${dest_dir} ..."
        # mkdir -p "${dest_dir}"
        # cp -r "${ckpt_path}/hf_ckpt" "${dest_dir}/"

        # if [ $? -ne 0 ]; then
        #     echo "    [ERROR] Move failed for step ${step}."
        #     return 1
        # fi

        # echo "    [OK] Moved to ${dest_dir}"
    } > "${log_file}" 2>&1

    local rc=$?
    cat "${log_file}"
    rm -f "${log_file}"
    return ${rc}
}

# ---------- parallel dispatch with concurrency limit ----------
running=0
for step in "${STEPS[@]}"; do
    convert_one_step "${step}" &
    running=$((running + 1))

    if [ ${running} -ge ${MAX_PARALLEL} ]; then
        wait -n  # wait for any one job to finish
        running=$((running - 1))
    fi
done

# wait for remaining jobs
wait

echo ""
echo "============================================================"
echo "[${EXP_NAME}] All done!"
echo "============================================================"
