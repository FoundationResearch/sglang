#!/bin/bash
# ============================================================
# Batch convert & move: loop over multiple experiments.
#
# For each experiment, specify:
#   EXP_NAME      — experiment name (also used as dest folder name)
#   CKPT_BASE_DIR — source checkpoint base dir
#   STEPS         — space-separated list of global steps
#
# Usage:
#   bash scripts/ckpt_transfer/batch_convert_and_move.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_and_move.sh"


# EXP_NAME="lhsa-olmo3-7B-8KA2K-wo-lmk-q-proj" \
# STEPS="12000 13000" \
# bash "${CONVERT_SCRIPT}"


EXP_NAME="lhsa-olmo3-7B-8KA2K-w-lmk-q-proj" \
STEPS="12000 13000" \
bash "${CONVERT_SCRIPT}"

echo ""
echo "############################################################"
echo "  All experiments finished!"
echo "############################################################"
