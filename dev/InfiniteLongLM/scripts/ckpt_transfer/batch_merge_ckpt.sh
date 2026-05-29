#!/bin/bash
# ============================================================
# Batch merge: average merge checkpoints from 1k to 10k steps.
#
# For each experiment, specify:
#   EXP_NAME — experiment name
#   STEPS    — space-separated list of global steps (1000 to 10000)
#   MEMO     — alias for the merged model
#
# Usage:
#   bash scripts/ckpt_transfer/batch_merge_ckpt.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_ckpt.sh"

# ==================== Add your experiments below ====================

EXP_NAME="olmo3-param-reuse-lr3e-4" \
STEPS="7000 8000 9000 10000 11000 12000 13000 13500 14000 14500" \
MEMO="avg-7k-to-14.5k" \
bash "${MERGE_SCRIPT}"


EXP_NAME="lhsa-olmo3-interleave" \
STEPS="7000 8000 9000 10000 11000 12000 13000 13500 14000 14500" \
MEMO="avg-7k-to-14.5k" \
bash "${MERGE_SCRIPT}"

# EXP_NAME="lhsa-olmo3-innerx-lr3e-4-warmup" \
# STEPS="7000 8000 9000 10000 11000 12000 13000 13250 13500 13750" \
# MEMO="avg-7k-to-13.75k" \
# bash "${MERGE_SCRIPT}"

echo ""
echo "############################################################"
echo "  All merge experiments finished!"
echo "############################################################"
