#!/usr/bin/env bash
# ============================================================================
# eval_sglang.sh — Single-GPU SGLang evaluation for one dataset
#
# Usage:
#   bash scripts/eval/eval_sglang.sh
#   CUDA_VISIBLE_DEVICES=3 bash scripts/eval/eval_sglang.sh
#
# For multi-GPU parallel evaluation, use eval_sglang_all.sh instead.
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# ── Paths ──
model_path=/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave-8KA1K-non-unified-no-noise-layerqk-64gpu-warmup1k/global_step_1000/hf_ckpt
config_path=configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_wo_noise.json
TOKENIZER_PATH=${TOKENIZER_PATH:-configs/olmo3_vocab}
OPENCOMPASS_PATH=${OPENCOMPASS_PATH:-/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/opencompass}
SGLANG_HSA_ROOT=${SGLANG_HSA_ROOT:-/apdcephfs_fsgm/share_303843174/user/guhao/SGLang-HSA}
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH="${REPO_ROOT}:${OPENCOMPASS_PATH}:${SGLANG_HSA_ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"

datasets=${1:-SuperGLUE_BoolQ_few_shot_ppl}

# ── Output ──
DATASET_TAG=$(printf '%s' "$datasets" | tr ',/' '__' | tr -cd '[:alnum:]_.-')
WORK_DIR="$SCRIPT_DIR/logs/eval_sglang_${DATASET_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR"

# ── Prepare resolved HF path ──
resolved_hf_path="$WORK_DIR/hf_with_tokenizer"
mkdir -p "$resolved_hf_path"

shopt -s nullglob dotglob
for src in "$model_path"/*;        do ln -sfn "$(readlink -f "$src")" "$resolved_hf_path/$(basename "$src")"; done
for src in "$TOKENIZER_PATH"/*;    do ln -sfn "$(readlink -f "$src")" "$resolved_hf_path/$(basename "$src")"; done
shopt -u nullglob dotglob

# Build config.json: checkpoint config (if exists) + HSA config merge
rm -f "$resolved_hf_path/config.json"
$PYTHON_BIN - "$model_path" "$config_path" "$resolved_hf_path/config.json" <<'PYEOF'
import json, os, sys
hf_dir, hsa_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
base_cfg_path = os.path.join(hf_dir, "config.json")
with open(hsa_path) as f: hsa = json.load(f)
if os.path.exists(base_cfg_path):
    with open(base_cfg_path) as f: config = json.load(f)
    config.update(hsa)
else:
    config = hsa
config["insert_landmarks"] = bool(config.get("insert_landmarks") or config.get("adjust_lmk_pos"))
config["adjust_lmk_pos"]   = bool(config.get("adjust_lmk_pos", False))
with open(dst_path, "w") as f: json.dump(config, f, indent=2); f.write("\n")
print(f"[config] arch={config.get('architectures')}, vocab_size={config.get('vocab_size')}, "
      f"insert_lmk={config['insert_landmarks']}, adjust_lmk_pos={config['adjust_lmk_pos']}, "
      f"base={'checkpoint' if os.path.exists(base_cfg_path) else 'hsa_config'}")
PYEOF

echo "=============================================="
echo " SGLang Evaluation"
echo " Model:    $model_path"
echo " Dataset:  $datasets"
echo " Work dir: $WORK_DIR"
echo "=============================================="


# ── Run ──
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON_BIN eval/eval_opencompass_sglang.py \
    --hf-path "$resolved_hf_path" \
    --hsa-config "$config_path" \
    --sglang-tp 1 \
    --sglang-page-size 64 \
    --sglang-max-total-tokens 65536 \
    --sglang-mem-fraction-static 0.95 \
    --sglang-batch-size 4 \
    --sglang-attention-backend hsa \
    --datasets "$datasets" \
    -w "$WORK_DIR" \
    --debug \
    2>&1 | tee "$WORK_DIR/run.log"

echo ""
echo "Done! Results in: $WORK_DIR"
