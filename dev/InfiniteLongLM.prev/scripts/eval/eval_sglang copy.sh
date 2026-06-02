set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# ── Paths ──
model_path=/apdcephfs_fsgm/share_303843174/user/guhao/Models/lhsa-olmo3-interleave/avg-1k-to-10k
config_path=configs/olmo3_7B/olmo3_lhsa_interleave.json
TOKENIZER_PATH=${TOKENIZER_PATH:-/apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_vocab}
OPENCOMPASS_PATH=${OPENCOMPASS_PATH:-/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/opencompass}
SGLANG_HSA_ROOT=${SGLANG_HSA_ROOT:-/apdcephfs_fsgm/share_303843174/user/guhao/SGLang-HSA}

# ── Python / PYTHONPATH ──
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH="${REPO_ROOT}:${OPENCOMPASS_PATH}:${SGLANG_HSA_ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"

datasets=ARC_c_few_shot_ppl
DATASET_TAG=$(printf '%s' "$datasets" | tr ',/' '__' | tr -cd '[:alnum:]_.-')
# ── Output ──
WORK_DIR="$SCRIPT_DIR/logs/eval_sglang_${DATASET_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR"

# ── Prepare resolved HF path: checkpoint + tokenizer + config ──
resolved_hf_path="$WORK_DIR/hf_with_tokenizer"
mkdir -p "$resolved_hf_path"

# Symlink checkpoint weights
shopt -s nullglob dotglob
for src in "$model_path"/*; do
    ln -sfn "$src" "$resolved_hf_path/$(basename "$src")"
done
# Symlink tokenizer files (overrides any from checkpoint)
for src in "$TOKENIZER_PATH"/*; do
    ln -sfn "$src" "$resolved_hf_path/$(basename "$src")"
done
shopt -u nullglob dotglob

# Build config.json: copy from HSA config, set landmark flags.
# NOTE: Do NOT override vocab_size — SGLang needs it to match the model weights.
# (The HF eval script overrides vocab_size for tokenizer compat, but SGLang's
#  flash_hsa.py handles vocab padding internally via _get_flashhsa_padded_vocab_size.)
$PYTHON_BIN - "$config_path" "$resolved_hf_path/config.json" <<'PYEOF'
import json, sys

src_cfg_path, dst_cfg_path = sys.argv[1], sys.argv[2]

with open(src_cfg_path, "r") as f:
    config = json.load(f)

insert_lmk = bool(config.get("insert_landmarks", False) or config.get("adjust_lmk_pos", False))
adjust_lmk_pos = bool(config.get("adjust_lmk_pos", False))

config["insert_landmarks"] = insert_lmk
config["adjust_lmk_pos"] = adjust_lmk_pos
# vocab_size is kept as-is from the original config

with open(dst_cfg_path, "w") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")

print(f"[config] vocab_size={config.get('vocab_size')}, insert_lmk={insert_lmk}, adjust_lmk_pos={adjust_lmk_pos}")
PYEOF

echo "=============================================="
echo " SGLang GPQA Minimal Verification"
echo "=============================================="
echo "Model:      $model_path"
echo "Config:     $config_path"
echo "Tokenizer:  $TOKENIZER_PATH"
echo "Resolved:   $resolved_hf_path"
echo "Work dir:   $WORK_DIR"
echo "=============================================="

# ── Run evaluation ──
# SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0: HSA landmark token insertion
# causes KV cache token counts to mismatch, triggering a false memory leak warning.
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1 2 3 4 5 6 7} $PYTHON_BIN eval/eval_opencompass_sglang.py \
    --hf-path "$resolved_hf_path" \
    --hsa-config "$config_path" \
    --sglang-tp 1 \
    --sglang-page-size 64 \
    --sglang-max-total-tokens 65536 \
    --sglang-mem-fraction-static 0.95 \
    --sglang-batch-size 4 \
    --datasets "$datasets" \
    -w "$WORK_DIR" \
    --debug \
    2>&1 | tee "$WORK_DIR/run.log"

    # --no-prefix-cache \


echo ""
echo "=============================================="
echo " Done! Results in: $WORK_DIR"
echo "=============================================="
