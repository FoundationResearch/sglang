#!/usr/bin/env bash
# ============================================================================
# eval_sglang_all.sh — Multi-GPU SGLang evaluation for multiple models/datasets
#
# Usage:
#   bash scripts/eval/eval_sglang_all.sh
#
# Each (model, dataset) pair runs on a single GPU via an independent
# sgl.Engine process. The script manages a GPU queue so all available
# GPUs are kept busy. Within each model, all datasets run in parallel
# (up to #GPUs). Models are evaluated sequentially.
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# ── Paths ──
TOKENIZER_PATH=${TOKENIZER_PATH:-configs/olmo3_vocab}
OPENCOMPASS_PATH=${OPENCOMPASS_PATH:-/apdcephfs_tj5/share_300719894/user/guhao/opencompass}
SGLANG_HSA_ROOT=${SGLANG_HSA_ROOT:-/apdcephfs_tj5/share_300719894/user/guhao/SGLang-HSA}
# Match the SGLang-HSA venv used for deps (plain `python` may be a different env).
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH="${REPO_ROOT}:${OPENCOMPASS_PATH}:${SGLANG_HSA_ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"

# ── Models ── (parallel arrays: name, config, weights, attention_backend)
#
# Two kinds of models:
#   - Standard OLMo3 (baseline/cpt): use the stock Olmo3 config (Olmo3ForCausalLM).
#     SGLang loads via its built-in models/olmo2.py. No --sglang-attention-backend.
#     Config: the original OLMo3 config_hf.json (architectures=Olmo3ForCausalLM).
#
#   - HSA/LHSA models: use HSA config (HSAForCausalLM) + attention_backend="hsa".
#     SGLang loads via models/flash_hsa.py.
#     Config: the training HSA config with insert_landmarks, hsa_heads, etc.
#
# For standard models whose checkpoint has no config.json, we provide the
# stock OLMo3 config path. prepare_hf_path will merge it in.
OLMO3_BASE_CONFIG=/apdcephfs_tj5/share_300719894/user/guhao/Models/OLMo-stage1-step999000/config_hf.json

MODEL_NAMES=(
    "8KA1K-w-noise-step12000"
    "8KA1K-w-noise-step13000"
)
HSA_CONFIGS=(
    # "$OLMO3_BASE_CONFIG"
    # "$OLMO3_BASE_CONFIG"
    configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_w_noise.json
    configs/olmo3_7B/olmo3_lhsa_interleave_8KA1K_non_unified_layerqk_w_noise.json
)
HF_PATHS=(
    /apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-interleave-8KA1K-non-unified-layerqk-w-noise-64gpu/global_step_12000/hf_ckpt
    /apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-interleave-8KA1K-non-unified-layerqk-w-noise-64gpu/global_step_13000/hf_ckpt
)
ATTN_BACKENDS=(
    "hsa"
    "hsa"
)

# ── Datasets ──
DATASET_LIST=(
    # PPL benchmarks (existing)
    # gpqa_few_shot_ppl_4b5a83
    # mmlu_ppl_ac766d
    # hellaswag_10shot_ppl_59c85e
    # ARC_c_few_shot_ppl
    # SuperGLUE_BoolQ_few_shot_ppl
    # race_few_shot_ppl
    # Generation benchmarks (new — configs in eval/configs/datasets/)
    gsm8k_gen
    math_gen
    cmath_gen
    humaneval_plus_gen
    mbpp_plus_gen
    cruxeval_o_gen
)

# ── GPUs & SGLang settings ──
GPU_IDS=(0 1 2 3 4 5 6 7)
SGLANG_TP=1
SGLANG_PAGE_SIZE=64
SGLANG_MAX_TOKENS=65536
SGLANG_MEM_FRACTION=0.95
SGLANG_BATCH_SIZE=1
DEBUG=${DEBUG:-1}

# ── Validation ──
if [ "${#MODEL_NAMES[@]}" -ne "${#HSA_CONFIGS[@]}" ] || \
   [ "${#MODEL_NAMES[@]}" -ne "${#HF_PATHS[@]}" ] || \
   [ "${#MODEL_NAMES[@]}" -ne "${#ATTN_BACKENDS[@]}" ]; then
    echo "MODEL_NAMES / HSA_CONFIGS / HF_PATHS / ATTN_BACKENDS length mismatch" >&2; exit 1
fi

# ── Output ──
RUN_TAG=$(date +%Y%m%d_%H%M%S)
MASTER_LOG_DIR="$SCRIPT_DIR/logs/eval_sglang_all_${RUN_TAG}"
mkdir -p "$MASTER_LOG_DIR"

echo "=============================================="
echo " SGLang Multi-GPU Evaluation"
echo "=============================================="
echo " Models:   ${MODEL_NAMES[*]}"
echo " Datasets: ${DATASET_LIST[*]}"
echo " GPUs:     ${GPU_IDS[*]}"
echo " Log dir:  $MASTER_LOG_DIR"
echo "=============================================="

# ── Prepare resolved HF path (symlink weights + tokenizer + build config.json) ──
prepare_hf_path() {
    local hf_path="$1" config_path="$2" resolved_hf_path="$3"
    rm -rf "$resolved_hf_path"
    mkdir -p "$resolved_hf_path"

    # Symlink checkpoint files first, then tokenizer files (tokenizer overrides)
    # Use readlink -f to resolve symlinks to real paths, ensuring they work across containers
    shopt -s nullglob dotglob
    for src in "$hf_path"/*;        do ln -sfn "$(readlink -f "$src")" "$resolved_hf_path/$(basename "$src")"; done
    for src in "$TOKENIZER_PATH"/*; do ln -sfn "$(readlink -f "$src")" "$resolved_hf_path/$(basename "$src")"; done
    shopt -u nullglob dotglob

    # Build config.json: start from checkpoint's own config (if exists), merge LMK flags from HSA config.
    # If checkpoint has no config.json, use HSA config directly as base.
    rm -f "$resolved_hf_path/config.json"
    "$PYTHON_BIN" - "$hf_path" "$config_path" "$resolved_hf_path/config.json" <<'PYEOF'
import json, os, sys
hf_dir, hsa_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
base_cfg_path = os.path.join(hf_dir, "config.json")
with open(hsa_path) as f: hsa = json.load(f)
if os.path.exists(base_cfg_path):
    # Start from checkpoint config, overlay ALL fields from HSA config
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
}

# ── GPU queue management ──
available_gpus=()
active_pids=()
active_gpus=()
active_labels=()
REAPED_GPU_ID=""
queue_failed=0

pop_gpu() { REAPED_GPU_ID="${available_gpus[0]}"; available_gpus=("${available_gpus[@]:1}"); }

reap_one_job() {
    local finished_pid wait_status=0
    wait -n -p finished_pid "${active_pids[@]}" || wait_status=$?
    [ "$wait_status" -ne 0 ] && queue_failed=1
    for idx in "${!active_pids[@]}"; do
        if [ "${active_pids[$idx]}" = "$finished_pid" ]; then
            available_gpus+=("${active_gpus[$idx]}")
            echo "[$(date '+%F %T')] Done: ${active_labels[$idx]} (gpu=${active_gpus[$idx]}, status=${wait_status})"
            unset 'active_pids[idx]' 'active_gpus[idx]' 'active_labels[idx]'
            active_pids=("${active_pids[@]}"); active_gpus=("${active_gpus[@]}"); active_labels=("${active_labels[@]}")
            return
        fi
    done
}

drain_all() { while [ "${#active_pids[@]}" -gt 0 ]; do reap_one_job; done; }

# ── Single (model, dataset) evaluation ──
run_one() {
    local gpu_id="$1" model_name="$2" hsa_config="$3" dataset="$4" work_dir="$5" attn_backend="$6"
    local resolved_hf_path="$work_dir/hf_with_tokenizer"
    local hf_path="${HF_PATHS[$model_idx]}"
    local run_log="$work_dir/run.log"

    prepare_hf_path "$hf_path" "$hsa_config" "$resolved_hf_path"

    local cmd=(
        "$PYTHON_BIN" eval/eval_opencompass_sglang.py
        --hf-path "$resolved_hf_path"
        --hsa-config "$hsa_config"
        --sglang-tp "$SGLANG_TP"
        --sglang-page-size "$SGLANG_PAGE_SIZE"
        --sglang-max-total-tokens "$SGLANG_MAX_TOKENS"
        --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION"
        --sglang-batch-size "$SGLANG_BATCH_SIZE"
        --datasets "$dataset"
        -w "$work_dir"
    )
    [ -n "$attn_backend" ] && cmd+=(--sglang-attention-backend "$attn_backend")
    [ "$DEBUG" = "1" ] && cmd+=(--debug)

    echo "[$(date '+%F %T')] Start: model=${model_name}, dataset=${dataset}, gpu=${gpu_id}"
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" 2>&1 | tee "$run_log"
}

# ── Main loop: flatten all (model, dataset) pairs into a single queue ──
# GPUs never sit idle as long as there are pending tasks.
available_gpus=("${GPU_IDS[@]}")
active_pids=(); active_gpus=(); active_labels=()

for model_idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$model_idx]}"
    hsa_config="${HSA_CONFIGS[$model_idx]}"
    attn_backend="${ATTN_BACKENDS[$model_idx]}"
    model_log_dir="$MASTER_LOG_DIR/$model_name"
    mkdir -p "$model_log_dir"

    for dataset in "${DATASET_LIST[@]}"; do
        # Wait for a GPU to become available
        [ "${#available_gpus[@]}" -eq 0 ] && reap_one_job
        pop_gpu; gpu_id="$REAPED_GPU_ID"

        ds_tag=$(printf '%s' "$dataset" | tr ',/' '__' | tr -cd '[:alnum:]_.-')
        work_dir="$model_log_dir/$ds_tag"
        mkdir -p "$work_dir"

        run_one "$gpu_id" "$model_name" "$hsa_config" "$dataset" "$work_dir" "$attn_backend" &
        active_pids+=($!); active_gpus+=("$gpu_id"); active_labels+=("${model_name}/${dataset}")
    done
done
drain_all
echo "[$(date '+%F %T')] All models and datasets done."

# ── Generate LaTeX summary ──
"$PYTHON_BIN" - "$MASTER_LOG_DIR" <<'PYEOF'
import csv, io, os, sys, re

master = sys.argv[1]
DATASET_ORDER = [
    ("mmlu_ppl_ac766d",              "MMLU(5-shot)"),
    ("gpqa_few_shot_ppl_4b5a83",     "GPQA(5-shot)"),
    ("hellaswag_10shot_ppl_59c85e",  "Hellaswag(10-shot)"),
    ("ARC_c_few_shot_ppl",           "ARC-c(25-shot)"),
    ("SuperGLUE_BoolQ_few_shot_ppl", "BoolQ(5-shot)"),
    ("race_few_shot_ppl",            "Race(3-shot)"),
    ("gsm8k_gen",                    "GSM8K(4-shot)"),
    ("math_gen",                     "MATH(4-shot)"),
    ("cmath_gen",                    "CMATH(4-shot)"),
    ("humaneval_plus_gen",           "HumanEval+(0-shot)"),
    ("mbpp_plus_gen",                "MBPP+(3-shot)"),
    ("cruxeval_o_gen",               "CRUxEval-O(2-shot)"),
]

def find_summary(d):
    for r, _, fs in os.walk(d):
        for f in sorted(fs):
            if f.startswith("summary_") and f.endswith(".txt"):
                return os.path.join(r, f)
    return None

def extract_score(path):
    with open(path) as f: text = f.read()
    idx = text.find("csv format\n")
    if idx < 0: return None
    csv_text = text[idx+len("csv format\n"):]
    nl = csv_text.find("\n")
    if nl < 0: return None
    csv_text = csv_text[nl+1:].lstrip("\n")
    lines = [l for l in csv_text.splitlines() if l.strip() and "," in l]
    if not lines: return None
    for row in csv.reader(io.StringIO("\n".join(lines))):
        if row and row[0].strip() == "overall_average":
            for cell in reversed(row):
                try: return float(cell.strip())
                except: pass
    # Fallback: last data row, last numeric column
    for row in csv.reader(io.StringIO("\n".join(lines))):
        if row and row[0].strip() != "dataset":
            for cell in reversed(row):
                try: return float(cell.strip())
                except: pass
    return None

models = sorted(d for d in os.listdir(master) if os.path.isdir(os.path.join(master, d)))
scores = {}
for m in models:
    scores[m] = {}
    for ds_internal, _ in DATASET_ORDER:
        ds_tag = re.sub(r"[^A-Za-z0-9_.\-]", "", ds_internal.replace(",","_").replace("/","_"))
        sf = find_summary(os.path.join(master, m, ds_tag))
        if sf:
            v = extract_score(sf)
            if v is not None: scores[m][ds_internal] = v

display = [dn for _, dn in DATASET_ORDER]
header = "& " + " & ".join(display + ["AVG"]) + "\\\\"
lines = ["% Auto-generated SGLang evaluation summary", f"% Log dir: {master}", "", header, "\\midrule"]
for m in models:
    vals, valid = [], []
    for ds, _ in DATASET_ORDER:
        v = scores[m].get(ds)
        if v is not None: vals.append(f"{v:.2f}"); valid.append(v)
        else: vals.append("-")
    avg = f"{sum(valid)/len(valid):.2f}" if valid else "-"
    vals.append(avg)
    lines.append(f"{m} & " + " & ".join(vals) + "\\\\")

out = "\n".join(lines) + "\n"
p = os.path.join(master, "summary.log")
with open(p, "w") as f: f.write(out)
print(f"Summary: {p}\n{out}")
PYEOF

echo ""
echo "=============================================="
echo " All done! Results in: $MASTER_LOG_DIR"
echo "=============================================="

[ "$queue_failed" -ne 0 ] && { echo "Some jobs failed." >&2; exit 1; }
