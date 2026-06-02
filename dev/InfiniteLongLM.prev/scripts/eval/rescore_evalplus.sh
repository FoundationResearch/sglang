#!/usr/bin/env bash
# ============================================================================
# rescore_evalplus.sh — re-score HumanEval+ / MBPP+ predictions offline
#
# Usage:
#   bash scripts/eval/rescore_evalplus.sh <run_log_dir>
#
# Example:
#   bash scripts/eval/rescore_evalplus.sh \
#     scripts/eval/logs/eval_sglang_all_20260420_120347
#
# Walks the log dir, finds every `humaneval_plus.json` and `mbpp_plus.json`
# predictions file, runs evalplus on each with the updated scoring rules,
# and prints the pass@1 scores.
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
RESCORE_PY="$REPO_ROOT/eval/rescore_evalplus.py"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <run_log_dir>" >&2
    echo "Example: $0 scripts/eval/logs/eval_sglang_all_20260420_120347" >&2
    exit 1
fi

run_dir=$1
if [ ! -d "$run_dir" ]; then
    echo "ERROR: directory not found: $run_dir" >&2
    exit 2
fi

echo "==================================================================="
echo " Re-scoring HumanEval+ / MBPP+ predictions under:"
echo "   $run_dir"
echo "==================================================================="
echo

rescore_one() {
    local pred_path=$1 dataset=$2 tag=$3
    if [ ! -e "$pred_path" ]; then
        return
    fi
    echo "-------------------------------------------------------------------"
    echo " $dataset  |  $tag"
    echo "   $pred_path"
    echo "-------------------------------------------------------------------"
    "$PYTHON_BIN" "$RESCORE_PY" "$pred_path" --dataset "$dataset" --tag "$tag"
    echo
}

# Collect predictions directories containing either a merged file or shard
# files (humaneval_plus_0.json, humaneval_plus_1.json, ...). We dedupe via
# an associative array keyed on the model tag so one dir is scored once.
declare -A seen_humaneval=() seen_mbpp=()

process_humaneval_dir() {
    local pred_dir=$1
    local tag
    tag=$(echo "$pred_dir" | sed -E 's|.*/([^/]+)/humaneval_plus_gen/.*|\1|')
    if [ -n "${seen_humaneval[$tag]:-}" ]; then return; fi
    seen_humaneval[$tag]=1

    if [ -f "$pred_dir/humaneval_plus.json" ]; then
        rescore_one "$pred_dir/humaneval_plus.json" humaneval "$tag"
    else
        # Shard mode: pass the directory; the Python side merges shards.
        rescore_one "$pred_dir" humaneval "$tag"
    fi
}

process_mbpp_dir() {
    local pred_dir=$1
    local tag
    tag=$(echo "$pred_dir" | sed -E 's|.*/([^/]+)/mbpp_plus_gen/.*|\1|')
    if [ -n "${seen_mbpp[$tag]:-}" ]; then return; fi
    seen_mbpp[$tag]=1

    if [ -f "$pred_dir/mbpp_plus.json" ]; then
        rescore_one "$pred_dir/mbpp_plus.json" mbpp "$tag"
    else
        rescore_one "$pred_dir" mbpp "$tag"
    fi
}

# HumanEval+: find any directory holding humaneval_plus*.json (merged or shard).
while IFS= read -r f; do
    process_humaneval_dir "$(dirname "$f")"
done < <(find "$run_dir" -path "*/humaneval_plus_gen/*/predictions/*/humaneval_plus*.json" 2>/dev/null | sort)

# MBPP+: find any directory holding mbpp_plus*.json.
while IFS= read -r f; do
    process_mbpp_dir "$(dirname "$f")"
done < <(find "$run_dir" -path "*/mbpp_plus_gen/*/predictions/*/mbpp_plus*.json" 2>/dev/null | sort)

echo "==================================================================="
echo " Done."
echo "==================================================================="
