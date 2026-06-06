#!/bin/bash
# Bench the REAL HSA backend path (no cuda graph delegation).
# Uses --disable-cuda-graph + --attention-backend hsa so that
# HSAAttnBackend.forward_decode -> _run_selection_decode + _compute_internal_swa_decode
# are actually exercised (rather than the dense delegate that cuda-graph capture uses).
#
# Comparison is apples-to-apples: both HSA and Dense disable cuda graph so per-step
# Python overhead is the same in both.

set -u
LENGTHS=("${1:-32768}")  # default 32K; override: bash bench_real_hsa.sh 8192 32768 131072
DECODE_LEN=64

if [ "$#" -gt 1 ]; then
    LENGTHS=("$@")
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

run_one() {
    local model_path=$1 input_len=$2 label=$3 ctx=$4 backend=$5
    echo "==== $label  input=${input_len}  ctx=${ctx}  backend=${backend} ===="
    timeout 1800 python -m sglang.bench_one_batch \
        --model-path "$model_path" \
        --load-format dummy \
        --tp 1 --batch-size 1 \
        --input-len "$input_len" --output-len "$DECODE_LEN" \
        --context-length "$ctx" \
        --attention-backend "$backend" \
        --disable-cuda-graph \
        --mem-fraction-static 0.50 \
        --trust-remote-code 2>&1 \
      | grep -E "Prefill\.|Decode\.  median|Total\.|Error|Traceback|OOM|out of memory|exceeds|too large|skipping" \
      | tail -10
    echo "(done)"
}

for L in "${LENGTHS[@]}"; do
    CTX=$((L + 200))
    echo "===================================================="
    echo "CONTEXT BUCKET: $L"
    echo "===================================================="
    run_one "/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real"    "$L" "HSA-345M-real"      "$CTX" "hsa"
    run_one "/home/hal-alex/workspace/sglang/dev/bench_models/dense345m_fair"  "$L" "Dense-Fair-345M"    "$CTX" "triton"
done
echo "=== Real-HSA sweep done ==="
