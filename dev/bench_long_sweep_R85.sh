#!/bin/bash
# Comprehensive HSA vs Dense bench: 8K -> 512K context.
# Both backends with --disable-cuda-graph for apples-to-apples per-step overhead.
# Page size 64 to satisfy HSA chunk_size requirement.

set -u
LENGTHS=("${@:-8192 16384 32768 65536 131072 262144 524288}")
DECODE_LEN=8

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
        --page-size 64 \
        --disable-cuda-graph \
        --mem-fraction-static 0.80 \
        --trust-remote-code 2>&1 \
      | grep -E "Prefill\.|Decode\.  median|Total\.|Error|Traceback|OOM|out of memory|exceeds|too large|skipping" \
      | tail -10
    echo "(done)"
}

for L in ${LENGTHS[@]}; do
    CTX=$((L + 200))
    echo "===================================================="
    echo "CONTEXT BUCKET: $L"
    echo "===================================================="
    run_one "/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real"    "$L" "HSA-345M-real"      "$CTX" "hsa"
    run_one "/home/hal-alex/workspace/sglang/dev/bench_models/dense345m_fair"  "$L" "Dense-Fair-345M"    "$CTX" "triton"
done
echo "=== Long-context sweep done ==="
