#!/bin/bash
# 8K-512K bench WITH CUDA Graph (prefill stays eager, decode under CG).
# Removes --disable-cuda-graph. Uses --cuda-graph-max-bs 1 for fast capture.

set -u
LENGTHS=("${@:-8192 16384 32768 65536 131072 262144 524288}")
DECODE_LEN=32

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

run_one() {
    local model_path=$1 input_len=$2 label=$3 ctx=$4 backend=$5 memfrac=$6
    echo "==== $label  input=${input_len}  ctx=${ctx}  backend=${backend}  CG=ON ===="
    timeout 1800 python -m sglang.bench_one_batch \
        --model-path "$model_path" \
        --load-format dummy \
        --tp 1 --batch-size 1 \
        --input-len "$input_len" --output-len "$DECODE_LEN" \
        --context-length "$ctx" \
        --attention-backend "$backend" \
        --page-size 64 \
        --cuda-graph-max-bs 1 \
        --mem-fraction-static "$memfrac" \
        --trust-remote-code 2>&1 \
      | grep -E "Prefill\.|Decode\.  median|Total\.|Error|Traceback|OOM|out of memory|exceeds|too large|skipping" \
      | tail -10
    echo "(done)"
}

for L in ${LENGTHS[@]}; do
    CTX=$((L + 200))
    # Lower mem-frac for >128K to leave room for HSA intermediate buffers.
    if [ "$L" -gt 131072 ]; then
        MF=0.45
    else
        MF=0.80
    fi
    echo "===================================================="
    echo "CONTEXT BUCKET: $L  (mem-frac=$MF)"
    echo "===================================================="
    run_one "/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real"    "$L" "HSA-345M-real"      "$CTX" "hsa"    "$MF"
    run_one "/home/hal-alex/workspace/sglang/dev/bench_models/dense345m_fair"  "$L" "Dense-Fair-345M"    "$CTX" "triton" "$MF"
done
echo "=== CG-on sweep done ==="
