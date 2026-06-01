#!/bin/bash
# Sweep latency across context-length buckets, compare HSA sglang vs Qwen dense sglang.
# Uses --cuda-graph-max-bs 1 to skip the 52-batch capture (only bench bs=1).
# Outputs ALL stdout/stderr so we can see errors; results are tail-greppable.
# Usage: bash bench_long_context_sweep.sh

set -u
LENGTHS=(8192 16384 32768 65536 131072 262144 524288)
DECODE_LEN=128
MEM_FRAC=${MEM_FRAC:-0.85}

run_one() {
    local model_path=$1 input_len=$2 label=$3 ctx=$4
    echo "==== $label  input=${input_len}  ctx=${ctx} ===="
    timeout 1800 python -m sglang.bench_one_batch \
        --model-path "$model_path" \
        --load-format dummy \
        --tp 1 --batch-size 1 \
        --input-len "$input_len" --output-len "$DECODE_LEN" \
        --context-length "$ctx" \
        --cuda-graph-max-bs 1 \
        --mem-fraction-static "$MEM_FRAC" \
        --trust-remote-code 2>&1 \
      | grep -E "Prefill\.|Decode\.  median|Total\.|Error|Traceback|OOM|out of memory|signal|exceeds|too large" \
      | tail -10
    echo "(done)"
}

for L in "${LENGTHS[@]}"; do
    CTX=$((L + 200))
    echo "===================================================="
    echo "CONTEXT BUCKET: $L"
    echo "===================================================="
    run_one "/home/hal-alex/workspace/hsa7b_dummy"   "$L" "HSA-7B"     "$CTX"
    run_one "Qwen/Qwen2.5-7B-Instruct"             "$L" "Qwen2.5-7B" "$CTX"
done
echo "=== Sweep done ==="
