#!/bin/bash
# Clean R87 sweep — explicitly sleeps between runs for full GPU memory release.
# Logs to /tmp/sweep_R87_clean.log.

set -u
LENGTHS=("${@:-8192 16384 32768 65536 131072 262144 524288}")
DECODE_LEN=32

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
LOG=/tmp/sweep_R87_clean.log
> "$LOG"

run_one() {
    local model_path=$1 input_len=$2 label=$3 ctx=$4 backend=$5 memfrac=$6
    echo "==== $label  input=${input_len}  ctx=${ctx}  backend=${backend}  CG=ON ====" >> "$LOG"
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
        --trust-remote-code >> "$LOG" 2>&1
    echo "(done)" >> "$LOG"
    # Force GPU memory release between runs.
    sleep 5
}

for L in ${LENGTHS[@]}; do
    CTX=$((L + 200))
    if [ "$L" -gt 131072 ]; then
        MF=0.45
    else
        MF=0.80
    fi
    echo "====================================================" >> "$LOG"
    echo "CONTEXT BUCKET: $L  (mem-frac=$MF)" >> "$LOG"
    echo "====================================================" >> "$LOG"
    run_one "/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real"    "$L" "HSA-345M-real"      "$CTX" "hsa"    "$MF"
    run_one "/home/hal-alex/workspace/sglang/dev/bench_models/dense345m_fair"  "$L" "Dense-Fair-345M"    "$CTX" "triton" "$MF"
done
echo "=== sweep done ===" >> "$LOG"
echo "EXIT=0" >> "$LOG"
