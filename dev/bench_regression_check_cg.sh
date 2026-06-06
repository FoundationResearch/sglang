#!/bin/bash
# Same as bench_regression_check.sh but WITH cuda-graph enabled — matches the
# R35 changelog methodology.  Numbers should land in the R35 ranges (HSA decode
# ~2.8-2.9 ms at 16K, etc.) when default `headwise_topk_softmax=True` is used.

set -u
LENGTHS_DEFAULT="8192 16384 32768"  # CG path doesn't scale to 128K reliably
LENGTHS="${LENGTHS:-$LENGTHS_DEFAULT}"
N_RUNS="${N_RUNS:-5}"
DECODE_LEN="${DECODE_LEN:-32}"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
HSA_MODEL="/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real"
DENSE_MODEL="/home/hal-alex/workspace/sglang/dev/bench_models/dense345m_fair"

bench_one() {
    local L=$1 backend=$2 model=$3
    timeout 600 /home/hal-alex/miniconda3/envs/alexsg/bin/python -m sglang.bench_one_batch \
        --model-path "$model" \
        --load-format dummy \
        --tp 1 --batch-size 1 \
        --input-len "$L" --output-len "$DECODE_LEN" \
        --context-length $((L + 200)) \
        --attention-backend "$backend" \
        --cuda-graph-max-bs 1 \
        --mem-fraction-static 0.50 \
        --trust-remote-code 2>&1
}

echo "==== regression sweep (CG enabled) ===="
echo "LENGTHS=${LENGTHS}  N_RUNS=${N_RUNS}  decode_len=${DECODE_LEN}"
echo ""
printf "%-7s %-9s %-6s %-12s %-14s\n" "L" "backend" "run" "prefill_ms" "decode_med_ms"
echo "------------------------------------------------------------"
for L in $LENGTHS; do
    for backend_spec in "hsa:$HSA_MODEL" "triton:$DENSE_MODEL"; do
        backend=${backend_spec%%:*}
        model=${backend_spec#*:}
        for run in $(seq 1 ${N_RUNS}); do
            out=$(bench_one "$L" "$backend" "$model")
            prefill_s=$(echo "$out" | grep -E "Prefill\. latency:" | awk 'NR==2{print}' | \
                sed -n 's/.*latency:\s*\([0-9.]*\)\s*s.*/\1/p')
            decode_s=$(echo "$out" | grep -E "Decode\.\s+median latency:" | awk 'NR==2{print}' | \
                sed -n 's/.*latency:\s*\([0-9.]*\)\s*s.*/\1/p')
            prefill_ms=$(awk -v s="$prefill_s" 'BEGIN{printf "%.2f", s*1000}')
            decode_ms=$(awk -v s="$decode_s" 'BEGIN{printf "%.3f", s*1000}')
            printf "%-7s %-9s %-6s %-12s %-14s\n" "$L" "$backend" "$run" "$prefill_ms" "$decode_ms"
        done
    done
done
echo ""
echo "=== CG regression sweep done ==="
