#!/bin/bash
# Spot-check the new --no-hsa-headwise-topk-softmax mode at L=8K/16K/32K.
# Compare with the default `--hsa-headwise-topk-softmax` baseline.
#
# Expected: maxpool path should be SAME OR FASTER than default — never slower.
# Earlier alignment test (commit e8b96139e validation) already proved output
# bit-identical (max_abs_diff = 0, KL = 0).  This bench just confirms perf.

set -u
LENGTHS="${LENGTHS:-8192 16384 32768}"
N_RUNS="${N_RUNS:-3}"
DECODE_LEN="${DECODE_LEN:-32}"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
HSA_MODEL="/home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real"

bench_one() {
    local L=$1 maxpool_flag=$2
    # --page-size 64 — HSA requires page_size == chunk_size, default 1 degenerates.
    timeout 600 /home/hal-alex/miniconda3/envs/alexsg/bin/python -m sglang.bench_one_batch \
        --model-path "$HSA_MODEL" \
        --load-format dummy \
        --tp 1 --batch-size 1 \
        --input-len "$L" --output-len "$DECODE_LEN" \
        --context-length $((L + 200)) \
        --attention-backend hsa \
        --page-size 64 \
        --cuda-graph-max-bs 1 \
        --mem-fraction-static 0.50 \
        --trust-remote-code \
        $maxpool_flag 2>&1
}

echo "==== maxpool spot-check (CG enabled) ===="
echo "LENGTHS=${LENGTHS}  N_RUNS=${N_RUNS}"
echo ""
printf "%-7s %-10s %-6s %-12s %-14s\n" "L" "mode" "run" "prefill_ms" "decode_med_ms"
echo "------------------------------------------------------------"
for L in $LENGTHS; do
    for mode_spec in "default:" "maxpool:--no-hsa-headwise-topk-softmax"; do
        mode=${mode_spec%%:*}
        flag=${mode_spec#*:}
        for run in $(seq 1 ${N_RUNS}); do
            out=$(bench_one "$L" "$flag")
            prefill_s=$(echo "$out" | grep -E "Prefill\. latency:" | awk 'NR==2{print}' | \
                sed -n 's/.*latency:\s*\([0-9.]*\)\s*s.*/\1/p')
            decode_s=$(echo "$out" | grep -E "Decode\.\s+median latency:" | awk 'NR==2{print}' | \
                sed -n 's/.*latency:\s*\([0-9.]*\)\s*s.*/\1/p')
            prefill_ms=$(awk -v s="$prefill_s" 'BEGIN{printf "%.2f", s*1000}')
            decode_ms=$(awk -v s="$decode_s" 'BEGIN{printf "%.3f", s*1000}')
            printf "%-7s %-10s %-6s %-12s %-14s\n" "$L" "$mode" "$run" "$prefill_ms" "$decode_ms"
        done
    done
done
echo ""
echo "=== spot-check done ==="
