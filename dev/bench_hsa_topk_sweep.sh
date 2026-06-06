#!/bin/bash
# Sweep --hsa-topk in {8,16,32,64,128} at multiple L.
# Confirms HSA selector is actually doing work: if decode time stays flat
# across topk values, selector is not really being exercised.
#
# Each (L, topk) combination runs in a FRESH process N_RUNS times.
# We take the WARM "Decode. median latency" (the SECOND one per bench, after JIT).

set -u
LENGTHS_DEFAULT="16384 32768 65536"
LENGTHS="${LENGTHS:-$LENGTHS_DEFAULT}"
N_RUNS="${N_RUNS:-3}"
DECODE_LEN="${DECODE_LEN:-16}"
TOPKS=(8 16 32 64 128)

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
MODEL_PATH="/home/hal-alex/workspace/hsa345m_real"

echo "==== HSA topk sensitivity sweep ===="
echo "LENGTHS=${LENGTHS}  N_RUNS=${N_RUNS}  decode_len=${DECODE_LEN}"
echo ""
printf "%-8s %-6s %-8s %-12s %-12s %-12s\n" "L" "topk" "run" "warm_ms" "tput_tok/s" "raw_line"
echo "---------------------------------------------------------------------------"

for L in $LENGTHS; do
    CTX=$((L + 200))
    for topk in "${TOPKS[@]}"; do
        for run in $(seq 1 ${N_RUNS}); do
            # Take only the SECOND "Decode.  median" line (warm), tail -1 of 2 == the second.
            line=$(timeout 600 /home/hal-alex/miniconda3/envs/alexsg/bin/python -m sglang.bench_one_batch \
                --model-path "$MODEL_PATH" \
                --load-format dummy \
                --tp 1 --batch-size 1 \
                --input-len "$L" --output-len "$DECODE_LEN" \
                --context-length "$CTX" \
                --attention-backend hsa \
                --hsa-topk "$topk" \
                --disable-cuda-graph \
                --mem-fraction-static 0.50 \
                --cuda-graph-max-bs 1 \
                --trust-remote-code 2>&1 \
              | grep -E "Decode\.\s+median" | tail -1)
            ms=$(echo "$line" | sed -n 's/.*latency:\s*\([0-9.]*\)\s*s.*/\1/p' | awk '{printf "%.2f", $1*1000}')
            tput=$(echo "$line" | sed -n 's/.*throughput:\s*\([0-9.]*\)\s*token.*/\1/p')
            printf "%-8s %-6s %-8s %-12s %-12s %-s\n" "$L" "$topk" "$run" "$ms" "$tput" "$line"
        done
    done
done
echo ""
echo "=== sweep done ==="
