#!/bin/bash
# Quick HSA-only prefill bench at 3 lengths to confirm R90 selector speedup
# Logs to /tmp/sweep_R90_quick.log
set -u
LENGTHS=("${@:-16384 32768 65536 131072}")
DECODE_LEN=4

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
LOG=/tmp/sweep_R90_quick.log
> "$LOG"

for L in ${LENGTHS[@]}; do
    CTX=$((L + 200))
    echo "==== HSA  input=${L}  ctx=${CTX} ====" >> "$LOG"
    timeout 600 /home/hal-alex/miniconda3/envs/alexsg/bin/python -m sglang.bench_one_batch \
        --model-path /home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real \
        --load-format dummy \
        --tp 1 --batch-size 1 \
        --input-len "$L" --output-len "$DECODE_LEN" \
        --context-length "$CTX" \
        --attention-backend hsa \
        --page-size 64 \
        --disable-cuda-graph \
        --mem-fraction-static 0.80 \
        --trust-remote-code 2>&1 \
      | grep -E "Prefill\.|Decode\.  median|Total\.|Error|Traceback|OOM" \
      | tail -8 >> "$LOG"
    echo "(done)" >> "$LOG"
done
echo "=== sweep done ===" >> "$LOG"
echo "EXIT=0" >> "$LOG"
