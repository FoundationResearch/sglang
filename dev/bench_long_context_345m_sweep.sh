#!/bin/bash
# Sweep latency across context-length buckets for 345M models.
# - HSA 345M (qwen_lhsa, 16L, 16 q heads, 2 kv heads, hsa_qk_ratio=8)
# - Dense-Fair 345M (qwen3, SAME arch — 16L, 16 q heads, 2 kv heads, full attention).
#   This is the apples-to-apples comparison the paper uses; the architecture is
#   identical to HSA, only the attention kernel differs.
# - Baseline-24L 345M (qwen3, 24L, 4 kv heads) is kept for reference but the arch
#   differs so the speedup is partly architectural, not from HSA sparsity.

set -u
LENGTHS=(8192 16384 32768 65536 131072 262144 524288)
DECODE_LEN=128

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
        --mem-fraction-static 0.85 \
        --trust-remote-code 2>&1 \
      | grep -E "Prefill\.|Decode\.  median|Total\.|Error|Traceback|OOM|out of memory|signal|exceeds|too large|skipping" \
      | tail -10
    echo "(done)"
}

for L in "${LENGTHS[@]}"; do
    CTX=$((L + 200))
    echo "===================================================="
    echo "CONTEXT BUCKET: $L"
    echo "===================================================="
    run_one "/home/hal-alex/workspace/hsa345m_real"        "$L" "HSA-345M"        "$CTX"
    run_one "/home/hal-alex/workspace/dense345m_fair"      "$L" "Dense-Fair-345M" "$CTX"
done
echo "=== Sweep done ==="
