#!/bin/bash
# Compact A/B latency check for the batch_size>1 work: HSA prefill (eager) and
# HSA decode (CUDA graph) at two context lengths, batch 1 — i.e. exactly the
# path the published numbers were measured on, to confirm the per-sequence
# refactor did not cost anything at B=1.
#
# Usage: GPU=4 dev/bench_batch_refactor_ab.sh <label>
set -u
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
unset SGLANG_HSA_DISABLE_AUTO_POOL_INIT
GPU=${GPU:-4}
LABEL=${1:-run}
LENGTHS=${LENGTHS:-"8192 32768"}
PREFILL_MODEL=dev/bench_models/hsa345m_real
DECODE_MODEL=dev/bench_models/hsa345m_real_hd128

for L in $LENGTHS; do
  ctx=$((L + 256))
  r=$(CUDA_VISIBLE_DEVICES=$GPU timeout 1800 python -m sglang.bench_one_batch \
      --model-path "$PREFILL_MODEL" --load-format dummy --tp 1 --batch-size 1 \
      --input-len "$L" --output-len 4 --max-running-requests 1 --context-length "$ctx" \
      --attention-backend hsa --page-size 64 --disable-cuda-graph \
      --mem-fraction-static 0.5 --trust-remote-code 2>&1)
  v=$(echo "$r" | grep -E "Prefill\. latency" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  echo "[$LABEL] PREFILL L=$L -> ${v:-FAIL}"

  r=$(CUDA_VISIBLE_DEVICES=$GPU timeout 1800 python -m sglang.bench_one_batch \
      --model-path "$DECODE_MODEL" --load-format dummy --tp 1 --batch-size 1 \
      --input-len "$L" --output-len 32 --max-running-requests 1 --context-length "$ctx" \
      --attention-backend hsa --page-size 64 --cuda-graph-max-bs 1 \
      --mem-fraction-static 0.5 --trust-remote-code 2>&1)
  v=$(echo "$r" | grep -E "Decode\. *median" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  echo "[$LABEL] DECODE  L=$L -> ${v:-FAIL}"
done
