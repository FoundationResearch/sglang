#!/bin/bash
# CLEAN serial sweep on ONE GPU (no cross-GPU power contention) — WITH P3.
#   prefill -> hd64 eager;  decode -> hd128 CUDA-graph.  maxpool, pool ON, req=1.
#   HSA mem-frac 0.50 (no over-sized SWA pool OOM); Dense 0.85 (room for full KV).
# For a clean P3-vs-paper regression check across 8K..512K.
set -u
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
unset SGLANG_HSA_DISABLE_AUTO_POOL_INIT
GPU=${GPU:-0}
OUT=/tmp/sweep_serial.txt; : > "$OUT"
LENGTHS=(8192 16384 32768 65536 131072 262144 524288)

pf() { local m=$1 b=$2 L=$3 mf=$4 ctx=$(($3+256)) r v
  r=$(CUDA_VISIBLE_DEVICES=$GPU timeout 3000 python -m sglang.bench_one_batch \
     --model-path "$m" --load-format dummy --tp 1 --batch-size 1 \
     --input-len "$L" --output-len 4 --max-running-requests 1 --context-length "$ctx" \
     --attention-backend "$b" --page-size 64 --disable-cuda-graph \
     --mem-fraction-static "$mf" --trust-remote-code 2>&1)
  v=$(echo "$r" | grep -E "Prefill\. latency" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  printf "PREFILL %-6s L=%-7s -> %s\n" "$b" "$L" "${v:-FAIL}" | tee -a "$OUT"; }

dc() { local m=$1 b=$2 L=$3 mf=$4 ctx=$(($3+256)) r v
  r=$(CUDA_VISIBLE_DEVICES=$GPU timeout 3000 python -m sglang.bench_one_batch \
     --model-path "$m" --load-format dummy --tp 1 --batch-size 1 \
     --input-len "$L" --output-len 32 --max-running-requests 1 --context-length "$ctx" \
     --attention-backend "$b" --page-size 64 --cuda-graph-max-bs 1 \
     --mem-fraction-static "$mf" --trust-remote-code 2>&1)
  v=$(echo "$r" | grep -E "Decode\.  median" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  printf "DECODE  %-6s L=%-7s -> %s\n" "$b" "$L" "${v:-FAIL}" | tee -a "$OUT"; }

for L in "${LENGTHS[@]}"; do
  echo "==== L=$L ====" | tee -a "$OUT"
  pf dev/bench_models/hsa345m_real         hsa    "$L" 0.50
  pf dev/bench_models/dense345m_fair       triton "$L" 0.85
  dc dev/bench_models/hsa345m_real_hd128   hsa    "$L" 0.50
  dc dev/bench_models/dense345m_fair_hd128 triton "$L" 0.85
done
echo "==== SERIAL SWEEP DONE ====" | tee -a "$OUT"
