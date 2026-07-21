#!/bin/bash
# HSA vs dense attention across BATCH SIZE at 8K..64K context.
#
# Methodology matches dev/sweep_8k_512k.sh so the numbers are comparable to the
# published batch-1 table:
#   prefill -> head_dim=64 models  (hsa345m_real / dense345m_fair),   eager
#   decode  -> head_dim=128 models (hsa345m_real_hd128 / dense..._hd128), CG ON
#   page_size 64 on both sides, maxpool selection, pool auto-init ON.
#
# bench_one_batch bypasses the scheduler, so `--batch-size B` gives one extend
# forward holding B sequences of `--input-len L` and then B-wide decode steps.
# (Before batch_size>1 support the extend path asserted B==1, so this sweep
# could not be run at all.)
#
# Usage: dev/sweep_batch_vs_dense.sh <LENGTH> <GPU> [OUT]
set -u
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
unset SGLANG_HSA_DISABLE_AUTO_POOL_INIT

L=${1:?length}; GPU=${2:?gpu}; OUT=${3:-/tmp/sweep_batch_L${L}.txt}
DECODE_BS=${DECODE_BS:-"1 4 16 32"}
PREFILL_BS=${PREFILL_BS:-"1 4"}
MEM=${MEM:-0.6}
: > "$OUT"

ctx=$((L + 256))

prefill_run() { # model backend bs
  local m=$1 b=$2 bs=$3 r v err
  r=$(CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python -m sglang.bench_one_batch \
      --model-path "$m" --load-format dummy --tp 1 --batch-size "$bs" \
      --input-len "$L" --output-len 4 --max-running-requests "$bs" \
      --context-length "$ctx" --attention-backend "$b" --page-size 64 \
      --disable-cuda-graph --mem-fraction-static "$MEM" --trust-remote-code 2>&1)
  v=$(echo "$r" | grep -E "Prefill\. latency" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  err=$(echo "$r" | grep -ciE "out of memory|OutOfMemory|refusing to silently|CUDA error|AssertionError")
  echo "PREFILL $b L=$L bs=$bs -> ${v:-FAIL} (err=$err)" | tee -a "$OUT"
}

decode_run() { # model backend bs
  local m=$1 b=$2 bs=$3 r v err
  r=$(CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python -m sglang.bench_one_batch \
      --model-path "$m" --load-format dummy --tp 1 --batch-size "$bs" \
      --input-len "$L" --output-len 32 --max-running-requests "$bs" \
      --context-length "$ctx" --attention-backend "$b" --page-size 64 \
      --cuda-graph-max-bs "$bs" --mem-fraction-static "$MEM" --trust-remote-code 2>&1)
  v=$(echo "$r" | grep -E "Decode\. *median" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  err=$(echo "$r" | grep -ciE "out of memory|OutOfMemory|refusing to silently|CUDA error|AssertionError")
  echo "DECODE  $b L=$L bs=$bs -> ${v:-FAIL} (err=$err)" | tee -a "$OUT"
}

for bs in $PREFILL_BS; do
  prefill_run dev/bench_models/hsa345m_real        hsa    "$bs"
  prefill_run dev/bench_models/dense345m_fair      triton "$bs"
done
for bs in $DECODE_BS; do
  decode_run dev/bench_models/hsa345m_real_hd128   hsa    "$bs"
  decode_run dev/bench_models/dense345m_fair_hd128 triton "$bs"
done
