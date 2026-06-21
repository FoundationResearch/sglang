#!/bin/bash
# Distributed 8K..512K sweep across 8 GPUs, WITH P3 (runtime length params) committed.
#   prefill -> hd64 models, eager;  decode -> hd128 models, CUDA graph ON.
#   maxpool selection, pool auto-init ON, --max-running-requests 1 (tiny lmk pool, no OOM).
#   HSA uses mem-frac 0.50 (avoids over-sized SWA pool forward-OOM at long ctx);
#   Dense uses 0.85 (no lmk pool, needs room for full KV).
# Compares against the paper Efficiency table to detect any P3 regression.
set -u
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
unset SGLANG_HSA_DISABLE_AUTO_POOL_INIT
RESDIR=/tmp/sweep_p3_res; rm -rf "$RESDIR"; mkdir -p "$RESDIR"
LENGTHS=(8192 16384 32768 65536 131072 262144 524288)

run_job() { # gpu jid L type backend model memfrac
  local gpu=$1 jid=$2 L=$3 typ=$4 b=$5 m=$6 mf=$7 ctx=$(($3+256)) out v
  if [ "$typ" = "prefill" ]; then
    out=$(CUDA_VISIBLE_DEVICES=$gpu timeout 3000 python -m sglang.bench_one_batch \
      --model-path "$m" --load-format dummy --tp 1 --batch-size 1 \
      --input-len "$L" --output-len 4 --max-running-requests 1 --context-length "$ctx" \
      --attention-backend "$b" --page-size 64 --disable-cuda-graph \
      --mem-fraction-static "$mf" --trust-remote-code 2>&1)
    v=$(echo "$out" | grep -E "Prefill\. latency" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  else
    out=$(CUDA_VISIBLE_DEVICES=$gpu timeout 3000 python -m sglang.bench_one_batch \
      --model-path "$m" --load-format dummy --tp 1 --batch-size 1 \
      --input-len "$L" --output-len 32 --max-running-requests 1 --context-length "$ctx" \
      --attention-backend "$b" --page-size 64 --cuda-graph-max-bs 1 \
      --mem-fraction-static "$mf" --trust-remote-code 2>&1)
    v=$(echo "$out" | grep -E "Decode\.  median" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  fi
  local err=$(echo "$out" | grep -ciE "out of memory|refusing to silently|RuntimeError")
  printf "%-7s %-6s L=%-7s -> %s (err=%s)\n" "$typ" "$b" "$L" "${v:-FAIL}" "$err" > "$RESDIR/$(printf '%02d' "$jid").txt"
}

declare -A GJ
jid=0
for L in "${LENGTHS[@]}"; do
  while IFS= read -r spec; do
    [ -z "$spec" ] && continue
    g=$((jid % 8))
    GJ[$g]="${GJ[$g]:-}|$jid $L $spec"
    jid=$((jid+1))
  done <<EOF
prefill hsa dev/bench_models/hsa345m_real 0.50
prefill triton dev/bench_models/dense345m_fair 0.85
decode hsa dev/bench_models/hsa345m_real_hd128 0.50
decode triton dev/bench_models/dense345m_fair_hd128 0.85
EOF
done

for g in $(seq 0 7); do
  (
    IFS='|' read -ra items <<< "${GJ[$g]:-}"
    for it in "${items[@]}"; do
      [ -z "$it" ] && continue
      # shellcheck disable=SC2086
      run_job "$g" $it
    done
  ) &
done
wait
echo "=== ALL DONE ==="; cat "$RESDIR"/*.txt 2>/dev/null
