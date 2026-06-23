#!/bin/bash
# Consistent Efficiency sweep for the paper, post-methodology-fix:
#   prefill  -> head_dim=64 models (hsa345m_real / dense345m_fair), eager (CG irrelevant for prefill)
#   decode   -> head_dim=128 models (hsa345m_real_hd128 / dense345m_fair_hd128), CUDA graph ON
#   selection -> maxpool (headwise_topk_softmax=False, default), pool auto-init ON (DISABLE_AUTO_POOL_INIT unset)
# The hard-error pool-init guard is in place: any silent groupwise downgrade now crashes loudly.
# Goal: 8K..512K, confirm NO OOM, produce one internally-consistent table.
set -u
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
unset SGLANG_HSA_DISABLE_AUTO_POOL_INIT   # pool ON (correct per-q-head path)
GPU=${GPU:-5}
LENGTHS=(8192 16384 32768 65536 131072 262144 524288)
OUT=/tmp/sweep_results.txt
: > "$OUT"

# A single bench request needs only ~L KV-cache tokens, but mem-fraction 0.85
# reserves a multi-million-token KV cache (e.g. 7.4M at 512K) that starves the
# HSA prefill activations (selection scores grow with L) -> spurious OOM/FAIL.
# Drop the static fraction at long context so prefill has room; the KV cache is
# still vastly larger than one request needs.  Override with MEM_FRAC=... .
mem_frac_for() { # L
  if [ -n "${MEM_FRAC:-}" ]; then echo "$MEM_FRAC"; return; fi
  if [ "$1" -ge 262144 ]; then echo 0.5; else echo 0.85; fi
}

prefill_run() { # model backend len
  local m=$1 b=$2 L=$3 ctx=$(($3+256)) mf=$(mem_frac_for "$3")
  local r
  r=$(CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python -m sglang.bench_one_batch \
      --model-path "$m" --load-format dummy --tp 1 --batch-size 1 \
      --input-len "$L" --output-len 4 --max-running-requests 1 --context-length "$ctx" \
      --attention-backend "$b" --page-size 64 --disable-cuda-graph \
      --mem-fraction-static "$mf" --trust-remote-code 2>&1)
  local v=$(echo "$r" | grep -E "Prefill\. latency" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  local err=$(echo "$r" | grep -ciE "out of memory|OutOfMemory|refusing to silently|RuntimeError|CUDA error")
  echo "PREFILL  $b  L=$L  -> ${v:-FAIL}  (err_lines=$err)" | tee -a "$OUT"
}

decode_run() { # model backend len
  local m=$1 b=$2 L=$3 ctx=$(($3+256)) mf=$(mem_frac_for "$3")
  local r
  r=$(CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python -m sglang.bench_one_batch \
      --model-path "$m" --load-format dummy --tp 1 --batch-size 1 \
      --input-len "$L" --output-len 32 --max-running-requests 1 --context-length "$ctx" \
      --attention-backend "$b" --page-size 64 --cuda-graph-max-bs 1 \
      --mem-fraction-static "$mf" --trust-remote-code 2>&1)
  local v=$(echo "$r" | grep -E "Decode\.  median" | tail -1 | grep -oE "[0-9.]+ s" | head -1)
  local err=$(echo "$r" | grep -ciE "out of memory|OutOfMemory|refusing to silently|RuntimeError|CUDA error")
  echo "DECODE   $b  L=$L  -> ${v:-FAIL}  (err_lines=$err)" | tee -a "$OUT"
}

for L in "${LENGTHS[@]}"; do
  echo "==== L=$L ====" | tee -a "$OUT"
  prefill_run dev/bench_models/hsa345m_real        hsa    "$L"
  prefill_run dev/bench_models/dense345m_fair      triton "$L"
  decode_run  dev/bench_models/hsa345m_real_hd128  hsa    "$L"
  decode_run  dev/bench_models/dense345m_fair_hd128 triton "$L"
done
echo "==== SWEEP DONE ====" | tee -a "$OUT"
