#!/bin/bash
# Re-run ONLY the sglang side of the friend's verify trio and compare against a
# cached HF worker pickle.  The HF worker does prefill-rollout (~minutes) and its
# result depends only on the checkpoint + input ids, so it never needs re-running
# while iterating on the sglang backend.
#
# Prime the cache once with:
#   bash dev/run_cg_vs_hf_verify.sh <CKPT> on <GPU> <PT> <MAXNEW> 0
#   cp /tmp/verify_hf_cgon.pkl dev/_verify_run/b1_baseline/hf_pt<PT>.pkl
#
# Usage: dev/reverify_align.sh <PT> <GPU> [MAXNEW] [CKPT] [on|off]
set -u
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
PT=${1:-512}; GPU=${2:-0}; MAXNEW=${3:-12}
CKPT=${4:-dev/align/ckpt_345m_bench_hf}; CG=${5:-on}
CE=dev/InfiniteLongLM/code_exp
HF_PKL=dev/_verify_run/b1_baseline/hf_pt${PT}.pkl
SG_PKL=/tmp/reverify_sg_pt${PT}.pkl
CFG=/tmp/reverify_cfg_pt${PT}.json
DIS=$([ "$CG" = "on" ] && echo false || echo true)

if [ ! -f "$HF_PKL" ]; then
  echo "missing cached HF pickle: $HF_PKL" >&2; exit 2
fi

python - "$CKPT" "$HF_PKL" "$SG_PKL" "$DIS" "$PT" "$MAXNEW" "$CFG" <<'PY'
import json, sys
ck, hf, sg, dis, pt, mx, cfgpath = sys.argv[1:8]
ids = json.load(open("/tmp/verify_ids.json"))["input_ids"]
json.dump({
  "checkpoint_path": ck, "vocab_dir": ck, "device": "cuda:0",
  "prompt": "", "input_ids": ids,
  "prompt_tokens": int(pt), "max_new_tokens": int(mx), "top_k": 10,
  "hf_output_path": hf, "sglang_output_path": sg,
  "disable_cuda_graph": (dis == "true"), "num_prefill_requests": 0,
  "sglang_page_size": 64, "sglang_max_total_tokens": 16384,
  "sglang_chunked_prefill_size": 8192,
}, open(cfgpath, "w"))
PY

echo "=========== SGLang worker (PT=$PT, CG=$CG) ==========="
CUDA_VISIBLE_DEVICES=$GPU CFG=$CFG python $CE/verify_sglang_worker.py $CFG 2>&1 \
  | grep -E "SGLang|Decode|Error|RuntimeError|AssertionError|Traceback" | tail -20
echo "=========== COMPARE (PT=$PT) ==========="
CUDA_VISIBLE_DEVICES="" CFG=$CFG python $CE/verify_sglang_vs_hf.py $CFG 2>&1 \
  | grep -E "匹配率|总一致率|KL|完全一致|大部分|差异" | tail -10
