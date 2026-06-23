#!/bin/bash
# Drive the friend's HF-vs-sglang verify suite with raw token ids.
#   HF worker: eager attn (no flash_attn3), modeling_qwen_lhsa.
#   sglang worker: real sgl.Engine, CUDA graph ON/OFF per CG arg, skip_tokenizer_init.
# Compares HF-official greedy decode vs sglang greedy decode (token/top2/KL).
#
# Usage: dev/run_cg_vs_hf_verify.sh <CKPT_DIR> <on|off> <GPU> [PROMPT_TOKENS] [MAX_NEW] [NUM_PREFILL]
set -u
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
CKPT=$1; CG=$2; GPU=${3:-0}; PT=${4:-2048}; MAXNEW=${5:-32}; NPRE=${6:-0}
CE=dev/InfiniteLongLM/code_exp
TAG="cg${CG}"
HF_PKL=/tmp/verify_hf_${TAG}.pkl
SG_PKL=/tmp/verify_sg_${TAG}.pkl
CFG=/tmp/verify_cfg_${TAG}.json
DIS=$([ "$CG" = "on" ] && echo false || echo true)

python - "$CKPT" "$HF_PKL" "$SG_PKL" "$DIS" "$PT" "$MAXNEW" "$NPRE" "$CFG" <<'PY'
import json, sys
ck,hf,sg,dis,pt,mx,npre,cfgpath = sys.argv[1:9]
ids = json.load(open("/tmp/verify_ids.json"))["input_ids"]
cfg = {
  "checkpoint_path": ck, "vocab_dir": ck, "device": "cuda:0",
  "prompt": "", "input_ids": ids,
  "prompt_tokens": int(pt), "max_new_tokens": int(mx), "top_k": 10,
  "hf_output_path": hf, "sglang_output_path": sg,
  "disable_cuda_graph": (dis=="true"),
  "num_prefill_requests": int(npre),
  "sglang_page_size": 64, "sglang_max_total_tokens": 16384,
  "sglang_chunked_prefill_size": 8192,
}
json.dump(cfg, open(cfgpath,"w"))
print("wrote cfg", cfgpath, "cg_disabled=", dis, "prompt_tokens=", pt)
PY

echo "================ HF worker (eager) ================"
CUDA_VISIBLE_DEVICES=$GPU CFG=$CFG python $CE/verify_hf_worker_eager.py $CFG 2>&1 | grep -vE "^\s*$" | tail -25
echo "================ SGLang worker (CG=$CG) ================"
CUDA_VISIBLE_DEVICES=$GPU CFG=$CFG python $CE/verify_sglang_worker.py $CFG 2>&1 | grep -E "SGLang|Decode|Token|Error|Traceback|disable_cuda|Using raw|Warmup|dummy" | tail -30
echo "================ COMPARE ================"
CUDA_VISIBLE_DEVICES="" CFG=$CFG python $CE/verify_sglang_vs_hf.py $CFG 2>&1 | grep -E "匹配率|总一致率|KL|Logprob|完全一致|大部分|差异|Vocab|decode tokens" | tail -30
