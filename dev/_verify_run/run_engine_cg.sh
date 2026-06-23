#!/bin/bash
set -u
cd /mnt/weka/home/hao.zhang/alex/sglang
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
PY=/mnt/weka/home/hao.zhang/conda/miniconda/envs/alexsg/bin/python
RD=dev/_verify_run
$PY - <<'PY'
import numpy as np, json
m=np.memmap("dev/align/wikitext103_tokenized/train_010.data", dtype=np.uint32, mode="r")
ids=[int(x) for x in m[1000:3500]]; assert max(ids)<100278
json.dump({"input_ids":ids}, open("dev/_verify_run/ids.json","w"))
for mode,out in (("off","dev/_verify_run/sg_off.pkl"),("on","dev/_verify_run/sg_on.pkl")):
    cfg={"checkpoint_path":"dev/align/ckpt_345m_bench_hf","vocab_dir":"dev/align/ckpt_345m_bench_hf",
     "device":"cuda:0","prompt":"","input_ids":ids,"prompt_tokens":2048,"max_new_tokens":24,"top_k":10,
     "hf_output_path":"dev/_verify_run/hf_x.pkl","sglang_output_path":out,
     "disable_cuda_graph":(mode=="off"),"num_prefill_requests":0,
     "sglang_page_size":64,"sglang_max_total_tokens":16384,"sglang_chunked_prefill_size":8192,
     "max_running_requests":1,"log_level":"error"}
    json.dump(cfg,open(f"dev/_verify_run/cfg_{mode}.json","w"))
print("prepared")
PY
echo "=== Engine CG-OFF ==="; CUDA_VISIBLE_DEVICES=0 $PY dev/InfiniteLongLM/code_exp/verify_sglang_worker.py $RD/cfg_off.json > $RD/off.log 2>&1; echo "off exit=$? $(grep 'Token IDs' $RD/off.log|tail -1)"
echo "=== Engine CG-ON ===";  CUDA_VISIBLE_DEVICES=0 $PY dev/InfiniteLongLM/code_exp/verify_sglang_worker.py $RD/cfg_on.json  > $RD/on.log  2>&1; echo "on exit=$? $(grep 'Token IDs' $RD/on.log|tail -1)"
echo "=== DIFF CG-on vs CG-off ==="; $PY dev/diff_sglang_cg.py $RD/sg_off.pkl $RD/sg_on.pkl 2>&1 | tail -12
echo "ALLDONE_ENGINE"
