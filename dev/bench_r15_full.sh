#!/bin/bash
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alexsg
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
cd /home/hal-alex/workspace/sglang

run() {
    local model=$1 input_len=$2 label=$3 ctx=$4 backend=$5 mode=$6 dummy=${7:-}
    local cg_args="" extra=""
    if [ "$mode" = "cg" ]; then
        cg_args="--cuda-graph-max-bs 1"
    else
        cg_args="--disable-cuda-graph"
    fi
    if [ "$dummy" = "dummy" ]; then
        extra="--load-format dummy"
    fi
    echo "== $label  L=${input_len}  ${mode} =="
    timeout 1200 python -m sglang.bench_one_batch \
        --model-path "$model" --tp 1 --batch-size 1 \
        --input-len "$input_len" --output-len 32 \
        --context-length "$ctx" \
        --attention-backend "$backend" $cg_args $extra \
        --mem-fraction-static 0.40 --trust-remote-code 2>&1 \
      | grep -E "Decode\.  median|Prefill\. lat|OOM|Error" | tail -4
}

for L in 8192 32768 131072 262144 524288; do
    CTX=$((L + 200))
    echo "==================== L=$L ===================="
    run /home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real    $L "HSA-no-CG"   $CTX hsa    nocg
    run /home/hal-alex/workspace/sglang/dev/bench_models/hsa345m_real    $L "HSA-R15-CG"  $CTX hsa    cg
    run /home/hal-alex/workspace/sglang/dev/bench_models/dense345m_fair  $L "Dense-no-CG" $CTX triton nocg dummy
    run /home/hal-alex/workspace/sglang/dev/bench_models/dense345m_fair  $L "Dense-CG"    $CTX triton cg   dummy
done
