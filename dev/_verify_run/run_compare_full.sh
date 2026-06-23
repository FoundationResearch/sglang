#!/bin/bash
set -u
cd /mnt/weka/home/hao.zhang/alex/sglang
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
PY=/mnt/weka/home/hao.zhang/conda/miniconda/envs/alexsg/bin/python
echo "######## G=1 (manifest_345m_g1) ########"
CUDA_VISIBLE_DEVICES=0 $PY dev/align/compare.py --manifest dev/align/manifest_345m_g1.json --no-verify-sha > dev/_verify_run/cmp_g1.log 2>&1
echo "g1 exit=$?"
echo "######## G=8 (manifest_my345m) ########"
CUDA_VISIBLE_DEVICES=1 $PY dev/align/compare.py --manifest dev/align/manifest_my345m.json --no-verify-sha > dev/_verify_run/cmp_g8.log 2>&1
echo "g8 exit=$?"
echo "ALLDONE_COMPARE"
