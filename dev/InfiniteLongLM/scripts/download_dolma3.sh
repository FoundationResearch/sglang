#!/bin/bash
# 下载 allenai/dolma3_dolmino_mix-100B-1125 数据集

SAVE_DIR=${1:-"/apdcephfs_tj5/share_300719894/shared/data/dolma3_dolmino_mix-100B-1125"}

mkdir -p "$SAVE_DIR"

# 方法1: 使用 huggingface-cli（推荐，支持断点续传）
huggingface-cli download \
    allenai/dolma3_dolmino_mix-100B-1125 \
    --repo-type dataset \
    --local-dir "$SAVE_DIR" \
    --resume-download \
    --token "$HF_TOKEN"

# 如果需要指定 token（私有数据集），取消下面的注释：
# huggingface-cli download \
#     allenai/dolma3_dolmino_mix-100B-1125 \
#     --repo-type dataset \
#     --local-dir "$SAVE_DIR" \
#     --resume-download \
#     --token "hf_YOUR_TOKEN_HERE"
