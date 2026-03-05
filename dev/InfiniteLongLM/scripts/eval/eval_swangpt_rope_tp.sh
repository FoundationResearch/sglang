#!/bin/bash
export PYTHONPATH=./

# TP配置
TP_SIZE=2  # 可以改为4, 8等
WORLD_SIZE=$TP_SIZE

# 使用torchrun启动多GPU评估
torchrun --nproc_per_node=$TP_SIZE --nnodes=1 --node_rank=0 \
    eval/eval_ppl.py \
    --config_path configs/swan_gpt_tiny/config_rope.json \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/swan-gpt-rope/checkpoints/global_step_30000 \
    --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len 65536 \
    --max_samples 1000 \
    --tp_size $TP_SIZE \
    --parallel_mode fsdp2