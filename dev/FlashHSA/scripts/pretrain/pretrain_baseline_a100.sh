export PYTHONPATH=./

# 50000 * 4K * 8 * 16 = 25.6B tokens
bash train.sh tasks/pretrain.py configs/baselines/full_attn_tiny.yaml \
    --model.config_path configs/full_attn_tiny/config.json \
    --data.train_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_long_tokenized/ \
    --data.max_seq_len 4096 \
    --data.train_size 10000000000 \
    --data.data_type numpy \
    --data.datasets_type olmo3 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.wandb_name full_attn_baseline \
    --train.rmpad false \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size 4 \
    --train.global_batch_size 128 \
    --train.lr 7e-4 \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 10000 \
    --train.max_steps 50000 \
    --train.output_dir /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/full-attn-tiny-a100-baseline

# /apdcephfs_fsgm/share_303843174/shared/data/dolma3_long_tokenized
