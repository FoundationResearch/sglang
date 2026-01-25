export PYTHONPATH=./

# 50000 * 4K * 8 * 16 = 25.6B tokens
bash train.sh tasks/pretrain.py configs/baselines/full_attn_tiny_cos.yaml \
    --model.config_path configs/swan_gpt_tiny/config_rope_full.json \
    --data.train_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --data.max_seq_len 8192 \
    --data.train_size 10000000000 \
    --data.data_type numpy \
    --data.datasets_type olmo3 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.enable_gradient_checkpointing true \
    --train.wandb_name full-attn-rope-8K \
    --train.rmpad false \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size 16 \
    --train.global_batch_size 128 \
    --train.lr 3e-4 \
    --train.lr_min 3e-5 \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 10000 \
    --train.max_steps 30000 \
    --train.load_checkpoint_path auto \
    --train.output_dir /apdcephfs_fsgm/share_303843174/user/shawnxxxhu/checkpoints/full-attn-rope-tiny-olmo3pt-8K

# /apdcephfs_fsgm/share_303843174/shared/data/dolma3_long_tokenized
#      \
