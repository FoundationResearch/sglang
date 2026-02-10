export PYTHONPATH=./

bash train.sh tasks/pretrain.py configs/baselines/full_attn_tiny.yaml \
    --model.config_path configs/swan_gpt_tiny/config_rope_full.json \
    --data.train_path /apdcephfs_fsgm/share_303843174/shared/data/dolma3_long_tokenized/ \
    --data.max_seq_len 4096 \
    --data.train_size 10000000000 \
    --data.data_type numpy \
    --data.datasets_type olmo3 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.rmpad false \
    --train.wandb_name swan_gpt_tiny_rope_baseline \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size 16 \
    --train.global_batch_size 128 \
    --train.lr 7e-4 \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 10000 \
    --train.max_steps 50000 \
    --train.output_dir /apdcephfs_fsgm/share_303843174/user/shawnxxxhu/checkpoints/swan-gpt-rope-tiny-baseline

# /apdcephfs_fsgm/share_303843174/shared/data/dolma3_long_tokenized
