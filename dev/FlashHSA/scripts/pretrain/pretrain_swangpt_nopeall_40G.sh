export PYTHONPATH=./

bash train.sh tasks/pretrain.py configs/baselines/full_attn_tiny.yaml \
    --model.config_path configs/swan_gpt_tiny/config_nope_all.json \
    --data.train_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_long_tokenized \
    --data.max_seq_len 4096 \
    --data.train_size 10000000000 \
    --data.data_type numpy \
    --data.datasets_type olmo3 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.rmpad false \
    --train.wandb_name swan_gpt_tiny_nope_all_baseline \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size 4 \
    --train.global_batch_size 128 \
    --train.lr 7e-4 \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 10000 \
    --train.max_steps 50000 \
    --train.output_dir /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/swan-gpt-nope-all-tiny-baseline

# /apdcephfs_fsgm/share_303843174/shared/data/dolma3_long_tokenized
