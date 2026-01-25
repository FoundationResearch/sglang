export PYTHONPATH=./

bash train_dist.sh tasks/pretrain.py configs/baselines/full_attn_tiny.yaml \
    --model.config_path configs/swan_gpt_tiny/config_nope_all.json \
    --data.train_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --data.max_seq_len 8192 \
    --data.train_size 10000000000 \
    --data.data_type numpy \
    --data.datasets_type olmo3 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.rmpad false \
    --train.enable_gradient_checkpointing true \
    --train.wandb_name full-attn-nope-tiny_olmo3pt_8K \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size 8 \
    --train.global_batch_size 128 \
    --train.lr 7e-4 \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 10000 \
    --train.max_steps 50000 \
    --train.output_dir /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/full-attn-nope-tiny-olmo3pt-8K

# /apdcephfs_fsgm/share_303843174/shared/data/dolma3_long_tokenized
