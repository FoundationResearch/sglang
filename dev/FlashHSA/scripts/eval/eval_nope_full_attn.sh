export PYTHONPATH=./

python eval/eval_ppl.py \
    --config_path ./configs/swan_gpt_tiny/config_nope_full.json \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path ../checkpoints/full-attn-nope-tiny-olmo3pt-8K/checkpoints/global_step_30000 \
    --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len 8192 \
    --max_samples 1000

# export MASTER_PORT=22341

# bash train.sh eval/eval_ppl.py configs/baselines/full_attn_tiny_cos.yaml \
#     --model.config_path configs/swan_gpt_tiny/config_nope_full.json \
#     --data.train_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
#     --data.max_seq_len 8192 \
#     --data.train_size 10000000000 \
#     --data.data_type numpy \
#     --data.datasets_type olmo3 \
#     --train.init_device meta \
#     --train.use_wandb true \
#     --train.enable_gradient_checkpointing true \
#     --train.rmpad false \
#     --train.rmpad_with_pos_ids false \
#     --train.enable_mixed_precision \
#     --train.micro_batch_size 1 \
#     --train.global_batch_size 8 \
#     --train.lr 3e-4 \
#     --train.lr_min 3e-5 \
#     --train.ulysses_parallel_size 1 \
#     --train.save_steps 10000 \
#     --train.max_steps 30000 \
#     --train.load_checkpoint_path auto \
#     --train.output_dir /apdcephfs_fsgm/share_303843174/user/shawnxxxhu/checkpoints/full-attn-nope-tiny-olmo3pt-8K
