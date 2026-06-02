export PYTHONPATH=./

bash train.sh tasks/ruler_sft.py configs/baselines/full_attn_tiny_cos.yaml \
    --model.config_path configs/flash_hsa/config_nope_dense.json \
    --data.train_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --data.max_seq_len 8064 \
    --data.train_size 10000000000 \
    --data.data_type numpy \
    --data.datasets_type olmo3 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.enable_gradient_checkpointing true \
    --train.wandb_name hsa_nope_dense_win512_ruler \
    --train.rmpad false \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size 16 \
    --train.global_batch_size 128 \
    --train.lr 3e-4 \
    --train.lr_min 3e-5 \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 1000 \
    --train.max_steps 3000 \
    --train.output_dir /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa_nope_dense_win512_ruler \
    --model.model_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-nope-dense-fixlabel/checkpoints/global_step_30000/hf_ckpt \
    --train.wandb_project ruler-sft \
