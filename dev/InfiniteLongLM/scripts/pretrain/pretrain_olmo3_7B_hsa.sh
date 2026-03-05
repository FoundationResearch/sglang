export PYTHONPATH=./

export MODEL_CONFIG="configs/olmo3_7B/olmo3_hsa.json"
export CORPUS_PATH="/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="olmo3-7B-hsa"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/Olmo3-7B-hsa"

DATA_TYPE=${DTYPE:-ruler_0.05}

bash train.sh tasks/ruler_sft.py configs/baselines/full_attn_tiny.yaml \
    --model.config_path $MODEL_CONFIG \
    --data.train_path $CORPUS_PATH \
    --data.max_seq_len $MAX_SEQ_LEN \
    --data.train_size 10000000000 \
    --data.data_type $DATA_TYPE \
    --data.datasets_type olmo3 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.enable_gradient_checkpointing true \
    --train.rmpad false \
    --train.wandb_project ruler_pretrain_5per \
    --train.wandb_name $WANDB_NAME \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size 8 \
    --train.global_batch_size 64 \
    --train.lr 3e-4 \
    --train.lr_min 3e-5 \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 10000 \
    --train.max_steps 30000 \
    --train.load_checkpoint_path auto \
    --train.output_dir $OUTPUT_DIR