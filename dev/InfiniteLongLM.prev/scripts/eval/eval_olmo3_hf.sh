export PYTHONPATH=./
export HF_ENABLE_PARALLEL_LOADING=true
export HF_PARALLEL_LOADING_WORKERS=3
SEQ_LEN_LIST=(
    # 64
    # 128
    # 512
    # 2048
    # 4096
    $((8*1024))
) 

# ckpt_path=/apdcephfs_sh8/share_300719895/shared/models/OLMO3/OLMo-stage1-step999000
# Test Length: 8192, Final Mean Loss: 1.3762, PPL: 3.9598
# ckpt_path=/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/OLMo-stage1-step999000
# ckpt_path=/apdcephfs_fsgm/share_303843174/user/guhao/Models/OLMo-stage1-step999000
ckpt_path=/apdcephfs_sh8/share_300719895/guhao/checkpoints/OLMo-stage1-step999000-modified/
# Test Length: 8192, Final Mean Loss: 1.4614, PPL: 4.3120





idx=6
for MAX_SEQ_LEN in "${SEQ_LEN_LIST[@]}"; do
    GPU_ID=$idx   # 0~6，对应 7 张卡；如果以后长度>8，也可以改成 idx%8

    pkill -f "burner.*--gpu $GPU_ID"

    echo "GPU $GPU_ID: Testing with max_seq_len = $MAX_SEQ_LEN"

    # CUDA_VISIBLE_DEVICES=$GPU_ID python eval/eval_ppl.py \
    #     --use_hf_model \
    #     --vocab_dir $ckpt_path \
    #     --checkpoint_path $ckpt_path \
    #     --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    #     --max_seq_len $MAX_SEQ_LEN \
    #     --max_samples 1000 \
    #     --last_k_tokens 512 &

    CUDA_VISIBLE_DEVICES=$GPU_ID python eval/eval_ppl_hf.py \
        --vocab_dir $ckpt_path \
        --checkpoint_path $ckpt_path \
        --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
        --max_seq_len $MAX_SEQ_LEN \
        --max_samples 1000 \
        --last_k_tokens 512 &

    idx=$((idx + 1))
done
wait



# # 从命令行参数获取 task_id，默认为 0
# TASK_ID=${1:-0}

# echo "Running with TASK_ID=$TASK_ID"

# # eval ruler for full attn rope theta10000 ruler-pretrain
# python eval/eval_ruler.py \
#     --config_path $model_config  \
#     --vocab_dir configs/olmo3_vocab/ \
#     --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
#     --checkpoint_path $ckpt_path \
#     --task_id $TASK_ID \
#     --segment_size -1 \
#     --max_seq_len $((8*1024))  \
#     --max_samples 100 \
#     --print_every 1 \
#     --tp_size 1 \


# python eval/eval_ruler.py \
#     --config_path configs/swan_gpt_tiny/config_rope_full_theta10000.json  \
#     --vocab_dir configs/olmo3_vocab/ \
#     --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
#     --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/full-attn-rope-theta10000-ruler-5per/checkpoints/global_step_10000 \
#     --task_id $TASK_ID \
#     --segment_size -1 \
#     --max_seq_len $((16*1024))  \
#     --max_samples 100 \
#     --print_every 10 \
#     --tp_size 1 \

# python eval/eval_ruler.py \
#     --config_path configs/swan_gpt_tiny/config_rope_full_theta10000.json  \
#     --vocab_dir configs/olmo3_vocab/ \
#     --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
#     --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/full-attn-rope-theta10000-ruler-5per/checkpoints/global_step_10000 \
#     --task_id $TASK_ID \
#     --segment_size -1 \
#     --max_seq_len $((64*1024))  \
#     --max_samples 100 \
#     --print_every 10 \
#     --tp_size 1 \


# python eval/eval_ppl.py \
#     --config_path ./configs/swan_gpt_tiny/config_rope_full_theta10000.json \
#     --vocab_dir ./configs/olmo3_vocab/ \
#     --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/full-attn-rope-theta10000-ruler-5per/checkpoints/global_step_10000 \
#     --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
#     --max_seq_len $((4*1024)) \
#     --max_samples 20
