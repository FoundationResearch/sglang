export PYTHONPATH=./

# python eval/eval_ppl.py \
#     --config_path configs/swan_gpt_tiny/config_std.json \
#     --vocab_dir ./configs/olmo3_vocab/ \
#     --checkpoint_path /apdcephfs_fsgm/share_303843174/user/shawnxxxhu/checkpoints/swan-gpt-std/checkpoints/global_step_30000 \
#     --data_path /apdcephfs_fsgm/share_303843174/shared/data/dolma3_mix-6T-1025-partial-tokenized \
#     --max_seq_len 65536 \
#     --max_samples 100





TASK_ID=${1:-0}

echo "Running with TASK_ID=$TASK_ID"

# eval ruler for swan gpt std
python eval/eval_ruler.py \
    --config_path configs/swan_gpt_tiny/config_std_scaling.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/swan-gpt-std-ruler/checkpoints/global_step_3000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((8*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --tp_size 1 \
    