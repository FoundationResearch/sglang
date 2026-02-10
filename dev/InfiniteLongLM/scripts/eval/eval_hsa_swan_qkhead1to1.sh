# export PYTHONPATH=./



# 从命令行参数获取 task_id，默认为 0
TASK_ID=${1:-0}

echo "Running with TASK_ID=$TASK_ID"

export PYTHONPATH=./
# python eval/eval_ruler.py \
#     --config_path configs/flash_hsa/config_swan_lmk_win512_eval.json  \
#     --vocab_dir configs/olmo3_vocab/ \
#     --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
#     --checkpoint_path /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/swan_lmk_win512_step20000-ruler/checkpoints/global_step_1000 \
#     --task_id $TASK_ID \
#     --segment_size -1 \
#     --max_seq_len $((8*1024))  \
#     --max_samples 100 \
#     --print_every 10 \
#     --insert_lmk \
#     --tp_size 1 \
    
# python eval/eval_ruler.py \
#     --config_path configs/flash_hsa/config_swan_lmk_win512_eval.json  \
#     --vocab_dir configs/olmo3_vocab/ \
#     --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
#     --checkpoint_path /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/swan_lmk_win512_step20000-ruler/checkpoints/global_step_1000 \
#     --task_id $TASK_ID \
#     --segment_size -1 \
#     --max_seq_len $((16*1024))  \
#     --max_samples 100 \
#     --print_every 10 \
#     --insert_lmk \
#     --tp_size 1 \
    
# python eval/eval_ruler.py \
#     --config_path configs/flash_hsa/config_swan_lmk_win512_eval.json  \
#     --vocab_dir configs/olmo3_vocab/ \
#     --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
#     --checkpoint_path /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/swan_lmk_win512_step20000-ruler/checkpoints/global_step_1000 \
#     --task_id $TASK_ID \
#     --segment_size 1024 \
#     --max_seq_len $((64*1024))  \
#     --max_samples 100 \
#     --print_every 10 \
#     --insert_lmk \
#     --tp_size 1 \
    
python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_lmk_win512_eval.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/swan_lmk_win512_step20000-ruler/checkpoints/global_step_1000 \
    --task_id $TASK_ID \
    --segment_size 1024 \
    --max_seq_len $((128*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \
    
# python eval/eval_ruler.py \
#     --config_path configs/flash_hsa/config_swan_lmk_win512_eval.json  \
#     --vocab_dir configs/olmo3_vocab/ \
#     --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
#     --checkpoint_path /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/swan_lmk_win512_step20000-ruler/checkpoints/global_step_1000 \
#     --task_id $TASK_ID \
#     --segment_size -1 \
#     --max_seq_len $((256*1024))  \
#     --max_samples 100 \
#     --print_every 10 \
#     --insert_lmk \
#     --tp_size 1 \
    
# python eval/eval_ruler.py \
#     --config_path configs/flash_hsa/config_swan_lmk_win512_eval.json  \
#     --vocab_dir configs/olmo3_vocab/ \
#     --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
#     --checkpoint_path /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/swan_lmk_win512_step20000-ruler/checkpoints/global_step_1000 \
#     --task_id $TASK_ID \
#     --segment_size -1 \
#     --max_seq_len $((512*1024))  \
#     --max_samples 100 \
#     --print_every 10 \
#     --insert_lmk \
#     --tp_size 1 \
