# export PYTHONPATH=./

# python eval/eval_ppl.py \
#     --config_path ./configs/flash_hsa/config_nope_dense.json \
#     --vocab_dir ./configs/olmo3_vocab/ \
#     --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-nope-dense/checkpoints/global_step_30000 \
#     --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
#     --max_seq_len 65536 \
#     --insert_lmk \
#     --max_samples 100




# 从命令行参数获取 task_id，默认为 0
TASK_ID=${1:-0}

echo "Running with TASK_ID=$TASK_ID"

# eval ruler for hsa swan sparse top32 win512
export PYTHONPATH=./
python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_dense_win512_eval.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-dense-win512/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((8*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \
    


python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_dense_win512_eval.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-dense-win512/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((16*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \
    

python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_dense_win512_eval.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-dense-win512/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((64*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \


python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_dense_win512_eval.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-dense-win512/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((128*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \
