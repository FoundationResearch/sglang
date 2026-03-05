
# export PYTHONPATH=./
# SEQ_LEN_LIST=(
#     64
#     128
#     512
#     $((8*1024))
#     $((16*1024))
#     $((64*1024))
#     $((128*1024))
# ) 

# for MAX_SEQ_LEN in "${SEQ_LEN_LIST[@]}"; do

#     echo "Testing with max_seq_len = $MAX_SEQ_LEN"
    
#     python eval/eval_ppl.py \
#         --config_path configs/flash_hsa/config_swan_nope_sparse_innerx_win512_gate.json \
#         --vocab_dir ./configs/olmo3_vocab/ \
#         --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-innerx-top32-win512-ruler-gate/checkpoints/global_step_30000 \
#         --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
#         --max_seq_len $MAX_SEQ_LEN \
#         --insert_lmk \
#         --max_samples 1000 \
#         --last_k_tokens 512
    
#     echo ""
# done





TASK_ID=${1:-0}

echo "Running with TASK_ID=$TASK_ID"

export PYTHONPATH=./
python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_sparse_innerx_win512_gate.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-innerx-top32-win512-ruler-gate/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((8*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \


python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_sparse_innerx_win512_gate.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-innerx-top32-win512-ruler-gate/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((16*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \

python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_sparse_innerx_win512_gate.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-innerx-top32-win512-ruler-gate/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((64*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \

python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_sparse_innerx_win512_gate.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-innerx-top32-win512-ruler-gate/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((128*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \

python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_sparse_innerx_win512_gate.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-innerx-top32-win512-ruler-gate/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((256*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \

python eval/eval_ruler.py \
    --config_path configs/flash_hsa/config_swan_nope_sparse_innerx_win512_gate.json  \
    --vocab_dir configs/olmo3_vocab/ \
    --corpus_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-innerx-top32-win512-ruler-gate/checkpoints/global_step_30000 \
    --task_id $TASK_ID \
    --segment_size -1 \
    --max_seq_len $((512*1024))  \
    --max_samples 100 \
    --print_every 10 \
    --insert_lmk \
    --tp_size 1 \
