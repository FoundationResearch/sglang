export PYTHONPATH=./

# python eval/eval_ppl.py \
#     --config_path ./configs/flash_hsa/config_nope_top32_win512.json \
#     --vocab_dir ./configs/olmo3_vocab/ \
#     --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-spasrse-top32-win512/checkpoints/global_step_30000 \
#     --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
#     --max_seq_len 8192 \
#     --insert_lmk \
#     --max_samples 1000

python eval/eval_ppl.py \
    --config_path ./configs/flash_hsa/config_nope_top32_win512.json \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path /apdcephfs_fsgm/share_303843174/user/shawnxxxhu/checkpoints/hsa-swan-spasrse-top32-win512/checkpoints/global_step_30000 \
    --data_path /apdcephfs_fsgm/share_303843174/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len 8192 \
    --insert_lmk \
    --max_samples 1000

# python eval/eval_ppl.py \
#     --config_path ./configs/flash_hsa/config_nope_top32_win512.json \
#     --vocab_dir ./configs/olmo3_vocab/ \
#     --checkpoint_path /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa_sparse_swan_top32_win512-ruler/checkpoints/global_step_3000 \
#     --data_path /apdcephfs_fsgm/share_303843174/shared/data/dolma3_mix-6T-1025-partial-tokenized \
#     --max_seq_len 8192 \
#     --insert_lmk \
#     --max_samples 1000