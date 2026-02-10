export PYTHONPATH=./

python eval/eval_ppl.py \
    --config_path configs/swan_gpt_tiny/config_rope.json \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path /apdcephfs_fsgm/share_303843174/user/shawnxxxhu/checkpoints/swan-gpt-rope/checkpoints/global_step_30000 \
    --data_path /apdcephfs_fsgm/share_303843174/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len 65536 \
    --max_samples 1000
