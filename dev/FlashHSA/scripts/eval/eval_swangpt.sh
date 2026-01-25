export PYTHONPATH=./

python eval/eval_ppl.py \
    --config_path configs/swan_gpt_tiny/config_std.json \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/swan-gpt-std/checkpoints/global_step_30000 \
    --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len 16384 \
    --max_samples 1000
