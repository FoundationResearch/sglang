export PYTHONPATH=./

python eval/eval_ppl.py \
    --config_path ./configs/flash_hsa/config_swan_nope_sparse.json \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path /apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-dense-correct/checkpoints/global_step_10000 \
    --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len 16384 \
    --insert_lmk \
    --max_samples 1000
