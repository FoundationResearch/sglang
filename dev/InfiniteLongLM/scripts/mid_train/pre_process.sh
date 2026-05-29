mkdir -p /apdcephfs_sh8/share_300719895/shared/data/dolmino_ingredient1_raw
for d in /apdcephfs_sh8/share_300719895/shared/data/dolma3_dolmino_mix-100B-1125/data/ingredient1-*; do                                                                                             
    ln -sf "$d" /apdcephfs_sh8/share_300719895/shared/data/dolmino_ingredient1_raw/$(basename "$d")
done                                                                                                                                                                                                
                                                                
# 2. Tokenize
python preprocess/build_olmo3_datasets.py \
    --vocab_dir /apdcephfs_fsgm/share_303843174/user/guhao/InfiniteLongLM/configs/olmo3_vocab \
    --corpus_dir /apdcephfs_sh8/share_300719895/shared/data/dolmino_ingredient1_raw/ \
    --output_dir /apdcephfs_sh8/share_300719895/shared/data/dolmino-midtrain-100B-tokenized/ \
    --num_workers 128
