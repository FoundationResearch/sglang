python utils/strip_ckpt_keys.py /apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-7B-8KA2K-w-lmk-q-proj/pytorch_model.bin \
    --pattern '*.layers.3.self_attn.*' \
    --pattern '*.layers.7.self_attn.*' \
    --pattern '*.layers.11.self_attn.*' \
    --pattern '*.layers.15.self_attn.*' \
    --pattern '*.layers.19.self_attn.*' \
    --pattern '*.layers.23.self_attn.*' \
    --pattern '*.layers.27.self_attn.*' \
    --pattern '*.layers.31.self_attn.*' \
    --output /apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-7B-8KA2K-alibi-remove-fullattn/pytorch_model.bin
