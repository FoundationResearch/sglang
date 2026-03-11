head lhsa(swa_len, need_retrieval=False, pos_emb=rope, chunk_align_=False):
    exp_sum = Sliding Window Attention(swa_len, pos_emb, chunk_align)
    # chunk align = true means 如果swa_len=16 那么先计算看到多少chunk 然后根据chunk数计算真正的swa_len是多少
    # e.g. current_len = 70, swa_len=16, chunk_size=64. effective_chunk = ceil(70/64)=2, effctive_swa_len=70
    Hierachical Sparse Attention(effective_chunk(swa的chunk就不看了), exp_sum 要权重融合)

every layer:
a lhsa(2048, False, rope, Flase) + b lhsa(2048, True, nope, True)

考虑多轮prefilling 观察是否支持lhsa的prefilling

todo: impl, prefill, bench, 