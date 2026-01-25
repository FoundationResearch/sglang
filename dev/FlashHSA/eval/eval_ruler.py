

if __name__ == '__main__':
    input_ids = xxx
    segment_len = 1024 # 单需要考虑model 是否自动insert landmark，如果自动insert，则需要是63的整数倍 或其他更优雅的解决思路
    past_key_values = None
    cache_position = torch.zeros(1)
    for segment in input_ids[::segment_len]:
        out = model(segment, past_key_values=past_key_values, use_cache=True, cache_position=cache_position)
        past_key_values = out.past_key_values
        cache_position += segment.shape[-1]
