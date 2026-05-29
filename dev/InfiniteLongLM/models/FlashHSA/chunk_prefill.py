"""
自动 Chunk Prefill 模块

将长序列分 chunk 逐段通过所有 decoder layers，利用 KV Cache 保持因果性，
从而将 MLP 中间激活的峰值显存从 O(seq_len) 降低到 O(chunk_size)。

用法：
    在 HSAModel.forward 中，当推理模式下序列长度超过阈值时，
    调用 chunked_forward 替代原始的逐层循环。
"""

import torch
from typing import Optional, Tuple
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from utils.hsa_cache_utils import HSADynamicLayer

# 硬编码的 chunk 大小（token 数）
CHUNK_PREFILL_SIZE = 1024

# 默认的自动开启阈值（token 数），序列长度超过此值时自动启用 chunk prefill
DEFAULT_CHUNK_PREFILL_THRESHOLD = 128 * 1024  # 128K

def chunked_forward(
    model,  # HSAModel 实例
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs,
) -> BaseModelOutputWithPast:
    """
    将长序列分 chunk 逐段通过所有 decoder layers 的 forward。

    核心思路：
    1. 先做 embedding 和 position_embeddings（全局计算，显存开销小）
    2. 将 hidden_states 按 CHUNK_PREFILL_SIZE 切分
    3. 每个 chunk 依次通过所有 decoder layers，KV Cache 自动累积
    4. 拼接所有 chunk 的输出，过 final norm

    Args:
        model: HSAModel 实例
        其余参数与 HSAModel.forward 一致
    """
    config = model.config

    # chunk prefill 内部必须启用 KV Cache，否则后续 chunk 无法看到前面 chunk 的 KV
    # 但返回时根据调用者的原始意图决定是否返回 past_key_values
    caller_use_cache = use_cache if use_cache is not None else config.use_cache
    use_cache = True  # 内部强制启用

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # 1. Embedding
    if inputs_embeds is None:
        inputs_embeds = model.embed_tokens(input_ids)

    seq_len = inputs_embeds.shape[1]

    # 2. 初始化 KV Cache
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if use_cache and isinstance(past_key_values, DynamicCache):
        required_layers = 2 * config.num_hidden_layers
        while len(past_key_values.layers) < required_layers:
            idx = len(past_key_values.layers)
            if idx >= config.num_hidden_layers:
                past_key_values.layers.append(HSADynamicLayer())
            else:
                past_key_values.layers.append(DynamicLayer())

    # 3. 计算全局 cache_position 和 position_ids
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # 4. 计算全局 position_embeddings（RoPE，显存开销很小）
    position_embeddings = model.rotary_emb(inputs_embeds, position_ids)

    # 5. 分 chunk 处理
    chunk_prefill_size = CHUNK_PREFILL_SIZE
    num_chunks = (seq_len + chunk_prefill_size - 1) // chunk_prefill_size
    all_hidden_chunks = []
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for c in range(num_chunks):
        start = c * chunk_prefill_size
        end = min((c + 1) * chunk_prefill_size, seq_len)

        # 切分当前 chunk 的各项输入
        chunk_hidden = inputs_embeds[:, start:end, :]
        chunk_position_ids = position_ids[:, start:end]
        chunk_cache_position = cache_position[start:end]
        chunk_pos_emb = (
            position_embeddings[0][:, start:end, :],
            position_embeddings[1][:, start:end, :],
        )

        # 为当前 chunk 构建 causal mask
        # 注意：需要传入 chunk 对应的 cache_position，这样 mask 会考虑已有的 KV Cache
        mask_kwargs = {
            "config": config,
            "input_embeds": chunk_hidden,
            "attention_mask": attention_mask,
            "cache_position": chunk_cache_position,
            "past_key_values": past_key_values,
            "position_ids": chunk_position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        # 逐层通过 decoder layers
        # 注意：position_ids 会在 Olmo3Attention.forward 中被 pop 掉，
        # 不会透传到 flash attention 内部，因此无需在此计算 cu_seq_lens
        for layer_idx, decoder_layer in enumerate(model.layers[: config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (chunk_hidden,)

            attention_type = getattr(
                decoder_layer.self_attn,
                "attention_type",
                config.layer_types[layer_idx] if config.layer_types is not None else "full_attention",
            )
            layer_outputs = decoder_layer(
                chunk_hidden,
                attention_mask=causal_mask_mapping[attention_type],
                position_ids=chunk_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=chunk_cache_position,
                position_embeddings=chunk_pos_emb,
                **flash_attn_kwargs,
            )

            chunk_hidden = layer_outputs[0]

            if output_attentions and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)

        all_hidden_chunks.append(chunk_hidden)

    # 6. 拼接所有 chunk 的输出
    hidden_states = torch.cat(all_hidden_chunks, dim=1)

    # 7. Final norm
    hidden_states = model.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if caller_use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
