import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def cu_seqlens_to_doc_ids(cu_seq_lens, total_len, device):
    """Convert cu_seq_lens (e.g. [0, 100, 256]) to a per-token doc_id tensor of shape (total_len,).
    Vectorized: no Python loop, pure tensor ops on GPU."""
    if not isinstance(cu_seq_lens, torch.Tensor):
        cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int32, device=device)
    else:
        cu_seq_lens = cu_seq_lens.to(device)
    # Each boundary (except the first) increments the doc_id by 1
    doc_ids = torch.zeros(total_len, dtype=torch.int32, device=device)
    # Scatter +1 at each sequence start (skip index 0)
    boundaries = cu_seq_lens[1:-1]  # interior boundaries only
    if boundaries.numel() > 0:
        doc_ids[boundaries.long()] = 1
        doc_ids = doc_ids.cumsum(0)
    return doc_ids


@torch.compile(dynamic=False)
def _compiled_flex_attention_train(q, k, v, window_size, chunk_size):
    L = q.shape[-2]

    def block_causal_mask(b, h, q_idx, kv_idx):
        start = q_idx - window_size + 1
        chunk_start = (start // chunk_size) * chunk_size
        return (kv_idx >= chunk_start) & (kv_idx <= q_idx) & ((kv_idx + 1) % chunk_size != 0)

    block_mask = create_block_mask(block_causal_mask, B=None, H=None, Q_LEN=L, KV_LEN=L)
    return flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=True,
        return_lse=True,
    )


@torch.compile(dynamic=False)
def _compiled_flex_attention_train_varlen(q, k, v, window_size, chunk_size, doc_ids):
    L = q.shape[-2]

    def block_causal_mask(b, h, q_idx, kv_idx):
        same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
        start = q_idx - window_size + 1
        chunk_start = (start // chunk_size) * chunk_size
        return same_doc & (kv_idx >= chunk_start) & (kv_idx <= q_idx) & ((kv_idx + 1) % chunk_size != 0)

    block_mask = create_block_mask(block_causal_mask, B=None, H=None, Q_LEN=L, KV_LEN=L)
    return flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=True,
        return_lse=True,
    )


@torch.compile(dynamic=True)
def _compiled_flex_attention_inference_inner(q, k, v, window_size, chunk_size, kv_offset):
    Q_LEN = q.shape[-2]
    KV_LEN_truncated = k.shape[-2]
    ORIG_KV_LEN = KV_LEN_truncated + kv_offset

    def block_causal_mask(b, h, q_idx, kv_idx):
        real_q_idx = q_idx + (ORIG_KV_LEN - Q_LEN)
        real_kv_idx = kv_idx + kv_offset
        start = real_q_idx - window_size + 1
        chunk_start = (start // chunk_size) * chunk_size
        return (real_kv_idx >= chunk_start) & (real_kv_idx <= real_q_idx) & ((real_kv_idx + 1) % chunk_size != 0)

    block_mask = create_block_mask(block_causal_mask, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN_truncated)

    return flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=True,
        return_lse=True,
    )


def _compiled_flex_attention_inference(q, k, v, window_size, chunk_size):
    # 全量传入 KV，不做截断，保证 forward 全量 prefill 和 generate decode 的 block 边界对齐一致
    return _compiled_flex_attention_inference_inner(q, k, v, window_size, chunk_size, 0)

    # --- 原始实现：截断 KV 以减少计算量（可能导致 block 边界对齐差异） ---
    # Q_LEN = q.shape[-2]
    # KV_LEN = k.shape[-2]
    # earliest_real_q = KV_LEN - Q_LEN
    # kv_start_needed = ((earliest_real_q - window_size + 1) // chunk_size) * chunk_size
    # kv_start = max(0, kv_start_needed - chunk_size)
    # if kv_start > 0:
    #     k = k[:, :, kv_start:, :].contiguous()
    #     v = v[:, :, kv_start:, :].contiguous()
    # return _compiled_flex_attention_inference_inner(q, k, v, window_size, chunk_size, kv_start)


def flex_attn(q, k, v, window_size, chunk_size, training=True, doc_ids=None):
    if training:
        if doc_ids is not None:
            return _compiled_flex_attention_train_varlen(q, k, v, window_size, chunk_size, doc_ids)
        return _compiled_flex_attention_train(q, k, v, window_size, chunk_size)
    else:
        return _compiled_flex_attention_inference(q, k, v, window_size, chunk_size)
