import os
import types

import pytest
import torch

# Patch DP-attention globals before importing backends.
from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)

_HSA_VERBOSE = os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") == "1"


def _vprint(*args):
    if _HSA_VERBOSE:
        print(*args, flush=True)


def _torch_swa_decode_window(
    *,
    q: torch.Tensor,  # [B,HQ,D] bf16
    k_cache: torch.Tensor,  # [Nloc,H,D] bf16
    v_cache: torch.Tensor,  # [Nloc,H,D] bf16
    page_table_1: torch.Tensor,  # [B,MAX_T] int32
    seq_lens: torch.Tensor,  # [B] int32
    page_size: int,
    window_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """Return out_swa in float32 with LMK slots excluded."""
    B, HQ, D = q.shape
    _, H, _ = k_cache.shape
    assert HQ % H == 0
    G = HQ // H

    out = torch.zeros((B, HQ, D), device=q.device, dtype=torch.float32)
    seq_lens_i64 = seq_lens.to(torch.int64)

    for b in range(B):
        seqlen = int(seq_lens_i64[b].item())
        if seqlen <= 0:
            continue
        w = min(seqlen, int(window_size))
        start = seqlen - w
        tok_pos = torch.arange(start, seqlen, device=q.device, dtype=torch.int64)
        keep = (tok_pos % int(page_size)) != (int(page_size) - 1)
        tok_pos = tok_pos[keep]
        if tok_pos.numel() == 0:
            continue
        token_locs = page_table_1[b, tok_pos].to(torch.int64)
        q_hgd = q[b].view(H, G, D).to(torch.float32)
        for kv_h in range(H):
            k_win = k_cache[token_locs, kv_h, :].to(torch.float32)  # [S,D]
            v_win = v_cache[token_locs, kv_h, :].to(torch.float32)
            logits = (q_hgd[kv_h] @ k_win.transpose(0, 1)) * float(sm_scale)  # [G,S]
            p = torch.softmax(logits, dim=-1)
            o = p @ v_win
            hq_start = kv_h * G
            out[b, hq_start : hq_start + G, :] = o
    return out


def _torch_hsa_decode_from_weights(
    *,
    q: torch.Tensor,  # [B,HQ,D] bf16
    k_cache: torch.Tensor,  # [Nloc,H,D] bf16
    v_cache: torch.Tensor,  # [Nloc,H,D] bf16
    page_table_1: torch.Tensor,  # [B,MAX_T] int32
    selected_page_ids: torch.Tensor,  # [B,H,K] int32
    hsa_weights: torch.Tensor,  # [B,HQ,K] float32
    page_size: int,
    sm_scale: float,
    mask_last_token: bool,
) -> torch.Tensor:
    B, HQ, D = q.shape
    _, H, _ = k_cache.shape
    assert HQ % H == 0
    G = HQ // H
    K = int(selected_page_ids.shape[2])

    out = torch.zeros((B, HQ, D), device=q.device, dtype=torch.float32)
    for b in range(B):
        for hq in range(HQ):
            kv_h = hq // G
            qq = q[b, hq].float()
            for ki in range(K):
                pid = int(selected_page_ids[b, kv_h, ki].item())
                w = float(hsa_weights[b, hq, ki].item())
                if pid < 0 or w == 0.0:
                    continue
                token_start = pid * int(page_size)
                token_end = token_start + int(page_size)
                token_locs = page_table_1[b, token_start:token_end].to(torch.int64)
                k = k_cache[token_locs, kv_h].float()
                v = v_cache[token_locs, kv_h].float()
                if mask_last_token:
                    k = k[:-1]
                    v = v[:-1]
                logits = (k @ qq) * float(sm_scale)
                p = torch.softmax(logits, dim=0)
                out[b, hq] += w * (p @ v)
    return out


@pytest.mark.skipif(
    torch.cuda.device_count() == 0, reason="CUDA device required for this test"
)
def test_hsa_backend_end_to_end_decode_headwise_split_is_math_correct_cuda():
    """
    Head-wise split correctness test for decode:
    - first half kv-head groups use HSA retrieval (paged kernel)
    - second half kv-head groups use SWA window attention (LMK excluded)
    - no SWA→HSA merged gating; outputs are stitched by head index
    """
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    device = "cuda"
    dtype = torch.bfloat16

    B = 1
    H = 2
    G = 2
    HQ = H * G
    D = 16
    page_size = 4
    topk = 1
    window_size = 4
    sm_scale = float(D) ** -0.5

    prefill_len = 12
    decode_len = 1
    total_len = prefill_len + decode_len

    max_batch_size = 8
    max_context_len = 64
    max_total_num_tokens = max_batch_size * max_context_len

    model_runner = types.SimpleNamespace()
    model_runner.device = device
    model_runner.dtype = dtype
    model_runner.gpu_id = 0
    model_runner.page_size = page_size
    model_runner.sliding_window_size = None
    model_runner.hybrid_gdn_config = None
    model_runner.kimi_linear_config = None
    model_runner.model_config = types.SimpleNamespace(
        context_len=max_context_len,
        num_attention_heads=HQ,
        is_encoder_decoder=False,
        get_num_kv_heads=lambda _tp: H,
    )
    model_runner.model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            hsa_topk=topk,
            hsa_selection_strategy="head",
            enable_swa_hsa_merging=False,
        )
    )
    model_runner.server_args = types.SimpleNamespace(
        attention_backend="hsa",
        speculative_num_draft_tokens=0,
        speculative_num_steps=0,
        triton_attention_num_kv_splits=8,
        triton_attention_split_tile_size=None,
        enable_deterministic_inference=False,
        # Override-only HSA args
        hsa_topk=topk,
        hsa_selection_strategy="head",
        hsa_layers="0",
        hsa_window_size=window_size,
        hsa_enable_swa_merging=False,
        # Experimental head-wise split
        hsa_headwise_swa_split=True,
        hsa_headwise_hsa_kv_heads=H // 2,
        hsa_lmk_id=-1,
    )

    req_to_token = torch.zeros(
        (max_batch_size, max_context_len), dtype=torch.int32, device=device
    )
    model_runner.req_to_token_pool = types.SimpleNamespace(
        size=max_batch_size, req_to_token=req_to_token
    )
    model_runner.token_to_kv_pool = MHATokenToKVPool(
        size=max_total_num_tokens,
        page_size=page_size,
        dtype=dtype,
        head_num=H,
        head_dim=D,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    model_runner.token_to_kv_pool_allocator = object()

    backend = HSAAttnBackend(model_runner)
    layer = RadixAttention(
        num_heads=HQ,
        head_dim=D,
        scaling=sm_scale,
        num_kv_heads=H,
        layer_id=0,
    )

    token_locs = torch.arange(0, total_len, dtype=torch.int32, device=device)
    model_runner.req_to_token_pool.req_to_token[0, :total_len] = token_locs

    # Prefill KV (including LMKs at 3,7,11)
    prefill_loc = token_locs[:prefill_len].to(torch.int64)
    cache_k = torch.randn((prefill_len, H, D), device=device, dtype=dtype)
    cache_v = torch.randn_like(cache_k)

    # Poison LMK V slots so masking/exclusion is enforced.
    cache_v[3, :, :] = 10000.0
    cache_v[7, :, :] = 10000.0
    cache_v[11, :, :] = 10000.0

    model_runner.token_to_kv_pool.set_kv_buffer(layer, prefill_loc, cache_k, cache_v)

    # Decode at loc=12
    q3 = torch.randn((B, HQ, D), device=device, dtype=dtype)
    q = q3.reshape(B, HQ * D)
    k_new = torch.randn((B, H, D), device=device, dtype=dtype)
    v_new = torch.randn_like(k_new)
    out_cache_loc = token_locs[-1:].to(torch.int64)

    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=B,
        input_ids=torch.randint(0, 100, (B, 1), device=device),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([total_len], device=device, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=total_len,
        seq_lens_cpu=torch.tensor([total_len], device="cpu", dtype=torch.int32),
        attn_backend=backend,
    )
    forward_batch.req_to_token_pool = model_runner.req_to_token_pool
    forward_batch.token_to_kv_pool = model_runner.token_to_kv_pool

    backend.init_forward_metadata(forward_batch)
    out_backend = backend.forward_decode(q, k_new, v_new, layer, forward_batch, save_kv_cache=True)
    out_backend_3 = out_backend.view(B, HQ, D)

    md = backend.forward_metadata
    assert md is not None

    # Selection should only be populated for the first H//2 kv heads; remainder must be invalid.
    hsa_kv = H // 2
    assert torch.all(md.hsa_selected_page_ids[:, hsa_kv:, :] == -1)

    # Torch reference:
    # - HSA heads: HSA-only weights from selected_scores
    selected_page_ids = md.hsa_selected_page_ids
    selected_scores = md.hsa_selected_scores
    valid = selected_page_ids >= 0
    scores = selected_scores.masked_fill(~valid, float("-inf"))
    w_kv = torch.softmax(scores, dim=-1)
    w_kv = torch.nan_to_num(w_kv, nan=0.0)
    w_q = w_kv[:, :, None, :].expand(B, H, G, topk).reshape(B, HQ, topk)

    out_hsa = _torch_hsa_decode_from_weights(
        q=q3,
        k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
        v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
        page_table_1=md.page_table_1,
        selected_page_ids=selected_page_ids,
        hsa_weights=w_q,
        page_size=page_size,
        sm_scale=layer.scaling,
        mask_last_token=True,
    )
    out_swa = _torch_swa_decode_window(
        q=q3,
        k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
        v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
        page_table_1=md.page_table_1,
        seq_lens=md.cache_seqlens_int32,
        page_size=page_size,
        window_size=window_size,
        sm_scale=layer.scaling,
    )

    hsa_q_end = hsa_kv * G
    out_ref = torch.empty_like(out_backend_3, dtype=torch.float32)
    out_ref[:, :hsa_q_end, :] = out_hsa[:, :hsa_q_end, :]
    out_ref[:, hsa_q_end:, :] = out_swa[:, hsa_q_end:, :]

    _vprint("### headwise split ref comparison")
    _vprint(f"- hsa_kv={hsa_kv} hsa_q_end={hsa_q_end} HQ={HQ}")
    _vprint(f"- max_abs_err={(out_backend_3.float() - out_ref).abs().max().item()}")

    torch.testing.assert_close(out_backend_3.float(), out_ref, rtol=3e-2, atol=3e-2)


