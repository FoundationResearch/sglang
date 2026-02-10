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


def _torch_swa_decode_window_innerx(
    *,
    q: torch.Tensor,  # [B,HQ,D] bf16
    k_cache: torch.Tensor,  # [Nloc,H,D] bf16
    v_cache: torch.Tensor,  # [Nloc,H,D] bf16
    page_table_1: torch.Tensor,  # [B,MAX_T] int32
    seq_lens: torch.Tensor,  # [B] int32
    window_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """SWA decode over last `window_size` tokens (InnerX: LMK NOT excluded)."""
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


def test_hsa_backend_innerx_split_head_decode_is_math_correct_cuda():
    """
    End-to-end correctness for InnerX split-head HSA decode (single layer):
    - Backend: uses `hsa_split_head_info` + `hsa_selection_q`:
      - SWA heads: sliding-window attention (LMK included)
      - HSA heads: selection on LMK-K + paged HSA kernel (mask_last_token=True)
      - output is head-wise concatenation [SWA | HSA]
    - Torch ref: recompute selection using `hsa_selection_q`, then compute SWA+HSA separately and stitch.
    """
    from sglang.srt.layers.attention.hsa.selector import (
        build_active_page_candidates,
        select_topk_pages_decode,
    )
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Small, deterministic-ish setup.
    B = 1
    page_size = 4
    total_len = 8  # two full pages, LMK at 3 and 7
    prefill_len = total_len
    max_context_len = 16
    max_total_num_tokens = 64

    # Head partitions: [SWA | HSA]
    HQ_total, H_total, D = 4, 2, 16
    HQ_swa, HQ_hsa = 2, 2
    H_swa, H_hsa = 1, 1
    assert HQ_swa + HQ_hsa == HQ_total
    assert H_swa + H_hsa == H_total
    sm_scale = float(D) ** -0.5
    window_size = 4
    topk = 2

    # Minimal ModelRunner stub.
    model_runner = types.SimpleNamespace()
    model_runner.device = device
    model_runner.page_size = page_size
    model_runner.sliding_window_size = None
    model_runner.model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            hsa_topk=topk,
            hsa_selection_strategy="head",
            # disable merged mode: InnerX uses split-head stitch
            enable_swa_hsa_merging=False,
            use_sliding_window_merging=True,
            sliding_window_merging_size=window_size,
        )
    )
    model_runner.model_config = types.SimpleNamespace(
        is_encoder_decoder=False,
        num_attention_heads=HQ_total,
        num_key_value_heads=H_total,
        head_dim=D,
        context_len=max_context_len,
    )
    model_runner.model_config.get_num_kv_heads = lambda tp_size: H_total // int(tp_size)
    model_runner.hybrid_gdn_config = None
    model_runner.kimi_linear_config = None
    model_runner.gpu_id = 0
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
        hsa_window_size=None,
        hsa_enable_swa_merging=False,
        hsa_lmk_id=-1,
    )

    req_to_token = torch.zeros((B, max_context_len), dtype=torch.int32, device=device)
    model_runner.req_to_token_pool = types.SimpleNamespace(size=B, req_to_token=req_to_token)
    model_runner.token_to_kv_pool = MHATokenToKVPool(
        size=max_total_num_tokens,
        page_size=page_size,
        dtype=dtype,
        head_num=H_total,
        head_dim=D,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    model_runner.token_to_kv_pool_allocator = object()

    backend = HSAAttnBackend(model_runner)
    layer = RadixAttention(
        num_heads=HQ_total,
        head_dim=D,
        scaling=sm_scale,
        num_kv_heads=H_total,
        layer_id=0,
    )

    token_locs = torch.arange(0, total_len + 1, dtype=torch.int32, device=device)
    model_runner.req_to_token_pool.req_to_token[0, : total_len + 1] = token_locs

    # Prefill KV (including LMKs).
    prefill_loc = token_locs[:prefill_len].to(torch.int64)
    cache_k = torch.randn((prefill_len, H_total, D), device=device, dtype=dtype)
    cache_v = torch.randn_like(cache_k)

    # Deterministic LMK-K for the HSA kv head (index 1).
    e0 = torch.zeros((D,), device=device, dtype=dtype)
    e0[0] = 10.0
    e1 = torch.zeros((D,), device=device, dtype=dtype)
    e1[1] = 10.0
    cache_k[3, 1, :] = e0  # page0 LMK
    cache_k[7, 1, :] = e1  # page1 LMK

    model_runner.token_to_kv_pool.set_kv_buffer(layer, prefill_loc, cache_k, cache_v)

    # Decode at loc=8 (token_locs[8]).
    q3 = torch.zeros((B, HQ_total, D), device=device, dtype=dtype)
    q3[0, :HQ_swa, :] = e0  # SWA heads
    q3[0, HQ_swa:, :] = e1  # HSA heads (used by kernel, not selection)
    q = q3.reshape(B, HQ_total * D)
    k_new = torch.randn((B, H_total, D), device=device, dtype=dtype)
    v_new = torch.randn_like(k_new)
    out_cache_loc = token_locs[-1:].to(torch.int64)

    # Selection query for decode: use e0 so it should prefer page0 over page1.
    sel_q = torch.zeros((B, HQ_hsa, D), device=device, dtype=dtype)
    sel_q[0, :, :] = e0

    split_info = dict(
        hq_swa=HQ_swa,
        hq_hsa=HQ_hsa,
        h_swa=H_swa,
        h_hsa=H_hsa,
        swa_window_size=window_size,
        swa_exclude_lmk=False,
    )

    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=B,
        input_ids=torch.randint(0, 100, (B, 1), device=device),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([total_len + 1], device=device, dtype=torch.int32),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=total_len + 1,
        seq_lens_cpu=torch.tensor([total_len + 1], device="cpu", dtype=torch.int32),
        attn_backend=backend,
    )
    forward_batch.req_to_token_pool = model_runner.req_to_token_pool
    forward_batch.token_to_kv_pool = model_runner.token_to_kv_pool

    backend.init_forward_metadata(forward_batch)
    out_backend = backend.forward_decode(
        q,
        k_new,
        v_new,
        layer,
        forward_batch,
        save_kv_cache=True,
        hsa_split_head_info=split_info,
        hsa_selection_q=sel_q,
    )
    out_backend_3 = out_backend.view(B, HQ_total, D).to(torch.float32)

    md = backend.forward_metadata
    assert md is not None

    # Torch recompute selection on HSA kv head slice only.
    cand_page_ids, cand_mask = build_active_page_candidates(
        page_table_1=md.page_table_1,
        seq_lens=md.cache_seqlens_int32,
        page_size=page_size,
        window_size=0,  # InnerX: no window exclusion in candidate construction
    )
    completed_pages = torch.div(
        md.cache_seqlens_int32.to(torch.int64), int(page_size), rounding_mode="floor"
    )
    cand_completed = cand_mask & (cand_page_ids.to(torch.int64) < completed_pages[:, None])
    safe_page_ids = cand_page_ids.clamp_min(0).to(torch.int64)
    lmk_locs = safe_page_ids * int(page_size) + (int(page_size) - 1)
    flat_lmk = lmk_locs.reshape(-1)
    k_cache_full = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
    flat_repr = k_cache_full[flat_lmk][:, H_swa : H_swa + H_hsa, :]  # HSA kv slice
    cand_repr = flat_repr.view(B, cand_page_ids.shape[1], H_hsa, D)

    sel = select_topk_pages_decode(
        q=sel_q,
        cand_page_ids=cand_page_ids,
        cand_mask=cand_mask,
        cand_chunk_repr=cand_repr,
        cand_chunk_repr_valid=cand_completed,
        topk=topk,
        selection_strategy="head",
        sm_scale=layer.scaling,
    )
    torch.testing.assert_close(md.hsa_selected_page_ids, sel.selected_page_ids)
    torch.testing.assert_close(md.hsa_selected_scores, sel.selected_scores)

    valid = sel.selected_page_ids >= 0
    scores = sel.selected_scores.masked_fill(~valid, float("-inf"))
    w_kv = torch.softmax(scores, dim=-1)
    w_kv = torch.nan_to_num(w_kv, nan=0.0)
    # Expand kv-head weights to q-head weights (G = HQ_hsa/H_hsa).
    assert HQ_hsa % H_hsa == 0
    G_hsa = HQ_hsa // H_hsa
    w_q = (
        w_kv[:, :, None, :]
        .expand(B, H_hsa, G_hsa, topk)
        .reshape(B, HQ_hsa, topk)
        .contiguous()
    )

    # Torch ref SWA + HSA then stitch.
    page_table_1 = md.page_table_1
    seq_lens = md.cache_seqlens_int32
    out_swa = _torch_swa_decode_window_innerx(
        q=q3[:, :HQ_swa, :],
        k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)[:, :H_swa, :],
        v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)[:, :H_swa, :],
        page_table_1=page_table_1,
        seq_lens=seq_lens,
        window_size=window_size,
        sm_scale=sm_scale,
    )
    out_hsa = _torch_hsa_decode_from_weights(
        q=q3[:, HQ_swa:, :],
        k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :],
        v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)[:, H_swa : H_swa + H_hsa, :],
        page_table_1=page_table_1,
        selected_page_ids=sel.selected_page_ids,
        hsa_weights=w_q,
        page_size=page_size,
        sm_scale=sm_scale,
        mask_last_token=True,
    )
    out_ref = torch.empty((B, HQ_total, D), device=device, dtype=torch.float32)
    out_ref[:, :HQ_swa, :] = out_swa
    out_ref[:, HQ_swa:, :] = out_hsa

    max_abs = (out_backend_3 - out_ref).abs().max().item()
    _vprint("### InnerX split-head decode check")
    _vprint(f"- cand_page_ids={cand_page_ids[0].tolist()} cand_mask={cand_mask[0].tolist()}")
    _vprint(f"- selected_page_ids={sel.selected_page_ids[0, 0].tolist()}")
    _vprint(f"- max_abs_err={max_abs}")

    torch.testing.assert_close(out_backend_3, out_ref, rtol=5e-2, atol=5e-2)

