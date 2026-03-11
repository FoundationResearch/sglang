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


def _torch_internal_swa_decode(
    *,
    q: torch.Tensor,  # [B,HQ_hsa,D] bf16
    k_cache: torch.Tensor,  # [Nloc,H_hsa,D] bf16 (HSA kv heads only)
    v_cache: torch.Tensor,  # [Nloc,H_hsa,D] bf16
    page_table_1: torch.Tensor,  # [B,MAX_T] int32
    seq_lens: torch.Tensor,  # [B] int32
    window_size: int,
    page_size: int,
    sm_scale: float,
) -> tuple:
    """Internal SWA on HSA heads: chunk-aligned window, LMK excluded.

    Returns (swa_o [B,HQ,D] f32, lse_kv [B,H] f32).
    """
    B, HQ, D = q.shape
    _, H, _ = k_cache.shape
    assert HQ % H == 0
    G = HQ // H
    swa_o = torch.zeros((B, HQ, D), device=q.device, dtype=torch.float32)
    lse_kv = torch.full((B, H), float("-inf"), device=q.device, dtype=torch.float32)
    seq_lens_i64 = seq_lens.to(torch.int64)
    for b in range(B):
        seqlen = int(seq_lens_i64[b].item())
        if seqlen <= 0:
            continue
        q_pos = seqlen - 1
        raw_start = q_pos - window_size + 1
        chunk_start = max(0, (raw_start // page_size) * page_size) if raw_start >= 0 else 0
        tok_pos = torch.arange(chunk_start, seqlen, device=q.device, dtype=torch.int64)
        keep = (tok_pos % page_size) != (page_size - 1)
        tok_pos = tok_pos[keep]
        if tok_pos.numel() == 0:
            continue
        token_locs = page_table_1[b, tok_pos].to(torch.int64)
        q_hgd = q[b].view(H, G, D).to(torch.float32)
        lse_per_q = torch.full((H, G), float("-inf"), device=q.device, dtype=torch.float32)
        for kv_h in range(H):
            k_win = k_cache[token_locs, kv_h, :].to(torch.float32)
            v_win = v_cache[token_locs, kv_h, :].to(torch.float32)
            logits = (q_hgd[kv_h] @ k_win.transpose(0, 1)) * sm_scale
            lse_per_q[kv_h] = torch.logsumexp(logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            o = p @ v_win
            hq_start = kv_h * G
            swa_o[b, hq_start : hq_start + G, :] = o
        lse_kv[b] = torch.logsumexp(lse_per_q, dim=-1)
    return swa_o, lse_kv


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
    End-to-end correctness for InnerX split-head HSA decode with LHSA semantics:
    - Backend: uses `hsa_split_head_info` + `hsa_selection_q`:
      - SWA heads: sliding-window attention (LMK included)
      - HSA heads: internal SWA → merged softmax → paged HSA kernel → fusion
      - output is head-wise concatenation [SWA | HSA]
    - Torch ref: recompute internal SWA + selection + merged softmax + fusion.
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
            enable_softmax1=False,
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

    # ---- Torch reference: LHSA semantics ----
    page_table_1 = md.page_table_1
    seq_lens = md.cache_seqlens_int32
    k_cache_full = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
    v_cache_full = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

    # 1) Outer SWA on SWA heads (unchanged).
    out_swa = _torch_swa_decode_window_innerx(
        q=q3[:, :HQ_swa, :],
        k_cache=k_cache_full[:, :H_swa, :],
        v_cache=v_cache_full[:, :H_swa, :],
        page_table_1=page_table_1,
        seq_lens=seq_lens,
        window_size=window_size,
        sm_scale=sm_scale,
    )

    # 2) Internal SWA on HSA heads → output + logsumexp.
    swa_o_inner, lse_kv = _torch_internal_swa_decode(
        q=q3[:, HQ_swa:, :],
        k_cache=k_cache_full[:, H_swa : H_swa + H_hsa, :],
        v_cache=v_cache_full[:, H_swa : H_swa + H_hsa, :],
        page_table_1=page_table_1,
        seq_lens=seq_lens,
        window_size=window_size,
        page_size=page_size,
        sm_scale=sm_scale,
    )

    # 3) Selection with window exclusion (SWA-covered chunks excluded).
    cand_page_ids, cand_mask = build_active_page_candidates(
        page_table_1=page_table_1,
        seq_lens=seq_lens,
        page_size=page_size,
        window_size=window_size,
    )
    completed_pages = torch.div(
        seq_lens.to(torch.int64), int(page_size), rounding_mode="floor"
    )
    cand_completed = cand_mask & (cand_page_ids.to(torch.int64) < completed_pages[:, None])
    safe_page_ids = cand_page_ids.clamp_min(0).to(torch.int64)
    lmk_locs = safe_page_ids * int(page_size) + (int(page_size) - 1)
    flat_lmk = lmk_locs.reshape(-1)
    flat_repr = k_cache_full[flat_lmk][:, H_swa : H_swa + H_hsa, :]
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

    # 4) Merged softmax: cat([chunk_scores, swa_lse]) → globally normalized.
    valid = sel.selected_page_ids >= 0
    scores = sel.selected_scores.masked_fill(~valid, float("-inf"))
    cat_scores = torch.cat([scores, lse_kv.unsqueeze(-1)], dim=-1)  # [B, H_hsa, K+1]
    merged_w = torch.softmax(cat_scores, dim=-1)
    merged_w = torch.nan_to_num(merged_w, nan=0.0)
    w_kv = merged_w[:, :, :topk]
    swa_w_kv = merged_w[:, :, -1]  # [B, H_hsa]

    assert HQ_hsa % H_hsa == 0
    G_hsa = HQ_hsa // H_hsa
    w_q = (
        w_kv[:, :, None, :]
        .expand(B, H_hsa, G_hsa, topk)
        .reshape(B, HQ_hsa, topk)
        .contiguous()
    )

    # 5) HSA kernel with chunk weights.
    out_hsa_chunks = _torch_hsa_decode_from_weights(
        q=q3[:, HQ_swa:, :],
        k_cache=k_cache_full[:, H_swa : H_swa + H_hsa, :],
        v_cache=v_cache_full[:, H_swa : H_swa + H_hsa, :],
        page_table_1=page_table_1,
        selected_page_ids=sel.selected_page_ids,
        hsa_weights=w_q,
        page_size=page_size,
        sm_scale=sm_scale,
        mask_last_token=True,
    )

    # 6) Weighted fusion: o_lower = hsa_o + swa_o * swa_weight.
    swa_w_q = (
        swa_w_kv[:, :, None]
        .expand(B, H_hsa, G_hsa)
        .reshape(B, HQ_hsa)
    )
    out_hsa = out_hsa_chunks + swa_o_inner * swa_w_q[:, :, None]

    # 7) Stitch.
    out_ref = torch.empty((B, HQ_total, D), device=device, dtype=torch.float32)
    out_ref[:, :HQ_swa, :] = out_swa
    out_ref[:, HQ_swa:, :] = out_hsa

    max_abs = (out_backend_3 - out_ref).abs().max().item()
    _vprint("### InnerX split-head decode check (LHSA semantics)")
    _vprint(f"- cand_page_ids={cand_page_ids[0].tolist()} cand_mask={cand_mask[0].tolist()}")
    _vprint(f"- selected_page_ids={sel.selected_page_ids[0, 0].tolist()}")
    _vprint(f"- lse_kv={lse_kv[0].tolist()}")
    _vprint(f"- swa_w_kv={swa_w_kv[0].tolist()}")
    _vprint(f"- max_abs_err={max_abs}")

    torch.testing.assert_close(out_backend_3, out_ref, rtol=5e-2, atol=5e-2)


def _run_lhsa_decode_reference(
    *,
    q3: torch.Tensor,       # [B, HQ_total, D]
    sel_q: torch.Tensor,    # [B, HQ_hsa, D]
    k_cache: torch.Tensor,  # [Nloc, H_total, D]
    v_cache: torch.Tensor,  # [Nloc, H_total, D]
    page_table_1: torch.Tensor,  # [B, MAX_T]
    seq_lens: torch.Tensor,      # [B]
    HQ_swa: int, HQ_hsa: int, H_swa: int, H_hsa: int,
    window_size: int, page_size: int, topk: int, sm_scale: float,
    enable_softmax1: bool = False,
) -> torch.Tensor:
    """Torch reference for LHSA decode (supports B>=1, different seq_lens)."""
    from sglang.srt.layers.attention.hsa.selector import (
        build_active_page_candidates,
        select_topk_pages_decode,
    )

    B = q3.shape[0]
    HQ_total = HQ_swa + HQ_hsa
    D = q3.shape[2]
    device = q3.device

    # 1) Outer SWA on SWA heads.
    out_swa = _torch_swa_decode_window_innerx(
        q=q3[:, :HQ_swa, :],
        k_cache=k_cache[:, :H_swa, :],
        v_cache=v_cache[:, :H_swa, :],
        page_table_1=page_table_1,
        seq_lens=seq_lens,
        window_size=window_size,
        sm_scale=sm_scale,
    )

    # 2) Internal SWA on HSA heads.
    swa_o_inner, lse_kv = _torch_internal_swa_decode(
        q=q3[:, HQ_swa:, :],
        k_cache=k_cache[:, H_swa : H_swa + H_hsa, :],
        v_cache=v_cache[:, H_swa : H_swa + H_hsa, :],
        page_table_1=page_table_1,
        seq_lens=seq_lens,
        window_size=window_size,
        page_size=page_size,
        sm_scale=sm_scale,
    )

    # 3) Selection with window exclusion.
    cand_page_ids, cand_mask = build_active_page_candidates(
        page_table_1=page_table_1,
        seq_lens=seq_lens,
        page_size=page_size,
        window_size=window_size,
    )
    completed_pages = torch.div(
        seq_lens.to(torch.int64), int(page_size), rounding_mode="floor"
    )
    cand_completed = cand_mask & (cand_page_ids.to(torch.int64) < completed_pages[:, None])
    safe_page_ids = cand_page_ids.clamp_min(0).to(torch.int64)
    lmk_locs = safe_page_ids * int(page_size) + (int(page_size) - 1)
    flat_lmk = lmk_locs.reshape(-1)
    flat_repr = k_cache[flat_lmk][:, H_swa : H_swa + H_hsa, :]
    cand_repr = flat_repr.view(B, cand_page_ids.shape[1], H_hsa, D)

    sel = select_topk_pages_decode(
        q=sel_q,
        cand_page_ids=cand_page_ids,
        cand_mask=cand_mask,
        cand_chunk_repr=cand_repr,
        cand_chunk_repr_valid=cand_completed,
        topk=topk,
        selection_strategy="head",
        sm_scale=sm_scale,
    )

    # 4) Merged softmax.
    valid = sel.selected_page_ids >= 0
    scores = sel.selected_scores.masked_fill(~valid, float("-inf"))
    if not enable_softmax1:
        cat_scores = torch.cat([scores, lse_kv.unsqueeze(-1)], dim=-1)
        swa_weight_idx = -1
    else:
        cat_scores = torch.cat([
            scores,
            lse_kv.unsqueeze(-1),
            torch.zeros(B, H_hsa, 1, device=device, dtype=scores.dtype),
        ], dim=-1)
        swa_weight_idx = -2
    merged_w = torch.softmax(cat_scores, dim=-1)
    merged_w = torch.nan_to_num(merged_w, nan=0.0)
    w_kv = merged_w[:, :, :topk]
    swa_w_kv = merged_w[:, :, swa_weight_idx]

    G_hsa = HQ_hsa // H_hsa
    w_q = (
        w_kv[:, :, None, :]
        .expand(B, H_hsa, G_hsa, topk)
        .reshape(B, HQ_hsa, topk)
        .contiguous()
    )

    # 5) HSA kernel.
    out_hsa_chunks = _torch_hsa_decode_from_weights(
        q=q3[:, HQ_swa:, :],
        k_cache=k_cache[:, H_swa : H_swa + H_hsa, :],
        v_cache=v_cache[:, H_swa : H_swa + H_hsa, :],
        page_table_1=page_table_1,
        selected_page_ids=sel.selected_page_ids,
        hsa_weights=w_q,
        page_size=page_size,
        sm_scale=sm_scale,
        mask_last_token=True,
    )

    # 6) Weighted fusion.
    swa_w_q = (
        swa_w_kv[:, :, None]
        .expand(B, H_hsa, G_hsa)
        .reshape(B, HQ_hsa)
    )
    out_hsa = out_hsa_chunks + swa_o_inner * swa_w_q[:, :, None]

    # 7) Stitch.
    out_ref = torch.empty((B, HQ_total, D), device=device, dtype=torch.float32)
    out_ref[:, :HQ_swa, :] = out_swa
    out_ref[:, HQ_swa:, :] = out_hsa
    return out_ref


def test_hsa_backend_innerx_split_head_decode_batch_continuous_batching_cuda():
    """
    B>1 correctness with different sequence lengths (continuous batching scenario).

    Req 0: seqlen=9  (2 full pages + 1 token — chunk-aligned window covers page 1)
    Req 1: seqlen=13 (3 full pages + 1 token — chunk-aligned window covers page 2,
           pages 0,1 available for selection)

    Verifies that the backend produces correct per-request LHSA output
    matching an independent torch reference for each request.
    """
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = 2
    page_size = 4
    # Req 0: 8 prefill tokens (2 full pages), decode at pos 8 → seqlen=9
    # Req 1: 12 prefill tokens (3 full pages), decode at pos 12 → seqlen=13
    prefill_lens = [8, 12]
    seq_lens_after_decode = [9, 13]
    max_prefill = max(prefill_lens)
    max_context_len = 32
    max_total_num_tokens = 256

    HQ_total, H_total, D = 4, 2, 16
    HQ_swa, HQ_hsa = 2, 2
    H_swa, H_hsa = 1, 1
    sm_scale = float(D) ** -0.5
    window_size = 4
    topk = 2

    # ModelRunner stub.
    model_runner = types.SimpleNamespace()
    model_runner.device = device
    model_runner.page_size = page_size
    model_runner.sliding_window_size = None
    model_runner.model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            hsa_topk=topk,
            hsa_selection_strategy="head",
            enable_swa_hsa_merging=False,
            use_sliding_window_merging=True,
            sliding_window_merging_size=window_size,
            enable_softmax1=False,
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

    torch.manual_seed(42)

    # Allocate separate KV slots for each request (non-overlapping).
    # Req 0: locations [0, prefill_lens[0])  + decode at prefill_lens[0]
    # Req 1: locations [64, 64+prefill_lens[1]) + decode at 64+prefill_lens[1]
    loc_offsets = [0, 64]
    for b in range(B):
        locs = torch.arange(
            loc_offsets[b], loc_offsets[b] + prefill_lens[b] + 1,
            dtype=torch.int32, device=device,
        )
        req_to_token[b, : prefill_lens[b] + 1] = locs

    # Prefill KV for both requests.
    e0 = torch.zeros((D,), device=device, dtype=dtype)
    e0[0] = 10.0
    e1 = torch.zeros((D,), device=device, dtype=dtype)
    e1[1] = 10.0
    e2 = torch.zeros((D,), device=device, dtype=dtype)
    e2[2] = 10.0

    for b in range(B):
        prefill_locs = torch.arange(
            loc_offsets[b], loc_offsets[b] + prefill_lens[b],
            dtype=torch.int64, device=device,
        )
        cache_k = torch.randn((prefill_lens[b], H_total, D), device=device, dtype=dtype)
        cache_v = torch.randn((prefill_lens[b], H_total, D), device=device, dtype=dtype)
        # Set deterministic LMK-K for HSA kv head (index 1) at page boundaries.
        for p in range(prefill_lens[b] // page_size):
            lmk_pos = p * page_size + (page_size - 1)
            cache_k[lmk_pos, 1, :] = [e0, e1, e2][p % 3]
        model_runner.token_to_kv_pool.set_kv_buffer(layer, prefill_locs, cache_k, cache_v)

    # Decode queries and new KV.
    q3 = torch.randn((B, HQ_total, D), device=device, dtype=dtype)
    q = q3.reshape(B, HQ_total * D)
    k_new = torch.randn((B, H_total, D), device=device, dtype=dtype)
    v_new = torch.randn_like(k_new)
    out_cache_locs = torch.tensor(
        [loc_offsets[b] + prefill_lens[b] for b in range(B)],
        dtype=torch.int64, device=device,
    )

    # Selection queries (different for each request).
    sel_q = torch.randn((B, HQ_hsa, D), device=device, dtype=dtype)

    split_info = dict(
        hq_swa=HQ_swa,
        hq_hsa=HQ_hsa,
        h_swa=H_swa,
        h_hsa=H_hsa,
        swa_window_size=window_size,
        swa_exclude_lmk=False,
    )

    seq_lens_t = torch.tensor(seq_lens_after_decode, device=device, dtype=torch.int32)
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=B,
        input_ids=torch.randint(0, 100, (B, 1), device=device),
        req_pool_indices=torch.arange(B, device=device, dtype=torch.int32),
        seq_lens=seq_lens_t,
        out_cache_loc=out_cache_locs,
        seq_lens_sum=int(seq_lens_t.sum().item()),
        seq_lens_cpu=seq_lens_t.to("cpu"),
        attn_backend=backend,
    )
    forward_batch.req_to_token_pool = model_runner.req_to_token_pool
    forward_batch.token_to_kv_pool = model_runner.token_to_kv_pool

    backend.init_forward_metadata(forward_batch)
    out_backend = backend.forward_decode(
        q, k_new, v_new, layer, forward_batch,
        save_kv_cache=True,
        hsa_split_head_info=split_info,
        hsa_selection_q=sel_q,
    )
    out_backend_3 = out_backend.view(B, HQ_total, D).to(torch.float32)

    md = backend.forward_metadata
    assert md is not None

    # Torch reference (supports B>1 natively).
    k_cache_full = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
    v_cache_full = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
    out_ref = _run_lhsa_decode_reference(
        q3=q3,
        sel_q=sel_q,
        k_cache=k_cache_full,
        v_cache=v_cache_full,
        page_table_1=md.page_table_1,
        seq_lens=md.cache_seqlens_int32,
        HQ_swa=HQ_swa, HQ_hsa=HQ_hsa, H_swa=H_swa, H_hsa=H_hsa,
        window_size=window_size, page_size=page_size,
        topk=topk, sm_scale=sm_scale,
    )

    max_abs = (out_backend_3 - out_ref).abs().max().item()
    _vprint("### InnerX split-head decode B>1 continuous batching check")
    _vprint(f"- seq_lens={seq_lens_after_decode}")
    _vprint(f"- max_abs_err={max_abs}")
    for b in range(B):
        err_b = (out_backend_3[b] - out_ref[b]).abs().max().item()
        _vprint(f"  req{b}: seqlen={seq_lens_after_decode[b]}, max_abs_err={err_b}")

    torch.testing.assert_close(out_backend_3, out_ref, rtol=5e-2, atol=5e-2)
