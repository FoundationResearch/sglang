"""
Integration test for HSA extend (prefill) with InnerX split-head LHSA semantics.

Verifies that `HSAAttnBackend.forward_extend` produces outputs matching
a pure-PyTorch per-token reference implementation.

Each query token t at position pos_t goes through:
  1. Internal SWA (chunk-aligned, LMK-excluded) → output + LSE
  2. TopK page selection (only completed pages before pos_t, outside SWA window)
  3. Paged sparse attention over selected pages (mask_last_token)
  4. Merged softmax fusion
  5. Head-wise stitch with SWA heads (dense attention)
"""

import os
import types

import pytest
import torch

from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)

_HSA_VERBOSE = os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") == "1"


def _vprint(*args):
    if _HSA_VERBOSE:
        print(*args, flush=True)


# ---- PyTorch reference helpers ----


def _torch_internal_swa_extend_ref(
    *,
    q_hsa: torch.Tensor,  # [T, HQ_hsa, D]
    k_cache: torch.Tensor,  # [Nloc, H_hsa, D]
    v_cache: torch.Tensor,  # [Nloc, H_hsa, D]
    page_table_1: torch.Tensor,  # [B, MAX_T]
    token_positions: torch.Tensor,  # [T]
    token_to_seq_id: torch.Tensor,  # [T]
    window_size: int,
    page_size: int,
    sm_scale: float,
) -> tuple:
    """Per-token internal SWA on HSA heads.

    Returns (swa_o [T, HQ_hsa, D] f32, lse_kv [T, H_hsa] f32).
    """
    T, HQ, D = q_hsa.shape
    _, H, _ = k_cache.shape
    assert HQ % H == 0
    G = HQ // H
    device = q_hsa.device

    swa_o = torch.zeros((T, HQ, D), device=device, dtype=torch.float32)
    lse_kv = torch.full((T, H), float("-inf"), device=device, dtype=torch.float32)

    for t in range(T):
        pos_t = int(token_positions[t].item())
        b = int(token_to_seq_id[t].item())

        raw_start = pos_t - window_size + 1
        chunk_start = max(0, (raw_start // page_size) * page_size) if raw_start >= 0 else 0

        tok_pos = torch.arange(chunk_start, pos_t + 1, device=device, dtype=torch.int64)
        keep = (tok_pos % page_size) != (page_size - 1)
        tok_pos = tok_pos[keep]
        if tok_pos.numel() == 0:
            continue

        token_locs = page_table_1[b, tok_pos].to(torch.int64)
        q_hgd = q_hsa[t].view(H, G, D).to(torch.float32)

        lse_per_q = torch.full((H, G), float("-inf"), device=device, dtype=torch.float32)
        for kv_h in range(H):
            k_win = k_cache[token_locs, kv_h, :].to(torch.float32)
            v_win = v_cache[token_locs, kv_h, :].to(torch.float32)
            logits = (q_hgd[kv_h] @ k_win.transpose(0, 1)) * sm_scale
            lse_per_q[kv_h] = torch.logsumexp(logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            o = p @ v_win
            hq_start = kv_h * G
            swa_o[t, hq_start : hq_start + G, :] = o
        lse_kv[t] = torch.logsumexp(lse_per_q, dim=-1)

    return swa_o, lse_kv


def _torch_topk_selection_extend_ref(
    *,
    selection_q: torch.Tensor,  # [T, HQ_sel, D]
    k_cache: torch.Tensor,  # [Nloc, H_hsa, D]
    page_table_1: torch.Tensor,  # [B, MAX_T]
    token_positions: torch.Tensor,  # [T]
    token_to_seq_id: torch.Tensor,  # [T]
    page_size: int,
    window_size: int,
    topk: int,
    sm_scale: float,
    selection_strategy: str = "head",
) -> tuple:
    """Per-token TopK page selection.

    Returns (selected_page_ids [T, H, K], selected_scores [T, H, K]).
    """
    from sglang.srt.layers.attention.hsa.selector import select_topk_pages_decode

    T = selection_q.shape[0]
    _, H, D = k_cache.shape
    device = selection_q.device

    all_page_ids = []
    all_scores = []

    for t in range(T):
        pos_t = int(token_positions[t].item())
        b = int(token_to_seq_id[t].item())

        completed = pos_t // page_size
        if completed <= 0:
            all_page_ids.append(torch.full((1, H, topk), -1, device=device, dtype=torch.int32))
            all_scores.append(torch.full((1, H, topk), float("-inf"), device=device, dtype=torch.float32))
            continue

        cand_pages = torch.arange(completed, device=device, dtype=torch.int32)

        # Exclude pages within SWA window.
        if window_size > 0:
            raw_start = pos_t - window_size + 1
            chunk_start = max(0, (raw_start // page_size) * page_size) if raw_start >= 0 else 0
            window_start_page = chunk_start // page_size
            cand_pages = cand_pages[cand_pages < window_start_page]

        C = int(cand_pages.numel())
        if C == 0:
            all_page_ids.append(torch.full((1, H, topk), -1, device=device, dtype=torch.int32))
            all_scores.append(torch.full((1, H, topk), float("-inf"), device=device, dtype=torch.float32))
            continue

        lmk_tok_pos = cand_pages.to(torch.int64) * page_size + (page_size - 1)
        lmk_locs = page_table_1[b, lmk_tok_pos].to(torch.int64)
        lmk_repr = k_cache[lmk_locs]  # [C, H, D]

        sel = select_topk_pages_decode(
            q=selection_q[t:t+1],
            cand_page_ids=cand_pages.unsqueeze(0),
            cand_mask=torch.ones((1, C), device=device, dtype=torch.bool),
            cand_chunk_repr=lmk_repr.unsqueeze(0),
            cand_chunk_repr_valid=torch.ones((1, C), device=device, dtype=torch.bool),
            topk=topk,
            selection_strategy=selection_strategy,
            sm_scale=sm_scale,
        )
        all_page_ids.append(sel.selected_page_ids)
        all_scores.append(sel.selected_scores)

    return torch.cat(all_page_ids, dim=0), torch.cat(all_scores, dim=0)


def _torch_hsa_sparse_attn_extend_ref(
    *,
    q_hsa: torch.Tensor,  # [T, HQ_hsa, D]
    k_cache: torch.Tensor,  # [Nloc, H_hsa, D]
    v_cache: torch.Tensor,  # [Nloc, H_hsa, D]
    page_table_1: torch.Tensor,  # [B, MAX_T]
    token_to_seq_id: torch.Tensor,  # [T]
    selected_page_ids: torch.Tensor,  # [T, H, K]
    hsa_weights: torch.Tensor,  # [T, HQ_hsa, K]
    page_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """Per-token paged HSA sparse attention.

    Returns out [T, HQ_hsa, D] f32.
    """
    T, HQ, D = q_hsa.shape
    _, H, _ = k_cache.shape
    assert HQ % H == 0
    G = HQ // H
    K = int(selected_page_ids.shape[2])
    device = q_hsa.device
    out = torch.zeros((T, HQ, D), device=device, dtype=torch.float32)

    for t in range(T):
        b = int(token_to_seq_id[t].item())
        for hq in range(HQ):
            kv_h = hq // G
            qq = q_hsa[t, hq].float()
            for ki in range(K):
                pid = int(selected_page_ids[t, kv_h, ki].item())
                w = float(hsa_weights[t, hq, ki].item())
                if pid < 0 or w == 0.0:
                    continue
                token_start = pid * page_size
                token_end = token_start + page_size
                tok_pos = torch.arange(token_start, token_end, device=device, dtype=torch.int64)
                token_locs = page_table_1[b, tok_pos].to(torch.int64)
                k_page = k_cache[token_locs, kv_h].float()  # [PS, D]
                v_page = v_cache[token_locs, kv_h].float()  # [PS, D]
                # mask_last_token
                k_page = k_page[:-1]
                v_page = v_page[:-1]
                logits = (k_page @ qq) * sm_scale  # [PS-1]
                p = torch.softmax(logits, dim=0)
                out[t, hq] += w * (p @ v_page)
    return out


def _run_lhsa_extend_reference(
    *,
    q3: torch.Tensor,  # [T, HQ_total, D]
    sel_q: torch.Tensor,  # [T, HQ_hsa, D]
    k_cache: torch.Tensor,  # [Nloc, H_total, D]
    v_cache: torch.Tensor,  # [Nloc, H_total, D]
    page_table_1: torch.Tensor,  # [B, MAX_T]
    token_positions: torch.Tensor,  # [T]
    token_to_seq_id: torch.Tensor,  # [T]
    HQ_swa: int, HQ_hsa: int, H_swa: int, H_hsa: int,
    window_size: int, page_size: int, topk: int, sm_scale: float,
) -> torch.Tensor:
    """Full LHSA extend reference for HSA heads only.

    Returns out_hsa [T, HQ_hsa, D] f32 (fused HSA + internal SWA).
    """
    T = q3.shape[0]
    D = q3.shape[2]
    device = q3.device

    q_hsa = q3[:, HQ_swa:, :]

    # 1) Internal SWA.
    swa_o_inner, lse_kv = _torch_internal_swa_extend_ref(
        q_hsa=q_hsa,
        k_cache=k_cache[:, H_swa:H_swa+H_hsa, :],
        v_cache=v_cache[:, H_swa:H_swa+H_hsa, :],
        page_table_1=page_table_1,
        token_positions=token_positions,
        token_to_seq_id=token_to_seq_id,
        window_size=window_size,
        page_size=page_size,
        sm_scale=sm_scale,
    )

    # 2) Selection.
    selected_page_ids, selected_scores = _torch_topk_selection_extend_ref(
        selection_q=sel_q,
        k_cache=k_cache[:, H_swa:H_swa+H_hsa, :],
        page_table_1=page_table_1,
        token_positions=token_positions,
        token_to_seq_id=token_to_seq_id,
        page_size=page_size,
        window_size=window_size,
        topk=topk,
        sm_scale=sm_scale,
        selection_strategy="head",
    )

    # 3) Merged softmax.
    valid = selected_page_ids >= 0
    scores = selected_scores.masked_fill(~valid, float("-inf"))
    cat_scores = torch.cat([scores, lse_kv.unsqueeze(-1)], dim=-1)
    merged_w = torch.softmax(cat_scores, dim=-1)
    merged_w = torch.nan_to_num(merged_w, nan=0.0)
    w_kv = merged_w[:, :, :topk]
    swa_w_kv = merged_w[:, :, -1]

    G_hsa = HQ_hsa // H_hsa
    w_q = (
        w_kv[:, :, None, :]
        .expand(T, H_hsa, G_hsa, topk)
        .reshape(T, HQ_hsa, topk)
        .contiguous()
    )

    # 4) Sparse attention.
    out_hsa_chunks = _torch_hsa_sparse_attn_extend_ref(
        q_hsa=q_hsa,
        k_cache=k_cache[:, H_swa:H_swa+H_hsa, :],
        v_cache=v_cache[:, H_swa:H_swa+H_hsa, :],
        page_table_1=page_table_1,
        token_to_seq_id=token_to_seq_id,
        selected_page_ids=selected_page_ids,
        hsa_weights=w_q,
        page_size=page_size,
        sm_scale=sm_scale,
    )

    # 5) Fusion.
    swa_w_q = (
        swa_w_kv[:, :, None]
        .expand(T, H_hsa, G_hsa)
        .reshape(T, HQ_hsa)
    )
    out_hsa = out_hsa_chunks + swa_o_inner * swa_w_q[:, :, None]
    return out_hsa


# ---- Actual tests ----


def _make_model_runner_stub(
    *, B, page_size, HQ_total, H_total, D, max_context_len,
    max_total_num_tokens, topk, window_size, dtype=torch.bfloat16,
):
    """Create a minimal ModelRunner stub for HSAAttnBackend."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    device = torch.device("cuda")
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
    return model_runner


def test_hsa_backend_innerx_split_head_extend_single_seq_cuda():
    """
    Single-sequence extend: prefix_len=12 (3 full pages), extend_len=4 (1 page).

    The extend tokens are at positions 12,13,14,15. For tokens at later positions,
    there are more completed pages available for selection. This tests the per-token
    nature of the HSA extend pipeline.
    """
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    B = 1
    page_size = 4
    prefix_len = 12  # 3 full pages (LMKs at 3, 7, 11)
    extend_len = 4
    total_len = prefix_len + extend_len  # 16
    max_context_len = 32
    max_total_num_tokens = 256

    HQ_total, H_total, D = 4, 2, 16
    HQ_swa, HQ_hsa = 2, 2
    H_swa, H_hsa = 1, 1
    sm_scale = float(D) ** -0.5
    window_size = 4
    topk = 2

    model_runner = _make_model_runner_stub(
        B=B, page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
        max_context_len=max_context_len, max_total_num_tokens=max_total_num_tokens,
        topk=topk, window_size=window_size,
    )

    backend = HSAAttnBackend(model_runner)
    layer = RadixAttention(
        num_heads=HQ_total, head_dim=D, scaling=sm_scale,
        num_kv_heads=H_total, layer_id=0,
    )

    req_to_token = model_runner.req_to_token_pool.req_to_token

    # Allocate slot locations: 0..total_len-1
    token_locs = torch.arange(0, total_len, dtype=torch.int32, device=device)
    req_to_token[0, :total_len] = token_locs

    # Pre-fill prefix KV into cache.
    prefill_locs = token_locs[:prefix_len].to(torch.int64)
    cache_k = torch.randn((prefix_len, H_total, D), device=device, dtype=dtype)
    cache_v = torch.randn_like(cache_k)

    # Set deterministic LMK keys for HSA head (index 1).
    e0 = torch.zeros((D,), device=device, dtype=dtype); e0[0] = 10.0
    e1 = torch.zeros((D,), device=device, dtype=dtype); e1[1] = 10.0
    e2 = torch.zeros((D,), device=device, dtype=dtype); e2[2] = 10.0
    cache_k[3, 1, :] = e0   # page 0 LMK
    cache_k[7, 1, :] = e1   # page 1 LMK
    cache_k[11, 1, :] = e2  # page 2 LMK

    model_runner.token_to_kv_pool.set_kv_buffer(layer, prefill_locs, cache_k, cache_v)

    # Extend tokens: q, k, v for new tokens at positions [prefix_len, ..., prefix_len+extend_len-1].
    q_extend = torch.randn((extend_len, HQ_total * D), device=device, dtype=dtype)
    k_extend = torch.randn((extend_len, H_total * D), device=device, dtype=dtype)
    v_extend = torch.randn((extend_len, H_total * D), device=device, dtype=dtype)

    # Selection queries for extend tokens.
    sel_q = torch.randn((extend_len, HQ_hsa, D), device=device, dtype=dtype)

    # Positions for extend tokens.
    positions = torch.arange(prefix_len, total_len, device=device, dtype=torch.int64)

    # out_cache_loc for extend tokens.
    out_cache_loc = token_locs[prefix_len:total_len].to(torch.int64)

    extend_seq_lens = torch.tensor([extend_len], device=device, dtype=torch.int32)
    extend_prefix_lens = torch.tensor([prefix_len], device=device, dtype=torch.int32)
    extend_start_loc = torch.tensor([0], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([total_len], device=device, dtype=torch.int32)

    split_info = dict(
        hq_swa=HQ_swa, hq_hsa=HQ_hsa,
        h_swa=H_swa, h_hsa=H_hsa,
        swa_window_size=window_size,
        swa_exclude_lmk=False,
    )

    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=B,
        input_ids=torch.randint(0, 100, (extend_len,), device=device),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        seq_lens_sum=int(seq_lens.sum().item()),
        seq_lens_cpu=seq_lens.to("cpu"),
        attn_backend=backend,
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        extend_start_loc=extend_start_loc,
        extend_seq_lens_cpu=[extend_len],
        extend_prefix_lens_cpu=[prefix_len],
        positions=positions,
    )
    forward_batch.req_to_token_pool = model_runner.req_to_token_pool
    forward_batch.token_to_kv_pool = model_runner.token_to_kv_pool

    backend.init_forward_metadata(forward_batch)
    out_backend = backend.forward_extend(
        q_extend, k_extend.view(extend_len, H_total, D), v_extend.view(extend_len, H_total, D),
        layer, forward_batch,
        save_kv_cache=True,
        hsa_split_head_info=split_info,
        hsa_selection_q=sel_q,
    )
    out_backend_3 = out_backend.view(extend_len, HQ_total, D).to(torch.float32)

    # Now build the reference.
    # After save_kv_cache, all tokens 0..total_len-1 are in the paged cache.
    md = backend.forward_metadata
    k_cache_full = model_runner.token_to_kv_pool.get_key_buffer(layer.layer_id)
    v_cache_full = model_runner.token_to_kv_pool.get_value_buffer(layer.layer_id)
    page_table_1 = md.page_table_1
    token_positions = md.token_positions
    token_to_seq_id = md.token_to_seq_id

    q3 = q_extend.view(extend_len, HQ_total, D)

    # HSA heads reference.
    out_hsa_ref = _run_lhsa_extend_reference(
        q3=q3,
        sel_q=sel_q,
        k_cache=k_cache_full,
        v_cache=v_cache_full,
        page_table_1=page_table_1,
        token_positions=token_positions,
        token_to_seq_id=token_to_seq_id,
        HQ_swa=HQ_swa, HQ_hsa=HQ_hsa, H_swa=H_swa, H_hsa=H_hsa,
        window_size=window_size, page_size=page_size,
        topk=topk, sm_scale=sm_scale,
    )

    # Compare HSA heads only (SWA heads from dense backend are separately correct).
    out_hsa_backend = out_backend_3[:, HQ_swa:, :]
    out_hsa_ref_bf16 = out_hsa_ref.to(torch.bfloat16).to(torch.float32)

    max_abs = (out_hsa_backend - out_hsa_ref_bf16).abs().max().item()
    _vprint(f"### Extend single-seq (prefix={prefix_len}, extend={extend_len})")
    _vprint(f"  max_abs_err (HSA heads) = {max_abs}")

    torch.testing.assert_close(out_hsa_backend, out_hsa_ref_bf16, rtol=5e-2, atol=5e-2)


def test_hsa_backend_innerx_split_head_extend_multi_seq_cuda():
    """
    Multi-sequence extend (B=2) with different prefix/extend lengths.

    Req 0: prefix_len=8 (2 pages), extend_len=4
    Req 1: prefix_len=16 (4 pages), extend_len=4

    Verifies per-token HSA extend output matches reference for ragged batches.
    """
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(123)

    B = 2
    page_size = 4
    prefix_lens = [8, 16]
    extend_lens = [4, 4]
    total_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    total_extend_tokens = sum(extend_lens)
    max_context_len = 64
    max_total_num_tokens = 512

    HQ_total, H_total, D = 4, 2, 16
    HQ_swa, HQ_hsa = 2, 2
    H_swa, H_hsa = 1, 1
    sm_scale = float(D) ** -0.5
    window_size = 4
    topk = 2

    model_runner = _make_model_runner_stub(
        B=B, page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
        max_context_len=max_context_len, max_total_num_tokens=max_total_num_tokens,
        topk=topk, window_size=window_size,
    )

    backend = HSAAttnBackend(model_runner)
    layer = RadixAttention(
        num_heads=HQ_total, head_dim=D, scaling=sm_scale,
        num_kv_heads=H_total, layer_id=0,
    )

    req_to_token = model_runner.req_to_token_pool.req_to_token

    # Allocate non-overlapping KV slots.
    loc_offsets = [0, 128]
    for b in range(B):
        locs = torch.arange(
            loc_offsets[b], loc_offsets[b] + total_lens[b],
            dtype=torch.int32, device=device,
        )
        req_to_token[b, :total_lens[b]] = locs

    # Pre-fill prefix KV.
    e_vecs = []
    for i in range(D):
        e = torch.zeros((D,), device=device, dtype=dtype)
        e[i % D] = 10.0
        e_vecs.append(e)

    for b in range(B):
        prefill_locs = torch.arange(
            loc_offsets[b], loc_offsets[b] + prefix_lens[b],
            dtype=torch.int64, device=device,
        )
        cache_k = torch.randn((prefix_lens[b], H_total, D), device=device, dtype=dtype)
        cache_v = torch.randn((prefix_lens[b], H_total, D), device=device, dtype=dtype)
        # Set deterministic LMK keys for HSA kv head (index 1).
        for p in range(prefix_lens[b] // page_size):
            lmk_pos = p * page_size + (page_size - 1)
            cache_k[lmk_pos, 1, :] = e_vecs[p % D]
        model_runner.token_to_kv_pool.set_kv_buffer(layer, prefill_locs, cache_k, cache_v)

    # Build flat extend tensors.
    q_extend = torch.randn((total_extend_tokens, HQ_total * D), device=device, dtype=dtype)
    k_extend_flat = torch.randn((total_extend_tokens, H_total, D), device=device, dtype=dtype)
    v_extend_flat = torch.randn((total_extend_tokens, H_total, D), device=device, dtype=dtype)
    sel_q = torch.randn((total_extend_tokens, HQ_hsa, D), device=device, dtype=dtype)

    # Positions: for each seq, [prefix_len, prefix_len+1, ..., prefix_len+extend_len-1]
    positions = torch.cat([
        torch.arange(prefix_lens[b], total_lens[b], device=device, dtype=torch.int64)
        for b in range(B)
    ])

    # out_cache_loc: where to write extend KV.
    out_cache_loc = torch.cat([
        torch.arange(
            loc_offsets[b] + prefix_lens[b], loc_offsets[b] + total_lens[b],
            dtype=torch.int64, device=device,
        )
        for b in range(B)
    ])

    extend_seq_lens_t = torch.tensor(extend_lens, device=device, dtype=torch.int32)
    extend_prefix_lens_t = torch.tensor(prefix_lens, device=device, dtype=torch.int32)
    extend_start_loc = torch.zeros(B, device=device, dtype=torch.int32)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens_t[:-1], dim=0)
    seq_lens_t = torch.tensor(total_lens, device=device, dtype=torch.int32)

    split_info = dict(
        hq_swa=HQ_swa, hq_hsa=HQ_hsa,
        h_swa=H_swa, h_hsa=H_hsa,
        swa_window_size=window_size,
        swa_exclude_lmk=False,
    )

    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=B,
        input_ids=torch.randint(0, 100, (total_extend_tokens,), device=device),
        req_pool_indices=torch.arange(B, device=device, dtype=torch.int32),
        seq_lens=seq_lens_t,
        out_cache_loc=out_cache_loc,
        seq_lens_sum=int(seq_lens_t.sum().item()),
        seq_lens_cpu=seq_lens_t.to("cpu"),
        attn_backend=backend,
        extend_seq_lens=extend_seq_lens_t,
        extend_prefix_lens=extend_prefix_lens_t,
        extend_start_loc=extend_start_loc,
        extend_seq_lens_cpu=extend_lens,
        extend_prefix_lens_cpu=prefix_lens,
        positions=positions,
    )
    forward_batch.req_to_token_pool = model_runner.req_to_token_pool
    forward_batch.token_to_kv_pool = model_runner.token_to_kv_pool

    backend.init_forward_metadata(forward_batch)
    out_backend = backend.forward_extend(
        q_extend, k_extend_flat, v_extend_flat,
        layer, forward_batch,
        save_kv_cache=True,
        hsa_split_head_info=split_info,
        hsa_selection_q=sel_q,
    )
    out_backend_3 = out_backend.view(total_extend_tokens, HQ_total, D).to(torch.float32)

    # Build reference.
    md = backend.forward_metadata
    k_cache_full = model_runner.token_to_kv_pool.get_key_buffer(layer.layer_id)
    v_cache_full = model_runner.token_to_kv_pool.get_value_buffer(layer.layer_id)
    page_table_1 = md.page_table_1
    token_positions = md.token_positions
    token_to_seq_id = md.token_to_seq_id

    q3 = q_extend.view(total_extend_tokens, HQ_total, D)

    out_hsa_ref = _run_lhsa_extend_reference(
        q3=q3,
        sel_q=sel_q,
        k_cache=k_cache_full,
        v_cache=v_cache_full,
        page_table_1=page_table_1,
        token_positions=token_positions,
        token_to_seq_id=token_to_seq_id,
        HQ_swa=HQ_swa, HQ_hsa=HQ_hsa, H_swa=H_swa, H_hsa=H_hsa,
        window_size=window_size, page_size=page_size,
        topk=topk, sm_scale=sm_scale,
    )

    out_hsa_backend = out_backend_3[:, HQ_swa:, :]
    out_hsa_ref_bf16 = out_hsa_ref.to(torch.bfloat16).to(torch.float32)

    max_abs = (out_hsa_backend - out_hsa_ref_bf16).abs().max().item()
    _vprint(f"### Extend multi-seq B={B}")
    _vprint(f"  prefix_lens={prefix_lens}, extend_lens={extend_lens}")
    _vprint(f"  max_abs_err (HSA heads) = {max_abs}")

    # Per-sequence breakdown.
    offset = 0
    for b in range(B):
        ext_len = extend_lens[b]
        err_b = (out_hsa_backend[offset:offset+ext_len] - out_hsa_ref_bf16[offset:offset+ext_len]).abs().max().item()
        _vprint(f"  req{b}: prefix={prefix_lens[b]}, extend={ext_len}, max_abs_err={err_b}")
        offset += ext_len

    torch.testing.assert_close(out_hsa_backend, out_hsa_ref_bf16, rtol=5e-2, atol=5e-2)
