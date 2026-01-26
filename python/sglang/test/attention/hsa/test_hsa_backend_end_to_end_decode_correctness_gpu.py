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


def _torch_ref_hsa_decode_from_selected(
    *,
    q: torch.Tensor,  # [B,HQ,D] bf16
    k_cache: torch.Tensor,  # [Nloc,H,D] bf16
    v_cache: torch.Tensor,  # [Nloc,H,D] bf16
    page_table_1: torch.Tensor,  # [B,MAX_T] int32
    selected_page_ids: torch.Tensor,  # [B,H,K] int32
    selected_scores: torch.Tensor,  # [B,H,K] fp32
    page_size: int,
    sm_scale: float,
    mask_last_token: bool,
) -> torch.Tensor:
    """
    End-to-end torch reference:
    - Compute hsa_weights as softmax over selected_scores (mask invalid page_ids)
    - Expand weights from kv-head to q-head (G = HQ/H)
    - Paged attention per selected page with mask_last_token semantics
    """
    B, HQ, D = q.shape
    _, H, _ = k_cache.shape
    assert HQ % H == 0
    G = HQ // H
    K = int(selected_page_ids.shape[2])

    valid = selected_page_ids >= 0
    scores = selected_scores.masked_fill(~valid, float("-inf"))
    w_kv = torch.softmax(scores, dim=-1)
    w_kv = torch.nan_to_num(w_kv, nan=0.0).float()  # [B,H,K]
    w_q = w_kv[:, :, None, :].expand(B, H, G, K).reshape(B, HQ, K)  # [B,HQ,K]

    out = torch.zeros((B, HQ, D), device=q.device, dtype=torch.float32)
    for b in range(B):
        for hq in range(HQ):
            kv_h = hq // G
            qq = q[b, hq].float()
            for ki in range(K):
                pid = int(selected_page_ids[b, kv_h, ki].item())
                if pid < 0:
                    continue
                w = float(w_q[b, hq, ki].item())
                if w == 0.0:
                    continue
                token_start = pid * int(page_size)
                token_end = token_start + int(page_size)
                token_locs = page_table_1[b, token_start:token_end].to(torch.int64)
                k = k_cache[token_locs, kv_h].float()  # [S,D]
                v = v_cache[token_locs, kv_h].float()
                if mask_last_token:
                    k = k[:-1]
                    v = v[:-1]
                logits = (k @ qq) * float(sm_scale)  # [S']
                p = torch.softmax(logits, dim=0)
                out[b, hq] += w * (p @ v)
    return out


@pytest.mark.skipif(
    torch.cuda.device_count() == 0, reason="CUDA device required for this test"
)
def test_hsa_backend_end_to_end_decode_is_math_correct_cuda():
    """
    True end-to-end correctness test (GPU-only):
    - Real HSAAttnBackend + real Triton paged HSA decode kernel
    - Deterministic LMK-K setup so selection chooses known completed pages
    - Compare backend output vs torch reference that recomputes selection + weights + paged attention
    """
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.layers.attention.hsa.selector import (
        build_active_page_candidates,
        select_topk_pages_decode,
    )
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    device = "cuda"
    dtype = torch.bfloat16

    # Small shapes (but non-trivial: grouped heads)
    B = 1
    H = 2
    G = 2
    HQ = H * G
    D = 16
    page_size = 4
    topk = 1

    # Internal seq_len includes LMKs:
    # - pages 0 and 1 are completed (LMKs at token indices 3 and 7)
    # - page 2 is active but not completed; should be gated out
    prefill_len = 8
    decode_len = 1
    total_len = prefill_len + decode_len  # 9 => completed_pages = 2

    max_batch_size = 8
    max_context_len = 64
    max_total_num_tokens = max_batch_size * max_context_len

    # Minimal model_runner for TritonAttnBackend init under HSAAttnBackend.
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
            enable_swa_hsa_fusion=False,
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
        hsa_window_size=None,
        hsa_enable_swa_fusion=False,
        hsa_lmk_id=-1,
    )

    # Pools
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
        scaling=float(D) ** -0.5,
        num_kv_heads=H,
        layer_id=0,
    )

    # Token locs are real KV slots. Use contiguous slots so pages are well-defined:
    # page0: 0..3 (LMK at 3), page1: 4..7 (LMK at 7), page2 starts at 8 (incomplete).
    token_locs = torch.arange(0, total_len, dtype=torch.int32, device=device)
    model_runner.req_to_token_pool.req_to_token[0, :total_len] = token_locs

    # Prefill KV for first 8 tokens (includes LMKs at loc 3 and 7).
    prefill_loc = token_locs[:prefill_len].to(torch.int64)
    cache_k = torch.randn((prefill_len, H, D), device=device, dtype=dtype)
    cache_v = torch.randn_like(cache_k)

    # Make LMK-K deterministic to force selection:
    # - For kv head 0: page0 wins (lmk at 3 => +e0), page1 loses (lmk at 7 => +e1)
    # - For kv head 1: page1 wins (lmk at 7 => +e0), page0 loses (lmk at 3 => +e1)
    e0 = torch.zeros((D,), device=device, dtype=dtype)
    e0[0] = 10.0
    e1 = torch.zeros((D,), device=device, dtype=dtype)
    e1[1] = 10.0
    # loc=3 is LMK for page0
    cache_k[3, 0, :] = e0
    cache_k[3, 1, :] = e1
    # loc=7 is LMK for page1
    cache_k[7, 0, :] = e1
    cache_k[7, 1, :] = e0

    # Poison LMK V slots so masking is actually tested.
    cache_v[3, :, :] = 10000.0
    cache_v[7, :, :] = 10000.0

    model_runner.token_to_kv_pool.set_kv_buffer(layer, prefill_loc, cache_k, cache_v)

    # Decode: add 1 token at loc=8 (page2, incomplete).
    q3 = torch.zeros((B, HQ, D), device=device, dtype=dtype)
    # Make q align with desired winners under "head" strategy (max over group):
    # - kv head0 (hq 0,1) align to e0 => selects page0 for head0
    # - kv head1 (hq 2,3) align to e0 => selects page1 for head1 (because we set page1 lmk for head1 to e0)
    q3[0, 0, :] = e0
    q3[0, 1, :] = e0
    q3[0, 2, :] = e0
    q3[0, 3, :] = e0
    q = q3.reshape(B, HQ * D)

    k_new = torch.randn((B, H, D), device=device, dtype=dtype)
    v_new = torch.randn_like(k_new)
    out_cache_loc = token_locs[-1:].to(torch.int64)  # loc=8

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
    assert out_backend.shape == (B, HQ * D)

    md = backend.forward_metadata
    assert md is not None

    # --- Torch reference recomputes selection (same semantics) ---
    cand_page_ids, cand_mask = build_active_page_candidates(
        page_table_1=md.page_table_1,
        seq_lens=md.cache_seqlens_int32,
        page_size=page_size,
        window_size=None,
    )
    completed_pages = torch.div(
        md.cache_seqlens_int32.to(torch.int64), int(page_size), rounding_mode="floor"
    )
    cand_completed = cand_mask & (cand_page_ids.to(torch.int64) < completed_pages[:, None])
    safe_page_ids = cand_page_ids.clamp_min(0).to(torch.int64)
    lmk_locs = safe_page_ids * int(page_size) + (int(page_size) - 1)
    flat_lmk = lmk_locs.reshape(-1)
    k_cache_full = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
    flat_repr = k_cache_full[flat_lmk]  # [B*C, H, D]
    cand_repr = flat_repr.view(B, cand_page_ids.shape[1], H, D)

    sel = select_topk_pages_decode(
        q=q,
        cand_page_ids=cand_page_ids,
        cand_mask=cand_mask,
        cand_chunk_repr=cand_repr,
        cand_chunk_repr_valid=cand_completed,
        topk=topk,
        selection_strategy="head",
        sm_scale=layer.scaling,
    )

    # Sanity: backend and ref selection should match.
    torch.testing.assert_close(md.hsa_selected_page_ids, sel.selected_page_ids)

    # --- Torch reference output ---
    ref = _torch_ref_hsa_decode_from_selected(
        q=q3,
        k_cache=forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
        v_cache=forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
        page_table_1=md.page_table_1,
        selected_page_ids=sel.selected_page_ids,
        selected_scores=sel.selected_scores,
        page_size=page_size,
        sm_scale=layer.scaling,
        mask_last_token=True,
    )

    out_backend_3 = out_backend.view(B, HQ, D)
    torch.testing.assert_close(out_backend_3.float(), ref, rtol=3e-2, atol=3e-2)

    _vprint("### test_hsa_backend_end_to_end_decode_is_math_correct_cuda")
    _vprint(f"- total_len={total_len} page_size={page_size} completed_pages={total_len // page_size}")
    _vprint(f"- cand_page_ids={cand_page_ids[0].tolist()} cand_mask={cand_mask[0].tolist()}")
    _vprint(f"- selected_page_ids={sel.selected_page_ids[0].tolist()}")
    _vprint(f"- max_abs_err={(out_backend_3.float() - ref).abs().max().item()}")
    _vprint("=> Conclusion: HSAAttnBackend decode (selection + weights + Triton kernel) matches torch reference.")


