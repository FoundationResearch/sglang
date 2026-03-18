"""
Tests comparing optimized (production) HSA extend implementations against
their Python reference counterparts.

Each of the 3 optimized methods is tested independently:
  1. _compute_internal_swa_extend_batched vs _compute_internal_swa_extend_reference
  2. _run_selection_extend_batched vs _run_selection_extend_reference
  3. _hsa_sparse_attn_extend (Triton kernel) vs _hsa_sparse_attn_extend_reference

Additionally, the full forward_extend pipeline is tested end-to-end comparing
optimized vs reference output.

Edge cases tested: zero candidates, fewer candidates than topk, single-token
extend, mixed-length batch, large window covering all pages.
"""

import os
import types

import pytest
import torch

from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)


# ---- Shared helpers ----


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


def _setup_extend_scenario(
    *, B, prefix_lens, extend_lens, page_size, HQ_total, H_total, D,
    topk, window_size, seed=42,
):
    """Set up a full extend scenario and return (backend, layer, forward_batch, split_info, sel_q)."""
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(seed)

    total_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    total_extend_tokens = sum(extend_lens)
    max_context_len = max(total_lens) + 32
    max_total_num_tokens = sum(total_lens) + 256

    HQ_swa = HQ_total // 2
    HQ_hsa = HQ_total - HQ_swa
    H_swa = H_total // 2
    H_hsa = H_total - H_swa
    sm_scale = float(D) ** -0.5

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
    loc_offsets = [sum(total_lens[:i]) for i in range(B)]
    for b in range(B):
        locs = torch.arange(
            loc_offsets[b], loc_offsets[b] + total_lens[b],
            dtype=torch.int32, device=device,
        )
        req_to_token[b, :total_lens[b]] = locs

    # Pre-fill prefix KV with deterministic LMK keys.
    e_vecs = []
    for i in range(D):
        e = torch.zeros((D,), device=device, dtype=dtype)
        e[i % D] = 10.0
        e_vecs.append(e)

    for b in range(B):
        if prefix_lens[b] == 0:
            continue
        prefill_locs = torch.arange(
            loc_offsets[b], loc_offsets[b] + prefix_lens[b],
            dtype=torch.int64, device=device,
        )
        cache_k = torch.randn((prefix_lens[b], H_total, D), device=device, dtype=dtype)
        cache_v = torch.randn((prefix_lens[b], H_total, D), device=device, dtype=dtype)
        for p in range(prefix_lens[b] // page_size):
            lmk_pos = p * page_size + (page_size - 1)
            cache_k[lmk_pos, H_swa, :] = e_vecs[p % D]
        model_runner.token_to_kv_pool.set_kv_buffer(layer, prefill_locs, cache_k, cache_v)

    # Extend tokens.
    q_extend = torch.randn((total_extend_tokens, HQ_total * D), device=device, dtype=dtype)
    k_extend = torch.randn((total_extend_tokens, H_total, D), device=device, dtype=dtype)
    v_extend = torch.randn((total_extend_tokens, H_total, D), device=device, dtype=dtype)
    sel_q = torch.randn((total_extend_tokens, HQ_hsa, D), device=device, dtype=dtype)

    positions = torch.cat([
        torch.arange(prefix_lens[b], total_lens[b], device=device, dtype=torch.int64)
        for b in range(B)
    ])
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
    if B > 1:
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

    return (backend, layer, forward_batch, q_extend, k_extend, v_extend,
            split_info, sel_q, HQ_swa, HQ_hsa, H_swa, H_hsa)


# ---- Tests ----


class TestInternalSWAExtendBatchedVsReference:
    """Compare _compute_internal_swa_extend_batched vs _reference."""

    def _run_both(self, *, B, prefix_lens, extend_lens, page_size=4, window_size=4):
        HQ_total, H_total, D = 4, 2, 16
        topk = 2

        (backend, layer, forward_batch, q_extend, k_extend, v_extend,
         split_info, sel_q, HQ_swa, HQ_hsa, H_swa, H_hsa) = _setup_extend_scenario(
            B=B, prefix_lens=prefix_lens, extend_lens=extend_lens,
            page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
            topk=topk, window_size=window_size,
        )

        pool = forward_batch.token_to_kv_pool
        # Save extend KV so cache is populated.
        pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k_extend, v_extend)

        backend.init_forward_metadata(forward_batch)
        md = backend.forward_metadata
        page_table_1 = md.page_table_1

        T = sum(extend_lens)
        q3 = q_extend.view(T, HQ_total, D)
        q_hsa = q3[:, HQ_swa:, :]

        # Batched.
        swa_o_batch, lse_batch = backend._compute_internal_swa_extend_batched(
            q_hsa=q_hsa, layer=layer, forward_batch=forward_batch,
            page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
            HQ_hsa=HQ_hsa, hsa_window=window_size,
        )

        # Reference.
        swa_o_ref, lse_ref = backend._compute_internal_swa_extend_reference(
            q_hsa=q_hsa, layer=layer, forward_batch=forward_batch,
            page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
            HQ_hsa=HQ_hsa, hsa_window=window_size,
        )

        return swa_o_batch, lse_batch, swa_o_ref, lse_ref

    def test_single_seq(self):
        swa_o_b, lse_b, swa_o_r, lse_r = self._run_both(
            B=1, prefix_lens=[12], extend_lens=[4],
        )
        torch.testing.assert_close(swa_o_b, swa_o_r, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lse_b, lse_r, rtol=1e-4, atol=1e-4)

    def test_multi_seq(self):
        swa_o_b, lse_b, swa_o_r, lse_r = self._run_both(
            B=2, prefix_lens=[8, 16], extend_lens=[4, 4],
        )
        torch.testing.assert_close(swa_o_b, swa_o_r, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lse_b, lse_r, rtol=1e-4, atol=1e-4)

    def test_single_token_extend(self):
        """Degenerate case: extend_len=1 per sequence."""
        swa_o_b, lse_b, swa_o_r, lse_r = self._run_both(
            B=1, prefix_lens=[12], extend_lens=[1],
        )
        torch.testing.assert_close(swa_o_b, swa_o_r, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lse_b, lse_r, rtol=1e-4, atol=1e-4)

    def test_large_window_covers_all(self):
        """Window larger than sequence — all pages in window, no candidates outside."""
        swa_o_b, lse_b, swa_o_r, lse_r = self._run_both(
            B=1, prefix_lens=[12], extend_lens=[4], window_size=100,
        )
        torch.testing.assert_close(swa_o_b, swa_o_r, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lse_b, lse_r, rtol=1e-4, atol=1e-4)

    def test_zero_window(self):
        """hsa_window=0: should return zeros/neg-inf."""
        swa_o_b, lse_b, swa_o_r, lse_r = self._run_both(
            B=1, prefix_lens=[12], extend_lens=[4], window_size=0,
        )
        torch.testing.assert_close(swa_o_b, swa_o_r, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lse_b, lse_r, rtol=1e-4, atol=1e-4)


class TestSelectionExtendBatchedVsReference:
    """Compare _run_selection_extend_batched vs _reference."""

    def _run_both(self, *, B, prefix_lens, extend_lens, page_size=4,
                  window_size=4, topk=2):
        HQ_total, H_total, D = 4, 2, 16

        (backend, layer, forward_batch, q_extend, k_extend, v_extend,
         split_info, sel_q, HQ_swa, HQ_hsa, H_swa, H_hsa) = _setup_extend_scenario(
            B=B, prefix_lens=prefix_lens, extend_lens=extend_lens,
            page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
            topk=topk, window_size=window_size,
        )

        pool = forward_batch.token_to_kv_pool
        pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k_extend, v_extend)

        backend.init_forward_metadata(forward_batch)
        md = backend.forward_metadata
        page_table_1 = md.page_table_1

        # Batched.
        backend._run_selection_extend_batched(
            q=sel_q, layer=layer, forward_batch=forward_batch,
            selection_q=sel_q, page_table_1=page_table_1,
            kv_head_offset=H_swa, kv_head_count=H_hsa,
            hsa_window=window_size,
        )
        page_ids_batch = md.hsa_ext_selected_page_ids.clone()
        scores_batch = md.hsa_ext_selected_scores.clone()

        # Reference.
        backend._run_selection_extend_reference(
            q=sel_q, layer=layer, forward_batch=forward_batch,
            selection_q=sel_q, page_table_1=page_table_1,
            kv_head_offset=H_swa, kv_head_count=H_hsa,
            hsa_window=window_size,
        )
        page_ids_ref = md.hsa_ext_selected_page_ids.clone()
        scores_ref = md.hsa_ext_selected_scores.clone()

        return page_ids_batch, scores_batch, page_ids_ref, scores_ref

    def test_single_seq(self):
        pi_b, sc_b, pi_r, sc_r = self._run_both(
            B=1, prefix_lens=[12], extend_lens=[4],
        )
        torch.testing.assert_close(pi_b, pi_r)
        torch.testing.assert_close(sc_b, sc_r, rtol=1e-4, atol=1e-4)

    def test_multi_seq(self):
        pi_b, sc_b, pi_r, sc_r = self._run_both(
            B=2, prefix_lens=[8, 16], extend_lens=[4, 4],
        )
        torch.testing.assert_close(pi_b, pi_r)
        torch.testing.assert_close(sc_b, sc_r, rtol=1e-4, atol=1e-4)

    def test_zero_candidates(self):
        """Early position tokens have no completed pages."""
        pi_b, sc_b, pi_r, sc_r = self._run_both(
            B=1, prefix_lens=[0], extend_lens=[4], page_size=4,
        )
        torch.testing.assert_close(pi_b, pi_r)
        # All should be -1 / -inf.
        assert (pi_b == -1).all()

    def test_fewer_candidates_than_topk(self):
        """Only 1 completed page but topk=2."""
        pi_b, sc_b, pi_r, sc_r = self._run_both(
            B=1, prefix_lens=[8], extend_lens=[4], page_size=4,
            window_size=4, topk=3,
        )
        torch.testing.assert_close(pi_b, pi_r)
        torch.testing.assert_close(sc_b, sc_r, rtol=1e-4, atol=1e-4)

    def test_large_window_no_candidates(self):
        """Window covers everything — no candidates outside window."""
        pi_b, sc_b, pi_r, sc_r = self._run_both(
            B=1, prefix_lens=[12], extend_lens=[4], window_size=100,
        )
        torch.testing.assert_close(pi_b, pi_r)
        assert (pi_b == -1).all()


class TestSparseAttnExtendTritonVsReference:
    """Compare _hsa_sparse_attn_extend (Triton kernel) vs _reference."""

    def _run_both(self, *, B, prefix_lens, extend_lens, page_size=4,
                  window_size=4, topk=2):
        HQ_total, H_total, D = 4, 2, 16

        (backend, layer, forward_batch, q_extend, k_extend, v_extend,
         split_info, sel_q, HQ_swa, HQ_hsa, H_swa, H_hsa) = _setup_extend_scenario(
            B=B, prefix_lens=prefix_lens, extend_lens=extend_lens,
            page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
            topk=topk, window_size=window_size,
        )

        pool = forward_batch.token_to_kv_pool
        pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k_extend, v_extend)

        backend.init_forward_metadata(forward_batch)
        md = backend.forward_metadata
        page_table_1 = md.page_table_1

        T = sum(extend_lens)
        q3 = q_extend.view(T, HQ_total, D)
        q_hsa = q3[:, HQ_swa:, :]

        # Run selection first to get page_ids and weights.
        backend._run_selection_extend_reference(
            q=sel_q, layer=layer, forward_batch=forward_batch,
            selection_q=sel_q, page_table_1=page_table_1,
            kv_head_offset=H_swa, kv_head_count=H_hsa,
            hsa_window=window_size,
        )
        selected_page_ids = md.hsa_ext_selected_page_ids
        selected_scores = md.hsa_ext_selected_scores

        # Build merged weights (same as forward_extend does).
        swa_o_inner, lse_kv = backend._compute_internal_swa_extend_reference(
            q_hsa=q_hsa, layer=layer, forward_batch=forward_batch,
            page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
            HQ_hsa=HQ_hsa, hsa_window=window_size,
        )

        valid = selected_page_ids >= 0
        scores = selected_scores.masked_fill(~valid, float("-inf"))
        cat_scores = torch.cat([scores, lse_kv.unsqueeze(-1)], dim=-1)
        merged_w = torch.softmax(cat_scores, dim=-1)
        merged_w = torch.nan_to_num(merged_w, nan=0.0)
        TOPK = int(selected_page_ids.shape[2])
        w_kv = merged_w[:, :, :TOPK].to(q_hsa.dtype)

        Gh = HQ_hsa // H_hsa
        w_q = (
            w_kv[:, :, None, :]
            .expand(T, H_hsa, Gh, TOPK)
            .reshape(T, HQ_hsa, TOPK)
            .contiguous()
        )

        k_cache_hsa = pool.get_key_buffer(layer.layer_id)[:, H_swa:H_swa + H_hsa, :]
        v_cache_hsa = pool.get_value_buffer(layer.layer_id)[:, H_swa:H_swa + H_hsa, :]

        # Triton kernel (production).
        out_triton = backend._hsa_sparse_attn_extend(
            q_hsa=q_hsa, k_cache=k_cache_hsa, v_cache=v_cache_hsa,
            page_table_1=page_table_1, selected_page_ids=selected_page_ids,
            hsa_weights=w_q, H_hsa=H_hsa, HQ_hsa=HQ_hsa,
            sm_scale=getattr(layer, "scaling", None),
        )

        # Reference (Python loop).
        out_ref = backend._hsa_sparse_attn_extend_reference(
            q_hsa=q_hsa, k_cache=k_cache_hsa, v_cache=v_cache_hsa,
            page_table_1=page_table_1, selected_page_ids=selected_page_ids,
            hsa_weights=w_q, H_hsa=H_hsa, HQ_hsa=HQ_hsa,
            sm_scale=getattr(layer, "scaling", None),
        )

        return out_triton, out_ref

    def test_single_seq(self):
        out_t, out_r = self._run_both(B=1, prefix_lens=[12], extend_lens=[4])
        # Triton returns bf16, reference returns bf16 via the original code.
        torch.testing.assert_close(
            out_t.to(torch.float32), out_r.to(torch.float32),
            rtol=5e-2, atol=5e-2,
        )

    def test_multi_seq(self):
        out_t, out_r = self._run_both(B=2, prefix_lens=[8, 16], extend_lens=[4, 4])
        torch.testing.assert_close(
            out_t.to(torch.float32), out_r.to(torch.float32),
            rtol=5e-2, atol=5e-2,
        )

    def test_single_token_extend(self):
        out_t, out_r = self._run_both(B=1, prefix_lens=[12], extend_lens=[1])
        torch.testing.assert_close(
            out_t.to(torch.float32), out_r.to(torch.float32),
            rtol=5e-2, atol=5e-2,
        )

    def test_mixed_lengths(self):
        """Ragged batch with very different sequence lengths."""
        out_t, out_r = self._run_both(
            B=3, prefix_lens=[4, 12, 20], extend_lens=[4, 4, 4], page_size=4,
        )
        torch.testing.assert_close(
            out_t.to(torch.float32), out_r.to(torch.float32),
            rtol=5e-2, atol=5e-2,
        )


class TestFullForwardExtendOptimizedVsReference:
    """End-to-end: run forward_extend with optimized path, then with reference,
    and compare HSA head outputs."""

    def _run_e2e(self, *, B, prefix_lens, extend_lens, page_size=4,
                 window_size=4, topk=2):
        HQ_total, H_total, D = 4, 2, 16

        (backend, layer, forward_batch, q_extend, k_extend, v_extend,
         split_info, sel_q, HQ_swa, HQ_hsa, H_swa, H_hsa) = _setup_extend_scenario(
            B=B, prefix_lens=prefix_lens, extend_lens=extend_lens,
            page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
            topk=topk, window_size=window_size,
        )

        T = sum(extend_lens)

        # Run optimized path.
        backend._USE_EXTEND_REFERENCE = False
        backend.init_forward_metadata(forward_batch)
        out_opt = backend.forward_extend(
            q_extend.clone(), k_extend.clone(), v_extend.clone(),
            layer, forward_batch,
            save_kv_cache=True,
            hsa_split_head_info=split_info,
            hsa_selection_q=sel_q,
        )
        out_opt_hsa = out_opt.view(T, HQ_total, D)[:, HQ_swa:, :].to(torch.float32)

        # Reset cache: re-fill prefix KV.
        # We need to re-create the scenario since KV cache was modified.
        (backend2, layer2, forward_batch2, q_extend2, k_extend2, v_extend2,
         split_info2, sel_q2, _, _, _, _) = _setup_extend_scenario(
            B=B, prefix_lens=prefix_lens, extend_lens=extend_lens,
            page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
            topk=topk, window_size=window_size,
        )

        # Run reference path.
        backend2._USE_EXTEND_REFERENCE = True
        backend2.init_forward_metadata(forward_batch2)
        out_ref = backend2.forward_extend(
            q_extend2.clone(), k_extend2.clone(), v_extend2.clone(),
            layer2, forward_batch2,
            save_kv_cache=True,
            hsa_split_head_info=split_info2,
            hsa_selection_q=sel_q2,
        )
        out_ref_hsa = out_ref.view(T, HQ_total, D)[:, HQ_swa:, :].to(torch.float32)

        return out_opt_hsa, out_ref_hsa

    def test_single_seq_e2e(self):
        out_opt, out_ref = self._run_e2e(B=1, prefix_lens=[12], extend_lens=[4])
        torch.testing.assert_close(out_opt, out_ref, rtol=5e-2, atol=5e-2)

    def test_multi_seq_e2e(self):
        out_opt, out_ref = self._run_e2e(B=2, prefix_lens=[8, 16], extend_lens=[4, 4])
        torch.testing.assert_close(out_opt, out_ref, rtol=5e-2, atol=5e-2)

    def test_single_token_e2e(self):
        out_opt, out_ref = self._run_e2e(B=1, prefix_lens=[12], extend_lens=[1])
        torch.testing.assert_close(out_opt, out_ref, rtol=5e-2, atol=5e-2)

    def test_mixed_batch_e2e(self):
        out_opt, out_ref = self._run_e2e(
            B=3, prefix_lens=[4, 12, 20], extend_lens=[4, 4, 4],
        )
        torch.testing.assert_close(out_opt, out_ref, rtol=5e-2, atol=5e-2)
