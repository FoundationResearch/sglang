"""
Benchmark: HSA extend/decode vs vanilla dense (SWA) extend/decode.

Measures wall-clock time (GPU-synchronized) for:
  1. HSA forward_extend  (optimized production path)
  2. HSA forward_extend  (reference Python-loop path)
  3. Dense forward_extend (TritonAttnBackend — vanilla SWA baseline)
  4. HSA forward_decode   (Triton kernel)
  5. Dense forward_decode  (TritonAttnBackend — vanilla SWA baseline)

Also breaks down HSA extend into its 3 sub-components:
  - _compute_internal_swa_extend  (batched vs reference)
  - _run_selection_extend         (batched vs reference)
  - _hsa_sparse_attn_extend       (Triton kernel vs reference)

Usage:
  python bench_hsa_vs_dense_extend_decode.py [--B 4] [--prefix 256] [--extend 64] [--page-size 64] [--warmup 5] [--iters 20]
"""

import argparse
import time
import types

import torch

from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1


def _make_model_runner_stub(
    *, B, page_size, HQ_total, H_total, D, max_context_len,
    max_total_num_tokens, topk, window_size, dtype=torch.bfloat16,
    attention_backend="hsa",
):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    device = torch.device("cuda")
    model_runner = types.SimpleNamespace()
    model_runner.device = device
    model_runner.page_size = page_size
    model_runner.sliding_window_size = window_size if attention_backend != "hsa" else None
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
        attention_backend=attention_backend,
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


def _setup_scenario(
    *, B, prefix_len, extend_len, page_size, HQ_total, H_total, D,
    topk, window_size, attention_backend="hsa",
):
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    device = torch.device("cuda")
    dtype = torch.bfloat16

    prefix_lens = [prefix_len] * B
    extend_lens = [extend_len] * B
    total_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    total_extend_tokens = sum(extend_lens)
    max_context_len = max(total_lens) + 64
    max_total_num_tokens = sum(total_lens) + 512

    HQ_swa = HQ_total // 2
    HQ_hsa = HQ_total - HQ_swa
    H_swa = H_total // 2
    H_hsa = H_total - H_swa
    sm_scale = float(D) ** -0.5

    model_runner = _make_model_runner_stub(
        B=B, page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
        max_context_len=max_context_len, max_total_num_tokens=max_total_num_tokens,
        topk=topk, window_size=window_size, attention_backend=attention_backend,
    )

    if attention_backend == "hsa":
        from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
        backend = HSAAttnBackend(model_runner)
    else:
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        backend = TritonAttnBackend(model_runner)

    layer = RadixAttention(
        num_heads=HQ_total, head_dim=D, scaling=sm_scale,
        num_kv_heads=H_total, layer_id=0,
        sliding_window_size=window_size if attention_backend != "hsa" else None,
    )

    req_to_token = model_runner.req_to_token_pool.req_to_token
    loc_offsets = [i * (max_context_len) for i in range(B)]

    for b in range(B):
        locs = torch.arange(
            loc_offsets[b], loc_offsets[b] + total_lens[b],
            dtype=torch.int32, device=device,
        )
        req_to_token[b, :total_lens[b]] = locs

    # Pre-fill prefix KV.
    for b in range(B):
        if prefix_lens[b] == 0:
            continue
        prefill_locs = torch.arange(
            loc_offsets[b], loc_offsets[b] + prefix_lens[b],
            dtype=torch.int64, device=device,
        )
        cache_k = torch.randn((prefix_lens[b], H_total, D), device=device, dtype=dtype)
        cache_v = torch.randn_like(cache_k)
        # Set LMK keys at page boundaries.
        for p in range(prefix_lens[b] // page_size):
            lmk_pos = p * page_size + (page_size - 1)
            cache_k[lmk_pos, H_swa, :] = 0
            cache_k[lmk_pos, H_swa, p % D] = 10.0
        model_runner.token_to_kv_pool.set_kv_buffer(layer, prefill_locs, cache_k, cache_v)

    # Extend tensors.
    q_extend = torch.randn((total_extend_tokens, HQ_total * D), device=device, dtype=dtype)
    k_extend = torch.randn((total_extend_tokens, H_total, D), device=device, dtype=dtype)
    v_extend = torch.randn_like(k_extend)
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

    # Decode batch (single token per seq).
    q_decode = torch.randn((B, HQ_total * D), device=device, dtype=dtype)
    k_decode = torch.randn((B, H_total * D), device=device, dtype=dtype)
    v_decode = torch.randn_like(k_decode)
    sel_q_decode = torch.randn((B, HQ_hsa, D), device=device, dtype=dtype)

    decode_seq_lens = torch.tensor(total_lens, device=device, dtype=torch.int32)
    decode_batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=B,
        input_ids=torch.randint(0, 100, (B,), device=device),
        req_pool_indices=torch.arange(B, device=device, dtype=torch.int32),
        seq_lens=decode_seq_lens,
        out_cache_loc=out_cache_loc[:B],  # one loc per seq
        seq_lens_sum=int(decode_seq_lens.sum().item()),
        seq_lens_cpu=decode_seq_lens.to("cpu"),
        attn_backend=backend,
    )
    decode_batch.req_to_token_pool = model_runner.req_to_token_pool
    decode_batch.token_to_kv_pool = model_runner.token_to_kv_pool

    return {
        "backend": backend,
        "layer": layer,
        "forward_batch": forward_batch,
        "decode_batch": decode_batch,
        "q_extend": q_extend,
        "k_extend": k_extend,
        "v_extend": v_extend,
        "q_decode": q_decode,
        "k_decode": k_decode,
        "v_decode": v_decode,
        "split_info": split_info,
        "sel_q": sel_q,
        "sel_q_decode": sel_q_decode,
        "HQ_swa": HQ_swa, "HQ_hsa": HQ_hsa,
        "H_swa": H_swa, "H_hsa": H_hsa,
    }


def _timed(fn, warmup=3, iters=10):
    """Run fn with CUDA sync, return mean ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1000
    return elapsed


def bench_extend(args):
    B = args.B
    prefix_len = args.prefix
    extend_len = args.extend
    page_size = args.page_size
    HQ_total = args.hq
    H_total = args.h
    D = args.d
    topk = args.topk
    window_size = args.window
    warmup = args.warmup
    iters = args.iters

    print(f"\n{'='*70}")
    print(f"EXTEND BENCHMARK")
    print(f"  B={B}, prefix={prefix_len}, extend={extend_len}, page_size={page_size}")
    print(f"  HQ={HQ_total}, H={H_total}, D={D}, topk={topk}, window={window_size}")
    print(f"  warmup={warmup}, iters={iters}")
    print(f"{'='*70}\n")

    # ---- HSA (optimized) ----
    torch.manual_seed(42)
    s = _setup_scenario(
        B=B, prefix_len=prefix_len, extend_len=extend_len,
        page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
        topk=topk, window_size=window_size, attention_backend="hsa",
    )
    be = s["backend"]
    be._USE_EXTEND_REFERENCE = False
    be.init_forward_metadata(s["forward_batch"])

    def run_hsa_extend_opt():
        be.init_forward_metadata(s["forward_batch"])
        be.forward_extend(
            s["q_extend"].clone(), s["k_extend"].clone(), s["v_extend"].clone(),
            s["layer"], s["forward_batch"], save_kv_cache=True,
            hsa_split_head_info=s["split_info"], hsa_selection_q=s["sel_q"],
        )

    ms_hsa_opt = _timed(run_hsa_extend_opt, warmup=warmup, iters=iters)

    # ---- HSA (reference) ----
    be._USE_EXTEND_REFERENCE = True
    def run_hsa_extend_ref():
        be.init_forward_metadata(s["forward_batch"])
        be.forward_extend(
            s["q_extend"].clone(), s["k_extend"].clone(), s["v_extend"].clone(),
            s["layer"], s["forward_batch"], save_kv_cache=True,
            hsa_split_head_info=s["split_info"], hsa_selection_q=s["sel_q"],
        )

    ms_hsa_ref = _timed(run_hsa_extend_ref, warmup=warmup, iters=iters)

    # ---- HSA sub-component breakdown (optimized) ----
    be._USE_EXTEND_REFERENCE = False
    be.init_forward_metadata(s["forward_batch"])
    # Pre-save KV for sub-component benchmarks.
    pool = s["forward_batch"].token_to_kv_pool
    pool.set_kv_buffer(s["layer"], s["forward_batch"].out_cache_loc,
                       s["k_extend"], s["v_extend"])
    md = be.forward_metadata
    page_table_1 = md.page_table_1
    T = B * extend_len
    q3 = s["q_extend"].view(T, HQ_total, D)
    HQ_swa, HQ_hsa, H_swa, H_hsa = s["HQ_swa"], s["HQ_hsa"], s["H_swa"], s["H_hsa"]
    q_hsa = q3[:, HQ_swa:, :]

    def run_swa_batched():
        be._compute_internal_swa_extend_batched(
            q_hsa=q_hsa, layer=s["layer"], forward_batch=s["forward_batch"],
            page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
            HQ_hsa=HQ_hsa, hsa_window=window_size,
        )

    def run_swa_ref():
        be._compute_internal_swa_extend_reference(
            q_hsa=q_hsa, layer=s["layer"], forward_batch=s["forward_batch"],
            page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
            HQ_hsa=HQ_hsa, hsa_window=window_size,
        )

    ms_swa_batched = _timed(run_swa_batched, warmup=warmup, iters=iters)
    ms_swa_ref = _timed(run_swa_ref, warmup=warmup, iters=iters)

    def run_sel_batched():
        be._run_selection_extend_batched(
            q=s["sel_q"], layer=s["layer"], forward_batch=s["forward_batch"],
            selection_q=s["sel_q"], page_table_1=page_table_1,
            kv_head_offset=H_swa, kv_head_count=H_hsa, hsa_window=window_size,
        )

    def run_sel_ref():
        be._run_selection_extend_reference(
            q=s["sel_q"], layer=s["layer"], forward_batch=s["forward_batch"],
            selection_q=s["sel_q"], page_table_1=page_table_1,
            kv_head_offset=H_swa, kv_head_count=H_hsa, hsa_window=window_size,
        )

    ms_sel_batched = _timed(run_sel_batched, warmup=warmup, iters=iters)
    ms_sel_ref = _timed(run_sel_ref, warmup=warmup, iters=iters)

    # Sparse attn needs selection results first.
    be._run_selection_extend_batched(
        q=s["sel_q"], layer=s["layer"], forward_batch=s["forward_batch"],
        selection_q=s["sel_q"], page_table_1=page_table_1,
        kv_head_offset=H_swa, kv_head_count=H_hsa, hsa_window=window_size,
    )
    selected_page_ids = md.hsa_ext_selected_page_ids
    selected_scores = md.hsa_ext_selected_scores
    swa_o, lse_kv = be._compute_internal_swa_extend_batched(
        q_hsa=q_hsa, layer=s["layer"], forward_batch=s["forward_batch"],
        page_table_1=page_table_1, H_swa=H_swa, H_hsa=H_hsa,
        HQ_hsa=HQ_hsa, hsa_window=window_size,
    )
    # Build weights.
    valid = selected_page_ids >= 0
    scores = selected_scores.masked_fill(~valid, float("-inf"))
    cat_scores = torch.cat([scores, lse_kv.unsqueeze(-1)], dim=-1)
    merged_w = torch.softmax(cat_scores, dim=-1)
    merged_w = torch.nan_to_num(merged_w, nan=0.0)
    TOPK = int(selected_page_ids.shape[2])
    w_kv = merged_w[:, :, :TOPK].to(q_hsa.dtype)
    Gh = HQ_hsa // H_hsa
    w_q = w_kv[:, :, None, :].expand(T, H_hsa, Gh, TOPK).reshape(T, HQ_hsa, TOPK).contiguous()

    k_cache_hsa = pool.get_key_buffer(s["layer"].layer_id)[:, H_swa:H_swa + H_hsa, :]
    v_cache_hsa = pool.get_value_buffer(s["layer"].layer_id)[:, H_swa:H_swa + H_hsa, :]

    def run_sparse_triton():
        be._hsa_sparse_attn_extend(
            q_hsa=q_hsa, k_cache=k_cache_hsa, v_cache=v_cache_hsa,
            page_table_1=page_table_1, selected_page_ids=selected_page_ids,
            hsa_weights=w_q, H_hsa=H_hsa, HQ_hsa=HQ_hsa,
            sm_scale=getattr(s["layer"], "scaling", None),
        )

    def run_sparse_ref():
        be._hsa_sparse_attn_extend_reference(
            q_hsa=q_hsa, k_cache=k_cache_hsa, v_cache=v_cache_hsa,
            page_table_1=page_table_1, selected_page_ids=selected_page_ids,
            hsa_weights=w_q, H_hsa=H_hsa, HQ_hsa=HQ_hsa,
            sm_scale=getattr(s["layer"], "scaling", None),
        )

    ms_sparse_triton = _timed(run_sparse_triton, warmup=warmup, iters=iters)
    ms_sparse_ref = _timed(run_sparse_ref, warmup=warmup, iters=iters)

    # ---- Dense (vanilla SWA) ----
    torch.manual_seed(42)
    sd = _setup_scenario(
        B=B, prefix_len=prefix_len, extend_len=extend_len,
        page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
        topk=topk, window_size=window_size, attention_backend="triton",
    )
    dense_be = sd["backend"]
    dense_be.init_forward_metadata(sd["forward_batch"])

    def run_dense_extend():
        dense_be.init_forward_metadata(sd["forward_batch"])
        dense_be.forward_extend(
            sd["q_extend"].clone(), sd["k_extend"].clone(), sd["v_extend"].clone(),
            sd["layer"], sd["forward_batch"], save_kv_cache=True,
        )

    ms_dense = _timed(run_dense_extend, warmup=warmup, iters=iters)

    # ---- Print results ----
    print(f"{'EXTEND RESULTS':^70}")
    print(f"{'-'*70}")
    print(f"{'Method':<40} {'Time (ms)':>10} {'Speedup':>10}")
    print(f"{'-'*70}")
    print(f"{'Dense (vanilla SWA) extend':<40} {ms_dense:>10.3f} {'baseline':>10}")
    print(f"{'HSA extend (optimized)':<40} {ms_hsa_opt:>10.3f} {ms_dense/ms_hsa_opt:>10.2f}x")
    print(f"{'HSA extend (reference loops)':<40} {ms_hsa_ref:>10.3f} {ms_dense/ms_hsa_ref:>10.2f}x")
    print()
    print(f"{'HSA Sub-component Breakdown':^70}")
    print(f"{'-'*70}")
    print(f"{'Component':<40} {'Optimized':>10} {'Reference':>10} {'Speedup':>10}")
    print(f"{'-'*70}")
    print(f"{'Internal SWA':<40} {ms_swa_batched:>10.3f} {ms_swa_ref:>10.3f} {ms_swa_ref/max(ms_swa_batched,0.001):>10.1f}x")
    print(f"{'TopK Selection':<40} {ms_sel_batched:>10.3f} {ms_sel_ref:>10.3f} {ms_sel_ref/max(ms_sel_batched,0.001):>10.1f}x")
    print(f"{'Sparse Attention':<40} {ms_sparse_triton:>10.3f} {ms_sparse_ref:>10.3f} {ms_sparse_ref/max(ms_sparse_triton,0.001):>10.1f}x")
    print()


def bench_decode(args):
    B = args.B
    prefix_len = args.prefix
    extend_len = args.extend
    page_size = args.page_size
    HQ_total = args.hq
    H_total = args.h
    D = args.d
    topk = args.topk
    window_size = args.window
    warmup = args.warmup
    iters = args.iters

    print(f"\n{'='*70}")
    print(f"DECODE BENCHMARK")
    print(f"  B={B}, seq_len={prefix_len + extend_len}, page_size={page_size}")
    print(f"  HQ={HQ_total}, H={H_total}, D={D}, topk={topk}, window={window_size}")
    print(f"  warmup={warmup}, iters={iters}")
    print(f"{'='*70}\n")

    # ---- HSA decode ----
    torch.manual_seed(42)
    s = _setup_scenario(
        B=B, prefix_len=prefix_len, extend_len=extend_len,
        page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
        topk=topk, window_size=window_size, attention_backend="hsa",
    )
    be = s["backend"]
    # First do extend to populate KV cache.
    be.init_forward_metadata(s["forward_batch"])
    be.forward_extend(
        s["q_extend"].clone(), s["k_extend"].clone(), s["v_extend"].clone(),
        s["layer"], s["forward_batch"], save_kv_cache=True,
        hsa_split_head_info=s["split_info"], hsa_selection_q=s["sel_q"],
    )

    # Now decode.
    be.init_forward_metadata(s["decode_batch"])

    def run_hsa_decode():
        be.init_forward_metadata(s["decode_batch"])
        be.forward_decode(
            s["q_decode"], s["k_decode"].view(B, H_total, D), s["v_decode"].view(B, H_total, D),
            s["layer"], s["decode_batch"], save_kv_cache=True,
            hsa_split_head_info=s["split_info"], hsa_selection_q=s["sel_q_decode"],
        )

    ms_hsa_decode = _timed(run_hsa_decode, warmup=warmup, iters=iters)

    # ---- Dense decode ----
    torch.manual_seed(42)
    sd = _setup_scenario(
        B=B, prefix_len=prefix_len, extend_len=extend_len,
        page_size=page_size, HQ_total=HQ_total, H_total=H_total, D=D,
        topk=topk, window_size=window_size, attention_backend="triton",
    )
    dense_be = sd["backend"]
    dense_be.init_forward_metadata(sd["forward_batch"])
    dense_be.forward_extend(
        sd["q_extend"].clone(), sd["k_extend"].clone(), sd["v_extend"].clone(),
        sd["layer"], sd["forward_batch"], save_kv_cache=True,
    )

    dense_be.init_forward_metadata(sd["decode_batch"])

    def run_dense_decode():
        dense_be.init_forward_metadata(sd["decode_batch"])
        dense_be.forward_decode(
            sd["q_decode"], sd["k_decode"].view(B, H_total, D), sd["v_decode"].view(B, H_total, D),
            sd["layer"], sd["decode_batch"], save_kv_cache=True,
        )

    ms_dense_decode = _timed(run_dense_decode, warmup=warmup, iters=iters)

    print(f"{'DECODE RESULTS':^70}")
    print(f"{'-'*70}")
    print(f"{'Method':<40} {'Time (ms)':>10} {'Speedup':>10}")
    print(f"{'-'*70}")
    print(f"{'Dense (vanilla SWA) decode':<40} {ms_dense_decode:>10.3f} {'baseline':>10}")
    print(f"{'HSA decode':<40} {ms_hsa_decode:>10.3f} {ms_dense_decode/ms_hsa_decode:>10.2f}x")
    print()


def main():
    parser = argparse.ArgumentParser(description="HSA vs Dense attention benchmark")
    parser.add_argument("--B", type=int, default=4, help="Batch size")
    parser.add_argument("--prefix", type=int, default=256, help="Prefix (cached) length per seq")
    parser.add_argument("--extend", type=int, default=64, help="Extend (new) tokens per seq")
    parser.add_argument("--page-size", type=int, default=64, help="Page/chunk size")
    parser.add_argument("--hq", type=int, default=32, help="Num Q heads")
    parser.add_argument("--h", type=int, default=8, help="Num KV heads")
    parser.add_argument("--d", type=int, default=128, help="Head dim")
    parser.add_argument("--topk", type=int, default=4, help="HSA top-k pages")
    parser.add_argument("--window", type=int, default=64, help="SWA window size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument("--extend-only", action="store_true", help="Only run extend benchmark")
    parser.add_argument("--decode-only", action="store_true", help="Only run decode benchmark")
    args = parser.parse_args()

    if not args.decode_only:
        bench_extend(args)
    if not args.extend_only:
        bench_decode(args)


if __name__ == "__main__":
    main()
