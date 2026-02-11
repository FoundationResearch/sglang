import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)

_HSA_VERBOSE = os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") not in ("", "0", "false", "False")


def _vprint(*args):
    if _HSA_VERBOSE:
        print(*args, flush=True)


def test_innerx_radix_prefix_kv_reuse_runner_like_end2end_cuda():
    """
    End-to-end (runner-like) test that exercises a *radix prefix-cache reuse* scenario:

    - Req1: run EXTEND (prefill) to write KV for the full prompt.
    - Cache the request (radix cache).
    - Req2: same prompt, gets a page-aligned prefix hit (including LMK tokens).
      We *do not* recompute that prefix; we only EXTEND the uncached tail tokens.
    - Then run multiple DECODE steps for Req1+Req2 together with teacher-forced identical
      next token ids, and assert:
        (a) attention-module outputs for the two reqs match step-by-step
        (b) shared prefix KV slots are not overwritten (true reuse)

    This test is especially important for FlashHSA (InnerX) because LMK insertion + landmark
    positions + paged KV + selection are all sensitive to exact KV indexing.
    """
    from sglang.srt.configs.flash_hsa import FlashHSAConfig
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers import dp_attention as dpa
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        ForwardMode,
        compute_decode_positions_landmark,
        compute_position,
    )
    from sglang.srt.models.flash_hsa import HSAForCausalLM
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # --- Minimal single-process distributed + model-parallel init (runner-like) ---
    if not dist.is_initialized():
        _, path = tempfile.mkstemp(prefix="sglang_dist_", suffix=".tmp")
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{path}",
            rank=0,
            world_size=1,
        )
    if not ps.model_parallel_is_initialized():
        ps._WORLD = ps.init_world_group(ranks=[0], local_rank=0, backend="gloo")  # type: ignore[attr-defined]
        ps._TP = ps.init_model_parallel_group(  # type: ignore[attr-defined]
            group_ranks=[[0]],
            local_rank=0,
            backend="gloo",
            use_custom_allreduce=False,
            use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False,
            group_name="tp",
        )
        ps._PP = ps.init_model_parallel_group(  # type: ignore[attr-defined]
            group_ranks=[[0]],
            local_rank=0,
            backend="gloo",
            use_custom_allreduce=False,
            use_mscclpp_allreduce=False,
            use_torch_symm_mem_allreduce=False,
            group_name="pp",
        )

    # Minimal DP-attention globals (runner would call initialize_dp_attention()).
    if getattr(dpa, "_ATTN_TP_RANK", None) is None:
        dpa._ATTN_TP_RANK = 0  # type: ignore[attr-defined]
        dpa._ATTN_TP_SIZE = 1  # type: ignore[attr-defined]
        dpa._ATTN_DP_RANK = 0  # type: ignore[attr-defined]
        dpa._ATTN_DP_SIZE = 1  # type: ignore[attr-defined]
        dpa._LOCAL_ATTN_DP_RANK = 0  # type: ignore[attr-defined]
        dpa._LOCAL_ATTN_DP_SIZE = 1  # type: ignore[attr-defined]
        dpa._ENABLE_DP_ATTENTION_FLAG = False  # type: ignore[attr-defined]
        dpa._ATTN_TP_GROUP = ps.get_tp_group()  # type: ignore[attr-defined]

    # Minimal global ServerArgs (some layers read this).
    sa = ServerArgs(model_path="dummy")
    sa.attention_backend = "hsa"
    sa.enable_dp_lm_head = False
    set_global_server_args_for_scheduler(sa)

    # --- InnerX-ish tiny config; head_dim must be >=16 for Triton extend kernel ---
    page_size = 4
    cfg = FlashHSAConfig(
        model_type="flash_hsa_innerx",
        architectures=["HSAForCausalLM"],
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=16,
        rms_norm_eps=1e-6,
        attention_bias=False,
        chunk_size=page_size,
        hsa_topk=2,
        hsa_mode="sparse",
        full_attn_interleave=1,
        hsa_heads=4,
        hsa_qk_ratio=4,
        enable_gate=False,
        use_sliding_window_merging=True,
        sliding_window_merging_size=page_size,
        use_sliding_window_attention=False,
        sliding_window_attention_size=None,
        tie_word_embeddings=False,
    )

    model = HSAForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()

    # --- Pools + radix cache (this is the "radix attention scenario") ---
    max_context_len = 64
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=max_context_len, device=device, enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=4096,
        page_size=page_size,
        dtype=dtype,
        head_num=int(cfg.num_key_value_heads),
        head_dim=int(cfg.head_dim),
        layer_num=int(cfg.num_hidden_layers),
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=4096, dtype=torch.int64, device=device, kvcache=kv_pool, need_sort=False
    )
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            enable_kv_cache_events=False,
        )
    )

    lmk_id = int(cfg.vocab_size)
    origin = [5, 6, 7, 8, 9, 10, 11]  # 7 real tokens => fill_ids length 9 with page_size=4

    # Req1: full prefill.
    req1 = Req("r1", "", origin, SamplingParams(max_new_tokens=16))
    req1.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)
    req1.req_pool_idx = req_to_token_pool.alloc(1)[0]
    req1.init_next_round_input(cache)
    fill1 = list(req1.fill_ids)
    prompt_len = len(fill1)
    assert prompt_len > 0

    kv_slots1 = allocator.alloc(prompt_len)
    assert kv_slots1 is not None
    req_to_token_pool.req_to_token[req1.req_pool_idx, :prompt_len] = kv_slots1.to(torch.int32)

    # Runner-like model_runner stub.
    model_runner = types.SimpleNamespace()
    model_runner.device = device
    model_runner.page_size = page_size
    model_runner.sliding_window_size = None
    model_runner.model = model
    model_runner.model_config = types.SimpleNamespace(
        is_encoder_decoder=False,
        context_len=max_context_len,
        num_attention_heads=int(cfg.num_attention_heads),
        get_num_kv_heads=lambda tp_size: int(cfg.num_key_value_heads) // int(tp_size),
    )
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
        hsa_topk=None,
        hsa_selection_strategy=None,
        hsa_layers=None,
        hsa_window_size=None,
        hsa_enable_swa_merging=None,
        hsa_lmk_id=lmk_id,
    )
    model_runner.req_to_token_pool = req_to_token_pool
    model_runner.token_to_kv_pool = kv_pool
    model_runner.token_to_kv_pool_allocator = allocator

    backend = HSAAttnBackend(model_runner)

    # Prefill Req1.
    ext_prefix = torch.tensor([0], device=device, dtype=torch.int32)
    ext_len = torch.tensor([prompt_len], device=device, dtype=torch.int32)
    pos1, start1 = compute_position(
        attn_backend="hsa",
        extend_prefix_lens=ext_prefix,
        extend_seq_lens=ext_len,
        extend_seq_lens_sum=int(prompt_len),
        page_size=page_size,
        enable_landmark_positions=True,
    )
    fb1 = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor(fill1, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([req1.req_pool_idx], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([prompt_len], device=device, dtype=torch.int32),
        out_cache_loc=kv_slots1.to(torch.int64),
        seq_lens_sum=int(prompt_len),
        seq_lens_cpu=torch.tensor([prompt_len], device="cpu", dtype=torch.int32),
        positions=pos1,
        extend_prefix_lens=ext_prefix,
        extend_seq_lens=ext_len,
        extend_start_loc=start1,
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[prompt_len],
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=kv_pool,
        attn_backend=backend,
    )
    backend.init_forward_metadata(fb1)
    _ = model.model(fb1.input_ids, fb1.positions, fb1)

    # Cache unfinished Req1 (radix prefix-cache).
    cache.cache_unfinished_req(req1)

    # Req2: prefix hit and tail extend only.
    req2 = Req("r2", "", origin, SamplingParams(max_new_tokens=16))
    req2.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)
    req2.req_pool_idx = req_to_token_pool.alloc(1)[0]
    req2.init_next_round_input(cache)
    fill2 = list(req2.fill_ids)
    assert fill2 == fill1  # same prompt
    prefix_slots = req2.prefix_indices.to(torch.int64)
    prefix_len = int(prefix_slots.numel())
    assert prefix_len % page_size == 0 and prefix_len > 0
    tail = fill2[prefix_len:]
    tail_len = len(tail)
    assert tail_len >= 0

    req_to_token_pool.req_to_token[req2.req_pool_idx, :prefix_len] = prefix_slots.to(torch.int32)
    if tail_len > 0:
        kv_tail2 = allocator.alloc(tail_len)
        assert kv_tail2 is not None
        req_to_token_pool.req_to_token[
            req2.req_pool_idx, prefix_len : prefix_len + tail_len
        ] = kv_tail2.to(torch.int32)
    else:
        kv_tail2 = torch.empty((0,), device=device, dtype=torch.int64)

    if tail_len > 0:
        ext_prefix2 = torch.tensor([prefix_len], device=device, dtype=torch.int32)
        ext_len2 = torch.tensor([tail_len], device=device, dtype=torch.int32)
        pos2, start2 = compute_position(
            attn_backend="hsa",
            extend_prefix_lens=ext_prefix2,
            extend_seq_lens=ext_len2,
            extend_seq_lens_sum=int(tail_len),
            page_size=page_size,
            enable_landmark_positions=True,
        )
        fb2 = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.tensor(tail, device=device, dtype=torch.int64),
            req_pool_indices=torch.tensor([req2.req_pool_idx], device=device, dtype=torch.int32),
            seq_lens=torch.tensor([prefix_len + tail_len], device=device, dtype=torch.int32),
            out_cache_loc=kv_tail2.to(torch.int64),
            seq_lens_sum=int(prefix_len + tail_len),
            seq_lens_cpu=torch.tensor([prefix_len + tail_len], device="cpu", dtype=torch.int32),
            positions=pos2,
            extend_prefix_lens=ext_prefix2,
            extend_seq_lens=ext_len2,
            extend_start_loc=start2,
            extend_prefix_lens_cpu=[prefix_len],
            extend_seq_lens_cpu=[tail_len],
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=kv_pool,
            attn_backend=backend,
        )
        backend.init_forward_metadata(fb2)
        _ = model.model(fb2.input_ids, fb2.positions, fb2)

    # Shared prefix KV slots should not be overwritten ever.
    key0 = kv_pool.get_key_buffer(0)[prefix_slots].detach().clone()
    val0 = kv_pool.get_value_buffer(0)[prefix_slots].detach().clone()

    # Hook attention module output (clone to avoid buffer aliasing).
    attn_mod = model.model.layers[0].self_attn
    outs = {}

    def _hook(_m, _args, kwargs, out):
        outs["attn_out"] = out.detach().clone()

    h = attn_mod.register_forward_hook(_hook, with_kwargs=True)
    try:
        decode_steps = 8
        visible_token_id = 13
        saw_lmk_step = False
        for t in range(decode_steps):
            # Cross LMK slot: feed LMK exactly at page boundary positions.
            # `pos_idx` is the engine-visible position to be appended this step.
            pos_idx = prompt_len + t
            if (pos_idx % page_size) == (page_size - 1):
                token_id = lmk_id
                saw_lmk_step = True
            else:
                token_id = visible_token_id

            # Allocate per-req KV slot for this new token.
            loc1 = allocator.alloc(1)
            loc2 = allocator.alloc(1)
            assert loc1 is not None and loc2 is not None
            loc1 = loc1.to(torch.int64)
            loc2 = loc2.to(torch.int64)

            # Update req_to_token_pool mapping at the new position.
            req_to_token_pool.req_to_token[req1.req_pool_idx, pos_idx] = loc1[0].to(torch.int32)
            req_to_token_pool.req_to_token[req2.req_pool_idx, pos_idx] = loc2[0].to(torch.int32)

            seq_lens = torch.tensor(
                [prompt_len + t + 1, prompt_len + t + 1], device=device, dtype=torch.int32
            )
            positions = compute_decode_positions_landmark(seq_lens, page_size=page_size)
            fb = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=2,
                input_ids=torch.tensor([token_id, token_id], device=device, dtype=torch.int64),
                req_pool_indices=torch.tensor(
                    [req1.req_pool_idx, req2.req_pool_idx], device=device, dtype=torch.int32
                ),
                seq_lens=seq_lens,
                out_cache_loc=torch.cat([loc1, loc2], dim=0),
                seq_lens_sum=int(seq_lens.sum().item()),
                seq_lens_cpu=seq_lens.to("cpu"),
                positions=positions,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool=kv_pool,
                attn_backend=backend,
            )
            backend.init_forward_metadata(fb)
            _ = model.model(fb.input_ids, fb.positions, fb)

            out = outs["attn_out"]
            # Only assert equality for user-visible steps; LMK-step outputs are internal.
            if token_id != lmk_id:
                torch.testing.assert_close(out[0], out[1], rtol=5e-2, atol=5e-2)
                assert torch.isfinite(out[0]).all()
                assert torch.isfinite(out[1]).all()

        assert saw_lmk_step, "Expected decode sequence to cross an LMK slot (at least one LMK input step)."

        # Ensure shared prefix KV entries were never overwritten.
        torch.testing.assert_close(kv_pool.get_key_buffer(0)[prefix_slots], key0, rtol=0, atol=0)
        torch.testing.assert_close(
            kv_pool.get_value_buffer(0)[prefix_slots], val0, rtol=0, atol=0
        )
    finally:
        h.remove()

    _vprint("### radix prefix-kv reuse end2end")
    _vprint(f"- prompt_len(engine_visible,with_lmk)={prompt_len}")
    _vprint(f"- prefix_len(page_aligned)={prefix_len} tail_len={tail_len}")
    _vprint("=> OK: req2 reused req1 prefix KV slots and decode outputs matched.")

