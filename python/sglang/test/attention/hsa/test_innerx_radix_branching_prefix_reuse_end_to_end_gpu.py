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


def test_innerx_radix_branching_prefix_reuse_runner_like_end2end_cuda():
    """
    A more complex radix-attention E2E test that validates *branching* + *page-aligned reuse*:

    - Prefill req1 and cache it in RadixCache (unfinished caching).
    - req2 shares a long prefix but diverges mid-page -> only page-aligned prefix is reused.
      We verify:
        * prefix_len == expected page-aligned length
        * req2 reuse decode outputs match a fresh-prefill req2_full (no reuse)
        * shared prefix KV slots are not overwritten during decode
        * decode sequence crosses LMK slots (LMK inputs appear at page boundary positions)
    - req3 diverges inside the first page -> only 1 page is reused; similarly verify against req3_full.
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

    # Minimal global ServerArgs (some model code reads this during __init__).
    # Must be set before instantiating HSAForCausalLM.
    page_size = 4
    sa = ServerArgs(model_path="dummy")
    sa.attention_backend = "hsa"
    sa.page_size = page_size
    sa.enable_dp_lm_head = False
    set_global_server_args_for_scheduler(sa)

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

    # --- InnerX-ish tiny config; head_dim must be >=16 for Triton extend kernel ---
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

    # --- Pools + radix cache ---
    max_context_len = 128
    req_to_token_pool = ReqToTokenPool(
        size=16, max_context_len=max_context_len, device=device, enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=8192,
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
        size=8192, dtype=torch.int64, device=device, kvcache=kv_pool, need_sort=False
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
    sa.hsa_lmk_id = lmk_id

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

    def _prefill_req_full(req: Req) -> int:
        """Allocate KV for full fill_ids and run an EXTEND forward."""
        fill = list(req.fill_ids)
        prompt_len = len(fill)
        kv_slots = allocator.alloc(prompt_len)
        assert kv_slots is not None
        kv_slots = kv_slots.to(torch.int64)
        req_to_token_pool.req_to_token[req.req_pool_idx, :prompt_len] = kv_slots.to(
            torch.int32
        )
        ext_prefix = torch.tensor([0], device=device, dtype=torch.int32)
        ext_len = torch.tensor([prompt_len], device=device, dtype=torch.int32)
        pos, start = compute_position(
            attn_backend="hsa",
            extend_prefix_lens=ext_prefix,
            extend_seq_lens=ext_len,
            extend_seq_lens_sum=int(prompt_len),
            page_size=page_size,
            enable_landmark_positions=True,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.tensor(fill, device=device, dtype=torch.int64),
            req_pool_indices=torch.tensor([req.req_pool_idx], device=device, dtype=torch.int32),
            seq_lens=torch.tensor([prompt_len], device=device, dtype=torch.int32),
            out_cache_loc=kv_slots,
            seq_lens_sum=int(prompt_len),
            seq_lens_cpu=torch.tensor([prompt_len], device="cpu", dtype=torch.int32),
            positions=pos,
            extend_prefix_lens=ext_prefix,
            extend_seq_lens=ext_len,
            extend_start_loc=start,
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[prompt_len],
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=kv_pool,
            attn_backend=backend,
        )
        backend.init_forward_metadata(fb)
        _ = model.model(fb.input_ids, fb.positions, fb)
        return prompt_len

    def _prefill_req_reuse_prefix(req: Req) -> tuple[int, int, torch.Tensor]:
        """Use radix prefix_indices mapping (already materialized in req.prefix_indices) and extend tail only."""
        fill = list(req.fill_ids)
        prefix_slots = req.prefix_indices.to(torch.int64)
        prefix_len = int(prefix_slots.numel())
        assert prefix_len % page_size == 0 and prefix_len > 0
        tail = fill[prefix_len:]
        tail_len = len(tail)

        req_to_token_pool.req_to_token[req.req_pool_idx, :prefix_len] = prefix_slots.to(
            torch.int32
        )
        if tail_len > 0:
            kv_tail = allocator.alloc(tail_len)
            assert kv_tail is not None
            kv_tail = kv_tail.to(torch.int64)
            req_to_token_pool.req_to_token[
                req.req_pool_idx, prefix_len : prefix_len + tail_len
            ] = kv_tail.to(torch.int32)

            ext_prefix = torch.tensor([prefix_len], device=device, dtype=torch.int32)
            ext_len = torch.tensor([tail_len], device=device, dtype=torch.int32)
            pos, start = compute_position(
                attn_backend="hsa",
                extend_prefix_lens=ext_prefix,
                extend_seq_lens=ext_len,
                extend_seq_lens_sum=int(tail_len),
                page_size=page_size,
                enable_landmark_positions=True,
            )
            fb = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=torch.tensor(tail, device=device, dtype=torch.int64),
                req_pool_indices=torch.tensor([req.req_pool_idx], device=device, dtype=torch.int32),
                seq_lens=torch.tensor([prefix_len + tail_len], device=device, dtype=torch.int32),
                out_cache_loc=kv_tail,
                seq_lens_sum=int(prefix_len + tail_len),
                seq_lens_cpu=torch.tensor([prefix_len + tail_len], device="cpu", dtype=torch.int32),
                positions=pos,
                extend_prefix_lens=ext_prefix,
                extend_seq_lens=ext_len,
                extend_start_loc=start,
                extend_prefix_lens_cpu=[prefix_len],
                extend_seq_lens_cpu=[tail_len],
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool=kv_pool,
                attn_backend=backend,
            )
            backend.init_forward_metadata(fb)
            _ = model.model(fb.input_ids, fb.positions, fb)
        return prefix_len, tail_len, prefix_slots

    def _decode_compare_two_reqs(
        *,
        req_a: Req,
        req_b: Req,
        prompt_len: int,
        prefix_slots_to_protect: torch.Tensor,
        decode_steps: int = 8,
        visible_token_id: int = 13,
    ):
        # Snapshot shared prefix KV.
        key0 = kv_pool.get_key_buffer(0)[prefix_slots_to_protect].detach().clone()
        val0 = kv_pool.get_value_buffer(0)[prefix_slots_to_protect].detach().clone()

        # Hook attention output (clone to avoid aliasing).
        attn_mod = model.model.layers[0].self_attn
        outs = {}

        def _hook(_m, _args, kwargs, out):
            outs["attn_out"] = out.detach().clone()

        h = attn_mod.register_forward_hook(_hook, with_kwargs=True)
        saw_lmk_step = False
        try:
            for t in range(decode_steps):
                pos_idx = prompt_len + t
                if (pos_idx % page_size) == (page_size - 1):
                    token_id = lmk_id
                    saw_lmk_step = True
                else:
                    token_id = visible_token_id

                loc_a = allocator.alloc(1)
                loc_b = allocator.alloc(1)
                assert loc_a is not None and loc_b is not None
                loc_a = loc_a.to(torch.int64)
                loc_b = loc_b.to(torch.int64)

                req_to_token_pool.req_to_token[req_a.req_pool_idx, pos_idx] = loc_a[0].to(
                    torch.int32
                )
                req_to_token_pool.req_to_token[req_b.req_pool_idx, pos_idx] = loc_b[0].to(
                    torch.int32
                )

                seq_lens = torch.tensor(
                    [prompt_len + t + 1, prompt_len + t + 1],
                    device=device,
                    dtype=torch.int32,
                )
                positions = compute_decode_positions_landmark(seq_lens, page_size=page_size)
                fb = ForwardBatch(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=2,
                    input_ids=torch.tensor([token_id, token_id], device=device, dtype=torch.int64),
                    req_pool_indices=torch.tensor(
                        [req_a.req_pool_idx, req_b.req_pool_idx],
                        device=device,
                        dtype=torch.int32,
                    ),
                    seq_lens=seq_lens,
                    out_cache_loc=torch.cat([loc_a, loc_b], dim=0),
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

            assert saw_lmk_step, "Expected decode sequence to cross an LMK slot."
            torch.testing.assert_close(
                kv_pool.get_key_buffer(0)[prefix_slots_to_protect], key0, rtol=0, atol=0
            )
            torch.testing.assert_close(
                kv_pool.get_value_buffer(0)[prefix_slots_to_protect], val0, rtol=0, atol=0
            )
        finally:
            h.remove()

    # ---------- Build req1 and cache ----------
    # Common prefix (7 real tokens -> 9 engine tokens with LMK for page_size=4).
    common_real = [5, 6, 7, 8, 9, 10, 11]
    origin1 = common_real + [12, 13, 14, 15]

    req1 = Req("r1", "", origin1, SamplingParams(max_new_tokens=16))
    req1.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)
    req1.req_pool_idx = req_to_token_pool.alloc(1)[0]
    req1.init_next_round_input(cache)  # no cache hit expected on empty cache
    prompt_len1 = _prefill_req_full(req1)
    cache.cache_unfinished_req(req1)

    # ---------- Branch A: long shared prefix but diverge mid-page ----------
    # Diverge after common prefix, but only page-aligned prefix should be reused.
    origin2 = common_real + [99, 100, 101]
    req2 = Req("r2", "", origin2, SamplingParams(max_new_tokens=16))
    req2.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)
    req2.req_pool_idx = req_to_token_pool.alloc(1)[0]
    req2.init_next_round_input(cache)
    fill2 = list(req2.fill_ids)
    prefix_len2 = int(req2.prefix_indices.numel())
    _vprint(f"- req2 prefix_len(engine,page_aligned)={prefix_len2} fill_len={len(fill2)}")
    # With common_real=7 and page_size=4, fill(common_real) has len 9, page-aligned prefix is 8.
    assert prefix_len2 == 8, f"Expected page-aligned prefix_len=8, got {prefix_len2}"
    prefix_len2, tail_len2, prefix_slots2 = _prefill_req_reuse_prefix(req2)

    # Fresh req2_full (no reuse) for correctness.
    req2_full = Req("r2_full", "", origin2, SamplingParams(max_new_tokens=16))
    req2_full.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)
    req2_full.req_pool_idx = req_to_token_pool.alloc(1)[0]
    req2_full.init_next_round_input(None)
    prompt_len2 = _prefill_req_full(req2_full)
    assert prompt_len2 == len(fill2)

    _decode_compare_two_reqs(
        req_a=req2,
        req_b=req2_full,
        prompt_len=prompt_len2,
        prefix_slots_to_protect=prefix_slots2,
        decode_steps=8,
        visible_token_id=13,
    )

    # ---------- Branch B: diverge inside first page (reuse only 1 page) ----------
    origin3 = [5, 6, 7, 200, 201, 202, 203]
    req3 = Req("r3", "", origin3, SamplingParams(max_new_tokens=16))
    req3.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)
    req3.req_pool_idx = req_to_token_pool.alloc(1)[0]
    req3.init_next_round_input(cache)
    prefix_len3 = int(req3.prefix_indices.numel())
    _vprint(f"- req3 prefix_len(engine,page_aligned)={prefix_len3}")
    assert prefix_len3 == 4, f"Expected prefix_len=4 (one page), got {prefix_len3}"
    prefix_len3, tail_len3, prefix_slots3 = _prefill_req_reuse_prefix(req3)

    req3_full = Req("r3_full", "", origin3, SamplingParams(max_new_tokens=16))
    req3_full.enable_hsa_lmk(page_size=page_size, lmk_id=lmk_id)
    req3_full.req_pool_idx = req_to_token_pool.alloc(1)[0]
    req3_full.init_next_round_input(None)
    prompt_len3 = _prefill_req_full(req3_full)

    _decode_compare_two_reqs(
        req_a=req3,
        req_b=req3_full,
        prompt_len=prompt_len3,
        prefix_slots_to_protect=prefix_slots3,
        decode_steps=8,
        visible_token_id=17,
    )

    _vprint("### radix branching prefix reuse end2end")
    _vprint(f"- req1 prompt_len(engine_visible)={prompt_len1}")
    _vprint(f"- req2 prefix_len={prefix_len2} tail_len={tail_len2}")
    _vprint(f"- req3 prefix_len={prefix_len3} tail_len={tail_len3}")
    _vprint("=> OK: branching + page-aligned prefix reuse matched fresh-prefill decode outputs; prefix KV protected.")

