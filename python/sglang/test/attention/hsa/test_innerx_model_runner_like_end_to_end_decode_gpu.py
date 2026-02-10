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


def _torch_swa_decode_window_innerx(
    *,
    q: torch.Tensor,  # [B,HQ,D] bf16/fp16
    k_cache: torch.Tensor,  # [Nloc,H,D]
    v_cache: torch.Tensor,  # [Nloc,H,D]
    page_table_1: torch.Tensor,  # [B,MAX_T] int32
    seq_lens: torch.Tensor,  # [B] int32
    window_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """SWA decode over last `window_size` tokens (InnerX: LMK included by default)."""
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
    q: torch.Tensor,  # [B,HQ,D]
    k_cache: torch.Tensor,  # [Nloc,H,D]
    v_cache: torch.Tensor,  # [Nloc,H,D]
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


def test_innerx_end_to_end_model_forward_extend_then_decode_matches_ultra_semantics_cuda():
    """
    Runner-like end-to-end test (no monkeypatch of backend):
    - Instantiate real `HSAForCausalLM` (InnerX) with a tiny config.
    - Run EXTEND (prefill) through the model to populate KV cache (runner-style ForwardBatch).
    - Run DECODE for one token through the model (real RadixAttention -> HSAAttnBackend).
    - Capture the InnerX HSA attention module's decode input/output via a forward hook.
    - Independently recompute the *ultra InnerX* decode semantics in torch:
        SWA heads (sliding window, LMK included) + HSA heads (selection on LMK-K, mask_last_token) + stitch + o_proj (+ optional gate)
      and assert the module output matches.
    """
    from sglang.srt.configs.flash_hsa import FlashHSAConfig
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers import dp_attention as dpa
    from sglang.srt.layers.attention.hsa.selector import (
        build_active_page_candidates,
        select_topk_pages_decode,
    )
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        ForwardMode,
        compute_decode_positions_landmark,
        compute_position,
    )
    from sglang.srt.models.flash_hsa import HSAForCausalLM
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Minimal single-process model-parallel init so `get_pp_group()` works.
    # This is still "runner-level" in spirit (ModelRunner would do this).
    if not dist.is_initialized():
        _, path = tempfile.mkstemp(prefix="sglang_dist_", suffix=".tmp")
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{path}",
            rank=0,
            world_size=1,
        )
    if not ps.model_parallel_is_initialized():
        # world / tp / pp are all size-1 groups.
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

    # Minimal DP-attention globals (ModelRunner would call initialize_dp_attention()).
    # For this unit test we only need TP=1.
    if getattr(dpa, "_ATTN_TP_RANK", None) is None:
        dpa._ATTN_TP_RANK = 0  # type: ignore[attr-defined]
        dpa._ATTN_TP_SIZE = 1  # type: ignore[attr-defined]
        dpa._ATTN_DP_RANK = 0  # type: ignore[attr-defined]
        dpa._ATTN_DP_SIZE = 1  # type: ignore[attr-defined]
        dpa._LOCAL_ATTN_DP_RANK = 0  # type: ignore[attr-defined]
        dpa._LOCAL_ATTN_DP_SIZE = 1  # type: ignore[attr-defined]
        dpa._ENABLE_DP_ATTENTION_FLAG = False  # type: ignore[attr-defined]
        dpa._ATTN_TP_GROUP = ps.get_tp_group()  # type: ignore[attr-defined]

    # Minimal global ServerArgs (ModelRunner would set this).
    try:
        _ = ServerArgs(model_path="dummy")
    except Exception:
        # Some repos may alias ServerArgs differently; keep test robust.
        _ = None
    if _ is not None:
        sa = ServerArgs(model_path="dummy")
        sa.attention_backend = "hsa"
        sa.enable_dp_lm_head = False
        set_global_server_args_for_scheduler(sa)

    # InnerX-ish small config (still consistent with ultra constraints).
    page_size = 4  # chunk_size
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
        full_attn_interleave=1,  # layer0 is HSA layer
        # split-head knobs (ultra)
        hsa_heads=4,  # denom=4, split 3/4 + 1/4
        hsa_qk_ratio=4,
        enable_gate=False,
        # sliding window for SWA branch inside HSA layer
        use_sliding_window_merging=True,
        sliding_window_merging_size=page_size,
        # sliding layers irrelevant for 1-layer test
        use_sliding_window_attention=False,
        sliding_window_attention_size=None,
        tie_word_embeddings=False,
    )

    model = HSAForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()

    # --- Build a prompt with LMKs (runner-visible sequence includes LMKs) ---
    lmk_id = int(cfg.vocab_size)
    real_prompt = [5, 6, 7, 8, 9, 10]  # len=6 => inserts 2 LMKs when page_size=4
    fill_ids = Req._hsa_insert_lmk_prompt(real_prompt, page_size=page_size, lmk_id=lmk_id)
    # Expected: [5,6,7,lmk, 8,9,10,lmk]
    assert len(fill_ids) == 8
    prefill_len = len(fill_ids)

    # --- Minimal model_runner stub for HSAAttnBackend + dense backend plumbing ---
    max_context_len = 32
    max_total_num_tokens = 256
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
        # HSA args (override-only; we keep them minimal)
        hsa_topk=None,
        hsa_selection_strategy=None,
        hsa_layers=None,
        hsa_window_size=None,
        hsa_enable_swa_merging=None,
        hsa_lmk_id=lmk_id,
    )

    req_to_token = torch.zeros((1, max_context_len), dtype=torch.int32, device=device)
    model_runner.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=req_to_token)
    model_runner.token_to_kv_pool = MHATokenToKVPool(
        size=max_total_num_tokens,
        page_size=page_size,
        dtype=dtype,
        head_num=int(cfg.num_key_value_heads),
        head_dim=int(cfg.head_dim),
        layer_num=int(cfg.num_hidden_layers),
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    model_runner.token_to_kv_pool_allocator = object()

    backend = HSAAttnBackend(model_runner)

    # Token locations (simple contiguous slots).
    token_locs = torch.arange(0, max_context_len, dtype=torch.int32, device=device)
    model_runner.req_to_token_pool.req_to_token[0, :prefill_len] = token_locs[:prefill_len]

    # --- EXTEND (prefill) ---
    extend_prefix_lens = torch.tensor([0], device=device, dtype=torch.int32)
    extend_seq_lens = torch.tensor([prefill_len], device=device, dtype=torch.int32)
    positions_ext, extend_start_loc = compute_position(
        attn_backend="hsa",
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_sum=int(prefill_len),
        page_size=page_size,
        enable_landmark_positions=True,
    )

    out_cache_loc_ext = token_locs[:prefill_len].to(torch.int64)
    fb_ext = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor(fill_ids, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([prefill_len], device=device, dtype=torch.int32),
        out_cache_loc=out_cache_loc_ext,
        seq_lens_sum=int(prefill_len),
        seq_lens_cpu=torch.tensor([prefill_len], device="cpu", dtype=torch.int32),
        positions=positions_ext,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        extend_start_loc=extend_start_loc,
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[prefill_len],
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        attn_backend=backend,
    )
    backend.init_forward_metadata(fb_ext)
    _ = model.model(fb_ext.input_ids, fb_ext.positions, fb_ext)  # populate KV

    # --- DECODE (one token) ---
    next_token_id = 11  # a visible token id (not LMK)
    seq_len_decode = prefill_len + 1
    model_runner.req_to_token_pool.req_to_token[0, prefill_len] = token_locs[prefill_len]

    pos_dec = compute_decode_positions_landmark(
        torch.tensor([seq_len_decode], device=device, dtype=torch.int32),
        page_size=page_size,
    )
    out_cache_loc_dec = token_locs[prefill_len : prefill_len + 1].to(torch.int64)
    fb_dec = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=torch.tensor([next_token_id], device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([seq_len_decode], device=device, dtype=torch.int32),
        out_cache_loc=out_cache_loc_dec,
        seq_lens_sum=int(seq_len_decode),
        seq_lens_cpu=torch.tensor([seq_len_decode], device="cpu", dtype=torch.int32),
        positions=pos_dec,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        attn_backend=backend,
    )

    # Hook the only layer's self_attn (InnerX HSA), its inner RadixAttention, and o_proj.
    attn_mod = model.model.layers[0].self_attn
    radix_attn = attn_mod.attn
    o_proj = attn_mod.o_proj
    captured = {}

    def _hook(_m, _args, kwargs, out):
        captured["positions"] = kwargs["positions"]
        captured["hidden_states_in"] = kwargs["hidden_states"]
        captured["forward_batch"] = kwargs["forward_batch"]
        captured["out"] = out.detach().clone()

    def _hook_radix(_m, args, _kwargs, out):
        # args: (q, k, v, forward_batch)
        captured["radix_q"] = args[0]
        captured["radix_k"] = args[1]
        captured["radix_v"] = args[2]
        captured["radix_out"] = out.detach().clone()

    h = attn_mod.register_forward_hook(_hook, with_kwargs=True)
    h2 = radix_attn.register_forward_hook(_hook_radix, with_kwargs=True)
    def _hook_o_proj(_m, _args, out):
        out0 = out[0] if isinstance(out, (tuple, list)) else out
        captured["o_proj_out"] = out0.detach().clone()
    h3 = o_proj.register_forward_hook(_hook_o_proj)
    try:
        backend.init_forward_metadata(fb_dec)
        _ = model.model(fb_dec.input_ids, fb_dec.positions, fb_dec)
    finally:
        h.remove()
        h2.remove()
        h3.remove()

    assert "out" in captured, "forward hook did not fire"
    hs_in = captured["hidden_states_in"]  # [B?, hidden]
    out_mod = captured["out"]  # [B?, hidden]
    attn_out_mod = captured["radix_out"]  # [B?, hidden]
    o_proj_out_mod = captured["o_proj_out"]  # [B?, hidden]

    # Normalize shapes to [B, hidden]
    if hs_in.dim() == 1:
        hs_in = hs_in.unsqueeze(0)
    if out_mod.dim() == 1:
        out_mod = out_mod.unsqueeze(0)
    if attn_out_mod.dim() == 1:
        attn_out_mod = attn_out_mod.unsqueeze(0)
    if o_proj_out_mod.dim() == 1:
        o_proj_out_mod = o_proj_out_mod.unsqueeze(0)
    assert hs_in.shape[0] == 1 and out_mod.shape[0] == 1

    # --- Torch reference: ultra InnerX decode semantics for this module ---
    with torch.no_grad():
        # Projections (tp=1, Column/Row parallel behave like normal linear)
        swa_q = attn_mod.q_proj(hs_in)[0].view(1, attn_mod.hq_swa, attn_mod.head_dim)
        swa_k = attn_mod.k_proj(hs_in)[0].view(1, attn_mod.hk_swa, attn_mod.head_dim)
        swa_v = attn_mod.v_proj(hs_in)[0].view(1, attn_mod.hk_swa, attn_mod.head_dim)
        hsa_q = attn_mod.hsa_q_proj(hs_in)[0].view(1, attn_mod.hq_hsa, attn_mod.head_dim)
        hsa_k = attn_mod.hsa_k_proj(hs_in)[0].view(1, attn_mod.hk_hsa, attn_mod.head_dim)
        hsa_v = attn_mod.hsa_v_proj(hs_in)[0].view(1, attn_mod.hk_hsa, attn_mod.head_dim)

        # Apply QK norm and RoPE (SWA only)
        def _rmsnorm(x, w, eps):
            x_f = x.float()
            var = x_f.pow(2).mean(dim=-1, keepdim=True)
            y = x_f * torch.rsqrt(var + float(eps))
            return (y * w.float()).to(x.dtype)

        swa_qn = _rmsnorm(swa_q, attn_mod.q_norm.weight, attn_mod.q_norm.variance_epsilon)
        swa_kn = _rmsnorm(swa_k, attn_mod.k_norm.weight, attn_mod.k_norm.variance_epsilon)
        # RoPE uses packed q/k in [T, H*D] form; reuse module rotary embedding (semantic equality).
        q_pack = swa_qn.reshape(1, -1)
        k_pack = swa_kn.reshape(1, -1)
        q_pack, k_pack = attn_mod.rotary_emb(fb_dec.positions, q_pack, k_pack)
        swa_qn = q_pack.view(1, attn_mod.hq_swa, attn_mod.head_dim)
        swa_kn = k_pack.view(1, attn_mod.hk_swa, attn_mod.head_dim)

        hsa_qn = _rmsnorm(hsa_q, attn_mod.q_norm.weight, attn_mod.q_norm.variance_epsilon)
        hsa_kn = _rmsnorm(hsa_k, attn_mod.k_norm.weight, attn_mod.k_norm.variance_epsilon)

        # Selection query (LMK-Q norm), expanded to q-head space.
        lmk_q = attn_mod.lmk_q_proj(hs_in)[0].view(1, attn_mod.hk_hsa, attn_mod.head_dim)
        lmk_qn = _rmsnorm(lmk_q, attn_mod.lmk_q_norm.weight, attn_mod.lmk_q_norm.variance_epsilon)
        assert attn_mod.hq_hsa % attn_mod.hk_hsa == 0
        G_hsa = attn_mod.hq_hsa // attn_mod.hk_hsa
        sel_q = (
            lmk_qn[:, :, None, :]
            .expand(1, attn_mod.hk_hsa, G_hsa, attn_mod.head_dim)
            .reshape(1, attn_mod.hq_hsa, attn_mod.head_dim)
            .contiguous()
        )

        md = backend.forward_metadata
        assert md is not None
        page_table_1 = md.page_table_1
        seq_lens = md.cache_seqlens_int32

        # SWA output on SWA head partition from cache.
        out_swa = _torch_swa_decode_window_innerx(
            q=swa_qn.to(dtype).view(1, attn_mod.hq_swa, attn_mod.head_dim),
            k_cache=fb_dec.token_to_kv_pool.get_key_buffer(0)[:, : attn_mod.hk_swa, :],
            v_cache=fb_dec.token_to_kv_pool.get_value_buffer(0)[:, : attn_mod.hk_swa, :],
            page_table_1=page_table_1,
            seq_lens=seq_lens,
            window_size=int(cfg.sliding_window_merging_size),
            sm_scale=float(attn_mod.scaling),
        )  # [1,hq_swa,D] float32

        # HSA selection (torch reference): recompute and verify it matches backend metadata.
        cand_page_ids, cand_mask = build_active_page_candidates(
            page_table_1=page_table_1,
            seq_lens=seq_lens,
            page_size=page_size,
            window_size=0,
        )
        completed_pages = torch.div(
            seq_lens.to(torch.int64), int(page_size), rounding_mode="floor"
        )
        cand_completed = cand_mask & (cand_page_ids.to(torch.int64) < completed_pages[:, None])
        safe_page_ids = cand_page_ids.clamp_min(0).to(torch.int64)
        lmk_locs = safe_page_ids * int(page_size) + (int(page_size) - 1)
        flat_lmk = lmk_locs.reshape(-1)
        k_cache_full = fb_dec.token_to_kv_pool.get_key_buffer(0)
        flat_repr = k_cache_full[flat_lmk][:, attn_mod.hk_swa : attn_mod.hk_swa + attn_mod.hk_hsa, :]
        cand_repr = flat_repr.view(1, cand_page_ids.shape[1], attn_mod.hk_hsa, attn_mod.head_dim)

        sel = select_topk_pages_decode(
            q=sel_q,
            cand_page_ids=cand_page_ids,
            cand_mask=cand_mask,
            cand_chunk_repr=cand_repr,
            cand_chunk_repr_valid=cand_completed,
            topk=int(cfg.hsa_topk),
            selection_strategy="head",
            sm_scale=float(attn_mod.scaling),
        )

        assert md.hsa_selected_page_ids is not None and md.hsa_selected_scores is not None
        torch.testing.assert_close(md.hsa_selected_page_ids, sel.selected_page_ids)
        torch.testing.assert_close(md.hsa_selected_scores, sel.selected_scores, rtol=0, atol=0)

        valid = sel.selected_page_ids >= 0
        scores = sel.selected_scores.masked_fill(~valid, float("-inf"))
        w_kv = torch.softmax(scores, dim=-1)
        w_kv = torch.nan_to_num(w_kv, nan=0.0)
        w_q = (
            w_kv[:, :, None, :]
            .expand(1, attn_mod.hk_hsa, G_hsa, int(cfg.hsa_topk))
            .reshape(1, attn_mod.hq_hsa, int(cfg.hsa_topk))
            .contiguous()
        )

        out_hsa = _torch_hsa_decode_from_weights(
            q=hsa_qn.to(dtype).view(1, attn_mod.hq_hsa, attn_mod.head_dim),
            k_cache=fb_dec.token_to_kv_pool.get_key_buffer(0)[:, attn_mod.hk_swa : attn_mod.hk_swa + attn_mod.hk_hsa, :],
            v_cache=fb_dec.token_to_kv_pool.get_value_buffer(0)[:, attn_mod.hk_swa : attn_mod.hk_swa + attn_mod.hk_hsa, :],
            page_table_1=page_table_1,
            selected_page_ids=sel.selected_page_ids,
            hsa_weights=w_q.to(torch.float32),
            page_size=page_size,
            sm_scale=float(attn_mod.scaling),
            mask_last_token=True,
        )  # [1,hq_hsa,D] float32

        out_stitch = torch.cat([out_swa, out_hsa], dim=1).reshape(1, -1).to(dtype)
        out_ref = attn_mod.o_proj(out_stitch)[0]  # [1, hidden]

    max_abs = (out_mod.float() - out_ref.float()).abs().max().item()
    max_abs_attn = (attn_out_mod.float() - out_stitch.float()).abs().max().item()
    max_abs_o_proj = (o_proj_out_mod.float() - out_ref.float()).abs().max().item()
    _vprint("### InnerX runner-like end2end (decode self_attn) check")
    _vprint(f"- fill_ids(len={prefill_len})={fill_ids}")
    _vprint(f"- max_abs_err={max_abs}")
    _vprint(f"- max_abs_err_pre_o_proj={max_abs_attn}")
    _vprint(f"- max_abs_err_o_proj_hook_vs_ref={max_abs_o_proj}")

    torch.testing.assert_close(out_mod, out_ref, rtol=5e-2, atol=5e-2)

