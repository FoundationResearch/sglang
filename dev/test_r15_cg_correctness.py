"""End-to-end cuda graph correctness test for R15.2.

Uses compare.py's HSA model + backend setup (known-good, produces finite
logits via aligned weights) and wraps a single decode step in
`torch.cuda.graph()` capture + replay.  Compares logits:
  * reference: decode step run WITHOUT cuda graph
  * test:      same step replayed from a captured cuda graph

If they match (within bf16 reduction noise), the R15.2 cuda-graph
integration is numerically correct end-to-end.
"""
import sys, os, json, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "align"))

import bootstrap  # noqa
import torch
from safetensors.torch import load_file

from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

SGConfig = bootstrap.SGConfig
SGModel = bootstrap.SGModel
compute_position = bootstrap.compute_position
Req = bootstrap.Req


def build_sglang(cfg_dict, device, dtype):
    kw = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
    sw = kw.pop("sliding_window", 64)
    use_sw = kw.pop("use_sliding_window", True)
    kw["hsa_sliding_window"] = sw
    kw["use_sliding_window_merging"] = use_sw
    kw["sliding_window_merging_size"] = sw
    if "decoder_variant" not in kw:
        kw["decoder_variant"] = "qwen" if cfg_dict.get("model_type") == "qwen_lhsa" else "olmo"
    cfg = SGConfig(**kw)
    return cfg, SGModel(cfg).to(device=device, dtype=dtype).eval()


def setup_backend_and_prefill(sg_model, sg_cfg, real_prompt, device, dtype):
    """Build HSAAttnBackend, allocate CG buffers, run prefill."""
    PS = sg_cfg.chunk_size
    lmk_id = int(sg_cfg.vocab_size)
    mc = max(2 * (len(real_prompt) * 2 + 16), 512) + 1024

    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(
        size=mc + 256, page_size=PS, dtype=dtype,
        head_num=int(sg_cfg.num_key_value_heads), head_dim=int(sg_cfg.head_dim),
        layer_num=int(sg_cfg.num_hidden_layers), device=device,
        enable_memory_saver=False, enable_alt_stream=False,
    )
    mr = types.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=sg_model,
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, context_len=mc,
            num_attention_heads=int(sg_cfg.num_attention_heads),
            get_num_kv_heads=lambda tp: int(sg_cfg.num_key_value_heads) // tp,
        ),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend="hsa", speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id,
        ),
    )
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool
    mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)

    _hq, _hk = int(sg_cfg.num_attention_heads), int(sg_cfg.num_key_value_heads)
    if _hq > _hk:
        from sglang.srt.mem_cache.landmark_pool import LandmarkLmkKPool, ReqToChunkPool
        _max_chunks = (mc + PS - 1) // PS
        be.lmk_k_pool = LandmarkLmkKPool(
            num_chunk_slots=_max_chunks * 2,
            num_layers=int(sg_cfg.num_hidden_layers), h_q=_hq,
            head_dim=int(sg_cfg.head_dim), dtype=dtype, device=device,
        )
        be.req_to_chunk_pool = ReqToChunkPool(num_reqs=1, max_chunks_per_req=_max_chunks, device=device)

    # R15: pre-allocate cuda graph buffers
    be.init_cuda_graph_state(max_bs=4, max_num_tokens=mc)

    # ---- Prefill ----
    fi = Req._hsa_insert_lmk_prompt(real_prompt, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    r2t[0, :pl] = tl[:pl]
    ep = torch.tensor([0], device=device, dtype=torch.int32)
    es = torch.tensor([pl], device=device, dtype=torch.int32)
    pos, esl = compute_position("hsa", ep, es, pl, page_size=PS, enable_landmark_positions=True)
    fb_p = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.tensor(fi, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([pl], device=device, dtype=torch.int32),
        out_cache_loc=tl[:pl].to(torch.int64), seq_lens_sum=pl,
        seq_lens_cpu=torch.tensor([pl], device="cpu", dtype=torch.int32),
        positions=pos, extend_prefix_lens=ep, extend_seq_lens=es,
        extend_start_loc=esl, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[pl],
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool, attn_backend=be,
    )
    be.init_forward_metadata(fb_p)
    with torch.no_grad():
        _ = sg_model.model(fb_p.input_ids, fb_p.positions, fb_p)
    return be, mr, pool, r2t, pl


def make_decode_fb(token_id, pl, mc, mr, pool, r2t, PS, device):
    """Construct a forward_batch for one decode step at position `pl`."""
    r2t[0, pl] = pl  # next-token slot
    import bootstrap
    dpos = bootstrap.compute_decode_positions_landmark(
        torch.tensor([pl], device=device, dtype=torch.int32), page_size=PS,
    )
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE, batch_size=1,
        input_ids=torch.tensor([token_id], device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([pl + 1], device=device, dtype=torch.int32),
        out_cache_loc=torch.tensor([pl], device=device, dtype=torch.int64),
        seq_lens_sum=pl + 1,
        seq_lens_cpu=torch.tensor([pl + 1], device="cpu", dtype=torch.int32),
        positions=dpos,
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool, attn_backend=None,
    )


def main():
    bootstrap.init_sglang_dist()

    device = "cuda"
    dtype = torch.bfloat16

    cfg_path = HERE / "align" / "config_345m.json"
    wts_path = HERE / "align" / "weights_345m" / "model.safetensors"
    cfg_dict = json.loads(cfg_path.read_text())
    sg_cfg, sg_model = build_sglang(cfg_dict, device, dtype)
    state = load_file(str(wts_path))
    sg_model.load_state_dict(state, strict=False)
    sg_model.eval()
    # Fix RoPE cache dtype (compare.py does the same — sglang kernel needs fp32).
    for m in sg_model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)
    VS = int(sg_cfg.vocab_size)

    real_prompt = list(range(1, 65))  # 64 valid tokens
    decode_token = 100
    PS = sg_cfg.chunk_size

    be, mr, pool, r2t, pl = setup_backend_and_prefill(
        sg_model, sg_cfg, real_prompt, device, dtype
    )
    fb_d = make_decode_fb(decode_token, pl, None, mr, pool, r2t, PS, device)
    fb_d.attn_backend = be

    # ---- Reference: regular decode (no cuda graph) ----
    be.init_forward_metadata(fb_d)
    with torch.no_grad():
        h_ref = sg_model.model(fb_d.input_ids, fb_d.positions, fb_d)
    if isinstance(h_ref, tuple):
        h_ref = h_ref[0]
    logits_ref = (sg_model.model.norm(h_ref) @ sg_model.lm_head.weight[:VS].t()).float().cpu()
    print(f"Reference (no CG): hidden finite={torch.isfinite(h_ref).all().item()}, "
          f"logits finite={torch.isfinite(logits_ref).all().item()}, "
          f"argmax={logits_ref.argmax(-1).tolist()}")

    # ---- Capture cuda graph ----
    # Mimic sglang: invoke the HSA capture hook, then capture model.forward.
    be.init_forward_metadata_capture_cuda_graph(
        bs=1, num_tokens=1,
        req_pool_indices=fb_d.req_pool_indices,
        seq_lens=fb_d.seq_lens,
        encoder_lens=None,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
    )

    static_input_ids = fb_d.input_ids.clone()
    static_positions = fb_d.positions.clone()
    out_buffer = torch.empty_like(h_ref)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        # Warm up before capture
        for _ in range(2):
            with torch.no_grad():
                _ = sg_model.model(static_input_ids, static_positions, fb_d)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            h_graph = sg_model.model(static_input_ids, static_positions, fb_d)
            if isinstance(h_graph, tuple):
                h_graph = h_graph[0]
            out_buffer.copy_(h_graph)

    # ---- Replay (use the SAME backend state but updated buffers) ----
    be.init_forward_metadata_replay_cuda_graph(
        bs=1,
        req_pool_indices=fb_d.req_pool_indices,
        seq_lens=fb_d.seq_lens,
        seq_lens_sum=int(fb_d.seq_lens_sum),
        encoder_lens=None,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens_cpu=fb_d.seq_lens_cpu,
    )
    graph.replay()
    torch.cuda.synchronize()

    logits_graph = (sg_model.model.norm(out_buffer) @ sg_model.lm_head.weight[:VS].t()).float().cpu()
    print(f"Replayed graph: hidden finite={torch.isfinite(out_buffer).all().item()}, "
          f"logits finite={torch.isfinite(logits_graph).all().item()}, "
          f"argmax={logits_graph.argmax(-1).tolist()}")

    # ---- Compare ----
    max_abs = (logits_ref - logits_graph).abs().max().item()
    max_ref = logits_ref.abs().max().item()
    rel = max_abs / (max_ref + 1e-9)
    top5_ref = logits_ref.topk(5).indices.tolist()
    top5_grp = logits_graph.topk(5).indices.tolist()
    print(f"\nlogits max_abs_diff = {max_abs:.4e}  (ref max abs = {max_ref:.4f})")
    print(f"logits max_rel_diff = {rel:.4e}")
    print(f"argmax match = {logits_ref.argmax(-1).equal(logits_graph.argmax(-1))}")
    print(f"top5 match   = {top5_ref == top5_grp}")
    print(f"ref top5: {top5_ref}")
    print(f"grp top5: {top5_grp}")

    # bf16 reduction noise ≤ ~1e-2 at scale 1 absolute (but logits can be O(10))
    if torch.isfinite(out_buffer).all() and rel < 5e-2:
        print("\n*** [PASS] CUDA GRAPH CAPTURE+REPLAY (step 0) MATCHES REFERENCE ***")
    else:
        print(f"\n*** [FAIL] step 0 divergence rel={rel:.4e} ***")

    # ---- Multi-step test: capture at step 0, replay across many decode
    # iterations (different seq_lens / out_cache_loc each step), comparing
    # each replay to a fresh no-CG forward at the same step. ----
    print("\n=== Multi-step replay test ===")
    pl_base = pl
    NUM_STEPS = 4

    # Path A: no-CG reference — run NUM_STEPS decodes incrementally, save logits.
    be_ref, mr_ref, pool_ref, r2t_ref, _ = setup_backend_and_prefill(
        SGModel(sg_cfg).to(device=device, dtype=dtype).eval()
        .load_state_dict(state, strict=False) and sg_model,
        sg_cfg, real_prompt, device, dtype,
    ) if False else (be, mr, pool, r2t, pl_base)

    # Re-prefill freshly into a separate KV pool for the reference path.
    sg_model2 = SGModel(sg_cfg).to(device=device, dtype=dtype).eval()
    sg_model2.load_state_dict(state, strict=False)
    for m in sg_model2.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)
    be_ref, mr_ref, pool_ref, r2t_ref, pl_ref = setup_backend_and_prefill(
        sg_model2, sg_cfg, real_prompt, device, dtype,
    )

    ref_argmaxes = []
    cur_tok = decode_token
    for step in range(NUM_STEPS):
        fb_step = make_decode_fb(cur_tok, pl_ref + step, None, mr_ref, pool_ref, r2t_ref, PS, device)
        fb_step.attn_backend = be_ref
        be_ref.init_forward_metadata(fb_step)
        with torch.no_grad():
            h_step = sg_model2.model(fb_step.input_ids, fb_step.positions, fb_step)
        if isinstance(h_step, tuple):
            h_step = h_step[0]
        lg = (sg_model2.model.norm(h_step) @ sg_model2.lm_head.weight[:VS].t()).float().cpu()
        nxt = int(lg.argmax(-1).item())
        ref_argmaxes.append((step, nxt, lg.max().item()))
        cur_tok = nxt
    print(f"Reference (no CG) tokens: {[a[1] for a in ref_argmaxes]}")

    # Path B: replay the captured graph across NUM_STEPS, updating metadata
    # via init_forward_metadata_replay_cuda_graph each time.
    cur_tok = decode_token
    cg_argmaxes = []
    for step in range(NUM_STEPS):
        new_seq_lens = torch.tensor([pl_base + step + 1], device=device, dtype=torch.int32)
        new_out_cache_loc = torch.tensor([pl_base + step], device=device, dtype=torch.int64)
        # Update req_to_token so the new slot is mapped (bench does this).
        r2t[0, pl_base + step] = pl_base + step
        # Update the GRAPH'S static input tensors in place (so replay sees them).
        static_input_ids.copy_(torch.tensor([cur_tok], device=device, dtype=torch.int64))
        # Update fb_d's seq_lens / out_cache_loc to reflect this step (so the
        # replay hook receives the right metadata).
        fb_d.seq_lens.copy_(new_seq_lens)
        fb_d.out_cache_loc = new_out_cache_loc
        fb_d.seq_lens_sum = int(new_seq_lens.item())

        be.init_forward_metadata_replay_cuda_graph(
            bs=1,
            req_pool_indices=fb_d.req_pool_indices,
            seq_lens=new_seq_lens,
            seq_lens_sum=int(new_seq_lens.item()),
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            seq_lens_cpu=new_seq_lens.cpu(),
        )
        graph.replay()
        torch.cuda.synchronize()
        lg = (sg_model.model.norm(out_buffer) @ sg_model.lm_head.weight[:VS].t()).float().cpu()
        nxt = int(lg.argmax(-1).item())
        cg_argmaxes.append((step, nxt, lg.max().item()))
        cur_tok = nxt
    print(f"CG-replay tokens:           {[a[1] for a in cg_argmaxes]}")

    match = [a[1] for a in ref_argmaxes] == [a[1] for a in cg_argmaxes]
    if match:
        print("\n*** [PASS] MULTI-STEP CG REPLAY MATCHES NO-CG REFERENCE ***")
    else:
        print("\n*** [FAIL] multi-step divergence ***")


if __name__ == "__main__":
    main()
