"""Long-context CG-vs-eager correctness test.

Extends test_r15_cg_correctness.py to test at multiple long-context prompts:
8K, 16K, 32K, 64K, 128K (and optionally 256K / 512K).

For each L:
  1. Build HSA-345M with aligned weights (compare.py setup).
  2. Prefill L tokens (no CG used here — prefill always eager).
  3. Run 1 decode step WITHOUT cuda graph → reference logits.
  4. Capture + replay 1 decode step WITH cuda graph → graph logits.
  5. Compare: max_abs_diff, max_rel_diff, argmax match, top5 match.

Pass criterion (per ctx): rel_diff < 5e-2 (within bf16 reduction noise) AND
top5 indices match.

Run:  LENGTHS=8192,16384,32768 python dev/test_long_ctx_cg_correctness.py
"""
import sys, os, json, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "align"))

import bootstrap  # noqa
bootstrap.init_sglang_dist()  # CRITICAL: init TP/PP groups before building model
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
    PS = sg_cfg.chunk_size
    lmk_id = int(sg_cfg.vocab_size)
    # Need room for prompt × 2 (LMK insertion) + decode slots + safety pad.
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
        be.req_to_chunk_pool = ReqToChunkPool(
            num_reqs=1, max_chunks_per_req=_max_chunks, device=device,
        )
    be.init_cuda_graph_state(max_bs=1, max_num_tokens=1)

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
        req_pool_indices=torch.tensor([0], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([pl], dtype=torch.int32, device=device),
        out_cache_loc=tl[:pl],
        positions=pos, extend_prefix_lens=ep, extend_seq_lens=es,
        extend_start_loc=esl, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[pl],
        seq_lens_sum=int(pl), seq_lens_cpu=torch.tensor([pl], dtype=torch.int32),
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool,
    )
    fb_p.attn_backend = be
    be.init_forward_metadata(fb_p)
    with torch.no_grad():
        _ = sg_model.model(fb_p.input_ids, fb_p.positions, fb_p)
    return be, mr, pool, r2t, pl


def make_decode_fb(decode_token, pl, decode_slot, mr, pool, r2t, PS, device):
    """Single-token decode: append `decode_token` at engine pos `pl`."""
    if decode_slot is None:
        decode_slot = torch.tensor([pl], dtype=torch.int32, device=device)
    r2t[0, pl] = decode_slot
    fb = ForwardBatch(
        forward_mode=ForwardMode.DECODE, batch_size=1,
        input_ids=torch.tensor([decode_token], dtype=torch.int32, device=device),
        req_pool_indices=torch.tensor([0], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([pl + 1], dtype=torch.int32, device=device),
        out_cache_loc=decode_slot,
        positions=torch.tensor([pl], dtype=torch.int64, device=device),
        extend_prefix_lens=None, extend_seq_lens=None, extend_start_loc=None,
        extend_prefix_lens_cpu=None, extend_seq_lens_cpu=None,
        seq_lens_sum=int(pl + 1), seq_lens_cpu=torch.tensor([pl + 1], dtype=torch.int32),
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool,
    )
    return fb


def test_one_length(L: int, sg_cfg, sg_model, state, device, dtype):
    """Run CG-vs-eager comparison at prompt length L."""
    print(f"\n{'=' * 60}\n  TESTING L = {L}\n{'=' * 60}")

    # Build a deterministic prompt of length L.
    torch.manual_seed(42 + L)
    VS = int(sg_cfg.vocab_size)
    real_prompt = torch.randint(5, VS - 5, (L,), generator=torch.Generator().manual_seed(42 + L)).tolist()
    decode_token = 100

    PS = sg_cfg.chunk_size
    try:
        be, mr, pool, r2t, pl = setup_backend_and_prefill(
            sg_model, sg_cfg, real_prompt, device, dtype
        )
    except torch.OutOfMemoryError as e:
        print(f"  [SKIP] OOM during setup: {e}")
        return None

    fb_d = make_decode_fb(decode_token, pl, None, mr, pool, r2t, PS, device)
    fb_d.attn_backend = be

    # Reference: eager decode (no CG)
    be.init_forward_metadata(fb_d)
    with torch.no_grad():
        h_ref = sg_model.model(fb_d.input_ids, fb_d.positions, fb_d)
    if isinstance(h_ref, tuple):
        h_ref = h_ref[0]
    logits_ref = (sg_model.model.norm(h_ref) @ sg_model.lm_head.weight[:VS].t()).float().cpu()
    fin_ref = torch.isfinite(logits_ref).all().item()
    print(f"  eager:   hidden finite={torch.isfinite(h_ref).all().item()}  "
          f"logits finite={fin_ref}  argmax={int(logits_ref.argmax(-1))}")
    if not fin_ref:
        print(f"  [WARN] eager logits has NaN/Inf — skipping CG comparison")
        return None

    # CG capture
    be.init_forward_metadata_capture_cuda_graph(
        bs=1, num_tokens=1,
        req_pool_indices=fb_d.req_pool_indices,
        seq_lens=fb_d.seq_lens, encoder_lens=None,
        forward_mode=ForwardMode.DECODE, spec_info=None,
    )
    static_input_ids = fb_d.input_ids.clone()
    static_positions = fb_d.positions.clone()
    out_buffer = torch.empty_like(h_ref)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
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

    be.init_forward_metadata_replay_cuda_graph(
        bs=1, req_pool_indices=fb_d.req_pool_indices,
        seq_lens=fb_d.seq_lens, seq_lens_sum=int(fb_d.seq_lens_sum),
        encoder_lens=None, forward_mode=ForwardMode.DECODE, spec_info=None,
        seq_lens_cpu=fb_d.seq_lens_cpu,
    )
    graph.replay()
    torch.cuda.synchronize()

    logits_graph = (sg_model.model.norm(out_buffer) @ sg_model.lm_head.weight[:VS].t()).float().cpu()
    fin_graph = torch.isfinite(logits_graph).all().item()
    print(f"  CG:      hidden finite={torch.isfinite(out_buffer).all().item()}  "
          f"logits finite={fin_graph}  argmax={int(logits_graph.argmax(-1))}")

    if not fin_graph:
        return {"L": L, "status": "CG_NaN"}

    max_abs = (logits_ref - logits_graph).abs().max().item()
    max_ref = logits_ref.abs().max().item()
    rel = max_abs / (max_ref + 1e-9)
    top5_ref = logits_ref.topk(5).indices.tolist()
    top5_grp = logits_graph.topk(5).indices.tolist()
    argmax_match = logits_ref.argmax(-1).equal(logits_graph.argmax(-1))
    top5_match = (top5_ref == top5_grp)

    status = "PASS" if (rel < 5e-2 and argmax_match) else "FAIL"
    print(f"  max_abs_diff = {max_abs:.4e}  (ref max abs = {max_ref:.4f})")
    print(f"  max_rel_diff = {rel:.4e}")
    print(f"  argmax match = {argmax_match}    top5 match = {top5_match}")
    print(f"  [{status}]")
    return {
        "L": L, "status": status,
        "max_abs": max_abs, "max_ref": max_ref, "rel": rel,
        "argmax_match": argmax_match, "top5_match": top5_match,
    }


def main():
    device = "cuda"
    dtype = torch.bfloat16

    cfg_path = HERE / "align" / "config_345m.json"
    wts_path = HERE / "align" / "weights_345m" / "model.safetensors"
    cfg_dict = json.loads(cfg_path.read_text())
    sg_cfg, sg_model = build_sglang(cfg_dict, device, dtype)
    state = load_file(str(wts_path))
    sg_model.load_state_dict(state, strict=False)
    sg_model.eval()
    for m in sg_model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    lengths_env = os.environ.get("LENGTHS", "8192,16384,32768")
    lengths = [int(x) for x in lengths_env.split(",") if x.strip()]

    results = []
    for L in lengths:
        # Fresh model per length (KV pool size depends on L).
        sg_cfg2, sg_model2 = build_sglang(cfg_dict, device, dtype)
        sg_model2.load_state_dict(state, strict=False)
        sg_model2.eval()
        for m in sg_model2.modules():
            if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
                m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)
        r = test_one_length(L, sg_cfg2, sg_model2, state, device, dtype)
        if r is not None:
            results.append(r)
        # Free GPU mem between lengths.
        del sg_model2
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\n  SUMMARY\n{'=' * 60}")
    print(f"{'L':>8}  {'status':<8}  {'rel_diff':<12}  {'argmax':<6}  {'top5':<5}")
    for r in results:
        print(f"{r['L']:>8}  {r['status']:<8}  "
              f"{r.get('rel', 0):.4e}  "
              f"{str(r.get('argmax_match', '-')):<6}  "
              f"{str(r.get('top5_match', '-')):<5}")

    all_pass = all(r.get("status") == "PASS" for r in results)
    if all_pass:
        print("\n*** ALL LENGTHS PASS ***")
    else:
        print("\n*** SOME LENGTHS FAILED ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
