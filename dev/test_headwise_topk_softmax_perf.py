"""Validate the new max-pooling prefill topk path.

Per L:
  1. Build sglang HSA backend with `headwise_topk_softmax=True` (default,
     unchanged slow path) — run prefill, time it, save last-pos logits.
  2. Build sglang HSA backend with `headwise_topk_softmax=False` (new fast
     path via online_topk_head + swa_lse) — same prefill, time it, save logits.
  3. (Optional) Build official HSAForCausalLM via compare.py harness — run
     prefill, save last-pos logits as ground truth.
  4. Report:
       - softmax path KL vs official (sanity: should equal previous ~3e-5)
       - maxpool path KL vs official (new path's quality)
       - softmax vs maxpool divergence (intrinsic algorithmic gap)
       - softmax vs maxpool prefill wall time (speedup)

IMPORTANT: each (L, mode) runs in a FRESH PROCESS to avoid the long-ctx
stateful contamination documented in test_nan_at_long_ctx.py.

Usage:
    for mode in softmax maxpool; do
        LEN=16384 MODE=$mode python dev/test_headwise_topk_softmax_perf.py
    done
"""
import sys, os, json, types, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "align"))

import bootstrap  # noqa
bootstrap.init_sglang_dist()

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


def setup_and_prefill(sg_model, sg_cfg, real_prompt, device, dtype, headwise_softmax):
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
            hsa_headwise_topk_softmax=headwise_softmax,
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
        out_cache_loc=tl[:pl].to(torch.int64),
        positions=pos, extend_prefix_lens=ep, extend_seq_lens=es,
        extend_start_loc=esl, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[pl],
        seq_lens_sum=int(pl), seq_lens_cpu=torch.tensor([pl], dtype=torch.int32),
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool, attn_backend=be,
    )
    be.init_forward_metadata(fb_p)

    # WARMUP — first forward includes JIT compile.
    with torch.no_grad():
        _ = sg_model.model(fb_p.input_ids, fb_p.positions, fb_p)
    torch.cuda.synchronize()

    # TIMED forward.
    t0 = time.perf_counter()
    with torch.no_grad():
        h = sg_model.model(fb_p.input_ids, fb_p.positions, fb_p)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000  # ms
    if isinstance(h, tuple):
        h = h[0]
    return be, mr, pool, r2t, pl, h, elapsed


def main():
    device = "cuda"
    dtype = torch.bfloat16
    L = int(os.environ.get("LEN", "16384"))
    mode = os.environ.get("MODE", "softmax").lower()  # softmax | maxpool

    assert mode in ("softmax", "maxpool"), f"MODE must be softmax or maxpool, got {mode}"
    headwise = True if mode == "softmax" else False

    cfg_path = HERE / "align" / "config_345m.json"
    wts_path = HERE / "align" / "weights_345m" / "model.safetensors"
    cfg_dict = json.loads(cfg_path.read_text())
    VS = int(cfg_dict["vocab_size"])

    g = torch.Generator().manual_seed(42 + L)
    real_prompt = torch.randint(5, VS - 5, (L,), generator=g).tolist()

    print(f"\n{'=' * 60}\n  L={L}  MODE={mode}  headwise_topk_softmax={headwise}\n{'=' * 60}", flush=True)

    sg_cfg, sg_model = build_sglang(cfg_dict, device, dtype)
    state = load_file(str(wts_path))
    sg_model.load_state_dict(state, strict=False)
    sg_model.eval()
    for m in sg_model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    be, mr, pool, r2t, pl, h, elapsed = setup_and_prefill(
        sg_model, sg_cfg, real_prompt, device, dtype, headwise_softmax=headwise,
    )
    maxpool_calls = getattr(be, "_maxpool_call_count", 0)
    print(f"  maxpool_kernel_calls = {maxpool_calls}  (expect >0 iff MODE=maxpool)", flush=True)

    h_finite_per_pos = torch.isfinite(h).all(dim=-1)
    finite_count = int(h_finite_per_pos.sum())
    total_pos = int(h_finite_per_pos.numel())
    print(f"  finite: {finite_count}/{total_pos}", flush=True)
    sg_last = (sg_model.model.norm(h[-1:]) @ sg_model.lm_head.weight[:VS].t()).float().squeeze(0)
    last_finite = bool(torch.isfinite(sg_last).all())
    if last_finite:
        argmax = int(sg_last.argmax())
        max_abs = float(sg_last.abs().max())
        print(f"  last_pos finite=True  argmax={argmax}  max|logit|={max_abs:.3f}", flush=True)
    print(f"  prefill_wall_ms={elapsed:.1f}", flush=True)
    # Save logits to file for inter-mode comparison.
    out_dir = HERE / "_headwise_topk_artifacts"
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / f"L{L}_{mode}.pt"
    torch.save({
        "L": L, "mode": mode, "headwise": headwise,
        "elapsed_ms": elapsed,
        "last_pos_logits": sg_last.cpu() if last_finite else None,
        "finite_count": finite_count, "total_pos": total_pos,
        "argmax": argmax if last_finite else None,
    }, save_path)
    print(f"  saved -> {save_path}", flush=True)

    print(f"@@RESULT@@ {json.dumps({'L': L, 'mode': mode, 'elapsed_ms': round(elapsed,2), 'last_finite': last_finite, 'argmax': argmax if last_finite else None})}")


if __name__ == "__main__":
    main()
