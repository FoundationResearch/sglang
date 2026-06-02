"""Bisect WHICH layer / WHICH step first produces NaN in sglang HSA prefill at 16K.

For L=16384, instrument the forward pass to:
  1. Check hidden_state finiteness after each layer
  2. Check K/V cache pool finiteness after each layer's write
  3. Report which layer first goes NaN and what input/output looks like
"""
import sys, os, json, types
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


def build_sglang_model(cfg_dict, device, dtype):
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


def main():
    device = "cuda"
    dtype = torch.bfloat16
    L = int(os.environ.get("LEN", "16384"))

    cfg_path = HERE / "align" / "config_345m.json"
    cfg_dict = json.loads(cfg_path.read_text())
    sg_cfg, sg_model = build_sglang_model(cfg_dict, device, dtype)
    state = load_file(str(HERE / "align" / "weights_345m" / "model.safetensors"))
    sg_model.load_state_dict(state, strict=False)
    sg_model.eval()
    for m in sg_model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    g = torch.Generator().manual_seed(42 + L)
    VS = int(sg_cfg.vocab_size)
    real_prompt = torch.randint(5, VS - 5, (L,), generator=g).tolist()
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

    # Instrument each layer's forward.
    print(f"\nL={L}  pl={pl}  num_layers={sg_cfg.num_hidden_layers}")
    layer_input_finite = []
    layer_output_finite = []

    def make_hook(layer_idx):
        def hook(module, args, output):
            inp = args[0] if isinstance(args, tuple) else args
            inp_finite = bool(torch.isfinite(inp).all()) if isinstance(inp, torch.Tensor) else None
            out = output[0] if isinstance(output, tuple) else output
            out_finite = bool(torch.isfinite(out).all()) if isinstance(out, torch.Tensor) else None
            inp_max = float(inp.abs().max()) if isinstance(inp, torch.Tensor) and inp_finite else float("nan")
            out_max = float(out.abs().max()) if isinstance(out, torch.Tensor) and out_finite else float("nan")
            print(f"  layer {layer_idx:2d}: in finite={inp_finite} max|x|={inp_max:.4f}  "
                  f"-> out finite={out_finite} max|x|={out_max:.4f}")
            layer_input_finite.append(inp_finite)
            layer_output_finite.append(out_finite)
        return hook

    hooks = []
    layers = sg_model.model.layers if hasattr(sg_model.model, "layers") else None
    if layers is not None:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        h = sg_model.model(fb_p.input_ids, fb_p.positions, fb_p)
    if isinstance(h, tuple):
        h = h[0]

    for h_ in hooks:
        h_.remove()

    print(f"\n  final hidden finite: {torch.isfinite(h).all().item()}")
    if torch.isfinite(h).all():
        print(f"  final max|x|: {h.abs().max().item():.4f}")
    print(f"  first NaN layer: {next((i for i, f in enumerate(layer_output_finite) if not f), 'none')}")

    # Also check K/V cache pool finiteness per layer.
    print(f"\n  ---- K/V cache check after prefill ----")
    for layer_idx in range(sg_cfg.num_hidden_layers):
        k_buf = pool.get_key_buffer(layer_idx)
        v_buf = pool.get_value_buffer(layer_idx)
        # Only check the used slots (first pl positions of slot space).
        k_used = k_buf[:pl]
        v_used = v_buf[:pl]
        k_fin = bool(torch.isfinite(k_used).all())
        v_fin = bool(torch.isfinite(v_used).all())
        if not (k_fin and v_fin):
            k_nan_pos = (~torch.isfinite(k_used).all(dim=(-1, -2))).nonzero(as_tuple=True)[0]
            first_nan = int(k_nan_pos[0]) if len(k_nan_pos) > 0 else -1
            print(f"  layer {layer_idx}: K finite={k_fin}  V finite={v_fin}  "
                  f"first NaN pos in K: {first_nan}/{pl}")
        else:
            print(f"  layer {layer_idx}: K finite={k_fin}  V finite={v_fin}  (all OK)")


if __name__ == "__main__":
    main()
