"""Layer-by-layer divergence fingerprint between official Qwen-LHSA and sglang.

For the 345M alignment harness, the compare.py KL on SHORT 65 prefill sits at
~0.11 (WARN).  All algorithmic paths are aligned, so the residual is
cumulative bf16 numerical drift across the 16 transformer layers.  This
script bisects which layer / which sub-op (q_proj, k_proj, attention,
MLP, layernorm, etc.) is the largest contributor.

It runs the same SHORT-65-style prefill on:
  * the official HSAForCausalLM (qwen variant), and
  * the sglang HSAForCausalLM via the same fb-fake construction compare.py uses,

after loading the same weights into both, and registers ``forward_hook``s on
every sub-module of each so it can side-by-side compare outputs.

Output: one row per (layer_id, op) with abs-mean / abs-max / L2 / cosine
similarity between the two implementations' tensors.

Usage:
    python dev/align/fingerprint_345m.py --prefill 65
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap  # noqa: F401,E402  (side-effect mocks)

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


REPO = HERE.parents[1]


def _build_official(cfg_dict: dict, device, dtype):
    """Same as compare._build_official."""
    OfficialConfig = bootstrap.OfficialConfig
    kw = {k: v for k, v in cfg_dict.items() if not k.startswith('_')}
    cfg = OfficialConfig(**kw)
    cfg._attn_implementation = 'eager'
    cfg.pad_token_id = None
    if not hasattr(cfg, 'num_swa_layers'):
        cfg.num_swa_layers = 0
    cls = bootstrap.OfficialModel
    if cfg_dict.get('model_type') == 'qwen_lhsa':
        from models.FlashHSA.modeling_qwen_lhsa import HSAForCausalLM as cls
    return cfg, cls(cfg).to(device=device, dtype=dtype).eval()


def _build_sglang(cfg_dict: dict, device, dtype):
    """Same translation compare._build_sglang does."""
    SGConfig = bootstrap.SGConfig
    SGModel = bootstrap.SGModel
    kw = {k: v for k, v in cfg_dict.items() if not k.startswith('_')}
    sw = kw.pop('sliding_window', 64)
    use_sw = kw.pop('use_sliding_window', True)
    kw['hsa_sliding_window'] = sw
    kw['sliding_window_merging_size'] = sw
    kw['use_sliding_window_merging'] = use_sw
    original_mt = str(kw.get('model_type', 'olmo_lhsa')).lower()
    kw['model_type'] = 'flash_hsa_innerx'
    kw['architectures'] = ['HSAForCausalLM']
    if 'decoder_variant' not in kw:
        kw['decoder_variant'] = 'qwen' if original_mt == 'qwen_lhsa' else 'olmo'
    cfg = SGConfig(**kw)
    return cfg, SGModel(cfg).to(device=device, dtype=dtype).eval()


def _short_prefill_input(n_real: int, page_size: int, vocab_size: int, device, seed: int = 123):
    """Build the same SHORT-style prefill input compare.py uses."""
    from sglang.srt.managers.schedule_batch import Req
    g = torch.Generator(device='cpu').manual_seed(seed)
    real_base = torch.randint(5, vocab_size - 5, (n_real,), generator=g).tolist()
    fi = Req._hsa_insert_lmk_prompt(real_base, page_size=page_size, lmk_id=vocab_size)
    return fi, real_base


@torch.no_grad()
def _run_official_layer_outputs(om, cfg_dict, fi, device, deep_layer: int = -1):
    """Forward the official model with a hook on each transformer layer's output.
    If ``deep_layer >= 0``, ALSO hook all sub-modules of that layer."""
    from utils.landmark_utils import create_position_ids_with_landmarks
    ids = torch.tensor([fi], device=device, dtype=torch.long)
    n_real = sum(1 for t in fi if t != cfg_dict['vocab_size'])
    pos = create_position_ids_with_landmarks(None, n_real, cfg_dict['chunk_size'], device)

    layer_outputs: dict[int, torch.Tensor] = {}
    deep_outputs: dict[str, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(om.model.layers):
        def make_hook(idx):
            def hook(module, args, out):
                if isinstance(out, tuple):
                    out = out[0]
                layer_outputs[idx] = out.detach().to(torch.float32).cpu()
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))

    if deep_layer >= 0 and deep_layer < len(om.model.layers):
        target = om.model.layers[deep_layer]
        # Hook every named child of the target decoder layer.
        for name, sub in target.named_modules():
            if name == "":
                continue
            def make_dhook(n):
                def hook(module, args, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    if isinstance(out, torch.Tensor):
                        deep_outputs[n] = out.detach().to(torch.float32).cpu()
                return hook
            handles.append(sub.register_forward_hook(make_dhook(name)))

    try:
        om(input_ids=ids, position_ids=pos, attention_mask=None, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return layer_outputs, deep_outputs


@torch.no_grad()
def _run_sglang_layer_outputs(sm, cfg, fi, device, dtype, deep_layer: int = -1):
    """Forward sglang prefill with the same fake forward_batch compare.py builds,
    and hook each decoder layer's hidden_states output."""
    import types as _t
    HSAAttnBackend = bootstrap.HSAAttnBackend
    MHATokenToKVPool = bootstrap.MHATokenToKVPool
    ForwardBatch = bootstrap.ForwardBatch
    ForwardMode = bootstrap.ForwardMode
    compute_position = bootstrap.compute_position

    PS = cfg.chunk_size
    VS = cfg.vocab_size
    pl = len(fi)
    mc = pl + 64

    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(
        size=mc + 64, page_size=PS, dtype=dtype,
        head_num=int(cfg.num_key_value_heads), head_dim=int(cfg.head_dim),
        layer_num=int(cfg.num_hidden_layers), device=device,
        enable_memory_saver=False, enable_alt_stream=False,
    )
    mr = _t.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=sm,
        model_config=_t.SimpleNamespace(
            is_encoder_decoder=False, context_len=mc,
            num_attention_heads=int(cfg.num_attention_heads),
            get_num_kv_heads=lambda tp: int(cfg.num_key_value_heads) // tp,
        ),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=_t.SimpleNamespace(
            attention_backend='hsa', speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=int(VS),
        ),
    )
    mr.req_to_token_pool = _t.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool
    mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)

    # Per-q-head landmark pools (we're testing the qwen path with hq_hsa > h_hsa).
    if int(cfg.num_attention_heads) > int(cfg.num_key_value_heads):
        from sglang.srt.mem_cache.landmark_pool import LandmarkLmkKPool, ReqToChunkPool
        _max_chunks = (mc + PS - 1) // PS
        be.lmk_k_pool = LandmarkLmkKPool(
            num_chunk_slots=_max_chunks * 2,
            num_layers=int(cfg.num_hidden_layers),
            h_q=int(cfg.num_attention_heads),
            head_dim=int(cfg.head_dim),
            dtype=dtype, device=device,
        )
        be.req_to_chunk_pool = ReqToChunkPool(
            num_reqs=1, max_chunks_per_req=_max_chunks, device=device,
        )

    tl = torch.arange(0, mc, dtype=torch.int32, device=device)
    r2t[0, :pl] = tl[:pl]
    ep = torch.tensor([0], device=device, dtype=torch.int32)
    es = torch.tensor([pl], device=device, dtype=torch.int32)
    pos, esl = compute_position('hsa', ep, es, pl, page_size=PS, enable_landmark_positions=True)
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.tensor(fi, device=device, dtype=torch.int64),
        req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([pl], device=device, dtype=torch.int32),
        out_cache_loc=tl[:pl].to(torch.int64), seq_lens_sum=pl,
        seq_lens_cpu=torch.tensor([pl], device='cpu', dtype=torch.int32),
        positions=pos, extend_prefix_lens=ep, extend_seq_lens=es,
        extend_start_loc=esl, extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[pl],
        req_to_token_pool=mr.req_to_token_pool, token_to_kv_pool=pool, attn_backend=be,
    )
    be.init_forward_metadata(fb)

    layer_outputs: dict[int, torch.Tensor] = {}
    deep_outputs: dict[str, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(sm.model.layers):
        def make_hook(idx):
            def hook(module, args, out):
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 2:
                    out = out.unsqueeze(0)
                layer_outputs[idx] = out.detach().to(torch.float32).cpu()
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))

    if deep_layer is not None and 0 <= deep_layer < len(sm.model.layers):
        target = sm.model.layers[deep_layer]
        for name, sub in target.named_modules():
            if name == "":
                continue
            def make_dhook(n):
                def hook(module, args, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    if isinstance(out, torch.Tensor):
                        deep_outputs[n] = out.detach().to(torch.float32).cpu()
                return hook
            handles.append(sub.register_forward_hook(make_dhook(name)))

    try:
        sm.model(fb.input_ids, fb.positions, fb)
    finally:
        for h in handles:
            h.remove()
    return layer_outputs, deep_outputs


def _fingerprint(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compare two same-shape tensors.  Returns abs/rel/cosine metrics."""
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    n = a.numel()
    diff = (a - b).abs()
    norm_a = a.norm().item()
    norm_b = b.norm().item()
    cos = (a @ b).item() / (norm_a * norm_b + 1e-12)
    rel = diff.norm().item() / (norm_a + 1e-12)
    return dict(
        n=n,
        abs_mean=diff.mean().item(),
        abs_max=diff.max().item(),
        rel_l2=rel,
        cos=cos,
        norm_off=norm_a, norm_sg=norm_b,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=str(HERE / 'manifest_345m.json'))
    ap.add_argument('--prefill', type=int, default=65, help='real (pre-LMK) prefill length')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    cfg_path = (REPO / manifest['config_path']).resolve()
    weights_path = (REPO / manifest['weights_path']).resolve()

    cfg_dict = json.loads(cfg_path.read_text())
    print(f'[fingerprint] config={cfg_path.name}  weights={weights_path.name}')
    print(f'[fingerprint] prefill_real={args.prefill} chunk_size={cfg_dict["chunk_size"]}')

    bootstrap.init_sglang_dist()
    device = torch.device(args.device)
    dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[args.dtype]

    oc, om = _build_official(cfg_dict, device, dtype)
    sc, sm = _build_sglang(cfg_dict, device, dtype)
    state = load_file(str(weights_path), device='cpu')
    state = {k: v.to(device=device, dtype=dtype) for k, v in state.items()}
    om.load_state_dict(state, strict=False)
    bootstrap.force_native_rmsnorm(sm)
    loaded, total, skipped = bootstrap.transfer_official_to_sglang(om, sm)
    print(f'[fingerprint] transfer: {loaded}/{total} loaded, {len(skipped)} skipped')
    # sgl_kernel's apply_rope_with_cos_sin_cache_inplace requires fp32 cache.
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    # Same input both sides.
    fi, _ = _short_prefill_input(args.prefill, int(cfg_dict['chunk_size']),
                                  int(cfg_dict['vocab_size']), device, args.seed)
    pl = len(fi)
    print(f'[fingerprint] engine prefill length = {pl}')

    print('\n[fingerprint] official forward...')
    off_outs, off_deep = _run_official_layer_outputs(om, cfg_dict, fi, device, deep_layer=0)
    print('[fingerprint] sglang forward...')
    sg_outs, sg_deep = _run_sglang_layer_outputs(sm, sc, fi, device, dtype, deep_layer=0)

    # Embed comparison
    print('\n[fingerprint] embed_tokens output diff:')
    _off_emb = om.model.embed_tokens(torch.tensor([fi], device=device, dtype=torch.long))
    _sg_emb = sm.model.embed_tokens(torch.tensor(fi, device=device, dtype=torch.long))
    if _sg_emb.dim() == 2:
        _sg_emb = _sg_emb.unsqueeze(0)
    fp = _fingerprint(_off_emb, _sg_emb)
    print(f'  abs_mean={fp["abs_mean"]:.6f} abs_max={fp["abs_max"]:.6f} '
          f'rel_L2={fp["rel_l2"]:.6f} cos={fp["cos"]:.6f}')

    print('\n' + '=' * 110)
    print(f'{"layer":<6} {"shape":<22} {"abs_mean":>11} {"abs_max":>11} {"rel_L2":>10} {"cos":>10}   {"|off|":>10}  {"|sg|":>10}')
    print('-' * 110)
    for i in range(int(cfg_dict['num_hidden_layers'])):
        if i not in off_outs or i not in sg_outs:
            print(f'layer{i:>3}  [MISSING]')
            continue
        a = off_outs[i].squeeze(0)  # [T, D]
        b = sg_outs[i].squeeze(0)
        Tmin = min(a.shape[0], b.shape[0])
        a, b = a[:Tmin], b[:Tmin]
        if a.shape != b.shape:
            print(f'layer{i:>3}  SHAPE MISMATCH off={tuple(a.shape)} sg={tuple(b.shape)}')
            continue
        fp = _fingerprint(a, b)
        print(f'layer{i:>3}  {str(tuple(a.shape)):<22} '
              f'{fp["abs_mean"]:>11.6f} {fp["abs_max"]:>11.6f} '
              f'{fp["rel_l2"]:>10.6f} {fp["cos"]:>10.6f}   '
              f'{fp["norm_off"]:>10.3f}  {fp["norm_sg"]:>10.3f}')

    print('=' * 110)

    # Layer 0 deep dive
    print(f'\n[fingerprint] off_deep captured {len(off_deep)} sub-modules, '
          f'sg_deep captured {len(sg_deep)} sub-modules')
    print('\n--- OFFICIAL layer-0 sub-modules ---')
    for k in sorted(off_deep.keys()):
        v = off_deep[k]
        print(f'  {k:<50} {tuple(v.shape)} '
              f'mean={v.float().mean().item():+.4f} std={v.float().std().item():+.4f}')
    print('\n--- SGLANG layer-0 sub-modules ---')
    for k in sorted(sg_deep.keys()):
        v = sg_deep[k]
        print(f'  {k:<50} {tuple(v.shape)} '
              f'mean={v.float().mean().item():+.4f} std={v.float().std().item():+.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
