"""
Compare logits between the official OLMo-LHSA reference and SGLang's HSA
inference path, using identical *trained* weights produced by train.py.

This isolates kernel/backend divergence from top-k selection noise: with random
init, two implementations can disagree purely because the selector picks
different pages under tiny numerical perturbations. After a few hundred
training steps, scores have signal and top-k is stable.

Usage:
    python dev/align/compare.py
    python dev/align/compare.py --manifest dev/align/manifest.json
    python dev/align/compare.py --no-verify-sha   # skip the integrity check

The script refuses to run if the on-disk weights' SHA-256 disagrees with the
manifest — a stale checkpoint vs. config mismatch is the most pernicious
silent failure mode here. Pass --no-verify-sha if you know what you're doing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import bootstrap  # noqa: E402,F401  side-effect import (mocks)

import torch  # noqa: E402
import types  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

OfficialConfig = bootstrap.OfficialConfig
OfficialModel = bootstrap.OfficialModel
SGConfig = bootstrap.SGConfig
SGModel = bootstrap.SGModel
HSAAttnBackend = bootstrap.HSAAttnBackend
MHATokenToKVPool = bootstrap.MHATokenToKVPool
ForwardBatch = bootstrap.ForwardBatch
ForwardMode = bootstrap.ForwardMode
compute_position = bootstrap.compute_position
compute_decode_positions_landmark = bootstrap.compute_decode_positions_landmark
Req = bootstrap.Req
insert_special_tokens = bootstrap.insert_special_tokens
create_position_ids_with_landmarks = bootstrap.create_position_ids_with_landmarks


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _build_official(cfg_dict: dict, device, dtype):
    kw = {k: v for k, v in cfg_dict.items() if not k.startswith('_')}
    cfg = OfficialConfig(**kw)
    cfg._attn_implementation = 'eager'
    cfg.pad_token_id = None
    if not hasattr(cfg, 'num_swa_layers'):
        cfg.num_swa_layers = 0
    return cfg, OfficialModel(cfg).to(device=device, dtype=dtype).eval()


def _build_sglang(cfg_dict: dict, device, dtype):
    """Translate the official config dict into the FlashHSAConfig field names
    sglang uses (sliding_window → hsa_sliding_window / sliding_window_merging_size,
    use_sliding_window → use_sliding_window_merging)."""
    kw = {k: v for k, v in cfg_dict.items() if not k.startswith('_')}
    sw = kw.pop('sliding_window', 64)
    use_sw = kw.pop('use_sliding_window', True)
    kw['hsa_sliding_window'] = sw
    kw['sliding_window_merging_size'] = sw
    kw['use_sliding_window_merging'] = use_sw
    kw['model_type'] = 'flash_hsa_innerx'
    kw['architectures'] = ['HSAForCausalLM']
    cfg = SGConfig(**kw)
    return cfg, SGModel(cfg).to(device=device, dtype=dtype).eval()


def official_prefill_logits(model, tokens, page_size: int, vocab_size: int, device):
    """Forward the full tokens (with LMK inserted) through the official model.
    Returns logits[:, :, :vocab_size] in fp32."""
    ids = insert_special_tokens(torch.tensor([tokens]), vocab_size, page_size).to(device)
    pos = create_position_ids_with_landmarks(len(tokens), page_size, device)
    with torch.no_grad():
        out = model(input_ids=ids, position_ids=pos, attention_mask=None, use_cache=False)
    return out.logits[:, :, :vocab_size].float()


def sglang_prefill_and_decode(sg_model, cfg, real_base, decode_tokens, device, dtype,
                              progress_every: int = 100):
    """Mirror sglang's engine path: prefill `real_base` (with engine-side LMK
    insertion), then DECODE through the full LMK-inserted suffix one engine
    position at a time.

    Returns:
        prefill_logits[1, eng_len, V] — fp32 logits over the prefill
        engine_decode_outputs: list of (engine_position, logits[V] fp32) for
            every engine step (real-token AND LMK-internal steps). The caller
            picks which positions to compare; KL on LMK steps is also valid
            (and informative — divergence on internal steps would point to a
            kernel bug).

    For decode > (page_size-1) real tokens, the engine inserts LMK tokens at
    page boundaries (see Req.hsa_decode_postprocess_sampled_token). We mirror
    that by feeding the precomputed LMK-inserted suffix `fi_full[pl:]` rather
    than just real tokens.
    """
    PS = cfg.chunk_size
    lmk_id = int(cfg.vocab_size)
    VS = int(cfg.vocab_size)
    # KV pool sizing: needs enough room for fi_full + some slack
    fi_full_len_est = (len(real_base) + len(decode_tokens)) + (
        (len(real_base) + len(decode_tokens)) // (PS - 1) + 2
    )
    mc = max(fi_full_len_est * 2, 512) + 1024

    r2t = torch.zeros((1, mc), dtype=torch.int32, device=device)
    pool = MHATokenToKVPool(
        size=mc + 256, page_size=PS, dtype=dtype,
        head_num=int(cfg.num_key_value_heads), head_dim=int(cfg.head_dim),
        layer_num=int(cfg.num_hidden_layers), device=device,
        enable_memory_saver=False, enable_alt_stream=False,
    )
    mr = types.SimpleNamespace(
        device=device, page_size=PS, sliding_window_size=None, model=sg_model,
        model_config=types.SimpleNamespace(
            is_encoder_decoder=False, context_len=mc,
            num_attention_heads=int(cfg.num_attention_heads),
            get_num_kv_heads=lambda tp: int(cfg.num_key_value_heads) // tp,
        ),
        hybrid_gdn_config=None, kimi_linear_config=None, gpu_id=0,
        server_args=types.SimpleNamespace(
            attention_backend='hsa', speculative_num_draft_tokens=0, speculative_num_steps=0,
            triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None,
            enable_deterministic_inference=False, hsa_topk=None, hsa_selection_strategy=None,
            hsa_layers=None, hsa_window_size=None, hsa_enable_swa_merging=None, hsa_lmk_id=lmk_id,
        ),
    )
    mr.req_to_token_pool = types.SimpleNamespace(size=1, req_to_token=r2t)
    mr.token_to_kv_pool = pool
    mr.token_to_kv_pool_allocator = object()
    be = HSAAttnBackend(mr)

    # Prefill
    fi = Req._hsa_insert_lmk_prompt(real_base, page_size=PS, lmk_id=lmk_id)
    pl = len(fi)
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
    with torch.no_grad():
        h_prefill = sg_model.model(fb.input_ids, fb.positions, fb)
    if isinstance(h_prefill, tuple):
        h_prefill = h_prefill[0]
    h_normed = sg_model.model.norm(h_prefill)
    prefill_logits = (h_normed @ sg_model.lm_head.weight[:VS, :].t()).float().unsqueeze(0)

    # Decode through the full LMK-inserted suffix, NOT just the real decode
    # tokens. fi_full[pl:] is the engine-visible token sequence to feed.
    fi_full = Req._hsa_insert_lmk_prompt(
        list(real_base) + list(decode_tokens), page_size=PS, lmk_id=lmk_id,
    )
    suffix = fi_full[pl:]  # what the decode loop will iterate over
    n_steps = len(suffix)

    current_len = pl
    engine_decode_outputs: list[tuple[int, torch.Tensor]] = []
    for step, tok in enumerate(suffix):
        current_len += 1
        r2t[0, current_len - 1] = tl[current_len - 1]
        dp = compute_decode_positions_landmark(
            torch.tensor([current_len], device=device, dtype=torch.int32), page_size=PS,
        )
        fbd = ForwardBatch(
            forward_mode=ForwardMode.DECODE, batch_size=1,
            input_ids=torch.tensor([tok], device=device, dtype=torch.int64),
            req_pool_indices=torch.tensor([0], device=device, dtype=torch.int32),
            seq_lens=torch.tensor([current_len], device=device, dtype=torch.int32),
            out_cache_loc=tl[current_len-1:current_len].to(torch.int64),
            seq_lens_sum=current_len,
            seq_lens_cpu=torch.tensor([current_len], device='cpu', dtype=torch.int32),
            positions=dp, req_to_token_pool=mr.req_to_token_pool,
            token_to_kv_pool=pool, attn_backend=be,
        )
        be.init_forward_metadata(fbd)
        with torch.no_grad():
            h = sg_model.model(fbd.input_ids, fbd.positions, fbd)
        if isinstance(h, tuple):
            h = h[0]
        h = sg_model.model.norm(h)
        logits = (h @ sg_model.lm_head.weight.t())[:, :VS].float().view(-1)
        # `current_len - 1` is the engine position the input token landed at;
        # logits predict the token at engine position `current_len`.
        engine_decode_outputs.append((current_len - 1, logits))
        if progress_every and (step + 1) % progress_every == 0:
            print(f'    decode {step + 1}/{n_steps}  engine_pos={current_len - 1}', flush=True)
    return prefill_logits, engine_decode_outputs


def kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(softmax(p) || softmax(q)) — one scalar."""
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    return float(torch.sum(p * (torch.log(p.clamp_min(1e-10)) - torch.log(q.clamp_min(1e-10))), dim=-1).mean())


def status(kl: float) -> str:
    if kl < 0.001:
        return 'IDENTICAL'
    if kl < 0.01:
        return 'PASS'
    if kl < 0.1:
        return 'CLOSE'
    if kl < 1.0:
        return 'WARN'
    return 'FAIL'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(HERE / 'manifest.json'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--no-verify-sha', action='store_true')
    parser.add_argument('--seed', type=int, default=123,
                        help='Seed for the synthetic comparison input (independent of train seed).')
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f'[compare] manifest not found at {manifest_path}; '
              f'run dev/align/train.py first.', file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())

    weights_path = (REPO / manifest['weights_path']).resolve()
    config_path = (REPO / manifest['config_path']).resolve()
    if not weights_path.exists():
        print(f'[compare] weights not found at {weights_path}', file=sys.stderr)
        return 2
    if not args.no_verify_sha:
        cfg_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if cfg_sha != manifest['config_sha256']:
            print(f'[compare] config SHA mismatch — manifest says {manifest["config_sha256"][:16]}…, '
                  f'on-disk config is {cfg_sha[:16]}…; retrain or pass --no-verify-sha',
                  file=sys.stderr)
            return 3
        w_sha = _sha256(weights_path)
        if w_sha != manifest['weights_sha256']:
            print(f'[compare] weights SHA mismatch — manifest says {manifest["weights_sha256"][:16]}…, '
                  f'on-disk weights are {w_sha[:16]}…; retrain or pass --no-verify-sha',
                  file=sys.stderr)
            return 3
        print(f'[compare] integrity OK  config_sha={cfg_sha[:16]}…  weights_sha={w_sha[:16]}…')

    bootstrap.init_sglang_dist()
    device = torch.device(args.device)
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

    cfg_dict = json.loads(config_path.read_text())

    print(f'[compare] manifest: {manifest_path.relative_to(REPO)}')
    print(f'[compare]   trained_at={manifest.get("trained_at_utc")}')
    print(f'[compare]   git_commit={manifest["env"]["git_commit"]} dirty={manifest["env"]["git_dirty"]}')
    print(f'[compare]   training: steps={manifest["training"]["steps"]} '
          f'final_loss={manifest["training"]["final_loss"]:.4f}')

    # Build models
    oc, om = _build_official(cfg_dict, device, dtype)
    sc, sm = _build_sglang(cfg_dict, device, dtype)

    # Load saved weights into the official model first, then transfer to sglang
    state = load_file(str(weights_path), device='cpu')
    state = {k: v.to(device=device, dtype=dtype) for k, v in state.items()}
    miss, unexp = om.load_state_dict(state, strict=False)
    if miss or unexp:
        print(f'[compare]   official load: missing={len(miss)} unexpected={len(unexp)}')
        for k in list(miss)[:5]:
            print(f'    MISSING:    {k}')
        for k in list(unexp)[:5]:
            print(f'    UNEXPECTED: {k}')

    bootstrap.force_native_rmsnorm(sm)
    loaded, total, skipped = bootstrap.transfer_official_to_sglang(om, sm)
    print(f'[compare]   sglang weight transfer: {loaded}/{total} loaded, {len(skipped)} skipped')
    for s in skipped[:5]:
        print(f'    SKIP: {s}')
    for m in sm.modules():
        if hasattr(m, 'cos_sin_cache') and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    # ---- Comparison ----
    PS = oc.chunk_size
    VS = oc.vocab_size

    rng = torch.Generator(device='cpu').manual_seed(args.seed)
    decode_toks_short = torch.randint(5, VS - 5, (8,), generator=rng).tolist()
    decode_toks_long = decode_toks_short[:5]
    # Long-decode case stresses cross-decode KV writes: the engine must add
    # newly decoded tokens (and the LMKs auto-inserted at page boundaries) to
    # the KV cache and have subsequent decode steps correctly attend to them.
    decode_toks_huge = torch.randint(5, VS - 5, (1024,), generator=rng).tolist()

    test_cases = [
        (10, decode_toks_short, 'SHORT 10'),
        (63, decode_toks_short, 'SHORT 63 (1 page-1)'),
        (65, decode_toks_short, 'SHORT 65 (1 page+1)'),
        (128, decode_toks_short, 'SHORT 128 (2 pages)'),
        (256, decode_toks_short, 'MID 256 (4 pages)'),
        (512, decode_toks_long, 'MID 512 (8 pages)'),
        (1024, decode_toks_long, 'MID 1024 (16 pages)'),
        (2048, decode_toks_long, 'LONG 2048 (32 pages)'),
        # Long prefill + long decode: 16 pages worth of tokens written during
        # decode itself; later decode steps must select among pages that
        # include freshly-written ones. This is the case the user flagged.
        (2048, decode_toks_huge, 'XLONG 2048 + 1024 decode (cross-page decode KV)'),
    ]

    summary = []
    for n_prefill, decode_toks, desc in test_cases:
        # Use deterministic synthetic input — independent rng so prompt is reproducible
        prompt_rng = torch.Generator(device='cpu').manual_seed(args.seed * 1000 + n_prefill)
        real_base = torch.randint(5, VS - 5, (n_prefill,), generator=prompt_rng).tolist()
        real_full = real_base + decode_toks

        print(f'\n--- {desc}  prefill={n_prefill} decode={len(decode_toks)} ---')

        try:
            off_logits = official_prefill_logits(om, real_full, PS, VS, device)
        except Exception as e:
            print(f'  official forward FAILED: {e}')
            continue

        try:
            sg_prefill_logits, sg_engine_decode = sglang_prefill_and_decode(
                sm, sc, real_base, decode_toks, device, dtype,
            )
        except Exception as e:
            print(f'  sglang forward FAILED: {e}')
            continue

        # Prefill: align on engine-len (with LMK)
        fi_base = Req._hsa_insert_lmk_prompt(real_base, page_size=PS, lmk_id=VS)
        eng_len = len(fi_base)
        if off_logits.shape[1] >= eng_len and sg_prefill_logits.shape[1] >= eng_len:
            off_p = off_logits[0, :eng_len]
            sg_p = sg_prefill_logits[0, :eng_len]
            kl = kl_div(off_p, sg_p)
            argmax = (off_p.argmax(-1) == sg_p.argmax(-1)).float().mean().item()
            max_err = (off_p - sg_p).abs().max().item()
            print(f'  PREFILL: KL={kl:.6f}  argmax_match={argmax:.1%}  max_err={max_err:.4f}  [{status(kl)}]')
            summary.append((desc, 'prefill', kl, argmax, status(kl)))
        else:
            print(f'  PREFILL: shape mismatch off={off_logits.shape} sg={sg_prefill_logits.shape}')

        # Decode: per-engine-position KL (both real-input AND LMK-input steps).
        # We separate the buckets in the report so a regression on cross-page
        # decode (LMK-input steps) is easy to spot.
        kls_real, kls_lmk = [], []
        argmax_match_real = 0
        argmax_total_real = 0
        n_print = 0
        verbose_threshold = 8  # for short cases, print each step; for long, print summary only
        is_long = len(sg_engine_decode) > 16
        for eng_pos, sg_l in sg_engine_decode:
            if eng_pos >= off_logits.shape[1] - 1:
                # Off-by-one: official has logits[i] predicting fi_full[i+1]; we
                # need eng_pos < len(fi_full) for both prefill+decode tokens to
                # have a defined NEXT prediction. Skip the very last step.
                continue
            off_l = off_logits[0, eng_pos]
            kl = kl_div(off_l.unsqueeze(0), sg_l.unsqueeze(0))
            am_off = int(off_l.argmax().item())
            am_sg = int(sg_l.argmax().item())
            input_was_lmk = (
                eng_pos < len(Req._hsa_insert_lmk_prompt(
                    list(real_base) + list(decode_toks), page_size=PS, lmk_id=VS,
                )) and Req._hsa_insert_lmk_prompt(
                    list(real_base) + list(decode_toks), page_size=PS, lmk_id=VS,
                )[eng_pos] == VS
            )
            if input_was_lmk:
                kls_lmk.append(kl)
            else:
                kls_real.append(kl)
                argmax_total_real += 1
                if am_off == am_sg:
                    argmax_match_real += 1
            if not is_long and n_print < verbose_threshold:
                mark = 'OK' if am_off == am_sg else 'MISS'
                tag = '  (LMK input)' if input_was_lmk else ''
                print(f'  decode@{eng_pos}: KL={kl:.6f}  argmax off={am_off} sg={am_sg} {mark}{tag}  [{status(kl)}]')
                n_print += 1

        def _bucket_stats(kls):
            if not kls:
                return None
            t = torch.tensor(kls)
            return {
                'n': int(t.numel()),
                'mean': float(t.mean()),
                'p50': float(t.quantile(0.5)),
                'p95': float(t.quantile(0.95)),
                'max': float(t.max()),
            }

        s_real = _bucket_stats(kls_real)
        s_lmk = _bucket_stats(kls_lmk)
        if s_real is not None:
            print(f'  decode REAL n={s_real["n"]:4d}  KL mean={s_real["mean"]:.6f} '
                  f'p95={s_real["p95"]:.6f} max={s_real["max"]:.6f}  '
                  f'argmax_match={argmax_match_real}/{argmax_total_real}={argmax_match_real/max(1,argmax_total_real):.1%}  '
                  f'[{status(s_real["max"])}]')
            summary.append((desc, f'decode_real(n={s_real["n"]})', s_real['mean'],
                            argmax_match_real / max(1, argmax_total_real), status(s_real['max'])))
        if s_lmk is not None:
            print(f'  decode LMK  n={s_lmk["n"]:4d}  KL mean={s_lmk["mean"]:.6f} '
                  f'p95={s_lmk["p95"]:.6f} max={s_lmk["max"]:.6f}  [{status(s_lmk["max"])}]')
            summary.append((desc, f'decode_lmk(n={s_lmk["n"]})', s_lmk['mean'],
                            float('nan'), status(s_lmk['max'])))

    print(f'\n{"=" * 70}\nSUMMARY\n{"=" * 70}')
    for row in summary:
        desc, kind, kl, am, st = row
        kl_s = f'KL={kl:.6f}' if kl == kl else 'KL=---'  # NaN check
        am_s = f'argmax={am:.1%}' if am == am else ''
        print(f'  {desc:30s}  {kind:14s}  {kl_s:18s} {am_s:14s} [{st}]')
    print('\nKL bands: IDENTICAL <0.001 | PASS <0.01 | CLOSE <0.1 | WARN <1.0 | FAIL >=1.0')
    print('Top-k selection noise should look like CLOSE/WARN at most for trained weights;')
    print('persistent FAIL on prefill or decode points to a real backend bug.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
