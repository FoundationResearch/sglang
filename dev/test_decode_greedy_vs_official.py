"""Verify sglang HSA *decode* logits match official HSAForCausalLM at long ctx.

Two-stage: official and sglang run in separate processes (avoids long-ctx
stateful contamination documented in test_nan_at_long_ctx.py).

Stage 1 (--stage official):
  - Generate L deterministic random tokens.
  - Greedy-chain N+1 forwards through OFFICIAL HSAForCausalLM (eager attn).
  - Each forward gives a logit at the LAST REAL TOKEN position; argmax → tok_i.
  - Save chosen tokens + per-step fp32 logits to a .pt file.

Stage 2 (--stage sglang):
  - Load chosen decode tokens + per-step logits from stage 1.
  - Build sglang HSA, prefill L tokens, feed the same tok_0..tok_{N-1} one
    at a time via sglang's decode path.
  - For each sglang step i, compare with official's logit at step i+1
    (both predict the same engine position).
  - Report per-step KL, argmax match, top-5 overlap.

Stage 3 (--stage compare): load both side, compute and print.

Run:
    for L in 8192 16384 32768; do
        LEN=$L bash -c '
          python dev/test_decode_greedy_vs_official.py --stage official &&
          python dev/test_decode_greedy_vs_official.py --stage sglang
        '
    done
"""
import sys, os, json, types, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART_DIR = HERE / "_decode_greedy_artifacts"
ART_DIR.mkdir(exist_ok=True)


def stage_official():
    """Run official-only path: build official, greedy N+1 forwards, save logits."""
    sys.path.insert(0, str(HERE / "align"))
    import bootstrap  # noqa
    bootstrap.init_sglang_dist()
    import torch
    from safetensors.torch import load_file
    from compare import _build_official, insert_special_tokens, create_position_ids_with_landmarks

    device = "cuda"
    dtype = torch.bfloat16
    L = int(os.environ.get("LEN", "8192"))
    N = int(os.environ.get("N_DECODE", "4"))
    cfg_path = HERE / "align" / "config_345m.json"
    wts_path = HERE / "align" / "weights_345m" / "model.safetensors"
    cfg_dict = json.loads(cfg_path.read_text())
    PS = int(cfg_dict.get("chunk_size", 64))
    VS = int(cfg_dict["vocab_size"])

    g = torch.Generator().manual_seed(42 + L)
    base_tokens = torch.randint(5, VS - 5, (L,), generator=g).tolist()

    print(f"\n[OFFICIAL] L={L}  N={N}  building model...", flush=True)
    oc, om = _build_official(cfg_dict, device, dtype)
    state = load_file(str(wts_path))
    om.load_state_dict(state, strict=False)
    om.eval()

    chain_toks: list[int] = []
    chain_logits: list[torch.Tensor] = []
    cur = list(base_tokens)
    lmk_id = VS
    for i in range(N + 1):
        torch.cuda.empty_cache()
        ids = insert_special_tokens(torch.tensor([cur]), VS, PS).to(device)
        pos = create_position_ids_with_landmarks(None, len(cur), PS, device)
        with torch.no_grad():
            out = om(input_ids=ids, position_ids=pos, attention_mask=None, use_cache=False)
        eng_tokens = ids[0]
        real_mask = (eng_tokens != lmk_id)
        last_real = int(real_mask.nonzero(as_tuple=True)[0][-1])
        logit = out.logits[0, last_real, :VS].float().cpu()
        tok = int(logit.argmax())
        if tok >= VS:
            tok = VS - 1
        chain_toks.append(tok)
        chain_logits.append(logit)
        print(f"  forward {i+1}/{N+1}  cur_len={len(cur)}  next_tok={tok}  "
              f"max|logit|={float(logit.abs().max()):.3f}", flush=True)
        cur.append(tok)

    art_path = ART_DIR / f"official_L{L}_N{N}.pt"
    torch.save({
        "L": L, "N": N, "PS": PS, "VS": VS,
        "base_tokens": base_tokens,
        "decode_tokens": chain_toks[:N],
        "chain_logits": torch.stack(chain_logits, dim=0),   # [N+1, V]
    }, art_path)
    print(f"[OFFICIAL] saved -> {art_path}", flush=True)


def stage_sglang():
    """Load official artifact, build sglang HSA, prefill + decode N steps."""
    sys.path.insert(0, str(HERE / "align"))
    import bootstrap  # noqa
    bootstrap.init_sglang_dist()
    import torch
    from safetensors.torch import load_file
    from compare import sglang_prefill_and_decode

    SGConfig = bootstrap.SGConfig
    SGModel = bootstrap.SGModel
    Req = bootstrap.Req

    device = "cuda"
    dtype = torch.bfloat16
    L = int(os.environ.get("LEN", "8192"))
    N = int(os.environ.get("N_DECODE", "4"))
    cfg_path = HERE / "align" / "config_345m.json"
    wts_path = HERE / "align" / "weights_345m" / "model.safetensors"
    cfg_dict = json.loads(cfg_path.read_text())
    PS = int(cfg_dict.get("chunk_size", 64))
    VS = int(cfg_dict["vocab_size"])

    art_path = ART_DIR / f"official_L{L}_N{N}.pt"
    if not art_path.exists():
        print(f"[ERROR] artifact missing: {art_path}. Run --stage official first.")
        sys.exit(2)
    art = torch.load(art_path, weights_only=False)
    assert art["L"] == L and art["N"] == N
    base_tokens = art["base_tokens"]
    decode_tokens = art["decode_tokens"]
    chain_logits = art["chain_logits"]   # [N+1, V] fp32 cpu

    print(f"\n[SGLANG] L={L}  N={N}  decode_tokens={decode_tokens}", flush=True)

    kw = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
    sw = kw.pop("sliding_window", 64); use_sw = kw.pop("use_sliding_window", True)
    kw["hsa_sliding_window"] = sw; kw["use_sliding_window_merging"] = use_sw
    kw["sliding_window_merging_size"] = sw
    if "decoder_variant" not in kw:
        kw["decoder_variant"] = "qwen" if cfg_dict.get("model_type") == "qwen_lhsa" else "olmo"
    sg_cfg = SGConfig(**kw)
    sg_model = SGModel(sg_cfg).to(device=device, dtype=dtype).eval()
    state = load_file(str(wts_path))
    sg_model.load_state_dict(state, strict=False)
    sg_model.eval()
    for m in sg_model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            m.cos_sin_cache = m.cos_sin_cache.to(torch.float32)

    print(f"[SGLANG] prefill + decode...", flush=True)
    _, engine_outs = sglang_prefill_and_decode(
        sg_model, sg_cfg, list(base_tokens), list(decode_tokens),
        device, dtype, progress_every=0,
    )

    # Filter engine_outs to keep only the steps where input == real decode token.
    fi_base = Req._hsa_insert_lmk_prompt(list(base_tokens), page_size=PS, lmk_id=VS)
    fi_full = Req._hsa_insert_lmk_prompt(
        list(base_tokens) + list(decode_tokens), page_size=PS, lmk_id=VS,
    )
    pl = len(fi_base)
    suffix = fi_full[pl:]
    real_logits: list[torch.Tensor] = []
    for i, tok in enumerate(suffix):
        if int(tok) != VS:
            eng_pos, logit = engine_outs[i]
            real_logits.append(logit.cpu())
    assert len(real_logits) == N

    # Pair sg_real[i] ↔ chain_logits[i+1] (both predict tok_{i+1}).
    print(f"\n  {'step':<5} {'KL(sg||off)':<13} {'argmax_match':<13} {'top5_overlap':<13} "
          f"{'off_argmax':<10} {'sg_argmax':<10} {'max|sg|':<8} {'max|off|':<8}")
    bad = 0
    rows = []
    for i in range(N):
        sg = real_logits[i].float()
        off = chain_logits[i + 1].float()
        la = torch.log_softmax(sg, dim=-1)
        lb = torch.log_softmax(off, dim=-1)
        kl = torch.nn.functional.kl_div(la, lb, reduction="batchmean", log_target=True).item()
        sg_arg = int(sg.argmax())
        off_arg = int(off.argmax())
        sg_top5 = set(sg.topk(5).indices.tolist())
        off_top5 = set(off.topk(5).indices.tolist())
        ovlp = len(sg_top5 & off_top5)
        am = (sg_arg == off_arg)
        finite = bool(torch.isfinite(sg).all())
        if not am or not finite or (finite and kl > 5e-4):
            bad += 1
        print(f"  {i:<5} {kl:.3e}    {str(am):<13} {ovlp}/5          "
              f"{off_arg:<10} {sg_arg:<10} {float(sg.abs().max()):<8.2f} "
              f"{float(off.abs().max()):<8.2f}")
        rows.append({"step": i, "kl": kl, "argmax_match": am,
                     "top5_overlap": ovlp, "off_argmax": off_arg, "sg_argmax": sg_arg,
                     "sg_finite": finite})

    print(f"\n  >>> L={L}  N={N}  PASS={'YES' if bad == 0 else 'NO'}  bad_steps={bad}")
    print(f"@@RESULT@@ {json.dumps({'L': L, 'N': N, 'bad_steps': bad, 'rows': rows})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["official", "sglang"], required=True)
    args = ap.parse_args()
    if args.stage == "official":
        stage_official()
    else:
        stage_sglang()


if __name__ == "__main__":
    main()
