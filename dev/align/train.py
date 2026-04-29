"""
Train a small 1-layer OLMo-LHSA model and persist weights + provenance manifest.

The whole point: the HSA top-k page selector is non-differentiable in a way that
makes random-init alignment unreliable — two implementations with identical
weights but tiny numerical differences in the score computation can pick
disjoint top-k pages, blowing up the comparison. Even a few hundred training
steps on synthetic data is enough to give selection scores meaningful signal,
so top-k becomes robust under small perturbations.

Usage:
    python dev/align/train.py                 # defaults
    python dev/align/train.py --steps 500     # longer training
    python dev/align/train.py --config dev/align/config.json --seed 0

Outputs:
    dev/align/weights/model.safetensors  — trained state_dict
    dev/align/manifest.json              — provenance: git commit, config sha,
                                           weights sha, seed, steps, loss curve

Re-train any time the HSA implementation, config, or kernel changes — the
manifest's git_commit field plus the file's git history give you the timestamp
trail.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# bootstrap MUST come before any torch / DRT / sglang import
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap  # noqa: E402,F401  side-effect import (mocks)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

OfficialConfig = bootstrap.OfficialConfig
OfficialModel = bootstrap.OfficialModel
insert_special_tokens = bootstrap.insert_special_tokens
create_position_ids_with_landmarks = bootstrap.create_position_ids_with_landmarks


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ['git', '-C', str(HERE.parents[1]), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ['git', '-C', str(HERE.parents[1]), 'status', '--porcelain'],
            stderr=subprocess.DEVNULL, text=True,
        )
        return bool(out.strip())
    except Exception:
        return True


def build_official(cfg_dict: dict, device: torch.device, dtype: torch.dtype):
    """Build official OLMo-LHSA from config dict. Drops `_comment`-prefixed keys."""
    kwargs = {k: v for k, v in cfg_dict.items() if not k.startswith('_')}
    cfg = OfficialConfig(**kwargs)
    cfg._attn_implementation = 'eager'
    cfg.pad_token_id = None
    if not hasattr(cfg, 'num_swa_layers'):
        cfg.num_swa_layers = 0
    model = OfficialModel(cfg).to(device=device, dtype=dtype)
    return cfg, model


def synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, page_size: int,
                    device: torch.device, generator: torch.Generator):
    """One synthetic batch: random tokens in [5, vocab_size-5), with the LMK id
    inserted at every page boundary (matching how the engine sees the sequence).

    Returns (input_ids_with_lmk, position_ids, labels_with_ignore_at_lmk).
    Loss is masked at LMK positions because LMK tokens are model-internal and
    shouldn't contribute to LM loss."""
    real = torch.randint(
        low=5, high=vocab_size - 5,
        size=(batch_size, seq_len),
        generator=generator, device=device,
    )
    # insert_special_tokens expects a [B, L] LongTensor on CPU (it uses python ints)
    ids_with_lmk = insert_special_tokens(real.cpu(), vocab_size, page_size).to(device)
    pos = create_position_ids_with_landmarks(seq_len, page_size, device).expand(batch_size, -1)
    # Labels: shifted ids; ignore LMK positions (id == vocab_size)
    labels = ids_with_lmk.clone()
    labels[ids_with_lmk == vocab_size] = -100
    return ids_with_lmk, pos, labels


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token CE with shift, ignoring positions labeled -100."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(HERE / 'config.json'))
    parser.add_argument('--out-weights', default=str(HERE / 'weights' / 'model.safetensors'))
    parser.add_argument('--out-manifest', default=str(HERE / 'manifest.json'))
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=384,
                        help='Real (pre-LMK) sequence length per sample.')
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--save-dtype', default='float16',
                        help='Dtype for saved weights (small file size).')
    parser.add_argument('--allow-dirty', action='store_true',
                        help='Skip the clean-tree check (manifest will note dirty=true).')
    args = parser.parse_args()

    bootstrap.init_sglang_dist()  # not strictly needed for training, but cheap and consistent

    cfg_path = Path(args.config).resolve()
    with cfg_path.open('rb') as fh:
        cfg_bytes = fh.read()
    config_sha = _sha256_bytes(cfg_bytes)
    cfg_dict = json.loads(cfg_bytes)

    git_commit = _git_commit()
    is_dirty = _git_dirty()
    if is_dirty and not args.allow_dirty:
        print('[train] working tree is dirty — pass --allow-dirty to proceed; '
              'manifest will record dirty=true.', file=sys.stderr)
    print(f'[train] config={cfg_path.name} config_sha256={config_sha[:16]}…')
    print(f'[train] git_commit={git_commit} dirty={is_dirty}')

    device = torch.device(args.device)
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    save_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.save_dtype]

    # Determinism — best effort. tilelang JIT may inject non-determinism; that's
    # acceptable here since manifest fixes weights_sha256 to whatever was saved.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    cfg, model = build_official(cfg_dict, device=device, dtype=dtype)
    # NOTE: keep eval() mode — the model's training-mode forward path calls a
    # 4-arg create_position_ids_with_landmarks that doesn't exist in this
    # version of InfiniteLongLM/utils/. We pre-insert LMK in synthetic_batch,
    # so eval mode is correct and avoids that broken branch. Dropout is off,
    # which is fine and actually preferable for deterministic alignment runs.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[train] built 1-layer OLMo-LHSA: hidden={cfg.hidden_size} '
          f'kv_heads={cfg.num_key_value_heads} hsa_heads={cfg.hsa_heads} '
          f'page={cfg.chunk_size} topk={cfg.hsa_topk}  params={n_params/1e6:.2f}M')

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    losses: list[float] = []
    for step in range(1, args.steps + 1):
        input_ids, position_ids, labels = synthetic_batch(
            args.batch_size, args.seq_len, cfg.vocab_size, cfg.chunk_size,
            device=device, generator=gen,
        )
        out = model(input_ids=input_ids, position_ids=position_ids,
                    attention_mask=None, use_cache=False)
        loss = lm_loss(out.logits.float(), labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(float(loss.item()))
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f'  step {step:4d}/{args.steps}  loss={loss.item():.4f}')

    # Save weights — convert to save_dtype to keep the file tiny.
    out_w = Path(args.out_weights).resolve()
    out_w.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to(save_dtype).cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out_w))
    weights_sha = _sha256(out_w)
    weights_size = out_w.stat().st_size
    print(f'[train] saved weights: {out_w}  ({weights_size/1e6:.2f} MB, sha={weights_sha[:16]}…)')

    # Manifest — careful about types; everything must be JSON-serializable.
    import torch as _torch
    try:
        import tilelang as _tl
        tl_ver = _tl.__version__
    except Exception:
        tl_ver = None
    try:
        import transformers as _tx
        tx_ver = _tx.__version__
    except Exception:
        tx_ver = None
    try:
        import triton as _tt
        tt_ver = _tt.__version__
    except Exception:
        tt_ver = None

    manifest = {
        'config_path': str(cfg_path.relative_to(HERE.parents[1])),
        'config_sha256': config_sha,
        'weights_path': str(out_w.relative_to(HERE.parents[1])),
        'weights_sha256': weights_sha,
        'weights_size_bytes': weights_size,
        'weights_dtype': str(save_dtype).replace('torch.', ''),
        'training': {
            'seed': args.seed,
            'steps': args.steps,
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
            'lr': args.lr,
            'optimizer': 'AdamW(betas=(0.9,0.95), wd=0.01, clip=1.0)',
            'dtype': args.dtype,
            'final_loss': losses[-1],
            'best_loss': min(losses),
            'loss_first': losses[0],
            'loss_curve_p50_p90_p99': [
                float(_torch.tensor(losses).quantile(0.5)),
                float(_torch.tensor(losses).quantile(0.9)),
                float(_torch.tensor(losses).quantile(0.99)),
            ],
        },
        'env': {
            'git_commit': git_commit,
            'git_dirty': is_dirty,
            'torch': _torch.__version__,
            'cuda': _torch.version.cuda,
            'device_name': _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
            'tilelang': tl_ver,
            'transformers': tx_ver,
            'triton': tt_ver,
            'python': sys.version.split()[0],
        },
        'trained_at_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    out_m = Path(args.out_manifest).resolve()
    out_m.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'[train] wrote manifest: {out_m}')
    print(f'[train] final loss={losses[-1]:.4f} (started at {losses[0]:.4f})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
