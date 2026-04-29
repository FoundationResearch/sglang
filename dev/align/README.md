# `dev/align/` — HSA logits alignment harness

Train a tiny 1-layer OLMo-LHSA reference, save deterministic weights, then
compare the official (HuggingFace-style, `models.DRT.modeling_olmo_lhsa`) vs.
SGLang's HSA inference path on identical weights. Training removes top-k
selection noise that makes random-init alignment unreliable.

## Files

| File              | Purpose                                            | Committed? |
|-------------------|----------------------------------------------------|------------|
| `bootstrap.py`    | veomni mocks + DRT/sglang import patches            | yes        |
| `config.json`     | 1-layer model arch (versioned, SHA tracked)        | yes        |
| `train.py`        | trains, writes `weights/` + `manifest.json`         | yes        |
| `compare.py`      | loads weights into both impls, diffs logits        | yes        |
| `manifest.json`   | provenance (git commit, SHAs, training dynamics)   | yes        |
| `weights/model.safetensors` | trained state_dict (fp16, ~2-30 MB)        | yes        |

## Quickstart

```bash
conda activate alexsg
python dev/align/train.py            # ~1-2 min on H100 with defaults
python dev/align/compare.py
```

`compare.py` refuses to run unless the on-disk weights' SHA-256 matches
`manifest.json`. Pass `--no-verify-sha` to bypass during local iteration.

## When to retrain

Retrain whenever any of these changes:

- HSA selection / kernel logic (`hsa-kernel-main/ops/`, `hsa_backend.py`,
  `selector.py`, `flash_hsa.py`)
- The official reference (`InfiniteLongLM/models/DRT/`)
- Positional encoding (e.g. swap RoPE → ALiBi)
- The `config.json` itself
- The training procedure (loss, optimizer, seed)

The provenance trail is:

1. `manifest.json` records `git_commit`, `config_sha256`, `weights_sha256`.
2. The git history of `manifest.json` + `weights/model.safetensors` gives you
   the timestamp of every retraining round (`git log dev/align/manifest.json`).
3. `compare.py`'s SHA check refuses to silently use a stale checkpoint.

## Recommended workflow when changing HSA

```bash
# 1. Make your HSA change (e.g. swap positional encoding)
$EDITOR python/sglang/srt/models/flash_hsa.py
$EDITOR dev/InfiniteLongLM/models/DRT/modeling_olmo_lhsa.py

# 2. (Optional) update config if the change adds new fields
$EDITOR dev/align/config.json

# 3. Retrain — must be from a clean tree to get a meaningful git_commit
git status              # should be clean (or pass --allow-dirty)
python dev/align/train.py

# 4. Compare
python dev/align/compare.py

# 5. Commit weights + manifest together with the HSA change
git add dev/align/manifest.json dev/align/weights/model.safetensors \
        dev/align/config.json   # if changed
git commit -m "align: retrain after <change description>"
```

## Configuration knobs

`config.json` is intentionally tiny so `weights/model.safetensors` stays
committable to git (a few MB at fp16). If you need to bump the architecture
(e.g. test multi-layer behaviour), do so in a *new* `config_<name>.json` and
pass `--config dev/align/config_<name>.json` — keep the default config stable.

The training defaults (`--steps 200 --batch-size 2 --seq-len 384`) are tuned
to drop the LM loss into a regime where the selector has signal but the run
finishes in 1-2 minutes on an H100. Crank `--steps` if you want sharper
alignment.
