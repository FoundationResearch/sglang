"""Convert train_345m.py's final.pt (torch state_dict) into the
safetensors + manifest layout that compare.py expects.

Usage:
    python dev/align/convert_g1_ckpt.py \
        --ckpt dev/align/ckpt_345m_g1/final.pt \
        --config dev/align/config_345m_g1.json \
        --out-weights dev/align/weights_345m_g1/model.safetensors \
        --out-manifest dev/align/manifest_345m_g1.json \
        --final-loss <loss>
"""
import argparse, hashlib, json, os, datetime
from pathlib import Path
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-weights", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--final-loss", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu")
    state = blob["model"] if "model" in blob else blob
    steps = args.steps or int(blob.get("step", 0))
    loss = args.final_loss or float(blob.get("loss_ema", 0.0) or 0.0)

    # safetensors refuses shared storage (tie_word_embeddings ties lm_head to
    # embed). Clone+contiguous every tensor to break aliasing; cast to fp16 to
    # match the existing weights_345m manifest convention.
    clean = {k: v.detach().to(torch.float16).clone().contiguous() for k, v in state.items()}

    out_w = Path(args.out_weights)
    out_w.parent.mkdir(parents=True, exist_ok=True)
    save_file(clean, str(out_w))

    cfg_path = Path(args.config)
    manifest = {
        "config_path": os.path.relpath(cfg_path, REPO),
        "config_sha256": sha256(cfg_path),
        "weights_path": os.path.relpath(out_w, REPO),
        "weights_sha256": sha256(out_w),
        "weights_size_bytes": out_w.stat().st_size,
        "weights_dtype": "float16",
        "training": {"steps": steps, "final_loss": loss, "best_loss": loss},
        "env": {"git_commit": "g1-align", "git_dirty": True},
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "_note": "345M qwen_lhsa G=1 MHA (hsa_heads=16,qk_ratio=1,kv=16) + prior_query, wikitext-103, for G=1 alignment repro",
    }
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[convert] wrote {out_w} ({out_w.stat().st_size/1e6:.1f} MB, {len(clean)} tensors)")
    print(f"[convert] wrote {args.out_manifest}  steps={steps} final_loss={loss:.4f}")


if __name__ == "__main__":
    main()
