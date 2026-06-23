"""Minimal training loop for the 345M Qwen-LHSA config on wikitext-103.

Bypasses veomni: reuses the veomni mocks from dev/align/bootstrap.py, loads
HSAForCausalLM (Qwen variant) directly, builds an AdamW + cosine schedule,
streams uint32 .data files, logs to wandb. DDP across all visible GPUs.

Launch (single node):
    torchrun --standalone --nproc-per-node=4 dev/align/train_345m.py \\
        --config dev/InfiniteLongLM/configs/flash_hsa/config_hsa_8KA2K_HoPE_345M_lmk_bias_priorq_wloralmkq_loradim64.json \\
        --data /home/hal-alex/workspace/hsa_train/wikitext103_tokenized \\
        --steps 10000 --seq-len 8192 --micro-bs 4 --grad-accum 2

Each global step processes micro_bs * grad_accum * world_size sequences.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# bootstrap MUST come first — it sets up veomni mocks before the FlashHSA
# imports below try to resolve veomni symbols.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402


def setup_dist():
    if dist.is_initialized():
        return
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, world, local_rank


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log0(*a, **kw):
    if is_main():
        print(*a, **kw, flush=True)


class ShardedTokenStream:
    """Reads all .data files in a directory (uint32 memmap, one stream per
    file), concatenates virtually, and yields contiguous seq_len windows from
    random offsets. Per-rank shard via stride.
    """

    def __init__(self, data_dir: str, seq_len: int, rank: int, world: int, seed: int = 0):
        files = sorted(Path(data_dir).glob("*.data"))
        if not files:
            raise FileNotFoundError(f"no .data files in {data_dir}")
        self.maps = [np.memmap(f, dtype=np.uint32, mode="r") for f in files]
        self.lens = [m.shape[0] for m in self.maps]
        self.cumlens = np.cumsum(self.lens)
        self.total = int(self.cumlens[-1])
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed + rank * 7919)
        self.rank = rank
        self.world = world
        log0(f"[data] {len(files)} files  total_tokens={self.total:,}  rank={rank}/{world}")

    def _read_window(self, start: int) -> np.ndarray:
        end = start + self.seq_len + 1  # +1 for labels-shift
        # Find which file holds start
        file_idx = int(np.searchsorted(self.cumlens, start, side="right"))
        if end <= self.cumlens[file_idx]:
            local_start = start - (self.cumlens[file_idx - 1] if file_idx > 0 else 0)
            return np.asarray(self.maps[file_idx][local_start : local_start + self.seq_len + 1])
        # Spans a boundary — fall back: wrap to next file's beginning
        local_start = start - (self.cumlens[file_idx - 1] if file_idx > 0 else 0)
        first = self.maps[file_idx][local_start:]
        remaining = self.seq_len + 1 - first.shape[0]
        next_idx = (file_idx + 1) % len(self.maps)
        second = self.maps[next_idx][:remaining]
        return np.concatenate([first, second])

    def sample_batch(self, micro_bs: int) -> torch.Tensor:
        # max valid start: total - seq_len - 1
        max_start = self.total - self.seq_len - 2
        starts = self.rng.integers(0, max_start, size=micro_bs)
        wins = [self._read_window(int(s)) for s in starts]
        ids = np.stack(wins)  # [B, seq_len+1]
        return torch.from_numpy(ids.astype(np.int64))


def build_model(config_path: str, device: torch.device):
    """Build the Qwen-LHSA HSAForCausalLM from the JSON config."""
    # Pick the variant by config.model_type
    import json as _json
    with open(config_path) as f:
        cfg_raw = _json.load(f)
    model_type = cfg_raw.get("model_type", "olmo_lhsa")

    from models.FlashHSA.configuration_hsa import HSAConfig

    if model_type == "qwen_lhsa":
        from models.FlashHSA.modeling_qwen_lhsa import HSAForCausalLM
    else:
        from models.FlashHSA.modeling_olmo_lhsa import HSAForCausalLM

    cfg_raw = {k: v for k, v in cfg_raw.items() if not k.startswith("_")}
    # Force eager attention impl (we don't have flash-attn-3 wired up here).
    cfg_raw["_attn_implementation"] = "eager"
    cfg = HSAConfig(**cfg_raw)
    cfg.pad_token_id = None
    if not hasattr(cfg, "num_swa_layers"):
        cfg.num_swa_layers = 0

    model = HSAForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    return cfg, model


def cosine_lr(step: int, max_steps: int, base_lr: float, min_lr: float, warmup_ratio: float = 0.01) -> float:
    warmup = max(1, int(max_steps * warmup_ratio))
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="dir of .data uint32 files")
    ap.add_argument("--out", default="/home/hal-alex/workspace/hsa_train/ckpt/run")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="ruler_pretrain_5per_345M")
    ap.add_argument("--wandb-name", default="hsa_8KA2K_HoPE-priorq-wloralmkq-loradim64")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--overfit", action="store_true",
                    help="Reuse ONE fixed batch every step so the model memorises it "
                         "(loss->0, sharp/varied greedy output). For a non-trivial "
                         "CG-vs-eager consistency test on a from-scratch model that "
                         "would otherwise collapse to a high-frequency token.")
    args = ap.parse_args()

    rank, world, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    os.makedirs(args.out, exist_ok=True)

    log0(f"[init] world={world} rank={rank} local={local_rank} device={device}")
    log0(f"[init] config={args.config}")
    log0(f"[init] data={args.data}")
    log0(f"[init] out={args.out}")
    log0(f"[init] steps={args.steps} seq_len={args.seq_len} micro_bs={args.micro_bs} "
         f"grad_accum={args.grad_accum} -> global_bs={args.micro_bs * args.grad_accum * world}")

    cfg, model = build_model(args.config, device)
    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        log0(f"[model] params={n_params/1e6:.2f}M  vocab={cfg.vocab_size}  "
             f"layers={cfg.num_hidden_layers} hidden={cfg.hidden_size}  "
             f"hsa_heads={getattr(cfg,'hsa_heads',None)} hsa_qk_ratio={getattr(cfg,'hsa_qk_ratio',None)}  "
             f"chunk_size={cfg.chunk_size} topk={cfg.hsa_topk}")

    model.train()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    no_decay = {"bias"}
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nm = n.split(".")[-1]
        if "norm" in n.lower() or "embed" in n.lower() or nm in no_decay:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95),
    )

    stream = ShardedTokenStream(args.data, args.seq_len, rank, world, seed=args.seed)

    use_wandb = (not args.no_wandb) and is_main()
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   config={**vars(args),
                           "model.params_M": n_params / 1e6,
                           "model.vocab_size": cfg.vocab_size,
                           "model.layers": cfg.num_hidden_layers,
                           "model.hsa_heads": getattr(cfg, "hsa_heads", None),
                           "model.hsa_qk_ratio": getattr(cfg, "hsa_qk_ratio", None),
                           "world_size": world})

    # --overfit: one fixed batch (per rank), reused every step.
    fixed_ids = None
    if args.overfit:
        fixed_ids = [stream.sample_batch(args.micro_bs).to(device) for _ in range(args.grad_accum)]
        log0(f"[train] --overfit: memorising {args.grad_accum} fixed micro-batch(es) every step")
        if is_main():
            import json as _json
            mem = fixed_ids[0][0].tolist()  # first sequence of the first micro-batch
            os.makedirs(args.out, exist_ok=True)
            _json.dump({"input_ids": mem}, open(os.path.join(args.out, "memorized_ids.json"), "w"))
            log0(f"[train] dumped memorized sequence ({len(mem)} tokens) -> {args.out}/memorized_ids.json")

    t0 = time.time()
    loss_ema = None
    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.lr, args.lr_min)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        step_loss_sum = 0.0
        for accum in range(args.grad_accum):
            ids = (fixed_ids[accum] if fixed_ids is not None
                   else stream.sample_batch(args.micro_bs).to(device, non_blocking=True))
            # input/label shift
            input_ids = ids[:, :-1].contiguous()
            labels = ids[:, 1:].contiguous()

            is_last = (accum == args.grad_accum - 1)
            ctx = model.no_sync() if (not is_last and world > 1) else torch.enable_grad()
            with ctx:
                out = model(input_ids=input_ids, labels=None,
                            attention_mask=None, use_cache=False)
                logits = out.logits  # [B, L', V] — L' may differ from L due to LMK insertion
                # Align labels: HSAForCausalLM may have inserted LMK tokens.
                # When insert_landmarks=True and self.training, the model
                # internally extends input_ids with LMK ids at chunk boundaries.
                # Logits shape becomes [B, L_with_lmk, V]. To compute LM loss
                # we shift logits/labels to match: take logits[:, :L-1] vs
                # labels[:, 1:]. If insert added tokens, just truncate to min.
                L_logits = logits.shape[1]
                L_labels = labels.shape[1]
                Lmin = min(L_logits, L_labels)
                logits_use = logits[:, :Lmin].reshape(-1, logits.shape[-1])
                labels_use = labels[:, :Lmin].reshape(-1)
                loss = F.cross_entropy(logits_use.float(), labels_use, ignore_index=-100)
                (loss / args.grad_accum).backward()
                step_loss_sum += loss.detach().float().item()

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        avg_loss = step_loss_sum / args.grad_accum
        loss_ema = avg_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * avg_loss

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            tps = (args.micro_bs * args.grad_accum * world * args.seq_len) * step / dt
            log0(f"step {step:5d}/{args.steps}  loss={avg_loss:.4f}  ema={loss_ema:.4f}  "
                 f"lr={lr:.2e}  {tps/1000:.1f}K tok/s  elapsed={dt/60:.1f}min")
            if use_wandb:
                import wandb as _wb
                _wb.log({"train/loss": avg_loss, "train/loss_ema": loss_ema, "train/lr": lr,
                         "train/tok_per_s": tps, "train/elapsed_min": dt / 60}, step=step)

        if step % args.save_every == 0 and is_main():
            save_dir = os.path.join(args.out, f"step_{step}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save({"model": model.module.state_dict(),
                        "step": step, "loss_ema": loss_ema, "config_path": args.config},
                       os.path.join(save_dir, "ckpt.pt"))
            log0(f"[save] -> {save_dir}/ckpt.pt")

    if is_main():
        torch.save({"model": model.module.state_dict(), "step": args.steps},
                   os.path.join(args.out, "final.pt"))
        log0(f"[done] saved final to {args.out}/final.pt")
    if use_wandb:
        import wandb as _wb
        _wb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
