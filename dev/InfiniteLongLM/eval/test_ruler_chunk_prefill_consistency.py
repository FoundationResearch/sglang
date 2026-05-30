import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import SequentialSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data import RulerSynthesizer, build_numpy_dataset
from models.FlashHSA.configuration_hsa import HSAConfig
from utils.landmark_utils import create_position_ids_with_landmarks, insert_special_tokens


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_model_type(config_path: Optional[str], checkpoint_path: Optional[str]) -> str:
    paths = []
    if config_path:
        paths.append(config_path)
    if checkpoint_path:
        paths.append(os.path.join(checkpoint_path, "config.json"))
    for path in paths:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("model_type", "")
    return ""


def resolve_hsa_class(config_path: Optional[str], checkpoint_path: Optional[str]):
    model_type = _read_model_type(config_path, checkpoint_path)
    if "olmo" in model_type:
        from models.FlashHSA.modeling_olmo_lhsa import HSAForCausalLM

        print(f"Using OLMo LHSA implementation (model_type={model_type})")
    else:
        from models.FlashHSA.modeling_qwen_lhsa import HSAForCausalLM

        print(f"Using Qwen LHSA implementation (model_type={model_type or '<unknown>'})")
    return HSAForCausalLM, model_type


def register_hsa_model(config_path: Optional[str], checkpoint_path: Optional[str]):
    HSAForCausalLM, model_type = resolve_hsa_class(config_path, checkpoint_path)
    HSAForCausalLM.config_class = HSAConfig
    for name in {model_type, "olmo_lhsa", "flash_hsa"}:
        if not name:
            continue
        try:
            AutoConfig.register(name, HSAConfig)
        except ValueError:
            pass
    try:
        AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM)
    except ValueError:
        pass


def load_model(args, device):
    if args.insert_lmk and args.auto_insert_lmk:
        raise ValueError("--insert_lmk and --auto_insert_lmk are mutually exclusive")

    register_hsa_model(args.config_path, args.checkpoint_path)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": args.attn_implementation,
        "device_map": device,
    }

    if args.checkpoint_path:
        if args.auto_insert_lmk:
            model_kwargs["auto_insert_lmk"] = True
        if args.config_path:
            config = AutoConfig.from_pretrained(args.config_path)
            if args.auto_insert_lmk:
                config.auto_insert_lmk = True
            elif args.insert_lmk:
                config.auto_insert_lmk = False
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, config=config, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **model_kwargs)
    else:
        if args.config_path is None:
            raise ValueError("Either --checkpoint_path or --config_path is required")
        config = AutoConfig.from_pretrained(args.config_path)
        config.auto_insert_lmk = bool(args.auto_insert_lmk)
        if args.insert_lmk:
            config.auto_insert_lmk = False
        model = AutoModelForCausalLM.from_config(config, **model_kwargs).to(device)

    if args.insert_lmk:
        if hasattr(model, "auto_insert_lmk"):
            model.auto_insert_lmk = False
        if hasattr(model, "config"):
            model.config.auto_insert_lmk = False
    model.eval()
    return model


@contextmanager
def auto_chunk_threshold(threshold: int):
    import models.FlashHSA.chunk_prefill as chunk_prefill_module

    saved = chunk_prefill_module.DEFAULT_CHUNK_PREFILL_THRESHOLD
    chunk_prefill_module.DEFAULT_CHUNK_PREFILL_THRESHOLD = threshold
    try:
        yield
    finally:
        chunk_prefill_module.DEFAULT_CHUNK_PREFILL_THRESHOLD = saved


def build_dataloader(args, tokenizer):
    dataset = build_numpy_dataset(args.corpus_path, args.max_seq_len, namespace="test")
    task_kwargs = {}
    if args.needle_len > 0:
        task_kwargs["length"] = args.needle_len
    if args.total_var > 0:
        task_kwargs["total_var"] = args.total_var
    if args.num_queries > 0:
        task_kwargs["num_queries"] = args.num_queries
    ruler_synthesizer = RulerSynthesizer(tokenizer, task_id=args.task_id, **task_kwargs)
    return data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=ruler_synthesizer.single_token_eval_collate_fn,
        sampler=SequentialSampler(dataset),
        num_workers=args.num_workers,
    )


def prepare_sample(batch, args, model, tokenizer, device) -> Dict[str, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    answer_len = labels.shape[1]
    chunk_size = getattr(model.config, "chunk_size", 64)
    lmk_id = tokenizer.vocab_size
    pos_ids = None

    if args.insert_lmk:
        orig_seq_len = input_ids.shape[1]
        orig_answer_start = orig_seq_len - answer_len
        label_ids = input_ids.clone()

        input_ids = insert_special_tokens(input_ids, fill_id=lmk_id, chunk_size=chunk_size)
        label_ids = torch.roll(label_ids, shifts=-1, dims=-1)
        label_ids[:, -1] = -100
        label_ids = insert_special_tokens(label_ids, fill_id=-100, chunk_size=chunk_size)
        label_ids = torch.roll(label_ids, shifts=1, dims=-1)

        if args.adjust_lmk_pos:
            pos_ids = create_position_ids_with_landmarks(
                None, orig_seq_len, chunk_size=chunk_size, device=device
            )

        answer_start = orig_answer_start + (orig_answer_start // (chunk_size - 1))
        answer_len_with_lmk = input_ids.shape[1] - answer_start
    else:
        label_ids = input_ids.clone()
        answer_len_with_lmk = answer_len

    seq_len = input_ids.shape[1]
    answer_logits_start = seq_len - answer_len_with_lmk - 1
    if answer_logits_start < 0:
        raise ValueError(
            f"answer_logits_start={answer_logits_start} is invalid. "
            f"seq_len={seq_len}, answer_len_with_lmk={answer_len_with_lmk}"
        )

    return {
        "input_ids": input_ids,
        "position_ids": pos_ids,
        "label_ids": label_ids,
        "answer_labels": label_ids[:, -answer_len_with_lmk:],
        "answer_len_with_lmk": torch.tensor(answer_len_with_lmk, device=device),
        "answer_logits_start": torch.tensor(answer_logits_start, device=device),
    }


def _position_ids_for_model(sample, args):
    return sample["position_ids"] if args.adjust_lmk_pos else None


def run_full_answer_logits(model, sample, args, internal_threshold: int) -> torch.Tensor:
    input_ids = sample["input_ids"]
    seq_len = input_ids.shape[1]
    answer_len = int(sample["answer_len_with_lmk"].item())
    cache_pos = torch.arange(0, seq_len, device=input_ids.device)
    with auto_chunk_threshold(internal_threshold):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            out = model(
                input_ids=input_ids,
                cache_position=cache_pos,
                use_cache=False,
                logits_to_keep=answer_len + 1,
                position_ids=_position_ids_for_model(sample, args),
            )
    logits = out.logits[:, :-1, :].float().cpu()
    del out
    torch.cuda.empty_cache()
    return logits


def run_manual_segment_answer_logits(model, sample, args) -> torch.Tensor:
    input_ids = sample["input_ids"]
    seq_len = input_ids.shape[1]
    segment_size = args.segment_size
    answer_len = int(sample["answer_len_with_lmk"].item())
    answer_logits_start = int(sample["answer_logits_start"].item())
    first_answer_segment = answer_logits_start // segment_size
    num_segments = (seq_len + segment_size - 1) // segment_size
    pos_ids = _position_ids_for_model(sample, args)

    answer_logits_list = []
    past_key_values = None
    with auto_chunk_threshold(0):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, seq_len)
                seg_input_ids = input_ids[:, start_idx:end_idx]
                seg_cache_pos = torch.arange(start_idx, end_idx, device=input_ids.device)
                seg_pos_ids = pos_ids[:, start_idx:end_idx] if pos_ids is not None else None
                seg_logits_to_keep = end_idx - start_idx if i >= first_answer_segment else 1

                out = model(
                    input_ids=seg_input_ids,
                    cache_position=seg_cache_pos,
                    use_cache=True,
                    past_key_values=past_key_values,
                    logits_to_keep=seg_logits_to_keep,
                    position_ids=seg_pos_ids,
                )
                past_key_values = out.past_key_values
                if i >= first_answer_segment:
                    answer_logits_list.append(out.logits.float().cpu())
                del out

    answer_region_logits = torch.cat(answer_logits_list, dim=1)
    offset = answer_logits_start - first_answer_segment * segment_size
    logits = answer_region_logits[:, offset : offset + answer_len, :]
    del answer_logits_list, answer_region_logits, past_key_values
    torch.cuda.empty_cache()
    return logits


@dataclass
class Metrics:
    max_abs: float
    rmse_ratio: float
    argmax_match: float
    valid_argmax_match: float
    valid_mismatch_count: int


def compare_logits(name: str, ref: torch.Tensor, test: torch.Tensor, answer_labels: torch.Tensor) -> Metrics:
    if ref.shape != test.shape:
        raise AssertionError(f"{name} shape mismatch: ref={tuple(ref.shape)}, test={tuple(test.shape)}")

    diff = ref - test
    max_abs = diff.abs().max().item()
    rmse = diff.square().mean().sqrt().item()
    base = ref.square().mean().sqrt().item()
    rmse_ratio = rmse / (base + 1e-12)
    ref_argmax = ref.argmax(dim=-1)
    test_argmax = test.argmax(dim=-1)
    argmax_match = (ref_argmax == test_argmax).float().mean().item()

    valid_mask = (answer_labels.cpu() != -100)
    valid_indices = valid_mask.flatten().nonzero(as_tuple=False).flatten()
    if valid_indices.numel() > 0:
        # eval_ruler_hf drops the final answer token (usually EOS) before scoring.
        valid_indices = valid_indices[:-1]
    if valid_indices.numel() == 0:
        valid_argmax_match = 1.0
        valid_mismatch_count = 0
    else:
        ref_valid = ref_argmax.flatten()[valid_indices]
        test_valid = test_argmax.flatten()[valid_indices]
        valid_match = ref_valid == test_valid
        valid_argmax_match = valid_match.float().mean().item()
        valid_mismatch_count = int((~valid_match).sum().item())

    print(
        f"[{name}] max_abs={max_abs:.6f}, rmse_ratio={rmse_ratio:.6f}, "
        f"argmax_match={argmax_match * 100:.2f}%, "
        f"valid_argmax_match={valid_argmax_match * 100:.2f}%, "
        f"valid_mismatch={valid_mismatch_count}"
    )
    return Metrics(max_abs, rmse_ratio, argmax_match, valid_argmax_match, valid_mismatch_count)


def check_metrics(name: str, metrics: Metrics, args) -> bool:
    ok = True
    if metrics.rmse_ratio > args.rtol:
        print(f"  ERROR {name}: rmse_ratio {metrics.rmse_ratio:.6f} > {args.rtol}")
        ok = False
    if metrics.valid_argmax_match < args.min_valid_argmax_match:
        print(
            f"  ERROR {name}: valid_argmax_match {metrics.valid_argmax_match:.6f} "
            f"< {args.min_valid_argmax_match}"
        )
        ok = False
    if metrics.max_abs > args.max_abs_tol:
        print(f"  WARN  {name}: max_abs {metrics.max_abs:.6f} > {args.max_abs_tol}")
    return ok


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir, trust_remote_code=True)
    dataloader = build_dataloader(args, tokenizer)

    any_failed = False
    checked = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < args.sample_offset:
            continue
        if checked >= args.max_samples:
            break

        sample = prepare_sample(batch, args, model, tokenizer, device)
        seq_len = sample["input_ids"].shape[1]
        answer_len = int(sample["answer_len_with_lmk"].item())
        print("\n" + "=" * 80)
        print(
            f"Sample {batch_idx}: seq_len={seq_len}, answer_len_with_lmk={answer_len}, "
            f"segment_size={args.segment_size}"
        )

        ref_logits = run_full_answer_logits(model, sample, args, internal_threshold=0)
        print(f"[full] logits shape={tuple(ref_logits.shape)}")

        if args.segment_size > 0:
            manual_logits = run_manual_segment_answer_logits(model, sample, args)
            manual_metrics = compare_logits(
                "manual_segment_prefill",
                ref_logits,
                manual_logits,
                sample["answer_labels"],
            )
            any_failed = (not check_metrics("manual_segment_prefill", manual_metrics, args)) or any_failed
            del manual_logits

        if not args.skip_internal_auto:
            auto_logits = run_full_answer_logits(model, sample, args, internal_threshold=args.internal_threshold)
            auto_metrics = compare_logits(
                f"internal_auto_chunk_threshold_{args.internal_threshold}",
                ref_logits,
                auto_logits,
                sample["answer_labels"],
            )
            any_failed = (not check_metrics("internal_auto_chunk", auto_metrics, args)) or any_failed
            del auto_logits

        del ref_logits, sample
        torch.cuda.empty_cache()
        checked += 1

    if checked == 0:
        raise RuntimeError("No samples were checked. Check --max_samples and --sample_offset.")
    if any_failed:
        raise SystemExit(1)
    print(f"\nAll {checked} sample(s) passed chunk prefill consistency checks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compare RULER full forward vs chunk prefill logits")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--vocab_dir", required=True, type=str)
    parser.add_argument("--corpus_path", required=True, type=str)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--task_id", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--segment_size", type=int, default=4096)
    parser.add_argument("--insert_lmk", action="store_true")
    parser.add_argument("--adjust_lmk_pos", action="store_true")
    parser.add_argument("--auto_insert_lmk", action="store_true")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--needle_len", type=int, default=-1)
    parser.add_argument("--total_var", type=int, default=-1)
    parser.add_argument("--num_queries", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_3")
    parser.add_argument("--internal_threshold", type=int, default=1)
    parser.add_argument("--skip_internal_auto", action="store_true")
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--max_abs_tol", type=float, default=1.0)
    parser.add_argument("--min_valid_argmax_match", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())