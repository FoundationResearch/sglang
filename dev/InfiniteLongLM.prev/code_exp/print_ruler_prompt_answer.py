#!/usr/bin/env python3
"""Print RULER examples and inspect task-5 answer digit statistics."""

import argparse
import os
import random
import re
import sys
from collections import Counter
import torch
import numpy as np
from torch.utils import data as torch_data
from torch.utils.data import SequentialSampler
from transformers import AutoTokenizer


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import RulerSynthesizer, build_numpy_dataset  # noqa: E402
from utils.landmark_utils import insert_special_tokens  # noqa: E402


TASKS = [
    (0, "Single NIAH", "generate_single_niah"),
    (1, "Multi Query", "generate_multi_query"),
    (2, "Variable Tracking", "generate_variable_tracking"),
    (3, "Frequent Words Extraction", "generate_frequent_words_extraction"),
    (4, "PMVL", "generate_positional_multi_value_lookup"),
    (5, "PCVL", "generate_positional_char_in_value_lookup"),
]

DEFAULT_CORPUS_PATH = "/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
DEFAULT_SEQ_LENS = [8192, 16384, 32768, 65536, 131072, 262144, 524288]

FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore. "
    "All work and no play makes Jack a dull boy. "
    "It was the best of times, it was the worst of times. "
    "In a hole in the ground there lived a hobbit. "
    "To be or not to be, that is the question. "
)


def build_filler_ids(tokenizer, length):
    text_parts = []
    ids = []
    while len(ids) < length:
        text_parts.append(FILLER_TEXT)
        ids = tokenizer.encode("".join(text_parts), add_special_tokens=False)
    return np.array(ids[:length], dtype=np.int64)


def format_text(text, max_chars):
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    omitted = len(text) - keep * 2
    return f"{text[:keep]}\n... <{omitted} chars omitted> ...\n{text[-keep:]}"


def parse_task_ids(raw):
    if raw.lower() == "all":
        return [task_id for task_id, _, _ in TASKS]
    task_ids = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            task_ids.append(int(item))
    return task_ids


def parse_seq_len(value):
    raw = value.strip().lower()
    if raw.endswith("k"):
        return int(raw[:-1]) * 1024
    if raw.endswith("m"):
        return int(raw[:-1]) * 1024 * 1024
    return int(raw)


def parse_seq_lens(raw):
    if raw.lower() == "default":
        return DEFAULT_SEQ_LENS
    return [parse_seq_len(item) for item in raw.split(",") if item.strip()]


def seq_len_name(value):
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(0)
    return tokenizer


def print_examples(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer)
    inputs = build_filler_ids(tokenizer, args.length)
    synth = RulerSynthesizer(tokenizer, enable_ruler_plus=True)
    selected_task_ids = set(parse_task_ids(args.tasks))

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Full sequence length: {args.length}")
    print(f"Tasks: {','.join(str(x) for x in sorted(selected_task_ids))}")

    for task_id, task_name, fn_name in TASKS:
        if task_id not in selected_task_ids:
            continue

        print("\n" + "=" * 100)
        print(f"Task {task_id}: {task_name} ({fn_name})")
        print("=" * 100)

        fn = getattr(synth, fn_name)
        try:
            full_ids, prompt_ids, answer_ids = fn(inputs.copy())
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            continue

        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
        full_text = tokenizer.decode(full_ids, skip_special_tokens=False)

        print(f"Token lengths: full={len(full_ids)}, prompt={len(prompt_ids)}, answer={len(answer_ids)}")
        print("\n[PROMPT]")
        if args.full_prompt:
            print(prompt_text)
        else:
            print(format_text(prompt_text, args.max_prompt_chars))
        print("\n[ANSWER]")
        print(repr(answer_text))
        print("\n[FULL TAIL]")
        print(format_text(full_text[-1000:], args.max_prompt_chars))


def extract_task5_answer_digit(tokenizer, answer_ids):
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    digits = re.findall(r"\d", answer_text)
    if not digits:
        return None, answer_text
    return digits[0], answer_text


def collect_task5_digit_stats_for_length(args, tokenizer, max_seq_len):
    dataset = build_numpy_dataset(args.corpus_path, max_seq_len, namespace="test")
    ruler_synthesizer = RulerSynthesizer(
        tokenizer,
        task_id=5,
        enable_ruler_plus=True,
    )
    dataloader = torch_data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=ruler_synthesizer.single_token_eval_collate_fn,
        sampler=SequentialSampler(dataset),
        num_workers=args.num_workers,
    )

    counter = Counter()
    bad_examples = []
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if args.max_samples > 0 and batch_idx >= args.max_samples:
            break

        labels = batch["labels"][0].tolist()
        digit, answer_text = extract_task5_answer_digit(tokenizer, labels)
        if digit is None:
            bad_examples.append((batch_idx, labels, answer_text))
        else:
            counter[digit] += 1
        total_samples += 1

    return counter, total_samples, bad_examples


def print_digit_stats(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer)
    seq_lens = parse_seq_lens(args.seq_lens)
    digits = [str(i) for i in range(10)]

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Corpus path: {args.corpus_path}")
    print(f"Task: 5 (PCVL)")
    print(f"Max samples per length: {args.max_samples}")
    print(f"Seq lens: {' '.join(seq_len_name(x) for x in seq_lens)}")
    print()

    header = ["Length", "Samples"] + digits
    print("\t".join(header))

    for max_seq_len in seq_lens:
        counter, total_samples, bad_examples = collect_task5_digit_stats_for_length(args, tokenizer, max_seq_len)
        row = [seq_len_name(max_seq_len), str(total_samples)] + [str(counter[digit]) for digit in digits]
        print("\t".join(row))

        if bad_examples:
            print(f"Warning: {seq_len_name(max_seq_len)} has {len(bad_examples)} answers without digits", file=sys.stderr)
            for sample_idx, labels, answer_text in bad_examples[:3]:
                print(
                    f"  sample={sample_idx}, labels={labels}, decoded={answer_text!r}",
                    file=sys.stderr,
                )


def _decode_ids(tokenizer, ids):
    return tokenizer.decode([int(x) for x in ids], skip_special_tokens=False)


def _non_special_answer_ids(tokenizer, labels):
    ids = [int(x) for x in labels]
    if ids and tokenizer.eos_token_id is not None and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    return ids


def _current_eval_valid_label(input_ids, labels, insert_lmk, chunk_size):
    answer_len = labels.shape[1]
    if insert_lmk:
        orig_seq_len = input_ids.shape[1]
        orig_answer_start = orig_seq_len - answer_len
        label_ids = input_ids.clone()
        inserted_input_ids = insert_special_tokens(input_ids, fill_id=-999999, chunk_size=chunk_size)
        label_ids = torch.roll(label_ids, shifts=-1, dims=-1)
        label_ids[:, -1] = -100
        label_ids = insert_special_tokens(label_ids, fill_id=-100, chunk_size=chunk_size)
        label_ids = torch.roll(label_ids, shifts=1, dims=-1)
        answer_start_with_lmk = orig_answer_start + (orig_answer_start // (chunk_size - 1))
        answer_len_with_lmk = inserted_input_ids.shape[1] - answer_start_with_lmk
        answer_labels = label_ids[:, -answer_len_with_lmk:]
        valid_mask = answer_labels != -100
        valid_label = answer_labels[valid_mask][:-1]
        debug = {
            "orig_seq_len": orig_seq_len,
            "orig_answer_start": int(orig_answer_start),
            "orig_answer_start_mod": int(orig_answer_start % (chunk_size - 1)),
            "inserted_seq_len": int(inserted_input_ids.shape[1]),
            "answer_start_with_lmk": int(answer_start_with_lmk),
            "answer_len_with_lmk": int(answer_len_with_lmk),
            "raw_answer_labels": answer_labels.flatten().tolist(),
        }
    else:
        answer_labels = input_ids[:, -answer_len:]
        valid_label = answer_labels.flatten()[:-1]
        debug = {
            "orig_seq_len": int(input_ids.shape[1]),
            "orig_answer_start": int(input_ids.shape[1] - answer_len),
            "orig_answer_start_mod": None,
            "inserted_seq_len": int(input_ids.shape[1]),
            "answer_start_with_lmk": int(input_ids.shape[1] - answer_len),
            "answer_len_with_lmk": int(answer_len),
            "raw_answer_labels": answer_labels.flatten().tolist(),
        }
    return valid_label.tolist(), debug


def _proposed_valid_label_and_positions(labels, orig_seq_len, insert_lmk, chunk_size):
    answer_len = labels.shape[1]
    orig_answer_start = orig_seq_len - answer_len
    orig_answer_token_pos = torch.arange(orig_answer_start, orig_answer_start + answer_len, dtype=torch.long)
    orig_logit_pos = orig_answer_token_pos - 1
    if insert_lmk:
        answer_logit_pos = orig_logit_pos + (orig_logit_pos // (chunk_size - 1))
    else:
        answer_logit_pos = orig_logit_pos
    valid_label = labels.flatten()[:-1]
    return valid_label.tolist(), answer_logit_pos.tolist()


def _get_one_eval_batch(dataset, tokenizer, task_id, num_workers):
    ruler_synthesizer = RulerSynthesizer(
        tokenizer,
        task_id=task_id,
        enable_ruler_plus=task_id in (4, 5),
    )
    dataloader = torch_data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=ruler_synthesizer.single_token_eval_collate_fn,
        sampler=SequentialSampler(dataset),
        num_workers=num_workers,
    )
    return next(iter(dataloader))


def print_eval_label_debug(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer)
    seq_lens = parse_seq_lens(args.label_debug_seq_lens)
    task_ids = parse_task_ids(args.label_debug_tasks)
    insert_modes = []
    for item in args.label_debug_insert_modes.split(","):
        item = item.strip().lower()
        if item in ("0", "false", "no", "none", "no_lmk"):
            insert_modes.append(False)
        elif item in ("1", "true", "yes", "lmk", "insert_lmk"):
            insert_modes.append(True)
        elif item == "both":
            insert_modes.extend([False, True])
        elif item:
            raise ValueError(f"Unsupported insert mode: {item}")
    if not insert_modes:
        insert_modes = [False, True]

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Corpus path: {args.corpus_path}")
    print(f"Tasks: {','.join(str(x) for x in task_ids)}")
    print(f"Seq lens: {' '.join(seq_len_name(x) for x in seq_lens)}")
    print(f"Insert modes: {', '.join('insert_lmk' if x else 'no_lmk' for x in insert_modes)}")
    print(f"Chunk size: {args.chunk_size}")

    for max_seq_len in seq_lens:
        dataset = build_numpy_dataset(args.corpus_path, max_seq_len, namespace="test")
        for task_id in task_ids:
            batch = _get_one_eval_batch(dataset, tokenizer, task_id, args.num_workers)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            gt_valid = _non_special_answer_ids(tokenizer, labels[0].tolist())
            gt_text = _decode_ids(tokenizer, gt_valid)
            answer_text = tokenizer.decode(labels[0].tolist(), skip_special_tokens=False)

            print("\n" + "=" * 100)
            print(f"Length={seq_len_name(max_seq_len)} task_id={task_id}")
            print(f"Original seq len={input_ids.shape[1]} answer_len={labels.shape[1]}")
            print(f"Answer ids full: {labels[0].tolist()}")
            print(f"Answer text full: {answer_text!r}")
            print(f"Ground-truth valid ids: {gt_valid}")
            print(f"Ground-truth valid text: {gt_text!r}")

            for insert_lmk in insert_modes:
                current_ids, current_debug = _current_eval_valid_label(
                    input_ids,
                    labels,
                    insert_lmk=insert_lmk,
                    chunk_size=args.chunk_size,
                )
                proposed_ids, proposed_logit_pos = _proposed_valid_label_and_positions(
                    labels,
                    orig_seq_len=input_ids.shape[1],
                    insert_lmk=insert_lmk,
                    chunk_size=args.chunk_size,
                )

                current_ok = current_ids == gt_valid
                proposed_ok = proposed_ids == gt_valid
                mode_name = "insert_lmk" if insert_lmk else "no_lmk"

                print("-" * 100)
                print(f"Mode: {mode_name}")
                print(f"  orig_answer_start={current_debug['orig_answer_start']} mod={current_debug['orig_answer_start_mod']}")
                print(f"  inserted_seq_len={current_debug['inserted_seq_len']}")
                print(f"  current answer_start_with_lmk={current_debug['answer_start_with_lmk']}")
                print(f"  current answer_len_with_lmk={current_debug['answer_len_with_lmk']}")
                print(f"  current raw answer_labels: {current_debug['raw_answer_labels']}")
                print(f"  current valid ids:   {current_ids}")
                print(f"  current valid text:  {_decode_ids(tokenizer, current_ids)!r}")
                print(f"  current matches GT?  {current_ok}")
                print(f"  proposed valid ids:  {proposed_ids}")
                print(f"  proposed valid text: {_decode_ids(tokenizer, proposed_ids)!r}")
                print(f"  proposed matches GT? {proposed_ok}")
                print(f"  proposed logit pos:  {proposed_logit_pos}")


def main():
    parser = argparse.ArgumentParser(description="Print RULER examples or task-5 answer digit statistics")
    parser.add_argument("--tokenizer", default=os.path.join(REPO_ROOT, "configs", "olmo3_vocab"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--length", type=int, default=4096, help="Synthetic full sequence token length")
    parser.add_argument("--tasks", default="all", help="Comma-separated task ids, or all")
    parser.add_argument("--max-prompt-chars", type=int, default=3000)
    parser.add_argument("--full-prompt", action="store_true")

    parser.add_argument("--digit-stats", action="store_true", help="Count task-5 answer digits using the eval_ruler data pipeline")
    parser.add_argument("--task4-probe", action="store_true", help="Probe task-4 answer token structure (BPE boundary, value token length, etc.)")
    parser.add_argument("--eval-label-debug", action="store_true", help="Compare current eval_ruler label slicing with the proposed position-gather label logic")
    parser.add_argument("--corpus-path", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--seq-lens", default="default", help="Comma-separated lengths, e.g. 8k,16k,32k, or default")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10, help="Top-k frequent tokens to show in task-4 probe")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size used by insert_lmk label debug")
    parser.add_argument("--label-debug-tasks", default="0,1,2,4,5", help="Task ids for --eval-label-debug")
    parser.add_argument("--label-debug-seq-lens", default="8k,16k", help="Seq lens for --eval-label-debug")
    parser.add_argument("--label-debug-insert-modes", default="both", help="no_lmk,insert_lmk,both or comma-separated aliases")

    args = parser.parse_args()

    if args.digit_stats:
        print_digit_stats(args)
    elif args.task4_probe:
        print_task4_token_probe(args)
    elif args.eval_label_debug:
        print_eval_label_debug(args)
    else:
        print_examples(args)


if __name__ == "__main__":
    main()
