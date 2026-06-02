"""Inspect Ruler / Ruler-Plus synthesized samples.

直接运行：
    cd /data/workspace/shanwxxxhu_gz/InfiniteLongLM
    python code_exp/inspect_ruler_synth.py

可选环境变量：
    RULER_TASK=4          # 0..5，指定单个任务；默认遍历所有
    RULER_PLUS=1          # 启用 ruler_plus（任务 4/5 必须）
    RULER_LEN=4096        # 总序列长度
    RULER_TOKENIZER=...   # 覆盖 tokenizer 路径
"""

import os
import sys
import numpy as np
import random

# Make repo importable when run from any cwd
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoTokenizer  # noqa: E402

from data.data_transform import RulerSynthesizer  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOKENIZER_PATH = os.environ.get(
    "RULER_TOKENIZER",
    os.path.join(REPO_ROOT, "configs", "olmo3_vocab"),
)
TOTAL_LEN = int(os.environ.get("RULER_LEN", "4096"))
ENABLE_PLUS = bool(int(os.environ.get("RULER_PLUS", "1")))
ONLY_TASK = os.environ.get("RULER_TASK", None)
SEED = 42

# Some unobtrusive filler text we tile to fill TOTAL_LEN tokens.
FILLER_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore. "
    "All work and no play makes Jack a dull boy. "
    "It was the best of times, it was the worst of times. "
    "In a hole in the ground there lived a hobbit. "
    "To be or not to be, that is the question. "
    "Curiouser and curiouser, cried Alice. "
)

TASK_INFO = [
    (0, "single_niah",                       False, "generate_single_niah"),
    (1, "multi_query",                       False, "generate_multi_query"),
    (2, "variable_tracking",                 False, "generate_variable_tracking"),
    (3, "frequent_words_extraction",         False, "generate_frequent_words_extraction"),
    (4, "positional_multi_value_lookup",     True,  "generate_positional_multi_value_lookup"),
    (5, "positional_char_in_value_lookup",   True,  "generate_positional_char_in_value_lookup"),
]


def _build_filler_input_ids(tokenizer, length: int) -> np.ndarray:
    """Tokenize a long-enough piece of natural text and pad/truncate to length."""
    text = ""
    while True:
        text += FILLER_PARAGRAPH
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= length:
            break
    return np.array(ids[:length], dtype=np.int64)


def _section(title: str):
    bar = "=" * 80
    print(f"\n{bar}\n {title}\n{bar}")


def _show_split(tokenizer, full_ids, q_ids, a_ids, max_chars=4000):
    """Pretty-print question (truncated middle) + answer."""
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    a_text = tokenizer.decode(a_ids, skip_special_tokens=False)
    q_text = tokenizer.decode(q_ids, skip_special_tokens=False)

    print(f"[lengths] full={len(full_ids)}  q={len(q_ids)}  a={len(a_ids)}")
    # Truncate middle of full text for readability
    if len(full_text) > max_chars:
        head = full_text[: max_chars // 2]
        tail = full_text[-max_chars // 2:]
        full_show = f"{head}\n... <{len(full_text) - max_chars} chars elided> ...\n{tail}"
    else:
        full_show = full_text

    print("\n--- FULL (q+a, middle elided) ---")
    print(full_show)
    print("\n--- QUESTION TAIL (last 400 chars) ---")
    print(q_text[-400:])
    print("\n--- ANSWER ---")
    print(repr(a_text))


def _validate_pml(tokenizer, full_ids, a_text):
    """Sanity check: the answer string should appear in the full text."""
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    needle = a_text.strip().rstrip(".").strip()
    if not needle:
        return None
    occurrences = full_text.count(needle)
    return occurrences


def main():
    print(f"[ruler-inspect] tokenizer = {TOKENIZER_PATH}")
    print(f"[ruler-inspect] total_len = {TOTAL_LEN}, enable_ruler_plus = {ENABLE_PLUS}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
    # Make sure eos is set; most ruler funcs use it
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(0)
    print(f"[ruler-inspect] vocab_size={tokenizer.vocab_size}, "
          f"eos_token={tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    random.seed(SEED)
    np.random.seed(SEED)

    inputs = _build_filler_input_ids(tokenizer, TOTAL_LEN)
    print(f"[ruler-inspect] filler input_ids shape = {inputs.shape}")

    synth = RulerSynthesizer(tokenizer, enable_ruler_plus=ENABLE_PLUS)

    selected = None
    if ONLY_TASK is not None:
        selected = int(ONLY_TASK)

    for task_id, name, needs_plus, fn_name in TASK_INFO:
        if selected is not None and task_id != selected:
            continue
        if needs_plus and not ENABLE_PLUS:
            continue

        _section(f"task {task_id}: {name}  (fn={fn_name})")
        fn = getattr(synth, fn_name)
        try:
            new_ids, q, a = fn(inputs.copy())
        except Exception as e:
            print(f"[ERROR] {name}: {e!r}")
            continue

        _show_split(tokenizer, new_ids, q, a)

        # Sanity check for PMVL / PCVL / NIAH / multi-query: answer should be in text
        if task_id in (0, 1, 4, 5):
            a_text = tokenizer.decode(a, skip_special_tokens=False)
            occ = _validate_pml(tokenizer, new_ids, a_text)
            if occ is not None:
                print(f"\n[sanity] answer string occurs in full text: {occ} time(s)")

    _section("done")


if __name__ == "__main__":
    main()
