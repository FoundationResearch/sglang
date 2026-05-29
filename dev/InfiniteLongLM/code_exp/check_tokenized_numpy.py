import os
import numpy as np
from transformers import AutoTokenizer


DATA_PATH = "/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-500B.numpy"
VOCAB_DIR = "configs/olmo3_vocab"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    if not os.path.exists(VOCAB_DIR):
        raise FileNotFoundError(f"Vocab dir not found: {VOCAB_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(VOCAB_DIR)
    vocab_size = tokenizer.vocab_size

    tokens = np.memmap(DATA_PATH, dtype=np.uint32, mode="r")
    total = tokens.shape[0]

    print(f"Data file: {DATA_PATH}")
    print(f"Total tokens: {total}")
    print(f"Vocab size: {vocab_size}")

    # Sample decode
    sample_len = min(256, total)
    sample_ids = tokens[:sample_len].tolist()
    print("Sample decode:")
    print(tokenizer.decode(sample_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
