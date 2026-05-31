"""Tokenize wikitext-103 (raw v1) with dolma2-tokenizer, save as uint32 .data
files in the layout the olmo3 dataset loader expects.

Layout:
    OUT_DIR/
        train_000.data   # numpy uint32 memmap of token IDs
        train_001.data
        ...

The LazyChunkedLoader reads any .data files it finds and treats the union as
the training corpus. Splitting into multiple files just helps with
parallelism + memory.
"""
import os, sys, argparse, gc
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', default='/home/hal-alex/workspace/hsa_train/tokenizer')
    ap.add_argument('--out', default='/home/hal-alex/workspace/hsa_train/wikitext103_tokenized')
    ap.add_argument('--chunk-docs', type=int, default=10000,
                    help='How many docs to tokenize per .data file (controls file count).')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f'[tok] loading tokenizer from {args.tokenizer}')
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f'[tok] vocab_size={tok.vocab_size}  eos={tok.eos_token_id}')

    print('[tok] loading wikitext-103-raw-v1 (train split)')
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
    print(f'[tok] {len(ds)} docs')

    file_idx = 0
    buf: list[int] = []
    total_tokens = 0
    n_docs_in_chunk = 0
    EOS = tok.eos_token_id

    def flush():
        nonlocal file_idx, buf, n_docs_in_chunk
        if not buf:
            return
        path = os.path.join(args.out, f'train_{file_idx:03d}.data')
        arr = np.asarray(buf, dtype=np.uint32)
        arr.tofile(path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f'[tok] flushed {path}  tokens={len(arr):,}  size={size_mb:.1f}MB')
        file_idx += 1
        buf = []
        n_docs_in_chunk = 0

    for i, doc in enumerate(tqdm(ds, desc='tokenize', mininterval=5.0, ncols=100)):
        text = doc['text']
        if not text.strip():
            continue
        ids = tok.encode(text, add_special_tokens=False)
        buf.extend(ids)
        buf.append(EOS)
        total_tokens += len(ids) + 1
        n_docs_in_chunk += 1
        if n_docs_in_chunk >= args.chunk_docs:
            flush()
    flush()

    print(f'\n[tok] done. total tokens={total_tokens:,}  files={file_idx}')
    print(f'[tok] out dir: {args.out}')


if __name__ == '__main__':
    main()
