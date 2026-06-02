import argparse
import sys
import os
import json

EVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFINITE_ROOT = os.path.abspath(os.path.join(EVAL_ROOT, "..", "InfiniteLongLM"))

if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from data import build_numpy_dataset

if INFINITE_ROOT not in sys.path:
    sys.path.insert(0, INFINITE_ROOT)

import models
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils import data
from torch.utils.data import SequentialSampler
from veomni.models import build_foundation_model
from veomni.checkpoint import ckpt_to_state_dict, build_checkpointer
from utils.misc import get_model_fingerprint
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks
from veomni.checkpoint import build_checkpointer
from transformers import AutoModelForCausalLM


class KahanSum:
    """Kahan 求和算法，减少浮点累加误差"""
    def __init__(self):
        self.sum = 0.0
        self.c = 0.0  # 误差补偿
        
    def add(self, value):
        y = value - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
        
    def get(self):
        return self.sum


def append_summary_log(summary_log, summary_text):
    log_dir = os.path.dirname(os.path.abspath(summary_log))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(summary_log, "a", encoding="utf-8") as fout:
        fout.write(summary_text)
        fout.write("\n\n")


def get_last_k_tokens_with_landmarks(raw_seq_len, raw_last_k_tokens, chunk_size):
    if raw_last_k_tokens <= 0:
        return raw_last_k_tokens

    raw_last_k_tokens = min(raw_last_k_tokens, raw_seq_len)
    if raw_last_k_tokens <= 0:
        return raw_last_k_tokens

    raw_answer_start = raw_seq_len - raw_last_k_tokens
    total_landmarks = raw_seq_len // (chunk_size - 1)
    landmarks_before_answer = raw_answer_start // (chunk_size - 1)
    return raw_last_k_tokens + (total_landmarks - landmarks_before_answer)


def is_dcp_checkpoint(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))


def load_state_dict_from_path(path):
    if os.path.isdir(path):
        safetensors_path = os.path.join(path, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            return load_file(safetensors_path)

        safetensors_index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(safetensors_index):
            from safetensors.torch import load_file
            with open(safetensors_index, "r", encoding="utf-8") as f:
                index = json.load(f)
            state_dict = {}
            for shard in sorted(set(index["weight_map"].values())):
                shard_path = os.path.join(path, shard)
                state_dict.update(load_file(shard_path))
            return state_dict

        bin_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")

        bin_index = os.path.join(path, "pytorch_model.bin.index.json")
        if os.path.exists(bin_index):
            with open(bin_index, "r", encoding="utf-8") as f:
                index = json.load(f)
            state_dict = {}
            for shard in sorted(set(index["weight_map"].values())):
                shard_path = os.path.join(path, shard)
                state_dict.update(torch.load(shard_path, map_location="cpu"))
            return state_dict

        raise FileNotFoundError(f"No model weight files found in directory: {path}")

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)

    return torch.load(path, map_location="cpu")


def build_eval_model(args, device):
    if args.use_hf_model:
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required when --use_hf_model is set")
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.to(device)
        return model

    if not args.config_path:
        raise ValueError("--config_path is required unless --use_hf_model is set")

    model = build_foundation_model(
        config_path=args.config_path,
        torch_dtype="bfloat16",
    )

    if args.checkpoint_path:
        if is_dcp_checkpoint(args.checkpoint_path):
            Checkpointer = build_checkpointer(dist_backend='fsdp2', ckpt_manager='dcp')
            Checkpointer.load(args.checkpoint_path, {"model": model})
        else:
            state_dict = load_state_dict_from_path(args.checkpoint_path)
            incompatible = model.load_state_dict(state_dict, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Checkpoint mismatch. missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )

    model.to(device)
    return model

def main(args):

    # create dataloader 
    # tokenizer = build_tokenizer(args.vocab_dir)
    device = torch.device('cuda:0')

    # 打印max_seq_len
    print(f'Max Sequence Length for Evaluation: {args.max_seq_len}')

    dataset = build_numpy_dataset(args.data_path, args.max_seq_len, namespace='test')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    def vanilla_collate_fn(examples):
        return {
            'input_ids': torch.tensor(examples),
            'labels': torch.tensor(examples)
        }

    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=vanilla_collate_fn,
        sampler=SequentialSampler(dataset),
        num_workers=1
    )

    # model, _ = create_model(
    #     config_path=args.config_path,
    # )
    model = build_eval_model(args, device)

    # fingerprint = get_model_fingerprint(model)
    # print(f'Model Fingerprint (MD5): {fingerprint}')

    model.eval()

    loss_accum = KahanSum()
    steps = 0
    for inputs in dataloader:
        steps += 1
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        input_ids = inputs['input_ids']
        label_ids = input_ids
        pos_ids = None
        orig_seq_len = input_ids.shape[1]
        eval_last_k_tokens = args.last_k_tokens
        if args.insert_lmk:
            input_ids = insert_special_tokens(input_ids, fill_id=tokenizer.vocab_size, chunk_size=args.chunk_size)
            label_ids = torch.roll(label_ids, shifts=-1, dims=-1)
            label_ids[:, -1] = -100
            label_ids = insert_special_tokens(label_ids, fill_id=-100, chunk_size=args.chunk_size)
            label_ids = torch.roll(label_ids, shifts=1, dims=-1)
            if args.last_k_tokens > 0:
                eval_last_k_tokens = get_last_k_tokens_with_landmarks(
                    orig_seq_len,
                    args.last_k_tokens,
                    args.chunk_size,
                )

            if args.adjust_lmk_pos:
                pos_ids = create_position_ids_with_landmarks(None, orig_seq_len, chunk_size=args.chunk_size, device=device)
        # label_ids = insert_special_tokens(, fill_id=-100, chunk_size=64)
        # print(f'{input_ids.shape}')
        # print(input_ids[:, 63::64])
        # print(tokenizer.decode(input_ids[0,:20]))


        # with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        #     result = model(input_ids, position_ids=pos_ids, use_cache=True)

        # ce_fct = nn.CrossEntropyLoss()
        # out_len = result.logits.shape[1]
        # if args.last_k_tokens is not None:
        #     out_len = min(out_len, args.last_k_tokens)
        # loss = ce_fct(result.logits[:, -out_len :-1, :].view(-1, result.logits.shape[-1]), label_ids[:, -out_len + 1:].view(-1).to(torch.long))

        # automatic logits_to_keep
        kwargs = {}
        if eval_last_k_tokens > 0 and not args.use_hf_model:
            kwargs['logits_to_keep'] = eval_last_k_tokens + 1

        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            result = model(input_ids, position_ids=pos_ids, use_cache=False, **kwargs)

        ce_fct = nn.CrossEntropyLoss()
        out_len = result.logits.shape[1]
        
        if eval_last_k_tokens > 0:
            out_len = min(out_len, eval_last_k_tokens + 1)
            
        loss = ce_fct(result.logits[:, -out_len :-1, :].view(-1, result.logits.shape[-1]), label_ids[:, -out_len + 1:].view(-1).to(torch.long))
        

        # mean_loss += (loss - mean_loss) / steps
        loss_accum.add(loss.item())
        if steps % 100 == 0:
            print(f'step: {steps}, mean_loss: {loss_accum.get() / steps}')

        if args.max_samples > 0 and steps >= args.max_samples:
            break
    # final ppl
    import math
    final_mean_loss = loss_accum.get() / steps
    ppl = math.exp(final_mean_loss)
    summary = (
        f'Test Length: {args.max_seq_len}, Final Mean Loss: {final_mean_loss:.4f}, '
        f'PPL: {ppl:.4f}\nModel: {args.checkpoint_path}'
    )
    print(summary)
    if args.summary_log:
        append_summary_log(args.summary_log, summary)


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('NCR pretraining setup')
    cmd.add_argument('--config_path', required=False, type=str, default=None)
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--data_path', required=True, type=str, help='path to the training corpus')
    # cmd.add_argument('--output_dir', default='/root/')
    cmd.add_argument('--max_seq_len', default=16384, type=int)
    cmd.add_argument('--chunk_size', default=64, type=int)
    cmd.add_argument('--insert_lmk', action='store_true')
    cmd.add_argument('--checkpoint_path', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--use_cache', action='store_true')
    cmd.add_argument('--last_k_tokens', type=int, default=-1)
    cmd.add_argument('--inference_segment', type=int, default=-1)
    cmd.add_argument('--max_samples', default=-1, type=int, help='max samples to eval')
    cmd.add_argument('--parallel_mode', default='fsdp1', type=str)
    cmd.add_argument('--adjust_lmk_pos', action='store_true')
    cmd.add_argument('--use_hf_model', action='store_true')
    cmd.add_argument('--summary_log', default=None, type=str, help='append final summary to this log file')
    args = cmd.parse_args(sys.argv[1:])
    print(args)
    main(args)
