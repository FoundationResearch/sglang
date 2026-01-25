import models
from data import build_numpy_dataset
import torch.nn as nn
import argparse
import sys
import torch
from transformers import AutoTokenizer
from torch.utils import data
from torch.utils.data import SequentialSampler
from veomni.models import build_foundation_model
from veomni.checkpoint import ckpt_to_state_dict, build_checkpointer
from utils.misc import get_model_fingerprint


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

def insert_special_tokens(input_ids, fill_id, chunk_size):
    """
    每隔 chunk_size 个 token 插入一个特殊 token
    支持 L 不能被 chunk_size 整除的情况
    
    Args:
        input_ids: (N, L) 输入 token ids
        fill_id: 要插入的特殊 token id
        add_last: 是否在最后一个不完整的 chunk 后也插入特殊 token
    
    Returns:
        chunked_input_ids: 插入特殊 token 后的 ids
    """
    N, L = input_ids.shape
    chunk_size = chunk_size
    
    # 计算完整的 chunk 数量和剩余的 token 数量
    full_chunks = L // (chunk_size - 1)
    remainder = L % (chunk_size - 1)
    
    result_parts = []
    
    # 处理完整的 chunks
    if full_chunks > 0:
        full_part = input_ids[:, :full_chunks * (chunk_size - 1)]  # (N, full_chunks * (chunk_size - 1))
        full_part = full_part.view(N, full_chunks, chunk_size - 1)  # (N, full_chunks, chunk_size - 1)
        
        # 在每个 chunk 后添加特殊 token
        chunk_id_padding = torch.full(
            (N, full_chunks, 1), fill_id, 
            device=input_ids.device, dtype=input_ids.dtype
        )
        full_part_with_special = torch.cat([full_part, chunk_id_padding], dim=2)  # (N, full_chunks, chunk_size+1)
        full_part_with_special = full_part_with_special.view(N, -1)  # (N, full_chunks * (chunk_size+1))
        result_parts.append(full_part_with_special)
    
    # 处理剩余部分（不完整的 chunk）
    if remainder > 0:
        remainder_part = input_ids[:, full_chunks * chunk_size:]  # (N, remainder)
        result_parts.append(remainder_part)
    
    chunked_input_ids = torch.cat(result_parts, dim=1)
    
    return chunked_input_ids

def main(args):

    # create dataloader 
    # tokenizer = build_tokenizer(args.vocab_dir)
    device = torch.device('cuda:0')

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
    Checkpointer = build_checkpointer(dist_backend='fsdp2', ckpt_manager='dcp')
    model = build_foundation_model(
        config_path=args.config_path,
        torch_dtype="bfloat16",
    )

    # print(model)
    state = {"model": model}
    Checkpointer.load(args.checkpoint_path, state)
    # model.to(device)

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
        if args.insert_lmk:
            input_ids = insert_special_tokens(input_ids, fill_id=tokenizer.vocab_size, chunk_size=64)
            label_ids = torch.roll(label_ids, shifts=1, dims=-1)
            label_ids = insert_special_tokens(label_ids, fill_id=-100, chunk_size=64)
            label_ids = torch.roll(label_ids, shifts=-1, dims=-1)

            # label_ids = insert_special_tokens(, fill_id=-100, chunk_size=64)
        # print(f'{input_ids.shape}')
        # print(input_ids[:, 63::64])
        # print(tokenizer.decode(input_ids[0,:20]))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            result = model(input_ids, use_cache=False)

        ce_fct = nn.CrossEntropyLoss()
        out_len = result.logits.shape[1]
        if args.last_k_tokens is not None:
            out_len = min(out_len, args.last_k_tokens)
        loss = ce_fct(result.logits[:, -out_len :-1, :].view(-1, result.logits.shape[-1]), label_ids[:, -out_len + 1:].view(-1).to(torch.long))
        # mean_loss += (loss - mean_loss) / steps
        loss_accum.add(loss.item())
        if steps % 10 == 0:
            print(f'step: {steps}, mean_loss: {loss_accum.get() / steps}')

        if args.max_samples > 0 and steps >= args.max_samples:
            break


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('NCR pretraining setup')
    cmd.add_argument('--config_path', required=False, type=str, default=None)
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--data_path', required=True, type=str, help='path to the training corpus')
    # cmd.add_argument('--output_dir', default='/root/')
    cmd.add_argument('--max_seq_len', default=16384, type=int)
    cmd.add_argument('--insert_lmk', action='store_true')
    cmd.add_argument('--checkpoint_path', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--use_cache', action='store_true')
    cmd.add_argument('--last_k_tokens', type=int, default=-1)
    cmd.add_argument('--inference_segment', type=int, default=-1)
    cmd.add_argument('--max_samples', default=-1, type=int, help='max samples to eval')
    cmd.add_argument('--parallel_mode', default='fsdp1', type=str)
    args = cmd.parse_args(sys.argv[1:])
    main(args)
