import models
from data import build_numpy_dataset
import torch.nn as nn
import argparse
import sys
import torch
from transformers import AutoTokenizer
from torch.utils import data
from torch.utils.data import SequentialSampler

from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

import sys
sys.path.insert(0, '/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/InfiniteLongLM')

from transformers import AutoConfig, AutoModelForCausalLM
from models.FlashHSA.configuration_hsa import HSAConfig
import json

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
    
def resolve_hsa_class(hsa_config=None):
    """根据 hsa_config 中的 model_type 动态选择 HSAForCausalLM 实现"""
    model_type = ""
    if hsa_config:
        with open(hsa_config, 'r') as f:
            model_type = json.load(f).get("model_type", "")
    if "olmo" in model_type:
        from models.FlashHSA.modeling_olmo_lhsa import HSAForCausalLM
        print("Using OLMo LHSA implementation")
    else:
        from models.FlashHSA.modeling_qwen_lhsa import HSAForCausalLM
        print("Using Qwen LHSA implementation")
    return HSAForCausalLM


def main(args):

    device = torch.device('cuda:0')

    HSAForCausalLM = resolve_hsa_class(args.hsa_config)
    AutoConfig.register("flash_hsa", HSAConfig)
    HSAForCausalLM.config_class = HSAConfig
    AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM)

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

    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_3',
        'device_map': device,
    }

    if args.checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **model_kwargs)
    else:
        assert args.config_path is not None, "必须提供 --config_path 或 --checkpoint_path"
        config = AutoConfig.from_pretrained(args.config_path)
        model = AutoModelForCausalLM.from_config(config, **model_kwargs).to(device)

    model.eval()
    if args.insert_lmk:
        chunk_size = model.config.chunk_size
        lmk_id = tokenizer.vocab_size
    else:
        chunk_size = None
        lmk_id = None
    use_chunk_prefill = args.segment_size > 0
    segment_size = args.segment_size if use_chunk_prefill else args.max_seq_len

    if use_chunk_prefill and args.last_k_tokens > 0:
        assert segment_size >= args.last_k_tokens

    print(f"\n{'='*60}")
    print(f"Max Seq Len: {args.max_seq_len}, Segment Size: {segment_size}")
    print(f"Insert LMK: {args.insert_lmk}, Adjust LMK Pos: {args.adjust_lmk_pos}")
    print(f"Chunk Prefill: {'Enabled' if use_chunk_prefill else 'Disabled (Full Inference)'}")
    print(f"{'='*60}\n")

    loss_accum = KahanSum()
    steps = 0
    ce_fct = nn.CrossEntropyLoss()

    for inputs in dataloader:
        steps += 1
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        input_ids = inputs['input_ids']
        label_ids = input_ids.clone()
        pos_ids = None

        if args.insert_lmk:
            orig_seq_len = input_ids.shape[1]
            input_ids = insert_special_tokens(input_ids, fill_id=lmk_id, chunk_size=chunk_size)
            label_ids = torch.roll(label_ids, shifts=-1, dims=-1)
            label_ids[:, -1] = -100
            label_ids = insert_special_tokens(label_ids, fill_id=-100, chunk_size=chunk_size)
            label_ids = torch.roll(label_ids, shifts=1, dims=-1)

            if args.adjust_lmk_pos:
                pos_ids = create_position_ids_with_landmarks(None, orig_seq_len, chunk_size=chunk_size, device=device)

        seq_len = input_ids.shape[1]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            if not use_chunk_prefill:
                # ==================== 全量推理模式 ====================
                kwargs = {}
                if args.last_k_tokens > 0:
                    kwargs['logits_to_keep'] = args.last_k_tokens + 1

                result = model(input_ids, position_ids=pos_ids, use_cache=True, **kwargs)

                out_len = result.logits.shape[1]
                if args.last_k_tokens > 0:
                    out_len = min(out_len, args.last_k_tokens + 1)

                if args.insert_lmk:
                    # 取最后 out_len 的 logits 和 labels，过滤掉 lmk 位置
                    answer_logits = result.logits[:, -out_len:-1, :]
                    answer_labels = label_ids[:, -out_len + 1:]
                    valid_mask = (answer_labels != -100).squeeze(0)
                    loss = ce_fct(
                        answer_logits[0, valid_mask, :],
                        answer_labels[0, valid_mask].to(torch.long)
                    )
                else:
                    loss = ce_fct(
                        result.logits[:, -out_len:-1, :].view(-1, result.logits.shape[-1]),
                        label_ids[:, -out_len + 1:].view(-1).to(torch.long)
                    )
                del result
            else:
                # ==================== Chunk Prefill 模式 ====================
                num_segments = (seq_len + segment_size - 1) // segment_size

                # 确定需要保留 logits 的范围
                if args.last_k_tokens > 0:
                    if args.insert_lmk:
                        orig_answer_start = orig_seq_len - args.last_k_tokens
                        answer_start_with_lmk = orig_answer_start + (orig_answer_start // (chunk_size - 1))
                        answer_len_with_lmk = seq_len - answer_start_with_lmk
                    else:
                        answer_len_with_lmk = args.last_k_tokens
                else:
                    answer_len_with_lmk = seq_len

                answer_logits_start = max(0, seq_len - answer_len_with_lmk - 1)
                first_answer_segment = max(0, answer_logits_start // segment_size)

                past_key_values = None
                answer_logits_list = []

                for i in range(num_segments):
                    start_idx = i * segment_size
                    end_idx = min((i + 1) * segment_size, seq_len)

                    seg_input_ids = input_ids[:, start_idx:end_idx]
                    seg_cache_pos = torch.arange(start_idx, end_idx, device=device)
                    seg_pos_ids = pos_ids[:, start_idx:end_idx] if pos_ids is not None else None

                    if i >= first_answer_segment:
                        seg_logits_to_keep = end_idx - start_idx
                    else:
                        seg_logits_to_keep = 1

                    out = model(
                        input_ids=seg_input_ids,
                        position_ids=seg_pos_ids,
                        cache_position=seg_cache_pos,
                        use_cache=True,
                        past_key_values=past_key_values,
                        logits_to_keep=seg_logits_to_keep,
                    )
                    past_key_values = out.past_key_values

                    if i >= first_answer_segment:
                        answer_logits_list.append(out.logits.cpu())

                    del out

                # 拼接 answer 相关的 logits（在 CPU 上拼接以节省显存）
                answer_region_logits = torch.cat(answer_logits_list, dim=1)
                del answer_logits_list
                torch.cuda.empty_cache()

                # 提取真正需要的 logits
                offset_in_region = answer_logits_start - first_answer_segment * segment_size
                answer_logits = answer_region_logits[:, offset_in_region:offset_in_region + answer_len_with_lmk, :]
                del answer_region_logits

                answer_logits = answer_logits.to(device)
                answer_labels = label_ids[:, -answer_len_with_lmk:]

                if args.insert_lmk:
                    if args.last_k_tokens > 0:
                        answer_logits = answer_logits[:, -args.last_k_tokens:, :]
                        answer_labels = answer_labels[:, -args.last_k_tokens:]
                    valid_mask = (answer_labels != -100).squeeze(0)
                    loss = ce_fct(
                        answer_logits[0, valid_mask, :],
                        answer_labels[0, valid_mask].to(torch.long)
                    )
                else:
                    if args.last_k_tokens > 0:
                        answer_logits = answer_logits[:, -args.last_k_tokens:, :]
                        answer_labels = answer_labels[:, -args.last_k_tokens:]
                    loss = ce_fct(
                        answer_logits.reshape(-1, answer_logits.shape[-1]),
                        answer_labels.reshape(-1).to(torch.long)
                    )

        loss_accum.add(loss.item())
        if steps % 100 == 0:
            print(f'step: {steps}, mean_loss: {loss_accum.get() / steps}')

        if args.max_samples > 0 and steps >= args.max_samples:
            break

    # 最终结果
    import math
    final_mean_loss = loss_accum.get() / steps
    ppl = math.exp(final_mean_loss)
    print(f'Test Length: {args.max_seq_len}, Final Mean Loss: {final_mean_loss:.4f}, PPL: {ppl:.4f}')


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('Chunk Prefill PPL Test')
    cmd.add_argument('--config_path', required=False, type=str, default=None)
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--data_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--max_seq_len', default=16384, type=int)
    cmd.add_argument('--chunk_size', default=64, type=int)
    cmd.add_argument('--checkpoint_path', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--insert_lmk', action='store_true', help='在外部对数据插入 LMK token')
    cmd.add_argument('--adjust_lmk_pos', action='store_true', help='调整 LMK 位置的 position ids')
    cmd.add_argument('--last_k_tokens', type=int, default=-1, help='只用最后 k 个 token 计算 loss')
    cmd.add_argument('--segment_size', type=int, default=-1, help='Chunk prefill 的分段大小，<=0 表示全量推理')
    cmd.add_argument('--max_samples', default=-1, type=int, help='max samples to eval')
    cmd.add_argument('--hsa_config', type=str, default=None, help='HSA config.json 路径，用于提取 model_type 决定模型实现(olmo/qwen)')
    args = cmd.parse_args(sys.argv[1:])
    print(args)
    main(args)


"""
# 全量推理测试 PPL
python code_exp/test_chunkprefill_ppl_hf.py \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path "/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-lmk-qproj-halfswa-win512-ruler-5per/checkpoints/global_step_30000/hf_ckpt" \
    --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len $((8*1024)) \
    --max_samples 100 \
    --last_k_tokens 512 \
    --insert_lmk \
    --adjust_lmk_pos

    

# Chunk Prefill 测试 PPL
python code_exp/test_chunkprefill_ppl_hf.py \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path "/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-lmk-qproj-halfswa-win512-ruler-5per/checkpoints/global_step_30000/hf_ckpt" \
    --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len $((8*1024)) \
    --max_samples 100 \
    --last_k_tokens 512 \
    --segment_size 256 \
    --insert_lmk \
    --adjust_lmk_pos
    

    
    # olmo
python code_exp/test_chunkprefill_ppl_hf.py \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path "/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/OLMo-stage1-step999000" \
    --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len $((8*1024)) \
    --max_samples 1000 \
    --last_k_tokens 512 \
    --segment_size 4096

python code_exp/test_chunkprefill_ppl_hf.py \
    --vocab_dir ./configs/olmo3_vocab/ \
    --checkpoint_path "/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/OLMo-stage1-step999000" \
    --data_path /apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized \
    --max_seq_len $((8*1024)) \
    --max_samples 1000 \
    --last_k_tokens 512 \
    --segment_size -1

"""
