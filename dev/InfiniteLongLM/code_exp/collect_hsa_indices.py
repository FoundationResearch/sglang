"""
HSA Indices 收集脚本
====================
从真实模型中收集每一层 HSA 的 topk indices、scores 和 chunk_weights，
保存为 .pt 文件供后续 bench 测试使用。

同时提供重合率统计函数，分析相邻 token 的 indices 重合度。

用法:
    # 设置环境变量启用收集（最新 head-wise 模型示例）
    HSA_COLLECT_INDICES=1 python code_exp/collect_hsa_indices.py \
        --config_path configs/flash_hsa/config_hsa_8KA2K_HoPE_345M_lmk_bias_priorq_wloralmkq_loradim64.json \
        --checkpoint_path /apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/hsa_8KA2K_HoPE_full_345M_dist-priorq-wloralmkq-loradim64/checkpoints/global_step_<LATEST_STEP> \
        --vocab_dir /apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/hsa_8KA2K_HoPE_full_345M_dist-priorq-wloralmkq-loradim64/model_assets \
        --data_path /apdcephfs_tj5/share_300719894/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
        --seq_lens 8192,16384,32768,65536,131072 \
        --num_samples 10 \
        --output_dir /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup_headwise

    # 仅统计已保存的 .pt 文件的重合率
    python code_exp/collect_hsa_indices.py --analyze /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup_headwise/indices_8192.pt
"""

import os
import sys
import argparse
import json
import math
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import SequentialSampler


def analyze_overlap(pt_path: str):
    """
    统计 .pt 文件中每一层、每个 sample 的 indices 重合率。

    重合率定义（M 路并集）：
        对于连续 M 个 token，每个 token 选了 S 个 indices，
        overlap_ratio = (M * S - |union|) / ((M - 1) * S)
        其中 |union| 是 M 个 token 的 indices 取并集后的大小。

    对每一层输出：
        - 每个 block_M 下的平均重合率
        - 按 token 位置分段的重合率分布
    """
    print(f"\n{'='*100}")
    print(f"分析文件: {pt_path}")
    print(f"{'='*100}")

    saved = torch.load(pt_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    samples = saved["samples"]
    num_samples = len(samples)

    print(f"配置: seq_len={config['seq_len']}, chunk_size={config['chunk_size']}, "
          f"hsa_topk={config['hsa_topk']}, num_attention_heads={config['num_attention_heads']}, "
          f"num_key_value_heads={config['num_key_value_heads']}")
    print(f"样本数: {num_samples}")

    # 收集所有层的 layer_idx
    all_layer_idxs = set()
    for sample in samples:
        all_layer_idxs.update(sample["layers"].keys())
    all_layer_idxs = sorted(all_layer_idxs)

    S = config["hsa_topk"]
    block_M_list = [1, 2, 4, 8, 16, 32, 64]

    # 先确定 H（kv head 数）
    first_sample = samples[0]
    first_layer = list(first_sample["layers"].values())[0]
    H = first_sample["layers"][all_layer_idxs[0]]["indices"].shape[2]
    print(f"KV head 数: {H}")

    for layer_idx in all_layer_idxs:
        print(f"\n{'─'*80}")
        print(f"Layer {layer_idx}")
        print(f"{'─'*80}")

        for block_M in block_M_list:
            # 按 kv head 分别收集重合率
            overlap_per_head = {h: [] for h in range(H)}

            for sample in samples:
                if layer_idx not in sample["layers"]:
                    continue
                indices = sample["layers"][layer_idx]["indices"]  # [B, L, H, S]
                B, L, cur_H, topk = indices.shape

                if block_M > L:
                    continue

                for b in range(B):
                    for h in range(cur_H):
                        # 按 block_M 大小的窗口滑动
                        t = 0
                        while t + block_M <= L:
                            window_indices = indices[b, t:t+block_M, h, :]  # [block_M, S]
                            # 过滤掉 -1（无效位）
                            valid_sets = []
                            for m in range(block_M):
                                valid = window_indices[m][window_indices[m] >= 0]
                                valid_sets.append(set(valid.tolist()))

                            if not valid_sets or all(len(s) == 0 for s in valid_sets):
                                t += block_M
                                continue

                            # 计算并集大小
                            union_set = set()
                            total_count = 0
                            for s in valid_sets:
                                union_set |= s
                                total_count += len(s)

                            union_size = len(union_set)
                            if block_M > 1 and total_count > 0:
                                # overlap = (M*S - |union|) / ((M-1)*S)
                                avg_s = total_count / block_M
                                if avg_s > 0 and block_M > 1:
                                    overlap = (total_count - union_size) / ((block_M - 1) * avg_s)
                                    overlap = max(0.0, min(1.0, overlap))
                                    overlap_per_head[h].append(overlap)
                            t += block_M

            # 输出每个 kv head 的统计结果
            head_avgs = []
            has_data = False
            for h in range(H):
                ratios = overlap_per_head[h]
                if ratios:
                    has_data = True
                    avg_h = sum(ratios) / len(ratios)
                    sorted_h = sorted(ratios)
                    median_h = sorted_h[len(sorted_h) // 2]
                    head_avgs.append(avg_h)
                    print(f"  block_M={block_M:>3}, kv_head={h}: "
                          f"avg={avg_h:.4f}, "
                          f"median={median_h:.4f}, "
                          f"min={min(ratios):.4f}, "
                          f"max={max(ratios):.4f}, "
                          f"num_windows={len(ratios)}")
                else:
                    print(f"  block_M={block_M:>3}, kv_head={h}: 无有效数据")

            # 输出所有 kv head 的平均值
            if has_data and head_avgs:
                overall_avg = sum(head_avgs) / len(head_avgs)
                print(f"  block_M={block_M:>3}, ** avg across heads **: {overall_avg:.4f}")

    print(f"\n{'='*100}")
    print("分析完成")
    print(f"{'='*100}")


def collect_indices(args):
    """
    加载模型，对每个序列长度采样 num_samples 条文本，
    跑前向收集每层 HSA 的 indices/scores/chunk_weights。
    """
    # 确保环境变量已设置
    assert os.environ.get("HSA_COLLECT_INDICES", "0") == "1", \
        "请设置环境变量 HSA_COLLECT_INDICES=1 来启用收集功能"

    import models
    from data import build_numpy_dataset
    from transformers import AutoTokenizer
    from veomni.models import build_foundation_model
    from veomni.checkpoint import build_checkpointer
    from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

    # 延迟导入，确保环境变量已生效
    from models.FlashHSA.lhsa_layer import _hsa_collected_data

    device = torch.device('cuda:0')

    # 解析序列长度列表
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    num_samples = args.num_samples

    print(f"{'='*100}")
    print(f"HSA Indices 收集")
    print(f"{'='*100}")
    print(f"配置文件: {args.config_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"数据路径: {args.data_path}")
    print(f"序列长度: {seq_lens}")
    print(f"每个长度采样数: {num_samples}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*100}")

    # 读取模型配置
    with open(args.config_path, 'r') as f:
        model_config = json.load(f)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("\n加载模型...")
    Checkpointer = build_checkpointer(dist_backend='fsdp2', ckpt_manager='dcp')
    model = build_foundation_model(
        config_path=args.config_path,
        torch_dtype="bfloat16",
    )
    state = {"model": model}
    Checkpointer.load(args.checkpoint_path, state)
    model.eval()
    print("模型加载完成")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    chunk_size = model_config.get("chunk_size", 64)

    for seq_len in seq_lens:
        print(f"\n{'#'*100}")
        print(f"### 序列长度: {seq_len} ({seq_len//1024}k)")
        print(f"{'#'*100}")

        # 构建数据集
        dataset = build_numpy_dataset(args.data_path, seq_len, namespace='test')

        def vanilla_collate_fn(examples):
            return {
                'input_ids': torch.tensor(examples),
            }

        dataloader = data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=vanilla_collate_fn,
            sampler=SequentialSampler(dataset),
            num_workers=1,
        )

        # 收集结果
        all_samples = []
        sample_count = 0

        for inputs in dataloader:
            if sample_count >= num_samples:
                break

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            input_ids = inputs['input_ids']

            # 插入 landmark token
            input_ids = insert_special_tokens(
                input_ids, fill_id=tokenizer.vocab_size, chunk_size=chunk_size
            )
            pos_ids = create_position_ids_with_landmarks(
                None, seq_len, chunk_size=chunk_size, device=device
            )

            # 清空全局收集字典
            _hsa_collected_data.clear()

            # 前向推理
            with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
                _ = model(input_ids, position_ids=pos_ids, use_cache=False)

            # 从全局字典中取出收集到的数据
            if not _hsa_collected_data:
                print(f"  ⚠ 样本 {sample_count+1}: 未收集到任何 HSA 层数据，跳过")
                continue

            sample_data = {
                "layers": {}
            }
            for layer_idx, layer_data in sorted(_hsa_collected_data.items()):
                sample_data["layers"][layer_idx] = {
                    "indices": layer_data["indices"].clone(),
                    "scores": layer_data["scores"].clone(),
                    "chunk_weights": layer_data["chunk_weights"].clone(),
                }

            all_samples.append(sample_data)
            sample_count += 1

            layer_idxs = sorted(_hsa_collected_data.keys())
            first_layer = layer_idxs[0]
            idx_shape = _hsa_collected_data[first_layer]["indices"].shape
            print(f"  样本 {sample_count}/{num_samples}: "
                  f"收集了 {len(layer_idxs)} 层 HSA 数据, "
                  f"层索引={layer_idxs}, "
                  f"indices shape={list(idx_shape)}")

        # 保存
        save_data = {
            "config": {
                "seq_len": seq_len,
                "chunk_size": chunk_size,
                "hsa_topk": model_config.get("hsa_topk", 16),
                "num_attention_heads": model_config.get("num_attention_heads", 16),
                "num_key_value_heads": model_config.get("num_key_value_heads", 4),
                "hsa_heads": model_config.get("hsa_heads", 16),
                "hsa_qk_ratio": model_config.get("hsa_qk_ratio", 4),
                "hsa_sliding_window": model_config.get("hsa_sliding_window", 512),
                "head_dim": model_config.get("head_dim", 64),
            },
            "samples": all_samples,
        }

        output_path = os.path.join(args.output_dir, f"indices_{seq_len}.pt")
        torch.save(save_data, output_path)
        print(f"\n  ✓ 已保存 {sample_count} 条样本到 {output_path}")

        # 可选：立即对当前长度进行重合率分析
        if args.with_analyze:
            print(f"\n  --- 重合率分析 (seq_len={seq_len}) ---")
            analyze_overlap(output_path)

        # 释放显存
        del all_samples
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("所有序列长度收集完成！")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*100}")


def parse_args():
    parser = argparse.ArgumentParser(description="HSA Indices 收集与重合率分析")

    # 收集模式参数
    parser.add_argument("--config_path", type=str, default=None,
                        help="模型配置文件路径")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="模型 checkpoint 路径")
    parser.add_argument("--vocab_dir", type=str, default=None,
                        help="tokenizer 路径")
    parser.add_argument("--data_path", type=str, default=None,
                        help="数据集路径")
    parser.add_argument("--seq_lens", type=str, default="8192,16384,32768,65536,131072",
                        help="序列长度列表，逗号分隔（默认: 8192,16384,32768,65536,131072）")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="每个序列长度采样的文本条数（默认: 10）")
    parser.add_argument("--output_dir", type=str, default="code_exp/real_indices",
                        help="输出目录（默认: code_exp/real_indices）")

    # 分析模式参数
    parser.add_argument("--analyze", type=str, default=None,
                        help="仅分析已保存的 .pt 文件的重合率（传入文件路径）")
    parser.add_argument("--with_analyze", action="store_true", default=False,
                        help="收集完成后自动统计重合率（默认关闭）")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.analyze:
        # 仅分析模式
        analyze_overlap(args.analyze)
    else:
        # 收集模式
        collect_indices(args)


"""

HSA_COLLECT_INDICES=1 python code_exp/collect_hsa_indices.py \
    --config_path configs/flash_hsa/config_hsa_8KA2K_HoPE_345M_lmk_bias_priorq_wloralmkq_loradim64.json \
    --checkpoint_path /apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/hsa_8KA2K_HoPE_full_345M_dist-priorq-wloralmkq-loradim64/checkpoints/global_step_<LATEST_STEP> \
    --vocab_dir /apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/hsa_8KA2K_HoPE_full_345M_dist-priorq-wloralmkq-loradim64/model_assets \
    --data_path /apdcephfs_tj5/share_300719894/shared/data/dolma3_mix-6T-1025-partial-tokenized/ \
    --seq_lens 8192,16384,32768,65536,131072 \
    --num_samples 10 \
    --output_dir /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup_headwise

"""