"""
对比 HF 和 SGLang 推理结果的一致性。

纯 CPU 脚本，只加载两个 pickle 文件做对比，不依赖 GPU 或特殊环境。
由 run_verify.sh 在两个 worker 执行完后调用。

用法（由 run_verify.sh 调用）：
  python verify_sglang_vs_hf.py <config.json>
"""
import logging
logging.getLogger('tilelang.cache.kernel_cache').setLevel(logging.ERROR)

import json
import math
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# 展示用的 top-K 数量（打印时只展示前 display_k 个，不影响 KL 计算）
DISPLAY_K = 4
# KL 散度阈值：逐 token KL < 此值视为"一致"
KL_THRESHOLD = 0.01


def compute_kl_divergence_from_topk(
    hf_logprobs: torch.Tensor,
    sg_top_idx: List[int],
    sg_top_val: List[float],
) -> float:
    """基于两侧 top-K logprobs 近似计算 KL(HF || SG)。

    原理：取 SGLang 返回的 top-K token 集合，在这些 token 上用 HF 的完整概率分布
    和 SGLang 的概率分布做 KL 散度。当 top-K 足够大（如100）时，这些 token 几乎
    覆盖了全部概率质量，近似非常准确。

    Args:
        hf_logprobs: HF 侧的完整 log_softmax 输出, shape (vocab_size,)
        sg_top_idx: SGLang 返回的 top-K token IDs
        sg_top_val: SGLang 返回的 top-K logprobs

    Returns:
        KL 散度值（非负浮点数），越小表示两侧分布越一致
    """
    if not sg_top_idx or not sg_top_val:
        return float('nan')

    # 取 SGLang top-K token 上的 HF logprobs
    sg_indices = torch.tensor(sg_top_idx, dtype=torch.long)
    hf_lp_on_sg_tokens = hf_logprobs[sg_indices].float().numpy()  # (K,)
    sg_lp = np.array(sg_top_val, dtype=np.float32)  # (K,)

    # 转为概率
    hf_p = np.exp(hf_lp_on_sg_tokens)
    sg_q = np.exp(sg_lp)

    # 归一化（在 top-K 子集上重新归一化）
    hf_p_sum = hf_p.sum()
    sg_q_sum = sg_q.sum()
    if hf_p_sum < 1e-10 or sg_q_sum < 1e-10:
        return float('nan')
    hf_p = hf_p / hf_p_sum
    sg_q = sg_q / sg_q_sum

    # KL(P || Q) = sum(P * log(P / Q))，避免 log(0)
    eps = 1e-10
    kl = np.sum(hf_p * np.log((hf_p + eps) / (sg_q + eps)))
    return max(float(kl), 0.0)  # 数值误差可能导致微小负值


def compare_decode_results(
    hf_results: Dict,
    sg_results: Dict,
    tokenizer,
    vocab_size: int,
    top_k: int,
):
    """对比 decode 阶段的结果。"""
    print("\n" + "=" * 80)
    print("📊 Decode 阶段对比")
    print("=" * 80)

    hf_ids = hf_results["decode_token_ids"]
    sg_ids = sg_results["decode_token_ids"]
    hf_logits_list = hf_results["decode_logits"]
    meta = sg_results["meta_info"]

    # SGLang 返回格式: output_token_logprobs = [(logprob, token_id, text), ...]
    #                   output_top_logprobs = [[(logprob, token_id, text), ...], ...]
    sg_raw_logprobs = meta.get("output_token_logprobs", None)
    sg_raw_top = meta.get("output_top_logprobs", None)

    # 解析 tuple 格式
    sg_output_logprobs_val = []
    sg_output_logprobs_idx = []
    if sg_raw_logprobs:
        for item in sg_raw_logprobs:
            if item is None or (isinstance(item, (list, tuple)) and item[0] is None):
                sg_output_logprobs_val.append(None)
                sg_output_logprobs_idx.append(None)
            else:
                sg_output_logprobs_val.append(float(item[0]))
                sg_output_logprobs_idx.append(int(item[1]))

    sg_output_top_val = []
    sg_output_top_idx = []
    if sg_raw_top:
        for entry in sg_raw_top:
            if entry is None:
                sg_output_top_val.append(None)
                sg_output_top_idx.append(None)
            else:
                vals = [float(t[0]) for t in entry]
                idxs = [int(t[1]) for t in entry]
                sg_output_top_val.append(vals)
                sg_output_top_idx.append(idxs)

    print(f"  [DEBUG] output logprobs: {len(sg_output_logprobs_val)} entries parsed")
    print(f"  [DEBUG] output top logprobs: {len(sg_output_top_val)} entries parsed")
    # 打印前3个样本
    for j in range(min(3, len(sg_output_logprobs_val))):
        print(f"    sample[{j}]: val={sg_output_logprobs_val[j]}, idx={sg_output_logprobs_idx[j]}")
    print()

    max_len = max(len(hf_ids), len(sg_ids))
    num_total = min(len(hf_ids), len(sg_ids))
    num_match = 0
    top2_match_count = 0  # top1不一致但top2集合一致的计数
    logprob_diffs = []
    kl_values = []  # 每个 decode step 的 KL 散度

    for i in range(max_len):
        hf_tok = hf_ids[i] if i < len(hf_ids) else None
        sg_tok = sg_ids[i] if i < len(sg_ids) else None

        match = (hf_tok == sg_tok) if (hf_tok is not None and sg_tok is not None) else False
        if match:
            num_match += 1

        # 判断 top2 是否一致
        top2_match = False
        if not match and i < len(hf_logits_list):
            hf_lps = F.log_softmax(hf_logits_list[i].float(), dim=-1)
            hf_top2_vals, hf_top2_idxs = hf_lps.topk(min(2, len(hf_lps)))
            hf_top2_set = set(idx.item() for idx in hf_top2_idxs)
            if (sg_output_top_val and i < len(sg_output_top_val)
                    and sg_output_top_val[i] is not None
                    and len(sg_output_top_idx[i]) >= 2):
                sg_top2_set = set(sg_output_top_idx[i][:2])
                top2_match = (hf_top2_set == sg_top2_set)
            if top2_match:
                top2_match_count += 1

        hf_text = tokenizer.decode([hf_tok]) if hf_tok is not None else "<N/A>"
        sg_text = tokenizer.decode([sg_tok]) if sg_tok is not None else "<N/A>"
        status = "✅" if match else ("🔶" if top2_match else "❌")

        # HF logits -> logprobs top-k
        hf_top_str = ""
        if i < len(hf_logits_list):
            logits = hf_logits_list[i]
            lps = F.log_softmax(logits.float(), dim=-1)
            top_vals, top_idxs = lps.topk(DISPLAY_K)
            hf_top_str = "  HF_top=[" + ", ".join(
                f"'{tokenizer.decode([idx.item()])}' ({val:.3f})"
                for idx, val in zip(top_idxs, top_vals)
            ) + "]"

        # SGLang logprobs
        lp_diff = float("nan")
        if sg_output_logprobs_val and i < len(sg_output_logprobs_val) and sg_output_logprobs_val[i] is not None:
            sg_lp = sg_output_logprobs_val[i]

            if i < len(hf_logits_list):
                hf_lps = F.log_softmax(hf_logits_list[i].float(), dim=-1)
                if sg_tok is not None and sg_tok < vocab_size:
                    hf_lp_for_sg_tok = hf_lps[sg_tok].item()
                    lp_diff = abs(hf_lp_for_sg_tok - sg_lp)
                    logprob_diffs.append(lp_diff)

        # KL 散度计算
        if (i < len(hf_logits_list)
                and sg_output_top_val and i < len(sg_output_top_val)
                and sg_output_top_val[i] is not None
                and sg_output_top_idx[i] is not None):
            hf_lps_for_kl = F.log_softmax(hf_logits_list[i].float(), dim=-1)
            kl_val = compute_kl_divergence_from_topk(
                hf_lps_for_kl, sg_output_top_idx[i], sg_output_top_val[i]
            )
            if not math.isnan(kl_val):
                kl_values.append(kl_val)

        # SGLang top logprobs
        sg_top_str = ""
        if sg_output_top_val and i < len(sg_output_top_val) and sg_output_top_val[i] is not None:
            sg_tops = list(zip(sg_output_top_idx[i], sg_output_top_val[i]))
            sg_top_str = "  SG_top=[" + ", ".join(
                f"'{tokenizer.decode([int(idx)])}' ({val:.3f})"
                for idx, val in sg_tops[:DISPLAY_K]
            ) + "]"

        kl_display = kl_values[-1] if kl_values else float('nan')
        print(f"  Step {i:3d}: {status}  "
              f"HF={hf_tok}('{hf_text}')  SG={sg_tok}('{sg_text}')  "
              f"lp_diff={lp_diff:.4f}  KL={kl_display:.6f}"
              f"{hf_top_str}{sg_top_str}")

    # 汇总
    print("-" * 80)
    if num_total > 0:
        print(f"  Token 匹配率: {num_match}/{num_total} ({num_match / num_total * 100:.1f}%)")
        print(f"  Decode top2一致(🔶): {top2_match_count}/{num_total - num_match} (在token不一致的step中)")
        print(f"  Decode token+top2 总一致率: {num_match + top2_match_count}/{num_total} ({(num_match + top2_match_count) / num_total * 100:.1f}%)")
    else:
        print("  无可对比 token")

    if logprob_diffs:
        avg_diff = sum(logprob_diffs) / len(logprob_diffs)
        max_diff = max(logprob_diffs)
        print(f"  Logprob 差值: 平均={avg_diff:.6f}, 最大={max_diff:.6f}")

    if kl_values:
        kl_pass_count = sum(1 for kl in kl_values if kl < KL_THRESHOLD)
        avg_kl = sum(kl_values) / len(kl_values)
        max_kl = max(kl_values)
        median_kl = float(np.median(kl_values))
        print(f"  KL 散度 (top-{top_k} 近似): 平均={avg_kl:.6f}, 中位数={median_kl:.6f}, 最大={max_kl:.6f}")
        print(f"  KL < {KL_THRESHOLD} 的 step 数: {kl_pass_count}/{len(kl_values)} ({kl_pass_count / len(kl_values) * 100:.1f}%)")

    if num_match == num_total and num_total > 0:
        print("  🎉 Decode 生成完全一致！")
    elif num_match > num_total * 0.8:
        print("  ⚠️  Decode 大部分一致，少数差异可能是 bf16 精度导致。")
    else:
        print("  ❌ Decode 结果存在较大差异，请排查 SGLang 侧的 HSA 实现。")

    # 完整文本对比
    print(f"\n  --- HF 生成文本 ---")
    print(f"  {hf_results['decode_text']}")
    print(f"\n  --- SGLang 生成文本 ---")
    print(f"  {sg_results['decode_text']}")


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_sglang_vs_hf.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        cfg = json.load(f)

    hf_output_path = cfg["hf_output_path"]
    sglang_output_path = cfg["sglang_output_path"]
    vocab_dir = cfg.get("vocab_dir") or cfg["checkpoint_path"]
    top_k = cfg.get("top_k", 5)

    print("=" * 80)
    print("📊 加载推理结果并对比")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(vocab_dir)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")

    # 加载 HF 结果
    with open(hf_output_path, "rb") as f:
        hf_results = pickle.load(f)
    print(f"  HF 结果已加载 (decode tokens: {len(hf_results['decode_token_ids'])})")

    # 加载 SGLang 结果
    with open(sglang_output_path, "rb") as f:
        sg_results = pickle.load(f)
    print(f"  SGLang 结果已加载 (decode tokens: {len(sg_results['decode_token_ids'])})")

    # ===== DEBUG: 打印 SGLang meta_info 完整 key 结构 =====
    meta = sg_results.get("meta_info", {})
    print(f"\n  [DEBUG] SGLang meta_info keys: {sorted(meta.keys())}")
    for k, v in sorted(meta.items()):
        if isinstance(v, (list, tuple)):
            vinfo = f"list len={len(v)}"
            if len(v) > 0:
                first = v[0]
                if isinstance(first, (list, tuple)):
                    vinfo += f", [0] is list len={len(first)}"
                elif isinstance(first, (int, float)):
                    vinfo += f", [0]={first}"
                else:
                    vinfo += f", [0] type={type(first).__name__}"
        elif isinstance(v, (int, float, str, bool)):
            vinfo = f"{v}"
        else:
            vinfo = f"type={type(v).__name__}"
        print(f"    {k}: {vinfo}")
    print()

    # 只对比 decode 阶段
    compare_decode_results(hf_results, sg_results, tokenizer, vocab_size, top_k)

    # ===== Decode 耗时对比 =====
    hf_time = hf_results.get("decode_time")
    sg_time = sg_results.get("decode_time")
    hf_ntok = hf_results.get("num_generated_tokens", len(hf_results["decode_token_ids"]))
    sg_ntok = sg_results.get("num_generated_tokens", len(sg_results["decode_token_ids"]))

    print(f"\n{'=' * 80}")
    print("⏱️  Decode 耗时对比")
    print("=" * 80)
    if hf_time is not None:
        print(f"  HF  : {hf_time:.3f}s  ({hf_ntok} tokens, {hf_time / max(hf_ntok, 1) * 1000:.1f} ms/token)")
    else:
        print("  HF  : 未记录耗时（请更新 verify_hf_worker.py）")
    if sg_time is not None:
        print(f"  SG  : {sg_time:.3f}s  ({sg_ntok} tokens, {sg_time / max(sg_ntok, 1) * 1000:.1f} ms/token)")
    else:
        print("  SG  : 未记录耗时（请更新 verify_sglang_worker.py）")
    if hf_time is not None and sg_time is not None:
        speedup = hf_time / sg_time if sg_time > 0 else float("inf")
        faster = "SGLang" if sg_time < hf_time else "HF"
        print(f"  加速比: {faster} 快 {abs(speedup - 1) * 100:.1f}%  (HF/SG = {speedup:.2f}x)")

    print(f"\n{'=' * 80}")
    print("✅ 对比完成！")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
