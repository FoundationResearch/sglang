"""
统计 HSA 模型 config 相比于 baseline (rope_full_theta10000_345M) 的参数量增量百分比。

用法 (需要在 InfiniteLongLM 项目根目录下运行):
    python code_exp/count_param_overhead.py
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models  # noqa: F401  注册模型
from veomni.models import build_foundation_model
import torch


def count_parameters(model):
    """统计模型中所有可训练参数的数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """统计模型中所有参数的数量（含不可训练）。"""
    return sum(p.numel() for p in model.parameters())


def build_and_count(config_path):
    """加载 config 并返回可训练参数量和总参数量。"""
    model = build_foundation_model(config_path=config_path)
    trainable = count_parameters(model)
    total = count_all_parameters(model)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return trainable, total


def main():
    # ========== Baseline ==========
    baseline_config = "configs/swan_gpt_tiny/config_rope_full_theta10000_345M.json"

    # ========== HSA 模型 configs ==========
    hsa_configs = [
        # pretrain_hsa_8KA2K_partial-RoPE_full_345M_tjdata_lmk_bias_priorq.sh
        "configs/flash_hsa/config_hsa_8KA2K_HoPE_345M_lmk_bias_priorq.json",
        # pretrain_hsa_8KA2K_partial-RoPE_full_345M_tjdata_lmk_bias_priorq_wlmkq.sh
        "configs/flash_hsa/config_hsa_8KA2K_HoPE_345M_lmk_bias_priorq_wlmkq.json",
        # pretrain_hsa_8KA2K_partial-RoPE_full_345M_tjdata_lmk_bias_priorq_wloralmkq.sh
        "configs/flash_hsa/config_hsa_8KA2K_HoPE_345M_lmk_bias_priorq_wloralmkq.json",
        # pretrain_hsa_8KA2K_partial-RoPE_full_345M_tjdata_lmk_bias_priorq_wloralmkq_loradim64.sh
        "configs/flash_hsa/config_hsa_8KA2K_HoPE_345M_lmk_bias_priorq_wloralmkq_loradim64.json",
    ]

    # 给每个 config 一个简短的别名，方便阅读
    hsa_names = [
        "priorq (no lmk_q)",
        "priorq + wlmkq",
        "priorq + wloralmkq (lora_dim=128)",
        "priorq + wloralmkq (lora_dim=64)",
    ]

    print("=" * 80)
    print("  HSA 模型参数量统计 —— 相比 Baseline 的增量百分比")
    print("=" * 80)

    # ---------- 统计 Baseline ----------
    print(f"\n[Baseline] {baseline_config}")
    baseline_trainable, baseline_total = build_and_count(baseline_config)
    print(f"  可训练参数: {baseline_trainable:>15,}")
    print(f"  总参数:     {baseline_total:>15,}")

    # ---------- 统计 HSA 模型 ----------
    results = []
    for name, config_path in zip(hsa_names, hsa_configs):
        print(f"\n[HSA] {name}")
        print(f"  config: {config_path}")
        trainable, total = build_and_count(config_path)

        trainable_overhead = (trainable - baseline_trainable) / baseline_trainable * 100
        total_overhead = (total - baseline_total) / baseline_total * 100
        trainable_diff = trainable - baseline_trainable

        print(f"  可训练参数: {trainable:>15,}  (相比baseline +{trainable_diff:,} = +{trainable_overhead:.4f}%)")
        print(f"  总参数:     {total:>15,}  (相比baseline +{total - baseline_total:,} = +{total_overhead:.4f}%)")

        results.append({
            "name": name,
            "config": config_path,
            "trainable": trainable,
            "total": total,
            "trainable_diff": trainable_diff,
            "trainable_overhead_pct": trainable_overhead,
            "total_overhead_pct": total_overhead,
        })

    # ---------- 汇总表格 ----------
    print("\n")
    print("=" * 80)
    print("  汇总表格")
    print("=" * 80)
    header = f"{'模型':<40s} {'可训练参数':>14s} {'增量':>12s} {'增量%':>10s}"
    print(header)
    print("-" * 80)
    print(f"{'[Baseline] rope_full_345M':<40s} {baseline_trainable:>14,} {'---':>12s} {'---':>10s}")
    for r in results:
        print(f"{r['name']:<40s} {r['trainable']:>14,} {r['trainable_diff']:>+12,} {r['trainable_overhead_pct']:>+9.4f}%")
    print("-" * 80)
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
