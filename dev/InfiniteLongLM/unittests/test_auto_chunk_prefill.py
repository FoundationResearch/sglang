"""
测试自动 Chunk Prefill 的正确性和最大处理长度。

验证逻辑：
1. 正确性测试：对比 chunk prefill 和全量 forward 的输出 logits 是否一致
2. 最大长度测试：测试 chunk prefill 能否成功处理指定长度的序列

用法：
    python unittests/test_auto_chunk_prefill.py --test_seq_len 16384
"""

import sys
import os
import argparse
import time
import torch

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoConfig, AutoModelForCausalLM

# 注册自定义模型
from models.FlashHSA.configuration_hsa import HSAConfig
from models.FlashHSA.modeling_olmo_lhsa import HSAForCausalLM

HSAConfig.model_type = "olmo_lhsa"
AutoConfig.register("olmo_lhsa", HSAConfig)
HSAForCausalLM.config_class = HSAConfig
AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM)

from models.FlashHSA.chunk_prefill import CHUNK_PREFILL_SIZE, DEFAULT_CHUNK_PREFILL_THRESHOLD

DEFAULT_CKPT_PATH = (
    "/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/lhsa-olmo3-interleave"
)

# 正确性对比时只对比最后 N 个 token 的 logits
COMPARE_LAST_N = 1024


def load_model(checkpoint_path, device):
    """加载模型到指定设备，启用 auto_insert_lmk"""
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_3",
        "device_map": device,
        "auto_insert_lmk": True,
    }
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    model.eval()
    return model


def enable_chunk_prefill(model, threshold=DEFAULT_CHUNK_PREFILL_THRESHOLD):
    """启用自动 chunk prefill，序列长度超过 threshold 时自动开启"""
    model.model._chunk_prefill_threshold = threshold
    print(f"[INFO] 已启用自动 chunk prefill，threshold={threshold}, chunk_size={CHUNK_PREFILL_SIZE}")


def disable_chunk_prefill(model):
    """禁用自动 chunk prefill"""
    model.model._chunk_prefill_threshold = 0
    print(f"[INFO] 已禁用自动 chunk prefill")


def generate_random_input(seq_len, vocab_size, device):
    """生成随机输入 token ids"""
    return torch.randint(0, vocab_size, (1, seq_len), device=device)


def test_correctness(model, args, device):
    """
    正确性测试：对比 chunk prefill 和全量 forward 的输出 logits。
    通过 model.forward 的 logits_to_keep 参数只计算最后 COMPARE_LAST_N 个 token 的 logits，
    大幅减少 logits 显存占用。
    """
    print("\n" + "=" * 60)
    print("正确性测试：Chunk Prefill vs 全量 Forward")
    print(f"序列长度: {args.test_seq_len}, Chunk 大小: {CHUNK_PREFILL_SIZE}")
    print(f"对比范围: 最后 {COMPARE_LAST_N} 个 token (logits_to_keep={COMPARE_LAST_N})")
    print("=" * 60)

    vocab_size = model.config.vocab_size
    input_ids = generate_random_input(args.test_seq_len, vocab_size, device)

    # 1. 全量 forward（不启用 chunk prefill）
    print("\n[1/2] 运行全量 forward ...")
    disable_chunk_prefill(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        outputs_full = model(input_ids, logits_to_keep=COMPARE_LAST_N)
    t1 = time.time()
    mem_full = torch.cuda.max_memory_allocated() / 1024**3
    logits_full = outputs_full.logits
    print(f"  全量 forward 完成: {t1 - t0:.2f}s, 峰值显存: {mem_full:.2f} GB")
    print(f"  输出 logits shape: {logits_full.shape}")

    # 释放 outputs 中的其他引用
    del outputs_full
    torch.cuda.empty_cache()

    # 2. Chunk prefill forward
    print(f"\n[2/2] 运行 chunk prefill forward (chunk_size={CHUNK_PREFILL_SIZE}) ...")
    # 正确性测试时将阈值设为 1，这样任何长度 > 1 都会走 chunk prefill
    enable_chunk_prefill(model, threshold=1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        outputs_chunk = model(input_ids, logits_to_keep=COMPARE_LAST_N)
    t1 = time.time()
    mem_chunk = torch.cuda.max_memory_allocated() / 1024**3
    logits_chunk = outputs_chunk.logits
    print(f"  Chunk prefill 完成: {t1 - t0:.2f}s, 峰值显存: {mem_chunk:.2f} GB")
    print(f"  输出 logits shape: {logits_chunk.shape}")

    del outputs_chunk, input_ids
    torch.cuda.empty_cache()

    # 3. 对比结果
    print(f"\n--- 对比结果（最后 {COMPARE_LAST_N} 个 token）---")
    assert logits_full.shape == logits_chunk.shape, (
        f"Shape 不一致: full={logits_full.shape}, chunk={logits_chunk.shape}"
    )

    # 辅助校验函数
    def get_abs_err(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        return (x[mask] - y[mask]).abs().max().item()

    def get_err_ratio(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        err = (x[mask] - y[mask]).square().mean().sqrt().item()
        base = (x[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    def assert_close(prefix, ref, tri, ratio=0.005):
        abs_err = get_abs_err(ref, tri)
        rel_ratio = get_err_ratio(ref, tri)
        msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio, msg

    # 验证 logits 一致性
    assert_close("Logits", logits_full.float(), logits_chunk.float())

    # 对比 argmax 是否一致
    argmax_full = logits_full.argmax(dim=-1)
    argmax_chunk = logits_chunk.argmax(dim=-1)
    match_rate = (argmax_full == argmax_chunk).float().mean().item()
    print(f"  Argmax 匹配率: {match_rate * 100:.2f}%")

    print(f"\n✅ 正确性测试通过！")
    print(f"\n显存对比: 全量={mem_full:.2f} GB, Chunk Prefill={mem_chunk:.2f} GB (节省 {mem_full - mem_chunk:.2f} GB)")

    # 清理
    del logits_full, logits_chunk
    torch.cuda.empty_cache()


def test_max_length(model, args, device):
    """
    最大长度测试：使用 chunk prefill 测试指定长度的序列能否成功处理。
    """
    seq_len = args.test_seq_len
    print("\n" + "=" * 60)
    print("最大长度测试（仅 Chunk Prefill）")
    print(f"序列长度: {seq_len} ({seq_len // 1024}K), Chunk 大小: {CHUNK_PREFILL_SIZE}")
    print("=" * 60)

    vocab_size = model.config.vocab_size
    enable_chunk_prefill(model, threshold=1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    input_ids = generate_random_input(seq_len, vocab_size, device)

    try:
        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_ids, logits_to_keep=1)
        t1 = time.time()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n✅ 成功! 序列长度: {seq_len} ({seq_len // 1024}K)")
        print(f"   耗时: {t1 - t0:.2f}s, 峰值显存: {mem:.2f} GB, logits shape: {outputs.logits.shape}")
        del outputs
    except torch.cuda.OutOfMemoryError as e:
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n❌ OOM! 序列长度: {seq_len} ({seq_len // 1024}K), 峰值显存: {mem:.2f} GB")
        print(f"   错误信息: {e}")
        import traceback
        print("\n--- 完整 OOM Traceback ---")
        traceback.print_exc()
        print("--- Traceback 结束 ---")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        del input_ids
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser("自动 Chunk Prefill 测试")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CKPT_PATH,
                        help="模型 checkpoint 路径")
    parser.add_argument("--test_seq_len", type=int, default=4*1024*1024,
                        help="测试的序列长度")

    args = parser.parse_args()

    device = torch.device("cuda:0")
    print(f"加载模型: {args.checkpoint_path}")
    model = load_model(args.checkpoint_path, device)
    print(f"模型加载完成, config.chunk_size={model.config.chunk_size}, "
          f"insert_landmarks={getattr(model.config, 'insert_landmarks', True)}, "
          f"auto_insert_lmk={model.auto_insert_lmk}")

    # test_correctness(model, args, device)
    test_max_length(model, args, device)


if __name__ == "__main__":
    main()

# python unittests/test_auto_chunk_prefill.py
