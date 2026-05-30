import os
import torch
import argparse
import time
import numpy as np

# ── 注册自定义模型到 transformers ──
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config
from models.FlashHSA.configuration_hsa import HSAConfig
from models.FlashHSA.modeling_qwen_lhsa_forbench import HSAForCausalLM

AutoConfig.register("olmo_lhsa", HSAConfig)
HSAForCausalLM.config_class = HSAConfig
AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM)

# 注册 SWANGPT：将 qwen3 model_type 对应的 AutoModel 覆盖为 SWANGPTForCausalLM
from models.SWANGPT.modeling_swan_gpt import SWANGPTForCausalLM
AutoModelForCausalLM.register(Qwen3Config, SWANGPTForCausalLM, exist_ok=True)

from data import build_numpy_dataset
from torch.utils.data import DataLoader, SequentialSampler

# ========================= 配置 =========================
# Full Attention 模型 (100k steps hf_ckpt)
FULL_ATTN_HF_PATH = "/apdcephfs_sh8/share_300719895/user/qqzxywei/wxy/checkpoints/rope_full_theta10000_345M_dist/checkpoints/global_step_30000/hf_ckpt"

# HSA 模型 (100k steps hf_ckpt)
HSA_HF_PATH = "/apdcephfs/share_300719895/user/qqzxywei/wxy/checkpoints/hsa_8KA2K_HoPE_full_345M_dist-priorq-wloralmkq-loradim64/checkpoints/global_step_30000/hf_ckpt"

# ── 测试配置列表 ──
# (prefill_seq_len, max_new_tokens)
DEFAULT_TEST_CONFIGS = [
    # (1024,   1),     # 仅测 prefill
    # (4096,   1),
    # (8192,   1),
    # (16384,  1),
    # (32768,  1),
    # (65536,  1),
    # (131072, 1),
    # (262144, 1),
    # (524288, 1),
    # (1024,   128),   # 测 prefill + decode
    # (4096,   128),
    (8192,   20),
    # (16384,  128),
    # (32768,  128),
    # (65536,  128),
    # (131072, 128),
    # (262144, 128),
    # (524288, 128),
]


def load_model(hf_path, device, model_kwargs=None):
    """使用 transformers 加载模型"""
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_3",
    )
    if model_kwargs:
        kwargs.update(model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(hf_path, **kwargs)
    model.to(device).eval()
    return model


def load_real_data(data_path, seq_len, num_samples=1):
    """从真实数据集加载 num_samples 条序列，返回 list[Tensor]，每个 shape=(1, seq_len)"""
    dataset = build_numpy_dataset(data_path, seq_len, namespace='test')
    loader = DataLoader(dataset, batch_size=1,
                        collate_fn=lambda x: torch.tensor(x),
                        sampler=SequentialSampler(dataset), num_workers=0)
    samples = []
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        samples.append(batch)  # (1, seq_len)
    print(f"  加载了 {len(samples)} 条序列 (seq_len={seq_len})")
    return samples


@torch.no_grad()
def benchmark_generate(model, input_ids, max_new_tokens, num_runs, warmup, profile_lhsa=False, profile_label=None):
    """统一使用 model.generate 测试延迟（prefill + decode）
    
    - max_new_tokens=1 时，几乎全部时间为 prefill
    - max_new_tokens>1 时，总时间 = prefill + decode
    """
    profile_helpers = None
    if profile_lhsa:
        from models.FlashHSA.lhsa_layer_forbench import (
            enable_lhsa_profile,
            print_lhsa_profile_summary,
            reset_lhsa_profile,
        )
        profile_helpers = (
            enable_lhsa_profile,
            print_lhsa_profile_summary,
            reset_lhsa_profile,
        )

    # warmup
    for _ in range(warmup):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()

    if profile_helpers is not None:
        enable_lhsa_profile, _print_lhsa_profile_summary, reset_lhsa_profile = profile_helpers
        reset_lhsa_profile()
        enable_lhsa_profile(True)

    # 正式计时
    times = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    if profile_helpers is not None:
        enable_lhsa_profile, print_lhsa_profile_summary, _reset_lhsa_profile = profile_helpers
        enable_lhsa_profile(False)
        print_lhsa_profile_summary(total_ms=sum(times), label=profile_label)

    return times


def run_benchmark(args):
    device = torch.device("cuda:0")

    # 解析测试配置
    test_configs = []
    for i in range(0, len(args.test_configs), 2):
        prefill_len = args.test_configs[i]
        max_new = args.test_configs[i + 1]
        test_configs.append((prefill_len, max_new))

    results = {}  # (model_name, prefill_len, max_new_tokens, phase) -> (avg, std, n_samples)

    model_list = []
    if not args.skip_fullattn:
        model_list.append(("FullAttn", FULL_ATTN_HF_PATH, {}))
    if not args.skip_hsa:
        hsa_kwargs = {"auto_insert_lmk": True}
        # 初始化时使用 prefill 的 kernel 参数（首次测的是 prefill）
        if args.block_M_fwd is not None:
            hsa_kwargs["bench_block_M_fwd"] = args.block_M_fwd
        if args.block_M_bwd is not None:
            hsa_kwargs["bench_block_M_bwd"] = args.block_M_bwd
        if args.num_threads_fwd is not None:
            hsa_kwargs["bench_num_threads_fwd"] = args.num_threads_fwd
        if args.num_threads_bwd is not None:
            hsa_kwargs["bench_num_threads_bwd"] = args.num_threads_bwd
        model_list.append(("HSA", HSA_HF_PATH, hsa_kwargs))

    for name, hf_path, extra_kwargs in model_list:
        print(f"\n{'='*60}")
        print(f"Loading {name} model from: {hf_path}")
        model = load_model(hf_path, device, model_kwargs=extra_kwargs)

        # 模型加载后显式设置 HSA kernel 参数（from_pretrained 可能不会把自定义 kwargs 传给 __init__）
        if name == "HSA" and hasattr(model, 'set_hsa_kernel_params'):
            model.set_hsa_kernel_params(
                args.block_M_fwd, args.block_M_bwd,
                args.num_threads_fwd, args.num_threads_bwd,
            )

        for prefill_len, max_new_tokens in test_configs:
            print(f"\n  --- {name} | prefill_len={prefill_len}, max_new_tokens={max_new_tokens} ---")

            # 加载真实数据
            samples = load_real_data(args.data_path, prefill_len, num_samples=args.num_samples)

            prefill_times_all = []
            decode_times_all = []
            generate_times_all = []

            for sid, raw_input_ids in enumerate(samples):
                input_ids = raw_input_ids.to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # ── 始终先测 prefill（generate 1 token）──
                    # prefill 阶段：使用用户指定的 block_M（已在模型初始化时设置）
                    if name == "HSA" and hasattr(model, 'set_hsa_kernel_params'):
                        model.set_hsa_kernel_params(
                            args.block_M_fwd, args.block_M_bwd,
                            args.num_threads_fwd, args.num_threads_bwd,
                        )
                    p_times = benchmark_generate(
                        model, input_ids, 1,
                        num_runs=args.num_runs, warmup=args.warmup,
                        profile_lhsa=(args.profile_lhsa and name == "HSA"),
                        profile_label=f"sample={sid}, prefill_len={prefill_len}, max_new_tokens=1",
                    )
                    p_avg = np.mean(p_times)
                    print(f"    sample {sid} prefill : avg={p_avg:.2f}ms  ({[f'{t:.1f}' for t in p_times]})")
                    prefill_times_all.extend(p_times)

                    # ── 当 max_new_tokens > 1 时，再测 generate(N)，差值得到纯 decode ──
                    if max_new_tokens > 1:
                        # Decode baseline and generate(N) must use the same kernel params.
                        if name == "HSA" and hasattr(model, 'set_hsa_kernel_params'):
                            model.set_hsa_kernel_params(
                                None, None,
                                args.num_threads_fwd, args.num_threads_bwd,
                            )
                        p_decode_base_times = benchmark_generate(
                            model, input_ids, 1,
                            num_runs=args.num_runs, warmup=args.warmup,
                            profile_lhsa=False,
                            profile_label=None,
                        )
                        p_decode_base_avg = np.mean(p_decode_base_times)
                        g_times = benchmark_generate(
                            model, input_ids, max_new_tokens,
                            num_runs=args.num_runs, warmup=args.warmup,
                            profile_lhsa=(args.profile_lhsa and name == "HSA"),
                            profile_label=f"sample={sid}, prefill_len={prefill_len}, max_new_tokens={max_new_tokens}",
                        )
                        g_avg = np.mean(g_times)
                        # Subtract the prefill baseline measured with the same decode kernel params.
                        d_times = [g - p for g, p in zip(g_times, p_decode_base_times)]
                        d_avg = np.mean(d_times)
                        per_tok = d_avg / (max_new_tokens - 1)
                        print(f"    sample {sid} decode baseline(1tok): avg={p_decode_base_avg:.2f}ms")
                        print(f"    sample {sid} generate({max_new_tokens}tok): avg={g_avg:.2f}ms")
                        print(f"    sample {sid} decode   : avg={d_avg:.2f}ms (pure), per_token={per_tok:.2f}ms")
                        generate_times_all.extend(g_times)
                        decode_times_all.extend(d_times)

            # 汇总 prefill
            p_avg_all = np.mean(prefill_times_all)
            p_std_all = np.std(prefill_times_all)
            results[(name, prefill_len, max_new_tokens, "prefill")] = (p_avg_all, p_std_all, len(samples))
            print(f"  [{name}] prefill_len={prefill_len} | prefill avg={p_avg_all:.2f}ms, std={p_std_all:.2f}ms")

            # 汇总 decode（去除 prefill 开销）
            if max_new_tokens > 1 and decode_times_all:
                d_avg_all = np.mean(decode_times_all)
                d_std_all = np.std(decode_times_all)
                per_token = d_avg_all / (max_new_tokens - 1)
                g_avg_all = np.mean(generate_times_all)
                results[(name, prefill_len, max_new_tokens, "decode")] = (d_avg_all, d_std_all, len(samples))
                print(f"  [{name}] prefill_len={prefill_len} | decode(pure) avg={d_avg_all:.2f}ms, "
                      f"per_token={per_token:.2f}ms, std={d_std_all:.2f}ms  (generate_total={g_avg_all:.2f}ms)")

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # ==================== 汇总对比 ====================
    print(f"\n{'='*80}")
    print(f"{'Model':<12} {'Phase':<10} {'PrefillLen':>10} {'MaxNewTok':>10} "
          f"{'Samples':>8} {'Avg(ms)':>10} {'Std(ms)':>10} {'PerTok(ms)':>12}")
    print(f"{'-'*80}")

    for key in sorted(results.keys(), key=lambda x: (x[0], x[1], x[2], x[3])):
        name, prefill_len, max_new_tokens, phase = key
        # 过滤掉 max_new_tokens > 1 时的 prefill 行（它只是 decode 差值的基线，不需要单独展示）
        if phase == "prefill" and max_new_tokens > 1:
            continue
        avg, std, n_samples = results[key]
        per_tok_str = ""
        if phase == "decode" and max_new_tokens > 1:
            per_tok_str = f"{avg / (max_new_tokens - 1):.2f}"
        print(f"{name:<12} {phase:<10} {prefill_len:>10} {max_new_tokens:>10} "
              f"{n_samples:>8} {avg:>10.2f} {std:>10.2f} {per_tok_str:>12}")

    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark Prefill & Decode Latency: FullAttn vs HSA")
    parser.add_argument('--data_path', type=str,
                        default='/apdcephfs_sh8/share_300719895/shared/data/dolma3_mix-6T-1025-partial-tokenized',
                        help='真实数据路径 (tokenized numpy)')
    parser.add_argument('--test_configs', type=int, nargs='+',
                        default=None,
                        help='测试配置列表，格式: prefill_len1 max_new1 prefill_len2 max_new2 ...')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='每个配置测试的不同序列数量')
    parser.add_argument('--num_runs', type=int, default=5, help='每条序列的正式计时次数')
    parser.add_argument('--warmup', type=int, default=3, help='每条序列的 warmup 次数')
    parser.add_argument('--skip_fullattn', action='store_true', help='跳过 FullAttn 模型测试')
    parser.add_argument('--skip_hsa', action='store_true', help='跳过 HSA 模型测试')
    parser.add_argument('--profile_lhsa', action='store_true', help='统计 HSA layer 内部四个核心模块的耗时')
    # HSA kernel 参数（仅 prefill 阶段使用，decode 阶段自动传 None）
    parser.add_argument('--block_M_fwd', type=int, default=8,
                        help='HSA prefill 阶段的 block_M_fwd（None 则由 kernel 自动选择）')
    parser.add_argument('--block_M_bwd', type=int, default=16,
                        help='HSA 的 block_M_bwd（benchmark 不走反向，一般不需要设置）')
    parser.add_argument('--num_threads_fwd', type=int, default=128,
                        help='HSA 前向 kernel 的线程数（None 则由 kernel 自动选择）')
    parser.add_argument('--num_threads_bwd', type=int, default=128,
                        help='HSA 反向 kernel 的线程数（benchmark 不走反向，一般不需要设置）')
    args = parser.parse_args()

    # 如果未指定 test_configs，使用默认配置
    if args.test_configs is None:
        args.test_configs = []
        for prefill_len, max_new in DEFAULT_TEST_CONFIGS:
            args.test_configs.extend([prefill_len, max_new])

    if len(args.test_configs) % 2 != 0:
        raise ValueError("--test_configs 必须成对出现: prefill_len max_new_tokens")

    run_benchmark(args)


# pkill -f "burner.*--gpu 6"; export CUDA_VISIBLE_DEVICES=6; python code_exp/bench_topk_hsa_vs_FA.py --skip_fullattn --profile_lhsa --num_samples 1
