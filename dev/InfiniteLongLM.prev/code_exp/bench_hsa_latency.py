"""
HSA Latency Benchmark
=====================
专门用于测试 HSA kernel 在不同 HQ:H 比例下的延迟表现。
支持 block_M_fwd / block_M_bwd 独立遍历，以及 num_threads_fwd / num_threads_bwd 独立遍历。

用法:
    python code_exp/bench_hsa_latency.py
    python code_exp/bench_hsa_latency.py --max_block_M 16 --overlap 0.6
    python code_exp/bench_hsa_latency.py --B 4 --SEQ_LEN 8192 --D 64 --S 16 --block_size 64

对于每个 HQ:H = G，block_M_fwd / block_M_bwd 分别从 max_block_M // G 开始翻倍遍历到 max_block_M:
    G=16 → M: [1, 2, 4, 8, 16]
    G=8  → M: [2, 4, 8, 16]
    G=4  → M: [4, 8, 16]
    G=2  → M: [8, 16]
    G=1  → M: [16]

num_threads_fwd / num_threads_bwd 分别遍历 [64, 128, 256, 512]。
"""

import torch
import torch.nn.functional as F
import math
import argparse
import random as _random


# =========================================================
# FlashAttention Baseline 延迟测试
# =========================================================
def bench_flash_attn(
    B: int,
    SEQ_LEN: int,
    H: int,
    HQ: int,
    D: int,
    num_warmup: int = 50,
    num_iters: int = 100,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    """
    测试 FlashAttention (flash_attn_func) 在给定配置下的 FWD / BWD / Total 延迟。
    作为 HSA kernel 的 baseline 对比。
    返回 (fwd_ms, bwd_ms, total_ms)。
    """
    from flash_attn_interface import flash_attn_func

    scale = 1.0 / math.sqrt(D)

    # ---------- 构造输入张量 ----------
    # flash_attn_func 期望的输入格式: (B, SEQ_LEN, H, D)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    Q_bench = Q.detach().clone().requires_grad_(True)

    # ---------- 计时辅助函数 ----------
    def measure_time(func):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters  # ms per iter

    # ---------- Warmup ----------
    for _ in range(num_warmup):
        O = flash_attn_func(
            Q_bench, K, V,
            softmax_scale=scale,
            causal=True,
        )
        O.backward(grad_output)

    # ---------- FWD only ----------
    def fa_fwd():
        with torch.no_grad():
            flash_attn_func(
                Q_bench, K, V,
                softmax_scale=scale,
                causal=True,
            )

    fwd_ms = measure_time(fa_fwd)

    # ---------- FWD + BWD ----------
    def fa_fwd_bwd():
        Q_bench.grad = K.grad = V.grad = None
        O = flash_attn_func(
            Q_bench, K, V,
            softmax_scale=scale,
            causal=True,
        )
        O.backward(grad_output)

    total_ms = measure_time(fa_fwd_bwd)
    bwd_ms = total_ms - fwd_ms

    return fwd_ms, bwd_ms, total_ms


# =========================================================
# build_block_indices_union_overlap: 直接控制 M 路并集大小
# =========================================================
def build_block_indices_union_overlap(
    B: int,
    SEQ_LEN: int,
    H: int,
    S: int,
    block_size: int,
    overlap_ratio: float = 0.5,
    block_M: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    """
    构造 block_indices 张量，直接控制 block_M 个 token 取并集后的大小。

    重合度定义（M 路并集）：
        overlap_ratio = (M * S - merged_len) / ((M - 1) * S)
    反推：
        merged_len = S * (M - overlap_ratio * (M - 1))

    overlap=1.0 → merged_len=S（M 个 token 选的 block 完全一样，最快）
    overlap=0.0 → merged_len=M*S（M 个 token 选的 block 完全不同，最慢）

    构造算法（两阶段：覆盖 + 填充）：
      1. 从可用 block 中随机选 merged_len 个组成 pool
      2. 覆盖阶段：round-robin 将 pool 中每个 block 分配给至少一个 token（保证并集 = merged_len）
      3. 填充阶段：每个 token 从 pool 中补选到恰好 S 个
      4. 排序写入
    """
    assert 0.0 <= overlap_ratio <= 1.0, "overlap_ratio 必须在 [0, 1]"
    assert block_M >= 1, "block_M 必须 >= 1"

    num_blocks = SEQ_LEN // block_size
    block_indices = torch.full((B, SEQ_LEN, H, S), -1, dtype=torch.int32, device=device)

    # --- debug 统计 ---
    debug_total_windows = 0
    debug_sum_actual_overlap = 0.0

    for b in range(B):
        for h in range(H):
            t = 0
            while t < SEQ_LEN:
                block_start = t
                actual_M = min(block_M, SEQ_LEN - t)  # 窗口末尾可能不足 block_M
                block_end = t + actual_M

                # 当前窗口内所有 token 可用的最大 block 数（取窗口内最后一个 token 的可用数）
                last_t = block_end - 1
                max_avail = min(last_t // block_size + 1, num_blocks)
                if max_avail <= 0 or actual_M <= 0:
                    t = block_end
                    continue

                num_select = min(S, max_avail)

                if actual_M == 1:
                    # 只有一个 token，直接随机选
                    idx = torch.randperm(max_avail, device=device)[:num_select]
                    idx_sorted = torch.sort(idx)[0]
                    block_indices[b, block_start, h, :num_select] = idx_sorted
                    t = block_end
                    continue

                # 计算目标 merged_len
                target_merged = int(round(num_select * (actual_M - overlap_ratio * (actual_M - 1))))
                target_merged = max(num_select, min(target_merged, actual_M * num_select))
                # 钳位到可用 block 数
                target_merged = min(target_merged, max_avail)

                # Step 1: 从可用 block 中随机选 target_merged 个组成 pool
                pool_indices = torch.randperm(max_avail, device=device)[:target_merged]
                pool_list = pool_indices.tolist()

                # Step 2: 覆盖阶段 - round-robin 分配，确保 pool 中每个 block 至少被一个 token 持有
                token_sets = [set() for _ in range(actual_M)]
                _random.shuffle(pool_list)
                robin_idx = 0
                for blk in pool_list:
                    # 找到下一个还没满的 token
                    attempts = 0
                    while len(token_sets[robin_idx]) >= num_select and attempts < actual_M:
                        robin_idx = (robin_idx + 1) % actual_M
                        attempts += 1
                    if attempts < actual_M:
                        token_sets[robin_idx].add(blk)
                        robin_idx = (robin_idx + 1) % actual_M

                # Step 3: 填充阶段 - 每个 token 从 pool 中补选到 num_select 个
                for m_idx in range(actual_M):
                    while len(token_sets[m_idx]) < num_select:
                        # 从 pool 中随机选一个自己还没有的
                        candidates = [blk for blk in pool_list if blk not in token_sets[m_idx]]
                        if len(candidates) > 0:
                            token_sets[m_idx].add(candidates[_random.randint(0, len(candidates) - 1)])
                        else:
                            # pool 中所有 block 都已被选中，理论上不会到这里
                            break

                # Step 4: 排序写入
                for m_idx in range(actual_M):
                    tt = block_start + m_idx
                    sorted_idx = sorted(token_sets[m_idx])
                    for j, v in enumerate(sorted_idx):
                        block_indices[b, tt, h, j] = v

                # --- debug: 计算实际并集大小和 overlap ---
                union_set = set()
                for m_idx in range(actual_M):
                    union_set |= token_sets[m_idx]
                actual_merged_len = len(union_set)
                if actual_M > 1:
                    actual_overlap = (actual_M * num_select - actual_merged_len) / ((actual_M - 1) * num_select)
                else:
                    actual_overlap = 1.0
                debug_total_windows += 1
                debug_sum_actual_overlap += actual_overlap

                t = block_end

    # --- debug print ---
    if debug_total_windows > 0:
        avg_overlap = debug_sum_actual_overlap / debug_total_windows
        print(f"[build_block_indices_union_overlap] "
              f"target overlap_ratio={overlap_ratio:.4f}, "
              f"actual avg overlap_ratio={avg_overlap:.4f}, "
              f"total windows={debug_total_windows}, "
              f"block_M={block_M}, S={S}")

    return block_indices


# =========================================================
# 单次 latency 测试
# =========================================================
def bench_single_config(
    B: int,
    SEQ_LEN: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    block_M_fwd: int,
    block_M_bwd: int,
    overlap_ratio: float,
    num_threads_fwd: int = 128,
    num_threads_bwd: int = 256,
    mask_last_token: bool = True,
    num_warmup: int = 50,
    num_iters: int = 100,
    dtype=torch.bfloat16,
    device: str = "cuda",
    block_indices=None,
):
    """
    对给定配置运行 HSA_block_M_group 的 FWD / BWD latency 测试。
    block_M_fwd / block_M_bwd 分别控制前向和反向的 block_M。
    num_threads_fwd / num_threads_bwd 分别控制前向和反向的线程数。
    可传入预构造的 block_indices 避免重复构造。
    返回 (fwd_ms, bwd_ms, total_ms)。
    """
    from ops.hsa_fwd_bwd_group_forbench import HSA_block_M_group

    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    # ---------- 构造 block_indices（如果未传入）----------
    if block_indices is None:
        # 用 max(block_M_fwd, block_M_bwd) 来构造 indices，保证两者都能用
        block_indices = build_block_indices_union_overlap(
            B=B, SEQ_LEN=SEQ_LEN, H=H, S=S,
            block_size=block_size,
            overlap_ratio=overlap_ratio,
            block_M=max(block_M_fwd, block_M_bwd),
            device=device,
        )

    # ---------- 构造输入张量 ----------
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)

    # 生成权重
    logits = torch.randn((B, SEQ_LEN, H, S), dtype=dtype, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, float('-inf'))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, nan=0.0).detach().requires_grad_(True)

    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    Q_tile = Q.detach().clone().requires_grad_(True)
    grad_output_tile = grad_output

    # ---------- 计时辅助函数 ----------
    def measure_time(func):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters  # ms per iter

    # ---------- Warmup ----------
    for _ in range(num_warmup):
        O = HSA_block_M_group(
            Q_tile, K, V, W, block_indices,
            block_size=block_size, sm_scale=scale,
            block_M_fwd=block_M_fwd, block_M_bwd=block_M_bwd,
            num_threads_fwd=num_threads_fwd, num_threads_bwd=num_threads_bwd,
            mask_last_token=mask_last_token,
        )
        O.backward(grad_output_tile)

    # ---------- FWD only ----------
    def tile_fwd():
        with torch.no_grad():
            HSA_block_M_group(
                Q_tile, K, V, W, block_indices,
                block_size=block_size, sm_scale=scale,
                block_M_fwd=block_M_fwd, block_M_bwd=block_M_bwd,
                num_threads_fwd=num_threads_fwd, num_threads_bwd=num_threads_bwd,
                mask_last_token=mask_last_token,
            )

    fwd_ms = measure_time(tile_fwd)

    # ---------- FWD + BWD ----------
    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_group(
            Q_tile, K, V, W, block_indices,
            block_size=block_size, sm_scale=scale,
            block_M_fwd=block_M_fwd, block_M_bwd=block_M_bwd,
            num_threads_fwd=num_threads_fwd, num_threads_bwd=num_threads_bwd,
            mask_last_token=mask_last_token,
        )
        O.backward(grad_output_tile)

    total_ms = measure_time(tile_fwd_bwd)
    bwd_ms = total_ms - fwd_ms

    return fwd_ms, bwd_ms, total_ms


# =========================================================
# 批量测试不同 HQ:H 比例
# =========================================================
def _get_block_M_list(G, max_block_M=16):
    """
    对于给定的 G (= HQ // H)，生成 block_M 遍历列表。
    从 16 // G 开始（16 固定），每次翻倍，最大到 max_block_M。
    额外约束：M * G <= 128，超过则跳过（BWD kernel shared memory 会超限）。
    例如（max_block_M=64）：
      G=16 → [1, 2, 4, 8]          (M=16 时 M*G=256 > 128，跳过)
      G=8  → [2, 4, 8, 16]         (M=32 时 M*G=256 > 128，跳过)
      G=4  → [4, 8, 16, 32]        (M=64 时 M*G=256 > 128，跳过)
      G=2  → [8, 16, 32, 64]       (M*G 最大=128，OK)
      G=1  → [16, 32, 64]          (M*G 最大=64，OK)
    """
    start_M = max(1, 16 // G)
    result = []
    m = start_M
    while m <= max_block_M:
        if m * G <= 128:
            result.append(m)
        m *= 2
    return result


# num_threads 遍历列表
NUM_THREADS_LIST = [64, 128]


def bench_hsa_latency(args):
    """
    批量测试不同 HQ:H 比例在给定 overlap_ratio 下的 HSA latency。
    遍历维度：
      - HQ:H 比例: 16:1, 8:1, 4:1, 2:1, 1:1
      - block_M_fwd × block_M_bwd（各自从 16//G 翻倍到 max_block_M）
      - num_threads_fwd × num_threads_bwd（各自遍历 [64, 128, 256, 512]）
    """
    # 基础配置
    B = args.B
    SEQ_LEN = args.SEQ_LEN
    D = args.D
    S = args.S
    block_size = args.block_size
    max_block_M = args.max_block_M
    overlap_ratio = args.overlap
    mask_last_token = args.mask_last_token

    # HQ:H 比例列表
    base_H = args.base_H
    ratios = [16, 8, 4, 2, 1]

    print("=" * 120)
    print(f"HSA Latency Benchmark (FWD/BWD 独立参数遍历)")
    print(f"Config: B={B}, L={SEQ_LEN}, D={D}, S={S}, block_size={block_size}, "
          f"max_block_M={max_block_M}, overlap={overlap_ratio}, base_H={base_H}")
    print(f"mask_last_token={mask_last_token}")
    print(f"block_M 遍历规则: 对于 G=HQ//H, M_fwd/M_bwd 各自从 16//G 开始翻倍到 {max_block_M}")
    print(f"num_threads 遍历: fwd/bwd 各自遍历 {NUM_THREADS_LIST}")
    print("=" * 120)
    print()

    # (ratio, H, HQ, G, M_fwd, M_bwd, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)
    results = []
    # 每个 ratio 的最优配置: {ratio: {"best_fwd": (...), "best_bwd": (...), "best_total": (...)}}
    best_configs = {}

    for ratio in ratios:
        G = ratio
        H = base_H
        HQ = H * G

        block_M_list = _get_block_M_list(G, max_block_M)

        print(f"\n{'='*100}")
        print(f">>> HQ:H = {ratio}:1  (H={H}, HQ={HQ}, G={G})")
        print(f">>> block_M 遍历列表: {block_M_list}")
        print(f">>> num_threads 遍历列表: {NUM_THREADS_LIST}")
        print(f"{'='*100}")

        # 预构造 block_indices（用最大的 block_M 来构造，所有子配置共享）
        max_M_in_list = max(block_M_list)
        block_indices = build_block_indices_union_overlap(
            B=B, SEQ_LEN=SEQ_LEN, H=H, S=S,
            block_size=block_size,
            overlap_ratio=overlap_ratio,
            block_M=max_M_in_list,
            device="cuda",
        )

        for block_M_fwd in block_M_list:
            for block_M_bwd in block_M_list:
                for nt_fwd in NUM_THREADS_LIST:
                    for nt_bwd in NUM_THREADS_LIST:
                        print(f"\n  --- M_fwd={block_M_fwd}, M_bwd={block_M_bwd}, "
                              f"nt_fwd={nt_fwd}, nt_bwd={nt_bwd} ---")
                        try:
                            fwd_ms, bwd_ms, total_ms = bench_single_config(
                                B=B,
                                SEQ_LEN=SEQ_LEN,
                                H=H,
                                HQ=HQ,
                                D=D,
                                S=S,
                                block_size=block_size,
                                block_M_fwd=block_M_fwd,
                                block_M_bwd=block_M_bwd,
                                overlap_ratio=overlap_ratio,
                                num_threads_fwd=nt_fwd,
                                num_threads_bwd=nt_bwd,
                                mask_last_token=mask_last_token,
                                block_indices=block_indices,
                            )
                        except Exception as e:
                            print(f"  ⚠ SKIPPED (M_fwd={block_M_fwd}, M_bwd={block_M_bwd}, "
                                  f"nt_fwd={nt_fwd}, nt_bwd={nt_bwd}): {type(e).__name__}: {e}")
                            continue

                        results.append((ratio, H, HQ, G, block_M_fwd, block_M_bwd,
                                        nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms))
                        print(f"  HQ:H={ratio}:1 | G={G} | M_fwd={block_M_fwd} M_bwd={block_M_bwd} | "
                              f"nt_fwd={nt_fwd} nt_bwd={nt_bwd} | "
                              f"FWD={fwd_ms:.3f}ms | BWD={bwd_ms:.3f}ms | Total={total_ms:.3f}ms")

        # --- 当前 ratio 遍历完毕，选出最优配置 ---
        ratio_results = [r for r in results if r[0] == ratio]
        if not ratio_results:
            print(f"\n  ⚠ HQ:H={ratio}:1 所有参数组合均失败，无最优配置")
            continue

        best_fwd_entry = min(ratio_results, key=lambda x: x[8])   # fwd_ms
        best_bwd_entry = min(ratio_results, key=lambda x: x[9])   # bwd_ms
        best_total_entry = min(ratio_results, key=lambda x: x[10]) # total_ms
        best_configs[ratio] = {
            "best_fwd": best_fwd_entry,
            "best_bwd": best_bwd_entry,
            "best_total": best_total_entry,
        }

        print(f"\n  {'*'*80}")
        print(f"  ★ HQ:H={ratio}:1 最优配置:")
        _r = best_fwd_entry
        print(f"    Best FWD:   M_fwd={_r[4]}, nt_fwd={_r[6]}  "
              f"→ FWD={_r[8]:.3f}ms")
        _r = best_bwd_entry
        print(f"    Best BWD:   M_bwd={_r[5]}, nt_bwd={_r[7]}  "
              f"→ BWD={_r[9]:.3f}ms")
        _r = best_total_entry
        print(f"    Best Total: M_fwd={_r[4]}, M_bwd={_r[5]}, nt_fwd={_r[6]}, nt_bwd={_r[7]}  "
              f"→ Total={_r[10]:.3f}ms (FWD={_r[8]:.3f}ms + BWD={_r[9]:.3f}ms)")
        print(f"  {'*'*80}")

    # 汇总表格
    print()
    print("=" * 140)
    print("Summary Table")
    print("=" * 140)
    header = (f"{'HQ:H':>8} | {'G':>4} | {'M_fwd':>5} | {'M_bwd':>5} | "
              f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
              f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12}")
    print(header)
    print("-" * len(header))
    prev_ratio = None
    for ratio, H, HQ, G, M_fwd, M_bwd, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms in results:
        if prev_ratio is not None and ratio != prev_ratio:
            print("-" * len(header))  # 不同 G 之间加分隔线
        prev_ratio = ratio
        print(f"{'%d:1' % ratio:>8} | {G:>4} | {M_fwd:>5} | {M_bwd:>5} | "
              f"{nt_fwd:>6} | {nt_bwd:>6} | "
              f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f}")
    print("=" * 140)

    # ========== 最优配置汇总（仅展示 Best Total 对应的 FWD/BWD/Total 拆分）==========
    print()
    print("=" * 140)
    print("Best Config per HQ:H Ratio (Best Total with FWD/BWD breakdown)")
    print("=" * 140)
    best_header = (f"{'HQ:H':>8} | {'G':>4} | {'M_fwd':>5} | {'M_bwd':>5} | "
                   f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
                   f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12}")
    print(best_header)
    print("-" * len(best_header))
    for ratio in ratios:
        if ratio not in best_configs:
            continue
        bc = best_configs[ratio]
        G = ratio  # G == ratio
        _r = bc["best_total"]
        print(f"{'%d:1' % ratio:>8} | {G:>4} | {_r[4]:>5} | {_r[5]:>5} | "
              f"{_r[6]:>6} | {_r[7]:>6} | "
              f"{_r[8]:>10.3f} | {_r[9]:>10.3f} | {_r[10]:>12.3f}")
    print("=" * len(best_header))


# =========================================================
# 多序列长度遍历搜索最优参数
# =========================================================
SEQ_LEN_LIST = [32768, 65536, 131072]

def bench_sweep_seq_len(args):
    """
    遍历多个序列长度 (8k, 16k, 32k, 64k, 128k)，
    对每个长度 × 每个 HQ:H 比例搜索最优的 (M_fwd, M_bwd, nt_fwd, nt_bwd) 参数。
    最终输出一张汇总表。
    """
    B = args.B
    D = args.D
    S = args.S
    block_size = args.block_size
    max_block_M = args.max_block_M
    overlap_ratio = args.overlap
    mask_last_token = args.mask_last_token
    base_H = args.base_H

    ratios = [16, 8, 4, 2, 1]
    seq_len_list = SEQ_LEN_LIST

    print("=" * 140)
    print(f"HSA Latency Benchmark (Sweep SEQ_LEN Mode)")
    print(f"Config: B={B}, D={D}, S={S}, block_size={block_size}, "
          f"max_block_M={max_block_M}, overlap={overlap_ratio}, base_H={base_H}")
    print(f"mask_last_token={mask_last_token}")
    print(f"SEQ_LEN list: {seq_len_list}")
    print(f"HQ:H ratios: {[f'{r}:1' for r in ratios]}")
    print(f"block_M 遍历规则: 对于 G=HQ//H, M_fwd/M_bwd 各自从 16//G 开始翻倍到 {max_block_M}")
    print(f"num_threads 遍历: fwd/bwd 各自遍历 {NUM_THREADS_LIST}")
    print("=" * 140)
    print()

    # 汇总结果: {(seq_len, ratio): {"best_total": (M_fwd, M_bwd, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)}}
    sweep_best = {}

    for seq_len in seq_len_list:
        print(f"\n{'#'*140}")
        print(f"### SEQ_LEN = {seq_len} ({seq_len//1024}k)")
        print(f"{'#'*140}")

        for ratio in ratios:
            G = ratio
            H = base_H
            HQ = H * G

            block_M_list = _get_block_M_list(G, max_block_M)

            print(f"\n  {'='*100}")
            print(f"  >>> SEQ_LEN={seq_len}, HQ:H = {ratio}:1  (H={H}, HQ={HQ}, G={G})")
            print(f"  >>> block_M 遍历列表: {block_M_list}")
            print(f"  {'='*100}")

            # 预构造 block_indices（用最大的 block_M 来构造）
            max_M_in_list = max(block_M_list)
            block_indices = build_block_indices_union_overlap(
                B=B, SEQ_LEN=seq_len, H=H, S=S,
                block_size=block_size,
                overlap_ratio=overlap_ratio,
                block_M=max_M_in_list,
                device="cuda",
            )

            best_total_ms = float('inf')
            best_entry = None

            for block_M_fwd in block_M_list:
                for block_M_bwd in block_M_list:
                    for nt_fwd in NUM_THREADS_LIST:
                        for nt_bwd in NUM_THREADS_LIST:
                            try:
                                fwd_ms, bwd_ms, total_ms = bench_single_config(
                                    B=B, SEQ_LEN=seq_len, H=H, HQ=HQ, D=D, S=S,
                                    block_size=block_size,
                                    block_M_fwd=block_M_fwd, block_M_bwd=block_M_bwd,
                                    overlap_ratio=overlap_ratio,
                                    num_threads_fwd=nt_fwd, num_threads_bwd=nt_bwd,
                                    mask_last_token=mask_last_token,
                                    block_indices=block_indices,
                                )
                            except Exception as e:
                                print(f"    ⚠ SKIPPED (M_fwd={block_M_fwd}, M_bwd={block_M_bwd}, "
                                      f"nt_fwd={nt_fwd}, nt_bwd={nt_bwd}): {type(e).__name__}: {e}")
                                continue

                            print(f"    M_fwd={block_M_fwd}, M_bwd={block_M_bwd}, "
                                  f"nt_fwd={nt_fwd}, nt_bwd={nt_bwd} | "
                                  f"FWD={fwd_ms:.3f}ms BWD={bwd_ms:.3f}ms Total={total_ms:.3f}ms")

                            if total_ms < best_total_ms:
                                best_total_ms = total_ms
                                best_entry = (block_M_fwd, block_M_bwd, nt_fwd, nt_bwd,
                                              fwd_ms, bwd_ms, total_ms)

            if best_entry is not None:
                sweep_best[(seq_len, ratio)] = best_entry
                print(f"\n  ★ Best for SEQ_LEN={seq_len}, HQ:H={ratio}:1: "
                      f"M_fwd={best_entry[0]}, M_bwd={best_entry[1]}, "
                      f"nt_fwd={best_entry[2]}, nt_bwd={best_entry[3]} → "
                      f"Total={best_entry[6]:.3f}ms (FWD={best_entry[4]:.3f}ms + BWD={best_entry[5]:.3f}ms)")
            else:
                print(f"\n  ⚠ SEQ_LEN={seq_len}, HQ:H={ratio}:1 所有参数组合均失败")

            # 释放 block_indices 显存
            del block_indices
            import gc; gc.collect()
            import torch as _torch; _torch.cuda.empty_cache()

    # ========== 汇总表格 ==========
    print()
    print("=" * 160)
    print("Sweep SEQ_LEN - Best Config Summary")
    print("=" * 160)
    sweep_header = (f"{'SEQ_LEN':>10} | {'HQ:H':>8} | {'G':>4} | {'M_fwd':>5} | {'M_bwd':>5} | "
                    f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
                    f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12}")
    print(sweep_header)
    print("-" * len(sweep_header))
    prev_seq_len = None
    for seq_len in seq_len_list:
        if prev_seq_len is not None and seq_len != prev_seq_len:
            print("-" * len(sweep_header))
        prev_seq_len = seq_len
        for ratio in ratios:
            key = (seq_len, ratio)
            if key not in sweep_best:
                continue
            entry = sweep_best[key]
            G = ratio
            print(f"{seq_len:>10} | {'%d:1' % ratio:>8} | {G:>4} | {entry[0]:>5} | {entry[1]:>5} | "
                  f"{entry[2]:>6} | {entry[3]:>6} | "
                  f"{entry[4]:>10.3f} | {entry[5]:>10.3f} | {entry[6]:>12.3f}")
    print("=" * len(sweep_header))

    # ========== 输出可直接用于 --config 的最优配置字符串 ==========
    print()
    print("=" * 160)
    print("Best --config strings per SEQ_LEN (可直接复制使用)")
    print("=" * 160)
    for seq_len in seq_len_list:
        config_parts = []
        for ratio in ratios:
            key = (seq_len, ratio)
            if key in sweep_best:
                entry = sweep_best[key]
                config_parts.append(f"{ratio}:{entry[0]}:{entry[1]}:{entry[2]}:{entry[3]}")
        if config_parts:
            config_str = ",".join(config_parts)
            print(f"  SEQ_LEN={seq_len:>6} ({seq_len//1024:>3}k): --config \"{config_str}\"")
    print("=" * 160)


# =========================================================
# 指定配置测试模式
# =========================================================
def bench_fixed_configs(args):
    """
    直接测试指定的一套最优配置，跳过全量搜索。
    通过 --config 参数传入，格式为:
        "G:M_fwd:M_bwd:nt_fwd:nt_bwd,..."
    例如:
        "16:4:8:128:128,8:8:16:128:128,4:4:32:64:128,2:16:16:128:64,1:16:32:128:64"
    """
    B = args.B
    SEQ_LEN = args.SEQ_LEN
    D = args.D
    S = args.S
    block_size = args.block_size
    overlap_ratio = args.overlap
    mask_last_token = args.mask_last_token
    base_H = args.base_H

    # 解析 config 字符串
    configs = []
    for item in args.config.split(","):
        parts = item.strip().split(":")
        assert len(parts) == 5, f"每个配置需要 5 个字段 (G:M_fwd:M_bwd:nt_fwd:nt_bwd)，得到: {item}"
        G, M_fwd, M_bwd, nt_fwd, nt_bwd = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        configs.append((G, M_fwd, M_bwd, nt_fwd, nt_bwd))

    print("=" * 120)
    print(f"HSA Latency Benchmark (Fixed Config Mode)")
    print(f"Config: B={B}, L={SEQ_LEN}, D={D}, S={S}, block_size={block_size}, "
          f"overlap={overlap_ratio}, base_H={base_H}")
    print(f"mask_last_token={mask_last_token}")
    print("=" * 120)
    print()

    # 结果表头
    header = (f"{'HQ:H':>8} | {'G':>4} | {'M_fwd':>5} | {'M_bwd':>5} | "
              f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
              f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12}")

    results = []
    for G, M_fwd, M_bwd, nt_fwd, nt_bwd in configs:
        H = base_H
        HQ = H * G
        ratio = G

        print(f"\n>>> Testing HQ:H = {ratio}:1  (H={H}, HQ={HQ}, G={G})")
        print(f"    M_fwd={M_fwd}, M_bwd={M_bwd}, nt_fwd={nt_fwd}, nt_bwd={nt_bwd}")

        try:
            fwd_ms, bwd_ms, total_ms = bench_single_config(
                B=B, SEQ_LEN=SEQ_LEN, H=H, HQ=HQ, D=D, S=S,
                block_size=block_size,
                block_M_fwd=M_fwd, block_M_bwd=M_bwd,
                overlap_ratio=overlap_ratio,
                num_threads_fwd=nt_fwd, num_threads_bwd=nt_bwd,
                mask_last_token=mask_last_token,
            )
            results.append((ratio, G, M_fwd, M_bwd, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms))
            print(f"    ✓ FWD={fwd_ms:.3f}ms | BWD={bwd_ms:.3f}ms | Total={total_ms:.3f}ms")
        except Exception as e:
            print(f"    ⚠ FAILED: {type(e).__name__}: {e}")

    # ========== FlashAttention Baseline ==========
    fa_results = []
    if getattr(args, 'with_baseline', False):
        print("\n" + "=" * 120)
        print("FlashAttention Baseline")
        print("=" * 120)
        # 对每个 ratio 测一次 FA baseline（同样的 B, SEQ_LEN, HQ, H, D）
        tested_ratios = set()
        for G, M_fwd, M_bwd, nt_fwd, nt_bwd in configs:
            if G in tested_ratios:
                continue
            tested_ratios.add(G)
            H = base_H
            HQ = H * G
            ratio = G
            print(f"\n>>> FA Baseline: HQ:H = {ratio}:1  (H={H}, HQ={HQ})")
            try:
                fa_fwd_ms, fa_bwd_ms, fa_total_ms = bench_flash_attn(
                    B=B, SEQ_LEN=SEQ_LEN, H=H, HQ=HQ, D=D,
                )
                fa_results.append((ratio, G, fa_fwd_ms, fa_bwd_ms, fa_total_ms))
                print(f"    ✓ FWD={fa_fwd_ms:.3f}ms | BWD={fa_bwd_ms:.3f}ms | Total={fa_total_ms:.3f}ms")
            except Exception as e:
                print(f"    ⚠ FAILED: {type(e).__name__}: {e}")

    # ========== 汇总表格 ==========
    # 构建 FA baseline 查找表
    fa_lookup = {r[0]: r for r in fa_results}  # ratio -> (ratio, G, fwd, bwd, total)

    print()
    print("=" * 120)
    print("Results (Fixed Config)")
    print("=" * 120)

    if fa_results:
        header_with_fa = (f"{'HQ:H':>8} | {'G':>4} | {'M_fwd':>5} | {'M_bwd':>5} | "
                          f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
                          f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12} | "
                          f"{'FA_FWD':>10} | {'FA_BWD':>10} | {'FA_Total':>12} | "
                          f"{'Speedup':>8}")
        print(header_with_fa)
        print("-" * len(header_with_fa))
        for ratio, G, M_fwd, M_bwd, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms in results:
            fa = fa_lookup.get(ratio)
            if fa:
                speedup = fa[4] / total_ms if total_ms > 0 else float('inf')
                print(f"{'%d:1' % ratio:>8} | {G:>4} | {M_fwd:>5} | {M_bwd:>5} | "
                      f"{nt_fwd:>6} | {nt_bwd:>6} | "
                      f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f} | "
                      f"{fa[2]:>10.3f} | {fa[3]:>10.3f} | {fa[4]:>12.3f} | "
                      f"{speedup:>7.2f}x")
            else:
                print(f"{'%d:1' % ratio:>8} | {G:>4} | {M_fwd:>5} | {M_bwd:>5} | "
                      f"{nt_fwd:>6} | {nt_bwd:>6} | "
                      f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f} | "
                      f"{'N/A':>10} | {'N/A':>10} | {'N/A':>12} | {'N/A':>8}")
        print("=" * len(header_with_fa))
    else:
        print(header)
        print("-" * len(header))
        for ratio, G, M_fwd, M_bwd, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms in results:
            print(f"{'%d:1' % ratio:>8} | {G:>4} | {M_fwd:>5} | {M_bwd:>5} | "
                  f"{nt_fwd:>6} | {nt_bwd:>6} | "
                  f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f}")
        print("=" * len(header))


# =========================================================
# 命令行入口
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="HSA Latency Benchmark - 测试不同 HQ:H 比例下的 HSA 延迟（FWD/BWD 独立参数）")
    parser.add_argument("--max_block_M", type=int, default=128, help="block_M 遍历的最大值（默认 128），M_fwd/M_bwd 各自从 max_block_M//G 开始翻倍到 max_block_M")
    parser.add_argument("--overlap", type=float, default=0.8, help="overlap_ratio（默认 0.8）")
    parser.add_argument("--B", type=int, default=4, help="batch size（默认 4）")
    parser.add_argument("--SEQ_LEN", type=int, default=8192*2, help="序列长度（默认 8192）")
    parser.add_argument("--D", type=int, default=64, help="head dim（默认 64）")
    parser.add_argument("--S", type=int, default=16, help="每个 query 选择的 block 数量（默认 16）")
    parser.add_argument("--block_size", type=int, default=64, help="block size（默认 64）")
    parser.add_argument("--base_H", type=int, default=4, help="基准 KV head 数，HQ = base_H * G（默认 4）")
    parser.add_argument("--mask_last_token", action="store_true", default=True, help="是否 mask last token（默认 True）")
    parser.add_argument("--no_mask_last_token", action="store_true", default=False, help="关闭 mask last token")
    parser.add_argument("--config", type=str, default=None,
                        help="指定固定配置直接测试，跳过全量搜索。"
                             "格式: 'G:M_fwd:M_bwd:nt_fwd:nt_bwd,...' "
                             "例如: '16:4:8:128:128,8:8:16:128:128,4:4:32:64:128,2:16:16:128:64,1:16:32:128:64'")
    parser.add_argument("--with_baseline", action="store_true", default=False,
                        help="同时测试 FlashAttention baseline 作为对比")
    parser.add_argument("--sweep_seq_len", action="store_true", default=False,
                        help="遍历多个序列长度 (8k~128k) 搜索每个长度下的最优参数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_mask_last_token:
        args.mask_last_token = False

    if args.sweep_seq_len:
        bench_sweep_seq_len(args)
    elif args.config:
        bench_fixed_configs(args)
    else:
        bench_hsa_latency(args)

# python code_exp/bench_hsa_latency.py

# python code_exp/bench_hsa_latency.py --config "16:4:8:128:128,8:8:16:128:128,4:4:32:64:128,2:16:16:128:64,1:16:32:128:64"


# python code_exp/bench_hsa_latency.py --config "16:4:8:128:128,8:8:16:128:128,4:4:32:64:128,2:16:16:128:64,1:16:32:128:64" --with_baseline
