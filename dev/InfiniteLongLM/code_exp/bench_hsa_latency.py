"""
HSA Latency Benchmark
=====================
专门用于测试 HSA kernel 在不同 HQ:H 比例下的延迟表现。
[head-wise 版] 前后向共用同一个 block_M，只遵循 num_threads_fwd / num_threads_bwd
独立。--config / --candidates / --hsa_configs 采用 4 段新格式。
  - --config / --candidates: G:M:nt_fwd:nt_bwd
  - --hsa_configs:           seq_len:G:M:nt_fwd:nt_bwd

用法:
    python code_exp/bench_hsa_latency.py
    python code_exp/bench_hsa_latency.py --max_block_M 16 --overlap 0.6
    python code_exp/bench_hsa_latency.py --B 4 --SEQ_LEN 8192 --D 64 --S 16 --block_size 64

对于每个 HQ:H = G，block_M 从 max_block_M // G 开始翻倍遍历到 max_block_M:
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
import os
import traceback


# =========================================================
# 加载真实 indices 数据
# =========================================================
def load_real_indices(pt_path: str, B: int, layer_idx: int = None, device: str = "cuda"):
    """
    从收集的 .pt 文件中加载真实的 indices 和 weights。

    参数:
        pt_path: .pt 文件路径
        B: 需要的 batch size，从 samples 中取前 B 条拼成 batch
        layer_idx: 使用哪一层的数据（默认 None，自动选第一个 HSA 层）
        device: 目标设备

    返回:
        (block_indices, weights, actual_seq_len)
        block_indices: [B, L, H_kv, S], int32
        weights: [B, L, H_kv, S], bfloat16
        actual_seq_len: 实际的序列长度 L（包含 landmark token）
    """
    print(f"\n加载真实 indices: {pt_path}")
    saved = torch.load(pt_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    samples = saved["samples"]
    num_samples = len(samples)

    print(f"  配置: seq_len={config['seq_len']}, chunk_size={config['chunk_size']}, "
          f"hsa_topk={config['hsa_topk']}, H_kv={config['num_key_value_heads']}")
    print(f"  样本数: {num_samples}, 需要 B={B}")

    assert num_samples >= B, (
        f"样本数 ({num_samples}) 不足以组成 batch ({B})，"
        f"请增加收集时的 --num_samples")

    # 确定使用哪一层
    all_layer_idxs = set()
    for sample in samples:
        all_layer_idxs.update(sample["layers"].keys())
    all_layer_idxs = sorted(all_layer_idxs)

    if layer_idx is None:
        layer_idx = all_layer_idxs[0]
        print(f"  自动选择第一个 HSA 层: layer_idx={layer_idx}")
    else:
        assert layer_idx in all_layer_idxs, (
            f"layer_idx={layer_idx} 不在可用层列表中: {all_layer_idxs}")
        print(f"  使用指定层: layer_idx={layer_idx}")

    print(f"  可用 HSA 层: {all_layer_idxs}")

    # 取前 B 条样本的 indices 和 chunk_weights
    indices_list = []
    weights_list = []
    S = config["hsa_topk"]

    for i in range(B):
        layer_data = samples[i]["layers"][layer_idx]
        idx = layer_data["indices"]    # [1, L, H_kv, S]
        cw = layer_data["chunk_weights"]  # [1, L, H_kv, S+1] or [1, L, H_kv, S+2]

        # chunk_weights 可能多了 swa 权重列，只取前 S 列
        cw = cw[:, :, :, :S]

        indices_list.append(idx)
        weights_list.append(cw)

    # 拼成 batch: [B, L, H_kv, S]
    block_indices = torch.cat(indices_list, dim=0).to(dtype=torch.int32, device=device)
    weights = torch.cat(weights_list, dim=0).to(dtype=torch.bfloat16, device=device)

    actual_seq_len = block_indices.shape[1]
    H_kv = block_indices.shape[2]

    print(f"  加载完成: block_indices={list(block_indices.shape)}, "
          f"weights={list(weights.shape)}, actual_seq_len={actual_seq_len}")

    return block_indices, weights, actual_seq_len


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
    backend: str = "fa3",
):
    """
    测试 FlashAttention (flash_attn_func) 在给定配置下的 FWD / BWD / Total 延迟。
    作为 HSA kernel 的 baseline 对比。

    backend:
        "fa3" -> from flash_attn_interface import flash_attn_func  (FlashAttention 3)
        "fa2" -> from flash_attn            import flash_attn_func  (FlashAttention 2)

    返回 (fwd_ms, bwd_ms, total_ms)。
    """
    if backend == "fa3":
        from flash_attn_interface import flash_attn_func
    elif backend == "fa2":
        from flash_attn import flash_attn_func
    else:
        raise ValueError(f"Unknown FlashAttention backend: {backend}")

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
# FlashAttention 2 (tilelang) Baseline 延迟测试
# =========================================================
def bench_flash_attn_tilelang(
    B: int,
    SEQ_LEN: int,
    H: int,
    HQ: int,
    D: int,
    num_warmup: int = 50,
    num_iters: int = 100,
    device: str = "cuda",
):
    """
    测试 ops.example_gqa_bwd.attention（tilelang 实现的 FA2 风格 GQA）在给定配置下的
    FWD / BWD / Total 延迟。作为 HSA kernel 的 tilelang 版 baseline 对比。

    注意:
        - kernel 内部 dtype 硬编码为 float16，这里强制构造 fp16 输入（不影响其他 baseline）。
        - 仅支持 causal=True（和 FA2/FA3 baseline 对齐）。
        - BWD 使用 atomic_add 版本。

    返回 (fwd_ms, bwd_ms, total_ms)。
    """
    from ops.example_gqa_bwd import attention as tl_attention

    dtype = torch.float16
    causal = True
    groups = HQ // H

    # ---------- 构造输入张量 ----------
    # tilelang attention 期望的输入格式: (B, SEQ_LEN, H, D)，dtype=fp16
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
        O = tl_attention(Q_bench, K, V, causal, groups, True)  # use_atomic=True
        O.backward(grad_output)

    # ---------- FWD only ----------
    def fa_fwd():
        with torch.no_grad():
            tl_attention(Q_bench, K, V, causal, groups, True)

    fwd_ms = measure_time(fa_fwd)

    # ---------- FWD + BWD ----------
    def fa_fwd_bwd():
        Q_bench.grad = K.grad = V.grad = None
        O = tl_attention(Q_bench, K, V, causal, groups, True)
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
    weights=None,
):
    """
    Run a single FWD / BWD latency measurement of HSA_block_M_head (head-wise).
    The head-wise kernel uses one shared block_M for fwd / bwd. We keep the
    legacy formal parameters ``block_M_fwd`` / ``block_M_bwd`` only for
    backward source compatibility; if the two values differ a warning is
    printed and ``block_M_fwd`` wins. ``num_threads_fwd`` / ``num_threads_bwd``
    remain independent. Pass a pre-built ``block_indices`` to skip rebuilding.
    Returns (fwd_ms, bwd_ms, total_ms).
    """
    from ops.hsa_fwd_bwd_head import HSA_block_M_head

    if block_M_bwd != block_M_fwd:
        # head-wise kernel doesn't separate fwd/bwd block_M; fall back to fwd's value.
        print(f"  [bench_single_config] head-wise kernel uses a single block_M; "
              f"got block_M_fwd={block_M_fwd}, block_M_bwd={block_M_bwd} -> use {block_M_fwd}")
    block_M = block_M_fwd

    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    # ---------- 构造 block_indices（如果未传入）----------
    if block_indices is None:
        block_indices = build_block_indices_union_overlap(
            B=B, SEQ_LEN=SEQ_LEN, H=H, S=S,
            block_size=block_size,
            overlap_ratio=overlap_ratio,
            block_M=block_M,
            device=device,
        )

    # ---------- 构造输入张量 ----------
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)

    # head-wise kernel expects per-q-head weights of shape (B, L, HQ, S).
    # If real weights (per-h_kv) are provided we broadcast them to HQ via
    # repeat_interleave; otherwise we sample fresh logits at HQ resolution.
    if weights is not None:
        if weights.shape[2] == HQ:
            W_src = weights
        elif weights.shape[2] == H and HQ % H == 0:
            W_src = weights.repeat_interleave(HQ // H, dim=2).contiguous()
        else:
            raise ValueError(
                f"weights head dim ({weights.shape[2]}) is neither HQ ({HQ}) nor H ({H})"
            )
        W = W_src.detach().to(dtype).requires_grad_(True)
    else:
        logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
        # block_indices is (B, L, H, S); broadcast valid_mask to HQ.
        valid_mask_h = (block_indices != -1)
        if HQ != H:
            valid_mask_hq = valid_mask_h.repeat_interleave(HQ // H, dim=2)
        else:
            valid_mask_hq = valid_mask_h
        logits = logits.masked_fill(~valid_mask_hq, float('-inf'))
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
    # NOTE: zero out grads each iteration so warmup does NOT keep accumulating
    # gradient tensors (which would otherwise grow throughout warmup and bias
    # later memory / cache state).
    for _ in range(num_warmup):
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_head(
            Q_tile, K, V, W, block_indices,
            block_size=block_size, sm_scale=scale,
            block_M=block_M,
            num_threads_fwd=num_threads_fwd, num_threads_bwd=num_threads_bwd,
            mask_last_token=mask_last_token,
            is_training=True,
        )
        O.backward(grad_output_tile)

    # ---------- FWD only ----------
    # NOTE: keep is_training=True (same code path as FWD+BWD) and do NOT wrap
    # with torch.no_grad(). The is_training=False / no_grad inference path can
    # exercise a different kernel branch that is much slower than the training
    # forward, which would inflate FWD time and produce a misleading
    # BWD = total - fwd. We only skip the backward call here; the forward
    # kernel itself is identical to the FWD+BWD case.
    def tile_fwd():
        HSA_block_M_head(
            Q_tile, K, V, W, block_indices,
            block_size=block_size, sm_scale=scale,
            block_M=block_M,
            num_threads_fwd=num_threads_fwd, num_threads_bwd=num_threads_bwd,
            mask_last_token=mask_last_token,
            is_training=True,
        )

    fwd_ms = measure_time(tile_fwd)

    # ---------- FWD + BWD ----------
    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_head(
            Q_tile, K, V, W, block_indices,
            block_size=block_size, sm_scale=scale,
            block_M=block_M,
            num_threads_fwd=num_threads_fwd, num_threads_bwd=num_threads_bwd,
            mask_last_token=mask_last_token,
            is_training=True,
        )
        O.backward(grad_output_tile)

    total_ms = measure_time(tile_fwd_bwd)
    bwd_ms = total_ms - fwd_ms

    return fwd_ms, bwd_ms, total_ms


# =========================================================
# HSA single baseline (只支持 HQ:H=16:1，其他比例通过 HQ padding 到 HQ_padded=16*H 实现)
# =========================================================
def bench_single_config_hsa_single(
    B: int,
    SEQ_LEN: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    overlap_ratio: float,
    num_threads_fwd: int = None,
    num_threads_bwd: int = None,
    mask_last_token: bool = True,
    num_warmup: int = 50,
    num_iters: int = 100,
    dtype=torch.bfloat16,
    device: str = "cuda",
    block_indices=None,
    weights=None,
):
    """
    Run FWD/BWD latency benchmark for the head-wise HSA_single baseline.
    HSA_single processes one query token per kernel block. The kernel parallelizes
    the G (= HQ // head_kv) group inside a thread block; tilelang requires every
    GEMM M dimension to be a multiple of 16, so when G < 16 we must pad the HQ
    dim of Q / W / grad_output (NOT K, V, block_indices) so that G_eff >= 16.

    Shape conventions (head-wise, before padding):
      Q              : (B, L, HQ, D)
      K, V           : (B, L, H,  D)
      W              : (B, L, HQ, S)   per-q-head weights
      block_indices  : (B, L, H,  S)   per-h_kv block indices

    After HQ padding to HQ_padded = pad_ratio * HQ where pad_ratio = ceil(16 / G):
      Q_s            : (B, L, HQ_padded, D)   real heads then zero-padded heads
      W_s            : (B, L, HQ_padded, S)   real weights then zero-padded weights
      grad_output_s  : (B, L, HQ_padded, D)   real grads then zero-padded grads

    Returns (fwd_ms, bwd_ms, total_ms).
    """
    from ops.hsa_fwd_bwd_single_tilelang import HSA_single_two_phase as HSA_single

    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    # ---------- Build block_indices if not provided ----------
    if block_indices is None:
        block_indices = build_block_indices_union_overlap(
            B=B, SEQ_LEN=SEQ_LEN, H=H, S=S,
            block_size=block_size,
            overlap_ratio=overlap_ratio,
            block_M=1,  # single kernel processes one token at a time
            device=device,
        )

    # ---------- Build input tensors ----------
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)

    # head-wise W: (B, L, HQ, S). Accept HQ-shaped weights directly; if the caller passes
    # group-wise weights (B, L, H, S), broadcast them to HQ via repeat_interleave.
    if weights is not None:
        if weights.shape[2] == HQ:
            W_src = weights
        elif weights.shape[2] == H and HQ % H == 0:
            W_src = weights.repeat_interleave(HQ // H, dim=2).contiguous()
        else:
            raise ValueError(
                f"weights head dim ({weights.shape[2]}) is neither HQ ({HQ}) nor H ({H})"
            )
        W = W_src.detach().to(dtype).requires_grad_(True)
    else:
        logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
        # block_indices is (B, L, H, S); broadcast valid_mask to HQ.
        valid_mask_h = (block_indices != -1)
        if HQ != H:
            valid_mask_hq = valid_mask_h.repeat_interleave(HQ // H, dim=2)
        else:
            valid_mask_hq = valid_mask_h
        logits = logits.masked_fill(~valid_mask_hq, float('-inf'))
        W = F.softmax(logits, dim=-1)
        W = torch.nan_to_num(W, nan=0.0).detach().requires_grad_(True)

    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    # ---------- HQ padding: ensure G_eff = G * pad_ratio >= 16 ----------
    # tilelang requires GEMM M-dim % 16 == 0; the single kernel parallelizes G inside
    # the thread block, so when G < 16 we pad the HQ dim of Q / W / grad_output.
    # K, V, block_indices are NOT padded (they live on the head_kv axis).
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * HQ
    need_padding = (HQ_padded > HQ)

    if need_padding:
        pad_heads = HQ_padded - HQ
        Q_s = torch.cat(
            [Q.detach(), torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)],
            dim=2,
        ).clone().requires_grad_(True)
        # head-wise: pad W along HQ dim with zeros so the padded q-heads contribute nothing.
        W_s = torch.cat(
            [W.detach(), torch.zeros(B, SEQ_LEN, pad_heads, S, dtype=dtype, device=device)],
            dim=2,
        ).clone().requires_grad_(True)
        grad_output_s = torch.cat(
            [grad_output, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)],
            dim=2,
        )
    else:
        Q_s = Q.detach().clone().requires_grad_(True)
        W_s = W
        grad_output_s = grad_output

    def measure_time(func):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters

    # Warmup
    for _ in range(num_warmup):
        O = HSA_single(
            Q_s, K, V, W_s, block_indices,
            block_size=block_size, sm_scale=scale,
            mask_last_token=mask_last_token,
            num_threads_fwd=num_threads_fwd,
            num_threads_bwd=num_threads_bwd,
        )
        O.backward(grad_output_s)

    def fwd_only():
        with torch.no_grad():
            HSA_single(
                Q_s, K, V, W_s, block_indices,
                block_size=block_size, sm_scale=scale,
                mask_last_token=mask_last_token,
                num_threads_fwd=num_threads_fwd,
                num_threads_bwd=num_threads_bwd,
            )

    fwd_ms = measure_time(fwd_only)

    def fwd_bwd():
        Q_s.grad = K.grad = V.grad = W_s.grad = None
        O = HSA_single(
            Q_s, K, V, W_s, block_indices,
            block_size=block_size, sm_scale=scale,
            mask_last_token=mask_last_token,
            num_threads_fwd=num_threads_fwd,
            num_threads_bwd=num_threads_bwd,
        )
        O.backward(grad_output_s)

    total_ms = measure_time(fwd_bwd)
    bwd_ms = total_ms - fwd_ms

    return fwd_ms, bwd_ms, total_ms


# =========================================================
# HSA Dense end-to-end latency benchmark
# =========================================================
def bench_dense_interface(
    B: int,
    SEQ_LEN: int,
    H: int,
    HQ: int,
    D: int,
    block_size: int,
    window_size: int,
    dense_block_M: int = 64,
    dense_num_threads: int = 128,
    mask_last_token: bool = True,
    num_warmup: int = 50,
    num_iters: int = 100,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    """
    Benchmark the end-to-end HSA-Dense path (lmk dense scoring + dense HSA attention)
    via ``HSA_dense_interface``. The dense path does NOT consume real indices; it
    computes attention weights directly from landmarks. ``SEQ_LEN`` here should be
    the lmk-inserted length (typically L_with_lmk = raw + raw // block_size), and
    must be divisible by ``block_size``.

    Returns (fwd_ms, bwd_ms, total_ms).
    """
    from ops.hsa_fwd_bwd_head_dense import HSA_dense_interface

    assert SEQ_LEN % block_size == 0, (
        f"dense path requires SEQ_LEN ({SEQ_LEN}) divisible by block_size ({block_size})")
    S_chunks = SEQ_LEN // block_size
    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    # Build inputs (random data is sufficient: dense path does not use real indices).
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    # lmk: per-h_kv chunk landmark, shape [B, S_chunks, H, D]
    lmk = torch.randn((B, S_chunks, H, D), dtype=dtype, device=device, requires_grad=True)
    # lse_swa: [B, L, HQ] (already in HQ form, accepted directly by the interface)
    lse_swa = torch.randn((B, SEQ_LEN, HQ), dtype=torch.float32, device=device)

    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    def _call():
        out, _, _ = HSA_dense_interface(
            q=Q, k=K, v=V, lmk=lmk, lse_swa=lse_swa,
            block_size=block_size, window_size=window_size,
            mask_last_token=mask_last_token,
            sm_scale=scale,
            # Do NOT pass G: when G is given, the interface requires lmk to be HQ-headed.
            # Our lmk is per-h_kv shape (B, S_chunks, H, D), so let G be auto-inferred.
        )
        return out

    def measure_time(fn):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters

    # Warmup (fwd+bwd to JIT-compile both kernels and amortize autotune cost).
    for _ in range(num_warmup):
        Q.grad = K.grad = V.grad = lmk.grad = None
        out = _call()
        out.backward(grad_output)

    # FWD only.
    def _fwd_only():
        with torch.no_grad():
            _call()
    fwd_ms = measure_time(_fwd_only)

    # FWD + BWD.
    def _fwd_bwd():
        Q.grad = K.grad = V.grad = lmk.grad = None
        out = _call()
        out.backward(grad_output)
    total_ms = measure_time(_fwd_bwd)
    bwd_ms = total_ms - fwd_ms
    return fwd_ms, bwd_ms, total_ms

# =========================================================
# FSA Select Attention baseline (不含 topk 选择，只做 chunk-wise selected attention)
# =========================================================
def _hsa_indices_to_fsa_topk_idx(block_indices: torch.Tensor):
    """
    将 HSA 的 [B, L, H_kv, S] 布局的 block_indices 转换为
    FSA 需要的 [H_kv, B*L, S] 布局的 topk_idx（varlen 打平 batch 维到 L 维上）。
    每个 query token 选 S 个 kv block，语义一致，-1 填充保留。
    """
    B, L, H_kv, S = block_indices.shape
    # [H_kv, B, L, S] → [H_kv, B*L, S]
    t = block_indices.permute(2, 0, 1, 3).contiguous().view(H_kv, B * L, S).contiguous()
    return t.to(torch.int32)


def bench_fsa_select_kernel(
    B: int,
    SEQ_LEN: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    block_indices: torch.Tensor,  # [B, L, H_kv, S]
    num_warmup: int = 50,
    num_iters: int = 100,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    """
    测试 FSA select attention kernel 的 FWD / BWD / Total 延迟。
    输入 block_indices 与 HSA 一致（[B, L, H_kv, S]），内部转换为 FSA 所需的
    topk_idx 布局 [H_kv, B*L, S]，以及 varlen 所需的 cu_seqlens。
    仅测试 select attention 部分（不含 topk 选择、compressed attention、sliding window）。
    """
    from fsa.ops.FSA_topk_sparse_attention import FSA_topk_sparse_attention

    scale = 1.0 / math.sqrt(D)

    # ---------- 构造输入张量（varlen: [total_len, H, D]）----------
    total_len = B * SEQ_LEN
    Q = torch.randn((total_len, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((total_len, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((total_len, H, D), dtype=dtype, device=device, requires_grad=True)

    # cu_seqlens: [0, L, 2L, ..., B*L]
    cu_seqlens = torch.arange(0, (B + 1) * SEQ_LEN, SEQ_LEN, dtype=torch.int32, device=device)

    # topk_idx: [H_kv, total_len, S]
    topk_idx = _hsa_indices_to_fsa_topk_idx(block_indices)

    grad_output = torch.randn((total_len, HQ, D), dtype=dtype, device=device)

    Q_f = Q.detach().clone().requires_grad_(True)
    K_f = K.detach().clone().requires_grad_(True)
    V_f = V.detach().clone().requires_grad_(True)

    def measure_time(func):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters

    # Warmup
    for _ in range(num_warmup):
        O = FSA_topk_sparse_attention(
            Q_f, K_f, V_f, topk_idx,
            block_size, cu_seqlens, scale,
        )
        O.backward(grad_output)

    def fwd_only():
        with torch.no_grad():
            FSA_topk_sparse_attention(
                Q_f, K_f, V_f, topk_idx,
                block_size, cu_seqlens, scale,
            )

    fwd_ms = measure_time(fwd_only)

    def fwd_bwd():
        Q_f.grad = K_f.grad = V_f.grad = None
        O = FSA_topk_sparse_attention(
            Q_f, K_f, V_f, topk_idx,
            block_size, cu_seqlens, scale,
        )
        O.backward(grad_output)

    total_ms = measure_time(fwd_bwd)
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
    Sweep HSA head-wise kernel latency across HQ:H ratios.
    Search dimensions:
      - HQ:H ratio: 16:1, 8:1, 4:1, 2:1, 1:1
      - block_M (single, shared by fwd / bwd; doubled from 16//G up to max_block_M)
      - num_threads_fwd x num_threads_bwd (independent)
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

    # 加载真实 indices（如果指定）
    real_block_indices = None
    real_weights = None
    if args.real_indices:
        real_block_indices, real_weights, actual_seq_len = load_real_indices(
            args.real_indices, B, layer_idx=args.layer_idx)
        SEQ_LEN = actual_seq_len
        print(f"  使用真实 indices，SEQ_LEN 已更新为 {SEQ_LEN}（包含 landmark token）")

    print("=" * 120)
    print(f"HSA Latency Benchmark (head-wise; single block_M, independent num_threads_fwd/bwd)")
    print(f"Config: B={B}, L={SEQ_LEN}, D={D}, S={S}, block_size={block_size}, "
          f"max_block_M={max_block_M}, overlap={overlap_ratio}, base_H={base_H}")
    print(f"mask_last_token={mask_last_token}")
    print(f"block_M sweep: for G=HQ//H, M doubled from 16//G up to {max_block_M}")
    print(f"num_threads sweep: fwd/bwd each over {NUM_THREADS_LIST}")
    print("=" * 120)
    print()

    # (ratio, H, HQ, G, M, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)
    results = []
    # 每个 ratio 的最优配置: {ratio: {"best_fwd": (...), "best_bwd": (...), "best_total": (...)}}
    best_configs = {}

    for ratio in ratios:
        G = ratio
        if hasattr(args, 'fixed_HQ') and args.fixed_HQ:
            HQ = args.fixed_HQ
            assert HQ % G == 0, f"fixed_HQ={HQ} 不是 G={G} 的整数倍"
            H = HQ // G
        else:
            H = base_H
            HQ = H * G

        block_M_list = _get_block_M_list(G, max_block_M)

        print(f"\n{'='*100}")
        print(f">>> HQ:H = {ratio}:1  (H={H}, HQ={HQ}, G={G})")
        print(f">>> block_M 遍历列表: {block_M_list}")
        print(f">>> num_threads 遍历列表: {NUM_THREADS_LIST}")
        print(f"{'='*100}")

        # 预构造 block_indices（用最大的 block_M 来构造，所有子配置共享）
        if real_block_indices is not None:
            # 真实 indices 的 H_kv 可能与当前 H 不一致（--fixed_HQ 模式下 H = HQ/G 会随 G 变化）
            real_h_kv = real_block_indices.shape[2]
            if H != real_h_kv:
                if H > real_h_kv:
                    assert H % real_h_kv == 0, (
                        f"H={H} 不是 real_h_kv={real_h_kv} 的整数倍")
                    repeat_factor = H // real_h_kv
                    block_indices = real_block_indices.repeat_interleave(
                        repeat_factor, dim=2).contiguous()
                    cur_weights = (real_weights.repeat_interleave(
                        repeat_factor, dim=2).contiguous()
                        if real_weights is not None else None)
                    print(f"  indices H_kv 适配: {real_h_kv} → {H} (repeat x{repeat_factor})")
                else:
                    block_indices = real_block_indices[:, :, :H, :].contiguous()
                    cur_weights = (real_weights[:, :, :H, :].contiguous()
                                   if real_weights is not None else None)
                    print(f"  indices H_kv 适配: {real_h_kv} → {H} (slice)")
            else:
                block_indices = real_block_indices
                cur_weights = real_weights
        else:
            max_M_in_list = max(block_M_list)
            block_indices = build_block_indices_union_overlap(
                B=B, SEQ_LEN=SEQ_LEN, H=H, S=S,
                block_size=block_size,
                overlap_ratio=overlap_ratio,
                block_M=max_M_in_list,
                device="cuda",
            )
            cur_weights = None

        for block_M in block_M_list:
            for nt_fwd in NUM_THREADS_LIST:
                for nt_bwd in NUM_THREADS_LIST:
                    print(f"\n  --- M={block_M}, nt_fwd={nt_fwd}, nt_bwd={nt_bwd} ---")
                    try:
                        fwd_ms, bwd_ms, total_ms = bench_single_config(
                            B=B,
                            SEQ_LEN=SEQ_LEN,
                            H=H,
                            HQ=HQ,
                            D=D,
                            S=S,
                            block_size=block_size,
                            block_M_fwd=block_M,
                            block_M_bwd=block_M,
                            overlap_ratio=overlap_ratio,
                            num_threads_fwd=nt_fwd,
                            num_threads_bwd=nt_bwd,
                            mask_last_token=mask_last_token,
                            block_indices=block_indices,
                            weights=cur_weights,
                        )
                    except Exception as e:
                        print(f"  ⚠ SKIPPED (M={block_M}, "
                              f"nt_fwd={nt_fwd}, nt_bwd={nt_bwd}): {type(e).__name__}: {e}")
                        continue

                    results.append((ratio, H, HQ, G, block_M,
                                    nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms))
                    print(f"  HQ:H={ratio}:1 | G={G} | M={block_M} | "
                          f"nt_fwd={nt_fwd} nt_bwd={nt_bwd} | "
                          f"FWD={fwd_ms:.3f}ms | BWD={bwd_ms:.3f}ms | Total={total_ms:.3f}ms")

        # --- 当前 ratio 遍历完毕，选出最优配置 ---
        ratio_results = [r for r in results if r[0] == ratio]
        if not ratio_results:
            print(f"\n  ⚠ HQ:H={ratio}:1 所有参数组合均失败，无最优配置")
            continue

        # tuple layout: (ratio, H, HQ, G, M, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)
        # indices:        0      1  2   3  4   5       6      7        8       9
        best_fwd_entry = min(ratio_results, key=lambda x: x[7])   # fwd_ms
        best_bwd_entry = min(ratio_results, key=lambda x: x[8])   # bwd_ms
        best_total_entry = min(ratio_results, key=lambda x: x[9]) # total_ms
        best_configs[ratio] = {
            "best_fwd": best_fwd_entry,
            "best_bwd": best_bwd_entry,
            "best_total": best_total_entry,
        }

        print(f"\n  {'*'*80}")
        print(f"  ★ HQ:H={ratio}:1 best configs:")
        _r = best_fwd_entry
        print(f"    Best FWD:   M={_r[4]}, nt_fwd={_r[5]}  "
              f"→ FWD={_r[7]:.3f}ms")
        _r = best_bwd_entry
        print(f"    Best BWD:   M={_r[4]}, nt_bwd={_r[6]}  "
              f"→ BWD={_r[8]:.3f}ms")
        _r = best_total_entry
        print(f"    Best Total: M={_r[4]}, nt_fwd={_r[5]}, nt_bwd={_r[6]}  "
              f"→ Total={_r[9]:.3f}ms (FWD={_r[7]:.3f}ms + BWD={_r[8]:.3f}ms)")
        print(f"  {'*'*80}")

    # 汇总表格
    print()
    print("=" * 140)
    print("Summary Table")
    print("=" * 140)
    header = (f"{'HQ:H':>8} | {'G':>4} | {'M':>5} | "
              f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
              f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12}")
    print(header)
    print("-" * len(header))
    prev_ratio = None
    for ratio, H, HQ, G, M, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms in results:
        if prev_ratio is not None and ratio != prev_ratio:
            print("-" * len(header))  # divider between G groups
        prev_ratio = ratio
        print(f"{'%d:1' % ratio:>8} | {G:>4} | {M:>5} | "
              f"{nt_fwd:>6} | {nt_bwd:>6} | "
              f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f}")
    print("=" * 140)

    # ========== 最优配置汇总（仅展示 Best Total 对应的 FWD/BWD/Total 拆分）==========
    print()
    print("=" * 140)
    print("Best Config per HQ:H Ratio (Best Total with FWD/BWD breakdown)")
    print("=" * 140)
    best_header = (f"{'HQ:H':>8} | {'G':>4} | {'M':>5} | "
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
        print(f"{'%d:1' % ratio:>8} | {G:>4} | {_r[4]:>5} | "
              f"{_r[5]:>6} | {_r[6]:>6} | "
              f"{_r[7]:>10.3f} | {_r[8]:>10.3f} | {_r[9]:>12.3f}")
    print("=" * len(best_header))


# =========================================================
# 多序列长度遍历搜索最优参数
# =========================================================
SEQ_LEN_LIST = [8192]

def bench_sweep_seq_len(args):
    """
    Sweep multiple sequence lengths (8k, 16k, 32k, 64k, 128k).
    For each (seq_len, HQ:H) search the best (M, nt_fwd, nt_bwd) configuration.
    Print one summary table at the end.
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
    print(f"block_M sweep: for G=HQ//H, M doubled from 16//G up to {max_block_M}")
    print(f"num_threads sweep: fwd/bwd each over {NUM_THREADS_LIST}")
    print("=" * 140)
    print()

    # sweep_best[(seq_len, ratio)] = (M, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)
    sweep_best = {}

    for seq_len in seq_len_list:
        print(f"\n{'#'*140}")
        print(f"### SEQ_LEN = {seq_len} ({seq_len//1024}k)")
        print(f"{'#'*140}")

        # 如果指定了 --real_indices_dir，预加载该 seq_len 的真实 indices（仅加载一次，ratio 之间复用）
        cur_seq_len = seq_len
        real_block_indices_seq = None
        real_weights_seq = None
        if getattr(args, 'real_indices_dir', None):
            import os as _os
            pt_path = _os.path.join(args.real_indices_dir, f"indices_{seq_len}.pt")
            if _os.path.exists(pt_path):
                real_block_indices_seq, real_weights_seq, actual_seq_len = load_real_indices(
                    pt_path, B, layer_idx=args.layer_idx)
                cur_seq_len = actual_seq_len
                print(f"  使用真实 indices: {pt_path}，SEQ_LEN 已更新为 {cur_seq_len}")
            else:
                print(f"  ⚠ 未找到 {pt_path}，该长度将使用随机构造的 indices")

        for ratio in ratios:
            G = ratio
            if getattr(args, 'fixed_HQ', None):
                HQ = args.fixed_HQ
                assert HQ % G == 0, f"fixed_HQ={HQ} 不是 G={G} 的整数倍"
                H = HQ // G
            else:
                H = base_H
                HQ = H * G

            block_M_list = _get_block_M_list(G, max_block_M)

            print(f"\n  {'='*100}")
            print(f"  >>> SEQ_LEN={cur_seq_len}, HQ:H = {ratio}:1  (H={H}, HQ={HQ}, G={G})")
            print(f"  >>> block_M 遍历列表: {block_M_list}")
            print(f"  {'='*100}")

            # 预构造 block_indices（用最大的 block_M 来构造），或使用真实 indices
            if real_block_indices_seq is not None:
                real_h_kv = real_block_indices_seq.shape[2]
                if H != real_h_kv:
                    if H > real_h_kv:
                        assert H % real_h_kv == 0, (
                            f"H={H} 不是 real_h_kv={real_h_kv} 的整数倍")
                        rf = H // real_h_kv
                        block_indices = real_block_indices_seq.repeat_interleave(
                            rf, dim=2).contiguous()
                        print(f"    indices H_kv 适配: {real_h_kv} → {H} (repeat x{rf})")
                    else:
                        block_indices = real_block_indices_seq[:, :, :H, :].contiguous()
                        print(f"    indices H_kv 适配: {real_h_kv} → {H} (slice)")
                else:
                    block_indices = real_block_indices_seq
            else:
                max_M_in_list = max(block_M_list)
                block_indices = build_block_indices_union_overlap(
                    B=B, SEQ_LEN=cur_seq_len, H=H, S=S,
                    block_size=block_size,
                    overlap_ratio=overlap_ratio,
                    block_M=max_M_in_list,
                    device="cuda",
                )

            best_total_ms = float('inf')
            best_entry = None

            for block_M in block_M_list:
                for nt_fwd in NUM_THREADS_LIST:
                    for nt_bwd in NUM_THREADS_LIST:
                        try:
                            fwd_ms, bwd_ms, total_ms = bench_single_config(
                                B=B, SEQ_LEN=cur_seq_len, H=H, HQ=HQ, D=D, S=S,
                                block_size=block_size,
                                block_M_fwd=block_M, block_M_bwd=block_M,
                                overlap_ratio=overlap_ratio,
                                num_threads_fwd=nt_fwd, num_threads_bwd=nt_bwd,
                                mask_last_token=mask_last_token,
                                block_indices=block_indices,
                            )
                        except Exception as e:
                            print(f"    ⚠ SKIPPED (M={block_M}, "
                                  f"nt_fwd={nt_fwd}, nt_bwd={nt_bwd}): {type(e).__name__}: {e}")
                            continue

                        print(f"    M={block_M}, "
                              f"nt_fwd={nt_fwd}, nt_bwd={nt_bwd} | "
                              f"FWD={fwd_ms:.3f}ms BWD={bwd_ms:.3f}ms Total={total_ms:.3f}ms")

                        if total_ms < best_total_ms:
                            best_total_ms = total_ms
                            best_entry = (block_M, nt_fwd, nt_bwd,
                                          fwd_ms, bwd_ms, total_ms)

            if best_entry is not None:
                sweep_best[(seq_len, ratio)] = best_entry
                print(f"\n  ★ Best for SEQ_LEN={seq_len}, HQ:H={ratio}:1: "
                      f"M={best_entry[0]}, "
                      f"nt_fwd={best_entry[1]}, nt_bwd={best_entry[2]} → "
                      f"Total={best_entry[5]:.3f}ms (FWD={best_entry[3]:.3f}ms + BWD={best_entry[4]:.3f}ms)")
            else:
                print(f"\n  ⚠ SEQ_LEN={seq_len}, HQ:H={ratio}:1 all configs failed")

            # 释放 block_indices 显存
            del block_indices
            import gc; gc.collect()
            import torch as _torch; _torch.cuda.empty_cache()

    # ========== 汇总表格 ==========
    print()
    print("=" * 160)
    print("Sweep SEQ_LEN - Best Config Summary")
    print("=" * 160)
    sweep_header = (f"{'SEQ_LEN':>10} | {'HQ:H':>8} | {'G':>4} | {'M':>5} | "
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
            print(f"{seq_len:>10} | {'%d:1' % ratio:>8} | {G:>4} | {entry[0]:>5} | "
                  f"{entry[1]:>6} | {entry[2]:>6} | "
                  f"{entry[3]:>10.3f} | {entry[4]:>10.3f} | {entry[5]:>12.3f}")
    print("=" * len(sweep_header))

    # ========== Print best --config strings per SEQ_LEN (4-segment new format) ==========
    print()
    print("=" * 160)
    print("Best --config strings per SEQ_LEN (copy-paste ready, format: G:M:nt_fwd:nt_bwd)")
    print("=" * 160)
    for seq_len in seq_len_list:
        config_parts = []
        for ratio in ratios:
            key = (seq_len, ratio)
            if key in sweep_best:
                entry = sweep_best[key]
                # entry: (M, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)
                config_parts.append(f"{ratio}:{entry[0]}:{entry[1]}:{entry[2]}")
        if config_parts:
            config_str = ",".join(config_parts)
            print(f"  SEQ_LEN={seq_len:>6} ({seq_len//1024:>3}k): --config \"{config_str}\"")
    print("=" * 160)


# =========================================================
# 候选配置搜索模式（缩小范围搜索）
# =========================================================
def bench_candidates(args):
    """
    For each G run a few user-specified candidate configs and pick the best.
    Faster than full sweep; use it to refine around known good configs.

    Pass via --candidates with format:
        "G:M:nt_fwd:nt_bwd|M:nt_fwd:nt_bwd|...;G:..."
    Different G are separated by ';'; multiple candidates of the same G are
    separated by '|'. The first item of each group must start with G; later
    items omit G.

    Example:
        "16:4:128:128;8:8:128:128;4:4:64:64|4:64:128|4:64:128;"
        "2:8:64:64|16:128:64;1:16:128:64"
    """
    B = args.B
    SEQ_LEN = args.SEQ_LEN
    D = args.D
    S = args.S
    block_size = args.block_size
    overlap_ratio = args.overlap
    mask_last_token = args.mask_last_token
    base_H = args.base_H

    # 加载真实 indices（如果指定）
    real_block_indices = None
    real_weights = None
    if args.real_indices:
        real_block_indices, real_weights, actual_seq_len = load_real_indices(
            args.real_indices, B, layer_idx=args.layer_idx)
        SEQ_LEN = actual_seq_len
        print(f"  使用真实 indices，SEQ_LEN 已更新为 {SEQ_LEN}（包含 landmark token）")

    # Parse candidates string.
    # New 4-segment format: "G:M:nt_fwd:nt_bwd|M:nt_fwd:nt_bwd;G:..."
    candidate_map = {}  # {G: [(M, nt_fwd, nt_bwd), ...]}
    for group in args.candidates.split(";"):
        group = group.strip()
        if not group:
            continue
        alternatives = group.split("|")
        G = None
        configs_for_G = []
        for i, alt in enumerate(alternatives):
            alt = alt.strip()
            parts = alt.split(":")
            if i == 0:
                # First item must start with G: G:M:nt_fwd:nt_bwd
                assert len(parts) == 4, (
                    f"first candidate needs 4 fields (G:M:nt_fwd:nt_bwd), got: {alt}")
                G = int(parts[0])
                configs_for_G.append((
                    int(parts[1]), int(parts[2]), int(parts[3])))
            else:
                # Later candidates omit G: M:nt_fwd:nt_bwd
                assert len(parts) == 3, (
                    f"follow-up candidate needs 3 fields (M:nt_fwd:nt_bwd), got: {alt}")
                configs_for_G.append((
                    int(parts[0]), int(parts[1]), int(parts[2])))
        if G is not None:
            candidate_map[G] = configs_for_G

    print("=" * 120)
    print(f"HSA Latency Benchmark (Candidates Search Mode)")
    print(f"Config: B={B}, L={SEQ_LEN}, D={D}, S={S}, block_size={block_size}, "
          f"overlap={overlap_ratio}, base_H={base_H}")
    if args.real_indices:
        print(f"Real indices: {args.real_indices} (layer_idx={args.layer_idx})")
    print(f"mask_last_token={mask_last_token}")
    print("=" * 120)
    for G in sorted(candidate_map.keys(), reverse=True):
        print(f"  G={G:>2} ({G}:1): {len(candidate_map[G])} candidates")
        for j, (m, nf, nb) in enumerate(candidate_map[G]):
            print(f"    [{j+1}] M={m}, nt_fwd={nf}, nt_bwd={nb}")
    print("=" * 120)
    print()

    # Iterate G from large to small.
    # all_results layout: (ratio, G, M, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)
    all_results = []
    # best_per_G layout: (M, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)
    best_per_G = {}

    for G in sorted(candidate_map.keys(), reverse=True):
        if hasattr(args, 'fixed_HQ') and args.fixed_HQ:
            HQ = args.fixed_HQ
            H = HQ // G
        else:
            H = base_H
            HQ = H * G
        ratio = G
        configs_for_G = candidate_map[G]

        print(f"\n{'='*100}")
        print(f">>> HQ:H = {ratio}:1  (H={H}, HQ={HQ}, G={G}), {len(configs_for_G)} 套候选")
        print(f"{'='*100}")

        # 预构造 block_indices（如果没有真实 indices，用所有候选中最大的 M 来构造）
        if real_block_indices is not None:
            # 真实 indices 的 H_kv 可能与当前 H 不一致（--fixed_HQ 模式下 H = HQ/G 会随 G 变化）
            real_h_kv = real_block_indices.shape[2]
            if H != real_h_kv:
                if H > real_h_kv:
                    assert H % real_h_kv == 0, (
                        f"H={H} 不是 real_h_kv={real_h_kv} 的整数倍")
                    repeat_factor = H // real_h_kv
                    block_indices = real_block_indices.repeat_interleave(
                        repeat_factor, dim=2).contiguous()
                    cur_weights = (real_weights.repeat_interleave(
                        repeat_factor, dim=2).contiguous()
                        if real_weights is not None else None)
                    print(f"  indices H_kv 适配: {real_h_kv} → {H} (repeat x{repeat_factor})")
                else:
                    block_indices = real_block_indices[:, :, :H, :].contiguous()
                    cur_weights = (real_weights[:, :, :H, :].contiguous()
                                   if real_weights is not None else None)
                    print(f"  indices H_kv 适配: {real_h_kv} → {H} (slice)")
            else:
                block_indices = real_block_indices
                cur_weights = real_weights
        else:
            max_M = max(m for m, _, _ in configs_for_G)
            block_indices = build_block_indices_union_overlap(
                B=B, SEQ_LEN=SEQ_LEN, H=H, S=S,
                block_size=block_size,
                overlap_ratio=overlap_ratio,
                block_M=max_M,
                device="cuda",
            )
            cur_weights = None

        best_total_ms = float('inf')
        best_entry = None

        for j, (M_block, nt_fwd, nt_bwd) in enumerate(configs_for_G):
            print(f"\n  [{j+1}/{len(configs_for_G)}] M={M_block}, "
                  f"nt_fwd={nt_fwd}, nt_bwd={nt_bwd}")
            try:
                fwd_ms, bwd_ms, total_ms = bench_single_config(
                    B=B, SEQ_LEN=SEQ_LEN, H=H, HQ=HQ, D=D, S=S,
                    block_size=block_size,
                    block_M_fwd=M_block, block_M_bwd=M_block,
                    overlap_ratio=overlap_ratio,
                    num_threads_fwd=nt_fwd, num_threads_bwd=nt_bwd,
                    mask_last_token=mask_last_token,
                    block_indices=block_indices,
                    weights=cur_weights,
                )
            except Exception as e:
                print(f"    ⚠ FAILED: {type(e).__name__}: {e}")
                continue

            all_results.append((ratio, G, M_block, nt_fwd, nt_bwd,
                                fwd_ms, bwd_ms, total_ms))
            print(f"    ✓ FWD={fwd_ms:.3f}ms | BWD={bwd_ms:.3f}ms | Total={total_ms:.3f}ms")

            if total_ms < best_total_ms:
                best_total_ms = total_ms
                best_entry = (M_block, nt_fwd, nt_bwd,
                              fwd_ms, bwd_ms, total_ms)

        if best_entry is not None:
            best_per_G[G] = best_entry
            print(f"\n  ★ G={G} best: M={best_entry[0]}, "
                  f"nt_fwd={best_entry[1]}, nt_bwd={best_entry[2]} → "
                  f"Total={best_entry[5]:.3f}ms (FWD={best_entry[3]:.3f}ms + BWD={best_entry[4]:.3f}ms)")

        # 释放 block_indices 显存（真实 indices 不释放，因为跨 G 共享）
        if real_block_indices is None:
            del block_indices
            import gc; gc.collect()
            torch.cuda.empty_cache()

    # ========== FlashAttention Baseline ==========
    fa_results = []
    if getattr(args, 'with_baseline', False):
        print("\n" + "=" * 120)
        print("FlashAttention Baseline")
        print("=" * 120)
        for G in sorted(candidate_map.keys(), reverse=True):
            if hasattr(args, 'fixed_HQ') and args.fixed_HQ:
                HQ = args.fixed_HQ
                H = HQ // G
            else:
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
    fa_lookup = {r[0]: r for r in fa_results}

    print()
    print("=" * 140)
    print("All Candidates Results")
    print("=" * 140)

    if fa_results:
        header = (f"{'HQ:H':>8} | {'G':>4} | {'M':>5} | "
                  f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
                  f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12} | "
                  f"{'FA_Total':>12} | {'Speedup':>8}")
    else:
        header = (f"{'HQ:H':>8} | {'G':>4} | {'M':>5} | "
                  f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
                  f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12}")
    print(header)
    print("-" * len(header))
    prev_G = None
    for ratio, G, M_block, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms in all_results:
        if prev_G is not None and G != prev_G:
            print("-" * len(header))
        prev_G = G
        line = (f"{'%d:1' % ratio:>8} | {G:>4} | {M_block:>5} | "
                f"{nt_fwd:>6} | {nt_bwd:>6} | "
                f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f}")
        if fa_results:
            fa = fa_lookup.get(ratio)
            if fa:
                speedup = fa[4] / total_ms if total_ms > 0 else float('inf')
                line += f" | {fa[4]:>12.3f} | {speedup:>7.2f}x"
            else:
                line += f" | {'N/A':>12} | {'N/A':>8}"
        print(line)
    print("=" * len(header))

    # ========== Best Config 汇总 ==========
    print()
    print("=" * 140)
    print("Best Config per HQ:H Ratio (Best Total with FWD/BWD breakdown)")
    print("=" * 140)
    best_header = (f"{'HQ:H':>8} | {'G':>4} | {'M':>5} | "
                   f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
                   f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12}")
    if fa_results:
        best_header += f" | {'FA_Total':>12} | {'Speedup':>8}"
    print(best_header)
    print("-" * len(best_header))
    for G in sorted(best_per_G.keys(), reverse=True):
        entry = best_per_G[G]
        ratio = G
        # entry: (M, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)
        line = (f"{'%d:1' % ratio:>8} | {G:>4} | {entry[0]:>5} | "
                f"{entry[1]:>6} | {entry[2]:>6} | "
                f"{entry[3]:>10.3f} | {entry[4]:>10.3f} | {entry[5]:>12.3f}")
        if fa_results:
            fa = fa_lookup.get(ratio)
            if fa:
                speedup = fa[4] / entry[5] if entry[5] > 0 else float('inf')
                line += f" | {fa[4]:>12.3f} | {speedup:>7.2f}x"
            else:
                line += f" | {'N/A':>12} | {'N/A':>8}"
        print(line)
    print("=" * len(best_header))

    # ========== Print best --config string (4-segment new format) ==========
    print()
    config_parts = []
    for G in sorted(best_per_G.keys(), reverse=True):
        entry = best_per_G[G]
        # entry: (M, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms)
        config_parts.append(f"{G}:{entry[0]}:{entry[1]}:{entry[2]}")
    if config_parts:
        config_str = ",".join(config_parts)
        print(f"Best --config string: \"{config_str}\"")
    print()


# =========================================================
# 指定配置测试模式
# =========================================================
def bench_fixed_configs(args):
    """
    Run only the user-specified configs, skipping the full sweep.
    Pass via --config with format:
        "G:M:nt_fwd:nt_bwd,..."
    Example:
        "16:4:128:128,8:8:128:128,4:4:64:64,2:16:128:64,1:16:128:64"
    """
    B = args.B
    SEQ_LEN = args.SEQ_LEN
    D = args.D
    S = args.S
    block_size = args.block_size
    overlap_ratio = args.overlap
    mask_last_token = args.mask_last_token
    base_H = args.base_H

    # 加载真实 indices（如果指定）
    real_block_indices = None
    real_weights = None
    if args.real_indices:
        real_block_indices, real_weights, actual_seq_len = load_real_indices(
            args.real_indices, B, layer_idx=args.layer_idx)
        SEQ_LEN = actual_seq_len
        print(f"  使用真实 indices，SEQ_LEN 已更新为 {SEQ_LEN}（包含 landmark token）")

    # Parse config string. New 4-segment format only.
    configs = []
    for item in args.config.split(","):
        parts = item.strip().split(":")
        assert len(parts) == 4, (
            f"each config needs 4 fields (G:M:nt_fwd:nt_bwd), got: {item}")
        G, M_block, nt_fwd, nt_bwd = (int(parts[0]), int(parts[1]),
                                      int(parts[2]), int(parts[3]))
        configs.append((G, M_block, nt_fwd, nt_bwd))

    print("=" * 120)
    print(f"HSA Latency Benchmark (Fixed Config Mode)")
    print(f"Config: B={B}, L={SEQ_LEN}, D={D}, S={S}, block_size={block_size}, "
          f"overlap={overlap_ratio}, base_H={base_H}")
    if args.real_indices:
        print(f"Real indices: {args.real_indices} (layer_idx={args.layer_idx})")
    print(f"mask_last_token={mask_last_token}")
    print("=" * 120)
    print()

    # Result table header
    header = (f"{'HQ:H':>8} | {'G':>4} | {'M':>5} | "
              f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
              f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12}")

    results = []
    for G, M_block, nt_fwd, nt_bwd in configs:
        if hasattr(args, 'fixed_HQ') and args.fixed_HQ:
            HQ = args.fixed_HQ
            H = HQ // G
        else:
            H = base_H
            HQ = H * G
        ratio = G

        print(f"\n>>> Testing HQ:H = {ratio}:1  (H={H}, HQ={HQ}, G={G})")
        print(f"    M={M_block}, nt_fwd={nt_fwd}, nt_bwd={nt_bwd}")

        # H_kv adaptation: real indices' H_kv may differ from current H.
        cur_block_indices = real_block_indices
        cur_weights = real_weights
        if real_block_indices is not None:
            real_h_kv = real_block_indices.shape[2]
            if H != real_h_kv:
                if H > real_h_kv:
                    assert H % real_h_kv == 0, (
                        f"H={H} not divisible by real_h_kv={real_h_kv}")
                    repeat_factor = H // real_h_kv
                    cur_block_indices = real_block_indices.repeat_interleave(
                        repeat_factor, dim=2).contiguous()
                    cur_weights = (real_weights.repeat_interleave(
                        repeat_factor, dim=2).contiguous()
                        if real_weights is not None else None)
                    print(f"    indices H_kv adapt: {real_h_kv} → {H} (repeat x{repeat_factor})")
                else:
                    cur_block_indices = real_block_indices[:, :, :H, :].contiguous()
                    cur_weights = (real_weights[:, :, :H, :].contiguous()
                                   if real_weights is not None else None)
                    print(f"    indices H_kv adapt: {real_h_kv} → {H} (slice)")

        try:
            fwd_ms, bwd_ms, total_ms = bench_single_config(
                B=B, SEQ_LEN=SEQ_LEN, H=H, HQ=HQ, D=D, S=S,
                block_size=block_size,
                block_M_fwd=M_block, block_M_bwd=M_block,
                overlap_ratio=overlap_ratio,
                num_threads_fwd=nt_fwd, num_threads_bwd=nt_bwd,
                mask_last_token=mask_last_token,
                block_indices=cur_block_indices,
                weights=cur_weights,
            )
            results.append((ratio, G, M_block, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms))
            print(f"    ✓ FWD={fwd_ms:.3f}ms | BWD={bwd_ms:.3f}ms | Total={total_ms:.3f}ms")
        except Exception as e:
            print(f"    ⚠ FAILED: {type(e).__name__}: {e}")

    # ========== FlashAttention Baseline ==========
    fa_results = []
    if getattr(args, 'with_baseline', False):
        print("\n" + "=" * 120)
        print("FlashAttention Baseline")
        print("=" * 120)
        # Test FA baseline once per ratio (same B, SEQ_LEN, HQ, H, D).
        tested_ratios = set()
        for G, M_block, nt_fwd, nt_bwd in configs:
            if G in tested_ratios:
                continue
            tested_ratios.add(G)
            if hasattr(args, 'fixed_HQ') and args.fixed_HQ:
                HQ = args.fixed_HQ
                H = HQ // G
            else:
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
        header_with_fa = (f"{'HQ:H':>8} | {'G':>4} | {'M':>5} | "
                          f"{'nt_fwd':>6} | {'nt_bwd':>6} | "
                          f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12} | "
                          f"{'FA_FWD':>10} | {'FA_BWD':>10} | {'FA_Total':>12} | "
                          f"{'Speedup':>8}")
        print(header_with_fa)
        print("-" * len(header_with_fa))
        for ratio, G, M_block, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms in results:
            fa = fa_lookup.get(ratio)
            if fa:
                speedup = fa[4] / total_ms if total_ms > 0 else float('inf')
                print(f"{'%d:1' % ratio:>8} | {G:>4} | {M_block:>5} | "
                      f"{nt_fwd:>6} | {nt_bwd:>6} | "
                      f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f} | "
                      f"{fa[2]:>10.3f} | {fa[3]:>10.3f} | {fa[4]:>12.3f} | "
                      f"{speedup:>7.2f}x")
            else:
                print(f"{'%d:1' % ratio:>8} | {G:>4} | {M_block:>5} | "
                      f"{nt_fwd:>6} | {nt_bwd:>6} | "
                      f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f} | "
                      f"{'N/A':>10} | {'N/A':>10} | {'N/A':>12} | {'N/A':>8}")
        print("=" * len(header_with_fa))
    else:
        print(header)
        print("-" * len(header))
        for ratio, G, M_block, nt_fwd, nt_bwd, fwd_ms, bwd_ms, total_ms in results:
            print(f"{'%d:1' % ratio:>8} | {G:>4} | {M_block:>5} | "
                  f"{nt_fwd:>6} | {nt_bwd:>6} | "
                  f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f}")
        print("=" * len(header))


# =========================================================
# 综合 LaTeX 表格测试（FA3 + TopK + HSA + HSA+TopK）
# =========================================================
def bench_latex_table(args):
    """
    固定 HQ，改变 H_KV 来实现不同 HQ:H 比例。
    对每个序列长度 × 每个比例，测试 FA3、TopK、HSA Attn、HSA+TopK 四个组件。
    输出可直接复制到 LaTeX 表格中的格式。

    HSA kernel configs come from --hsa_configs in 4-segment format:
        "seq_len:G:M:nt_fwd:nt_bwd,..."
    Example:
        "8192:16:4:128:128,8192:8:8:128:128,..."
    """
    from ops.topk_head_softmax import online_softmax_topk_head

    HQ = args.fixed_HQ  # 固定的 HQ（默认 32）
    B = args.B
    D = args.D
    block_size = args.block_size
    topk = args.S
    window_size = getattr(args, 'topk_window_size', 512)
    mask_last_token = args.mask_last_token
    overlap_ratio = args.overlap
    dtype = torch.bfloat16
    device = "cuda"

    num_warmup = 50
    num_iters = 100

    # 序列长度列表（原始长度，不含 lmk）
    raw_seq_lens = [int(x) for x in args.topk_seq_lens.split(",")]
    # HQ:H 比例列表
    ratios = [16, 8, 4, 2, 1]

    # ---- 测试开关 ----
    only_single = getattr(args, 'only_single', False)
    with_single = getattr(args, 'with_single', False) or only_single
    # only_single 模式下跳过其他 baseline（除非用户显式关闭）
    skip_fa = getattr(args, 'skip_fa', False) or only_single
    skip_topk = getattr(args, 'skip_topk', False) or only_single
    skip_hsa_group = only_single  # 只测 single 时跳过 group 版本
    skip_fsa = getattr(args, 'skip_fsa', False) or only_single

    # Parse HSA kernel configs.
    # Format: "seq_len:G:M:nt_fwd:nt_bwd,..." (4-segment, new format only)
    hsa_config_map = {}  # {(raw_seq_len, G): (M, nt_fwd, nt_bwd)}
    if args.hsa_configs:
        for item in args.hsa_configs.split(","):
            parts = item.strip().split(":")
            assert len(parts) == 5, (
                f"HSA config needs 5 fields (seq_len:G:M:nt_fwd:nt_bwd), got: {item}")
            sl, G, m, nf, nb = [int(x) for x in parts]
            hsa_config_map[(sl, G)] = (m, nf, nb)

    # 解析 HSA single kernel 配置（可选）
    # 格式: "seq_len:G:nt_fwd:nt_bwd,..."
    hsa_single_config_map = {}  # {(raw_seq_len, G): (nt_fwd, nt_bwd)}
    hsa_single_configs_str = getattr(args, 'hsa_single_configs', None)
    if hsa_single_configs_str:
        for item in hsa_single_configs_str.split(","):
            parts = item.strip().split(":")
            assert len(parts) == 4, (
                f"HSA single 配置需要 4 个字段 (seq_len:G:nt_fwd:nt_bwd)，得到: {item}")
            sl, G, nf, nb = [int(x) for x in parts]
            hsa_single_config_map[(sl, G)] = (nf, nb)

    # 加载真实 indices（按序列长度分别加载）
    real_indices_dir = args.real_indices_dir  # 目录路径，如 /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup

    print("=" * 140)
    print(f"HSA + TopK Comprehensive LaTeX Table Benchmark")
    print(f"Config: B={B}, HQ={HQ}(fixed), D={D}, block_size={block_size}, "
          f"topk={topk}, window_size={window_size}")
    print(f"原始序列长度: {raw_seq_lens}")
    print(f"HQ:H 比例: {[f'{r}:1' for r in ratios]}")
    print(f"warmup={num_warmup}, iters={num_iters}")
    skip_dense = getattr(args, 'skip_dense', False) or only_single
    dense_block_M = getattr(args, 'dense_block_M', 64)
    dense_num_threads = getattr(args, 'dense_num_threads', 128)
    print(f"开关: skip_fa={skip_fa}, skip_topk={skip_topk}, "
          f"skip_hsa_group={skip_hsa_group}, with_single={with_single}, only_single={only_single}, "
          f"skip_dense={skip_dense}, dense_block_M={dense_block_M}, dense_num_threads={dense_num_threads}")
    if real_indices_dir:
        print(f"Real indices 目录: {real_indices_dir}")
    if hsa_config_map:
        print(f"HSA kernel 配置:")
        for (sl, G), (m, nf, nb) in sorted(hsa_config_map.items()):
            print(f"  seq_len={sl}, G={G}: block_M={m}, nt_fwd={nf}, nt_bwd={nb}")
    if hsa_single_config_map:
        print(f"HSA single kernel 配置:")
        for (sl, G), (nf, nb) in sorted(hsa_single_config_map.items()):
            print(f"  seq_len={sl}, G={G}: nt_fwd={nf}, nt_bwd={nb}")
    print("=" * 140)
    print()

    # 结果存储
    # fa_data[raw_seq_len][ratio] = (fwd_ms, bwd_ms, total_ms)
    # topk_data[raw_seq_len][ratio] = (fwd_ms, bwd_ms, total_ms)
    # hsa_data[raw_seq_len][ratio] = (fwd_ms, bwd_ms, total_ms)
    # hsa_single_data[raw_seq_len][ratio] = (fwd_ms, bwd_ms, total_ms)
    fa_data = {}
    fa2tl_data = {}
    topk_data = {}
    hsa_data = {}
    hsa_single_data = {}
    fsa_data = {}
    # dense_data[raw_seq_len][ratio] = (fwd_ms, bwd_ms, total_ms) for HSA-Dense end-to-end
    dense_data = {}

    for raw_seq_len in raw_seq_lens:
        L_with_lmk = raw_seq_len + raw_seq_len // block_size
        S_chunks = L_with_lmk // block_size

        fa_data[raw_seq_len] = {}
        fa2tl_data[raw_seq_len] = {}
        topk_data[raw_seq_len] = {}
        hsa_data[raw_seq_len] = {}
        hsa_single_data[raw_seq_len] = {}
        fsa_data[raw_seq_len] = {}
        dense_data[raw_seq_len] = {}

        print(f"\n{'#'*140}")
        print(f"### 原始序列长度: {raw_seq_len} ({raw_seq_len//1024}k) → "
              f"插入 lmk 后: L={L_with_lmk}, S(chunks)={S_chunks}")
        print(f"{'#'*140}")

        # 加载真实 indices（如果有）
        real_block_indices = None
        real_weights = None
        actual_hsa_seq_len = L_with_lmk
        if real_indices_dir:
            pt_path = os.path.join(real_indices_dir, f"indices_{raw_seq_len}.pt")
            if os.path.exists(pt_path):
                real_block_indices, real_weights, actual_hsa_seq_len = load_real_indices(
                    pt_path, B, layer_idx=args.layer_idx)
                print(f"  使用真实 indices，HSA SEQ_LEN={actual_hsa_seq_len}")
            else:
                print(f"  ⚠ 未找到 {pt_path}，HSA 将使用构造的 indices")

        for ratio in ratios:
            G = ratio
            H_KV = HQ // G  # 固定 HQ，计算 H_KV

            print(f"\n  {'='*100}")
            print(f"  >>> HQ:H = {ratio}:1  (HQ={HQ}, H_KV={H_KV}, G={G})")
            print(f"  {'='*100}")

            # ========== 1. FA3 Baseline ==========
            # 注意: FA baseline 使用原始序列长度 raw_seq_len (不含 lmk token)，
            # 因为 FA 是 full attention baseline，不需要 landmark token。
            if skip_fa:
                print(f"\n    [FA3] 已跳过 (--skip_fa)")
            else:
                print(f"\n    [FA3] B={B}, L={raw_seq_len} (raw, no lmk), HQ={HQ}, H={H_KV}, D={D}")
                try:
                    fa_fwd, fa_bwd, fa_total = bench_flash_attn(
                        B=B, SEQ_LEN=raw_seq_len, H=H_KV, HQ=HQ, D=D,
                        num_warmup=num_warmup, num_iters=num_iters,
                        backend="fa3",
                    )
                    fa_data[raw_seq_len][ratio] = (fa_fwd, fa_bwd, fa_total)
                    print(f"    ✓ FA3: FWD={fa_fwd:.3f}ms | BWD={fa_bwd:.3f}ms | Total={fa_total:.3f}ms")
                except Exception as e:
                    print(f"    ⚠ FA3 FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()

            # ========== 1.6 FA2 (tilelang) Baseline ==========
            # 使用 ops.example_gqa_bwd.attention（tilelang FA2 风格 GQA），强制 fp16
            if skip_fa:
                print(f"\n    [FA2_tl] 已跳过 (--skip_fa)")
            else:
                print(f"\n    [FA2_tl] B={B}, L={raw_seq_len} (raw, no lmk), HQ={HQ}, H={H_KV}, D={D} (fp16)")
                try:
                    fa2tl_fwd, fa2tl_bwd, fa2tl_total = bench_flash_attn_tilelang(
                        B=B, SEQ_LEN=raw_seq_len, H=H_KV, HQ=HQ, D=D,
                        num_warmup=num_warmup, num_iters=num_iters,
                    )
                    fa2tl_data[raw_seq_len][ratio] = (fa2tl_fwd, fa2tl_bwd, fa2tl_total)
                    print(f"    ✓ FA2_tl: FWD={fa2tl_fwd:.3f}ms | BWD={fa2tl_bwd:.3f}ms | Total={fa2tl_total:.3f}ms")
                except Exception as e:
                    print(f"    ⚠ FA2_tl FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()

            # ========== 2. TopK ==========
            # head-wise topk: q is (B, L, h_kv, G, D); lmks is (B, S_chunks, h_kv, D)
            # 使用 per-KV-head landmark + per-q-head q，与训练一致。
            topk_h_kv = H_KV
            topk_h_q = HQ
            topk_G = HQ // H_KV
            if skip_topk:
                print(f"\n    [TopK] 已跳过 (--skip_topk)")
            else:
                print(f"\n    [TopK] h_q={topk_h_q}, h_kv={topk_h_kv}, G={topk_G}, topk={topk}")
                try:
                    # head-wise topk 期待 q: (B, L, h_kv, G, D)
                    q_topk = torch.randn(
                        B, L_with_lmk, topk_h_kv, topk_G, D,
                        dtype=dtype, device=device, requires_grad=True)
                    lmks_topk = torch.randn(
                        B, S_chunks, topk_h_kv, D,
                        dtype=dtype, device=device, requires_grad=True)
                    # lse_swa: (B, L, h_kv, G) used by online_softmax_topk_head
                    lse_swa = torch.randn(
                        B, L_with_lmk, topk_h_kv, topk_G,
                        dtype=dtype, device=device) * 5.0 + 10.0
                    # scores returned by head-wise kernel: (B, L, h_q, topk).
                    grad_topk = torch.randn(
                        B, L_with_lmk, topk_h_q, topk,
                        dtype=dtype, device=device)

                    # Warmup
                    for _ in range(num_warmup):
                        q_topk.grad = None
                        lmks_topk.grad = None
                        _, scores = online_softmax_topk_head(
                            q_topk, lmks_topk, lse_swa, topk, block_size, window_size, True,
                            is_training=True,
                        )
                        loss = (scores * grad_topk).sum()
                        loss.backward()
                    torch.cuda.synchronize()

                    # FWD only
                    # NOTE: keep is_training=True (same path as FWD+BWD) and do NOT wrap
                    # with torch.no_grad(). is_training=False/inference path can be much
                    # slower than the training path, which would otherwise inflate FWD
                    # time and yield negative BWD = total - fwd. We only skip the backward
                    # call here; the forward kernel itself is identical to the FWD+BWD case.
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(num_iters):
                        online_softmax_topk_head(
                            q_topk, lmks_topk, lse_swa, topk, block_size, window_size, True,
                            is_training=True,
                        )
                    end.record()
                    torch.cuda.synchronize()
                    topk_fwd = start.elapsed_time(end) / num_iters

                    # FWD + BWD
                    torch.cuda.synchronize()
                    start2 = torch.cuda.Event(enable_timing=True)
                    end2 = torch.cuda.Event(enable_timing=True)
                    start2.record()
                    for _ in range(num_iters):
                        q_topk.grad = None
                        lmks_topk.grad = None
                        _, scores = online_softmax_topk_head(
                            q_topk, lmks_topk, lse_swa, topk, block_size, window_size, True,
                            is_training=True,
                        )
                        loss = (scores * grad_topk).sum()
                        loss.backward()
                    end2.record()
                    torch.cuda.synchronize()
                    topk_total = start2.elapsed_time(end2) / num_iters
                    topk_bwd = topk_total - topk_fwd

                    topk_data[raw_seq_len][ratio] = (topk_fwd, topk_bwd, topk_total)
                    print(f"    ✓ TopK: FWD={topk_fwd:.3f}ms | BWD={topk_bwd:.3f}ms | Total={topk_total:.3f}ms")

                    del q_topk, lmks_topk, lse_swa, grad_topk
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    ⚠ TopK FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()

            # ========== 3. HSA Attn Only ==========
            hsa_key = (raw_seq_len, G)
            if skip_hsa_group:
                print(f"\n    [HSA] 已跳过 (--only_single)")
            elif hsa_key in hsa_config_map:
                M_block, nt_fwd, nt_bwd = hsa_config_map[hsa_key]
                print(f"\n    [HSA] G={G}, block_M={M_block}, "
                      f"nt_fwd={nt_fwd}, nt_bwd={nt_bwd}")
                try:
                    # 适配 block_indices 和 weights 的 H_kv 维度
                    cur_block_indices = real_block_indices
                    cur_weights = real_weights
                    if cur_block_indices is not None:
                        real_h_kv = cur_block_indices.shape[2]
                        if H_KV != real_h_kv:
                            if H_KV > real_h_kv:
                                # H_KV 更大：repeat_interleave 扩展
                                assert H_KV % real_h_kv == 0, (
                                    f"H_KV={H_KV} 不是 real_h_kv={real_h_kv} 的整数倍")
                                repeat_factor = H_KV // real_h_kv
                                cur_block_indices = cur_block_indices.repeat_interleave(
                                    repeat_factor, dim=2).contiguous()
                                if cur_weights is not None:
                                    cur_weights = cur_weights.repeat_interleave(
                                        repeat_factor, dim=2).contiguous()
                                print(f"    indices H_kv 适配: {real_h_kv} → {H_KV} (repeat x{repeat_factor})")
                            else:
                                # H_KV 更小：slice 取前 H_KV 个 head
                                cur_block_indices = cur_block_indices[:, :, :H_KV, :].contiguous()
                                if cur_weights is not None:
                                    cur_weights = cur_weights[:, :, :H_KV, :].contiguous()
                                print(f"    indices H_kv 适配: {real_h_kv} → {H_KV} (slice)")

                    hsa_fwd, hsa_bwd, hsa_total = bench_single_config(
                        B=B, SEQ_LEN=actual_hsa_seq_len, H=H_KV, HQ=HQ, D=D, S=topk,
                        block_size=block_size,
                        block_M_fwd=M_block, block_M_bwd=M_block,
                        overlap_ratio=overlap_ratio,
                        num_threads_fwd=nt_fwd, num_threads_bwd=nt_bwd,
                        mask_last_token=mask_last_token,
                        num_warmup=num_warmup, num_iters=num_iters,
                        block_indices=cur_block_indices,
                        weights=cur_weights,
                    )
                    hsa_data[raw_seq_len][ratio] = (hsa_fwd, hsa_bwd, hsa_total)
                    print(f"    ✓ HSA: FWD={hsa_fwd:.3f}ms | BWD={hsa_bwd:.3f}ms | Total={hsa_total:.3f}ms")
                except Exception as e:
                    print(f"    ⚠ HSA FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()
            else:
                print(f"\n    [HSA] ⚠ 未找到 seq_len={raw_seq_len}, G={G} 的 HSA 配置，跳过")

            # ========== 4. HSA Single Baseline（只支持 16:1，其他比例通过 HQ padding 实现）==========
            if with_single:
                nt_fwd_s, nt_bwd_s = hsa_single_config_map.get((raw_seq_len, G), (None, None))
                print(f"\n    [HSA-Single] G={G}, nt_fwd={nt_fwd_s}, nt_bwd={nt_bwd_s}"
                      f"{' (HQ padding to 16*H)' if G < 16 else ''}")
                try:
                    # 适配 block_indices 和 weights 的 H_kv 维度（与 HSA group 相同逻辑）
                    cur_bi_s = real_block_indices
                    cur_w_s = real_weights
                    if cur_bi_s is not None:
                        real_h_kv = cur_bi_s.shape[2]
                        if H_KV != real_h_kv:
                            if H_KV > real_h_kv:
                                assert H_KV % real_h_kv == 0, (
                                    f"H_KV={H_KV} 不是 real_h_kv={real_h_kv} 的整数倍")
                                repeat_factor = H_KV // real_h_kv
                                cur_bi_s = cur_bi_s.repeat_interleave(
                                    repeat_factor, dim=2).contiguous()
                                if cur_w_s is not None:
                                    cur_w_s = cur_w_s.repeat_interleave(
                                        repeat_factor, dim=2).contiguous()
                            else:
                                cur_bi_s = cur_bi_s[:, :, :H_KV, :].contiguous()
                                if cur_w_s is not None:
                                    cur_w_s = cur_w_s[:, :, :H_KV, :].contiguous()

                    hs_fwd, hs_bwd, hs_total = bench_single_config_hsa_single(
                        B=B, SEQ_LEN=actual_hsa_seq_len, H=H_KV, HQ=HQ, D=D, S=topk,
                        block_size=block_size,
                        overlap_ratio=overlap_ratio,
                        num_threads_fwd=nt_fwd_s, num_threads_bwd=nt_bwd_s,
                        mask_last_token=mask_last_token,
                        num_warmup=num_warmup, num_iters=num_iters,
                        block_indices=cur_bi_s,
                        weights=cur_w_s,
                    )
                    hsa_single_data[raw_seq_len][ratio] = (hs_fwd, hs_bwd, hs_total)
                    print(f"    ✓ HSA-Single: FWD={hs_fwd:.3f}ms | BWD={hs_bwd:.3f}ms | Total={hs_total:.3f}ms")
                except Exception as e:
                    print(f"    ⚠ HSA-Single FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()

            # ========== 5. FSA Select Attention Baseline ==========
            if skip_fsa:
                print(f"\n    [FSA] 已跳过 (--skip_fsa)")
            else:
                print(f"\n    [FSA] G={G}, HQ={HQ}, H_KV={H_KV}")
                try:
                    # 复用 HSA 的 block_indices（同一份真实 indices），适配 H_kv 维度
                    cur_bi_f = real_block_indices
                    if cur_bi_f is not None:
                        real_h_kv = cur_bi_f.shape[2]
                        if H_KV != real_h_kv:
                            if H_KV > real_h_kv:
                                assert H_KV % real_h_kv == 0, (
                                    f"H_KV={H_KV} 不是 real_h_kv={real_h_kv} 的整数倍")
                                repeat_factor = H_KV // real_h_kv
                                cur_bi_f = cur_bi_f.repeat_interleave(
                                    repeat_factor, dim=2).contiguous()
                            else:
                                cur_bi_f = cur_bi_f[:, :, :H_KV, :].contiguous()
                    else:
                        # 未提供真实 indices 时，构造假 indices（与 HSA 一致）
                        cur_bi_f = build_block_indices_union_overlap(
                            B=B, SEQ_LEN=actual_hsa_seq_len, H=H_KV, S=topk,
                            block_size=block_size,
                            overlap_ratio=overlap_ratio,
                            block_M=1,
                            device=device,
                        )

                    fsa_fwd, fsa_bwd, fsa_total = bench_fsa_select_kernel(
                        B=B, SEQ_LEN=actual_hsa_seq_len, H=H_KV, HQ=HQ, D=D, S=topk,
                        block_size=block_size,
                        block_indices=cur_bi_f,
                        num_warmup=num_warmup, num_iters=num_iters,
                        dtype=dtype, device=device,
                    )
                    fsa_data[raw_seq_len][ratio] = (fsa_fwd, fsa_bwd, fsa_total)
                    print(f"    ✓ FSA: FWD={fsa_fwd:.3f}ms | BWD={fsa_bwd:.3f}ms | Total={fsa_total:.3f}ms")
                except Exception as e:
                    print(f"    ⚠ FSA FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()

            # ========== 6. HSA-Dense end-to-end ==========
            # Dense path: lmk dense scoring + dense HSA attention. No real indices needed.
            # Use the lmk-inserted length L_with_lmk so the result is comparable with
            # FA3 / TopK / HSA which also operate on this length.
            if skip_dense:
                print(f"\n    [HSA-Dense] 已跳过 (--skip_dense)")
            else:
                print(f"\n    [HSA-Dense] G={G}, HQ={HQ}, H_KV={H_KV}, "
                      f"block_M={dense_block_M}, num_threads={dense_num_threads}")
                try:
                    dn_fwd, dn_bwd, dn_total = bench_dense_interface(
                        B=B, SEQ_LEN=L_with_lmk, H=H_KV, HQ=HQ, D=D,
                        block_size=block_size, window_size=window_size,
                        dense_block_M=dense_block_M,
                        dense_num_threads=dense_num_threads,
                        mask_last_token=mask_last_token,
                        num_warmup=num_warmup, num_iters=num_iters,
                        dtype=dtype, device=device,
                    )
                    dense_data[raw_seq_len][ratio] = (dn_fwd, dn_bwd, dn_total)
                    print(f"    ✓ HSA-Dense: FWD={dn_fwd:.3f}ms | BWD={dn_bwd:.3f}ms | Total={dn_total:.3f}ms")
                except Exception as e:
                    print(f"    ⚠ HSA-Dense FAILED: {type(e).__name__}: {e}")
                    traceback.print_exc()

    # ========== 汇总表格（终端可读格式）==========
    print()
    print("=" * 180)
    print("Comprehensive Latency Summary (ms)")
    print("=" * 180)

    for raw_seq_len in raw_seq_lens:
        print(f"\n--- {raw_seq_len//1024}K ---")
        print(f"{'Component':>16} | {'HQ:H':>6} | {'FWD':>8} | {'BWD':>8} | {'Total':>8} | {'Spd.':>6}")
        print("-" * 60)
        for ratio in ratios:
            # FA3
            if ratio in fa_data.get(raw_seq_len, {}):
                f, b, t = fa_data[raw_seq_len][ratio]
                print(f"{'FA3':>16} | {ratio:>2}:1   | {f:>8.2f} | {b:>8.2f} | {t:>8.2f} | {'1.00x':>6}")
            # FA2 (tilelang)
            if ratio in fa2tl_data.get(raw_seq_len, {}):
                f, b, t = fa2tl_data[raw_seq_len][ratio]
                fa3_t = fa_data.get(raw_seq_len, {}).get(ratio)
                spd = f"{fa3_t[2]/t:.2f}x" if fa3_t else "N/A"
                print(f"{'FA2_tl':>16} | {ratio:>2}:1   | {f:>8.2f} | {b:>8.2f} | {t:>8.2f} | {spd:>6}")
            # TopK
            if ratio in topk_data.get(raw_seq_len, {}):
                f, b, t = topk_data[raw_seq_len][ratio]
                print(f"{'TopK':>16} | {ratio:>2}:1   | {f:>8.2f} | {b:>8.2f} | {t:>8.2f} |")
            # HSA
            if ratio in hsa_data.get(raw_seq_len, {}):
                f, b, t = hsa_data[raw_seq_len][ratio]
                print(f"{'HSA (Attn)':>16} | {ratio:>2}:1   | {f:>8.2f} | {b:>8.2f} | {t:>8.2f} |")
            # HSA + TopK
            if (ratio in topk_data.get(raw_seq_len, {}) and
                ratio in hsa_data.get(raw_seq_len, {})):
                tf, tb, tt = topk_data[raw_seq_len][ratio]
                hf, hb, ht = hsa_data[raw_seq_len][ratio]
                sum_fwd = tf + hf
                sum_bwd = tb + hb
                sum_total = tt + ht
                fa_t = fa_data.get(raw_seq_len, {}).get(ratio)
                spd = f"{fa_t[2]/sum_total:.2f}x" if fa_t else "N/A"
                print(f"{'HSA + TopK':>16} | {ratio:>2}:1   | {sum_fwd:>8.2f} | {sum_bwd:>8.2f} | {sum_total:>8.2f} | {spd:>6}")
            # HSA Single
            if ratio in hsa_single_data.get(raw_seq_len, {}):
                f, b, t = hsa_single_data[raw_seq_len][ratio]
                print(f"{'HSA-Single':>16} | {ratio:>2}:1   | {f:>8.2f} | {b:>8.2f} | {t:>8.2f} |")
            # HSA Single + TopK
            if (ratio in topk_data.get(raw_seq_len, {}) and
                ratio in hsa_single_data.get(raw_seq_len, {})):
                tf, tb, tt = topk_data[raw_seq_len][ratio]
                hf, hb, ht = hsa_single_data[raw_seq_len][ratio]
                sum_fwd = tf + hf
                sum_bwd = tb + hb
                sum_total = tt + ht
                fa_t = fa_data.get(raw_seq_len, {}).get(ratio)
                spd = f"{fa_t[2]/sum_total:.2f}x" if fa_t else "N/A"
                print(f"{'HSA-Single+TopK':>16} | {ratio:>2}:1   | {sum_fwd:>8.2f} | {sum_bwd:>8.2f} | {sum_total:>8.2f} | {spd:>6}")
            # FSA
            if ratio in fsa_data.get(raw_seq_len, {}):
                f, b, t = fsa_data[raw_seq_len][ratio]
                print(f"{'FSA (Attn)':>16} | {ratio:>2}:1   | {f:>8.2f} | {b:>8.2f} | {t:>8.2f} |")
            # HSA-Dense (end-to-end, includes lmk dense scoring + dense attention)
            if ratio in dense_data.get(raw_seq_len, {}):
                f, b, t = dense_data[raw_seq_len][ratio]
                fa_t = fa_data.get(raw_seq_len, {}).get(ratio)
                spd = f"{fa_t[2]/t:.2f}x" if fa_t and t > 0 else "N/A"
                print(f"{'HSA-Dense':>16} | {ratio:>2}:1   | {f:>8.2f} | {b:>8.2f} | {t:>8.2f} | {spd:>6}")

    # ========== LaTeX 格式输出 ==========
    print()
    print("=" * 180)
    print("LaTeX Table Rows (可直接复制)")
    print("=" * 180)

    def fmt(val):
        """格式化数值为 LaTeX 表格中的字符串"""
        if val is None:
            return "--"
        return f"{val:.2f}"

    def fmt_spd(numer_total, denom_total):
        """计算 speedup 字符串：denom_total / numer_total = 加速比"""
        if numer_total is None or denom_total is None or numer_total <= 0:
            return "--"
        return f"{denom_total/numer_total:.2f}$\\times$"

    # 每个 cell 为: FWD & BWD & Total & vs. FA3 & vs. FA2_tl （共 5 列）
    # 加速比规则：
    #   - FA3 行：vs. FA3 = 1.00x, vs. FA2_tl = --
    #   - FA2 (tilelang) 行：vs. FA3 = --, vs. FA2_tl = 1.00x
    #   - TopK / HSA (Attn only) / HSA Padding (Attn only) / FSA (Attn only)：两列都 --
    #   - 其他（HSA+TopK / HSA Padding + TopK）：分别计算 vs. FA3 / vs. FA2_tl
    NO_SPEEDUP_COMPONENTS = {"TopK", "HSA (Attn only)", "HSA Padding (Attn only)", "FSA (Attn only)"}

    def make_row(get_triple, component_name):
        """
        get_triple(raw_seq_len, ratio) -> (fwd, bwd, total) or None
        """
        lines = []
        for ratio in ratios:
            parts = []
            for raw_seq_len in raw_seq_lens:
                triple = get_triple(raw_seq_len, ratio)
                fa3 = fa_data.get(raw_seq_len, {}).get(ratio)
                fa2tl = fa2tl_data.get(raw_seq_len, {}).get(ratio)
                if triple is None:
                    parts.append("& -- & -- & -- & -- & --")
                    continue
                f, b, t = triple
                if component_name == "FA3":
                    vs_fa3 = "1.00$\\times$"
                    vs_fa2tl = "--"
                elif component_name == "FA2 (tilelang)":
                    vs_fa3 = "--"
                    vs_fa2tl = "1.00$\\times$"
                elif component_name in NO_SPEEDUP_COMPONENTS:
                    vs_fa3 = "--"
                    vs_fa2tl = "--"
                else:
                    vs_fa3 = fmt_spd(t, fa3[2]) if fa3 else "--"
                    vs_fa2tl = fmt_spd(t, fa2tl[2]) if fa2tl else "--"
                parts.append(f"& {fmt(f)} & {fmt(b)} & {fmt(t)} & {vs_fa3} & {vs_fa2tl}")
            line = f"& ${ratio}{{:}}1$  " + " ".join(parts) + " \\\\"
            lines.append(line)
        return lines

    def print_section(title, latex_name, get_triple, component_name, with_midrule=True):
        print(f"\n% === {title} ===")
        print(f"\\multirow{{5}}{{*}}{{{latex_name}}}")
        for line in make_row(get_triple, component_name):
            print(line)
        if with_midrule:
            print("\\midrule")

    # 组合 triple 的辅助函数
    def _sum_triple(a, b):
        if a is None or b is None:
            return None
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    # FA3 行
    print_section(
        "FA3", "FA3",
        lambda sl, r: fa_data.get(sl, {}).get(r),
        component_name="FA3",
    )

    # FA2 (tilelang) 行 —— 走 else 分支自动计算 vs. FA3
    print_section(
        "FA2 (tilelang)", "FA2 (tilelang)",
        lambda sl, r: fa2tl_data.get(sl, {}).get(r),
        component_name="FA2 (tilelang)",
    )

    # TopK 行
    print_section(
        "TopK", "TopK",
        lambda sl, r: topk_data.get(sl, {}).get(r),
        component_name="TopK",
    )

    # FSA (Attn only) —— 不计算加速比
    print_section(
        "FSA (Attn only)", "FSA (Attn only)",
        lambda sl, r: fsa_data.get(sl, {}).get(r),
        component_name="FSA (Attn only)",
    )

    # HSA Padding (Attn only) —— 即 HSA single（通过 HQ padding 到 16*H 实现）
    print_section(
        "HSA Padding (Attn only)", "HSA Padding (Attn only)",
        lambda sl, r: hsa_single_data.get(sl, {}).get(r),
        component_name="HSA Padding (Attn only)",
    )

    # HSA Padding + TopK
    print_section(
        "HSA Padding + TopK", "HSA Padding + TopK",
        lambda sl, r: _sum_triple(
            hsa_single_data.get(sl, {}).get(r),
            topk_data.get(sl, {}).get(r),
        ),
        component_name="HSA Padding + TopK",
    )

    # HSA (Attn only) —— group 版
    print_section(
        "HSA (Attn only)", "HSA (Attn only)",
        lambda sl, r: hsa_data.get(sl, {}).get(r),
        component_name="HSA (Attn only)",
    )

    # HSA + TopK —— group 版 + TopK
    print_section(
        "HSA + TopK", "HSA + TopK",
        lambda sl, r: _sum_triple(
            hsa_data.get(sl, {}).get(r),
            topk_data.get(sl, {}).get(r),
        ),
        component_name="HSA + TopK",
        with_midrule=True,
    )

    # HSA-Dense (end-to-end). Uses the generic else-branch in make_row to compute
    # vs. FA3 / vs. FA2_tl speedups (it is end-to-end, not attn-only).
    print_section(
        "HSA-Dense", "HSA-Dense",
        lambda sl, r: dense_data.get(sl, {}).get(r),
        component_name="HSA-Dense",
        with_midrule=False,
    )

    print()


# =========================================================
# TopK 延迟测试
# =========================================================
def bench_topk(args):
    """
    测试 online_softmax_topk_head（head-wise）在不同序列长度和 HQ:H 比例下的 FWD/BWD 延迟。
    序列长度使用插入 landmark token 后的长度。
    topk 不受数据影响，使用随机数据即可。
    """
    from ops.topk_head_softmax import online_softmax_topk_head

    B = args.B
    D = args.D
    block_size = args.block_size
    topk = args.S  # 复用 S 参数作为 topk
    window_size = getattr(args, 'topk_window_size', 512)
    base_H = args.base_H  # h_kv = 4
    is_causal = True
    dtype = torch.bfloat16
    device = "cuda"

    num_warmup = 30
    num_iters = 50

    # 序列长度列表（原始长度，不含 lmk）
    raw_seq_lens = [int(x) for x in args.topk_seq_lens.split(",")]
    # HQ:H 比例列表
    ratios = [16, 8, 4, 2, 1]

    print("=" * 140)
    print(f"TopK (online_softmax_topk_head, head-wise) Latency Benchmark")
    print(f"Config: B={B}, D={D}, block_size={block_size}, topk={topk}, "
          f"window_size={window_size}, base_H(h_kv)={base_H}")
    print(f"原始序列长度: {raw_seq_lens}")
    print(f"HQ:H 比例: {[f'{r}:1' for r in ratios]}")
    print(f"warmup={num_warmup}, iters={num_iters}")
    print("=" * 140)
    print()

    # 计算插入 lmk 后的实际长度
    # 每 chunk_size 个 token 后插入 1 个 lmk token
    # L_with_lmk = raw_seq_len + raw_seq_len // chunk_size
    # S (num_chunks) = L_with_lmk // chunk_size（近似）
    # 但实际上 topk 的 q 长度就是 L_with_lmk，lmks 长度是 S = L_with_lmk // chunk_size

    # 结果收集
    # (raw_seq_len, L_with_lmk, ratio, h_q, h_kv, S, fwd_ms, bwd_ms, total_ms)
    results = []

    for raw_seq_len in raw_seq_lens:
        L_with_lmk = raw_seq_len + raw_seq_len // block_size
        S = L_with_lmk // block_size  # landmark 数量

        print(f"\n{'#'*140}")
        print(f"### 原始序列长度: {raw_seq_len} ({raw_seq_len//1024}k) → 插入 lmk 后: L={L_with_lmk}, S(chunks)={S}")
        print(f"{'#'*140}")

        for ratio in ratios:
            G = ratio
            h_kv = base_H
            h_q = h_kv * G  # topk 的 q head 数 = h_kv * G

            print(f"\n  --- HQ:H = {ratio}:1  (h_q={h_q}, h_kv={h_kv}, G={G}) ---")

            # head-wise topk: q is 5-D (B, L, h_kv, G, D); lmks is per-h_kv
            q = torch.randn(B, L_with_lmk, h_kv, G, D, dtype=dtype,
                            device=device, requires_grad=True)
            lmks = torch.randn(B, S, h_kv, D, dtype=dtype,
                               device=device, requires_grad=True)
            # lse_swa: (B, L, h_kv, G)
            lse_swa = torch.randn(B, L_with_lmk, h_kv, G, dtype=dtype,
                                  device=device) * 5.0 + 10.0
            # scores returned by head-wise kernel: (B, L, h_q, topk)
            grad_output = torch.randn(B, L_with_lmk, h_q, topk,
                                      dtype=dtype, device=device)

            try:
                # Warmup
                for _ in range(num_warmup):
                    q.grad = None
                    lmks.grad = None
                    _, scores = online_softmax_topk_head(
                        q, lmks, lse_swa, topk, block_size, window_size, is_causal,
                        is_training=True,
                    )
                    loss = (scores * grad_output).sum()
                    loss.backward()
                torch.cuda.synchronize()

                # FWD only
                # NOTE: keep is_training=True (same path as FWD+BWD) and do NOT wrap
                # with torch.no_grad(). The inference path can be much slower than the
                # training path, which would otherwise inflate FWD time and produce a
                # negative BWD = total - fwd. Only skip the backward call here.
                torch.cuda.synchronize()
                start_fwd = torch.cuda.Event(enable_timing=True)
                end_fwd = torch.cuda.Event(enable_timing=True)
                start_fwd.record()
                for _ in range(num_iters):
                    online_softmax_topk_head(
                        q, lmks, lse_swa, topk, block_size, window_size, is_causal,
                        is_training=True,
                    )
                end_fwd.record()
                torch.cuda.synchronize()
                fwd_ms = start_fwd.elapsed_time(end_fwd) / num_iters

                # FWD + BWD
                torch.cuda.synchronize()
                start_all = torch.cuda.Event(enable_timing=True)
                end_all = torch.cuda.Event(enable_timing=True)
                start_all.record()
                for _ in range(num_iters):
                    q.grad = None
                    lmks.grad = None
                    _, scores = online_softmax_topk_head(
                        q, lmks, lse_swa, topk, block_size, window_size, is_causal,
                        is_training=True,
                    )
                    loss = (scores * grad_output).sum()
                    loss.backward()
                end_all.record()
                torch.cuda.synchronize()
                total_ms = start_all.elapsed_time(end_all) / num_iters
                bwd_ms = total_ms - fwd_ms

                results.append((raw_seq_len, L_with_lmk, ratio, h_q, h_kv, S,
                                fwd_ms, bwd_ms, total_ms))
                print(f"    ✓ FWD={fwd_ms:.3f}ms | BWD={bwd_ms:.3f}ms | Total={total_ms:.3f}ms")

            except Exception as e:
                print(f"    ⚠ FAILED: {type(e).__name__}: {e}")

            # 释放显存
            del q, lmks, grad_output
            torch.cuda.empty_cache()

    # ========== 汇总表格 ==========
    print()
    print("=" * 160)
    print("TopK (online_softmax_topk_head) Latency Summary")
    print("=" * 160)
    header = (f"{'SeqLen':>8} | {'L(+lmk)':>9} | {'S(chunks)':>9} | "
              f"{'HQ:H':>8} | {'h_q':>4} | {'h_kv':>4} | "
              f"{'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>12}")
    print(header)
    print("-" * len(header))
    prev_seq_len = None
    for raw_seq_len, L_with_lmk, ratio, h_q, h_kv, S, fwd_ms, bwd_ms, total_ms in results:
        if prev_seq_len is not None and raw_seq_len != prev_seq_len:
            print("-" * len(header))
        prev_seq_len = raw_seq_len
        print(f"{raw_seq_len:>8} | {L_with_lmk:>9} | {S:>9} | "
              f"{'%d:1' % ratio:>8} | {h_q:>4} | {h_kv:>4} | "
              f"{fwd_ms:>10.3f} | {bwd_ms:>10.3f} | {total_ms:>12.3f}")
    print("=" * len(header))


# =========================================================
# 命令行入口
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="HSA Latency Benchmark - 测试不同 HQ:H 比例下的 HSA 延迟（FWD/BWD 独立参数）")
    parser.add_argument("--max_block_M", type=int, default=128, help="block_M 遍历的最大值（默认 128），单一 block_M 从 max_block_M//G 开始翻倍到 max_block_M")
    parser.add_argument("--overlap", type=float, default=0.8, help="overlap_ratio（默认 0.8）")
    parser.add_argument("--B", type=int, default=4, help="batch size（默认 4）")
    parser.add_argument("--SEQ_LEN", type=int, default=8192*2, help="序列长度（默认 8192）")
    parser.add_argument("--D", type=int, default=128, help="head dim（默认 128）")
    parser.add_argument("--S", type=int, default=16, help="每个 query 选择的 block 数量（默认 16）")
    parser.add_argument("--block_size", type=int, default=64, help="block size（默认 64）")
    parser.add_argument("--base_H", type=int, default=4, help="基准 KV head 数，HQ = base_H * G（默认 4）")
    parser.add_argument("--mask_last_token", action="store_true", default=True, help="是否 mask last token（默认 True）")
    parser.add_argument("--no_mask_last_token", action="store_true", default=False, help="关闭 mask last token")
    parser.add_argument("--config", type=str, default=None,
                        help="指定固定配置直接测试，跳过全量搜索。"
                             "格式: 'G:M:nt_fwd:nt_bwd,...' "
                             "例如: '16:4:128:128,8:8:128:128,4:4:64:128,2:16:128:64,1:16:128:64'")
    parser.add_argument("--with_baseline", action="store_true", default=False,
                        help="同时测试 FlashAttention baseline 作为对比")
    parser.add_argument("--sweep_seq_len", action="store_true", default=False,
                        help="遍历多个序列长度 (8k~128k) 搜索每个长度下的最优参数")
    parser.add_argument("--candidates", type=str, default=None,
                        help="候选配置搜索模式，对每个 G 指定多套候选配置。"
                             "格式: 'G:M:nt_fwd:nt_bwd|M:nt_fwd:nt_bwd|...;G:...' "
                             "每个 G 用 ';' 分隔，同一 G 下多套候选用 '|' 分隔。"
                             "例如: '16:4:128:128;8:8:128:128;"
                             "4:4:64:64|4:64:128;2:8:64:64|16:128:64;1:16:128:64'")
    parser.add_argument("--real_indices", type=str, default=None,
                        help="使用真实收集的 indices 数据（.pt 文件路径）。"
                             "传入后将替代 build_block_indices_union_overlap 构造的假数据，"
                             "--overlap 参数将被忽略。"
                             "例如: /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup/indices_8192.pt")
    parser.add_argument("--layer_idx", type=int, default=None,
                        help="使用真实 indices 时，指定使用哪一层的数据（默认自动选第一个 HSA 层）")
    parser.add_argument("--bench_topk", action="store_true", default=False,
                        help="测试 online_softmax_topk_head 在不同序列长度和 HQ:H 比例下的延迟")
    parser.add_argument("--topk_seq_lens", type=str, default="8192,16384,32768,65536,131072",
                        help="topk 测试的原始序列长度列表（不含 lmk），逗号分隔（默认: 8192,16384,32768,65536,131072）")
    parser.add_argument("--topk_window_size", type=int, default=512,
                        help="topk 测试的 window_size（默认: 512）")
    parser.add_argument("--bench_latex", action="store_true", default=False,
                        help="综合测试 FA3+TopK+HSA，输出 LaTeX 表格格式")
    parser.add_argument("--fixed_HQ", type=int, default=32,
                        help="固定的 HQ 值（默认: 32），H_KV = HQ / G")
    parser.add_argument("--hsa_configs", type=str, default=None,
                        help="HSA kernel 配置，格式: 'seq_len:G:M:nt_fwd:nt_bwd,...' "
                             "例如: '8192:16:4:128:128,8192:8:8:128:128,...'")
    parser.add_argument("--real_indices_dir", type=str, default=None,
                        help="真实 indices 目录路径（如 /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup），"
                             "会自动加载 indices_{seq_len}.pt")
    parser.add_argument("--skip_fa", action="store_true", default=False,
                        help="跳过 FA3 baseline 测试（加速调试）")
    parser.add_argument("--skip_topk", action="store_true", default=False,
                        help="跳过 TopK 测试（加速调试）")
    parser.add_argument("--skip_fsa", action="store_true", default=False,
                        help="跳过 FSA select attention baseline 测试（加速调试）")
    parser.add_argument("--skip_dense", action="store_true", default=False,
                        help="Skip the HSA-Dense end-to-end component (default: enabled in --bench_latex mode)")
    parser.add_argument("--dense_block_M", type=int, default=64,
                        help="block_M for the dense HSA kernel (default: 64)")
    parser.add_argument("--dense_num_threads", type=int, default=128,
                        help="num_threads for the dense HSA kernel (default: 128)")
    parser.add_argument("--with_single", action="store_true", default=False,
                        help="额外测试 HSA_single baseline（HQ:H < 16:1 时通过 HQ padding 到 16*H 实现）")
    parser.add_argument("--only_single", action="store_true", default=False,
                        help="只测试 HSA_single baseline，跳过 FA3/TopK/HSA group（会自动启用 --with_single）")
    parser.add_argument("--hsa_single_configs", type=str, default=None,
                        help="HSA single kernel 配置（可选），格式: 'seq_len:G:nt_fwd:nt_bwd,...'，"
                             "未指定则使用 kernel 默认线程数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_mask_last_token:
        args.mask_last_token = False

    if args.bench_latex:
        bench_latex_table(args)
    elif args.bench_topk:
        bench_topk(args)
    elif args.sweep_seq_len:
        bench_sweep_seq_len(args)
    elif args.candidates:
        bench_candidates(args)
    elif args.config:
        bench_fixed_configs(args)
    else:
        bench_hsa_latency(args)

# python code_exp/bench_hsa_latency.py

# pkill -f "burner.*--gpu 7"
# # 8k
# python code_exp/bench_hsa_latency.py --config "16:4:128:128,8:8:128:128,4:4:64:64,2:8:64:64,1:16:64:64" --with_baseline --with_single --real_indices /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup/indices_8192.pt 

# # 16k
# python code_exp/bench_hsa_latency.py --config "16:4:128:128,8:8:128:128,4:4:64:64,2:8:64:64,1:16:128:64" --with_baseline --real_indices /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup/indices_16384.pt

# # 32k
# python code_exp/bench_hsa_latency.py --config "16:4:128:128,8:8:128:128,4:4:64:64,2:8:64:64,1:16:128:64" --with_baseline --real_indices /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup/indices_32768.pt

# # 64k
# python code_exp/bench_hsa_latency.py --config "16:4:128:128,8:8:128:128,4:4:64:64,2:8:64:64,1:16:128:64" --with_baseline --real_indices /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup/indices_65536.pt

# # 128k
# python code_exp/bench_hsa_latency.py --config "16:4:128:128,8:8:128:128,4:4:64:64,2:8:64:64,1:16:128:64" --with_baseline --real_indices /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup/indices_131072.pt
