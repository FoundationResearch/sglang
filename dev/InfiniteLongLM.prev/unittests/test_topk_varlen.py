"""
Test TopK varlen (sequence packing) support.

目标：让 online_topk_group 在打包场景（多个样本拼接成 B=1）下，
与逐样本独立计算的结果一致。

本文件包含：
1. 可视化 prepare_fa_kwargs_from_position_ids 生成 cu_seq_lens 的过程
1.5 extract_lmk_from_packed_hsa_k: 从打包的 hsa_k_norm 中提取 lmk_k 并生成 cu_seq_lens_k
2. ref_topk_no_packing: 逐样本独立计算 topk（参考基准）
3. ref_topk_varlen: 打包后 B=1 计算 topk（torch 纯实现）
4. 测试入口
"""

import os
import sys
import math
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ============================================================================
# 1. 可视化 prepare_fa_kwargs_from_position_ids
# ============================================================================

def prepare_fa_kwargs_from_position_ids(position_ids):
    """
    从 position_ids 推导出 cu_seq_lens（累积序列长度），
    用于 flash_attn_varlen_func 的 varlen 模式。
    
    核心原理：position_ids 中每当值回退到 0，就说明一个新的子序列开始了。
    通过找到所有 position_ids == 0 的位置，就能得到每个子序列的起始位置。
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.view(-1)
    # 找到所有 position_id == 0 的位置，即每个子序列的起始位置
    indices_q = (position_ids == 0).nonzero().view(-1)

    # cu_seq_lens = [start_0, start_1, ..., start_n, total_len]
    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )
    cu_seq_lens_k = cu_seq_lens_q

    max_length_q = cu_seq_lens_q.diff().max()
    max_length_q = max_length_q.item()
    max_length_k = max_length_q

    return (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k)


def visualize_cu_seq_lens():
    """
    可视化 prepare_fa_kwargs_from_position_ids 的工作过程。
    模拟 3 个长度不同的序列打包成一个 B=1 的序列。
    """
    print("=" * 80)
    print("可视化 prepare_fa_kwargs_from_position_ids 生成 cu_seq_lens 的过程")
    print("=" * 80)

    # 模拟 3 个子序列，长度分别为 5, 3, 4
    seq_lens = [5, 3, 4]
    print(f"\n子序列长度: {seq_lens}")
    print(f"总长度: {sum(seq_lens)}")

    # 构造 position_ids: 每个子序列从 0 开始递增
    pos_parts = [torch.arange(l) for l in seq_lens]
    packed_pos = torch.cat(pos_parts).unsqueeze(0)  # (1, total_len)

    print(f"\nposition_ids (shape={packed_pos.shape}):")
    print(f"  {packed_pos[0].tolist()}")
    print(f"  即: [0,1,2,3,4 | 0,1,2 | 0,1,2,3]")
    print(f"  每次回退到0表示一个新子序列的开始")

    # 调用函数
    (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(packed_pos)

    print(f"\n--- 生成结果 ---")
    print(f"cu_seq_lens_q = {cu_q.tolist()}")
    print(f"  含义: 子序列0从idx={cu_q[0].item()}开始, 子序列1从idx={cu_q[1].item()}开始, "
          f"子序列2从idx={cu_q[2].item()}开始, 总长={cu_q[3].item()}")
    print(f"  各子序列长度 = cu_seq_lens.diff() = {cu_q.diff().tolist()}")
    print(f"max_length_q = {max_q} (batch中最长的子序列长度)")

    # 更复杂的例子：模拟实际 SFT 场景
    print(f"\n{'=' * 80}")
    print("模拟实际 SFT 打包场景 (chunk_size=64)")
    print("=" * 80)

    chunk_size = 64
    seg = chunk_size - 1  # 63

    # 假设 4 个 SFT 样本，原始长度分别为：
    orig_lens = [200, 150, 100, 180]
    # 对齐到 seg 的整数倍（以便 lmk 插入后 chunk 对齐）
    aligned_lens = [((l + seg - 1) // seg) * seg for l in orig_lens]
    print(f"\n原始样本长度:    {orig_lens}")
    print(f"对齐到seg={seg}后: {aligned_lens}")
    print(f"各样本chunk数:    {[l // seg for l in aligned_lens]}")
    print(f"总token数:        {sum(aligned_lens)}")

    pos_parts = [torch.arange(l) for l in aligned_lens]
    packed_pos = torch.cat(pos_parts).unsqueeze(0)

    (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(packed_pos)

    print(f"\ncu_seq_lens_q = {cu_q.tolist()}")
    print(f"各子序列长度  = {cu_q.diff().tolist()}")
    print(f"max_length_q  = {max_q}")

    # 展示 position_ids 的前几个和后几个值
    pos_flat = packed_pos[0]
    print(f"\nposition_ids 前20个: {pos_flat[:20].tolist()}")
    print(f"position_ids 在第一个边界处 [{aligned_lens[0]-3}:{aligned_lens[0]+3}]: "
          f"{pos_flat[aligned_lens[0]-3:aligned_lens[0]+3].tolist()}")
    print(f"  -> 注意: ...{aligned_lens[0]-1} 然后回退到 0,1,...  表示新序列开始")

    # 展示 cu_seq_lens 如何划分 lmk_k
    total_after_lmk = sum(l // seg * chunk_size for l in aligned_lens)
    print(f"\n如果插入lmk后(每{seg}个token插1个lmk):")
    for i, al in enumerate(aligned_lens):
        num_chunks = al // seg
        after_lmk_len = num_chunks * chunk_size
        print(f"  样本{i}: {al}个token -> {num_chunks}个chunk -> 插入lmk后{after_lmk_len}个token, "
              f"lmk_k有{num_chunks}个")


# ============================================================================
# 1.5 从打包的 hsa_k_norm 中提取 lmk_k 并生成 cu_seq_lens_k
# ============================================================================

def extract_lmk_from_packed_hsa_k(hsa_k_norm, cu_seq_lens_q, chunk_size):
    """
    从打包的 hsa_k_norm（含 lmk token 的完整 key 序列）中，
    根据 cu_seq_lens_q（token 级别的子序列边界）提取 lmk keys，
    并生成 lmk 级别的 cu_seq_lens_k。

    背景：
        在 LandmarkHSA 中，每个 chunk 的最后一个 token 是 lmk token。
        预训练时直接用全局步长 hsa_k_norm[:, chunk_size-1::chunk_size, :, :]
        即可提取。但在序列打包场景下，多个子序列拼在一起，需要确保：
        1) 每个子序列的 lmk 从自身的 chunk 边界提取，不会跨子序列
        2) 生成 lmk 维度的 cu_seq_lens_k 供 topk kernel 使用

    支持两种情况：
        a) 每个子序列长度是 chunk_size 的整数倍（数据侧已对齐到 seg=chunk_size-1）
        b) 每个子序列长度不是 chunk_size 的整数倍（末尾有 remainder 尾巴，没有 lmk）
           此时只取前 L_i // chunk_size 个完整 chunk 的 lmk，忽略尾巴部分

    Args:
        hsa_k_norm: [B, L_total, h_kv, D]
            打包后的完整 key 序列（包含 lmk token），B 通常为 1。
            L_total = sum(L_i)。
        cu_seq_lens_q: [num_seqs + 1], int32
            token 级别的子序列边界，如 [0, 128, 320, 384]。
        chunk_size: int
            每个 chunk 的大小（包含 lmk token），如 64。

    Returns:
        lmk_k: [B, S_total, h_kv, D]
            提取出的所有 lmk keys。S_total = sum(S_i)，其中 S_i = L_i // chunk_size。
        cu_seq_lens_k: [num_seqs + 1], int32
            lmk 级别的子序列边界，如 [0, 2, 5, 6]。

    Example (对齐):
        >>> cu_seq_lens_q = [0, 128, 320, 384]   # 每段都是 64 的整数倍
        >>> lmk_k, cu_seq_lens_k = extract_lmk_from_packed_hsa_k(hsa_k_norm, cu_seq_lens_q, 64)
        >>> # cu_seq_lens_k = [0, 2, 5, 6], lmk_k.shape[1] = 6

    Example (非对齐):
        >>> cu_seq_lens_q = [0, 203, 406, 469]   # 203=3*64+11, 不是 64 的整数倍
        >>> lmk_k, cu_seq_lens_k = extract_lmk_from_packed_hsa_k(hsa_k_norm, cu_seq_lens_q, 64)
        >>> # 每段有 3, 3, 0 个完整 chunk
        >>> # cu_seq_lens_k = [0, 3, 6, 6], lmk_k.shape[1] = 6
    """
    B = hsa_k_norm.shape[0]
    device = hsa_k_norm.device
    num_seqs = len(cu_seq_lens_q) - 1

    # Step 1: 计算每个子序列的长度和 lmk 数量（向下取整）
    sub_lengths = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]  # [num_seqs]
    sub_lmk_counts = sub_lengths // chunk_size  # [num_seqs], 每个子序列的完整 chunk 数 = lmk 数量

    # 检查是否有子序列不对齐（用于日志/debug）
    remainders = sub_lengths % chunk_size
    has_remainder = (remainders > 0).any().item()

    # Step 2: 构造 cu_seq_lens_k（lmk 级别的子序列边界）
    cu_seq_lens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_seq_lens_k[1:] = sub_lmk_counts.cumsum(0).to(torch.int32)

    S_total = cu_seq_lens_k[-1].item()

    # Step 3: 从 hsa_k_norm 中逐子序列提取 lmk keys
    # 必须用逐子序列提取（方式A），因为：
    #   - 非对齐时全局步长会采到跨子序列边界的错误位置
    #   - 即使对齐时也更安全

    lmk_parts = []
    for i in range(num_seqs):
        q_start = cu_seq_lens_q[i].item()
        q_end = cu_seq_lens_q[i + 1].item()
        S_i = sub_lmk_counts[i].item()

        if S_i == 0:
            # 子序列长度不足一个完整 chunk，没有 lmk
            continue

        sub_hsa_k = hsa_k_norm[:, q_start:q_end, :, :]  # [B, L_i, h_kv, D]
        # 只取前 S_i 个完整 chunk 的 lmk（忽略末尾 remainder）
        sub_lmk = sub_hsa_k[:, chunk_size - 1::chunk_size, :, :]  # [B, S_i (或 S_i), h_kv, D]
        # sub_lmk 的 shape[1] = L_i // chunk_size = S_i，刚好是我们要的
        assert sub_lmk.shape[1] == S_i, (
            f"子序列 {i}: 期望提取 {S_i} 个 lmk，实际得到 {sub_lmk.shape[1]}。"
            f"L_i={q_end - q_start}, chunk_size={chunk_size}"
        )
        lmk_parts.append(sub_lmk)

    if len(lmk_parts) > 0:
        lmk_k = torch.cat(lmk_parts, dim=1)  # [B, S_total, h_kv, D]
    else:
        # 极端情况：所有子序列都不足一个完整 chunk
        h_kv = hsa_k_norm.shape[2]
        D = hsa_k_norm.shape[3]
        lmk_k = hsa_k_norm.new_empty(B, 0, h_kv, D)

    # 可选验证：当所有子序列都对齐到 chunk_size 时，全局步长应该等价
    if not has_remainder:
        lmk_k_global = hsa_k_norm[:, chunk_size - 1::chunk_size, :, :]
        if lmk_k_global.shape[1] == S_total:
            assert torch.equal(lmk_k, lmk_k_global), (
                "全局步长采样与逐子序列采样结果不一致！"
                "这说明存在子序列边界未对齐到 chunk_size 的情况。"
            )

    return lmk_k, cu_seq_lens_k


def _visualize_one_case(sub_lens_after_lmk, chunk_size, case_name):
    """
    可视化单个测试场景的 extract_lmk_from_packed_hsa_k 工作过程。
    """
    h_kv = 2
    D = 16

    L_total = sum(sub_lens_after_lmk)
    print(f"\n--- {case_name} ---")
    print(f"参数: chunk_size={chunk_size}")
    print(f"子序列长度 (插入 lmk 后): {sub_lens_after_lmk}")
    for i, sub_len in enumerate(sub_lens_after_lmk):
        full_chunks = sub_len // chunk_size
        remainder = sub_len % chunk_size
        aligned = "✅ 对齐" if remainder == 0 else f"⚠️ 有余数 {remainder}"
        print(f"  样本{i}: L={sub_len}, 完整chunk数={full_chunks}, {aligned}")
    print(f"总长度: L_total = {L_total}")

    cu_seq_lens_q = torch.tensor(
        [0] + list(torch.tensor(sub_lens_after_lmk).cumsum(0).tolist()),
        dtype=torch.int32
    )
    print(f"cu_seq_lens_q = {cu_seq_lens_q.tolist()}")

    # 构造 hsa_k_norm，在 lmk 位置填充特殊值以便验证
    # 非 lmk 位置填充 -1 以区分
    hsa_k_norm = torch.full((1, L_total, h_kv, D), -1.0)
    lmk_counter = 0
    for i, sub_len in enumerate(sub_lens_after_lmk):
        start = cu_seq_lens_q[i].item()
        num_chunks = sub_len // chunk_size
        remainder = sub_len % chunk_size
        for c in range(num_chunks):
            lmk_pos = start + (c + 1) * chunk_size - 1
            hsa_k_norm[0, lmk_pos, :, :] = (i * 100 + c)
            lmk_counter += 1
            print(f"  样本{i} chunk{c}: lmk 位于全局位置 {lmk_pos}, 填充值={i * 100 + c}")
        if remainder > 0:
            tail_start = start + num_chunks * chunk_size
            tail_end = start + sub_len
            print(f"  样本{i} 尾巴: 全局位置 [{tail_start}, {tail_end}), "
                  f"长度={remainder}, 无 lmk")

    print(f"共 {lmk_counter} 个 lmk 位置")

    # 调用转换函数
    lmk_k, cu_seq_lens_k = extract_lmk_from_packed_hsa_k(
        hsa_k_norm, cu_seq_lens_q, chunk_size
    )

    print(f"\n转换结果:")
    print(f"  lmk_k shape: {lmk_k.shape}  (期望 [1, {lmk_counter}, {h_kv}, {D}])")
    print(f"  cu_seq_lens_k = {cu_seq_lens_k.tolist()}")
    for i in range(len(sub_lens_after_lmk)):
        n_lmk = (cu_seq_lens_k[i + 1] - cu_seq_lens_k[i]).item()
        print(f"    样本{i}: {n_lmk} 个 lmk")

    # 验证提取的值
    print(f"验证提取值:")
    all_correct = True
    for s in range(lmk_k.shape[1]):
        val = lmk_k[0, s, 0, 0].item()
        if val == -1.0:
            print(f"    lmk_k[{s}] 值=-1 ← ❌ 采到了非 lmk 位置！")
            all_correct = False
        else:
            sample_id = int(val) // 100
            chunk_id = int(val) % 100
            print(f"    lmk_k[{s}] 值={val:.0f} → 来自样本{sample_id} chunk{chunk_id}")

    # 验证全局步长等价性
    lmk_k_global = hsa_k_norm[:, chunk_size - 1::chunk_size, :, :]
    any_remainder = any(l % chunk_size != 0 for l in sub_lens_after_lmk)
    if not any_remainder and lmk_k_global.shape[1] == lmk_k.shape[1]:
        is_equal = torch.equal(lmk_k, lmk_k_global)
        print(f"全局步长采样等价性: {'✅ 一致' if is_equal else '❌ 不一致'}")
    else:
        # 非对齐时，全局步长采样可能采到跨边界的错误位置
        print(f"全局步长采样: 跳过验证（子序列非对齐时全局步长不安全）")
        if lmk_k_global.shape[1] > 0:
            # 展示全局步长采到了什么
            global_vals = [lmk_k_global[0, s, 0, 0].item()
                           for s in range(lmk_k_global.shape[1])]
            print(f"  全局步长采到的值: {global_vals}")
            print(f"  全局步长 lmk 数: {lmk_k_global.shape[1]} vs 逐子序列: {lmk_k.shape[1]}")

    if all_correct:
        print(f"✅ {case_name} 通过")
    else:
        print(f"❌ {case_name} 失败")

    return all_correct


def visualize_extract_lmk():
    """
    可视化 extract_lmk_from_packed_hsa_k 的工作过程，
    分别测试对齐和非对齐两种场景。
    """
    print("\n" + "=" * 80)
    print("可视化 extract_lmk_from_packed_hsa_k（对齐 + 非对齐场景）")
    print("=" * 80)

    all_pass = True

    # ====== Case 1: 对齐场景（每个子序列都是 chunk_size 的整数倍）======
    all_pass &= _visualize_one_case(
        sub_lens_after_lmk=[128, 192, 64],
        chunk_size=64,
        case_name="Case 1: 全部对齐 (128, 192, 64)"
    )

    # ====== Case 2: 非对齐场景 ======
    # 模拟原始序列 200, 150, 100 tokens，chunk_size=64, seg=63
    # insert_special_tokens 后:
    #   200 // 63 = 3 full chunks, remainder = 200 % 63 = 11
    #     → 3*64 + 11 = 203
    #   150 // 63 = 2 full chunks, remainder = 150 % 63 = 24
    #     → 2*64 + 24 = 152
    #   100 // 63 = 1 full chunk,  remainder = 100 % 63 = 37
    #     → 1*64 + 37 = 101
    all_pass &= _visualize_one_case(
        sub_lens_after_lmk=[203, 152, 101],
        chunk_size=64,
        case_name="Case 2: 非对齐 (203=3*64+11, 152=2*64+24, 101=1*64+37)"
    )

    # ====== Case 3: 混合场景（部分对齐 + 部分非对齐）======
    all_pass &= _visualize_one_case(
        sub_lens_after_lmk=[128, 203, 64, 101],
        chunk_size=64,
        case_name="Case 3: 混合 (128✅, 203⚠️, 64✅, 101⚠️)"
    )

    # ====== Case 4: 极端场景（子序列长度不足一个 chunk）======
    all_pass &= _visualize_one_case(
        sub_lens_after_lmk=[30, 128, 10],
        chunk_size=64,
        case_name="Case 4: 极短子序列 (30<64, 128, 10<64)"
    )

    # ====== Case 5: 全部不足一个 chunk ======
    all_pass &= _visualize_one_case(
        sub_lens_after_lmk=[10, 20, 30],
        chunk_size=64,
        case_name="Case 5: 全部不足一个 chunk (10, 20, 30)"
    )

    print("\n" + "=" * 80)
    if all_pass:
        print("🎉 extract_lmk_from_packed_hsa_k 所有场景测试通过！")
    else:
        print("⚠️  部分场景失败，请检查！")
    print("=" * 80)


# ============================================================================
# 2. ref_topk_no_packing: 逐样本独立计算 topk（参考基准）
# ============================================================================

def ref_topk_single_sample(q, lmks, topk, block_size, window_size, is_causal,
                           q_offset=0, drop_mask=None, force_recent_chunks=0):
    """
    单个样本的 TopK 参考实现（纯 PyTorch，无 Triton kernel）。

    Args:
        q:     [1, L, h_q, D]  单个样本的 query
        lmks:  [1, S, h_kv, D] 单个样本的 landmark keys
        topk:  int
        block_size: int (= chunk_size)
        window_size: int (= hsa_sliding_window)
        is_causal: bool

        q_offset: int, optional
        drop_mask: [1, L, S] int32, optional
        force_recent_chunks: int, optional
            强制保留最近 N 个 chunk 的数量，这些 chunk 不参与竞争 topk，
            而是直接占据 topk 的最后 N 个位置。

    Returns:
        indices: [1, L, h_shared, topk]
        scores:  [1, L, h_shared, topk]
    """
    B, L, h_q, D = q.shape
    _, S, h_kv, D2 = lmks.shape
    assert B == 1, "ref_topk_single_sample 只处理单个样本"
    sm_scale = 1.0 / math.sqrt(D)

    # 处理 D 维度不等的情况
    if D != D2:
        assert D2 % D == 0
        d_ratio = D2 // D
        lmks = lmks.reshape(B, S, h_kv * d_ratio, D)
        h_kv = h_kv * d_ratio

    h_shared = min(h_q, h_kv)

    # 自动判断 group 方向
    if h_q >= h_kv:
        G = h_q // h_kv
        q_group_sum = q.view(B, L, h_kv, G, D).sum(dim=3)  # [B, L, h_kv, D]
        scores_ref = torch.einsum("blkd,bskd->blks", q_group_sum.float(), lmks.float()) * sm_scale
    else:
        G = h_kv // h_q
        lmk_group_sum = lmks.view(B, S, h_q, G, D).sum(dim=3)
        scores_ref = torch.einsum("blkd,bskd->blks", q.float(), lmk_group_sum.float()) * sm_scale

    effective_topk = min(topk - force_recent_chunks, S)

    # Causal mask: 每个 query 位置只能看到 window_size 之前的 chunk
    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1)
        i_idx_global = i_idx + q_offset
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)

        # chunk j 对 query i 可见的条件: j < (i_global - window_size + 1) / block_size
        limit_chunk_idx = (i_idx_global - window_size + 1).div(block_size, rounding_mode='floor')
        mask = j_idx >= limit_chunk_idx  # True = 不可见

        if force_recent_chunks > 0:
            # recent 区间: [recent_start, limit_chunk)，从竞争部分排除
            # 与 kernel 对齐: recent_start = max(limit_chunk - force_recent_chunks, 0)
            recent_start_idx = torch.clamp(limit_chunk_idx - force_recent_chunks, min=0)  # [L, 1]
            recent_mask = (j_idx >= recent_start_idx) & (j_idx < limit_chunk_idx)  # [L, S]
            compete_mask = mask | recent_mask
        else:
            compete_mask = mask

        scores_compete = scores_ref.masked_fill(compete_mask.unsqueeze(0).unsqueeze(2), float('-inf'))
    else:
        scores_compete = scores_ref.clone()

    if drop_mask is not None:
        drop_bool = drop_mask.bool()
        scores_compete = scores_compete.masked_fill(drop_bool.unsqueeze(2), float('-inf'))

    # 竞争部分 TopK selection
    scores_topk, indices_topk = torch.topk(scores_compete, k=effective_topk, dim=-1, sorted=False)

    if force_recent_chunks > 0 and is_causal:
        # 收集 recent chunk 的真实 score
        recent_offsets = torch.arange(force_recent_chunks, device=q.device)  # [F]
        recent_indices_per_q = recent_start_idx.squeeze(1).unsqueeze(-1) + recent_offsets  # [L, F]
        recent_indices_per_q = recent_indices_per_q.long()

        # 标记无效的 recent index（< 0 或 >= S）
        invalid_recent = (recent_indices_per_q < 0) | (recent_indices_per_q >= S)  # [L, F]

        ri_expanded = recent_indices_per_q.unsqueeze(0).unsqueeze(2).expand(
            1, L, h_shared, force_recent_chunks).clone()
        ri_safe = ri_expanded.clamp(0, max(S - 1, 0))
        recent_scores = torch.gather(scores_ref, dim=-1, index=ri_safe)  # [1, L, h_shared, F]

        invalid_recent_expanded = invalid_recent.unsqueeze(0).unsqueeze(2).expand(
            1, L, h_shared, force_recent_chunks)
        recent_scores[invalid_recent_expanded] = float('-inf')
        ri_expanded[invalid_recent_expanded] = -1

        # 拼接: [竞争部分 | recent 部分]
        indices_topk = torch.cat([indices_topk, ri_expanded], dim=-1)
        scores_topk = torch.cat([scores_topk, recent_scores], dim=-1)

    # 按 index 排序（与 kernel 行为一致）
    is_invalid = scores_topk == float('-inf')
    sort_keys = indices_topk.clone()
    sort_keys[is_invalid] = S + 1

    _, order = torch.sort(sort_keys, dim=-1)
    indices_sorted = torch.gather(indices_topk, -1, order)
    scores_sorted = torch.gather(scores_topk, -1, order)
    indices_sorted[scores_sorted == float('-inf')] = -1

    return indices_sorted, scores_sorted


def ref_topk_no_packing(q_list, lmks_list, topk, block_size, window_size,
                         is_causal, force_recent_chunks=0):
    """
    逐样本独立计算 TopK（不打包的参考基准）。

    Args:
        q_list:    list of [1, L_i, h_q, D]   每个样本的 query
        lmks_list: list of [1, S_i, h_kv, D]  每个样本的 lmk keys
        force_recent_chunks: int, optional
        其他参数同 online_topk_group

    Returns:
        indices_list: list of [1, L_i, h_shared, topk]
        scores_list:  list of [1, L_i, h_shared, topk]
    """
    indices_list = []
    scores_list = []

    for q_i, lmks_i in zip(q_list, lmks_list):
        idx_i, sc_i = ref_topk_single_sample(
            q_i, lmks_i, topk, block_size, window_size,
            is_causal=is_causal,
            q_offset=0,  # 每个样本独立，offset=0
            force_recent_chunks=force_recent_chunks,
        )
        indices_list.append(idx_i)
        scores_list.append(sc_i)

    return indices_list, scores_list


# ============================================================================
# 3. ref_topk_varlen: 打包后 B=1 计算 topk（torch 纯实现）
# ============================================================================

def ref_topk_varlen(q_list, lmks_list, topk, block_size, window_size,
                    is_causal, force_recent_chunks=0):
    """
    将多个样本打包成 B=1，然后利用 cu_seqlens 划分子序列边界，
    确保每个子序列的 query 只能检索自己对应的 lmk chunks。

    打包后的布局:
        packed_q    = [1, sum(L_i), h_q, D]
        packed_lmks = [1, sum(S_i), h_kv, D]
        cu_seqlens_q = [0, L_0, L_0+L_1, ..., sum(L_i)]
        cu_seqlens_k = [0, S_0, S_0+S_1, ..., sum(S_i)]

    计算方式:
        对每个子序列 i，提取 packed_q[cu_q[i]:cu_q[i+1]] 和
        packed_lmks[cu_k[i]:cu_k[i+1]]，独立计算 topk。
        最终将各子序列结果拼接回去。

    Args:
        q_list:    list of [1, L_i, h_q, D]
        lmks_list: list of [1, S_i, h_kv, D]
        force_recent_chunks: int, optional

    Returns:
        packed_indices: [1, sum(L_i), h_shared, topk]  (indices 相对于各子序列自身的 lmks)
        packed_scores:  [1, sum(L_i), h_shared, topk]
        cu_seqlens_q:   [num_seqs + 1]  query 子序列边界
        cu_seqlens_k:   [num_seqs + 1]  lmk key 子序列边界
    """
    num_seqs = len(q_list)
    device = q_list[0].device

    # --- Step 1: 打包 (沿 L/S 维度拼接) ---
    packed_q = torch.cat([q_i.squeeze(0) for q_i in q_list], dim=0)      # [sum(L_i), h_q, D]
    packed_lmks = torch.cat([l_i.squeeze(0) for l_i in lmks_list], dim=0)  # [sum(S_i), h_kv, D]

    # 构造 cu_seqlens
    q_lens = [q_i.shape[1] for q_i in q_list]
    k_lens = [l_i.shape[1] for l_i in lmks_list]

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_lens[i]
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + k_lens[i]

    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    _, h_q, D = packed_q.shape
    _, h_kv, D2 = packed_lmks.shape

    # 处理 D 维度不等的情况
    if D != D2:
        assert D2 % D == 0
        d_ratio = D2 // D
        packed_lmks = packed_lmks.reshape(total_k, h_kv * d_ratio, D)
        h_kv = h_kv * d_ratio

    h_shared = min(h_q, h_kv)
    sm_scale = 1.0 / math.sqrt(D)

    # --- Step 2: 逐子序列计算 scores 和 topk ---
    # 与直接逐样本计算不同的是，这里从打包后的连续张量中切片
    all_indices = []
    all_scores = []

    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        k_start = cu_seqlens_k[seq_idx].item()
        k_end = cu_seqlens_k[seq_idx + 1].item()

        L_i = q_end - q_start
        S_i = k_end - k_start

        q_i = packed_q[q_start:q_end]        # [L_i, h_q, D]
        lmks_i = packed_lmks[k_start:k_end]  # [S_i, h_kv, D]

        # 计算 scores: 先 group sum，再 einsum
        if h_q >= h_kv:
            G = h_q // h_kv
            q_group_sum = q_i.view(L_i, h_kv, G, D).sum(dim=2)  # [L_i, h_kv, D]
            scores = torch.einsum("lkd,skd->lks", q_group_sum.float(), lmks_i.float()) * sm_scale
        else:
            G = h_kv // h_q
            lmk_group_sum = lmks_i.view(S_i, h_q, G, D).sum(dim=2)  # [S_i, h_q, D]
            scores = torch.einsum("lkd,skd->lks", q_i.float(), lmk_group_sum.float()) * sm_scale

        # scores: [L_i, h_shared, S_i]

        effective_topk = min(topk - force_recent_chunks, S_i)

        # Causal mask
        if is_causal:
            i_idx = torch.arange(L_i, device=device).unsqueeze(1)     # [L_i, 1]
            j_idx = torch.arange(S_i, device=device).unsqueeze(0)     # [1, S_i]

            # q_offset=0 因为每个子序列独立
            limit_chunk_idx = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
            mask = j_idx >= limit_chunk_idx  # True = 不可见

            if force_recent_chunks > 0:
                # recent 区间: [recent_start, limit_chunk)，从竞争部分排除
                # 与 kernel 对齐: recent_start = max(limit_chunk - force_recent_chunks, 0)
                recent_start_idx = torch.clamp(limit_chunk_idx - force_recent_chunks, min=0)  # [L_i, 1]
                recent_mask = (j_idx >= recent_start_idx) & (j_idx < limit_chunk_idx)  # [L_i, S_i]
                compete_mask = mask | recent_mask
            else:
                compete_mask = mask

            # scores: [L_i, h_shared, S_i], mask: [L_i, S_i] -> [L_i, 1, S_i]
            scores_compete = scores.masked_fill(compete_mask.unsqueeze(1), float('-inf'))
        else:
            scores_compete = scores.clone()

        # 竞争部分 TopK
        scores_topk, indices_topk = torch.topk(scores_compete, k=effective_topk, dim=-1, sorted=False)
        # scores_topk, indices_topk: [L_i, h_shared, effective_topk]

        if force_recent_chunks > 0 and is_causal:
            # 收集 recent chunk 的真实 score
            recent_offsets = torch.arange(force_recent_chunks, device=device)  # [F]
            recent_indices_per_q = recent_start_idx.squeeze(1).unsqueeze(-1) + recent_offsets  # [L_i, F]
            recent_indices_per_q = recent_indices_per_q.long()

            # 标记无效的 recent index（< 0 或 >= S_i）
            invalid_recent = (recent_indices_per_q < 0) | (recent_indices_per_q >= S_i)  # [L_i, F]

            ri_expanded = recent_indices_per_q.unsqueeze(1).expand(
                L_i, h_shared, force_recent_chunks).clone()
            ri_safe = ri_expanded.clamp(0, max(S_i - 1, 0))
            recent_scores = torch.gather(scores, dim=-1, index=ri_safe)  # [L_i, h_shared, F]

            invalid_recent_expanded = invalid_recent.unsqueeze(1).expand(
                L_i, h_shared, force_recent_chunks)
            recent_scores[invalid_recent_expanded] = float('-inf')
            ri_expanded[invalid_recent_expanded] = -1

            # 拼接: [竞争部分 | recent 部分]
            indices_topk = torch.cat([indices_topk, ri_expanded], dim=-1)
            scores_topk = torch.cat([scores_topk, recent_scores], dim=-1)

        # 排序（与 kernel 行为一致：按 index 升序，invalid 放最后）
        is_invalid = scores_topk == float('-inf')
        sort_keys = indices_topk.clone()
        sort_keys[is_invalid] = S_i + 1

        _, order = torch.sort(sort_keys, dim=-1)
        indices_sorted = torch.gather(indices_topk, -1, order)
        scores_sorted = torch.gather(scores_topk, -1, order)
        indices_sorted[scores_sorted == float('-inf')] = -1

        all_indices.append(indices_sorted)
        all_scores.append(scores_sorted)

    # --- Step 3: 拼接结果 ---
    packed_indices = torch.cat(all_indices, dim=0).unsqueeze(0)  # [1, sum(L_i), h_shared, topk]
    packed_scores = torch.cat(all_scores, dim=0).unsqueeze(0)    # [1, sum(L_i), h_shared, topk]

    return packed_indices, packed_scores, cu_seqlens_q, cu_seqlens_k


# ============================================================================
# 4. 测试入口
# ============================================================================

def test_visualize_cu_seq_lens():
    """运行可视化，帮助理解 cu_seq_lens 生成过程。"""
    visualize_cu_seq_lens()


def test_ref_topk_no_packing():
    """
    验证 ref_topk_no_packing 的正确性：
    生成几个独立样本，逐个计算 topk，确认结果形状和基本语义正确。
    """
    print("\n" + "=" * 80)
    print("测试 ref_topk_no_packing 逐样本独立计算 TopK")
    print("=" * 80)

    device = "cpu"
    dtype = torch.float32
    torch.manual_seed(42)

    # 参数
    h_q = 8
    h_kv = 2
    D = 16
    topk = 4
    block_size = 64
    window_size = 128
    is_causal = True

    # 3 个样本，长度不同
    # 注意：L 必须 > window_size 才会有 chunk 可见
    sample_configs = [
        {"L": 512, "S": 8},   # 512个token, 8个chunk
        {"L": 384, "S": 6},   # 384个token, 6个chunk
        {"L": 640, "S": 10},  # 640个token, 10个chunk
    ]

    q_list = []
    lmks_list = []

    for i, cfg in enumerate(sample_configs):
        L, S = cfg["L"], cfg["S"]
        q_i = torch.randn(1, L, h_q, D, dtype=dtype, device=device)
        lmks_i = torch.randn(1, S, h_kv, D, dtype=dtype, device=device)
        q_list.append(q_i)
        lmks_list.append(lmks_i)
        print(f"  样本{i}: q=[1, {L}, {h_q}, {D}], lmks=[1, {S}, {h_kv}, {D}]")

    indices_list, scores_list = ref_topk_no_packing(
        q_list, lmks_list, topk, block_size, window_size,
        is_causal=is_causal,
    )

    h_shared = min(h_q, h_kv)
    for i, (idx, sc) in enumerate(zip(indices_list, scores_list)):
        L = sample_configs[i]["L"]
        S = sample_configs[i]["S"]
        print(f"\n  样本{i} 结果:")
        print(f"    indices shape: {idx.shape}  (期望 [1, {L}, {h_shared}, {topk}])")
        print(f"    scores  shape: {sc.shape}")
        # 打印几个位置的 topk 结果
        for pos in [0, window_size, L - 1]:
            if pos < L:
                print(f"    位置 {pos}: indices={idx[0, pos, 0, :].tolist()}, "
                      f"scores={sc[0, pos, 0, :].tolist()}")
        # 验证: 位置 0 ~ window_size-1 的 query 应该看不到任何 chunk（全是 -inf / -1）
        early_indices = idx[0, :min(window_size, L), 0, :]
        num_valid_early = (early_indices >= 0).sum().item()
        print(f"    前{min(window_size, L)}个位置中有效chunk数: {num_valid_early} "
              f"(前window_size个位置应该看不到chunk)")

    print("\n  ✅ ref_topk_no_packing 基本测试通过")


def test_topk_varlen_matches_no_packing():
    """
    测试 topk varlen (打包后 B=1 计算) 与 ref_topk_no_packing 逐样本计算的一致性。

    验证逻辑:
    1. 生成多个长度不同的样本
    2. 用 ref_topk_no_packing 逐样本独立计算 topk (基准)
    3. 用 ref_topk_varlen 将样本打包后计算 topk
    4. 比较两者的 indices 和 scores 是否完全一致
    """
    print("\n" + "=" * 80)
    print("测试 ref_topk_varlen 打包计算 vs ref_topk_no_packing 逐样本计算")
    print("=" * 80)

    device = "cpu"
    dtype = torch.float32
    torch.manual_seed(42)

    # ============ 测试配置 ============
    test_cases = [
        {
            "name": "基础测试: 3个等长样本",
            "h_q": 8, "h_kv": 2, "D": 16,
            "topk": 4, "block_size": 64, "window_size": 128,
            "is_causal": True,
            "samples": [
                {"L": 512, "S": 8},
                {"L": 512, "S": 8},
                {"L": 512, "S": 8},
            ],
        },
        {
            "name": "变长测试: 3个不等长样本",
            "h_q": 8, "h_kv": 2, "D": 16,
            "topk": 4, "block_size": 64, "window_size": 128,
            "is_causal": True,
            "samples": [
                {"L": 512, "S": 8},
                {"L": 384, "S": 6},
                {"L": 640, "S": 10},
            ],
        },
        {
            "name": "小 window: 有更多可见 chunks",
            "h_q": 4, "h_kv": 4, "D": 32,
            "topk": 3, "block_size": 32, "window_size": 64,
            "is_causal": True,
            "samples": [
                {"L": 256, "S": 8},
                {"L": 128, "S": 4},
                {"L": 384, "S": 12},
            ],
        },
        {
            "name": "非因果: 无 causal mask",
            "h_q": 8, "h_kv": 2, "D": 16,
            "topk": 4, "block_size": 64, "window_size": 128,
            "is_causal": False,
            "samples": [
                {"L": 256, "S": 4},
                {"L": 384, "S": 6},
            ],
        },
        {
            "name": "单个样本 (退化情况)",
            "h_q": 8, "h_kv": 2, "D": 16,
            "topk": 4, "block_size": 64, "window_size": 128,
            "is_causal": True,
            "samples": [
                {"L": 512, "S": 8},
            ],
        },
        {
            "name": "h_kv > h_q (lmk group sum)",
            "h_q": 2, "h_kv": 8, "D": 16,
            "topk": 4, "block_size": 64, "window_size": 128,
            "is_causal": True,
            "samples": [
                {"L": 512, "S": 8},
                {"L": 384, "S": 6},
            ],
        },
    ]

    all_pass = True
    for tc in test_cases:
        print(f"\n  --- {tc['name']} ---")
        h_q, h_kv, D = tc["h_q"], tc["h_kv"], tc["D"]
        topk = tc["topk"]
        block_size = tc["block_size"]
        window_size = tc["window_size"]
        is_causal = tc["is_causal"]

        q_list = []
        lmks_list = []
        for cfg in tc["samples"]:
            L, S = cfg["L"], cfg["S"]
            q_i = torch.randn(1, L, h_q, D, dtype=dtype, device=device)
            lmks_i = torch.randn(1, S, h_kv, D, dtype=dtype, device=device)
            q_list.append(q_i)
            lmks_list.append(lmks_i)

        # 方式1: 逐样本独立计算
        indices_ref_list, scores_ref_list = ref_topk_no_packing(
            q_list, lmks_list, topk, block_size, window_size,
            is_causal=is_causal,
        )

        # 方式2: 打包后 varlen 计算
        packed_indices, packed_scores, cu_q, cu_k = ref_topk_varlen(
            q_list, lmks_list, topk, block_size, window_size,
            is_causal=is_causal,
        )

        # 对比每个子序列的结果
        h_shared = min(h_q, h_kv)
        case_pass = True
        for i, (idx_ref, sc_ref) in enumerate(zip(indices_ref_list, scores_ref_list)):
            L_i = tc["samples"][i]["L"]
            q_start = cu_q[i].item()
            q_end = cu_q[i + 1].item()

            # 从打包结果中提取子序列 i 的结果
            idx_varlen = packed_indices[0, q_start:q_end, :, :]   # [L_i, h_shared, topk]
            sc_varlen = packed_scores[0, q_start:q_end, :, :]     # [L_i, h_shared, topk]

            idx_ref_sq = idx_ref.squeeze(0)   # [L_i, h_shared, topk]
            sc_ref_sq = sc_ref.squeeze(0)     # [L_i, h_shared, topk]

            # 比较 indices
            idx_match = torch.equal(idx_varlen, idx_ref_sq)
            # 比较 scores (用 allclose 处理浮点误差)
            # 对 -inf 位置特殊处理
            valid_mask = sc_ref_sq != float('-inf')
            if valid_mask.any():
                sc_match = torch.allclose(
                    sc_varlen[valid_mask], sc_ref_sq[valid_mask],
                    atol=1e-6, rtol=1e-5
                )
            else:
                sc_match = True
            # -inf 位置也要一致
            inf_match = torch.equal(
                (sc_varlen == float('-inf')),
                (sc_ref_sq == float('-inf'))
            )

            if not (idx_match and sc_match and inf_match):
                case_pass = False
                print(f"    ❌ 样本{i} 不一致!")
                if not idx_match:
                    # 找到第一个不同的位置
                    diff_pos = (idx_varlen != idx_ref_sq).nonzero()
                    if len(diff_pos) > 0:
                        pos = diff_pos[0]
                        print(f"       indices 首个差异位置: {pos.tolist()}")
                        print(f"         varlen: {idx_varlen[pos[0], pos[1], :].tolist()}")
                        print(f"         ref:    {idx_ref_sq[pos[0], pos[1], :].tolist()}")
                if not sc_match:
                    print(f"       scores 不匹配 (valid 区域)")
                if not inf_match:
                    print(f"       -inf 位置不匹配")
            else:
                print(f"    ✅ 样本{i} 一致 (L={L_i}, indices & scores 完全匹配)")

        if case_pass:
            print(f"  ✅ 测试通过: {tc['name']}")
        else:
            print(f"  ❌ 测试失败: {tc['name']}")
            all_pass = False

    print("\n" + "=" * 80)
    if all_pass:
        print("🎉 所有测试通过! ref_topk_varlen 与 ref_topk_no_packing 结果完全一致")
    else:
        print("⚠️  部分测试失败，请检查!")
    print("=" * 80)


def test_extract_lmk_non_aligned():
    """
    专项测试: extract_lmk_from_packed_hsa_k 在非对齐场景下的正确性。
    
    核心验证：
    1. 非对齐时逐子序列提取仍然只取到正确的 lmk 位置
    2. 非对齐时全局步长会采到错误位置（验证二者不等价）
    3. cu_seq_lens_k 正确反映每个子序列的 lmk 数量
    4. 尾巴中的 token 不会被错误提取为 lmk
    """
    print("\n" + "=" * 80)
    print("专项测试: extract_lmk_from_packed_hsa_k 非对齐场景")
    print("=" * 80)

    chunk_size = 64
    h_kv = 2
    D = 16
    all_pass = True

    # ---- 测试 1: 基本非对齐 ----
    print("\n  --- 测试1: 基本非对齐 ---")
    # 原始序列 200 tokens → insert_special_tokens 后 203 = 3*64+11
    # 原始序列 150 tokens → insert_special_tokens 后 152 = 2*64+24
    sub_lens = [203, 152]
    L_total = sum(sub_lens)
    cu_seq_lens_q = torch.tensor([0, 203, 355], dtype=torch.int32)

    # 构造 hsa_k_norm: lmk 位置填 1.0，其他填 0.0
    hsa_k_norm = torch.zeros(1, L_total, h_kv, D)
    for i, sub_len in enumerate(sub_lens):
        start = cu_seq_lens_q[i].item()
        num_chunks = sub_len // chunk_size
        for c in range(num_chunks):
            lmk_pos = start + (c + 1) * chunk_size - 1
            hsa_k_norm[0, lmk_pos, :, :] = 1.0  # lmk 标记

    lmk_k, cu_seq_lens_k = extract_lmk_from_packed_hsa_k(
        hsa_k_norm, cu_seq_lens_q, chunk_size
    )

    # 验证 cu_seq_lens_k
    expected_k = torch.tensor([0, 3, 5], dtype=torch.int32)  # 3个lmk + 2个lmk
    assert torch.equal(cu_seq_lens_k, expected_k), (
        f"cu_seq_lens_k 不正确: 期望 {expected_k.tolist()}, 得到 {cu_seq_lens_k.tolist()}"
    )
    print(f"    cu_seq_lens_k = {cu_seq_lens_k.tolist()} ✅")

    # 验证提取的全部是 lmk 位置（值应为 1.0）
    all_are_lmk = (lmk_k == 1.0).all().item()
    assert all_are_lmk, "提取到了非 lmk 位置的 token！"
    print(f"    所有提取的 token 都是 lmk 位置 ✅")

    # 验证全局步长会出错
    lmk_k_global = hsa_k_norm[:, chunk_size - 1::chunk_size, :, :]
    # 全局步长位置: 63, 127, 191, 255, 319
    # 其中 191 是样本0的 chunk2 lmk ✅
    # 但 255 = 203 + 52，这是样本1的第52个位置，不是 lmk！
    global_count = lmk_k_global.shape[1]
    local_count = lmk_k.shape[1]
    print(f"    全局步长采到 {global_count} 个 token，逐子序列提取 {local_count} 个")

    if global_count != local_count:
        print(f"    数量不同 → 全局步长不安全 ✅ (非对齐场景下必须用逐子序列提取)")
    else:
        is_equal = torch.equal(lmk_k, lmk_k_global)
        if not is_equal:
            # 找到不一致的位置
            global_vals_are_lmk = (lmk_k_global == 1.0).all(dim=-1).all(dim=-1)[0]  # [S]
            wrong_indices = (~global_vals_are_lmk).nonzero().view(-1).tolist()
            print(f"    全局步长在位置 {wrong_indices} 采到了非 lmk token ✅ (证明全局步长不安全)")
        else:
            print(f"    ⚠️ 全局步长恰好一致（偶然情况，不代表一般安全）")

    print(f"  ✅ 测试1 通过")

    # ---- 测试 2: 子序列长度不足一个 chunk ----
    print("\n  --- 测试2: 子序列长度不足一个 chunk ---")
    sub_lens = [30, 128, 10]  # 30 < 64, 10 < 64
    L_total = sum(sub_lens)
    cu_seq_lens_q = torch.tensor([0, 30, 158, 168], dtype=torch.int32)
    hsa_k_norm = torch.zeros(1, L_total, h_kv, D)
    # 只有样本1(128=2*64) 有 lmk
    for c in range(2):
        lmk_pos = 30 + (c + 1) * chunk_size - 1
        hsa_k_norm[0, lmk_pos, :, :] = 1.0

    lmk_k, cu_seq_lens_k = extract_lmk_from_packed_hsa_k(
        hsa_k_norm, cu_seq_lens_q, chunk_size
    )

    expected_k = torch.tensor([0, 0, 2, 2], dtype=torch.int32)  # 0, 2, 0 个 lmk
    assert torch.equal(cu_seq_lens_k, expected_k), (
        f"cu_seq_lens_k 不正确: 期望 {expected_k.tolist()}, 得到 {cu_seq_lens_k.tolist()}"
    )
    assert lmk_k.shape[1] == 2, f"期望 2 个 lmk，得到 {lmk_k.shape[1]}"
    assert (lmk_k == 1.0).all().item(), "提取到了非 lmk 位置的 token！"
    print(f"    cu_seq_lens_k = {cu_seq_lens_k.tolist()} ✅")
    print(f"    样本0(L=30): 0个lmk, 样本1(L=128): 2个lmk, 样本2(L=10): 0个lmk ✅")
    print(f"  ✅ 测试2 通过")

    # ---- 测试 3: 全部不足一个 chunk ----
    print("\n  --- 测试3: 全部不足一个 chunk ---")
    sub_lens = [10, 20, 30]
    L_total = sum(sub_lens)
    cu_seq_lens_q = torch.tensor([0, 10, 30, 60], dtype=torch.int32)
    hsa_k_norm = torch.zeros(1, L_total, h_kv, D)

    lmk_k, cu_seq_lens_k = extract_lmk_from_packed_hsa_k(
        hsa_k_norm, cu_seq_lens_q, chunk_size
    )

    expected_k = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    assert torch.equal(cu_seq_lens_k, expected_k), (
        f"cu_seq_lens_k 不正确: 期望 {expected_k.tolist()}, 得到 {cu_seq_lens_k.tolist()}"
    )
    assert lmk_k.shape[1] == 0, f"期望 0 个 lmk，得到 {lmk_k.shape[1]}"
    print(f"    cu_seq_lens_k = {cu_seq_lens_k.tolist()} ✅")
    print(f"    lmk_k.shape = {lmk_k.shape} (空张量) ✅")
    print(f"  ✅ 测试3 通过")

    # ---- 测试 4: 真实 SFT 场景模拟 ----
    print("\n  --- 测试4: 模拟真实 SFT 场景 ---")
    # 原始 SFT 样本长度: 200, 150, 63, 100
    # insert_special_tokens 后:
    #   200: 200//63=3 chunks, 200%63=11  → 3*64+11=203
    #   150: 150//63=2 chunks, 150%63=24  → 2*64+24=152
    #    63: 63//63=1 chunk,   63%63=0   → 1*64+0=64  (对齐!)
    #   100: 100//63=1 chunk,  100%63=37  → 1*64+37=101
    sub_lens = [203, 152, 64, 101]
    L_total = sum(sub_lens)  # 520
    cu_seq_lens_q = torch.tensor([0, 203, 355, 419, 520], dtype=torch.int32)
    hsa_k_norm = torch.randn(1, L_total, h_kv, D)  # 随机值

    # 在每个子序列的 lmk 位置填充唯一标记
    expected_lmk_positions = []
    for i, sub_len in enumerate(sub_lens):
        start = cu_seq_lens_q[i].item()
        num_chunks = sub_len // chunk_size
        for c in range(num_chunks):
            lmk_pos = start + (c + 1) * chunk_size - 1
            hsa_k_norm[0, lmk_pos, :, :] = float(1000 + i * 10 + c)
            expected_lmk_positions.append(lmk_pos)

    lmk_k, cu_seq_lens_k = extract_lmk_from_packed_hsa_k(
        hsa_k_norm, cu_seq_lens_q, chunk_size
    )

    # 验证: 3+2+1+1 = 7 个 lmk
    expected_k = torch.tensor([0, 3, 5, 6, 7], dtype=torch.int32)
    assert torch.equal(cu_seq_lens_k, expected_k), (
        f"cu_seq_lens_k 不正确: 期望 {expected_k.tolist()}, 得到 {cu_seq_lens_k.tolist()}"
    )
    assert lmk_k.shape[1] == 7, f"期望 7 个 lmk，得到 {lmk_k.shape[1]}"

    # 验证每个 lmk 的值与手动插入的标记一致
    for s in range(7):
        val = lmk_k[0, s, 0, 0].item()
        expected_val = hsa_k_norm[0, expected_lmk_positions[s], 0, 0].item()
        assert val == expected_val, (
            f"lmk_k[{s}] 值不匹配: 期望 {expected_val}, 得到 {val}"
        )

    print(f"    cu_seq_lens_k = {cu_seq_lens_k.tolist()} ✅")
    print(f"    4个子序列: 3+2+1+1=7 个lmk，全部正确提取 ✅")
    print(f"    全局 lmk 位置: {expected_lmk_positions} ✅")
    print(f"  ✅ 测试4 通过")

    print("\n" + "=" * 80)
    print("🎉 extract_lmk_from_packed_hsa_k 非对齐专项测试全部通过!")
    print("=" * 80)



# ============================================================================
# 6. online_topk_group 的 dense / varlen 路径 wrapper
# ============================================================================
def _build_unified_dense_wrapper(q, lmks, topk, block_size, window_size,
                                  is_causal,
                                  q_offset=0, is_training=True,
                                  drop_mask=None, force_recent_chunks=0):
    """
    用 online_topk_group (dense 路径) 走 unified kernel。
    返回 (indices, scores)，语义与 online_topk_group 完全一致。
    """
    from ops.topk_group import online_topk_group

    return online_topk_group(
        q, lmks, topk, block_size, window_size,
        is_causal=is_causal,
        q_offset=q_offset, is_training=is_training, drop_mask=drop_mask,
        force_recent_chunks=force_recent_chunks,
    )


def _build_unified_varlen_wrapper(q, lmks, topk, block_size, window_size,
                                   cu_seq_lens,
                                   is_causal=True, force_recent_chunks=0):
    """
    用 online_topk_group (varlen 路径) 走 unified kernel。
    返回 (indices, scores)，语义与 online_topk_group (varlen) 完全一致。
    """
    from ops.topk_group import online_topk_group

    return online_topk_group(
        q, lmks, topk, block_size, window_size,
        is_causal=is_causal,
        cu_seq_lens=cu_seq_lens,
        force_recent_chunks=force_recent_chunks,
    )


import pytest



# ============================================================================
# pytest 测试: unified kernel 两条路径 vs 纯 PyTorch ref (三角验证)
# ============================================================================

class TestUnifiedBothPathsVsRef:
    """
    综合验证: unified kernel 两条路径都与纯 PyTorch ref 一致。
    三角验证: unified dense (逐样本) vs unified varlen (打包) vs PyTorch ref。
    使用较宽松的 tolerance（bf16 vs fp32 精度差异）。
    """

    @pytest.mark.parametrize("samples,is_causal,h_q,h_kv,topk", [
        # 整数倍长度 (S = L // block_size)
        ([{"L": 512}, {"L": 384}, {"L": 640}],
         True, 4, 2, 4),
        # 非整数倍长度（S = L // 64 >= topk=4，L 需足够大使 causal mask 下有可见 chunk）
        ([{"L": 259}, {"L": 333}, {"L": 406}],
         True, 4, 2, 4),
        # 非因果 + 非整数倍（S = L // 64 >= topk=4）
        ([{"L": 290}, {"L": 310}],
         False, 4, 2, 4),
        # h_kv > h_q
        ([{"L": 512}, {"L": 384}],
         True, 2, 8, 4),
        # topk == S（部分子序列 S == topk，ref 中 clamp topk）
        ([{"L": 512}, {"L": 256}],
         False, 4, 2, 4),
        # 极短序列（L 接近 block_size，S = L // 64 >= topk=4）
        ([{"L": 260}, {"L": 270}],
         False, 4, 2, 4),
        # 多序列（8 个序列）
        ([{"L": 256}, {"L": 320}, {"L": 384},
          {"L": 256}, {"L": 320}, {"L": 384},
          {"L": 256}, {"L": 320}],
         False, 4, 2, 4),
    ], ids=[
        "aligned_causal",
        "unaligned_causal",
        "unaligned_non_causal",
        "hkv_gt_hq",
        "topk_eq_S",
        "short_seqs",
        "many_seqs_8",
    ])
    def test_triangle_verification(self, samples, is_causal, h_q, h_kv, topk):
        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(42)

        D = 64
        block_size = 64
        window_size = 128
        h_shared = min(h_q, h_kv)

        q_list = []
        lmks_list = []
        for cfg in samples:
            L = cfg["L"]
            S = L // block_size
            q_i = torch.randn(1, L, h_q, D, dtype=dtype, device=device)
            lmks_i = torch.randn(1, S, h_kv, D, dtype=dtype, device=device)
            q_list.append(q_i)
            lmks_list.append(lmks_i)

        # 路径1: unified dense (逐样本)
        dense_indices_list = []
        dense_scores_list = []
        for q_i, lmks_i in zip(q_list, lmks_list):
            idx_i, sc_i = _build_unified_dense_wrapper(
                q_i, lmks_i, topk, block_size, window_size, is_causal,
                q_offset=0, is_training=True,
            )
            dense_indices_list.append(idx_i)
            dense_scores_list.append(sc_i)

        # 路径2: unified varlen (打包)
        packed_q = torch.cat(q_list, dim=1)
        packed_lmks = torch.cat(lmks_list, dim=1)
        num_seqs = len(samples)
        cu_seq_lens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_seq_lens[i + 1] = cu_seq_lens[i] + samples[i]["L"]

        varlen_indices, varlen_scores = _build_unified_varlen_wrapper(
            packed_q, packed_lmks, topk, block_size, window_size,
            cu_seq_lens,
            is_causal=is_causal,
        )

        # 路径3: 纯 PyTorch ref（clamp topk 以避免 S < topk 时 torch.topk 越界）
        q_list_cpu = [q_i.float().cpu() for q_i in q_list]
        lmks_list_cpu = [l_i.float().cpu() for l_i in lmks_list]
        ref_topk_clamped = min(topk, min(cfg["L"] // block_size for cfg in samples))
        ref_indices, ref_scores, _, _ = ref_topk_varlen(
            q_list_cpu, lmks_list_cpu, ref_topk_clamped, block_size, window_size,
            is_causal=is_causal,
        )

        # 对比
        # 根据 cu_seq_lens 和 block_size 计算 k_start/k_end
        cu_seq_lens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_seq_lens_k[i + 1] = cu_seq_lens_k[i] + samples[i]["L"] // block_size
        for seq_idx in range(num_seqs):
            q_start = cu_seq_lens[seq_idx].item()
            q_end = cu_seq_lens[seq_idx + 1].item()
            k_start = cu_seq_lens_k[seq_idx].item()
            k_end = cu_seq_lens_k[seq_idx + 1].item()
            L_i = q_end - q_start
            S_i = k_end - k_start

            dense_idx = dense_indices_list[seq_idx][0]
            varlen_idx = varlen_indices[0, q_start:q_end].clone()
            valid_mask = varlen_idx >= 0
            varlen_idx[valid_mask] -= k_start
            ref_idx = ref_indices[0, q_start:q_end]

            # dense vs varlen 应该 bit-exact
            assert torch.equal(dense_idx.cpu(), varlen_idx.cpu()), (
                f"样本{seq_idx}: dense vs varlen indices 不一致"
            )

            # dense vs ref 用集合匹配率（bf16 vs fp32 精度差异）
            idx_match_count = 0
            idx_total = 0
            for l in range(L_i):
                for hh in range(h_shared):
                    d_set = set(dense_idx[l, hh, :].cpu().tolist())
                    r_set = set(ref_idx[l, hh, :].tolist())
                    d_set.discard(-1)
                    r_set.discard(-1)
                    if len(d_set) > 0 or len(r_set) > 0:
                        idx_total += 1
                        if d_set == r_set:
                            idx_match_count += 1

            if idx_total == 0:
                # 两者都没有有效 chunk（全 -1），视为完全匹配
                match_rate = 1.0
            else:
                match_rate = idx_match_count / idx_total
            assert match_rate >= 0.90, (
                f"样本{seq_idx} (L={L_i}, S={S_i}): "
                f"dense vs ref 匹配率 {match_rate:.1%} < 90%"
            )



# ============================================================================
# pytest 测试: unified varlen forward 直接 vs ref_topk_no_packing
# ============================================================================

class TestUnifiedVarlenVsNoPacking:
    """
    补全对比链的最后一环：unified varlen forward 的结果拆分后，
    直接与逐样本 ref_topk_no_packing 对比。

    对比链: ref_no_packing ↔ unified_varlen（直接对比，不经过中间环节）
    使用集合匹配率（bf16 vs fp32 精度差异）。
    """

    @staticmethod
    def _check(samples, is_causal, h_q=4, h_kv=2, topk=4,
               block_size=64, window_size=128, seed=42):
        device = "cuda"
        dtype = torch.bfloat16
        D = 64
        h_shared = min(h_q, h_kv)

        torch.manual_seed(seed)
        q_list = []
        lmks_list = []
        for cfg in samples:
            L = cfg["L"]
            S = L // block_size
            q_i = torch.randn(1, L, h_q, D, dtype=dtype, device=device)
            lmks_i = torch.randn(1, S, h_kv, D, dtype=dtype, device=device)
            q_list.append(q_i)
            lmks_list.append(lmks_i)

        # unified varlen
        packed_q = torch.cat(q_list, dim=1)
        packed_lmks = torch.cat(lmks_list, dim=1)
        num_seqs = len(samples)
        cu_seq_lens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_seq_lens[i + 1] = cu_seq_lens[i] + samples[i]["L"]

        varlen_indices, varlen_scores = _build_unified_varlen_wrapper(
            packed_q, packed_lmks, topk, block_size, window_size,
            cu_seq_lens,
            is_causal=is_causal,
        )

        # ref_topk_no_packing（纯 PyTorch fp32，逐样本独立）
        q_list_cpu = [q_i.float().cpu() for q_i in q_list]
        lmks_list_cpu = [l_i.float().cpu() for l_i in lmks_list]
        ref_topk_clamped = min(topk, min(cfg["L"] // block_size for cfg in samples))
        ref_indices_list, ref_scores_list = ref_topk_no_packing(
            q_list_cpu, lmks_list_cpu, ref_topk_clamped, block_size, window_size,
            is_causal=is_causal,
        )

        # 逐样本对比
        cu_seq_lens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_seq_lens_k[i + 1] = cu_seq_lens_k[i] + samples[i]["L"] // block_size
        for seq_idx in range(num_seqs):
            q_start = cu_seq_lens[seq_idx].item()
            q_end = cu_seq_lens[seq_idx + 1].item()
            k_start = cu_seq_lens_k[seq_idx].item()
            L_i = q_end - q_start

            # varlen indices 是 global index，转为 local
            v_idx = varlen_indices[0, q_start:q_end].clone().cpu()
            valid_mask = v_idx >= 0
            v_idx[valid_mask] -= k_start

            r_idx = ref_indices_list[seq_idx][0].cpu()  # [L_i, h_shared, topk]

            # 集合匹配率
            match_count = 0
            total = 0
            for l in range(L_i):
                for hh in range(h_shared):
                    v_set = set(v_idx[l, hh, :].tolist())
                    r_set = set(r_idx[l, hh, :].tolist())
                    v_set.discard(-1)
                    r_set.discard(-1)
                    if len(v_set) > 0 or len(r_set) > 0:
                        total += 1
                        if v_set == r_set:
                            match_count += 1

            if total == 0:
                rate = 1.0
            else:
                rate = match_count / total
            assert rate >= 0.90, (
                f"样本{seq_idx}: unified varlen vs ref_no_packing 匹配率 {rate:.1%} < 90%"
            )

    def test_aligned_causal(self):
        self._check(
            samples=[{"L": 512}, {"L": 384}, {"L": 640}],
            is_causal=True,
        )

    def test_unaligned_causal(self):
        self._check(
            samples=[{"L": 259}, {"L": 333}, {"L": 406}],
            is_causal=True,
        )

    def test_non_causal(self):
        self._check(
            samples=[{"L": 290}, {"L": 310}],
            is_causal=False,
        )

    def test_hkv_gt_hq(self):
        self._check(
            samples=[{"L": 512}, {"L": 384}],
            is_causal=True, h_q=2, h_kv=8,
        )


# ============================================================================
# pytest 测试: force_recent_chunks + varlen 联合正确性验证
# ============================================================================

class TestForceRecentChunksVarlen:
    """
    验证 force_recent_chunks 与 varlen 同时生效时的正确性。

    三角验证策略:
      1. unified dense (逐样本, force_recent_chunks>0) — 基线
      2. unified varlen (打包, force_recent_chunks>0) — 待验证
      3. 纯 PyTorch ref (ref_topk_no_packing, force_recent_chunks>0) — 参考

    对比链:
      - unified_varlen vs unified_dense (逐样本拆分后对比)
      - unified_varlen vs PyTorch ref
    """

    @staticmethod
    def _check(samples, force_recent_chunks, is_causal=True,
               h_q=4, h_kv=2, topk=4,
               block_size=64, window_size=128, seed=42):
        device = "cuda"
        dtype = torch.bfloat16
        D = 64
        h_shared = min(h_q, h_kv)

        # 确保 topk <= 最小 S_i，避免 k out of range
        min_S = min(cfg["L"] // block_size for cfg in samples)
        topk = min(topk, min_S)
        assert topk > force_recent_chunks, (
            f"topk({topk}) 必须大于 force_recent_chunks({force_recent_chunks})"
        )

        torch.manual_seed(seed)
        q_list = []
        lmks_list = []
        for cfg in samples:
            L = cfg["L"]
            S = L // block_size
            q_i = torch.randn(1, L, h_q, D, dtype=dtype, device=device)
            lmks_i = torch.randn(1, S, h_kv, D, dtype=dtype, device=device)
            q_list.append(q_i)
            lmks_list.append(lmks_i)

        # --- 路径1: unified dense (逐样本, force_recent_chunks>0) ---
        dense_indices_list = []
        dense_scores_list = []
        for q_i, lmks_i in zip(q_list, lmks_list):
            idx_i, sc_i = _build_unified_dense_wrapper(
                q_i, lmks_i, topk, block_size, window_size, is_causal,
                q_offset=0, is_training=True,
                force_recent_chunks=force_recent_chunks,
            )
            dense_indices_list.append(idx_i)
            dense_scores_list.append(sc_i)

        # --- 路径2: unified varlen (打包, force_recent_chunks>0) ---
        packed_q = torch.cat(q_list, dim=1)
        packed_lmks = torch.cat(lmks_list, dim=1)
        num_seqs = len(samples)
        cu_seq_lens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_seq_lens[i + 1] = cu_seq_lens[i] + samples[i]["L"]

        varlen_indices, varlen_scores = _build_unified_varlen_wrapper(
            packed_q, packed_lmks, topk, block_size, window_size,
            cu_seq_lens,
            is_causal=is_causal,
            force_recent_chunks=force_recent_chunks,
        )

        # --- 路径3: 纯 PyTorch ref (逐样本独立, fp32) ---
        q_list_cpu = [q_i.float().cpu() for q_i in q_list]
        lmks_list_cpu = [l_i.float().cpu() for l_i in lmks_list]
        ref_indices_list, ref_scores_list = ref_topk_no_packing(
            q_list_cpu, lmks_list_cpu, topk, block_size, window_size,
            is_causal=is_causal,
            force_recent_chunks=force_recent_chunks,
        )

        # --- 对比 ---
        cu_seq_lens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_seq_lens_k[i + 1] = cu_seq_lens_k[i] + samples[i]["L"] // block_size

        for seq_idx in range(num_seqs):
            q_start = cu_seq_lens[seq_idx].item()
            q_end = cu_seq_lens[seq_idx + 1].item()
            k_start = cu_seq_lens_k[seq_idx].item()
            L_i = q_end - q_start

            # --- varlen vs dense ---
            dense_idx = dense_indices_list[seq_idx][0].cpu()  # [L_i, h_shared, topk]
            varlen_idx = varlen_indices[0, q_start:q_end].clone().cpu()
            # varlen indices 是 global index，转为 local
            valid_mask = varlen_idx >= 0
            varlen_idx[valid_mask] -= k_start

            match_vd = 0
            total_vd = 0
            for l in range(L_i):
                for hh in range(h_shared):
                    v_set = set(varlen_idx[l, hh, :].tolist())
                    d_set = set(dense_idx[l, hh, :].tolist())
                    v_set.discard(-1)
                    d_set.discard(-1)
                    if len(v_set) > 0 or len(d_set) > 0:
                        total_vd += 1
                        if v_set == d_set:
                            match_vd += 1
            rate_vd = match_vd / total_vd if total_vd > 0 else 1.0
            assert rate_vd >= 0.95, (
                f"样本{seq_idx}: varlen vs dense 匹配率 {rate_vd:.1%} < 95% "
                f"(force_recent_chunks={force_recent_chunks})"
            )

            # --- varlen vs ref (逐样本独立 PyTorch ref) ---
            ref_idx = ref_indices_list[seq_idx][0].cpu()  # [L_i, h_shared, topk]
            match_vr = 0
            total_vr = 0
            for l in range(L_i):
                for hh in range(h_shared):
                    v_set = set(varlen_idx[l, hh, :].tolist())
                    r_set = set(ref_idx[l, hh, :].tolist())
                    v_set.discard(-1)
                    r_set.discard(-1)
                    if len(v_set) > 0 or len(r_set) > 0:
                        total_vr += 1
                        if v_set == r_set:
                            match_vr += 1
            rate_vr = match_vr / total_vr if total_vr > 0 else 1.0
            assert rate_vr >= 0.90, (
                f"样本{seq_idx}: varlen vs ref 匹配率 {rate_vr:.1%} < 90% "
                f"(force_recent_chunks={force_recent_chunks})"
            )

    # --- 基本测试: 不同 force_recent_chunks 值 ---
    @pytest.mark.parametrize("force_recent_chunks", [1, 2, 3])
    def test_aligned_causal(self, force_recent_chunks):
        """整数倍长度 + causal + 不同 force_recent_chunks"""
        self._check(
            samples=[{"L": 512}, {"L": 384}, {"L": 640}],
            force_recent_chunks=force_recent_chunks,
            is_causal=True,
        )

    @pytest.mark.parametrize("force_recent_chunks", [1, 2])
    def test_unaligned_causal(self, force_recent_chunks):
        """非整数倍长度 + causal"""
        self._check(
            samples=[{"L": 320}, {"L": 384}, {"L": 448}],
            force_recent_chunks=force_recent_chunks,
            is_causal=True,
        )

    def test_large_force_recent(self):
        """force_recent_chunks 较大（topk=4, force_recent=3）"""
        self._check(
            samples=[{"L": 1024}, {"L": 768}],
            force_recent_chunks=3,
            is_causal=True,
            topk=4,
        )

    def test_hkv_gt_hq(self):
        """h_kv > h_q 场景"""
        self._check(
            samples=[{"L": 512}, {"L": 384}],
            force_recent_chunks=2,
            is_causal=True,
            h_q=2, h_kv=8,
        )

    def test_many_seqs(self):
        """多序列打包（6 个序列）"""
        self._check(
            samples=[{"L": 512}, {"L": 640}, {"L": 384},
                     {"L": 512}, {"L": 640}, {"L": 384}],
            force_recent_chunks=2,
            is_causal=True,
        )

    def test_force_recent_zero_baseline(self):
        """force_recent_chunks=0 基线验证（应与原有 varlen 测试结果一致）"""
        self._check(
            samples=[{"L": 512}, {"L": 384}, {"L": 640}],
            force_recent_chunks=0,
            is_causal=True,
        )


# ============================================================================
# pytest 测试: varlen backward vs 纯 PyTorch ref 独立梯度验证
# ============================================================================

class TestVarlenBackwardVsRef:
    """
    独立验证 varlen backward 的数学正确性，不依赖旧的 online_topk_group (varlen)。

    策略：用纯 PyTorch 的 ref_topk_forward_with_grad（支持 autograd）计算梯度，
    与 unified varlen backward 的梯度对比。使用宽松 tolerance（bf16 vs fp32）。
    """

    @staticmethod
    def _check(samples, is_causal, h_q=4, h_kv=2,
               topk=4, block_size=64, window_size=128, seed=42):
        from ops.topk_group import online_topk_group

        device = "cuda"
        dtype = torch.bfloat16
        D = 64
        h_shared = min(h_q, h_kv)

        torch.manual_seed(seed)
        q_list = []
        lmks_list = []
        for cfg in samples:
            L = cfg["L"]
            S = L // block_size
            q_i = torch.randn(1, L, h_q, D, dtype=dtype, device=device)
            lmks_i = torch.randn(1, S, h_kv, D, dtype=dtype, device=device)
            q_list.append(q_i)
            lmks_list.append(lmks_i)

        packed_q_raw = torch.cat(q_list, dim=1)
        packed_lmks_raw = torch.cat(lmks_list, dim=1)

        num_seqs = len(samples)
        cu_seq_lens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_seq_lens[i + 1] = cu_seq_lens[i] + samples[i]["L"]

        # 根据 cu_seq_lens 和 block_size 计算 k_start/k_end（仅用于 ref 对比）
        cu_seq_lens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_seq_lens_k[i + 1] = cu_seq_lens_k[i] + samples[i]["L"] // block_size

        L_total = packed_q_raw.shape[1]
        grad_output = torch.randn(1, L_total, h_shared, topk, dtype=dtype, device=device)

        # ---- unified varlen backward ----
        q_unified = packed_q_raw.clone().detach().requires_grad_(True)
        lmks_unified = packed_lmks_raw.clone().detach().requires_grad_(True)
        indices_unified, scores_unified = online_topk_group(
            q_unified, lmks_unified, topk, block_size, window_size,
            is_causal=is_causal,
            cu_seq_lens=cu_seq_lens,
        )
        loss_unified = (scores_unified * grad_output).sum()
        loss_unified.backward()
        grad_q_unified = q_unified.grad.clone()
        grad_lmks_unified = lmks_unified.grad.clone()

        # ---- 纯 PyTorch ref backward（逐样本 fp32 计算）----
        # 逐样本独立计算梯度，然后拼接
        grad_q_ref_list = []
        grad_lmks_ref_list = []

        for seq_idx in range(num_seqs):
            q_start = cu_seq_lens[seq_idx].item()
            q_end = cu_seq_lens[seq_idx + 1].item()
            k_start = cu_seq_lens_k[seq_idx].item()
            k_end = cu_seq_lens_k[seq_idx + 1].item()
            L_i = q_end - q_start
            S_i = k_end - k_start

            q_i = packed_q_raw[0, q_start:q_end].float().detach().requires_grad_(True)
            lmks_i = packed_lmks_raw[0, k_start:k_end].float().detach().requires_grad_(True)

            # group sum + scores
            sm_scale = 1.0 / math.sqrt(D)
            if h_q >= h_kv:
                G = h_q // h_kv
                q_gs = q_i.view(L_i, h_kv, G, D).sum(dim=2)
                scores = torch.einsum("lhd,shd->lhs", q_gs, lmks_i) * sm_scale
            else:
                G = h_kv // h_q
                lmk_gs = lmks_i.view(S_i, h_q, G, D).sum(dim=2)
                scores = torch.einsum("lhd,shd->lhs", q_i, lmk_gs) * sm_scale

            # causal mask
            if is_causal:
                i_idx = torch.arange(L_i, device=device).unsqueeze(1)
                j_idx = torch.arange(S_i, device=device).unsqueeze(0)
                limit = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
                mask = j_idx >= limit
                scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

            # 使用 unified 的 indices 来选取对应的 scores（确保选取位置一致）
            idx_i = indices_unified[0, q_start:q_end].clone()
            valid = (idx_i >= k_start) & (idx_i < k_end)
            local_idx = idx_i.clone().to(torch.int64)
            local_idx[valid] -= k_start
            local_idx[~valid] = 0

            # gather scores at the same positions
            selected_scores = torch.gather(scores, 2, local_idx)  # [L_i, h_shared, topk]
            # mask invalid
            selected_scores = selected_scores * valid.float()

            grad_i = grad_output[0, q_start:q_end].float()  # [L_i, h_shared, topk]
            grad_i = grad_i * valid.float()

            loss_ref = (selected_scores * grad_i).sum()
            loss_ref.backward()

            grad_q_ref_list.append(q_i.grad.to(dtype))
            grad_lmks_ref_list.append(lmks_i.grad.to(dtype))

        grad_q_ref = torch.cat(grad_q_ref_list, dim=0).unsqueeze(0)
        grad_lmks_ref = torch.cat(grad_lmks_ref_list, dim=0).unsqueeze(0)

        # 辅助校验函数（RMSE ratio 方式，对 bf16 outlier 更稳健）
        def get_abs_err(x, y):
            mask = (x > -1e5) & (y > -1e5)
            if mask.sum() == 0:
                return 0.0
            return (x[mask] - y[mask]).abs().max().item()

        def get_err_ratio(x, y):
            mask = (x > -1e5) & (y > -1e5)
            if mask.sum() == 0:
                return 0.0
            err = (x[mask] - y[mask]).square().mean().sqrt().item()
            base = (x[mask]).square().mean().sqrt().item()
            return err / (base + 1e-12)

        def assert_close(prefix, ref, tri, ratio=0.005):
            abs_err = get_abs_err(ref, tri)
            rel_ratio = get_err_ratio(ref, tri)
            msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
            print(msg)
            assert rel_ratio < ratio, msg

        assert_close("BWD grad_q", grad_q_ref.float(), grad_q_unified.float())
        assert_close("BWD grad_lmks", grad_lmks_ref.float(), grad_lmks_unified.float())

    def test_basic_causal(self):
        self._check(
            samples=[{"L": 512}, {"L": 384}],
            is_causal=True,
        )

    def test_non_causal(self):
        self._check(
            samples=[{"L": 256}, {"L": 256}],
            is_causal=False,
        )

    def test_hkv_gt_hq(self):
        self._check(
            samples=[{"L": 300}, {"L": 200}],
            is_causal=True, h_q=2, h_kv=8,
        )


# ============================================================================
# 极简可视化: 直观打印 varlen 接口输出情况
# ============================================================================

def visualize_varlen_output():
    """
    极简可视化: 打印 varlen 接口在几个 case 上的输出概况。
    检查: indices 是否越界、无效位置(-1)占比、与 torch ref 的匹配率、scores 误差。
    用法: python test_topk_varlen.py --viz
    """
    from ops.topk_group import online_topk_group

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    D, block_size, window_size = 64, 64, 128
    h_q, h_kv, topk = 4, 2, 4
    h_shared = min(h_q, h_kv)

    cases = [
        ("causal_aligned",   [{"L": 512}, {"L": 384}], True),
        ("causal_unaligned", [{"L": 259}, {"L": 333}], True),
        ("non_causal",       [{"L": 290}, {"L": 310}], False),
    ]

    print("=" * 80)
    print("Varlen 接口输出可视化")
    print("=" * 80)

    for case_name, samples, is_causal in cases:
        # 构造输入
        q_list, lmks_list = [], []
        for cfg in samples:
            L = cfg["L"]
            S = L // block_size
            q_list.append(torch.randn(1, L, h_q, D, dtype=dtype, device=device))
            lmks_list.append(torch.randn(1, S, h_kv, D, dtype=dtype, device=device))

        packed_q = torch.cat(q_list, dim=1)
        packed_lmks = torch.cat(lmks_list, dim=1)
        num_seqs = len(samples)
        cu_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_q[i+1] = cu_q[i] + samples[i]["L"]

        # 计算 k 级别的边界（仅用于可视化对比）
        cu_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        for i in range(num_seqs):
            cu_k[i+1] = cu_k[i] + samples[i]["L"] // block_size

        # 调用 varlen 接口
        indices, scores = online_topk_group(
            packed_q, packed_lmks, topk, block_size, window_size,
            is_causal=is_causal,
            cu_seq_lens=cu_q,
        )

        # 逐样本 torch ref
        ref_indices_list, ref_scores_list = [], []
        for q_i, lmks_i in zip(q_list, lmks_list):
            ri, rs = ref_topk_single_sample(
                q_i.float().cpu(), lmks_i.float().cpu(),
                min(topk, lmks_i.shape[1]), block_size, window_size,
                is_causal=is_causal,
            )
            ref_indices_list.append(ri)
            ref_scores_list.append(rs)

        print(f"\n--- {case_name} (causal={is_causal}) ---")
        for seq_idx in range(num_seqs):
            q_s, q_e = cu_q[seq_idx].item(), cu_q[seq_idx+1].item()
            k_s, k_e = cu_k[seq_idx].item(), cu_k[seq_idx+1].item()
            L_i, S_i = q_e - q_s, k_e - k_s

            idx_i = indices[0, q_s:q_e]           # [L_i, h_shared, topk]
            sc_i  = scores[0, q_s:q_e]            # [L_i, h_shared, topk]
            ref_idx_i = ref_indices_list[seq_idx][0]  # [L_i, h_shared, topk]
            ref_sc_i  = ref_scores_list[seq_idx][0]

            # 1) indices 越界检查（有效 index 应在 [k_s, k_e) 范围内）
            valid = idx_i >= 0
            if valid.any():
                oob = ((idx_i[valid] < k_s) | (idx_i[valid] >= k_e)).sum().item()
            else:
                oob = 0

            # 2) 无效位置(-1)占比
            invalid_ratio = (~valid).sum().item() / idx_i.numel()

            # 3) 与 ref 的 index 集合匹配率
            local_idx = idx_i.clone()
            local_idx[valid] -= k_s  # 转为 local index 以便与 ref 对比
            match, total = 0, 0
            for l in range(L_i):
                for hh in range(h_shared):
                    d_set = set(local_idx[l, hh].cpu().tolist()) - {-1}
                    r_set = set(ref_idx_i[l, hh].tolist()) - {-1}
                    if d_set or r_set:
                        total += 1
                        if d_set == r_set:
                            match += 1
            match_rate = match / total if total > 0 else 1.0

            # 4) scores RMSE ratio（仅比较有效位置）
            sc_flat = sc_i.float().cpu().reshape(-1)
            ref_sc_flat = ref_sc_i.float().reshape(-1)
            mask = (sc_flat > -1e5) & (ref_sc_flat > -1e5)
            if mask.sum() > 0:
                rmse = (sc_flat[mask] - ref_sc_flat[mask]).square().mean().sqrt().item()
                rms_base = sc_flat[mask].square().mean().sqrt().item()
                score_ratio = rmse / (rms_base + 1e-12)
            else:
                score_ratio = 0.0

            print(f"  seq[{seq_idx}] L={L_i:4d} S={S_i:2d} | "
                  f"OOB={oob} invalid={invalid_ratio:.1%} | "
                  f"idx_match={match_rate:.1%} score_ratio={score_ratio:.4f}")

            # 5) 直观展示 indices 是 global 还是 local
            #    取最后一个 q token（最可能有完整 topk 有效值）的 head=0 的 topk indices
            sample_q = min(L_i - 1, L_i - 1)  # 最后一个 q token
            raw_vals = idx_i[sample_q, 0].cpu().tolist()  # head=0 的 topk 个 index
            ref_vals = ref_idx_i[sample_q, 0].tolist()    # ref 是 local index
            # 判断: 如果有效 index 都在 [0, S_i) 内 → local; 在 [k_s, k_e) 内 → global
            valid_raw = [v for v in raw_vals if v >= 0]
            in_local  = all(0 <= v < S_i for v in valid_raw) if valid_raw else True
            in_global = all(k_s <= v < k_e for v in valid_raw) if valid_raw else True
            if k_s == 0:
                kind = "global==local (k_start=0, 无法区分)"
            elif in_global and not in_local:
                kind = "GLOBAL (indices ∈ [k_start, k_end))"
            elif in_local and not in_global:
                kind = "LOCAL  (indices ∈ [0, S_i))"
            elif in_local and in_global:
                kind = "ambiguous (范围重叠)"
            else:
                kind = "ERROR (既不在 local 也不在 global 范围!)"
            print(f"         k_range=[{k_s}, {k_e})  raw_idx={raw_vals}  ref_local={ref_vals}")
            print(f"         → 判断: {kind}")

    print("\n" + "=" * 80)
    print("可视化完成")
    print("=" * 80)


# ============================================================================
# __main__ 入口（兼容直接运行）
# ============================================================================

if __name__ == "__main__":
    import sys
    if "--viz" in sys.argv:
        visualize_varlen_output()
    else:
        pytest.main([__file__, "-v", "--tb=short"])
