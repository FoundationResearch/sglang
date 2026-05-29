"""
简单测试：q 对应 kv 的中间部分时，topk 是否正确。

场景：
  总序列长度 10 个 chunk，q 是其中的 chunk 4/5/6（即 token 位置 4*bs ~ 7*bs）
  - 情况A：kv 只有 chunk 0~6（7个chunk），q_offset=4*bs
  - 情况B：kv 是完整的 chunk 0~9（10个chunk），q_offset=4*bs
  
  由于 causal mask，q 只能看到自己之前的 chunk。
  所以 chunk 7/8/9 对 q 来说是不可见的，两种情况的输出应该一致。
"""

import sys
sys.path.insert(0, '/data/workspace/wxy_local/InfiniteLongLM')

import torch
import math

from ops.topk_group import online_topk_group

def test_topk_sp_simple():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # ---- 参数 ----
    B = 1
    h_kv = 1
    G = 8
    h_q = h_kv * G
    D = 64
    block_size = 64
    window_size = 64
    topk = 4
    is_causal = True
    memory_window_size = -1

    num_total_chunks = 10         # 完整序列 10 个 chunk

    # q 对应 chunk 4,5,6 → token [256, 448)
    q_start_chunk = 4
    q_end_chunk = 7               # 不含
    q_offset = q_start_chunk * block_size  # 256
    local_q_len = (q_end_chunk - q_start_chunk) * block_size  # 192

    S_short = q_end_chunk  # 7 个 chunk
    S_all = num_total_chunks  # 10 个 chunk

    print(f"总序列: {num_total_chunks} chunks")
    print(f"q 范围: chunk {q_start_chunk}~{q_end_chunk-1}, q_offset={q_offset}")

    # ---- 直接生成数据（不从完整序列切片）----
    q_local = torch.randn(B, local_q_len, h_q, D, dtype=dtype, device=device)

   # 先生成 10 个 chunk 的 lmk 数据源
    lmk_all_data = torch.randn(B, S_all, h_kv, D, dtype=dtype, device=device)

    # 情况A: lmk 只有前 7 个 chunk
    # 用 torch.empty 新建 tensor，再 copy_ 进去，确保 stride 完全正确
    lmks_short = torch.empty(B, S_short, h_kv, D, dtype=dtype, device=device)
    lmks_short.copy_(lmk_all_data[:, :S_short, :, :])

    # 情况B: lmk 是完整的 10 个 chunk
    lmks_all = torch.empty(B, S_all, h_kv, D, dtype=dtype, device=device)
    lmks_all.copy_(lmk_all_data)

    print(f"lmks_short: {lmks_short.shape} (chunk 0~{S_short-1})")
    print(f"lmks_all:   {lmks_all.shape} (chunk 0~{S_all-1})")

    from ops.topk_group import _MODULE_CACHE

    # 清空缓存，确保为 S=7 编译新 kernel
    _MODULE_CACHE.clear()
    indices_fA, scores_fA = online_topk_group(
        q_local, lmks_short, topk, block_size, window_size, is_causal,
        memory_window_size=memory_window_size, q_offset=q_offset,
    )

    # 清空缓存，确保为 S=10 编译新 kernel（避免 stride 不匹配）
    _MODULE_CACHE.clear()
    indices_fB, scores_fB = online_topk_group(
        q_local, lmks_all, topk, block_size, window_size, is_causal,
        memory_window_size=memory_window_size, q_offset=q_offset,
    )

    valid_fA = scores_fA.float() > -1e9
    valid_fB = scores_fB.float() > -1e9

    if valid_fB.any():
        max_idx_fB = indices_fB[valid_fB].max().item()
        print(f"  情况B fused: 最大选中 chunk idx = {max_idx_fB} (应 < {S_short})")
        assert max_idx_fB < S_short, f"fused 选到了 chunk {max_idx_fB} >= {S_short}，causal mask 有问题！"

    both_valid_f = valid_fA & valid_fB
    if both_valid_f.any():
        score_diff_f = (scores_fA.float()[both_valid_f] - scores_fB.float()[both_valid_f]).abs().max().item()
        idx_match_f = (indices_fA[both_valid_f] == indices_fB[both_valid_f]).float().mean().item()
        print(f"  fused scores diff: {score_diff_f:.8f}")
        print(f"  fused indices match: {idx_match_f * 100:.2f}%")
        assert score_diff_f < 1e-3, f"fused scores diff = {score_diff_f}"
        assert idx_match_f > 0.9999, f"fused indices match = {idx_match_f * 100:.2f}%"
        print("  ✅ fused: 截断 kv 和完整 kv 输出一致")
    else:
        print("  ⚠️  没有共同的 valid 位置可比较")


if __name__ == "__main__":
    test_topk_sp_simple()
