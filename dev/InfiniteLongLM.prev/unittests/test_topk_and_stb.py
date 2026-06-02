"""
验证 topk 输出是否符合 stick breaking 的预期格式

核心检查项：
1. indices 的排序方式：Stick Breaking 期望【相对位置越近的 chunk 排在前面】
2. scores 对应的排序是否正确
3. 对比不同排序下 stick_breaking 的权重分布
"""
import sys
sys.path.insert(0, '/data/workspace/wxy_local/InfiniteLongLM')

import torch
import torch.nn.functional as F

def verify_topk_order_for_stick_breaking():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # 参数设置
    B, L = 2, 512
    h_kv = 2
    D = 64
    topk = 32
    chunk_size = 64
    S = L // chunk_size  # landmark 数量
    
    print(f"=== Config ===")
    print(f"B={B}, L={L}, h_kv={h_kv}, topk={topk}, chunk_size={chunk_size}, S={S}")
    
    # 模拟输入
    lmk_q_norm = torch.randn(B, L, h_kv, D, device=device, dtype=dtype)
    lmk_k = torch.randn(B, S, h_kv, D, device=device, dtype=dtype)
    
    from ops.topk_group import online_topk_group
    
    indices, scores = online_topk_group(
        lmk_q_norm, lmk_k, topk, 
        block_size=chunk_size, window_size=0, is_causal=True
    )
    
    print(f"\nindices shape: {indices.shape}")  # [B, L, h_kv, topk]
    print(f"scores shape:  {scores.shape}")
    
    # ============================================
    # 关键检查：indices 的排序方向
    # ============================================
    print(f"\n{'='*60}")
    print(f"=== 关键检查：TopK 返回的 indices 排序方向 ===")
    print(f"{'='*60}")
    
    # 取一个中间位置的 query（能看到多个 chunk）
    query_pos = L - 10  # 接近末尾，能看到所有 chunk
    sample_indices = indices[0, query_pos, 0, :].tolist()
    sample_scores = scores[0, query_pos, 0, :].float().tolist()
    
    # 计算当前 query 所在的 chunk
    current_chunk = query_pos // chunk_size
    print(f"\nQuery position: {query_pos} (in chunk {current_chunk})")
    print(f"Can see chunks: 0 to {current_chunk - 1}")
    
    # 分析 indices
    valid_indices = [i for i in sample_indices if i >= 0]
    print(f"\nTopK 返回的 indices (前16个): {sample_indices[:16]}")
    print(f"TopK 返回的 scores (前16个):  {[f'{s:.3f}' for s in sample_scores[:16]]}")
    
    # 检查排序方向
    if len(valid_indices) >= 2:
        is_descending = valid_indices[0] > valid_indices[-1]
        is_ascending = valid_indices[0] < valid_indices[-1]
        
        print(f"\n排序分析:")
        print(f"  - 第一个有效 index: {valid_indices[0]} (距离当前 chunk {current_chunk - valid_indices[0]} 个 chunk)")
        print(f"  - 最后一个有效 index: {valid_indices[-1]} (距离当前 chunk {current_chunk - valid_indices[-1]} 个 chunk)")
        
        if is_descending:
            print(f"\n❌ 当前排序: 【降序】(远的 chunk 在前，近的在后)")
            print(f"   这对 Stick Breaking 是【错误的】！")
            print(f"   Stick Breaking 会给远的 chunk 更大的权重")
        elif is_ascending:
            print(f"\n✅ 当前排序: 【升序】(近的 chunk 在前，远的在后)")
            print(f"   这对 Stick Breaking 是【正确的】！")
        else:
            print(f"\n⚠️  无法确定排序方向")
    
    # ============================================
    # 模拟 Stick Breaking 的权重分布
    # ============================================
    print(f"\n{'='*60}")
    print(f"=== Stick Breaking 权重分布对比 ===")
    print(f"{'='*60}")
    
    scores_float = scores[0, query_pos, 0, :].float()
    valid_mask = scores_float > -1e5
    
    # 当前顺序的 Stick Breaking 权重
    softplus_x = F.softplus(scores_float, threshold=15.0)
    softplus_cumsum = torch.cumsum(softplus_x, dim=-1)
    weights_current = (scores_float - softplus_cumsum).exp()
    
    # 反转顺序后的 Stick Breaking 权重
    scores_reversed = scores_float.flip(0)
    indices_reversed = torch.tensor(sample_indices).flip(0)
    softplus_rev = F.softplus(scores_reversed, threshold=15.0)
    softplus_cumsum_rev = torch.cumsum(softplus_rev, dim=-1)
    weights_reversed = (scores_reversed - softplus_cumsum_rev).exp()
    
    print(f"\n当前顺序 (indices 从大到小，即远的在前):")
    print(f"  indices:    {sample_indices[:8]}")
    print(f"  scores:     {[f'{s:.3f}' for s in scores_float[:8].tolist()]}")
    print(f"  SB weights: {[f'{w:.4f}' for w in weights_current[:8].tolist()]}")
    print(f"  权重和:     {weights_current[valid_mask].sum().item():.4f}")
    
    print(f"\n反转顺序 (indices 从小到大，即近的在前):")
    print(f"  indices:    {indices_reversed[:8].tolist()}")
    print(f"  scores:     {[f'{s:.3f}' for s in scores_reversed[:8].tolist()]}")
    print(f"  SB weights: {[f'{w:.4f}' for w in weights_reversed[:8].tolist()]}")
    print(f"  权重和:     {weights_reversed[valid_mask.flip(0)].sum().item():.4f}")
    
    # ============================================
    # 关键：哪个 chunk 获得了最大权重？
    # ============================================
    print(f"\n{'='*60}")
    print(f"=== 哪个 chunk 获得最大权重? ===")
    print(f"{'='*60}")
    
    max_weight_idx = weights_current.argmax().item()
    max_weight_chunk = sample_indices[max_weight_idx]
    max_weight_value = weights_current[max_weight_idx].item()
    
    print(f"\n当前顺序下:")
    print(f"  最大权重位置: index={max_weight_idx}")
    print(f"  对应 chunk:   {max_weight_chunk}")
    print(f"  权重值:       {max_weight_value:.4f}")
    print(f"  距离当前位置: {current_chunk - max_weight_chunk} 个 chunk")
    
    # 检查最近的 chunk 获得的权重
    nearest_chunk = current_chunk - 1  # 最近的可见 chunk
    if nearest_chunk in sample_indices:
        nearest_pos = sample_indices.index(nearest_chunk)
        nearest_weight = weights_current[nearest_pos].item()
        print(f"\n最近 chunk (chunk {nearest_chunk}) 的权重: {nearest_weight:.4f}")
        print(f"  在 topk 中的位置: {nearest_pos}")
    
    # 检查最远的 chunk 获得的权重
    farthest_chunk = 0
    if farthest_chunk in sample_indices:
        farthest_pos = sample_indices.index(farthest_chunk)
        farthest_weight = weights_current[farthest_pos].item()
        print(f"\n最远 chunk (chunk {farthest_chunk}) 的权重: {farthest_weight:.4f}")
        print(f"  在 topk 中的位置: {farthest_pos}")
    
    # ============================================
    # 结论和建议
    # ============================================
    print(f"\n{'='*60}")
    print(f"=== 结论 ===")
    print(f"{'='*60}")
    
    if len(valid_indices) >= 2 and valid_indices[0] > valid_indices[-1]:
        print(f"""
⚠️  发现问题：TopK 返回的 indices 是【降序排列】（远的 chunk 在前）

   Stick Breaking 的特性：
   - 第一个元素获得 sigmoid(s_0) 的权重
   - 后续元素获得的权重会被前面元素"吃掉"一部分
   - 因此【排在前面的元素倾向于获得更大的权重】

   当前问题：
   - 远的 chunk（index 大）排在前面
   - 导致 Stick Breaking 给远的 chunk 更大的权重
   - 这与 "近的 chunk 应该更重要" 的直觉相反

   建议修复方案：
   1. 在 topk_group kernel 中修改排序逻辑，改为【升序排列】
   2. 或者在 modeling 中调用 topk 后，手动反转 indices 和 scores
        """)
    else:
        print(f"✅ TopK 排序方向符合 Stick Breaking 的期望")


def verify_stick_breaking_math():
    """验证 Stick Breaking 的数学公式"""
    print(f"\n{'='*60}")
    print(f"=== Stick Breaking 数学验证 ===")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 假设有 4 个 chunk，scores 分别为 [2, 1, 0, -1]
    # chunk 0 最远，chunk 3 最近
    
    # 场景1：按当前顺序（远的在前）
    scores_far_first = torch.tensor([2.0, 1.0, 0.0, -1.0], device=device)  # chunk 3,2,1,0
    
    # 场景2：近的在前（期望的顺序）
    scores_near_first = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)  # chunk 0,1,2,3
    
    def compute_sb_weights(scores):
        softplus = F.softplus(scores, threshold=15.0)
        cumsum = torch.cumsum(softplus, dim=-1)
        return (scores - cumsum).exp()
    
    w_far_first = compute_sb_weights(scores_far_first)
    w_near_first = compute_sb_weights(scores_near_first)
    
    print(f"\n场景1: 远的 chunk 在前 (当前实现)")
    print(f"  位置:   [0,     1,     2,     3    ]")
    print(f"  chunk:  [3(远), 2,     1,     0(近)]")
    print(f"  scores: {scores_far_first.tolist()}")
    print(f"  weights:{[f'{w:.4f}' for w in w_far_first.tolist()]}")
    print(f"  -> 远的 chunk 3 获得权重: {w_far_first[0]:.4f}")
    print(f"  -> 近的 chunk 0 获得权重: {w_far_first[3]:.4f}")
    
    print(f"\n场景2: 近的 chunk 在前 (期望的实现)")
    print(f"  位置:   [0,     1,     2,     3    ]")
    print(f"  chunk:  [0(近), 1,     2,     3(远)]")
    print(f"  scores: {scores_near_first.tolist()}")
    print(f"  weights:{[f'{w:.4f}' for w in w_near_first.tolist()]}")
    print(f"  -> 近的 chunk 0 获得权重: {w_near_first[0]:.4f}")
    print(f"  -> 远的 chunk 3 获得权重: {w_near_first[3]:.4f}")
    
    print(f"\n关键观察:")
    print(f"  - 场景1 中，score=2 的元素（远的 chunk）获得最大权重 {w_far_first[0]:.4f}")
    print(f"  - 场景2 中，score=2 的元素（远的 chunk）只获得权重 {w_near_first[3]:.4f}")
    print(f"  - Stick Breaking 天然偏向【排在前面】的元素")


if __name__ == "__main__":
    verify_topk_order_for_stick_breaking()
    verify_stick_breaking_math()
