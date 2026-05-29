import torch


def test_topk_dropout_mask_effect(
    B: int = 2,
    L: int = 6,
    h_hsa_kv: int = 3,
    topk: int = 4,
    topk_dropout: float = 0.35,
    seed: int = 2026,
    device: str | None = None,
):
    """
    仅使用 torch 复现并验证如下逻辑：

        do_dropout = torch.rand(B, L, 1, 1, device=scores.device) < topk_dropout
        do_dropout = do_dropout.repeat(1, 1, h_hsa_kv, 1)
        scores = scores.masked_fill(do_dropout, float('-inf'))

    目标：打印并验证 mask 前后变化，确认触发后会把该 (b,l,h) 的整条 topk 维度都置为 -inf。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 构造一个可观察的 scores: [B, L, h_hsa_kv, topk]
    scores_before = torch.randn(B, L, h_hsa_kv, topk, device=device, dtype=torch.float32)

    # ====== 复现你的逻辑 ======
    do_dropout = torch.rand(B, L, 1, 1, device=scores_before.device) < topk_dropout
    do_dropout = do_dropout.repeat(1, 1, h_hsa_kv, 1)
    scores_after = scores_before.masked_fill(do_dropout, float("-inf"))
    # ==========================

    # 统计
    trigger_mask = do_dropout[..., 0]  # [B, L, h_hsa_kv]
    trigger_count = int(trigger_mask.sum().item())
    total = B * L * h_hsa_kv
    trigger_ratio = trigger_count / max(total, 1)

    inf_per_slot = torch.isinf(scores_after).sum(dim=-1)  # [B, L, h_hsa_kv]

    print("=" * 100)
    print("[test_topk_dropout_mask_effect] 仅 torch 的 dropout 后处理验证")
    print(f"device={device}, seed={seed}")
    print(f"scores.shape={tuple(scores_before.shape)}, topk_dropout={topk_dropout}")
    print(f"trigger_count={trigger_count}/{total}, trigger_ratio={trigger_ratio:.4f}")
    print("说明：每个 (b,l,h) 一旦触发，会把该位置整个 topk 向量都置为 -inf")

    # 打印部分样例前后对比
    show_b = min(B, 1)
    show_l = min(L, 6)
    show_h = min(h_hsa_kv, 3)
    for b in range(show_b):
        print("-" * 100)
        print(f"batch={b}")
        for l_idx in range(show_l):
            for h_idx in range(show_h):
                trig = bool(trigger_mask[b, l_idx, h_idx].item())
                inf_cnt = int(inf_per_slot[b, l_idx, h_idx].item())
                before_vec = scores_before[b, l_idx, h_idx].detach().cpu().tolist()
                after_vec = scores_after[b, l_idx, h_idx].detach().cpu().tolist()
                print(
                    f"(b={b}, l={l_idx}, h={h_idx}) trigger={trig}, inf_count={inf_cnt}/{topk}\n"
                    f"  before: {before_vec}\n"
                    f"  after : {after_vec}"
                )

    print("=" * 100)

    # 断言验证：
    # 1) 触发的 slot，topk 维应全部为 -inf
    triggered_ok = torch.all(inf_per_slot[trigger_mask] == topk) if trigger_count > 0 else True
    # 2) 未触发的 slot，不应该有 -inf（因为原始 scores_before 来自 randn）
    not_triggered_mask = ~trigger_mask
    not_triggered_ok = torch.all(inf_per_slot[not_triggered_mask] == 0)

    assert bool(triggered_ok), "触发 dropout 的位置未全部被置为 -inf"
    assert bool(not_triggered_ok), "未触发 dropout 的位置出现了意外 -inf"

    return {
        "scores_before": scores_before,
        "scores_after": scores_after,
        "do_dropout": do_dropout,
        "trigger_mask": trigger_mask,
        "inf_per_slot": inf_per_slot,
    }


def test_chunk_level_mask_projection_to_topk(
    B: int = 2,
    L: int = 6,
    h_hsa_kv: int = 3,
    S: int = 10,
    topk: int = 4,
    topk_dropout: float = 0.35,
    invalid_prob: float = 0.2,
    seed: int = 2027,
    device: str | None = None,
):
    """
    仅使用 torch 验证“先在 S 维做随机 mask，再投影到 topk 结果”的后处理逻辑：

        drop_s = (torch.rand(B, L, 1, S) < topk_dropout)
        drop_s = drop_s.expand(B, L, h_hsa_kv, S)
        valid = (indices >= 0)
        idx_safe = indices.clamp_min(0)
        drop_mask = torch.gather(drop_s, dim=-1, index=idx_safe) & valid
        scores = scores.masked_fill(drop_mask, -inf)

    目标：验证不同 token 可以 mask 0/1/2/... 个 topk 槽位（取决于其 indices 命中情况）。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 构造模拟 topk 输出
    # indices: [B, L, h_hsa_kv, topk], 值域 [0, S) 或 -1
    indices = torch.randint(0, S, (B, L, h_hsa_kv, topk), device=device, dtype=torch.int32)
    invalid_mask = torch.rand(B, L, h_hsa_kv, topk, device=device) < invalid_prob
    indices = torch.where(invalid_mask, torch.full_like(indices, -1), indices)

    scores_before = torch.randn(B, L, h_hsa_kv, topk, device=device, dtype=torch.float32)

    # 先在 S 维做随机 chunk mask（跨 head 共享）
    drop_s = (torch.rand(B, L, 1, S, device=device) < topk_dropout)
    drop_s = drop_s.expand(B, L, h_hsa_kv, S)

    # 映射到 topk 维
    valid = indices >= 0
    idx_safe = indices.clamp_min(0).to(torch.int64)
    gathered = torch.gather(drop_s, dim=-1, index=idx_safe)
    drop_mask = gathered & valid

    scores_after = scores_before.masked_fill(drop_mask, float("-inf"))

    # ---- 打印统计 ----
    masked_cnt_per_token = drop_mask.sum(dim=-1)  # [B, L, h]
    any_masked = (masked_cnt_per_token > 0).sum().item()
    total_slots = B * L * h_hsa_kv

    print("=" * 100)
    print("[test_chunk_level_mask_projection_to_topk] S维随机mask -> topk映射验证")
    print(f"device={device}, seed={seed}")
    print(f"indices.shape={tuple(indices.shape)}, S={S}, topk={topk}, topk_dropout={topk_dropout}")
    print(f"至少mask一个topk位置的slot数: {any_masked}/{total_slots}")
    print("说明：这里不会整条全清空；每个(b,l,h)被mask的topk个数可变（0/1/2/...）")

    show_b = min(B, 1)
    show_l = min(L, 6)
    show_h = min(h_hsa_kv, 3)
    for b in range(show_b):
        print("-" * 100)
        print(f"batch={b}")
        for l_idx in range(show_l):
            for h_idx in range(show_h):
                idx_vec = indices[b, l_idx, h_idx].detach().cpu().tolist()
                m_vec = drop_mask[b, l_idx, h_idx].detach().cpu().tolist()
                m_cnt = int(masked_cnt_per_token[b, l_idx, h_idx].item())
                before_vec = scores_before[b, l_idx, h_idx].detach().cpu().tolist()
                after_vec = scores_after[b, l_idx, h_idx].detach().cpu().tolist()
                print(
                    f"(b={b}, l={l_idx}, h={h_idx}) masked_count={m_cnt}/{topk}, indices={idx_vec}\n"
                    f"  drop_mask: {m_vec}\n"
                    f"  before   : {before_vec}\n"
                    f"  after    : {after_vec}"
                )

    print("=" * 100)

    # ---- 正确性断言 ----
    # 1) drop_mask 应严格等于“indices有效且对应S维位置被置True”
    expected_mask = torch.zeros_like(drop_mask)
    for b in range(B):
        for l_idx in range(L):
            for h_idx in range(h_hsa_kv):
                for k_idx in range(topk):
                    idx = int(indices[b, l_idx, h_idx, k_idx].item())
                    if idx >= 0:
                        expected_mask[b, l_idx, h_idx, k_idx] = bool(drop_s[b, l_idx, h_idx, idx].item())
    assert torch.equal(drop_mask, expected_mask), "drop_mask 与期望映射不一致"

    # 2) 被mask位置应为-inf，未mask位置应保持不变
    is_inf = torch.isinf(scores_after)
    assert torch.equal(is_inf, drop_mask), "-inf 分布与 drop_mask 不一致"

    unchanged_ok = torch.all(scores_after[~drop_mask] == scores_before[~drop_mask])
    assert bool(unchanged_ok), "未被mask的位置发生了意外变化"

    return {
        "indices": indices,
        "drop_s": drop_s,
        "drop_mask": drop_mask,
        "scores_before": scores_before,
        "scores_after": scores_after,
    }


def test_chunk_level_mask_projection_take_along_dim(
    B: int = 2,
    L: int = 6,
    h_hsa_kv: int = 3,
    S: int = 10,
    topk: int = 4,
    topk_dropout: float = 0.35,
    invalid_prob: float = 0.2,
    seed: int = 2028,
    device: str | None = None,
):
    """
    使用更简洁写法验证 S 维 mask 到 topk 的映射：

        valid = indices >= 0
        idx = indices.masked_fill(~valid, 0).long()
        drop_s = torch.rand(B, L, S) < p
        drop_mask = torch.take_along_dim(drop_s[:, :, None, :], idx, dim=-1) & valid

    目标：
    1) 验证该简洁写法与 gather+expand 写法数值一致
    2) 验证 masked_fill 后 -inf 分布与 drop_mask 一致
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    indices = torch.randint(0, S, (B, L, h_hsa_kv, topk), device=device, dtype=torch.int32)
    invalid_mask = torch.rand(B, L, h_hsa_kv, topk, device=device) < invalid_prob
    indices = torch.where(invalid_mask, torch.full_like(indices, -1), indices)

    scores_before = torch.randn(B, L, h_hsa_kv, topk, device=device, dtype=torch.float32)

    # --- 简洁写法 ---
    valid = indices >= 0
    idx = indices.masked_fill(~valid, 0).to(torch.int64)
    drop_s_compact = torch.rand(B, L, S, device=device) < topk_dropout
    drop_mask_compact = torch.take_along_dim(drop_s_compact[:, :, None, :], idx, dim=-1) & valid
    scores_after_compact = scores_before.masked_fill(drop_mask_compact, float("-inf"))

    # --- 参考写法（用于对照）---
    drop_s_ref = drop_s_compact[:, :, None, :].expand(B, L, h_hsa_kv, S)
    idx_safe_ref = indices.clamp_min(0).to(torch.int64)
    drop_mask_ref = torch.gather(drop_s_ref, dim=-1, index=idx_safe_ref) & valid
    scores_after_ref = scores_before.masked_fill(drop_mask_ref, float("-inf"))

    print("=" * 100)
    print("[test_chunk_level_mask_projection_take_along_dim] 简洁写法一致性验证")
    print(f"device={device}, seed={seed}")
    print(f"indices.shape={tuple(indices.shape)}, S={S}, topk={topk}, topk_dropout={topk_dropout}")

    masked_cnt_per_token = drop_mask_compact.sum(dim=-1)
    any_masked = int((masked_cnt_per_token > 0).sum().item())
    total_slots = B * L * h_hsa_kv
    print(f"至少mask一个topk位置的slot数: {any_masked}/{total_slots}")

    show_b = min(B, 1)
    show_l = min(L, 4)
    show_h = min(h_hsa_kv, 2)
    for b in range(show_b):
        print("-" * 100)
        print(f"batch={b}")
        for l_idx in range(show_l):
            for h_idx in range(show_h):
                idx_vec = indices[b, l_idx, h_idx].detach().cpu().tolist()
                m_vec = drop_mask_compact[b, l_idx, h_idx].detach().cpu().tolist()
                m_cnt = int(masked_cnt_per_token[b, l_idx, h_idx].item())
                print(
                    f"(b={b}, l={l_idx}, h={h_idx}) masked_count={m_cnt}/{topk}, indices={idx_vec}\n"
                    f"  drop_mask(compact): {m_vec}"
                )

    print("=" * 100)

    # 1) 映射一致
    assert torch.equal(drop_mask_compact, drop_mask_ref), "take_along_dim 与 gather+expand 映射不一致"

    # 2) 结果一致
    assert torch.equal(scores_after_compact, scores_after_ref), "两种写法 masked_fill 后结果不一致"

    # 3) -inf 分布一致
    is_inf = torch.isinf(scores_after_compact)
    assert torch.equal(is_inf, drop_mask_compact), "-inf 分布与 drop_mask 不一致"

    return {
        "indices": indices,
        "drop_s_compact": drop_s_compact,
        "drop_mask_compact": drop_mask_compact,
        "scores_before": scores_before,
        "scores_after_compact": scores_after_compact,
    }


def test_topk_slot_independent_dropout(
    B: int = 2,
    L: int = 6,
    h_hsa_kv: int = 3,
    topk: int = 4,
    topk_dropout: float = 0.35,
    seed: int = 2029,
    device: str | None = None,
):
    """
    验证最终简洁写法——每个 topk 槽位独立随机 mask，跨 head 共享：

        drop_mask = (torch.rand(B, L, 1, topk, device=scores.device) < topk_dropout)
        drop_mask = drop_mask.expand(B, L, h_hsa_kv, topk)
        scores = scores.masked_fill(drop_mask, float("-inf"))

    与旧写法（整条全清空）对比，验证：
    1) 每个 (b,l) 内所有 head 共享相同的 mask pattern
    2) 每个 topk 槽位独立被 mask（不会像旧写法一样整条全清空）
    3) 被 mask 位置为 -inf，未 mask 位置保持不变
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    scores_before = torch.randn(B, L, h_hsa_kv, topk, device=device, dtype=torch.float32)

    # ====== 最终简洁写法 ======
    drop_mask = (torch.rand(B, L, 1, topk, device=scores_before.device) < topk_dropout)
    drop_mask = drop_mask.expand(B, L, h_hsa_kv, topk)
    scores_after = scores_before.masked_fill(drop_mask, float("-inf"))
    # ==========================

    # 统计
    masked_per_slot = drop_mask.sum(dim=-1)  # [B, L, h_hsa_kv]
    total_masked = int(drop_mask.sum().item())
    total_elements = B * L * h_hsa_kv * topk
    mask_ratio = total_masked / max(total_elements, 1)

    # 检查跨 head 共享：所有 head 的 mask pattern 应该相同
    head0_mask = drop_mask[:, :, 0, :]  # [B, L, topk]
    cross_head_ok = True
    for h_idx in range(1, h_hsa_kv):
        if not torch.equal(drop_mask[:, :, h_idx, :], head0_mask):
            cross_head_ok = False
            break

    # 检查是否存在"部分 mask"（不是整条全清空也不是完全不 mask）
    partial_mask_count = 0
    all_mask_count = 0
    no_mask_count = 0
    total_slots = B * L * h_hsa_kv
    for b in range(B):
        for l_idx in range(L):
            for h_idx in range(h_hsa_kv):
                m_cnt = int(masked_per_slot[b, l_idx, h_idx].item())
                if m_cnt == 0:
                    no_mask_count += 1
                elif m_cnt == topk:
                    all_mask_count += 1
                else:
                    partial_mask_count += 1

    print("=" * 100)
    print("[test_topk_slot_independent_dropout] 每个topk槽位独立mask验证")
    print(f"device={device}, seed={seed}")
    print(f"scores.shape={tuple(scores_before.shape)}, topk_dropout={topk_dropout}")
    print(f"总mask元素: {total_masked}/{total_elements}, mask_ratio={mask_ratio:.4f}")
    print(f"跨head共享: {'✓' if cross_head_ok else '✗'}")
    print(f"slot统计: no_mask={no_mask_count}, partial_mask={partial_mask_count}, all_mask={all_mask_count} (total={total_slots})")
    print("说明：与旧写法不同，这里每个topk槽位独立决定mask，会出现部分mask的情况")

    show_b = min(B, 1)
    show_l = min(L, 6)
    show_h = min(h_hsa_kv, 3)
    for b in range(show_b):
        print("-" * 100)
        print(f"batch={b}")
        for l_idx in range(show_l):
            for h_idx in range(show_h):
                m_vec = drop_mask[b, l_idx, h_idx].detach().cpu().tolist()
                m_cnt = int(masked_per_slot[b, l_idx, h_idx].item())
                before_vec = scores_before[b, l_idx, h_idx].detach().cpu().tolist()
                after_vec = scores_after[b, l_idx, h_idx].detach().cpu().tolist()
                print(
                    f"(b={b}, l={l_idx}, h={h_idx}) masked_count={m_cnt}/{topk}\n"
                    f"  drop_mask: {m_vec}\n"
                    f"  before   : {before_vec}\n"
                    f"  after    : {after_vec}"
                )

    print("=" * 100)

    # ---- 断言 ----
    # 1) 跨 head 共享
    assert cross_head_ok, "不同 head 的 mask pattern 不一致，跨 head 共享失败"

    # 2) 被 mask 位置应为 -inf
    is_inf = torch.isinf(scores_after)
    assert torch.equal(is_inf, drop_mask), "-inf 分布与 drop_mask 不一致"

    # 3) 未 mask 位置应保持不变
    unchanged_ok = torch.all(scores_after[~drop_mask] == scores_before[~drop_mask])
    assert bool(unchanged_ok), "未被 mask 的位置发生了意外变化"

    # 4) 应该存在部分 mask 的情况（topk_dropout=0.35, topk=4 时大概率出现）
    assert partial_mask_count > 0, "没有出现部分mask的情况，可能逻辑有误"

    print("所有断言通过 ✓")

    return {
        "scores_before": scores_before,
        "scores_after": scores_after,
        "drop_mask": drop_mask,
        "masked_per_slot": masked_per_slot,
    }


if __name__ == "__main__":
    test_topk_dropout_mask_effect()
    test_chunk_level_mask_projection_to_topk()
    test_chunk_level_mask_projection_take_along_dim()
    test_topk_slot_independent_dropout()
