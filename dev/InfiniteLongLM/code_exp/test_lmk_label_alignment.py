"""
测试文件：验证 HSA 模型中 LMK 插入后 labels 对齐的正确性
以及 OpenCompass get_ppl 在 auto_insert_lmk 模式下是否能正确计算 PPL

不依赖 torch，纯用 Python 列表模拟所有操作。
"""

# ============================================================
# 模拟工具函数
# ============================================================

def insert_special_tokens(input_ids, fill_id, chunk_size):
    """模拟 landmark_utils.insert_special_tokens"""
    L = len(input_ids)
    full_chunks = L // (chunk_size - 1)
    remainder = L % (chunk_size - 1)
    
    parts = []
    for c in range(full_chunks):
        start = c * (chunk_size - 1)
        end = start + (chunk_size - 1)
        parts.extend(input_ids[start:end])
        parts.append(fill_id)
    
    if remainder > 0:
        parts.extend(input_ids[full_chunks * (chunk_size - 1):])
    
    return parts


def roll(lst, shifts):
    """模拟 torch.roll"""
    n = len(lst)
    shifts = shifts % n
    return lst[-shifts:] + lst[:-shifts]


# ============================================================
# Part 1: 验证训练时 labels 变换后各位置的 label
# ============================================================

def test_training_labels(chunk_size=64, seq_len=126):
    """
    验证训练时 labels 经过 roll(-1) -> set_last_-100 -> insert(-100) -> roll(1) 后，
    再经过 ForCausalLMLoss 内部的 shift 后，各位置的对齐关系。
    """
    print("=" * 80)
    print(f"Part 1: 训练时 labels 变换验证 (chunk_size={chunk_size}, seq_len={seq_len})")
    print("=" * 80)
    
    # 原始 input_ids 和 labels
    # 用 "t0", "t1", ... 表示 token
    original_input_ids = [f"t{i}" for i in range(seq_len)]
    original_labels = [f"t{i}" for i in range(seq_len)]
    
    LMK = "LMK"
    IGNORE = -100
    
    # --- 训练路径中 input_ids 的变换 ---
    internal_input_ids = insert_special_tokens(original_input_ids, LMK, chunk_size)
    
    # --- 训练路径中 labels 的变换 ---
    # Step 1: roll(-1)
    labels = roll(original_labels, -1)
    print(f"\nStep 1 - roll(-1): {labels[:5]} ... {labels[-3:]}")
    
    # Step 2: labels[-1] = -100
    labels[-1] = IGNORE
    print(f"Step 2 - set last to -100: {labels[:5]} ... {labels[-3:]}")
    
    # Step 3: insert_special_tokens(labels, -100, chunk_size)
    labels = insert_special_tokens(labels, IGNORE, chunk_size)
    print(f"Step 3 - insert -100: length={len(labels)}")
    
    # Step 4: roll(1)
    labels = roll(labels, 1)
    print(f"Step 4 - roll(1): length={len(labels)}")
    
    # --- 打印对齐表 ---
    print(f"\n内部序列长度: input_ids={len(internal_input_ids)}, labels={len(labels)}")
    assert len(internal_input_ids) == len(labels), "长度不匹配！"
    
    print(f"\n{'位置':>4} | {'input_ids':>8} | {'labels':>8} | 说明")
    print("-" * 60)
    
    lmk_positions = []
    for i in range(len(internal_input_ids)):
        inp = internal_input_ids[i]
        lab = labels[i]
        
        if inp == LMK:
            lmk_positions.append(i)
            note = "← LMK 位置"
            if lab == IGNORE:
                note += " (label=-100, 忽略)"
            else:
                note += f" (label={lab}, ⚠️ 不是-100!)"
        elif lab == IGNORE:
            note = "← label=-100 (忽略)"
        else:
            note = ""
        
        # 只打印关键位置（LMK 附近 ±2 和首尾）
        is_near_lmk = any(abs(i - p) <= 2 for p in lmk_positions) or any(abs(i - p) <= 2 for p in [len(internal_input_ids) - 1])
        is_boundary = i < 3 or i >= len(internal_input_ids) - 3
        
        if is_near_lmk or is_boundary:
            print(f"{i:>4} | {str(inp):>8} | {str(lab):>8} | {note}")
        elif i == 3:
            print(f"{'...':>4} | {'...':>8} | {'...':>8} |")
    
    # --- ForCausalLMLoss 内部的 shift ---
    print(f"\n--- ForCausalLMLoss 内部 shift 后的对齐 ---")
    print(f"shift_logits = logits[:-1]  (位置 0 到 {len(internal_input_ids)-2})")
    print(f"shift_labels = labels[1:]   (位置 1 到 {len(labels)-1})")
    
    shift_labels = labels[1:]  # labels[1:]
    
    print(f"\n{'i':>4} | {'logits[i]来自':>16} | {'shift_labels[i]':>16} | 说明")
    print("-" * 75)
    
    for i in range(len(shift_labels)):
        inp = internal_input_ids[i]  # logits[i] 来自 h[i]，h[i] 看到了 input_ids[0..i]
        lab = shift_labels[i]  # = labels[i+1]
        
        if inp == LMK:
            note = "← h[LMK位置] 的 logits"
        elif i + 1 < len(internal_input_ids) and internal_input_ids[i + 1] == LMK:
            note = "← LMK 前一个位置 (chunk 最后一个真实 token)"
        elif i > 0 and internal_input_ids[i - 1] == LMK:
            note = "← LMK 后一个位置 (下一个 chunk 第一个真实 token)"
        else:
            note = ""
        
        if lab == IGNORE:
            note += " [IGNORED]"
        
        # 只打印关键位置
        is_near_lmk = any(abs(i - p) <= 2 for p in lmk_positions)
        is_boundary = i < 3 or i >= len(shift_labels) - 3
        
        if is_near_lmk or is_boundary:
            context = f"h[{i}](看到..{inp})"
            print(f"{i:>4} | {context:>16} | {str(lab):>16} | {note}")
        elif i == 3:
            print(f"{'...':>4} | {'...':>16} | {'...':>16} |")
    
    # --- 验证关键位置 ---
    print(f"\n--- 关键位置验证 ---")
    for lmk_pos in lmk_positions:
        prev_pos = lmk_pos - 1  # LMK 前一个位置
        next_pos = lmk_pos + 1 if lmk_pos + 1 < len(shift_labels) else None
        
        print(f"\nLMK 在内部位置 {lmk_pos}:")
        
        # LMK 前一个位置
        if prev_pos >= 0 and prev_pos < len(shift_labels):
            lab = shift_labels[prev_pos]
            print(f"  位置 {prev_pos} (LMK前): logits[{prev_pos}] 来自 h[{prev_pos}](看到..{internal_input_ids[prev_pos]})")
            print(f"    shift_labels[{prev_pos}] = labels[{prev_pos+1}] = {lab}")
            if lab != IGNORE:
                print(f"    ✅ 模型在此位置被训练预测 {lab}")
            else:
                print(f"    ⚠️ 此位置被忽略")
        
        # LMK 位置本身
        if lmk_pos < len(shift_labels):
            lab = shift_labels[lmk_pos]
            print(f"  位置 {lmk_pos} (LMK): logits[{lmk_pos}] 来自 h[{lmk_pos}](看到..{internal_input_ids[lmk_pos]})")
            print(f"    shift_labels[{lmk_pos}] = labels[{lmk_pos+1}] = {lab}")
            if lab == IGNORE:
                print(f"    ✅ LMK 位置被忽略 (label=-100)")
            else:
                print(f"    ⚠️ LMK 位置没有被忽略! label={lab}")
        
        # LMK 后一个位置
        if next_pos is not None and next_pos < len(shift_labels):
            lab = shift_labels[next_pos]
            print(f"  位置 {next_pos} (LMK后): logits[{next_pos}] 来自 h[{next_pos}](看到..{internal_input_ids[next_pos]})")
            print(f"    shift_labels[{next_pos}] = labels[{next_pos+1}] = {lab}")
            if lab != IGNORE:
                print(f"    ✅ 模型在此位置被训练预测 {lab}")
            else:
                print(f"    ⚠️ 此位置被忽略")


# ============================================================
# Part 2: 验证 auto_insert_lmk 推理路径下 OpenCompass get_ppl 的对齐
# ============================================================

def test_opencompass_ppl_alignment(chunk_size=64, seq_len=126, mask_length=None):
    """
    模拟 OpenCompass get_ppl 在 auto_insert_lmk 模式下的计算过程，
    验证 shift_logits 和 shift_labels 的对齐是否正确。
    
    mask_length: 如果提供，模拟 OpenCompass 计算选项 PPL 时的 mask
    """
    print("\n" + "=" * 80)
    print(f"Part 2: OpenCompass get_ppl 对齐验证 (chunk_size={chunk_size}, seq_len={seq_len}, mask_length={mask_length})")
    print("=" * 80)
    
    # 原始 input_ids（OpenCompass 传给模型的）
    original_input_ids = [f"t{i}" for i in range(seq_len)]
    
    LMK = "LMK"
    
    # --- auto_insert_lmk 路径 ---
    # Step 1: 插入 LMK
    internal_input_ids = insert_special_tokens(original_input_ids, LMK, chunk_size)
    new_seq_len = len(internal_input_ids)
    
    # Step 2: 计算 non_lmk_mask
    non_lmk_mask = [not (i % chunk_size == chunk_size - 1) for i in range(new_seq_len)]
    
    # Step 3: 模型 forward 得到 hidden_states (长度 = new_seq_len)
    # 模拟 hidden_states 的来源标记
    internal_hidden = [f"h[{i}]" for i in range(new_seq_len)]
    
    # Step 4: _filter_lmk_hidden_states 过滤
    filtered_hidden = [internal_hidden[i] for i in range(new_seq_len) if non_lmk_mask[i]]
    
    # Step 5: lm_head 得到 logits (长度 = len(filtered_hidden))
    logits_source = filtered_hidden  # logits[i] 来自 filtered_hidden[i]
    
    print(f"\n原始 input_ids 长度: {seq_len}")
    print(f"插入 LMK 后内部长度: {new_seq_len}")
    print(f"过滤 LMK 后 logits 长度: {len(logits_source)}")
    print(f"长度是否与原始一致: {len(logits_source) == seq_len} {'✅' if len(logits_source) == seq_len else '❌'}")
    
    # --- OpenCompass get_ppl 的 shift ---
    # outputs = self.model(**tokens)[0]  → logits, shape (1, seq_len, vocab_size)
    # shift_logits = outputs[:, :-1, :]  → logits[0..seq_len-2]
    # shift_labels = tokens['input_ids'][:, 1:]  → original_input_ids[1..seq_len-1]
    
    shift_logits_source = logits_source[:-1]  # 长度 seq_len - 1
    shift_labels = original_input_ids[1:]      # 长度 seq_len - 1
    
    print(f"\nshift_logits 长度: {len(shift_logits_source)}")
    print(f"shift_labels 长度: {len(shift_labels)}")
    
    # --- 对齐分析 ---
    print(f"\n{'i':>4} | {'logits来源':>12} | {'内部看到的token':>16} | {'shift_label':>12} | {'训练时label':>12} | 匹配?")
    print("-" * 90)
    
    # 同时计算训练时的 labels 来做对比
    train_labels = [f"t{i}" for i in range(seq_len)]
    train_labels = roll(train_labels, -1)
    train_labels[-1] = -100
    train_labels = insert_special_tokens(train_labels, -100, chunk_size)
    train_labels = roll(train_labels, 1)
    # ForCausalLMLoss shift 后: shift_train_labels[i] = train_labels[i+1]
    shift_train_labels = train_labels[1:]
    
    # 建立 filtered 位置到内部位置的映射
    filtered_to_internal = [i for i in range(new_seq_len) if non_lmk_mask[i]]
    
    mismatch_count = 0
    ignored_count = 0
    
    for i in range(len(shift_logits_source)):
        internal_pos = filtered_to_internal[i]
        logits_src = logits_source[i]
        oc_label = shift_labels[i]  # OpenCompass 认为的 label
        
        # 训练时这个内部位置的 label
        train_lab = shift_train_labels[internal_pos] if internal_pos < len(shift_train_labels) else "N/A"
        
        # 判断是否匹配
        if train_lab == -100:
            match_str = "IGNORED_IN_TRAIN"
            ignored_count += 1
        elif str(oc_label) == str(train_lab):
            match_str = "✅"
        else:
            match_str = f"❌ 不匹配!"
            mismatch_count += 1
        
        # 判断是否是关键位置
        is_near_lmk = (internal_pos + 1 < new_seq_len and internal_input_ids[internal_pos + 1] == LMK) or \
                       (internal_pos > 0 and internal_input_ids[internal_pos - 1] == LMK)
        is_boundary = i < 3 or i >= len(shift_logits_source) - 3
        
        if is_near_lmk or is_boundary:
            seen_token = internal_input_ids[internal_pos]
            print(f"{i:>4} | {logits_src:>12} | {f'..{seen_token}':>16} | {str(oc_label):>12} | {str(train_lab):>12} | {match_str}")
        elif i == 3:
            print(f"{'...':>4} | {'...':>12} | {'...':>16} | {'...':>12} | {'...':>12} |")
    
    print(f"\n总结:")
    print(f"  总位置数: {len(shift_logits_source)}")
    print(f"  OpenCompass label 与训练 label 匹配: {len(shift_logits_source) - mismatch_count - ignored_count}")
    print(f"  训练时被忽略的位置: {ignored_count}")
    print(f"  不匹配的位置: {mismatch_count}")
    
    if mismatch_count > 0:
        print(f"  ❌ 存在 {mismatch_count} 个位置的 label 不匹配！")
    else:
        print(f"  ✅ 所有非忽略位置的 label 都匹配！")
    
    if ignored_count > 0:
        print(f"  ⚠️ 有 {ignored_count} 个位置在训练时被忽略 (label=-100)，但 OpenCompass 仍然计算了这些位置的 loss")
        print(f"     这些位置占比: {ignored_count}/{len(shift_logits_source)} = {ignored_count/len(shift_logits_source)*100:.2f}%")
        
        # 详细列出被忽略的位置
        print(f"\n  被忽略位置的详细信息:")
        for i in range(len(shift_logits_source)):
            internal_pos = filtered_to_internal[i]
            train_lab = shift_train_labels[internal_pos] if internal_pos < len(shift_train_labels) else "N/A"
            if train_lab == -100:
                oc_label = shift_labels[i]
                seen_token = internal_input_ids[internal_pos]
                # 判断这个位置在内部序列中的角色
                if internal_pos > 0 and internal_input_ids[internal_pos - 1] == LMK:
                    role = "LMK 后第一个真实 token"
                elif internal_pos + 1 < new_seq_len and internal_input_ids[internal_pos + 1] == LMK:
                    role = "LMK 前最后一个真实 token"  
                else:
                    role = "其他"
                print(f"    外部位置 {i}, 内部位置 {internal_pos}, token={seen_token}, "
                      f"OC_label={oc_label}, 角色={role}")
    
    # --- 如果有 mask_length，模拟选项 PPL 计算 ---
    if mask_length is not None:
        print(f"\n--- 模拟 OpenCompass 选项 PPL (mask_length={mask_length}) ---")
        print(f"只计算最后 {mask_length} 个 token 的 loss")
        
        # mask: 从 mask_length-1 开始到末尾为 1
        mask_start = len(shift_labels) - mask_length
        
        print(f"选项区域: 外部位置 {mask_start} 到 {len(shift_labels)-1}")
        
        option_has_ignored = False
        for i in range(mask_start, len(shift_labels)):
            internal_pos = filtered_to_internal[i]
            train_lab = shift_train_labels[internal_pos] if internal_pos < len(shift_train_labels) else "N/A"
            oc_label = shift_labels[i]
            seen_token = internal_input_ids[internal_pos]
            
            if train_lab == -100:
                option_has_ignored = True
                print(f"  ⚠️ 外部位置 {i}: token={seen_token}, OC_label={oc_label}, "
                      f"训练时label=-100 (此位置的 loss 不可靠!)")
            else:
                print(f"  ✅ 外部位置 {i}: token={seen_token}, OC_label={oc_label}, "
                      f"训练时label={train_lab}")
        
        if option_has_ignored:
            print(f"\n  ⚠️ 选项区域中存在训练时被忽略的位置，PPL 计算可能不准确！")
        else:
            print(f"\n  ✅ 选项区域中所有位置在训练时都有有效 label")


# ============================================================
# Part 3: 测试不同序列长度
# ============================================================

def test_various_lengths(chunk_size=64):
    """测试各种序列长度下的对齐情况"""
    print("\n" + "=" * 80)
    print(f"Part 3: 不同序列长度测试 (chunk_size={chunk_size})")
    print("=" * 80)
    
    test_cases = [
        (50, "短于一个 chunk (无 LMK 插入)"),
        (63, "恰好一个 chunk (1 × 63)"),
        (64, "一个 chunk + 1 余数"),
        (100, "一个完整 chunk + 37 余数"),
        (126, "恰好两个 chunk (2 × 63)"),
        (189, "恰好三个 chunk (3 × 63)"),
        (200, "三个完整 chunk + 11 余数"),
    ]
    
    for seq_len, desc in test_cases:
        print(f"\n--- seq_len={seq_len}: {desc} ---")
        
        original = [f"t{i}" for i in range(seq_len)]
        LMK = "LMK"
        
        internal = insert_special_tokens(original, LMK, chunk_size)
        new_len = len(internal)
        non_lmk_mask = [not (i % chunk_size == chunk_size - 1) for i in range(new_len)]
        filtered_len = sum(non_lmk_mask)
        
        num_lmk = new_len - seq_len
        
        # 计算训练时被忽略的位置数
        train_labels = [f"t{i}" for i in range(seq_len)]
        train_labels = roll(train_labels, -1)
        train_labels[-1] = -100
        train_labels = insert_special_tokens(train_labels, -100, chunk_size)
        train_labels = roll(train_labels, 1)
        shift_train_labels = train_labels[1:]
        
        # 过滤后的位置映射
        filtered_to_internal = [i for i in range(new_len) if non_lmk_mask[i]]
        
        # OpenCompass shift_labels
        shift_labels_oc = original[1:]
        
        ignored_in_option = 0
        mismatch = 0
        for i in range(len(shift_labels_oc)):
            internal_pos = filtered_to_internal[i]
            train_lab = shift_train_labels[internal_pos] if internal_pos < len(shift_train_labels) else "N/A"
            oc_lab = shift_labels_oc[i]
            
            if train_lab == -100:
                ignored_in_option += 1
            elif str(oc_lab) != str(train_lab):
                mismatch += 1
        
        status = "✅" if mismatch == 0 else "❌"
        print(f"  内部长度: {new_len}, LMK数: {num_lmk}, 过滤后: {filtered_len}, "
              f"匹配原始: {filtered_len == seq_len}")
        print(f"  {status} 不匹配: {mismatch}, 训练时忽略但OC计算: {ignored_in_option}/{len(shift_labels_oc)} "
              f"({ignored_in_option/max(len(shift_labels_oc),1)*100:.1f}%)")
        
        # 列出被忽略的位置
        if ignored_in_option > 0:
            ignored_positions = []
            for i in range(len(shift_labels_oc)):
                internal_pos = filtered_to_internal[i]
                train_lab = shift_train_labels[internal_pos] if internal_pos < len(shift_train_labels) else "N/A"
                if train_lab == -100:
                    role = ""
                    if internal_pos > 0 and internal[internal_pos - 1] == LMK:
                        role = "LMK后"
                    elif internal_pos + 1 < new_len and internal[internal_pos + 1] == LMK:
                        role = "LMK前最后"
                    ignored_positions.append(f"外部{i}(内部{internal_pos},{role})")
            print(f"  忽略位置: {', '.join(ignored_positions)}")


# ============================================================
# Part 4: 模拟 OpenCompass 选项 PPL 场景
# ============================================================

def test_option_ppl_scenario(chunk_size=64):
    """
    模拟 OpenCompass 中 MMLU 等任务计算选项 PPL 的场景。
    
    典型场景：prompt + 选项 A/B/C/D，计算每个选项的 PPL。
    prompt 长度固定，选项长度不同，关键是选项部分是否跨越 chunk 边界。
    """
    print("\n" + "=" * 80)
    print(f"Part 4: 模拟 OpenCompass 选项 PPL 场景 (chunk_size={chunk_size})")
    print("=" * 80)
    
    # 模拟不同的 prompt + option 组合
    scenarios = [
        # (prompt_len, option_len, description)
        (60, 3, "选项在第一个 chunk 内，不跨越边界"),
        (61, 3, "选项跨越第一个 chunk 边界 (prompt 末尾在 pos 60, LMK 在 pos 63)"),
        (62, 3, "选项从 chunk 边界附近开始"),
        (63, 3, "prompt 恰好填满一个 chunk"),
        (120, 10, "选项跨越第二个 chunk 边界"),
        (125, 5, "选项在第二个 chunk 末尾"),
    ]
    
    for prompt_len, option_len, desc in scenarios:
        total_len = prompt_len + option_len
        print(f"\n--- {desc} ---")
        print(f"  prompt_len={prompt_len}, option_len={option_len}, total_len={total_len}")
        
        original = [f"t{i}" for i in range(total_len)]
        LMK = "LMK"
        
        internal = insert_special_tokens(original, LMK, chunk_size)
        new_len = len(internal)
        non_lmk_mask = [not (i % chunk_size == chunk_size - 1) for i in range(new_len)]
        filtered_to_internal = [i for i in range(new_len) if non_lmk_mask[i]]
        
        # 训练时的 labels
        train_labels = [f"t{i}" for i in range(total_len)]
        train_labels = roll(train_labels, -1)
        train_labels[-1] = -100
        train_labels = insert_special_tokens(train_labels, -100, chunk_size)
        train_labels = roll(train_labels, 1)
        shift_train_labels = train_labels[1:]
        
        # OpenCompass 的 mask_length = option_len
        # 只计算最后 option_len 个 token 的 loss
        # mask 从 mask_length-1 开始（即 shift 后的位置 total_len - 1 - option_len 到 total_len - 2）
        # 实际上 OpenCompass 的 mask 逻辑是：
        #   for j in range(mask_length[i] - 1, len(mask[i])):
        #       mask[i][j] = 1
        # shift_labels 长度 = total_len - 1
        # mask 从 option_len - 1 开始到 total_len - 2
        # 等等，这里 mask_length 是从末尾算的
        # 实际上 mask_length 是选项的 token 数
        # mask 从 (total_len - 1) - option_len 开始... 不对
        # 让我重新看 OpenCompass 的代码：
        #   for j in range(mask_length[i] - 1, len(mask[i])):
        #       mask[i][j] = 1
        # 这里 mask_length[i] - 1 是起始位置
        # 但 mask_length 是什么？是 get_token_len(cont) 即选项的 token 长度
        # 而 shift_labels 的长度是 total_len - 1
        # 所以 mask 从位置 option_len - 1 到 total_len - 2 都是 1
        # 等等这不对，应该是从末尾开始的
        
        # 重新理解：OpenCompass 的 get_loglikelihood 调用 get_ppl(inputs, mask_length)
        # inputs = prompt + option (完整字符串)
        # mask_length = option 的 token 数
        # 在 get_ppl 中：
        #   mask = zeros(batch, total_len - 1)
        #   for j in range(mask_length - 1, total_len - 1):
        #       mask[j] = 1
        # 这意味着 mask 从位置 mask_length - 1 开始到末尾都是 1
        # 但这不对... mask_length 是选项长度，比如 3
        # 那 mask 从位置 2 开始到末尾都是 1？这会包含 prompt 部分的 loss
        
        # 等等，我再仔细看：
        # loss = loss * mask
        # ce_loss = loss.sum(-1) / lens
        # lens -= mask_length
        # 所以 lens = total_len - mask_length (非 pad token 数 - mask_length)
        
        # 不对，让我重新理解。mask_length 的含义是"前 mask_length 个 token 被 mask 掉"
        # 即 prompt 部分被 mask，只计算选项部分的 loss
        # 但代码是 for j in range(mask_length - 1, len(mask)):
        #     mask[j] = 1
        # 这意味着从位置 mask_length - 1 开始到末尾都是 1
        # 如果 mask_length = prompt_len，那么从 prompt_len - 1 开始到末尾都是 1
        # 但实际上 mask_length = option_len (选项的 token 数)
        
        # 让我再看一遍代码...
        # get_loglikelihood:
        #   mask_length = [get_token_len(c) for c in conts]  # conts 是选项文本
        #   return -get_ppl(inputs, mask_length)
        # 
        # get_ppl 中:
        #   for j in range(mask_length[i] - 1, len(mask[i])):
        #       mask[i][j] = 1
        #   loss = loss * mask
        #   lens -= mask_length
        
        # 如果 option_len = 3, total_len = 63+3 = 66
        # shift 后长度 = 65
        # mask 从位置 2 到 64 都是 1 → 这包含了 prompt 部分！
        
        # 这看起来不对... 让我再想想
        # 哦等等，mask_length 的语义是"要 mask 掉的前缀长度"
        # 不，看代码：lens -= mask_length
        # 如果 lens = 66 (非 pad 数), lens -= 3 = 63
        # loss 只在位置 2 到 64 有值
        # ce_loss = sum(loss) / 63
        
        # 这意味着 loss 包含了从位置 2 开始的所有 token
        # 但 prompt 部分也在里面...
        
        # 不对！我搞混了。让我重新理解：
        # mask_length 是选项的 token 数
        # 但 mask 的含义是"只保留最后 mask_length 个 token 的 loss"
        # 不，代码是 for j in range(mask_length - 1, len(mask)):
        # 如果 mask_length = 3, len(mask) = 65
        # 那 mask[2:65] = 1，这包含了几乎所有位置
        
        # 我觉得这个理解有问题。让我换个角度：
        # 实际上 OpenCompass 的 PPL 任务中，mask_length 是 prompt 的 token 数
        # 不，get_loglikelihood 中 mask_length = get_token_len(cont)
        # cont 是选项文本，所以 mask_length = 选项的 token 数
        
        # 好吧，让我直接按代码逻辑来：
        # mask_length = option_len = 3
        # mask = zeros(65)
        # mask[2:65] = 1  (从 mask_length-1=2 开始)
        # loss = loss * mask  (只保留位置 2 到 64 的 loss)
        # lens = 66 - 3 = 63
        # ce_loss = sum(loss[2:65]) / 63
        
        # 这确实包含了 prompt 部分的 loss！这看起来像是 OpenCompass 的设计
        # 但 lens 减去了 mask_length，所以分母是 63 而不是 66
        
        # 嗯，但这样的话 mask_length 的作用就是"跳过前 mask_length-1 个位置"
        # 对于选项 PPL，这意味着跳过选项的前几个 token？
        
        # 我觉得我理解错了。让我重新看：
        # 在 PPL inferencer 中，inputs 是完整的 "prompt + option"
        # mask_length 是 prompt 的 token 数（不是选项的）
        # 不，get_loglikelihood 中 mask_length = get_token_len(cont)
        # cont 是选项文本
        
        # 算了，这个分析太复杂了，让我简化
        # 关键问题是：选项部分的 token 是否跨越了 chunk 边界
        
        # 选项部分在原始序列中的位置: prompt_len 到 total_len - 1
        # 在 shift 后: prompt_len - 1 到 total_len - 2
        
        option_start_in_shift = prompt_len - 1  # shift 后选项开始的位置
        option_end_in_shift = total_len - 2      # shift 后选项结束的位置
        
        print(f"  选项在 shift 后的位置范围: [{option_start_in_shift}, {option_end_in_shift}]")
        
        has_issue = False
        for i in range(option_start_in_shift, option_end_in_shift + 1):
            if i >= len(filtered_to_internal):
                break
            internal_pos = filtered_to_internal[i]
            train_lab = shift_train_labels[internal_pos] if internal_pos < len(shift_train_labels) else "N/A"
            oc_lab = original[i + 1]  # shift_labels[i] = original_input_ids[i+1]
            
            if train_lab == -100:
                has_issue = True
                role = ""
                if internal_pos > 0 and internal[internal_pos - 1] == LMK:
                    role = "(LMK后第一个token)"
                print(f"    ⚠️ 位置 {i}: OC预测 {oc_lab}, 但训练时此位置 label=-100 {role}")
            elif str(oc_lab) != str(train_lab):
                has_issue = True
                print(f"    ❌ 位置 {i}: OC预测 {oc_lab}, 训练时预测 {train_lab}")
        
        if not has_issue:
            print(f"  ✅ 选项区域所有位置对齐正确")


# ============================================================
# 运行所有测试
# ============================================================

if __name__ == "__main__":
    # Part 1: 训练时 labels 变换验证
    test_training_labels(chunk_size=64, seq_len=126)
    
    # Part 2: OpenCompass get_ppl 对齐验证
    test_opencompass_ppl_alignment(chunk_size=64, seq_len=126)
    
    # Part 3: 不同序列长度测试
    test_various_lengths(chunk_size=64)
    
    # Part 4: 模拟选项 PPL 场景
    test_option_ppl_scenario(chunk_size=64)
