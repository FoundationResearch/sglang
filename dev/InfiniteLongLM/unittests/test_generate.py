"""
测试 HSAForCausalLM 的 generate 功能。
对比实验：generate 的 greedy decode 输出 vs 全量 forward 的 argmax 输出。

验证逻辑：
1. 用 model.generate() 对 prompt 做 greedy decode，得到输出 tokens
2. 把 prompt + generated tokens 拼起来，外部插入 LMK，调用 model.forward() 全量 prefill
3. 外部剔除 LMK 位置的 logits，做 argmax（带 shift），对比是否与 generate 输出一致

参考：eval/eval_ppl_hf.py 中的评测方式，LMK 的插入和剔除完全在模型外部实现
"""

import sys
import os
import math
import argparse
import torch
import torch.nn.functional as F

# 将项目根目录加入 sys.path，以便导入 models 和 utils
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 注册自定义模型到 transformers
from models.FlashHSA.configuration_hsa import HSAConfig
from models.FlashHSA.modeling_qwen_lhsa import HSAForCausalLM

# checkpoint 的 config.json 中 model_type="flash_hsa"，需要覆盖 HSAConfig 以匹配
HSAConfig.model_type = "flash_hsa"
AutoConfig.register("flash_hsa", HSAConfig)
HSAForCausalLM.config_class = HSAConfig
AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM)

from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks


DEFAULT_CKPT_PATH = (
    "/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/hsa-interleave-win512-hsadisturb0.2-hsadropout0.2-8KA512-unified-ruler-5per-single/checkpoints/global_step_30000/hf_ckpt"
)


def compute_ppl_from_logits(logits, target_ids):
    """
    从 logits 和 target token ids 计算 PPL。
    
    Args:
        logits: [gen_len, vocab_size] 的 logits，logits[i] 对应预测 target_ids[i]
        target_ids: [gen_len] 的 target token ids
    
    Returns:
        ppl: 困惑度
        avg_nll: 平均负对数似然
    """
    # 计算交叉熵（逐 token）
    log_probs = F.log_softmax(logits.float(), dim=-1)  # [gen_len, vocab_size]
    # 取每个位置对应 target token 的 log probability
    target_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [gen_len]
    # 平均负对数似然
    avg_nll = -target_log_probs.mean().item()
    ppl = math.exp(avg_nll)
    return ppl, avg_nll


def run_generate(model, tokenizer, prompt, device, max_new_tokens):
    """
    运行 model.generate()，greedy decode。
    返回 (input_ids, generated_token_ids, gen_scores)。
    其中 gen_scores 是每步 decode 的 logits（用于计算 PPL）。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decode
            output_scores=True,  # 返回每步的 logits
            return_dict_in_generate=True,
        )

    generated_ids = output.sequences[0, input_len:]  # 只取新生成的部分
    # output.scores 是一个 tuple，每个元素是 [batch_size, vocab_size] 的 logits
    # 拼接成 [gen_len, vocab_size]
    gen_scores = torch.stack(output.scores, dim=0).squeeze(1)  # [gen_len, vocab_size]
    return input_ids, generated_ids, gen_scores


def run_forward_baseline(model, tokenizer, input_ids, generated_ids, device):
    """
    全量 forward 对照组（参考 eval/eval_ppl_hf.py 的方式）：
    LMK 的插入和剔除完全在模型外部实现，这是最权威的参考基准。
    
    流程：
    1. 拼接 prompt + generated tokens
    2. 外部 insert_special_tokens 插入 LMK
    3. 外部 create_position_ids_with_landmarks 创建 position_ids
    4. 关闭 auto_insert_lmk，调用 model.forward() 得到含 LMK 的 logits
    5. 外部剔除 LMK 位置的 logits
    6. 做 argmax（带 causal shift），返回预测 tokens
    
    关键点：
    - 必须关闭 auto_insert_lmk 和 _gen_state.active，防止模型内部重复插入或过滤
    - 必须使用 use_cache=True，和 eval_ppl_hf.py 保持一致，
      否则 past_key_values 为 None，导致 LandmarkHSA 中选择训练版本的
      compiled_flex_attention（dynamic=False），而非推理版本
    """
    chunk_size = model.chunk_size
    lmk_id = model.lmk_id
    prompt_len = input_ids.shape[1]
    gen_len = generated_ids.shape[0]

    # 1. 拼接完整序列：prompt + generated
    full_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)  # [1, prompt_len + gen_len]
    full_len = full_ids.shape[1]
    print(f"  [Forward] 原始序列长度: {full_len} (prompt={prompt_len}, gen={gen_len})")

    # 2. 外部插入 LMK token（和 eval_ppl_hf.py 一致）
    full_ids_with_lmk = insert_special_tokens(full_ids, lmk_id, chunk_size)
    position_ids = create_position_ids_with_landmarks(None, full_len, chunk_size, device)
    new_seq_len = full_ids_with_lmk.shape[1]
    print(f"  [Forward] 插入 LMK 后序列长度: {new_seq_len}")

    # 3. 构建 LMK 位置的 mask（用于后续剔除 logits）
    #    insert_special_tokens 在每 (chunk_size-1) 个 real token 后插入一个 LMK
    #    LMK 位于 index % chunk_size == chunk_size - 1 的位置
    #    最后不完整 chunk 没有 LMK（remainder < chunk_size，不会触发该条件）
    pos_indices = torch.arange(new_seq_len, device=device)
    is_lmk = (pos_indices % chunk_size == chunk_size - 1)
    non_lmk_mask = ~is_lmk
    num_lmk = is_lmk.sum().item()
    num_non_lmk = non_lmk_mask.sum().item()
    print(f"  [Forward] LMK 数量: {num_lmk}, 非 LMK 数量: {num_non_lmk}")
    assert num_non_lmk == full_len, f"非 LMK 数量 {num_non_lmk} != 原始序列长度 {full_len}"

    # 4. 关闭 auto_insert_lmk 和 _gen_state.active，防止模型内部重复处理
    saved_auto_insert_lmk = model.auto_insert_lmk
    saved_gen_state_active = model._gen_state.active
    saved_lmk_positions = model._gen_state.lmk_positions_in_input
    model.auto_insert_lmk = False
    model._gen_state.active = False
    model._gen_state.lmk_positions_in_input = None

    # 使用 use_cache=True，和 eval_ppl_hf.py 保持一致
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids_with_lmk,
            position_ids=position_ids,
            use_cache=True,
            attention_mask=None,
        )

    # 恢复状态
    model.auto_insert_lmk = saved_auto_insert_lmk
    model._gen_state.active = saved_gen_state_active
    model._gen_state.lmk_positions_in_input = saved_lmk_positions

    # 5. 从 logits 中外部剔除 LMK 位置（和 eval_ppl_hf.py 中处理 label 的方式对应）
    logits = outputs.logits  # [1, new_seq_len, vocab_size]
    logits_no_lmk = logits[:, non_lmk_mask, :]  # [1, full_len, vocab_size]
    print(f"  [Forward] 剔除 LMK 后 logits 形状: {logits_no_lmk.shape}")

    # 6. Causal LM shift: logits[i] 预测 token[i+1]
    #    要预测 generated tokens，需要取 logits[prompt_len-1 : prompt_len+gen_len-1]
    pred_logits = logits_no_lmk[:, prompt_len - 1 : prompt_len + gen_len - 1, :]  # [1, gen_len, vocab_size]
    pred_tokens = pred_logits.argmax(dim=-1).squeeze(0)  # [gen_len]

    # 同时返回 pred_logits 用于计算 PPL
    return pred_tokens, pred_logits.squeeze(0)  # pred_logits: [gen_len, vocab_size]


def compare_results(generated_ids, pred_tokens, tokenizer):
    """
    对比 generate 输出和 forward argmax 输出。
    """
    gen_len = generated_ids.shape[0]
    pred_len = pred_tokens.shape[0]
    compare_len = min(gen_len, pred_len)

    match = (generated_ids[:compare_len] == pred_tokens[:compare_len])
    match_count = match.sum().item()
    total = compare_len

    print(f"\n  [对比结果] 总 token 数: {total}, 匹配数: {match_count}, 匹配率: {match_count/total*100:.2f}%")

    if match_count == total:
        print("  ✅ 完全匹配！Generate 和 Forward 的输出一致。")
    else:
        print("  ❌ 存在不匹配！")
        mismatch_indices = torch.where(~match)[0]
        print(f"  不匹配位置 (最多显示前10个): {mismatch_indices[:10].tolist()}")
        for idx in mismatch_indices[:10]:
            idx = idx.item()
            gen_tok = generated_ids[idx].item()
            pred_tok = pred_tokens[idx].item()
            gen_text = tokenizer.decode([gen_tok])
            pred_text = tokenizer.decode([pred_tok])
            print(f"    位置 {idx}: generate={gen_tok}('{gen_text}') vs forward={pred_tok}('{pred_text}')")

    return match_count == total


def compare_ppl(gen_ppl, gen_nll, fwd_ppl, fwd_nll):
    """
    对比 generate 和 forward 的 PPL。
    """
    print(f"\n  [PPL 对比]")
    print(f"    Generate PPL: {gen_ppl:.6f} (avg NLL: {gen_nll:.6f})")
    print(f"    Forward  PPL: {fwd_ppl:.6f} (avg NLL: {fwd_nll:.6f})")
    
    ppl_diff = abs(gen_ppl - fwd_ppl)
    nll_diff = abs(gen_nll - fwd_nll)
    # 使用相对误差判断，因为 bf16 精度下可能有微小差异
    rel_diff = ppl_diff / max(gen_ppl, fwd_ppl, 1e-8)
    
    print(f"    PPL 绝对差: {ppl_diff:.6f}, 相对差: {rel_diff:.6f}")
    print(f"    NLL 绝对差: {nll_diff:.6f}")
    
    # 允许 bf16 精度下的微小误差（相对误差 < 1%）
    ppl_threshold = 0.01
    if rel_diff < ppl_threshold:
        print(f"    ✅ PPL 一致！(相对差 {rel_diff:.6f} < 阈值 {ppl_threshold})")
        return True
    else:
        print(f"    ❌ PPL 不一致！(相对差 {rel_diff:.6f} >= 阈值 {ppl_threshold})")
        return False


def main(args):
    device = torch.device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # ---- 加载 tokenizer ----
    tokenizer_path = args.tokenizer_path or args.checkpoint_path
    print(f"[INFO] 加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 加载模型 ----
    print(f"[INFO] 加载模型: {args.checkpoint_path}")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device,
    }
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **model_kwargs)
    model.eval()
    print(f"[INFO] 模型加载完成，chunk_size={model.chunk_size}, lmk_id={model.lmk_id}")
    print(f"[INFO] auto_insert_lmk={model.auto_insert_lmk}")
    print(f"[INFO] 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ---- 准备测试 prompts ----
    prompts = args.prompts if args.prompts else [
        "Once a year I make the drive back to my hometown of Shreveport, Louisiana. My journey begins as the sun rises over our nation’s capital. Before long I’m moving through smaller cities that claim tobacco and the Confederate flag as symbols of pride, wondering how long it will be before the smell of factory smoke is replaced by the fertile aroma of livestock and chicken flocks. Then the narrow roads begin to unwind--hugged on either side by pastures, cows, horses, and shacks--and so too does my mind. As I ramble down bumpy paths, I stumble over the memory of a day spent fishing in a nearby bayou with my uncles, and the familiar smells of rank armpits and beer overwhelm my senses. Later, I see shiny pumps and a black veil waiting on top of an aging quilt and hear children running in bare feet. A tin of peanut brittle spotted at the counter of a country gas station lands my mind on my great-grandmother because that was her favorite candy. Big Momma was part of a chorus of tabernacle women who mothered me. She always said Shreveport was known as the city of churches because they sprout up on corners like strawberries in July. Word has it that there are more churches in my hometown than in any other city in the country. Hymns flow from their doors on Sunday mornings, while during the week the smiling church ladies greet you with words of encouragement as they skirt around the vestibule like bees on a honeycomb. “Baaaby, that was a fine prayer you did Sunday,” Mrs. Davis would say as I walked by. “Whose girl is dat with you? Bring her next Sunday.” But as is the case with all of Louisiana, our little city wobbles between extremes. Our local dishes of gumbo, catfish, and dirty rice must be spicy hot. Juke joints squat next to churches, and betting slips compete with offering envelopes. Fire-and-brimstone ministers point out that we drink and gamble too much. That is, until one in their congregation “hits.” Then it’s time to bring a tithe of the winnings to the altar. All of this is summed up neatly by the two billboards I notice as I cross the Texas Street Bridge into my city’s fold. One beckons you toward the horseshoe casino straight ahead. Opposite, another shouts, wanna win the jackpot? come to jesus. There is always the question of which road to take. To enter Shreveport’s downtown, travelers must cross our beloved Red River, which curls like a large garden snake around the city. The river is yet another contradiction. It does not remotely resemble the liquid silver color of the Mississippi. Instead, it pours out a murky clay red and flows as thick as soft mud across Louisiana. The only time it sparkles is at night, when the casino riverboats’ carnival lights illuminate the city. Following the curve of the river, crouched along the road leaving downtown, rests a neighborhood of little shacks that belie a city of over a quarter million. We call the houses shotgun because a bullet fired through the front entrance will pass through every room in the house before exiting the back door. It is a place where the children play dodgeball in the street but know to watch their manners, and every woman worth her salt can make a meal out of meat drippings, flour, eggs, and rice. The unpaved streets are filled with stray dogs, and after a rain the air smells like wet earth. On warm mornings, plump older women wearing blinding white maid uniforms congregate on corners and talk while awaiting the arrival of little blue buses that will take them to the homes where they work. “Child, Pastor Green liked to got the church on fire Sunday, didn’t he?” “Yeah, girl, and did you hear Mrs. Rogers shouting in the back? You know that boy of hers keeps her on her knees. He ain’t got good sense.” “Folks say what they want about her whiskey habit. That woman will give you the shirt off her back. That’s how I know she close to God.” That neighborhood was called Stoner Hill. I grew up listening to the women there. Everyone in that phalanx had a family church, and most believed in God and agreed that it was through Jesus Christ that we all gained salvation. Growing up, I don’t recall ever meeting someone who didn’t have a faith--at least no one who would admit such a thing out loud. It was a place where all doctrine was respected; even door-knocking Jehovah’s Witnesses were given the opportunity to speak their piece. Yes, Shreveport was the kind of city where everyone had a church, temple, or chapel they considered theirs, even if they’d only seen it from the inside a dozen times. A child from Stoner Hill seldom made it out of puberty without a distant cousin or a neighbor dragging the youngster off to recite New Testament Scripture in the Easter pageant or sing carols in the Christmas program, blessing the child with at least a C.M.E. membership: attendance at Christmas, Mother’s Day, and Easter. I’ve been reciting Bible verses since I was old enough to say, “Jesus wept.” My great-grandmother, Big Momma, used to say about the Bible, “Baby, you can find a word to carry you through anythang.” Still, my very religious family managed to pick and choose which Scriptures to live by. The men would pray up a miracle in the deacons’ corner and then enjoy a strong glass or two of Jack Daniel’s after church. The women sang in the choir but cursed like sailors when their team fumbled on Monday Night Football. “I could pull up my skirt and beat that sorry-ass receiver to the ball,” Big Momma would shout from the kitchen while stacking freshly washed dishes in cabinets. I spent most of my childhood summers down the road from home at Big Momma’s house. We began each day with the morning ritual she referred to as her labor of love--combing my hair. I would sit on the porch floor with my feet swinging over its edge while my head bobbed back and forth between Big Momma’s legs as she tugged, parted, and braided my long, thick, nappy hair. Big Momma always sat perfectly upright, sucking in her breath with each drag of the comb, then releasing the air from her hollow Cherokee cheeks, never once bending her back. After she finished the job, she’d pat me on my head and say, “Now you beautiful.” I’d rush to the bathroom, stand on the toilet seat, and peer over the sink into the mirror, eager to view this new and beautiful me. Of course, she never materialized. All I ever saw was my chubby face with a crown of lopsided plaits and a mouth full of what my momma teasingly called “beaver teeth” because they looked large enough to saw wood. Besides our grooming, Big Momma and her band of swearing sopranos made sure their offspring got a proper Christian upbringing. Every Sunday there was morning church school and Baptist Training Union. And for one week every August the young ones were herded to Grambling, Louisiana, a small college town, for a gigantic statewide revival called Youth-En-Camp. Although the drive took only a few hours, it had the feel of a great adventure. This was due in part to the parcel of sheets, dresses, and fried chicken that always accompanied me but also because the decreased supervision allowed me to experience free will. It was during one of these revivals that I became hopeful that I would one day look into the mirror and see beauty in myself. I was thirteen at the time--too old to be in one of the crayon classrooms but still too awkward to be cool. Before that summer I’d never thought that I could be beautiful--perhaps cute, on a good day, but never glamorous, radiant, or enchanting. Of course, up to that point, the only form of beauty I knew to desire was physical splendor, in which category I was sorely lacking. I was the tallest girl in my eighth-grade class, and when I tried to walk in dress shoes, my heels would slide out, causing me to trip over myself. Naturally, my only concern was ridding myself of awkwardness. Beauty was something I saw only in others. A woman’s even-colored skin and bright white teeth made her beautiful, never the inner peace that sparkled in her eyes. I greatly admired the little girl’s sunny Easter dress, adorned with white bows and ribbons, but gave no thought to the mother--needle in one hand, iron in the other, creating this lovely vision. And Big Momma’s front lawn with its velvet violets, deep purple grape suckers, and yellow sunflowers floating in the air like balloons was beautiful, but never once did I consider the care they were given even as the flowers’ first petals danced indiscriminately in the sunlight. I had always focused on my plainness, and it was this sorry image of myself that I took with me to Youth-En-Camp that summer. Only later would I understand that real beauty emanates from the heart. At camp that summer, our daily activities started with 5 a.m. prayer and devotion, during which I often volunteered to pray out loud so that everyone could hear my conversation with God. Somewhere along the way I got the notion that you were the biggest coward and hypocrite if you didn’t want to pray out loud. That to me suggested you were ashamed of the Lord, and even with all my insecurities and teenage angst, I wanted to be bigger than that. After breakfast, there was Bible-study class, lunch, and midday worship. There teenagers would offer testimonials, and thanks to those I referred to as our “holy staples” (they seemed as necessary to our religious experience as the flour and canned goods that lined the shelves of our neighborhood general store)--the girl who’d been suffering from multiple sclerosis who was walking for the first time in five years and the boys who overnight had been called to preach--the standard for godliness was set high. Following dinner and church service came the dating game, which commenced on a dusty bridge that stretched a half mile long and linked Grambling to the town of Ruston. As a symbolic gesture, the bridge was closed while the campers lined up at its foot, over a thousand of us girls on the right while the boys, far fewer in number, stood on the left. Once we "
            # "I want to "
    ]

    all_passed = True

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"[Prompt {i+1}] {prompt}")
        print(f"{'='*60}")

        # ---- Step 1: Generate (greedy decode) ----
        print("\n[Step 1] 运行 model.generate() (greedy decode)...")
        input_ids, generated_ids, gen_scores = run_generate(model, tokenizer, prompt, device, args.max_new_tokens)
        gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"  输入 token 数: {input_ids.shape[1]}")
        print(f"  生成 token 数: {generated_ids.shape[0]}")
        print(f"  生成文本: {gen_text}")

        # 重置 generate 模式状态
        model._gen_state.reset()

        # ---- Step 2: Forward baseline (全量 prefill) ----
        print("\n[Step 2] 运行 model.forward() (全量 prefill 对照组)...")
        pred_tokens, fwd_logits = run_forward_baseline(model, tokenizer, input_ids, generated_ids, device)

        # ---- Step 3: 对比 argmax ----
        print("\n[Step 3] 对比 generate vs forward argmax 结果...")
        argmax_passed = compare_results(generated_ids, pred_tokens, tokenizer)

        # ---- Step 4: 对比 PPL ----
        print("\n[Step 4] 对比 generate vs forward PPL...")
        # Generate 侧 PPL：用 generate 每步的 scores 和实际生成的 token 计算
        gen_ppl, gen_nll = compute_ppl_from_logits(gen_scores, generated_ids)
        # Forward 侧 PPL：用全量 forward 的 logits 和实际生成的 token 计算
        fwd_ppl, fwd_nll = compute_ppl_from_logits(fwd_logits, generated_ids)
        ppl_passed = compare_ppl(gen_ppl, gen_nll, fwd_ppl, fwd_nll)

        if not argmax_passed or not ppl_passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("[INFO] ✅ 所有测试通过！Generate 和 Forward 输出完全一致。")
    else:
        print("[INFO] ❌ 部分测试未通过，请检查上面的不匹配详情。")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 HSAForCausalLM generate vs forward 一致性")
    parser.add_argument(
        "--checkpoint_path", type=str, default=DEFAULT_CKPT_PATH,
        help="HF checkpoint 路径"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Tokenizer 路径，默认与 checkpoint_path 相同"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="运行设备，如 cuda:0 或 cpu"
    )
    parser.add_argument(
        "--attn_impl", type=str, default="flash_attention_3",
        help="注意力实现方式，如 flash_attention_3, sdpa, eager 等"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=100,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--prompts", nargs="+", type=str, default=None,
        help="自定义 prompt 列表"
    )

    args = parser.parse_args()
    print(f"[INFO] 参数: {args}")
    main(args)
