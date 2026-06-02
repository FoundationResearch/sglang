"""
HF Transformers 侧推理 Worker。
在 veomni 虚拟环境下运行，使用 model.generate() 做 greedy decode，结果序列化到 pickle 文件。

现在 HF 侧模型已支持内部自动插入 LMK（通过 prepare_inputs_for_generation 和 auto_insert_lmk），
因此调用方式与 SGLang 侧完全对齐：直接传入原始 token，无需手动处理 LMK。

用法（由 run_verify.sh 直接调用）：
  /root/VeOmni/.venv/bin/python verify_hf_worker.py <config.json>
"""
import logging
logging.getLogger('tilelang.cache.kernel_cache').setLevel(logging.ERROR)

import gc
import json
import os
import pickle
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 确保能导入 InfiniteLongLM 的模块
INFINITELONGLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if INFINITELONGLM_ROOT not in sys.path:
    sys.path.insert(0, INFINITELONGLM_ROOT)

from models.FlashHSA.configuration_hsa import HSAConfig


# def resolve_hsa_class(config_path=None):
#     """根据 config 中的 model_type 动态选择 HSAForCausalLM 实现"""
#     model_type = ""
#     if config_path:
#         with open(config_path, "r") as f:
#             model_type = json.load(f).get("model_type", "")
#     if "olmo" in model_type:
#         from models.FlashHSA.modeling_olmo_lhsa import HSAForCausalLM
#         print("[HF] Using OLMo LHSA implementation")
#     else:
#         from models.FlashHSA.modeling_qwen_lhsa import HSAForCausalLM
#         print("[HF] Using Qwen LHSA (generate) implementation")
#     return HSAForCausalLM


def run_hf_inference(cfg: dict) -> Dict:
    """HF Transformers 侧：使用 model.generate() 做 greedy decode。

    模型内部已支持自动插入 LMK：
    - generate 模式：prepare_inputs_for_generation 自动处理 LMK 插入和 position_ids
    - forward 模式：auto_insert_lmk=True 时自动插入 LMK 并过滤 LMK 位置的 logits

    因此调用方式与 SGLang 侧完全对齐，无需手动处理 LMK。
    """
    checkpoint_path = cfg["checkpoint_path"]
    device_str = cfg["device"]
    prompt = cfg["prompt"]
    max_new_tokens = cfg["max_new_tokens"]
    vocab_dir = cfg.get("vocab_dir") or checkpoint_path

    device = torch.device(device_str)
    prompt_tokens = cfg.get("prompt_tokens", 0)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(vocab_dir)
    vocab_size = tokenizer.vocab_size

    # 注册模型（使用支持 generate 的版本）
    model_config_path = os.path.join(checkpoint_path, "config.json")
    # HSAForCausalLM = resolve_hsa_class(model_config_path)
    from models.FlashHSA.modeling_olmo_lhsa import HSAForCausalLM
    HSAConfig.model_type = "olmo_lhsa"
    AutoConfig.register("olmo_lhsa", HSAConfig)
    HSAForCausalLM.config_class = HSAConfig
    AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM)

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_3",
        "device_map": device,
    }
    print(f"[HF] Loading model from: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    model.eval()

    chunk_size = model.config.chunk_size
    print(f"[HF] Model loaded, chunk_size={chunk_size}, lmk_id={model.lmk_id}")
    print(f"[HF] auto_insert_lmk={model.auto_insert_lmk}")

    # Tokenize（支持 prompt_tokens 从长 prompt 截取指定长度）
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    full_len = input_ids.shape[1]
    if prompt_tokens > 0:
        if prompt_tokens > full_len:
            print(f"[HF] ⚠️ prompt_tokens={prompt_tokens} > full prompt length={full_len}, using full prompt")
        else:
            input_ids = input_ids[:, :prompt_tokens]
            print(f"[HF] prompt_tokens={prompt_tokens}, truncated from {full_len} -> {input_ids.shape[1]} tokens")
    orig_seq_len = input_ids.shape[1]
    print(f"[HF] Prompt tokens: {orig_seq_len}")

    # ==================== Generate (greedy decode) ====================
    # 使用 model.generate()，模型内部通过 prepare_inputs_for_generation 自动处理 LMK
    print(f"[HF] Running model.generate() with max_new_tokens={max_new_tokens}...")

    # 重置 generate 状态
    model._gen_state.reset()

    decode_start = time.time()
    with torch.no_grad():
        gen_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decode
            output_scores=True,  # 返回每步的 logits
            return_dict_in_generate=True,
        )

    # 重置 generate 状态
    model._gen_state.reset()

    decode_time = time.time() - decode_start

    # 提取生成结果
    generated_ids = gen_output.sequences[0, orig_seq_len:].tolist()
    # gen_output.scores 是 tuple，每个元素是 [batch_size, vocab_size] 的 logits
    decode_logits_list = [
        scores[0, :vocab_size].float().cpu() for scores in gen_output.scores
    ]

    decode_text = tokenizer.decode(generated_ids)
    print(f"[HF] Decode done: {len(generated_ids)} tokens in {decode_time:.2f}s "
          f"({decode_time / max(len(generated_ids), 1) * 1000:.1f} ms/token)")
    print(f"[HF] Generated text: {decode_text}")
    print(f"[HF] Token IDs: {generated_ids}")

    # ===== DEBUG: Decode 前几步详细信息 =====
    for step in range(min(10, len(generated_ids))):
        tok_id = generated_ids[step]
        _dbg_lp = F.log_softmax(decode_logits_list[step].float(), dim=-1)
        _dbg_v, _dbg_i = _dbg_lp.topk(5)
        _dbg_s = [f"'{tokenizer.decode([i.item()])}' id={i.item()} lp={v.item():.4f}" for i, v in zip(_dbg_i, _dbg_v)]
        print(f"[HF DEBUG] Step {step}: chose id={tok_id} '{tokenizer.decode([tok_id])}', "
              f"top5=[{', '.join(_dbg_s)}]")
    # ===== END DEBUG =====

    # 释放模型
    del model, gen_output
    gc.collect()
    torch.cuda.empty_cache()
    print("[HF] Model released, GPU memory freed.")

    return {
        "decode_token_ids": generated_ids,
        "decode_logits": decode_logits_list,
        "decode_text": decode_text,
        "vocab_size": vocab_size,
        "chunk_size": chunk_size,
        "decode_time": decode_time,
        "num_generated_tokens": len(generated_ids),
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_hf_worker.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        cfg = json.load(f)

    output_path = cfg["hf_output_path"]

    print("=" * 60)
    print("[HF Worker] 开始 HF Transformers 推理")
    print("=" * 60)

    results = run_hf_inference(cfg)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[HF Worker] 结果已保存到: {output_path}")
    print("[HF Worker] 完成！")


if __name__ == "__main__":
    main()
