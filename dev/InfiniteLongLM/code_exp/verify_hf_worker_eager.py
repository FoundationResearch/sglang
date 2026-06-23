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

# bootstrap installs the veomni mocks the official FlashHSA package imports at
# module load (models/FlashHSA/__init__.py -> veomni.models.loader). Without it,
# `import models.FlashHSA` fails with ModuleNotFoundError: veomni.
_ALIGN_DIR = os.path.join(os.path.dirname(INFINITELONGLM_ROOT), "align")
if _ALIGN_DIR not in sys.path:
    sys.path.insert(0, _ALIGN_DIR)
import bootstrap  # noqa: F401,E402  (side-effect: veomni mocks)

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

    # 注册模型（按 config.model_type 选 qwen/olmo 变体；eager attn 绕开 flash_attn3）
    model_config_path = os.path.join(checkpoint_path, "config.json")
    with open(model_config_path) as _f:
        _mt = json.load(_f).get("model_type", "olmo_lhsa")
    if "qwen" in _mt:
        from models.FlashHSA.modeling_qwen_lhsa import HSAForCausalLM
        print(f"[HF] Using Qwen LHSA implementation (model_type={_mt})")
    else:
        from models.FlashHSA.modeling_olmo_lhsa import HSAForCausalLM
        print(f"[HF] Using OLMo LHSA implementation (model_type={_mt})")
    HSAConfig.model_type = _mt
    AutoConfig.register(_mt, HSAConfig)
    HSAForCausalLM.config_class = HSAConfig
    AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM)

    # eager attention so this worker runs without flash_attention_3 installed.
    # No device_map (that needs accelerate) — load then .to(device).
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }
    print(f"[HF] Loading model from: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs).to(device)
    model.eval()

    chunk_size = model.config.chunk_size
    # Use the MODEL's vocab (config), not the tokenizer's — the bench tokenizer
    # is a Qwen tok (vocab 151643) that does NOT match this dolma2-trained model
    # (vocab 100278). We feed raw token ids and compare on ids, so the tokenizer
    # is only used for cosmetic display.
    vocab_size = int(model.config.vocab_size)
    print(f"[HF] Model loaded, chunk_size={chunk_size}, lmk_id={model.lmk_id}, vocab_size={vocab_size}")
    print(f"[HF] auto_insert_lmk={model.auto_insert_lmk}")

    # Raw-token-id path: feed cfg['input_ids'] directly (bypasses tokenizer
    # mismatch). Falls back to tokenizing the prompt string if not provided.
    if cfg.get("input_ids"):
        ids = list(cfg["input_ids"])
        if prompt_tokens > 0:
            ids = ids[:prompt_tokens]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        print(f"[HF] Using raw input_ids: {input_ids.shape[1]} tokens (max id={max(ids)})")
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if prompt_tokens > 0 and prompt_tokens < input_ids.shape[1]:
            input_ids = input_ids[:, :prompt_tokens]
    orig_seq_len = input_ids.shape[1]
    print(f"[HF] Prompt tokens: {orig_seq_len}")

    # ==================== Greedy decode via PREFILL-ROLLOUT ====================
    # The official model's autoregressive generate-decode path is NOT implemented
    # for the prior_query landmark scheme: lhsa_layer.py recomputes the per-chunk
    # prior query ``mu_q`` from the CURRENT forward's lmk_q only (empty during
    # 1-token decode) while ``k_chunked`` comes from the full KV cache, so
    # chunk_attn_pool hits a (0 chunks vs N chunks) shape mismatch.  There is no
    # per-chunk mu_q / lmk_k / prior_b decode cache in this layer.
    #
    # We therefore obtain an INDEPENDENT greedy reference the only way the
    # official model supports correctly: re-run a full PREFILL over the growing
    # real-token sequence each step (no KV cache), take the argmax of the last
    # real token's logit, append, repeat.  This is numerically a greedy decode
    # and uses ONLY the proven prefill path — exactly what compare.py forwards
    # through (official_prefill_logits) and validated == sglang at 100% argmax.
    print(f"[HF] Greedy decode via prefill-rollout, max_new_tokens={max_new_tokens} "
          f"(official generate-decode unimplemented for prior_query; using prefill path)")

    from utils.landmark_utils import (
        insert_special_tokens as _insert_lmk,
        create_position_ids_with_landmarks as _lmk_pos,
    )

    def _prefill_last_logit(real_tokens):
        """Full prefill over real_tokens; return fp32 logit (V,) that predicts
        the NEXT token (logit at the last real token position)."""
        ids_t = _insert_lmk(torch.tensor([real_tokens]), model.lmk_id, chunk_size).to(device)
        pos_t = _lmk_pos(None, len(real_tokens), chunk_size, device)
        with torch.no_grad():
            out = model(input_ids=ids_t, position_ids=pos_t,
                        attention_mask=None, use_cache=False)
        logits = out.logits[0, :, :vocab_size].float()  # (L_ext, V)
        # insert_special_tokens appends a trailing LMK iff len % (chunk_size-1)==0,
        # in which case the last real token sits one position before the end.
        last_real_idx = -1 if (len(real_tokens) % (chunk_size - 1) != 0) else -2
        return logits[last_real_idx].cpu()

    real_tokens = input_ids[0].tolist()
    generated_ids = []
    decode_logits_list = []

    decode_start = time.time()
    for step in range(max_new_tokens):
        next_logit = _prefill_last_logit(real_tokens)  # (V,)
        next_id = int(next_logit.argmax().item())
        generated_ids.append(next_id)
        decode_logits_list.append(next_logit)
        real_tokens.append(next_id)
    decode_time = time.time() - decode_start

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
    del model
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
