"""
SGLang 侧推理 Worker。
在 sglang 虚拟环境下运行，使用离线 Engine 做推理，结果序列化到 pickle 文件。

用法（由 run_verify.sh 直接调用）：
  /root/sglang/python/.venv/bin/python verify_sglang_worker.py <config.json>
"""

import gc
import json
import os
import pickle
import sys
import time
from typing import Dict, List


def run_sglang_inference(cfg: dict) -> Dict:
    """SGLang 侧：使用离线 Engine 做推理。"""
    import torch
    import sglang as sgl

    checkpoint_path = cfg["checkpoint_path"]
    prompt = cfg["prompt"]
    max_new_tokens = cfg["max_new_tokens"]
    prompt_tokens = cfg.get("prompt_tokens", 0)
    sglang_tp = cfg.get("sglang_tp", 1)
    sglang_max_total_tokens = cfg.get("sglang_max_total_tokens", 4096)
    sglang_page_size = cfg.get("sglang_page_size", 64)
    sglang_chunked_prefill_size = cfg.get("sglang_chunked_prefill_size", 8192)
    top_k = cfg.get("top_k", 5)

    print(f"[SGLang] Loading model with Engine: {checkpoint_path}")
    print(f"[SGLang] chunked_prefill_size={sglang_chunked_prefill_size}")
    engine = sgl.Engine(
        model_path=checkpoint_path,
        tp_size=sglang_tp,
        attention_backend="hsa",
        page_size=sglang_page_size,
        max_total_tokens=sglang_max_total_tokens,
        chunked_prefill_size=sglang_chunked_prefill_size,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
    )

    # 支持 prompt_tokens：从长 prompt 截取前 N 个 token
    if prompt_tokens > 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        full_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_len = len(full_ids)
        if prompt_tokens > full_len:
            print(f"[SGLang] ⚠️ prompt_tokens={prompt_tokens} > full prompt length={full_len}, using full prompt")
        else:
            truncated_ids = full_ids[:prompt_tokens]
            prompt = tokenizer.decode(truncated_ids)
            # 验证 roundtrip 一致性
            verify_ids = tokenizer.encode(prompt, add_special_tokens=False)
            print(f"[SGLang] prompt_tokens={prompt_tokens}, truncated from {full_len} -> {len(verify_ids)} tokens")
            if len(verify_ids) != prompt_tokens:
                print(f"[SGLang] ⚠️ roundtrip mismatch: {len(verify_ids)} != {prompt_tokens}, diff={len(verify_ids)-prompt_tokens}")
        del tokenizer

    # ===== Warmup: 预热一次短推理，触发 TileLang JIT 编译 topk kernel =====
    # warmup prompt 需要足够长（>page_size tokens）才能触发 HSA topk 路径
    warmup_prompt = "Hello " * max(sglang_page_size * 2, 1024)  # 生成一个足够长的 warmup prompt
    warmup_tokens = 3
    print(f"[SGLang] Warmup: generating {warmup_tokens} tokens to trigger TileLang JIT compilation...")
    warmup_start = time.time()
    _ = engine.generate(
        prompt=warmup_prompt,
        sampling_params={"max_new_tokens": warmup_tokens, "temperature": 0.0},
    )
    warmup_time = time.time() - warmup_start
    print(f"[SGLang] Warmup done in {warmup_time:.2f}s (JIT compilation included)")
    # ===== End Warmup =====

    # ===== 多请求预热（复现 eval 场景下跨请求状态泄露）=====
    num_prefill_requests = cfg.get("num_prefill_requests", 0)
    if num_prefill_requests > 0:
        print(f"[SGLang] Sending {num_prefill_requests} dummy requests (long prompts to trigger chunked prefill)...")
        from transformers import AutoTokenizer as _AT
        _tok = _AT.from_pretrained(checkpoint_path)
        # 用 cfg 里的原始 prompt（未截取），确保 engine tokens > chunked_prefill_size
        _full_ids = _tok.encode(cfg["prompt"], add_special_tokens=False)
        print(f"  Full prompt length for dummy: {len(_full_ids)} tokens")
        for ri in range(num_prefill_requests):
            # 每个 dummy 截取不同长度，保证都够长触发 chunked prefill
            dummy_len = len(_full_ids) - ri * 200
            if dummy_len < 5000:
                dummy_len = 5000 + ri * 100
            dummy_len = min(dummy_len, len(_full_ids))
            _ids = _full_ids[:dummy_len]
            dummy_prompt = _tok.decode(_ids)
            _ = engine.generate(
                prompt=dummy_prompt,
                sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
            )
            print(f"  dummy request {ri+1}/{num_prefill_requests} done (tokens={dummy_len})")
        del _tok
        print(f"[SGLang] Dummy requests done, now running the real one.")

    print(f"[SGLang] Generating with return_logprob=True, top_logprobs_num={top_k}")
    decode_start = time.time()
    result = engine.generate(
        prompt=prompt,
        sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=top_k,
    )

    decode_time = time.time() - decode_start

    decode_token_ids = result.get("output_ids", [])
    decode_text = result.get("text", "")
    meta_info = result.get("meta_info", {})

    num_generated = len(decode_token_ids)
    print(f"[SGLang] Decode done: {num_generated} tokens in {decode_time:.2f}s "
          f"({decode_time / max(num_generated, 1) * 1000:.1f} ms/token)")

    # ===== DEBUG: 打印 result 和 meta_info 的完整结构 =====
    print(f"\n[SGLang DEBUG] result top-level keys: {list(result.keys())}")
    print(f"[SGLang DEBUG] meta_info keys: {list(meta_info.keys())}")
    for k, v in meta_info.items():
        if isinstance(v, (list, tuple)):
            print(f"  meta_info['{k}']: type={type(v).__name__}, len={len(v)}")
            if len(v) > 0:
                first = v[0]
                if isinstance(first, (list, tuple)):
                    print(f"    [0]: type={type(first).__name__}, len={len(first)}, sample={first[:3]}")
                else:
                    print(f"    [0]: {first}")
            if len(v) > 1:
                print(f"    [1]: {v[1] if not isinstance(v[1], (list,tuple)) else f'len={len(v[1])}'}")
        else:
            print(f"  meta_info['{k}']: {v}")

    # DEBUG: 检查 logprob 相关的字段
    for logprob_key in ["input_token_logprobs_val", "input_token_logprobs_idx",
                         "output_token_logprobs_val", "output_token_logprobs_idx",
                         "input_top_logprobs_val", "input_top_logprobs_idx",
                         "output_top_logprobs_val", "output_top_logprobs_idx"]:
        val = meta_info.get(logprob_key, "NOT_FOUND")
        if val == "NOT_FOUND":
            print(f"[SGLang DEBUG] {logprob_key}: ❌ NOT FOUND in meta_info")
        elif isinstance(val, (list, tuple)):
            print(f"[SGLang DEBUG] {logprob_key}: len={len(val)}, sample[:3]={val[:3]}")
        else:
            print(f"[SGLang DEBUG] {logprob_key}: type={type(val).__name__}, val={val}")
    print("[SGLang DEBUG] ===== END =====\n")
    # ===== END DEBUG =====

    print(f"[SGLang] Generated text: {decode_text}")
    print(f"[SGLang] Token IDs: {decode_token_ids}")

    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    print("[SGLang] Engine shut down, GPU memory freed.")

    return {
        "decode_token_ids": decode_token_ids,
        "decode_text": decode_text,
        "meta_info": meta_info,
        "decode_time": decode_time,
        "num_generated_tokens": num_generated,
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_sglang_worker.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        cfg = json.load(f)

    output_path = cfg["sglang_output_path"]

    print("=" * 60)
    print("[SGLang Worker] 开始 SGLang Engine 推理")
    print("=" * 60)

    results = run_sglang_inference(cfg)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[SGLang Worker] 结果已保存到: {output_path}")
    print("[SGLang Worker] 完成！")


if __name__ == "__main__":
    main()
