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

    # Raw-token-id mode: skip the engine tokenizer entirely (the checkpoint dir
    # carries a mismatched Qwen tokenizer; our model is dolma2 vocab 100278). We
    # feed/compare on ids, so skip_tokenizer_init=True is both correct and avoids
    # out-of-range encodes.
    raw_ids = list(cfg.get("input_ids") or [])
    raw_mode = len(raw_ids) > 0
    if prompt_tokens > 0 and raw_mode:
        raw_ids = raw_ids[:prompt_tokens]

    print(f"[SGLang] Loading model with Engine: {checkpoint_path}  raw_ids_mode={raw_mode}")
    print(f"[SGLang] chunked_prefill_size={sglang_chunked_prefill_size}")
    engine = sgl.Engine(
        model_path=checkpoint_path,
        tp_size=sglang_tp,
        attention_backend="hsa",
        page_size=sglang_page_size,
        max_total_tokens=sglang_max_total_tokens,
        chunked_prefill_size=sglang_chunked_prefill_size,
        disable_cuda_graph=cfg.get("disable_cuda_graph", True),
        disable_overlap_schedule=True,
        skip_tokenizer_init=raw_mode,
        log_level=cfg.get("log_level", "error"),
        enable_nan_detection=cfg.get("enable_nan_detection", False),
        **({"context_length": cfg["context_length"]} if cfg.get("context_length") else {}),
        **({"mem_fraction_static": cfg["mem_fraction_static"]} if cfg.get("mem_fraction_static") else {}),
        **({"max_running_requests": cfg["max_running_requests"]} if cfg.get("max_running_requests") else {}),
    )
    print(f"[SGLang] disable_cuda_graph={cfg.get('disable_cuda_graph', True)} "
          f"(CUDA graph {'OFF' if cfg.get('disable_cuda_graph', True) else 'ON'})")

    # ===== Warmup: 触发 TileLang JIT 编译 topk kernel（用 ids 切片，> page_size）=====
    warmup_tokens = 3
    print(f"[SGLang] Warmup: generating {warmup_tokens} tokens to trigger TileLang JIT compilation...")
    warmup_start = time.time()
    if raw_mode:
        _wu = raw_ids[: max(sglang_page_size * 4, 1024)]
        _ = engine.generate(input_ids=[_wu],
                            sampling_params={"max_new_tokens": warmup_tokens, "temperature": 0.0})
    else:
        _ = engine.generate(prompt="Hello " * max(sglang_page_size * 2, 1024),
                            sampling_params={"max_new_tokens": warmup_tokens, "temperature": 0.0})
    print(f"[SGLang] Warmup done in {time.time()-warmup_start:.2f}s (JIT included)")

    # ===== 多请求预热（复现 eval 跨请求状态泄露，P5）=====
    num_prefill_requests = cfg.get("num_prefill_requests", 0)
    if num_prefill_requests > 0 and raw_mode:
        print(f"[SGLang] Sending {num_prefill_requests} dummy requests (long, trigger chunked prefill)...")
        for ri in range(num_prefill_requests):
            dummy_len = max(len(raw_ids) - ri * 200, 5000 + ri * 100)
            dummy_len = min(dummy_len, len(raw_ids))
            _ = engine.generate(input_ids=[raw_ids[:dummy_len]],
                                sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0})
            print(f"  dummy request {ri+1}/{num_prefill_requests} done (tokens={dummy_len})")
        print(f"[SGLang] Dummy requests done, now running the real one.")

    print(f"[SGLang] Generating with return_logprob=True, top_logprobs_num={top_k}")
    # Raw-token-id path: feed cfg['input_ids'] directly (bypasses the Qwen-vs-dolma2
    # tokenizer mismatch). Falls back to the prompt string if not provided.
    gen_kwargs = dict(
        sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=top_k,
    )
    if cfg.get("input_ids"):
        ids = list(cfg["input_ids"])
        if prompt_tokens > 0:
            ids = ids[:prompt_tokens]
        gen_kwargs["input_ids"] = [ids]
        print(f"[SGLang] Using raw input_ids: {len(ids)} tokens (max id={max(ids)})")
    else:
        gen_kwargs["prompt"] = prompt
    decode_start = time.time()
    result = engine.generate(**gen_kwargs)
    if isinstance(result, list):
        result = result[0]

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
