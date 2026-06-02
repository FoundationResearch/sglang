import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
import types  
import random
import numpy as np
from data import build_numpy_dataset
import torch.nn as nn
import argparse
import sys
import torch
from transformers import AutoTokenizer
from torch.utils import data
from torch.utils.data import SequentialSampler
from veomni.models import build_foundation_model
from veomni.checkpoint import ckpt_to_state_dict, build_checkpointer
from data import RulerSynthesizer, synthesize_ruler_example
import torch.distributed as dist
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.checkpoint import build_checkpointer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# 误差工具
def get_abs_err(x, y):
    return (x - y).flatten().abs().max().item()

def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg

# # 设置路径
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

# VeOmni Imports
from veomni.models import build_foundation_model
from veomni.checkpoint import build_checkpointer
from models.FlashHSA.configuration_hsa import HSAConfig

def insert_special_tokens(input_ids, fill_id, chunk_size):
    """每隔 chunk_size - 1 插入一个特殊 token"""
    N, L = input_ids.shape
    full_chunks = L // (chunk_size - 1)
    remainder = L % (chunk_size - 1)
    
    parts = []
    if full_chunks > 0:
        chunk_part = input_ids[:, :full_chunks * (chunk_size - 1)].view(N, full_chunks, chunk_size - 1)
        fill_tokens = torch.full((N, full_chunks, 1), fill_id, device=input_ids.device, dtype=input_ids.dtype)
        parts.append(torch.cat([chunk_part, fill_tokens], dim=2).view(N, -1))
    
    if remainder > 0:
        parts.append(input_ids[:, full_chunks * (chunk_size - 1):])
    
    return torch.cat(parts, dim=1)




def run_test():
    # 1. 基础配置
    step_dir = "/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-dense-correct/checkpoints/global_step_10000"
    config_path = "./configs/flash_hsa/config_swan_nope_sparse.json"
    vocab_dir = "./configs/olmo3_vocab/"
    device = "cuda"
    dtype = torch.bfloat16
    
    segment_size = 1024

    print(f"Loading tokenizer from: {vocab_dir}")
    tokenizer = AutoTokenizer.from_pretrained(vocab_dir, trust_remote_code=True, fix_mistral_regex=True)
    
    print(f"Building VeOmni model with config: {config_path}")
    model = build_foundation_model(config_path=config_path, torch_dtype="bfloat16")

    print(f"Loading DCP checkpoint from: {step_dir}")
    Checkpointer = build_checkpointer(dist_backend='fsdp2', ckpt_manager='dcp')
    Checkpointer.load(step_dir, {"model": model})
    model.to(device).eval()

    # =================================================================
    # MLP Patching (返回全0, 屏蔽 MLP 噪音)
    # =================================================================
    print("\n[DEBUG MODE] Patching MLP layers to return zeros...")
    def zero_out_forward(self, x): return torch.zeros_like(x)
    patched = 0
    for name, module in model.named_modules():
        if "MLP" in module.__class__.__name__:
            module.forward = types.MethodType(zero_out_forward, module)
            patched += 1
    print(f"Patched MLP layers: {patched}")
    # =================================================================

    if hasattr(model, "config"):
        chunk_size = model.config.chunk_size
        print(f"Chunk Size: {chunk_size}")
        lmk_id = tokenizer.vocab_size
    else:
        raise ValueError("Model config missing.")

    # 2. 构造长度为 (chunk_size-1) 的倍数，然后插入特殊 token
    base_len = (4096 // chunk_size) * (chunk_size - 1)
    raw_input_ids = tokenizer("H " * (base_len + 10), return_tensors="pt").input_ids.to(device)
    full_input_ids = insert_special_tokens(raw_input_ids[:, :base_len], lmk_id, chunk_size)
    print(f"After insertion, seq len = {full_input_ids.shape[1]} (should be multiple of chunk_size)")

    full_seq_len = full_input_ids.shape[1]
    print(f"\nTest Config: Full Len={full_seq_len}, Segment Size={segment_size}, Num Segments={full_seq_len // segment_size}")

    # ==========================================
    # 实验 A: Ground Truth (一次性全量)
    # ==========================================
    print("\n--- Running Experiment A: Full Context (Reference) ---")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        full_pos = torch.arange(0, full_seq_len, device=device)
        out_full = model(input_ids=full_input_ids, cache_position=full_pos, use_cache=True)
        logits_ref_all = out_full.logits.cpu().float()

    # ==========================================
    # 实验 B: Iterative (循环切片输入)
    # ==========================================
    print("\n--- Running Experiment B: Iterative Segments ---")
    past_key_values = None
    num_segments = full_seq_len // segment_size

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size
        
        seg_input_ids = full_input_ids[:, start_idx:end_idx]
        seg_cache_pos = torch.arange(start_idx, end_idx, device=device)
        
        print(f"Processing Segment {i} ({start_idx} -> {end_idx})...")
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            out_iter = model(
                input_ids=seg_input_ids,
                cache_position=seg_cache_pos,
                use_cache=True,
                past_key_values=past_key_values
            )
        
        past_key_values = out_iter.past_key_values
        logits_iter = out_iter.logits.cpu().float()
        logits_ref = logits_ref_all[:, start_idx:end_idx, :]

        assert_close(f"Segment {i}", logits_ref, logits_iter, 0.05)

    print("\nTest Finished.")



def test_memory_saving_prefill():
    """
    验证：全量超长序列可能 OOM，而分段 Prefill 不会 OOM，并记录峰值显存。
    """
    step_dir = "/apdcephfs_sh8/share_300719895/shawnxxxhu/checkpoints/hsa-swan-dense-correct/checkpoints/global_step_10000"
    config_path = "./configs/flash_hsa/config_swan_nope_sparse.json"
    vocab_dir = "./configs/olmo3_vocab/"
    device = "cuda"

    # 构建模型与 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(vocab_dir, trust_remote_code=True, fix_mistral_regex=True)
    model = build_foundation_model(config_path=config_path, torch_dtype="bfloat16")
    Checkpointer = build_checkpointer(dist_backend='fsdp2', ckpt_manager='dcp')
    Checkpointer.load(step_dir, {"model": model})
    model.to(device).eval()

    chunk_size = model.config.chunk_size
    lmk_id = tokenizer.vocab_size
    segment_size = 4096
    chunk_size = model.config.chunk_size  # 例如 64

    # 目标插入后的总长度
    target_full_len = segment_size * 1024

    # 反推插入前长度（必须是 (chunk_size-1) 的倍数）
    base_len = int(target_full_len * (chunk_size - 1) / chunk_size)
    base_len = (base_len // (chunk_size - 1)) * (chunk_size - 1)  # 对齐

    raw_input_ids = tokenizer("H " * (base_len + 10), return_tensors="pt").input_ids.to(device)
    full_input_ids = insert_special_tokens(raw_input_ids[:, :base_len], lmk_id, chunk_size)
    print(f"After insertion, seq len = {full_input_ids.shape[1]} (expect {target_full_len})")

    full_seq_len = full_input_ids.shape[1]
    num_segments = full_seq_len // segment_size
    print(f"Num Segments={num_segments}")

    # ==========================================
    # 1. 先跑 Chunk Prefill (预期成功)
    # ==========================================
    print("\n>>> Testing Chunked Prefill (Should Succeed)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    past_key_values = None
    print(f"[Chunk Prefill] num_segments={num_segments}, segment_size={segment_size}")
    
    try:
        with torch.no_grad():
            for i in range(num_segments):
                s, e = i * segment_size, (i + 1) * segment_size
                seg_ids = full_input_ids[:, s:e]
                seg_pos = torch.arange(s, e, device=device)
                
                # 简单打印进度
                if i % 10 == 0:
                        print(f"Processing segment {i}/{num_segments}...", end='\r')

                out = model(
                    input_ids=seg_ids,
                    cache_position=seg_pos,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
        
        peak_chunk = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n[Chunk Prefill] ✅ Completed without OOM. Peak Memory: {peak_chunk:.2f} GB")
    except Exception as e:
        print(f"\n[Chunk Prefill] ❌ Failed with error: {e}")
        # 如果分段都跑不过，全量肯定没戏，直接退出
        return

    # ==========================================
    # 2. 再跑全量 (预期 OOM/Crash)
    # ==========================================
    print("\n>>> Testing Full Inference (Should OOM/Crash)...")
    
    # 释放显存，给全量一个“公平”的机会（虽然预期它接不住）
    del past_key_values
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        full_pos = torch.arange(0, full_seq_len, device=device)
        print("Starting full forward pass (this might hang/crash)...")
        with torch.no_grad():
            _ = model(input_ids=full_input_ids, cache_position=full_pos, use_cache=True)
            
        peak_full = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[Full] peak {peak_full:.2f} GB).")
        
    except RuntimeError as e:
        peak_full = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[Full] OOM Peak before failure ~ {peak_full:.2f} GB")
        print(f"Error details: {str(e)[:100]}...")
        torch.cuda.empty_cache()







# debug only
if __name__ == "__main__":
    run_test()
    test_memory_saving_prefill()
