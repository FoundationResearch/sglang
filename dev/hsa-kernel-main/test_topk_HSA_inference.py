"""
真正的动态 KV Cache 实现 - 使用 concat 而非预分配
支持变长批推理，左对齐，cache 长度等于最长序列的长度
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from einops import rearrange

# 导入 FlashHSA 的接口
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ops'))

from ops.topk_head_softmax_prefill import online_softmax_topk_head as prefill_topk_func
from ops.topk_head_softmax_decode import topk_softmax_decode_interface as decode_topk_func
from ops.hsa_head_prefill import HSA_prefill_interface
from ops.hsa_head_decode import HSA_decode_interface


# ==================== 配置类 ====================

@dataclass
class HSAConfig:
    """HSA 模型配置"""
    hidden_size: int = 512
    num_query_heads: int = 8
    num_kv_heads: int = 2
    top_k: int = 4
    sliding_window: int = 64
    chunk_size: int = 32
    rms_norm_eps: float = 1e-6
    enable_softmax1: bool = False
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_query_heads
    
    @property
    def d_kv(self) -> int:
        return self.head_dim * self.num_kv_heads


# ==================== RMS Norm ====================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ==================== 真正的动态 KV Cache ====================

class DynamicKVCache:
    """
    真正的动态 KV Cache - 使用 concat 而非预分配
    
    特点：
    1. 初始时 cache 为空
    2. 每次 decode 时，将新的 k/v concat 到 cache 后面
    3. 支持左对齐的变长批处理
    4. cache 长度等于最长序列的长度
    
    布局：左对齐
    - Batch 0: [t0, t1, t2, ..., t_L0, 0, 0, ...]
    - Batch 1: [t0, t1, t2, ..., t_L1, 0, 0, ...]
    - Batch 2: [t0, t1, t2, ..., t_L2, 0, 0, ...]
    其中 L_max = max(L0, L1, L2)
    """
    
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache 初始为空（或很小的初始大小）
        # 形状: [B, 0, H, D] - 序列维度为 0
        self.k_cache = torch.empty(
            batch_size, 0, num_heads, head_dim,
            dtype=dtype, device=self.device
        )
        self.v_cache = torch.empty(
            batch_size, 0, num_heads, head_dim,
            dtype=dtype, device=self.device
        )
        
        # 每个 batch 元素的当前序列长度（用于 masking）
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        
        # 记录哪些序列已经完成（用于变长批处理）
        self.completed = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
    
    def update_prefill(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor,
        seq_lens: torch.Tensor
    ):
        """
        Prefill 阶段：一次性写入整个序列
        
        Args:
            k: [B, L, H, D] - 要写入的 keys
            v: [B, L, H, D] - 要写入的 values
            seq_lens: [B] - 每个序列的实际长度
        
        注意：这里假设输入已经是左对齐的，短序列后面会有 padding
        """
        B, L, H, D = k.shape
        assert B == self.batch_size
        assert H == self.num_heads
        assert D == self.head_dim
        
        # 直接赋值，因为这是第一次写入
        self.k_cache = k
        self.v_cache = v
        self.seq_lens = seq_lens
    
    def update_decode(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ):
        """
        Decode 阶段：将新的 k/v 写入到 cache 的对应位置
        
        Args:
            k_new: [B, 1, H, D] - 新生成的 keys（每个 batch 元素一个 token）
            v_new: [B, 1, H, D] - 新生成的 values
            positions: [B] - 可选，指定每个元素在 cache 中的写入位置
                            如果为 None，则自动使用当前 seq_lens 作为写入位置
        """
        B, L, H, D = k_new.shape
        assert L == 1, f"Decode should process single token, got L={L}"
        assert B == self.batch_size
        assert H == self.num_heads
        assert D == self.head_dim
        
        # 检查所有序列的 seq_lens 是否相同
        seq_lens_equal = (self.seq_lens == self.seq_lens[0]).all()
        
        if seq_lens_equal:
            # 简单情况：所有序列长度相同
            # 直接在序列维度 concat
            self.k_cache = torch.cat([self.k_cache, k_new], dim=1)  # [B, L+1, H, D]
            self.v_cache = torch.cat([self.v_cache, v_new], dim=1)
            self.seq_lens = self.seq_lens + 1
        else:
            # 复杂情况：变长批，不同序列长度不同
            # 需要扩展现有 cache，并将新 k/v 写入到对应位置
            
            # 1. 扩展 cache（增加 1 列）
            new_max_len = self.k_cache.shape[1] + 1
            k_cache_expanded = torch.zeros(
                B, new_max_len, H, D, dtype=self.dtype, device=self.device
            )
            v_cache_expanded = torch.zeros(
                B, new_max_len, H, D, dtype=self.dtype, device=self.device
            )
            
            # 2. 复制旧的 cache
            k_cache_expanded[:, :-1, :, :] = self.k_cache
            v_cache_expanded[:, :-1, :, :] = self.v_cache
            
            # 3. 在对应位置写入新的 k/v
            for b in range(B):
                if self.completed[b]:
                    continue
                
                write_pos = int(self.seq_lens[b])
                # 直接写入到指定位置，只写 1 个元素
                k_cache_expanded[b, write_pos, :, :] = k_new[b, 0, :, :]
                v_cache_expanded[b, write_pos, :, :] = v_new[b, 0, :, :]
            
            # 4. 更新 cache
            self.k_cache = k_cache_expanded
            self.v_cache = v_cache_expanded
            
            # 5. 更新 seq_lens
            for b in range(B):
                if not self.completed[b]:
                    self.seq_lens[b] += 1

    
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前的 KV cache
        
        Returns:
            k_cache: [B, max_len, H, D]
            v_cache: [B, max_len, H, D]
        """
        return self.k_cache, self.v_cache
    
    def get_landmarks(self, chunk_size: int) -> torch.Tensor:
        """
        获取 landmark keys（按 chunk_size 采样）
        
        Args:
            chunk_size: 采样的 chunk 大小
        
        Returns:
            lmks: [B, num_chunks, H, D]
        """
        # 按固定的 chunk_size 采样 landmarks
        lmks = self.k_cache[:, ::chunk_size, :, :]  # [B, num_chunks, H, D]
        return lmks
    
    def get_max_cache_len(self) -> int:
        """获取当前 cache 的最大长度"""
        return self.k_cache.shape[1]
    
    def reset(self):
        """重置 cache"""
        B = self.batch_size
        H = self.num_heads
        D = self.head_dim
        
        self.k_cache = torch.empty(B, 0, H, D, dtype=self.dtype, device=self.device)
        self.v_cache = torch.empty(B, 0, H, D, dtype=self.dtype, device=self.device)
        self.seq_lens.zero_()
        self.completed.zero_()


# ==================== 模拟的 FlashHSA 类 ====================


class FlashHSA_Inference(nn.Module):
    def __init__(self, config: HSAConfig):
        super().__init__()
        self.config = config
        self.layer_idx = 0
        
        # 从 config 获取参数
        self.d_model = config.hidden_size
        self.head_dim = config.head_dim
        self.d_kv = config.d_kv
        self.h_kv = config.num_kv_heads
        self.h_q = config.num_query_heads
        self.topk = config.top_k
        self.window_size = config.sliding_window
        self.chunk_size = config.chunk_size
        
        # 线性投影层
        self.q_proj = nn.Linear(self.d_model, self.d_model,).to(torch.bfloat16)
        self.kv_proj = nn.Linear(self.d_model, self.d_kv * 2).to(torch.bfloat16)
        self.o_proj = nn.Linear(self.d_model, self.d_model).to(torch.bfloat16)
        
        # 归一化层
        self.q_norm = RMSNorm(self.head_dim, dtype=torch.bfloat16)
        self.k_norm = RMSNorm(self.head_dim, dtype=torch.bfloat16)
        self.lmk_norm = RMSNorm(self.head_dim, dtype=torch.bfloat16)
        
        self.enable_softmax1 = config.enable_softmax1
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: DynamicKVCache,
        seq_lens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        统一的前向传播接口 - 根据 hidden_states 形状自动判断 prefill 或 decode
        
        Args:
            hidden_states: [B, L, d_model]
                          - L > 1: prefill 阶段（处理多个 token）
                          - L == 1: decode 阶段（处理单个 token）
            kv_cache: KV cache 管理器
            seq_lens: [B] 每个 batch 元素的序列长度（用于 masking）
            positions: [B] 可选，仅 decode 阶段有效，指定每个元素在 cache 中的写入位置
            use_cache: 是否使用 cache（默认 True）
            
        Returns:
            output: [B, L, d_model]
            past_key_value: 包含 KV cache 的字典（如果 use_cache=True），否则为 None
        """
        B, L, _ = hidden_states.shape
        device = hidden_states.device
        
        # ==================== DECODE 阶段 (L == 1) ====================
        if L == 1:
            # 1. 投影 Q, K, V
            q = self.q_proj(hidden_states)
            swa_kv = self.kv_proj(hidden_states)
            swa_k, swa_v = torch.split(swa_kv, self.d_kv, dim=2)
            
            # 2. 重塑并归一化 Q
            q = q.squeeze(1)  # [B, 1, d_model] -> [B, d_model]
            q = rearrange(q, 'B (h d) -> B h d', d=self.head_dim)  # [B, h_q, d]
            q = self.q_norm(q)
            
            # 3. 重塑 K, V
            hsa_k = rearrange(swa_k, 'B L (h d) -> B L h d', d=self.head_dim)  # [B, 1, h_kv, d]
            hsa_k = self.k_norm(hsa_k)
            hsa_v = rearrange(swa_v, 'B L (h d) -> B L h d', d=self.head_dim)  # [B, 1, h_kv, d]
            
            # 4. 从 cache 获取 landmarks
            lmks = kv_cache.get_landmarks(self.chunk_size)
            lmks = self.lmk_norm(lmks)
            
            # 5. 模拟 SWA 注意力并获取 LSE
            lse_swa = torch.randn(B, self.h_q, dtype=hidden_states.dtype, device=device) * 2.0
            
            # 6. TopK 选择 (Decode 接口)
            indices, scores = decode_topk_func(
                q, lmks, lse_swa, seq_lens, 
                self.topk, self.chunk_size, self.window_size
            )
            
            # 7. 合并分数
            if not self.enable_softmax1:
                cat_scores = torch.cat([scores, lse_swa.unsqueeze(-1)], dim=-1)
                swa_weight_idx = -1
            else:
                ones_tensor = torch.ones(*lse_swa.shape, 1, dtype=hidden_states.dtype, device=device)
                cat_scores = torch.cat([scores, lse_swa.unsqueeze(-1), ones_tensor], dim=-1)
                swa_weight_idx = -2
            
            # 8. Softmax 获取权重
            chunk_weights = F.softmax(cat_scores, dim=-1)
            
            # 9. HSA 注意力 (Decode 接口)
            k_cache, v_cache = kv_cache.get_kv()
            k_cache = k_cache.contiguous()
            v_cache = v_cache.contiguous()
            
            hsa_o = HSA_decode_interface(
                q, k_cache, v_cache,
                chunk_weights,
                indices, block_size=self.chunk_size,
            )
            
            # 10. 合并 SWA 和 HSA 输出
            swa_o_weight = chunk_weights[:, :, swa_weight_idx]  # [B, h_q]
            swa_o_weight = swa_o_weight.unsqueeze(1).unsqueeze(-1)  # [B, 1, h_q, 1]
            swa_o = q.unsqueeze(1) * swa_o_weight  # [B, 1, h_q, d]
            o = hsa_o.unsqueeze(1) + swa_o  # [B, 1, h_q, d]
            
            # 11. 输出投影
            o_reshaped = rearrange(o, 'B L h d -> B L (h d)')
            output = self.o_proj(o_reshaped)
            
            # 12. 更新 KV cache
            past_key_value = None
            if use_cache:
                kv_cache.update_decode(hsa_k, hsa_v)
                past_key_value = {
                    'key': kv_cache.k_cache,
                    'value': kv_cache.v_cache,
                    'seq_lens': kv_cache.seq_lens.clone()
                }
            
            return output, past_key_value
        
        # ==================== PREFILL 阶段 (L > 1) ====================
        else:
            # 1. 投影 Q, K, V
            q = self.q_proj(hidden_states)
            swa_kv = self.kv_proj(hidden_states)
            swa_k, swa_v = torch.split(swa_kv, self.d_kv, dim=2)
            
            # 2. 重塑并归一化 Q
            q = rearrange(q, 'B L (h d) -> B L h d', d=self.head_dim)  # [B, L, h_q, d]
            q = self.q_norm(q)
            
            # 3. 重塑 K, V
            hsa_k = rearrange(swa_k, 'B L (h d) -> B L h d', d=self.head_dim)  # [B, L, h_kv, d]
            hsa_k = self.k_norm(hsa_k)
            hsa_v = rearrange(swa_v, 'B L (h d) -> B L h d', d=self.head_dim)  # [B, L, h_kv, d]
            
            # 4. 从输入获取 landmarks
            lmks = hsa_k[:, ::self.chunk_size, :, :]
            lmks = self.lmk_norm(lmks)
            
            # 5. 模拟 SWA 注意力并获取 LSE
            lse_swa = torch.randn(B, L, self.h_q, dtype=hidden_states.dtype, device=device) * 2.0
            
            # 6. TopK 选择 (Prefill 接口)
            indices, scores = prefill_topk_func(
                q, lmks, lse_swa, seq_lens,
                self.topk, self.chunk_size, self.window_size, is_causal=True
            )
            
            # 7. 合并分数
            if not self.enable_softmax1:
                cat_scores = torch.cat([scores, lse_swa.unsqueeze(-1)], dim=-1)
                swa_weight_idx = -1
            else:
                ones_tensor = torch.ones(*lse_swa.shape, 1, dtype=hidden_states.dtype, device=device)
                cat_scores = torch.cat([scores, lse_swa.unsqueeze(-1), ones_tensor], dim=-1)
                swa_weight_idx = -2
            
            # 8. Softmax 获取权重
            chunk_weights = F.softmax(cat_scores, dim=-1)
            
            # 9. HSA 注意力 (Prefill 接口)
            hsa_k = hsa_k.contiguous()
            hsa_v = hsa_v.contiguous()
            
            hsa_o = HSA_prefill_interface(
                q, hsa_k, hsa_v, chunk_weights,
                indices, block_size=self.chunk_size,
            )
            
            # 10. 合并 SWA 和 HSA 输出
            swa_o_weight = chunk_weights[:, :, :, swa_weight_idx]  # [B, L, h_q]
            swa_o_weight = swa_o_weight.unsqueeze(-1)  # [B, L, h_q, 1]
            swa_o = q * swa_o_weight  # [B, L, h_q, d]
            o = hsa_o + swa_o  # [B, L, h_q, d]
            
            # 11. 输出投影
            o_reshaped = rearrange(o, 'B L h d -> B L (h d)')
            output = self.o_proj(o_reshaped)
            
            # 12. 更新 KV cache
            past_key_value = None
            if use_cache:
                kv_cache.update_prefill(hsa_k, hsa_v, seq_lens)
                past_key_value = {
                    'key': kv_cache.k_cache,
                    'value': kv_cache.v_cache,
                    'seq_lens': kv_cache.seq_lens.clone()
                }
            
            return output, past_key_value



# ==================== 模拟推理测试主程序 ====================

class MockLLMInference:
    """
    模拟 LLM 推理流程
    
    模拟完整的自回归生成过程，包括：
    1. Prefill 阶段处理初始输入
    2. Decode 阶段自回归生成多个 token
    3. 支持变长批处理
    """
    
    def __init__(self, config: HSAConfig, max_gen_tokens: int = 10):
        self.config = config
        self.max_gen_tokens = max_gen_tokens
        
        # 创建 FlashHSA 层
        self.flash_hsa = FlashHSA_Inference(config)
    
    def prefill(
        self,
        input_ids: List[List[int]],
        seq_lens: List[int]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Prefill 阶段 - 处理初始输入
        """
        B = len(input_ids)
        max_len = max(seq_lens)
        device = self.flash_hsa.q_proj.weight.device
        
        # 创建动态 KV cache（初始为空）
        self.kv_cache = DynamicKVCache(
            batch_size=B,
            num_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            dtype=torch.bfloat16,
            device=device
        )
        
        # 创建模拟的 hidden_states [B, L, d_model]
        hidden_states = torch.randn(
            B, max_len, self.config.hidden_size,
            dtype=torch.bfloat16, device=device
        )
        
        # 对 padding 位置进行掩码（左对齐）
        for b in range(B):
            if seq_lens[b] < max_len:
                hidden_states[b, seq_lens[b]:] = 0.0
        
        # 转换 seq_lens 为 tensor
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        
        # 调用统一的 forward 接口（L > 1 会自动走 prefill 逻辑）
        output, past_key_value = self.flash_hsa(
            hidden_states, self.kv_cache, seq_lens_tensor
        )
        
        return output, past_key_value
    
    def decode_step(
        self,
        current_seq_lens: List[int]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Decode 单步 - 生成一个新 token
        """
        B = len(current_seq_lens)
        device = self.flash_hsa.q_proj.weight.device
        
        # 创建模拟的 hidden_states [B, 1, d_model]
        hidden_states = torch.randn(
            B, 1, self.config.hidden_size,
            dtype=torch.bfloat16, device=device
        )
        
        # 转换 seq_lens 为 tensor
        seq_lens_tensor = torch.tensor(current_seq_lens, dtype=torch.int32, device=device)
        
        # 调用统一的 forward 接口（L == 1 会自动走 decode 逻辑）
        output, past_key_value = self.flash_hsa(
            hidden_states, self.kv_cache, seq_lens_tensor
        )
        
        return output, past_key_value
    
    def generate(
        self,
        input_ids: List[List[int]],
        seq_lens: List[int],
        max_new_tokens: Optional[int] = None
    ) -> List[List[int]]:
        """
        自回归生成
        
        Args:
            input_ids: 输入 token ID 列表
            seq_lens: 每个序列的实际长度
            max_new_tokens: 最大生成 token 数
            
        Returns:
            generated_ids: 生成的 token ID 列表
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_gen_tokens
        
        # Prefill 阶段
        print("\n" + "=" * 80)
        print("=== Prefill Phase ===")
        print("=" * 80)
        print(f"Batch size: {len(input_ids)}")
        print(f"Input seq lens: {seq_lens}")
        
        output, _ = self.prefill(input_ids, seq_lens)
        print(f"Prefill output shape: {output.shape}")
        print(f"Cache after prefill: max_len={self.kv_cache.get_max_cache_len()}, seq_lens={self.kv_cache.seq_lens.tolist()}")
        print("✅ Prefill completed successfully!")
        
        # Decode 阶段
        print("\n" + "=" * 80)
        print("=== Decode Phase (Auto-regressive Generation) ===")
        print("=" * 80)
        
        current_seq_lens = seq_lens.copy()
        generated_ids = [[] for _ in range(len(input_ids))]
        
        for step in range(max_new_tokens):
            print(f"\n--- Decode Step {step + 1}/{max_new_tokens} ---")
            print(f"Current seq lens: {current_seq_lens}")
            
            # Decode 单步
            output, _ = self.decode_step(current_seq_lens)
            print(f"Decode output shape: {output.shape}")
            print(f"Cache after decode: max_len={self.kv_cache.get_max_cache_len()}, seq_lens={self.kv_cache.seq_lens.tolist()}")
            
            # 模拟采样
            for b in range(len(current_seq_lens)):
                new_token = torch.randint(0, 1000, (1,)).item()
                generated_ids[b].append(new_token)
                current_seq_lens[b] += 1
            
            print(f"Generated tokens (step {step + 1}): {[gen_ids[-1] if gen_ids else None for gen_ids in generated_ids]}")
        
        print("\n✅ All decode steps completed successfully!")
        
        return generated_ids


# ==================== 测试函数 ====================

def test_varlen_batch_inference():
    """
    测试变长批推理场景
    
    测试配置：
    - Batch size: 4
    - 序列长度: [64, 128, 256, 32]
    - Prefill 阶段: 处理不同长度的输入
    - Decode 阶段: 自回归生成 5 个 token
    """
    print("\n" + "=" * 80)
    print("=== 测试变长批推理场景（使用真正的动态 Cache） ===")
    print("=" * 80)
    
    # 配置
    config = HSAConfig(
        hidden_size=512,
        num_query_heads=8,
        num_kv_heads=2,
        top_k=4,
        sliding_window=64,
        chunk_size=32,
        enable_softmax1=False
    )
    
    # 创建模拟推理器
    inference = MockLLMInference(config, max_gen_tokens=5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference.flash_hsa = inference.flash_hsa.to(device)
    
    # 准备变长输入
    input_ids = [
        [1, 2, 3, 4, 5, 6] * 10 + [7, 8],  # 序列 0: 长度 64
        [1, 2, 3, 4, 5, 6] * 20 + [7, 8, 9, 10, 11, 12, 13, 14],  # 序列 1: 长度 128
        [1, 2, 3, 4, 5, 6] * 40 + [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 序列 2: 长度 250
        [1, 2, 3, 4, 5, 6] * 5 + [7, 8],  # 序列 3: 长度 32
    ]
    
    seq_lens = [len(ids) for ids in input_ids]
    
    print("seqlens:",seq_lens)
    
    print(f"\n输入配置:")
    print(f"  Batch size: {len(input_ids)}")
    print(f"  序列长度: {seq_lens}")
    print(f"  模型配置:")
    print(f"    - Hidden size: {config.hidden_size}")
    print(f"    - Query heads: {config.num_query_heads}")
    print(f"    - KV heads: {config.num_kv_heads}")
    print(f"    - Top-K: {config.top_k}")
    print(f"    - Chunk size: {config.chunk_size}")
    print(f"    - Sliding window: {config.sliding_window}")
    
    # 执行推理
    try:
        generated_ids = inference.generate(input_ids, seq_lens, max_new_tokens=5)
        
        print("\n" + "=" * 80)
        print("=== 推理结果 ===")
        print("=" * 80)
        for i, gen_ids in enumerate(generated_ids):
            print(f"Batch {i}: Generated {len(gen_ids)} tokens")
            print(f"  Generated tokens: {gen_ids}")
        
        print("\n✅ 测试通过！动态 cache 正常工作。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败！错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False



# ==================== 主程序 ====================

if __name__ == "__main__":
    torch.manual_seed(42)
    
    test_varlen_batch_inference()