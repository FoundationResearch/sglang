import torch
import torch.nn.functional as F
import pytest
import math
import sys
import os

# 添加工作区根目录到 payload 以便做模块导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ops.topk_group import online_topk_group, ref_topk_forward_with_grad
from ops.hsa_fwd_bwd_group import HSA_block_M_group, hsa_torch_ref

def get_abs_err(x, y):
    mask = (x > -1e5) & (y > -1e5)
    if mask.sum() == 0: return 0.0
    return (x[mask] - y[mask]).abs().max().item()

def get_err_ratio(x, y):
    mask = (x > -1e5) & (y > -1e5)
    if mask.sum() == 0: return 0.0
    err = (x[mask] - y[mask]).square().mean().sqrt().item()
    base = (x[mask]).square().mean().sqrt().item()
    return err / (base + 1e-12)

def assert_close(prefix, ref, tri, ratio=0.01, atol=1e-3):
    abs_err = get_abs_err(ref, tri)
    rel_ratio = get_err_ratio(ref, tri)
    msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
    print(msg)
    # 宽松一点的检查，因为是两个 Kernel 串联，FP16/BF16 误差会累积
    if rel_ratio > ratio and abs_err > atol:
        raise AssertionError(msg)

@pytest.mark.parametrize("B, L, H_q, H_kv, D, S, block_size, topk, q_offset", [
    (1, 4096, 16, 2, 128, 64, 64, 16, 0),
    (2, 2048, 16, 2, 128, 64, 64, 16, 1024),
    (1, 1024, 8, 1, 64, 64, 64, 8, 3072),
    (1, 1385, 8, 1, 64, 64, 64, 8, 2048),
])
def test_joint_topk_hsa_correctness(B, L, H_q, H_kv, D, S, block_size, topk, q_offset):
    print(f"\n\n=== Testing Joint TopK+HSA [B={B}, L={L}, Offset={q_offset}] ===")
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # KV Cache 总长度
    L_kv = S * block_size
    
    # 确保 S 足够大以覆盖 offset + L (Assert 逻辑校验)
    min_required_s = (q_offset + L + block_size - 1) // block_size
    assert S >= min_required_s, f"Config Error: S={S} not enough for Offset+L={(q_offset+L)}"

    torch.manual_seed(42)

    # 1. 准备输入张量
    # Q: [B, L, H_q, D]
    q = torch.randn(B, L, H_q, D, dtype=dtype, device=device, requires_grad=True)
    
    # Lmks: [B, S, H_kv, D] (用于 TopK 计算索引)
    lmks = torch.randn(B, S, H_kv, D, dtype=dtype, device=device, requires_grad=True)
    
    # K, V: [B, S*block_size, H_kv, D] (用于 HSA 计算内容)
    # 注意: L_kv = S * block_size
    k = torch.randn(B, L_kv, H_kv, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, L_kv, H_kv, D, dtype=dtype, device=device, requires_grad=True)
    
    # 2. 复制两份用于对比 (Fused vs Ref)
    q_f = q.clone().detach().requires_grad_(True)
    lmks_f = lmks.clone().detach().requires_grad_(True)
    k_f = k.clone().detach().requires_grad_(True)
    v_f = v.clone().detach().requires_grad_(True)
    
    q_r = q.clone().detach().requires_grad_()
    lmks_r = lmks.clone().detach().requires_grad_()
    k_r = k.clone().detach().requires_grad_()
    v_r = v.clone().detach().requires_grad_()

    # ==========================================
    # Path A: Fused Implementations
    # ==========================================
    
    # A.1 TopK
    # indices: [B, L, H_kv, topk], scores: [B, L, H_kv, topk]
    # 使用 is_causal=True 模拟训练场景
    indices_f, scores_f = online_topk_group(
        q_f, lmks_f, topk, block_size, window_size=100, is_causal=True, q_offset=q_offset
    )
    
    # A.2 Softmax (生成 HSA 权重)
    # 简单的 Softmax 策略，你实际代码可能有 stick breaking，这里为了对比基础正确性用标准softmax
    weights_f = F.softmax(scores_f.float(), dim=-1).to(dtype)
    
    # A.3 HSA
    # HSA 接受的 indices 是 [B, L, H_kv, topk]，内部会对齐到 H_q
    # 注意 HSA 不需要 q_offset，它只认 indices 里的物理块 ID
    hsa_out_f = HSA_block_M_group(
        q_f, k_f, v_f, weights_f, indices_f, 
        block_size=block_size, mask_last_token=True
    )

    # ==========================================
    # Path B: Reference Implementations
    # ==========================================
    
    # B.1 TopK Ref
    # 使用 float32 保证 Ref 精度
    indices_r, scores_r = ref_topk_forward_with_grad(
        q_r.float(), lmks_r.float(), topk, block_size, window_size=100, is_causal=True, 
        dtype=torch.float32, q_offset=q_offset
    )
    
    # B.2 Softmax Ref
    weights_r = F.softmax(scores_r, dim=-1)
    
    # B.3 HSA Ref
    # HSA Ref 需要 indices 形状 [B, q_blocks, H, K]，这里 q_blocks = L (即 Ref 实现中的 block_q=1)
    # Ref 实现中假设 block_q=1 时，dim 1 就是 L
    hsa_out_r = hsa_torch_ref(
        q_r.float(), k_r.float(), v_r.float(), weights_r, indices_r,
        chunk_size=block_size, sm_scale=(1.0/math.sqrt(D)),
        block_q=1, mask_last_token=True
    )

    # ==========================================
    # Check Forward
    # ==========================================
    print(">>> Comparing HSA Output (Forward)")
    assert_close("HSA Output", hsa_out_r, hsa_out_f.float(), ratio=0.02)
    print("✅ Forward Joint Test Passed")

    # ==========================================
    # Check Backward (End-to-End)
    # ==========================================
    print(">>> Comparing Gradients (Backward)")
    
    grad_out = torch.randn_like(hsa_out_f)
    
    # Fused Backward
    hsa_out_f.backward(grad_out)
    
    # Ref Backward
    hsa_out_r.backward(grad_out.float())
    
    # 1. Check dQ (来自 HSA -> Q 和 TopK -> Q 的梯度叠加)
    assert_close("grad_Q", q_r.grad, q_f.grad.float(), ratio=0.05)
    
    # 2. Check dLmks (来自 TopK 的梯度)
    # 这是测试 TopK 反向是否正确处理了 q_offset 和 sparse gradient
    # nan_to_num 是为了防止 Ref 在某些极端 mask 下产生 NaN
    grad_lmks_r = torch.nan_to_num(lmks_r.grad, 0.0)
    grad_lmks_f = torch.nan_to_num(lmks_f.grad, 0.0)
    assert_close("grad_Lmks", grad_lmks_r, grad_lmks_f.float(), ratio=0.05)
    
    # 3. Check dK, dV (来自 HSA 的梯度, 稀疏回传到 dense tensor)
    # 这是测试 HSA 反向是否根据 indices 正确写回了 KV
    grad_k_r = torch.nan_to_num(k_r.grad, 0.0)
    grad_k_f = torch.nan_to_num(k_f.grad, 0.0)
    grad_v_r = torch.nan_to_num(v_r.grad, 0.0)
    grad_v_f = torch.nan_to_num(v_f.grad, 0.0)
    
    assert_close("grad_K", grad_k_r, grad_k_f.float(), ratio=0.05)
    assert_close("grad_V", grad_v_r, grad_v_f.float(), ratio=0.05)
    
    print("✅ Backward Joint Test Passed")

if __name__ == "__main__":
    # 手动运行测试
    params_list = [
        # B, L, H_q, H_kv, D, S, block_size, topk, q_offset
        # (1, 1024, 16, 2, 128, 64, 64, 16, 0),
        (1, 1001, 16, 2, 128, 32, 64, 16, 499),
    ]
    for p in params_list:
        test_joint_topk_hsa_correctness(*p)
