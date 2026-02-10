import torch
import torch.nn.functional as F
import pytest
import math
from einops import rearrange

from ops.hsa_fwd_bwd_head import HSA_block_M_head
from ops.hsa_fwd_bwd_head_dense import HSA_dense_interface
from ops.topk_head_softmax import online_softmax_topk_head

def get_abs_err(x, y):
    # 过滤掉 -inf (Masked) 的位置，避免 NaN
    # 注意：对于最终输出 O，通常没有 -inf，这个 mask 会全选，逻辑依然正确
    mask = (x > -1e5) & (y > -1e5)
    if mask.sum() == 0: return 0.0
    return (x[mask] - y[mask]).abs().max().item()

def get_err_ratio(x, y):
    mask = (x > -1e5) & (y > -1e5)
    if mask.sum() == 0: return 0.0
    # 计算差值的 RMS
    err = (x[mask] - y[mask]).square().mean().sqrt().item()
    # 计算基准值的 RMS
    base = (x[mask]).square().mean().sqrt().item()
    # 加上极小值防止除零
    return err / (base + 1e-12)

def assert_close(prefix, ref, tri, ratio=0.01):
    """
    prefix: 打印前缀
    ref: 参考值 (通常是 Dense)
    tri: 待测值 (通常是 Sparse)
    ratio: 允许的相对误差阈值
    """
    # 统一转为 float32 进行高精度对比
    ref_f = ref.float()
    tri_f = tri.float()
    
    abs_err = get_abs_err(ref_f, tri_f)
    rel_ratio = get_err_ratio(ref_f, tri_f)
    
    msg = f"{prefix} diff: {abs_err:.6e} ratio: {rel_ratio:.6e}"
    print(msg)
    
    assert rel_ratio < ratio, f"Assertion Failed: {msg}"



def get_lmks(k, chunk_size):
    """
    模拟从 K 生成 Landmarks (Mean Pooling)
    K: [B, L, H, D] -> LMK: [B, Num_Chunks, H, D]
    """
    # B, L, H, D = k.shape
    # num_chunks = L // chunk_size
    # k_reshaped = k.view(B, num_chunks, chunk_size, H, D)
    # lmks = k_reshaped.mean(dim=2) # [B, M, H, D]
    lmks = k[:, ::chunk_size, :, :]
    return lmks

@pytest.mark.parametrize("enable_softmax1", [False, True])
def test_dense_sparse_equivalence_full_coverage(enable_softmax1):
    """
    验证当 TopK 覆盖所有 Chunk 时，Sparse HSA 是否与 Dense HSA 等价。
    配置: L=4096, Chunk=64, TopK=64
    """
    # 1. 配置参数
    B = 1
    L = 4096
    H_q = 16
    H_kv = 4 # GQA
    D = 128
    chunk_size = 64
    window_size = 64
    topk = L // chunk_size # 64, 覆盖全量
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"\n\nTesting Equivalence: L={L}, TopK={topk}, Softmax1={enable_softmax1}")

    # 2. 构造输入数据
    torch.manual_seed(42)
    
    # Q, K, V (Normalized)
    q_norm = torch.randn(B, L, H_q, D, device=device, dtype=dtype)
    k_norm = torch.randn(B, L, H_kv, D, device=device, dtype=dtype)
    v = torch.randn(B, L, H_kv, D, device=device, dtype=dtype)
    
    # SWA 相关输入
    lse_sum = torch.randn(B, L, H_q, device=device, dtype=torch.float32) # LSE 通常是 float32
    swa_o = torch.randn(B, L, H_q * D, device=device, dtype=dtype) # 假设 swa_o 已经是 flatten 后的或者未 flatten
    # 为了配合你的代码 rearrange(o, 'B L h d->B L (h d)')，我们假设 swa_o 输入是 [B, L, H_q, D]
    swa_o = swa_o.view(B, L, H_q, D)

    # 生成 Landmarks
    chunk_lmks = get_lmks(k_norm, chunk_size)

    # -------------------------------------------------------------------------
    # Path A: Sparse Mode (模拟 self.hsa_mode == 'sparse')
    # -------------------------------------------------------------------------
    
    # A1. TopK Selection
    # indices: [B, L, H_q, K], scores: [B, L, H_q, K]
    indices, scores = online_softmax_topk_head(
        q_norm, 
        chunk_lmks, 
        lse_sum, 
        topk, 
        chunk_size, 
        window_size=window_size, 
        is_causal=True
    )

    # A2. Softmax & Weight Calculation (完全复刻你的代码逻辑)
    if not enable_softmax1:
        # (B, L, h_q, K + 1)
        cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1)], dim=-1)
        swa_weight_idx = -1
    else:
        # (B, L, h_q, K + 2)
        zeros = torch.zeros(B, L, H_q, 1, device=device, dtype=scores.dtype)
        cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1), zeros], dim=-1)
        swa_weight_idx = -2

    # 计算 Softmax
    chunk_weights = F.softmax(cat_scores, dim=-1).to(dtype) # (B, L, h_q, K_total)

    hsa_o_sparse = HSA_block_M_head(
        q_norm, 
        k_norm, 
        v, 
        chunk_weights, 
        indices, 
        block_size=chunk_size,
        mask_last_token=True # 你的代码中开启了这个
    )

    # A4. Residual Connection
    swa_o_weight_sparse = chunk_weights[:, :, :, swa_weight_idx] # (B, L, h_q)
    o_sparse = hsa_o_sparse + swa_o * swa_o_weight_sparse.unsqueeze(-1)
    o_sparse = rearrange(o_sparse, 'B L h d->B L (h d)')

    # -------------------------------------------------------------------------
    # Path B: Dense Mode (模拟 self.hsa_mode == 'dense')
    # -------------------------------------------------------------------------
    
    # B1. Dense Interface
    # 假设 HSA_dense_interface 返回 (output, all_weights, lse_index)
    hsa_o_dense_raw, chunk_weights_dense, lse_idx_dense = HSA_dense_interface(
        q=q_norm,
        k=k_norm,
        v=v,
        lmk=chunk_lmks,
        lse_swa=lse_sum,
        block_size=chunk_size,
        window_size=window_size,
        enable_softmax1=enable_softmax1,
        mask_last_token=True # 保持一致
    )
    
    chunk_weights_dense = chunk_weights_dense.to(dtype)

    # B2. Residual Connection
    swa_o_weight_dense = chunk_weights_dense[:, :, :, lse_idx_dense]
    o_dense = hsa_o_dense_raw + swa_o * swa_o_weight_dense.unsqueeze(-1)
    o_dense = rearrange(o_dense, 'B L h d->B L (h d)')

    # -------------------------------------------------------------------------
    # 验证结果
    # -------------------------------------------------------------------------
    assert_close("Final Output", o_dense, o_sparse, ratio=0.005)
    print("Test Passed!")

if __name__ == "__main__":
    # 手动运行测试
    test_dense_sparse_equivalence_full_coverage(False)
    test_dense_sparse_equivalence_full_coverage(True)
