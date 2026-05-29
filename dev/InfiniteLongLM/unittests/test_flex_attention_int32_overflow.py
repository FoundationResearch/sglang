"""
测试 compiled_flex_attention_inference 在超长序列下是否会触发 Triton 64-bit 索引溢出。

背景：
  - 在 lhsa_layer.py 中，compiled_flex_attention_inference 被 @torch.compile(dynamic=True) 装饰
  - 当 KV 张量总元素数超过 int32 上限 (2^31 - 1) 时，Triton 模板不支持 64-bit 索引，会报错：
    NotImplementedError: 64-bit indexing is not yet implemented for triton templates
  - 从实际报错的 traceback 可知张量形状为：
    Q: [1, 16, q_len, 64]   (B=1, h_q=16, head_dim=64, decode时q_len=1)
    K: [1, 4, kv_len, 64]    (B=1, h_kv=4, head_dim=64, enable_gqa=True)
    V: [1, 4, kv_len, 64]

用法（来自 lhsa_layer.py forward）：
    flex_attn_func = compiled_flex_attention_inference if past_key_value is not None else compiled_flex_attention
    swa_o, lse_sum = flex_attn_func(
        hsa_q_norm.transpose(1, 2),   # (B, h_q, L, head_dim)
        hsa_k_norm.transpose(1, 2),   # (B, h_hsa_kv, KV_LEN, head_dim)
        hsa_v.transpose(1, 2),        # (B, h_hsa_kv, KV_LEN, head_dim)
        window_size=self.hsa_sliding_window,
        chunk_size=self.chunk_size
    )

运行方式：
    python unittests/test_flex_attention_int32_overflow.py [--seq_len SEQ_LEN] [--q_len Q_LEN]
    
    默认 seq_len=8388608 (8M)，q_len=1（模拟 decode 阶段），可通过参数调整。
    
    示例：
    # 测试 8M decode（q_len=1，预期报错）
    python unittests/test_flex_attention_int32_overflow.py --seq_len 8388608
    
    # 测试 8M chunk prefill（q_len=4096）
    python unittests/test_flex_attention_int32_overflow.py --seq_len 8388608 --q_len 4096
    
    # 测试 2M decode（预期正常）
    python unittests/test_flex_attention_int32_overflow.py --seq_len 2097152
    
    # 测试 4M decode
    python unittests/test_flex_attention_int32_overflow.py --seq_len 4194304
"""

import argparse
import sys
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# ============================================================
# 与 lhsa_layer.py 中完全一致的函数定义
# ============================================================
@torch.compile(dynamic=True)
def compiled_flex_attention_inference(q, k, v, window_size, chunk_size):
    # inference (chunk prefill or decoding)
    Q_LEN = q.shape[-2]
    KV_LEN = k.shape[-2]
    
    def block_causal_mask(b, h, q_idx, kv_idx):
        real_q_idx = q_idx + (KV_LEN - Q_LEN)
        
        start = real_q_idx - window_size + 1
        chunk_start = (start // chunk_size) * chunk_size
        return (kv_idx >= chunk_start) & (kv_idx <= real_q_idx) & ((kv_idx + 1) % chunk_size != 0)
    
    block_mask = create_block_mask(block_causal_mask, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN)
    
    return flex_attention(
        q, 
        k, 
        v, 
        block_mask=block_mask, 
        enable_gqa=True, 
        return_lse=True
    )


def compute_numel_info(B, h_q, h_kv, q_len, kv_len, head_dim):
    """计算张量元素数并判断是否超过 int32 上限"""
    q_numel = B * h_q * q_len * head_dim
    k_numel = B * h_kv * kv_len * head_dim
    int32_max = 2**31 - 1
    return {
        "q_numel": q_numel,
        "k_numel": k_numel,
        "q_exceeds_int32": q_numel > int32_max,
        "k_exceeds_int32": k_numel > int32_max,
        "int32_max": int32_max,
    }


def test_compiled_flex_attention_inference(
    seq_len: int,
    q_len: int = 1,
    B: int = 1,
    h_q: int = 16,
    h_kv: int = 4,
    head_dim: int = 64,
    window_size: int = 4096,
    chunk_size: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    严格按照 lhsa_layer.py 中的用法构造输入，测试 compiled_flex_attention_inference。
    
    模拟 decode 阶段：Q 长度很短（默认 1），KV 长度为完整序列长度。
    
    参数说明（对应 lhsa_layer.py 中的配置）：
    - B: batch_size
    - h_q: hsa_heads (= num_attention_heads // hsa_denom)，对应 Q 的 head 数
    - h_kv: h_hsa_kv (= hsa_heads // hsa_qk_ratio)，对应 K/V 的 head 数（GQA）
    - head_dim: hidden_size // num_attention_heads
    - seq_len: KV 序列长度（即已缓存的 KV cache 长度）
    - q_len: Q 序列长度（decode 时为 1，chunk prefill 时可以更大）
    - window_size: hsa_sliding_window
    - chunk_size: chunk_size
    """
    
    kv_len = seq_len
    
    # 计算元素数信息
    info = compute_numel_info(B, h_q, h_kv, q_len, kv_len, head_dim)
    
    print("=" * 70)
    print(f"测试 compiled_flex_attention_inference")
    print(f"=" * 70)
    print(f"参数配置:")
    print(f"  B={B}, h_q={h_q}, h_kv={h_kv}, head_dim={head_dim}")
    print(f"  q_len={q_len}, kv_len={kv_len} ({kv_len / 1024 / 1024:.1f}M)")
    print(f"  window_size={window_size}, chunk_size={chunk_size}")
    print(f"  dtype={dtype}")
    print(f"")
    print(f"张量形状:")
    print(f"  Q: [{B}, {h_q}, {q_len}, {head_dim}]")
    print(f"  K: [{B}, {h_kv}, {kv_len}, {head_dim}]")
    print(f"  V: [{B}, {h_kv}, {kv_len}, {head_dim}]")
    print(f"")
    print(f"元素数分析:")
    print(f"  Q 元素数: {info['q_numel']:,} {'(超过 int32!)' if info['q_exceeds_int32'] else '(未超过 int32)'}")
    print(f"  K 元素数: {info['k_numel']:,} {'(超过 int32!)' if info['k_exceeds_int32'] else '(未超过 int32)'}")
    print(f"  int32 上限: {info['int32_max']:,}")
    print(f"")
    
    if info['q_exceeds_int32'] or info['k_exceeds_int32']:
        print(f"⚠️  预期会触发 Triton 64-bit 索引错误!")
    else:
        print(f"✅ 元素数在 int32 范围内，预期正常运行")
    print(f"")
    
    # 构造输入张量（与 lhsa_layer.py forward 中一致）
    # lhsa_layer.py 中的调用：
    #   hsa_q_norm.transpose(1, 2)  -> (B, h_q, Q_LEN, head_dim)
    #   hsa_k_norm.transpose(1, 2)  -> (B, h_hsa_kv, KV_LEN, head_dim)
    #   hsa_v.transpose(1, 2)       -> (B, h_hsa_kv, KV_LEN, head_dim)
    # decode 阶段：Q_LEN=1 (或很小), KV_LEN=完整序列长度
    print(f"正在分配张量...")
    
    # 使用 empty 而非 randn 来减少分配时间，对于测试编译阶段的错误足够了
    q = torch.empty(B, h_q, q_len, head_dim, dtype=dtype, device=device)
    k = torch.empty(B, h_kv, kv_len, head_dim, dtype=dtype, device=device)
    v = torch.empty(B, h_kv, kv_len, head_dim, dtype=dtype, device=device)
    
    q_mem = q.numel() * q.element_size() / 1024 / 1024 / 1024
    k_mem = k.numel() * k.element_size() / 1024 / 1024 / 1024
    v_mem = v.numel() * v.element_size() / 1024 / 1024 / 1024
    print(f"  Q 显存: {q_mem:.2f} GiB")
    print(f"  K 显存: {k_mem:.2f} GiB")
    print(f"  V 显存: {v_mem:.2f} GiB")
    print(f"  总显存: {q_mem + k_mem + v_mem:.2f} GiB")
    print(f"")
    
    # 调用 compiled_flex_attention_inference（与 lhsa_layer.py 中完全一致的调用方式）
    print(f"正在调用 compiled_flex_attention_inference (torch.compile dynamic=True)...")
    print(f"首次调用会触发编译，可能需要较长时间...")
    print(f"")
    
    
    with torch.no_grad():
        swa_o, lse_sum = compiled_flex_attention_inference(
            q, k, v,
            window_size=window_size,
            chunk_size=chunk_size,
        )
    
    print(f"✅ 调用成功!")
    print(f"  swa_o shape: {swa_o.shape}")
    print(f"  lse_sum shape: {lse_sum.shape}")
    return True


def test_multiple_lengths(q_len=1):
    """测试多个 KV 序列长度，找到触发 64-bit 索引错误的临界点"""
    
    # 模型配置（与实际报错时的配置一致）
    B = 1
    h_q = 16
    h_kv = 4
    head_dim = 64
    
    test_lengths = [
        ("2M", 2 * 1024 * 1024),
        ("4M", 4 * 1024 * 1024),
        ("8M", 8 * 1024 * 1024),
    ]
    
    print("=" * 70)
    print(f"元素数分析（不实际分配张量，q_len={q_len}，模拟{'decode' if q_len == 1 else 'chunk prefill'}阶段）")
    print("=" * 70)
    
    for name, kv_len in test_lengths:
        info = compute_numel_info(B, h_q, h_kv, q_len, kv_len, head_dim)
        q_ratio = info['q_numel'] / info['int32_max']
        k_ratio = info['k_numel'] / info['int32_max']
        print(f"\n{name} (q_len={q_len}, kv_len={kv_len:,}):")
        print(f"  Q [{B}, {h_q}, {q_len}, {head_dim}] -> {info['q_numel']:,} 元素 ({q_ratio:.4f}x int32_max) {'⚠️ 超限!' if info['q_exceeds_int32'] else '✅'}")
        print(f"  K [{B}, {h_kv}, {kv_len}, {head_dim}] -> {info['k_numel']:,} 元素 ({k_ratio:.2f}x int32_max) {'⚠️ 超限!' if info['k_exceeds_int32'] else '✅'}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="测试 compiled_flex_attention_inference 在超长序列下的 Triton 64-bit 索引溢出问题"
    )
    parser.add_argument(
        "--seq_len", type=int, default=8 * 1024 * 1024,
        help="KV 序列长度，默认 8M (8388608)"
    )
    parser.add_argument(
        "--q_len", type=int, default=1,
        help="Q 序列长度，默认 1（模拟 decode 阶段）"
    )
    parser.add_argument(
        "--B", type=int, default=1,
        help="batch size，默认 1"
    )
    parser.add_argument(
        "--h_q", type=int, default=16,
        help="Q 的 head 数（hsa_heads），默认 16"
    )
    parser.add_argument(
        "--h_kv", type=int, default=4,
        help="K/V 的 head 数（h_hsa_kv），默认 4"
    )
    parser.add_argument(
        "--head_dim", type=int, default=64,
        help="head 维度，默认 64"
    )
    parser.add_argument(
        "--window_size", type=int, default=4096,
        help="hsa_sliding_window，默认 4096"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512,
        help="chunk_size，默认 512"
    )
    parser.add_argument(
        "--analysis_only", action="store_true",
        help="仅做元素数分析，不实际运行"
    )
    args = parser.parse_args()
    
    # 先打印多个长度的元素数分析
    test_multiple_lengths(q_len=args.q_len)
    
    if args.analysis_only:
        print("仅分析模式，不实际运行。")
        return
    
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，无法运行测试")
        sys.exit(1)
    
    # 打印 GPU 信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GiB)")
    print()
    
    # 运行测试
    success = test_compiled_flex_attention_inference(
        seq_len=args.seq_len,
        q_len=args.q_len,
        B=args.B,
        h_q=args.h_q,
        h_kv=args.h_kv,
        head_dim=args.head_dim,
        window_size=args.window_size,
        chunk_size=args.chunk_size,
    )
    
    print()
    if success:
        print("🎉 测试通过，未触发 64-bit 索引错误")
    else:
        print("💥 测试失败，触发了错误（详见上方输出）")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
