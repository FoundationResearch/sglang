# """
# 最小化测试脚本：验证 compiled_flex_attention 前向 + 反向能否跑通
# 用法：python code_exp/test_flexattn_cute.py [--seq_len 8192]
# """

# import argparse
# import torch
# from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# @torch.compile(dynamic=False)
# def compiled_flex_attention(q, k, v, window_size, chunk_size):
#     """training only"""
#     L = q.shape[-2]

#     def block_causal_mask(b, h, q_idx, kv_idx):
#         start = q_idx - window_size + 1
#         chunk_start = (start // chunk_size) * chunk_size
#         return (kv_idx >= chunk_start) & (kv_idx <= q_idx) & ((kv_idx + 1) % chunk_size != 0)

#     block_mask = create_block_mask(block_causal_mask, B=None, H=None, Q_LEN=L, KV_LEN=L)
#     return flex_attention(
#         q, k, v,
#         block_mask=block_mask,
#         enable_gqa=True,
#         return_lse=True,
#     )


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--seq_len", type=int, default=8192)
#     parser.add_argument("--num_q_heads", type=int, default=4)
#     parser.add_argument("--num_kv_heads", type=int, default=1)
#     parser.add_argument("--head_dim", type=int, default=64)
#     parser.add_argument("--window_size", type=int, default=512)
#     parser.add_argument("--chunk_size", type=int, default=64)
#     args = parser.parse_args()

#     device = "cuda"
#     dtype = torch.bfloat16

#     assert torch.cuda.is_available(), "需要 CUDA GPU"
#     assert args.seq_len % args.chunk_size == 0, \
#         f"seq_len ({args.seq_len}) 必须是 chunk_size ({args.chunk_size}) 的整数倍"

#     print(f"构造数据: B={args.batch_size}, L={args.seq_len}, "
#           f"Q_H={args.num_q_heads}, KV_H={args.num_kv_heads}, D={args.head_dim}")

#     q = torch.randn(args.batch_size, args.num_q_heads, args.seq_len, args.head_dim,
#                      device=device, dtype=dtype, requires_grad=True)
#     k = torch.randn(args.batch_size, args.num_kv_heads, args.seq_len, args.head_dim,
#                      device=device, dtype=dtype, requires_grad=True)
#     v = torch.randn(args.batch_size, args.num_kv_heads, args.seq_len, args.head_dim,
#                      device=device, dtype=dtype, requires_grad=True)

#     # 前向
#     print("运行前向...")
#     out, lse = compiled_flex_attention(q, k, v,
#                                        window_size=args.window_size,
#                                        chunk_size=args.chunk_size)
#     torch.cuda.synchronize()
#     print(f"前向通过 ✓  out.shape={out.shape}, lse.shape={lse.shape}")

#     # 反向
#     print("运行反向...")
#     loss = out.sum() + lse.sum()
#     loss.backward()
#     torch.cuda.synchronize()
#     print(f"反向通过 ✓  q.grad.shape={q.grad.shape}")

#     print("全部通过！")


# if __name__ == "__main__":
#     main()




import torch
from flash_attn.cute.interface import flash_attn_func

B, S, H, D = 1, 128, 8, 128
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, causal=True)
print(out.shape)
