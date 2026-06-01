"""Triton kernels that update HSA's cuda-graph metadata buffers in place.

For cuda graph capture/replay to work, all per-step metadata must live in
buffers with stable addresses across replays.  These kernels write directly
into pre-allocated buffers (no intermediate allocations), which makes them
graph-safe.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def hsa_build_page_table_1_kernel(
    req_to_token_ptr,        # [num_req, total_kv]  int32
    req_pool_indices_ptr,    # [bs]                 int64 or int32
    page_table_1_ptr,        # [bs, max_seqlen_k]   int32  (output)
    stride_rt0: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy req_to_token[req_pool_indices[b], :max_seqlen_k] into page_table_1[b].

    Grid is (bs, cdiv(max_seqlen_k, BLOCK)) so the seq_len copy parallelises
    across SMs — critical at long context (R15.1).  The previous single-block
    grid spent ~15 ms per call serialising a 524K-element copy.
    """
    b = tl.program_id(0)
    chunk = tl.program_id(1)
    req_idx = tl.load(req_pool_indices_ptr + b).to(tl.int64)
    base_src = req_idx * stride_rt0
    base_dst = b * max_seqlen_k
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < max_seqlen_k
    src = tl.load(req_to_token_ptr + base_src + offs, mask=mask, other=0)
    tl.store(page_table_1_ptr + base_dst + offs, src, mask=mask)


@triton.jit
def hsa_copy_seq_lens_kernel(
    seq_lens_ptr,            # [bs]  int64 or int32
    cache_seqlens_i32_ptr,   # [bs]  int32  (output)
    SRC_IS_I64: tl.constexpr,
):
    """cache_seqlens_i32[b] = seq_lens[b]   (with dtype cast if needed)."""
    b = tl.program_id(0)
    if SRC_IS_I64:
        v = tl.load(seq_lens_ptr + b).to(tl.int32)
    else:
        v = tl.load(seq_lens_ptr + b)
    tl.store(cache_seqlens_i32_ptr + b, v)


def update_hsa_cg_buffers(
    *,
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    page_table_1_buf: torch.Tensor,
    cache_seqlens_i32_buf: torch.Tensor,
):
    """Update the two main HSA cuda-graph buffers in place via triton kernels.

    No intermediate Python-side allocations — safe to call inside a cuda
    graph capture context.
    """
    max_seqlen_k = page_table_1_buf.shape[1]
    # 4096-element blocks (8 warps × 512 ints) — small enough to fit lots of
    # SMs in parallel for the seq_len dimension; large enough that loop +
    # launch overhead stays small.  At max_seqlen_k = 524288 that's 128
    # parallel grid blocks per batch instead of one — saturates GB200 SMs.
    BLOCK = min(4096, triton.next_power_of_2(max_seqlen_k))
    num_chunks = (int(max_seqlen_k) + BLOCK - 1) // BLOCK

    hsa_build_page_table_1_kernel[(bs, num_chunks)](
        req_to_token,
        req_pool_indices,
        page_table_1_buf,
        stride_rt0=int(req_to_token.stride(0)),
        max_seqlen_k=int(max_seqlen_k),
        BLOCK=int(BLOCK),
        num_warps=8,
    )

    hsa_copy_seq_lens_kernel[(bs,)](
        seq_lens,
        cache_seqlens_i32_buf,
        SRC_IS_I64=(seq_lens.dtype == torch.int64),
        num_warps=1,
    )
