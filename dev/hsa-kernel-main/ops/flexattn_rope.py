import math
import gc
from typing import Optional, Tuple, Union

import torch
from torch.nn.attention.flex_attention import flex_attention, BlockMask, create_block_mask
from torch.utils.checkpoint import checkpoint

from ops.rope_tilelang_fp32 import rope_rotary_pos_emb_bhld
from liger_kernel.transformers.rope import liger_rotary_pos_emb

def create_causal_mask_with_window_size(window_size: int, chunk_size: int):
    """
    返回一个 mask_mod，用于 create_block_mask(...)
    """
    def block_causal_mask(b, h, q_idx, kv_idx, aux_tensors=None):
        # q_idx / kv_idx 通常是 0-d int tensor
        ws = kv_idx.new_full((), window_size)
        cs = kv_idx.new_full((), chunk_size)

        kv_clamped = torch.maximum(kv_idx, ws)          # max(kv_idx, window_size)
        start = (kv_clamped - ws) // cs * cs            # ((max - ws)//chunk)*chunk
        return q_idx >= start

    return block_causal_mask


def make_rope_cos_sin(L, D, device, dtype):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device=device).float() / D))
    t = torch.arange(L, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, :, :].to(dtype)  # (1, L, D)
    sin = emb.sin()[None, :, :].to(dtype)
    return cos, sin
def flex_attention_ckpt(
    q, k, v, cos, sin,
    block_mask: BlockMask,
    scale: float,
    enable_gqa: bool = False,
    return_lse: bool = True
):
    """
    Baseline: Checkpointed FlexAttention with RoPE applied inside using Liger Kernel.
    Input q, k, v are expected to be (B, L, H, D).
    """
    def _inner(q_in, k_in, v_in, c, s):
        # q_in: (B, L, H, D) -> (B, H, L, D)
        q_bhld = q_in.transpose(1, 2).contiguous()
        k_bhld = k_in.transpose(1, 2).contiguous()
        v_bhld = v_in.transpose(1, 2).contiguous()

        q_rope, k_rope = liger_rotary_pos_emb(q_bhld, k_bhld, c, s)
        out = flex_attention(
            q_rope, k_rope, v_bhld,
            score_mod=None,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=return_lse
        )
        
        if return_lse:
            out_tensor, lse = out
            # out_tensor: (B, H, L, D) -> (B, L, H, D)
            # lse: (B, H, L) -> (B, L, H)
            return out_tensor.transpose(1, 2).contiguous(), lse.transpose(1, 2).contiguous()
        else:
            return out.transpose(1, 2).contiguous()

    return checkpoint(
        _inner,
        q, k, v, cos, sin,
        use_reentrant=False,
        preserve_rng_state=False
    )


def _identity_score_mod(score, b, h, q_idx, kv_idx):
    return score

_IDENTITY_GRAPH_CACHE = {}  # key: (device, dtype) -> (fw_graph, joint_graph)

def _get_identity_graphs(device: torch.device, dtype: torch.dtype):
    """
    backward kernel 需要 fw_graph/joint_graph 参数。
    我们固定用 identity score_mod，所以 graphs 可以缓存起来复用。
    """
    key = (str(device), dtype)
    if key in _IDENTITY_GRAPH_CACHE:
        return _IDENTITY_GRAPH_CACHE[key]

    from torch.fx.experimental.proxy_tensor import make_fx

    example_score = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    example_b = torch.tensor(0, device=device, dtype=torch.int32)
    example_h = torch.tensor(0, device=device, dtype=torch.int32)
    example_q = torch.tensor(0, device=device, dtype=torch.int32)
    example_kv = torch.tensor(0, device=device, dtype=torch.int32)

    fw_graph = make_fx(_identity_score_mod)(example_score, example_b, example_h, example_q, example_kv)

    def joint_identity(score, b, h, q_idx, kv_idx, tangent):
        # d(identity)/dscore = 1
        return [tangent, None, None, None, None]

    example_tangent = torch.tensor(1.0, device=device, dtype=dtype)
    joint_graph = make_fx(joint_identity)(
        example_score, example_b, example_h, example_q, example_kv, example_tangent
    )

    _IDENTITY_GRAPH_CACHE[key] = (fw_graph, joint_graph)
    return fw_graph, joint_graph

from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch._higher_order_ops.flex_attention import flex_attention_backward as flex_attention_backward_hop
class FlexAttention_Rope(torch.autograd.Function):
    """
    自定义 FlexAttention + RoPE（BLHD）
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: BlockMask,
        scale: float,
        kernel_options: Optional[dict] = None,
        apply_rope: bool = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        kernel_options = kernel_options or {}
        kernel_options.setdefault("OUTPUT_LOGSUMEXP", True)

        # Input is (B, L, H, D), convert to (B, H, L, D) for kernel
        q_bhld = query.transpose(1, 2).contiguous()
        k_bhld = key.transpose(1, 2).contiguous()
        v_bhld = value.transpose(1, 2).contiguous()

        q_for_kernel = q_bhld
        k_for_kernel = k_bhld
        if apply_rope:
            q_for_kernel, k_for_kernel = rope_rotary_pos_emb_bhld(q_bhld, k_bhld, cos, sin, dtype=query.dtype)

        out_bhld, lse = flex_attention_hop(
            q_for_kernel,
            k_for_kernel,
            v_bhld,
            _identity_score_mod,
            block_mask.as_tuple(),
            scale,
            kernel_options,
            (),  # score_mod_other_buffers
            (),  # mask_mod_other_buffers
        )
        # Output back to (B, L, H, D)
        out_blhd = out_bhld.transpose(1, 2).contiguous()
        
        # Save BHLD tensors to avoid re-transpose in backward
        # lse is (B, H, L), keep it as is for backward kernel
        ctx.save_for_backward(query, key, value, out_blhd, lse)
        ctx.block_mask = block_mask
        ctx.scale = scale
        ctx.kernel_options = kernel_options
        ctx.apply_rope = apply_rope
        ctx.cos = cos
        ctx.sin = sin

        # Return lse as (B, L, H) to user
        lse_blh = lse.transpose(1, 2).contiguous()
        return out_blhd, lse_blh * math.log(2)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_lse: Optional[torch.Tensor]):

        # Retrieve BHLD tensors
        query, key, value, out_blhd, lse = ctx.saved_tensors # lse is (B, H, L)
        block_mask = ctx.block_mask
        scale = ctx.scale
        kernel_options = ctx.kernel_options

        # grad_out is (B, L, H, D), convert to (B, H, L, D)
        grad_out_bhld = grad_out.transpose(1, 2).contiguous()

        # base-e -> base-2 梯度
        if grad_lse is not None:
            # grad_lse is (B, L, H), convert back to (B, H, L)
            grad_lse_bhl = grad_lse.transpose(1, 2).contiguous()
            grad_logsumexp_base2 = grad_lse_bhl * math.log(2)
        else:
            grad_logsumexp_base2 = torch.zeros_like(lse)
            
        # Input is (B, L, H, D), convert to (B, H, L, D) for kernel
        q_bhld = query.transpose(1, 2).contiguous()
        k_bhld = key.transpose(1, 2).contiguous()
        v_bhld = value.transpose(1, 2).contiguous()
        out_bhld = out_blhd.transpose(1, 2).contiguous()
        
        q_for_kernel = q_bhld
        k_for_kernel = k_bhld
        if ctx.apply_rope:
            q_for_kernel, k_for_kernel = rope_rotary_pos_emb_bhld(q_bhld, k_bhld, ctx.cos, ctx.sin, dtype=q_bhld.dtype)

        fw_graph, joint_graph = _get_identity_graphs(q_bhld.device, q_bhld.dtype)

        grad_query_bhld, grad_key_bhld, grad_value_bhld = flex_attention_backward_hop(
            q_for_kernel,
            k_for_kernel,
            v_bhld,
            out_bhld,
            lse,
            grad_out_bhld,
            grad_logsumexp_base2,
            fw_graph,
            joint_graph,
            block_mask.as_tuple(),
            scale,
            kernel_options,
            (),  # score_mod_other_buffers
            (),  # mask_mod_other_buffers
        )

        if ctx.apply_rope:
            grad_query_bhld, grad_key_bhld = rope_rotary_pos_emb_bhld(
                grad_query_bhld, grad_key_bhld, ctx.cos, -ctx.sin, dtype=q_bhld.dtype
            )

        # Convert grads back to (B, L, H, D)
        grad_query = grad_query_bhld.transpose(1, 2).contiguous()
        grad_key = grad_key_bhld.transpose(1, 2).contiguous()
        grad_value = grad_value_bhld.transpose(1, 2).contiguous()

        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None




def flex_attention_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_mask: Optional[BlockMask] = None,
    apply_rope: Optional[bool] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    return_lse: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    B, L, H, D = q.shape
    
    out, lse = FlexAttention_Rope.apply(
        q, k, v,
        block_mask,
        scale,
        {},
        apply_rope,
        cos, sin
    )

    return (out, lse) if return_lse else out


import pytest

@pytest.mark.parametrize("B, H, L, D", [
    (2, 4, 128, 64),
    (2, 8, 512, 64),
    (4, 8, 1024, 64)
])
def test_flex_attention_rope_correctness(B, H, L, D):
    device = "cuda" 
    dtype = torch.float16
    window_size = 128
    chunk_size = 64
    
    print(f"\nConfig: B={B}, H={H}, L={L}, D={D}, dtype={dtype}")
    torch.manual_seed(42)

    # Generate (B, L, H, D) inputs
    q = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)

    cos, sin = make_rope_cos_sin(L, D, device, dtype)
    cos = cos.expand(B, -1, -1).contiguous()
    sin = sin.expand(B, -1, -1).contiguous()

    grad_output = torch.randn(B, L, H, D, device=device, dtype=dtype)

    q_target = q.detach().clone().requires_grad_(True)
    k_target = k.detach().clone().requires_grad_(True)
    v_target = v.detach().clone().requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    mask_mod = create_causal_mask_with_window_size(window_size, chunk_size)
    block_mask = create_block_mask(mask_mod, B, H, L, L, device=device)

    scale = 1.0 / math.sqrt(D)

    # Baseline
    out_ref, lse_ref = flex_attention_ckpt(
        q_ref, k_ref, v_ref, cos, sin,
        block_mask=block_mask,
        scale=scale,
        return_lse=True
    )

    # Target
    # out_target, lse_target = FlexAttention_Rope.apply(
    #     q_target, k_target, v_target,
    #     block_mask,
    #     scale,
    #     {},      # kernel_options
    #     True,    # apply_rope
    #     cos, sin
    # )
    out_target, lse_target = flex_attention_rope(
        q_target, k_target, v_target,
        block_mask=block_mask,
        apply_rope=True,
        cos=cos,
        sin=sin,
        scale=scale,
        return_lse=True
    )
    print(out_ref.shape)
    print(lse_target.shape)

    def get_abs_err(x, y):
        return (x - y).flatten().abs().max().item()

    def get_err_ratio(x, y):
        err = (x - y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base + 1e-12)

    def assert_close(prefix, ref, tri, ratio=0.05):
        abs_err = get_abs_err(ref, tri)
        rel_ratio = get_err_ratio(ref, tri)
        msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
        print(msg)
        assert rel_ratio <= ratio, f"❌ {prefix} Failed! {msg}"

    print("-" * 20 + " Forward Check " + "-" * 20)
    assert_close("OUT", out_ref, out_target)
    assert_close("LSE", lse_ref, lse_target)

    out_ref.backward(grad_output, retain_graph=True)
    out_target.backward(grad_output, retain_graph=True)

    print("-" * 20 + " Backward Check " + "-" * 20)
    assert_close("DQ", q_ref.grad, q_target.grad)
    assert_close("DK", k_ref.grad, k_target.grad)
    assert_close("DV", v_ref.grad, v_target.grad)
    
    print(f"✅ Test Passed for B={B}, H={H}, L={L}, D={D}")


def benchmark_flex_attention_rope():
    print("\n" + "-" * 40)
    print("Performance Test")
    print("-" * 40)
    
    device = "cuda"
    dtype = torch.float16
    B, H, L, D = 4, 8, 1024, 64
    window_size = 128
    chunk_size = 64
    
    torch.manual_seed(42)
    q = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    cos, sin = make_rope_cos_sin(L, D, device, dtype)
    cos = cos.expand(B, -1, -1).contiguous()
    sin = sin.expand(B, -1, -1).contiguous()
    
    mask_mod = create_causal_mask_with_window_size(window_size, chunk_size)
    block_mask = create_block_mask(mask_mod, B, H, L, L, device=device)
    scale = 1.0 / math.sqrt(D)

    # Prepare inputs for baseline
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    def zero_grads(tensors):
        for t in tensors:
            if t.grad is not None:
                t.grad.zero_()

    def run_benchmark(name, func, args, kwargs, grad_tensors):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        mem_start = torch.cuda.memory_allocated()

        res = func(*args, **kwargs)
        out = res[0] if isinstance(res, tuple) else res

        torch.cuda.synchronize()
        mem_after_fwd = torch.cuda.memory_allocated()
        activation_mem = (mem_after_fwd - mem_start) / 1024**2

        # warmup
        loss = out.sum()
        loss.backward()
        zero_grads(grad_tensors)

        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        loops = 20
        start_evt.record()
        for _ in range(loops):
            res = func(*args, **kwargs)
            out = res[0] if isinstance(res, tuple) else res
            loss = out.sum()
            loss.backward()
            zero_grads(grad_tensors)
        end_evt.record()
        torch.cuda.synchronize()

        avg_time = start_evt.elapsed_time(end_evt) / loops
        print(f"[{name}]")
        print(f"  - Activation Memory: {activation_mem:.2f} MB")
        print(f"  - Latency (Fwd+Bwd): {avg_time:.2f} ms")

    run_benchmark(
        "Baseline (Checkpoint + Liger RoPE + FlexAttention, block_mask only)",
        flex_attention_ckpt,
        args=(q_ref, k_ref, v_ref, cos, sin),
        kwargs={"block_mask": block_mask, "scale": scale, "return_lse": True},
        grad_tensors=[q_ref, k_ref, v_ref]
    )

    run_benchmark(
        "Target (Custom Autograd + BHLD RoPE + FlexAttention, block_mask only)",
        FlexAttention_Rope.apply,
        args=(q, k, v, block_mask, scale, {}, True, cos, sin),
        kwargs={},
        grad_tensors=[q, k, v]
    )


if __name__ == "__main__":
    # Run correctness tests manually
    params = [
        (2, 4, 128, 64),
        (2, 8, 512, 64),
    ]
    for p in params:
        test_flex_attention_rope_correctness(*p)
        
    # Run benchmark
    benchmark_flex_attention_rope()