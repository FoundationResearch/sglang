import torch
from torch.nn.attention.flex_attention import flex_attention
from torch.utils.checkpoint import checkpoint
from liger_kernel.transformers.rope import liger_rotary_pos_emb

from typing import Optional, Tuple, Callable
from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd


class FlashAttnRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        mask_mod: Optional[Callable] = None,
        block_sparse_tensors: Optional[object] = None,
    ):
        apply_rope = pos_emb is not None
        
        if apply_rope:
            cos, sin = pos_emb
            if cos.ndim == 3:
                cos = cos.unsqueeze(2)
                sin = sin.unsqueeze(2)
            q_rope, k_rope = liger_rotary_pos_emb(q, k, cos, sin)
        else:
            q_rope, k_rope = q, k

        out, lse = _flash_attn_fwd(
            q_rope, k_rope, v,
            softmax_scale=softmax_scale, causal=causal,
            window_size_left=window_size[0], window_size_right=window_size[1],
            learnable_sink=learnable_sink, softcap=softcap,
            num_splits=num_splits, pack_gqa=pack_gqa, mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
        )
        
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.params = (pos_emb, softmax_scale, causal, window_size, softcap, deterministic)
        
        return out, lse
    
    @staticmethod
    def backward(ctx, dout, dlse=None):
        q_nope, k_nope, v, out, lse = ctx.saved_tensors
        pos_emb, softmax_scale, causal, window_size, softcap, deterministic = ctx.params
        
        apply_rope = pos_emb is not None
        
        if apply_rope: 
            cos, sin = pos_emb
            if cos.ndim == 3:
                cos = cos.unsqueeze(2)
                sin = sin.unsqueeze(2)
            q_rope, k_rope = liger_rotary_pos_emb(q_nope, k_nope, cos, sin)
        else:
            q_rope, k_rope = q_nope, k_nope
        
        dq_rope, dk_rope, dv = _flash_attn_bwd(
            q_rope, k_rope, v, out, dout, lse,
            softmax_scale, causal, softcap,
            window_size_left=window_size[0], window_size_right=window_size[1],
            deterministic=deterministic,
        )
        
        if apply_rope:
            dq_nope, dk_nope = liger_rotary_pos_emb(dq_rope, dk_rope, cos, -sin)
        else:
            dq_nope, dk_nope = dq_rope, dk_rope
        
        return dq_nope, dk_nope, dv, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_rope_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[torch.Tensor] = None,
    full_block_idx: Optional[torch.Tensor] = None,
    mask_block_cnt: Optional[torch.Tensor] = None,
    mask_block_idx: Optional[torch.Tensor] = None,
):
    """
    FlashAttention with on-the-fly RoPE wrapper.
    API is compatible with standard flash_attn_func, with the addition of pos_emb.
    """
    
    block_sparse_tensors = None
    if any(t is not None for t in [full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx]):
        from flash_attn.cute.interface import BlockSparseTensorsTorch
        block_sparse_tensors = BlockSparseTensorsTorch(
            full_block_cnt=full_block_cnt,
            full_block_idx=full_block_idx,
            mask_block_cnt=mask_block_cnt,
            mask_block_idx=mask_block_idx,
        )

    return FlashAttnRoPEFunc.apply(
        q,
        k,
        v,
        pos_emb,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        mask_mod,
        block_sparse_tensors,
    )
    
    


def apply_rotary_pos_emb(q, k, cos, sin):
    return liger_rotary_pos_emb(q, k, cos, sin)


def make_rope_cos_sin(L, D, device, dtype):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device=device).float() / D))
    t = torch.arange(L, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :].to(dtype)
    sin = emb.sin()[None, None, :, :].to(dtype)
    return cos, sin

# torch的flexattention with checkpoint作为baseline
def flex_attention_ckpt(
    q, k, v, cos, sin,
    score_mod=None,
    block_mask=None,
    scale=None,
    enable_gqa=False,
    return_lse=True
):
    def _inner(q_in, k_in, v_in, c, s, mask, scale_val):
        
        q_rope, k_rope = apply_rotary_pos_emb(q_in, k_in, c, s)
        
        return flex_attention(
            q_rope, k_rope, v_in,
            score_mod=score_mod,
            block_mask=mask,
            scale=scale_val,
            enable_gqa=enable_gqa,
            return_lse=return_lse
        )
    
    return checkpoint(
        _inner,
        q, k, v, cos, sin, block_mask, scale,
        use_reentrant=False,
        preserve_rng_state=False
    )




def get_testing_inputs(B, H, L, D, device, dtype):
    torch.manual_seed(42)
    # 基础数据，形状为 (B, L, H, D) -> 符合 FlashAttention 标准布局
    q = torch.randn(B, L, H, D, device=device, dtype=dtype)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype)
    
    # RoPE cache
    cos, sin = make_rope_cos_sin(L, D, device, dtype)
    
    return q, k, v, cos, sin

def test_correctness(B=2, H=4, L=1024, D=64, dtype=torch.bfloat16):
    print(f"\n{'='*20} 数值正确性测试 (Correctness) {'='*20}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Skipping: CUDA not available.")
        return

    q_ref, k_ref, v_ref, cos, sin = get_testing_inputs(B, H, L, D, device, dtype)
    
    # --- 1. FlexAttention (Baseline) ---
    # Flex 需要 (B, H, L, D)
    q_flex = q_ref.permute(0, 2, 1, 3).clone().detach().requires_grad_(True)
    k_flex = k_ref.permute(0, 2, 1, 3).clone().detach().requires_grad_(True)
    v_flex = v_ref.permute(0, 2, 1, 3).clone().detach().requires_grad_(True)
    
    # Flex Mask (Causal)
    from torch.nn.attention.flex_attention import create_block_mask
    block_mask = create_block_mask(lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, B, H, L, L, device=device)
    
    out_flex, lse_flex = flex_attention_ckpt(
        q_flex, k_flex, v_flex, cos, sin,
        block_mask=block_mask,
        scale=1.0 / (D ** 0.5), # 显式指定 scale 以保持一致
        return_lse=True
    )
    loss_flex = out_flex.sum()
    loss_flex.backward()
    
    # --- 2. Custom FlashAttnRoPEFunc (Target) ---
    # Flash 需要 (B, L, H, D)
    q_flash = q_ref.clone().detach().requires_grad_(True)
    k_flash = k_ref.clone().detach().requires_grad_(True)
    v_flash = v_ref.clone().detach().requires_grad_(True)
    
    out_flash, lse_flash = flash_attn_rope_func(
        q_flash, k_flash, v_flash,
        pos_emb=(cos, sin),
        causal=True
    )
    
    # Flash 输出是 (B, L, H, D)，转置以便对比
    out_flash_permuted = out_flash.permute(0, 2, 1, 3)
    loss_flash = out_flash_permuted.sum()
    loss_flash.backward()
    
    # --- 3. 验证对比 ---
    
    # 验证 Forward Output
    diff_out = (out_flex - out_flash_permuted).abs().max()
    print(f"Max Output Difference: {diff_out.item():.6f}")
    assert diff_out < 1e-2, "Forward output mismatch!"
    
    # 验证 Backward Gradients
    # Flash 梯度是 (B, L, H, D)，转置回 (B, H, L, D) 对比
    grad_q_diff = (q_flex.grad - q_flash.grad.permute(0, 2, 1, 3)).abs().max()
    grad_k_diff = (k_flex.grad - k_flash.grad.permute(0, 2, 1, 3)).abs().max()
    grad_v_diff = (v_flex.grad - v_flash.grad.permute(0, 2, 1, 3)).abs().max()
    
    print(f"Max Grad Q Difference: {grad_q_diff.item():.6f}")
    print(f"Max Grad K Difference: {grad_k_diff.item():.6f}")
    print(f"Max Grad V Difference: {grad_v_diff.item():.6f}")
    
    assert grad_q_diff < 1e-2, "Grad Q mismatch!"
    assert grad_k_diff < 1e-2, "Grad K mismatch!"
    assert grad_v_diff < 1e-2, "Grad V mismatch!"
    print(">> SUCCESS: 数值一致性验证通过！")


def test_performance(B=4, H=16, L=4096, D=128, dtype=torch.bfloat16):
    print(f"\n{'='*20} 性能与显存测试 (Performance) {'='*20}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": return

    print(f"Config: B={B}, H={H}, L={L}, D={D}, dtype={dtype}")
    
    q_ref, k_ref, v_ref, cos, sin = get_testing_inputs(B, H, L, D, device, dtype)
    
    # 准备 Flex 数据
    q_flex = q_ref.permute(0, 2, 1, 3).clone().detach().requires_grad_(True)
    k_flex = k_ref.permute(0, 2, 1, 3).clone().detach().requires_grad_(True)
    v_flex = v_ref.permute(0, 2, 1, 3).clone().detach().requires_grad_(True)
    from torch.nn.attention.flex_attention import create_block_mask
    block_mask = create_block_mask(lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, B, H, L, L, device=device)
    
    # 准备 Flash 数据
    q_flash = q_ref.clone().detach().requires_grad_(True)
    k_flash = k_ref.clone().detach().requires_grad_(True)
    v_flash = v_ref.clone().detach().requires_grad_(True)
    
    import gc
    
    def run_benchmark(name, func, args, kwargs, input_tensor_grad):
        # --- 显存测试 ---
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        mem_start = torch.cuda.memory_allocated()
        
        # Forward
        res = func(*args, **kwargs)
        out = res[0] if isinstance(res, tuple) else res
        
        torch.cuda.synchronize()
        mem_after_fwd = torch.cuda.memory_allocated()
        activation_mem = (mem_after_fwd - mem_start) / 1024**2
        
        # Backward (warmup for timing)
        loss = out.sum()
        loss.backward()
        if input_tensor_grad.grad is not None: input_tensor_grad.grad.zero_()
        
        # --- 速度测试 ---
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
            if input_tensor_grad.grad is not None: input_tensor_grad.grad.zero_()
        end_evt.record()
        torch.cuda.synchronize()
        
        avg_time = start_evt.elapsed_time(end_evt) / loops
        
        print(f"[{name}]")
        print(f"  - Activation Memory (Forward后增量): {activation_mem:.2f} MB")
        print(f"  - Latency (Fwd+Bwd): {avg_time:.2f} ms")

    # 1. Test Flex
    run_benchmark(
        "FlexAttention (Checkpoint)",
        flex_attention_ckpt,
        args=(q_flex, k_flex, v_flex, cos, sin),
        kwargs={"block_mask": block_mask, "scale": 1.0/(D**0.5)},
        input_tensor_grad=q_flex
    )

    # 2. Test Custom Flash
    run_benchmark(
        "Custom FlashAttn RoPE",
        flash_attn_rope_func,
        args=(q_flash, k_flash, v_flash),
        kwargs={"pos_emb": (cos, sin), "causal": True, "softmax_scale": 1.0/(D**0.5)},
        input_tensor_grad=q_flash
    )

if __name__ == "__main__":
    test_correctness(B=1, H=8, L=512, D=64)
    test_performance(B=1, H=8, L=512, D=64)
