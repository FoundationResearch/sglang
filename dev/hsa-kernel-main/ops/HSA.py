import torch
from typing import Optional
from ops.hsa_fwd_bwd_block_M_tilelang_group import HSA_block_M_group
from ops.hsa_fwd_bwd_block_M_tilelang_head import HSA_block_M_head

def HSA(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    selection_strategy: str,
    block_size: int = 32,
    sm_scale: Optional[float] = None,
    block_M: int = 0,
    mask_last_token: bool = False
) -> torch.Tensor:
    """
    Unified interface for Hierarchical Sparse Attention (HSA) kernels with shape validation.

    Args:
        q: Query tensor. Shape: [B, L, HQ, D].
        k: Key tensor. Shape: [B, L, H, D].
        v: Value tensor. Shape: [B, L, H, D].
        weights: Attention weights.
                 Shape: [B, L, H, S] for 'group' strategy.
                 Shape: [B, L, HQ, S] for 'head' and 'softmax_head' strategies.
        indices: Selected block indices. Shape: [B, L, H, S].
        selection_strategy: 'group', 'head', or 'softmax_head'.
        ...
    """
    
    B, L, HQ, D = q.shape
    _, _, H, _ = k.shape
    _, _, _, S = indices.shape

    if selection_strategy == 'group':
        expected_shape = (B, L, H, S)
        if weights.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for strategy '{selection_strategy}': "
                f"Expected weights shape {expected_shape}, but got {weights.shape}. "
                f"(B={B}, L={L}, H={H}, S={S})"
            )
            
        return HSA_block_M_group(
            q=q, k=k, v=v, weights=weights, indices=indices,
            block_size=block_size, sm_scale=sm_scale, block_M=block_M,
            mask_last_token=mask_last_token
        )
        
    elif selection_strategy in ['head', 'softmax_head']:
        expected_shape = (B, L, HQ, S)
        if weights.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for strategy '{selection_strategy}': "
                f"Expected weights shape {expected_shape}, but got {weights.shape}. "
                f"(B={B}, L={L}, HQ={HQ}, S={S})"
            )
            
        return HSA_block_M_head(
            q=q, k=k, v=v, weights=weights, indices=indices,
            block_size=block_size, sm_scale=sm_scale, block_M=block_M,
            mask_last_token=mask_last_token
        )
        
    else:
        valid_strategies = ['group', 'head', 'softmax_head']
        raise ValueError(f"Unknown selection_strategy: '{selection_strategy}'. "
                         f"Valid options are: {valid_strategies}")