import torch
from typing import Optional, Tuple
from ops.topk_group import online_topk_group
from ops.topk_head import online_topk_head
from ops.topk_head_softmax import online_softmax_topk_head

def online_topk(
    q: torch.Tensor,
    lmks: torch.Tensor,
    topk: int,
    selection_strategy: str,
    block_size: int,
    is_causal: bool = False,
    lse_swa: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified interface for HSA TopK selection strategies.

    Args:
        q: Query tensor.
           Shape: [B, L, h_q, D] for 'group' strategy.
           Shape: [B, L, h_kv, G, D] for 'head' and 'softmax_head' strategies.
        lmks: Landmarks (Key/Value summary) tensor. Shape: [B, S, h_kv, D].
        topk: Number of blocks to select.
        selection_strategy: Strategy name. Options:
            - 'group': Selects topk blocks based on group-level aggregation (sum over heads).
            - 'head': Selects topk blocks independently for each head (max pooling).
            - 'softmax_head': Selects topk blocks using softmax-normalized scores combined with sliding window attention LSE.
        block_size: The size of the token block (chunk size). Used for causal masking.
        is_causal: Whether to apply causal masking (query cannot see future chunks).
        lse_swa: (Optional) LogSumExp of Sliding Window Attention. 
                 Required ONLY if selection_strategy is 'softmax_head'.
                 Shape: [B, L, h_q].

    Returns:
        indices: Selected block indices. Shape: [B, L, h_kv, topk].
        scores: Selection scores. 
                Shape: [B, L, h_kv, topk] for 'group'.
                Shape: [B, L, h_kv, G, topk] for 'head' and 'softmax_head'.
    """
    
    if selection_strategy == 'group':
        return online_topk_group(
            q=q, 
            lmks=lmks, 
            topk=topk, 
            block_size=block_size, 
            is_causal=is_causal
        )
        
    elif selection_strategy == 'head':
        return online_topk_head(
            q=q, 
            lmks=lmks, 
            topk=topk, 
            block_size=block_size, 
            is_causal=is_causal
        )
        
    elif selection_strategy == 'softmax_head':
        if lse_swa is None:
            raise ValueError("Argument 'lse_swa' is required when selection_strategy='softmax_head'.")
        
        return online_softmax_topk_head(
            q=q, 
            lmks=lmks, 
            lse_swa=lse_swa, 
            topk=topk, 
            block_size=block_size, 
            is_causal=is_causal
        )
        
    else:
        valid_strategies = ['group', 'head', 'softmax_head']
        raise ValueError(f"Unknown selection_strategy: '{selection_strategy}'. "
                         f"Valid options are: {valid_strategies}")