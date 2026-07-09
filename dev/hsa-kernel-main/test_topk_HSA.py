import torch
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional, Tuple
from ops.topk import online_topk
from ops.HSA import HSA
from ops.hsa_fwd_bwd_block_M_tilelang_head_dense import HSA_block_M_head_dense
torch.manual_seed(42)


def hsa_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    selection_strategy: str,
    chunk_size: int,
    sm_scale: float,
    mask_last_token: bool = False
) -> torch.Tensor:
    """
    Unified PyTorch reference for HSA.
    Supports 'group', 'head', and 'softmax_head' strategies.
    """
    B, L, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    
    valid_mask = (indices >= 0)  # (B, L, H, K)
    safe_indices = indices.clamp_min(0)

    N = L // chunk_size
    k_chunks = rearrange(k, 'B (N S) h d -> B N S h d', S=chunk_size)
    v_chunks = rearrange(v, 'B (N S) h d -> B N S h d', S=chunk_size)
    
    k_chunks_perm = k_chunks.permute(0, 3, 1, 2, 4) # (B, H, N, S, D)
    v_chunks_perm = v_chunks.permute(0, 3, 1, 2, 4)
    
    idx_perm = safe_indices.permute(0, 2, 1, 3) # (B, H, L, K)
    
    # k_flat: (B*H, N, S, D)
    k_flat = k_chunks_perm.reshape(B*H, N, chunk_size, D)
    v_flat = v_chunks_perm.reshape(B*H, N, chunk_size, D)
    
    # idx_flat: (B*H, L, K)
    idx_flat = idx_perm.reshape(B*H, L, indices.shape[-1])
    
    idx_flat_long = idx_flat.view(B*H, -1).long() # (B*H, L*K)
    
    idx_expanded = idx_flat_long.unsqueeze(-1).unsqueeze(-1) # (BH, LK, 1, 1)
    idx_expanded = idx_expanded.expand(-1, -1, chunk_size, D) # (BH, LK, S, D)
    
    gather_k_flat = k_flat.gather(1, idx_expanded) # (BH, LK, S, D)
    gather_v_flat = v_flat.gather(1, idx_expanded)
    
    # Reshape back: (B, H, L, K, S, D)
    gather_k = gather_k_flat.view(B, H, L, indices.shape[-1], chunk_size, D)
    gather_v = gather_v_flat.view(B, H, L, indices.shape[-1], chunk_size, D)
    
    # Current K: (B, H, L, K, S, D) -> (B, L, S, K, H, D)
    gather_k = gather_k.permute(0, 2, 4, 3, 1, 5) # (B, L, S, K, H, D)
    gather_v = gather_v.permute(0, 2, 4, 3, 1, 5)

    # K: (B, L, S, K, H, D) -> (B, L, S, K, HQ, D)
    k_ = torch.repeat_interleave(gather_k, dim=-2, repeats=G)
    v_ = torch.repeat_interleave(gather_v, dim=-2, repeats=G)

    # q: (B, L, HQ, D) -> (B, L, 1, 1, HQ, D) to broadcast over S and K
    q_exp = q.unsqueeze(2).unsqueeze(3) 

    # qk: (B, L, S, K, HQ)
    # q_exp: (B, L, 1, 1, HQ, D)
    # k_:    (B, L, S, K, HQ, D)
    # sum over D
    qk = torch.sum(q_exp.float() * k_.float(), dim=-1)
    qk = qk * float(sm_scale)
    
    if mask_last_token:
        qk[:, :, -1, :, :] = float("-inf")

    p = torch.softmax(qk, dim=2)

    # o_k: (B, L, K, HQ, D)
    # p:  (B, L, S, K, HQ) -> (B, L, S, K, HQ, 1)
    # v_: (B, L, S, K, HQ, D)
    # sum over S
    o_k = torch.sum(p.unsqueeze(-1) * v_.float(), dim=2)

    w_masked = weights.clone()
    
    if selection_strategy == 'group':
        # weights: (B, L, H, K)
        # valid_mask: (B, L, H, K)
        w_masked = w_masked.masked_fill(~valid_mask, 0)
        w_exp = torch.repeat_interleave(w_masked, dim=-2, repeats=G).float() # (B, L, HQ, K)
        
    elif selection_strategy in ['head', 'softmax_head']:
        valid_mask_expanded = torch.repeat_interleave(valid_mask, dim=-2, repeats=G)
        w_masked = w_masked.masked_fill(~valid_mask_expanded, 0)
        w_exp = w_masked.float() # (B, L, HQ, K)
    else:
        raise ValueError(f"Unknown strategy: {selection_strategy}")

    w_exp = w_exp.permute(0, 1, 3, 2).unsqueeze(-1)
    
    o_ref = torch.sum(o_k * w_exp, dim=2) # Sum over K -> (B, L, HQ, D)
    
    return o_ref.to(torch.float32)



def topk_torch_ref(
    q: torch.Tensor,
    lmks: torch.Tensor,
    topk: int,
    selection_strategy: str,
    block_size: int,
    is_causal: bool = False,
    lse_swa: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified PyTorch reference for TopK selection.
    """
    
    if selection_strategy == 'group':
        return _ref_topk_group(q, lmks, topk, block_size, is_causal)
    elif selection_strategy == 'head':
        return _ref_topk_head(q, lmks, topk, block_size, is_causal)
    elif selection_strategy == 'softmax_head':
        if lse_swa is None:
            raise ValueError("lse_swa required for softmax_head")
        return _ref_softmax_topk_head(q, lmks, lse_swa, topk, block_size, is_causal)
    else:
        raise ValueError(f"Unknown strategy: {selection_strategy}")

def _ref_topk_group(q, lmks, topk, block_size, is_causal):
    B, L, h_q, D = q.shape
    _, S, h_kv, _ = lmks.shape
    sm_scale = 1.0 / math.sqrt(D)
    G = h_q // h_kv
    q_group_sum = q.view(B, L, h_kv, G, D).sum(dim=3)
    scores_ref = torch.einsum("blkd,bskd->blks", q_group_sum.float(), lmks.float()) * sm_scale
    
    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1)
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)
        current_block_idx = i_idx // block_size
        causal_mask = j_idx >= current_block_idx
        scores_ref = scores_ref.masked_fill(causal_mask.unsqueeze(0).unsqueeze(2), float('-inf'))

    scores_topk, indices_topk = torch.topk(scores_ref, k=topk, dim=-1, sorted=False)
    indices_sorted, order = torch.sort(indices_topk, dim=-1)
    scores_sorted = torch.gather(scores_topk, -1, order)
    
    indices_sorted = indices_sorted.masked_fill(scores_sorted == float('-inf'), -1)
    
    return indices_sorted, scores_sorted

def _ref_topk_head(q, k_lmks, topk, block_size, is_causal):
    B, L, h_kv, G, D = q.shape
    S = k_lmks.shape[1]
    sm_scale = 1.0 / math.sqrt(D)
    scores_all = torch.einsum("blhgd,bshd->blhgs", q.float(), k_lmks.float())
    
    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1)
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)
        current_block_idx = i_idx // block_size
        causal_mask = j_idx >= current_block_idx
        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all = scores_all.masked_fill(causal_mask_expanded, float('-inf'))

    scores_max_pooling = scores_all.max(dim=3).values
    _, topk_indices = torch.topk(scores_max_pooling, k=topk, dim=-1, sorted=False)
    indices_sorted, order = torch.sort(topk_indices, dim=-1)
    order_expanded = order.unsqueeze(3).expand(-1, -1, -1, G, -1)
    indices_expanded = indices_sorted.unsqueeze(3).expand(-1, -1, -1, G, -1)
    scores_sorted = torch.gather(scores_all, -1, indices_expanded)
    scores_sorted = scores_sorted * sm_scale
    
    selected_max_scores = torch.gather(scores_max_pooling, -1, indices_sorted)
    indices_sorted = indices_sorted.masked_fill(selected_max_scores == float('-inf'), -1)
    
    return indices_sorted, scores_sorted

def _ref_softmax_topk_head(q, k_lmks, lse_swa, topk, block_size, is_causal):
    B, L, h_kv, G, D = q.shape
    S = k_lmks.shape[1]
    
    logits_hsa = torch.einsum("blhgd,bshd->blhgs", q.float(), k_lmks.float())
    sm_scale = 1.0 / math.sqrt(D)
    logits_hsa_scaled = logits_hsa * sm_scale

    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1)
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)
        current_block_idx = i_idx // block_size
        causal_mask = j_idx >= current_block_idx
        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        logits_hsa_scaled = logits_hsa_scaled.masked_fill(causal_mask_expanded, float('-inf'))

    lse_hsa = torch.logsumexp(logits_hsa_scaled, dim=-1)
    
    if lse_swa.dim() == 3:
        lse_swa_view = lse_swa.view(B, L, h_kv, G)
    else:
        lse_swa_view = lse_swa
        
    lse_total = torch.logaddexp(lse_swa_view, lse_hsa)
    log_probs = logits_hsa_scaled - lse_total.unsqueeze(-1)
    
    scores_max_pooling = log_probs.max(dim=3).values
    _, topk_indices = torch.topk(scores_max_pooling, k=topk, dim=-1, sorted=False)
    
    indices_sorted, order = torch.sort(topk_indices, dim=-1)
    
    order_expanded = order.unsqueeze(3).expand(-1, -1, -1, G, -1)
    indices_expanded = indices_sorted.unsqueeze(3).expand(-1, -1, -1, G, -1)
    
    scores_sorted = torch.gather(logits_hsa_scaled, -1, indices_expanded)
    
    selected_max_scores = torch.gather(scores_max_pooling, -1, indices_sorted)
    indices_sorted = indices_sorted.masked_fill(selected_max_scores == float('-inf'), -1)
    
    return indices_sorted, scores_sorted


def test_topk_hsa_unified_correctness():
    B, L, H, HQ, D = 1, 1024, 1, 8, 128
    block_size = 64
    topk = 8
    is_causal = True
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    block_M = 32 // G
    scale = 1.0 / math.sqrt(D)
    
    print(f"Config: B={B}, L={L}, H={H}, HQ={HQ}, D={D}, block_size={block_size}, topk={topk}, is_causal={is_causal}, dtype={dtype}")

    strategies = ['group']

    for strategy in strategies:
        print(f"\n>>> Testing Strategy: {strategy}")

        torch.manual_seed(42)

        Q = torch.randn((B, L, HQ, D), dtype=dtype, device=device, requires_grad=True)
        K = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)
        V = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)

        assert L % block_size == 0
        lmks = K[:, block_size - 1::block_size, :, :].detach().clone().requires_grad_(True)
        lse_swa = torch.randn((B, L, HQ), dtype=dtype, device=device)

        if strategy == 'group':
            Q_topk = Q.detach().clone().requires_grad_(True)  # [B, L, HQ, D]
        else:
            Q_topk = Q.detach().view(B, L, H, G, D).clone().requires_grad_(True)  # [B, L, H, G, D]

        print("\n--- TopK Forward Correctness ---")

        topk_indices, topk_scores = online_topk(
            q=Q_topk,
            lmks=lmks,
            topk=topk,
            selection_strategy=strategy,
            block_size=block_size,
            is_causal=is_causal,
            lse_swa=lse_swa if strategy == 'softmax_head' else None
        )

        if strategy == 'group':
            q_ref_topk = Q_topk.detach().clone()  # [B, L, HQ, D]
        else:
            q_ref_topk = Q_topk.detach().clone()  # [B, L, H, G, D]
        
        lmks_ref = lmks.detach().clone()

        ref_indices, ref_scores = topk_torch_ref(
            q=q_ref_topk,
            lmks=lmks_ref,
            topk=topk,
            selection_strategy=strategy,
            block_size=block_size,
            is_causal=is_causal,
            lse_swa=lse_swa if strategy == 'softmax_head' else None
        )
        
        valid_indices_mask = (ref_indices >= 0)
        indices_match = (topk_indices.to(ref_indices.dtype) == ref_indices)
        
        if valid_indices_mask.sum() > 0:
            match_rate = indices_match[valid_indices_mask].float().mean().item()
        else:
            match_rate = 1.0
            
        print(f"TopK Indices Match Rate (Valid Elements): {match_rate*100:.6f}%")

        valid_scores_mask = (ref_scores > -1e9) & (topk_scores.float() > -1e9)
        
        if valid_scores_mask.sum() == 0:
            print("Warning: No valid scores to compare (all masked?)")
            max_score_diff = 0.0
        else:
            score_diff = torch.abs(ref_scores[valid_scores_mask].float() - topk_scores[valid_scores_mask].float())
            max_score_diff = score_diff.max().item()
            
        print(f"TopK Scores Max Diff (Valid Only): {max_score_diff:.6e}")
        
        if match_rate >= 0.99 and max_score_diff < 1.0:
             print("✅ TopK Forward PASSED")
        else:
             print("❌ TopK Forward FAILED")

        print("\n--- TopK Backward Correctness ---")
        
        grad_output_topk = torch.randn_like(topk_scores)
        grad_output_topk[topk_scores < -1e4] = 0.0
        
        if Q_topk.grad is not None: Q_topk.grad.zero_()
        if lmks.grad is not None: lmks.grad.zero_()
        
        topk_indices_bwd, topk_scores_bwd = online_topk(
            q=Q_topk,
            lmks=lmks,
            topk=topk,
            selection_strategy=strategy,
            block_size=block_size,
            is_causal=is_causal,
            lse_swa=lse_swa if strategy == 'softmax_head' else None
        )
        
        topk_scores_bwd.backward(grad_output_topk)
        grad_q_online = Q_topk.grad.clone() if Q_topk.grad is not None else torch.zeros_like(Q_topk)
        grad_lmks_online = lmks.grad.clone() if lmks.grad is not None else torch.zeros_like(lmks)
        
        q_ref_bwd = Q_topk.detach().clone().requires_grad_(True)
        lmks_ref_bwd = lmks.detach().clone().requires_grad_(True)
        
        if strategy == 'group':
            B_ref, L_ref, HQ_ref, D_ref = q_ref_bwd.shape
            _, S_ref, H_ref, _ = lmks_ref_bwd.shape
            G_ref = HQ_ref // H_ref
            sm_scale_ref = 1.0 / math.sqrt(D_ref)
            
            q_group_sum = q_ref_bwd.view(B_ref, L_ref, H_ref, G_ref, D_ref).sum(dim=3)
            scores_all_ref = torch.einsum("blkd,bskd->blks", q_group_sum.float(), lmks_ref_bwd.float()) * sm_scale_ref
            
            if is_causal:
                i_idx = torch.arange(L_ref, device=device).unsqueeze(1)
                j_idx = torch.arange(S_ref, device=device).unsqueeze(0)
                causal_mask = (j_idx * block_size) > i_idx
                current_block_idx = i_idx // block_size
                causal_mask = j_idx >= current_block_idx
                scores_all_ref = scores_all_ref.masked_fill(causal_mask.unsqueeze(0).unsqueeze(2), float('-inf'))
                
            safe_indices = topk_indices.clone()
            safe_indices[safe_indices < 0] = 0
            scores_gathered_ref = torch.gather(scores_all_ref, -1, safe_indices.long())
            
        elif strategy == 'head':
            scores_all_ref = torch.einsum("blhgd,bshd->blhgs", q_ref_bwd.float(), lmks_ref_bwd.float())
            sm_scale_ref = 1.0 / math.sqrt(D)
            scores_all_ref = scores_all_ref * sm_scale_ref
            
            if is_causal:
                i_idx = torch.arange(L, device=device).unsqueeze(1)
                j_idx = torch.arange(S_ref, device=device).unsqueeze(0)
                current_block_idx = i_idx // block_size
                causal_mask = j_idx >= current_block_idx
                causal_mask_expanded = causal_mask.view(1, L, 1, 1, S_ref) # [1, L, 1, 1, S_ref]
                scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))
                
            safe_indices = topk_indices.clone()
            safe_indices[safe_indices < 0] = 0
            indices_expanded = safe_indices.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
            scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded)
            
        elif strategy == 'softmax_head':
            logits_hsa = torch.einsum("blhgd,bshd->blhgs", q_ref_bwd.float(), lmks_ref_bwd.float())
            sm_scale_ref = 1.0 / math.sqrt(D)
            logits_hsa_scaled = logits_hsa * sm_scale_ref
            
            if is_causal:
                i_idx = torch.arange(L, device=device).unsqueeze(1)
                j_idx = torch.arange(S_ref, device=device).unsqueeze(0)
                current_block_idx = i_idx // block_size
                causal_mask = j_idx >= current_block_idx
                causal_mask_expanded = causal_mask.view(1, L, 1, 1, S_ref)
                logits_hsa_scaled = logits_hsa_scaled.masked_fill(causal_mask_expanded, float('-inf'))
            
            safe_indices = topk_indices.clone()
            safe_indices[safe_indices < 0] = 0
            indices_expanded = safe_indices.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
            scores_gathered_ref = torch.gather(logits_hsa_scaled, -1, indices_expanded)

        loss_ref = (scores_gathered_ref * grad_output_topk.float()).sum()
        loss_ref.backward()
        grad_q_ref = q_ref_bwd.grad.clone() if q_ref_bwd.grad is not None else torch.zeros_like(q_ref_bwd)
        grad_lmks_ref = lmks_ref_bwd.grad.clone() if lmks_ref_bwd.grad is not None else torch.zeros_like(lmks_ref_bwd)
        
        grad_q_online = torch.nan_to_num(grad_q_online, 0.0)
        grad_lmks_online = torch.nan_to_num(grad_lmks_online, 0.0)
        grad_q_ref = torch.nan_to_num(grad_q_ref, 0.0)
        grad_lmks_ref = torch.nan_to_num(grad_lmks_ref, 0.0)
        
        diff_q = (grad_q_online - grad_q_ref).abs().max().item()
        diff_lmks = (grad_lmks_online - grad_lmks_ref).abs().max().item()
        
        norm_q_ref = grad_q_ref.norm().item()
        rel_err_q = diff_q / (norm_q_ref + 1e-6) if norm_q_ref > 0 else diff_q
        
        print(f"TopK Grad Q Max Diff: {diff_q:.6e}")
        print(f"TopK Grad LMKS Max Diff: {diff_lmks:.6e}")
        
        if rel_err_q < 0.1:
             print("✅ TopK Backward PASSED")
        else:
             print("❌ TopK Backward FAILED")

        print("\n--- HSA Forward Correctness ---")

        if strategy == 'group':
            # group: topk_scores [B, L, H, topk]
            scores_for_softmax = topk_scores
            lse_sum_sim = lse_swa.view(B, L, H, G).logsumexp(dim=3)
        else:
            # head/softmax_head: topk_scores [B, L, H, G, topk] -> [B, L, HQ, topk]
            scores_for_softmax = rearrange(topk_scores, 'b l h g s -> b l (h g) s')
            lse_sum_sim = lse_swa  # [B, L, HQ]

        cat_scores = torch.cat([scores_for_softmax, lse_sum_sim.unsqueeze(-1)], dim=-1)
        probs = F.softmax(cat_scores.float(), dim=-1).to(dtype)
        W_gen = probs[..., :-1]

        block_indices = topk_indices.int().detach()
        W = W_gen.detach().clone().requires_grad_(True)

        O_hsa = HSA(
            q=Q, k=K, v=V, weights=W, indices=block_indices,
            selection_strategy=strategy,
            block_size=block_size,
            sm_scale=scale,
            block_M=block_M,
            mask_last_token=True
        )

        O_ref = hsa_torch_ref(
            q=Q.float().detach(),
            k=K.float().detach(),
            v=V.float().detach(),
            weights=W.detach(),
            indices=block_indices,
            selection_strategy=strategy,
            chunk_size=block_size,
            sm_scale=scale,
            mask_last_token=True
        )

        fwd_diff = (O_hsa.float() - O_ref.float()).abs().max().item()
        print(f"HSA Forward Max Diff: {fwd_diff:.6e}")
        if fwd_diff < 1e-2:
            print("✅ HSA Forward PASSED")
        else:
            print("❌ HSA Forward FAILED")

        print("\n--- HSA Backward Correctness ---")
        grad_output = torch.randn_like(O_hsa)

        Q_ref = Q.detach().clone().float().requires_grad_(True)
        K_ref = K.detach().clone().float().requires_grad_(True)
        V_ref = V.detach().clone().float().requires_grad_(True)
        W_ref = W.detach().clone().float().requires_grad_(True)

        O_ref_bwd = hsa_torch_ref(
            q=Q_ref, k=K_ref, v=V_ref, weights=W_ref, indices=block_indices,
            selection_strategy=strategy,
            chunk_size=block_size,
            sm_scale=scale,
            mask_last_token=True
        )
        O_ref_bwd.backward(grad_output.float())

        if Q.grad is not None: Q.grad = None
        if K.grad is not None: K.grad = None
        if V.grad is not None: V.grad = None
        if W.grad is not None: W.grad = None

        O_hsa.backward(grad_output)

        for name, g_hsa, g_ref in [
            ("DQ", Q.grad, Q_ref.grad),
            ("DK", K.grad, K_ref.grad),
            ("DV", V.grad, V_ref.grad),
            ("DW", W.grad, W_ref.grad),
        ]:
            if g_hsa is None or g_ref is None:
                print(f"  {name}: None grad")
                continue

            diff = (g_hsa.float() - g_ref.float()).abs().max().item()
            print(f"  {name} Max Diff: {diff:.6e}")
            
            if diff < 0.5:
                print(f"  ✅ {name} PASSED")
            else:
                print(f"  ❌ {name} FAILED")
                



def FA3_ref_program(Q, K, V, is_causal, groups=1):
    # Q: [B, T, HQ, D_QK]
    # K: [B, T, HK, D_QK]
    # V: [B, T, HV, D_V]
    # HQ = HKV * groups
    
    # from flash_attn.flash_attn_interface import flash_attn_func
    # # print("Running FlashAttention3 reference implementation...")
    # output = flash_attn_func(Q, K, V, causal=is_causal)
    
    from example_gqa_bwd import attention
    # print("Running tilelang GQA reference implementation...")
    output = attention(Q, K, V, causal=is_causal, groups=groups, use_atomic=False)

    return output




# import torch.utils.benchmark as benchmark

# def build_block_indices_with_overlap(
#     B: int,
#     SEQ_LEN: int,
#     H: int,
#     S: int,
#     block_size: int,
#     target_overlap_ratio: float = 0.5,
#     device: str = "cuda",
# ) -> torch.Tensor:
#     """
#     构造 block_indices 张量，保证相邻token之间的重合度接近target_overlap_ratio
    
#     参数:
#         B: batch 大小
#         SEQ_LEN: 序列长度
#         H: head 数
#         S: 每个 query 选择的 block 数量
#         block_size: 每个 block 的大小
#         target_overlap_ratio: 目标重合度 [0,1]
#         device: 输出所在设备
#     """
#     assert 0.0 <= target_overlap_ratio <= 1.0, "target_overlap_ratio 必须在 [0, 1]"
    
#     num_blocks = (SEQ_LEN + block_size - 1) // block_size  # 向上取整
#     block_indices = torch.full((B, SEQ_LEN, H, S), -1, dtype=torch.int32, device=device)
    
#     for b in range(B):
#         for h in range(H):
#             # 为第一个token生成初始集合
#             t0 = 0
#             max_blocks_t0 = min((t0 // block_size) + 1, num_blocks)
#             num_select_t0 = min(S, max_blocks_t0)
            
#             if num_select_t0 > 0:
#                 idx_prev = torch.randperm(max_blocks_t0, device=device)[:num_select_t0]
#                 idx_prev_sorted = torch.sort(idx_prev)[0]
#                 block_indices[b, t0, h, :len(idx_prev_sorted)] = idx_prev_sorted
#             else:
#                 idx_prev = torch.tensor([], dtype=torch.int64, device=device)
            
#             # 为后续token生成集合，控制重合度
#             for t in range(1, SEQ_LEN):
#                 max_blocks_t = min((t // block_size) + 1, num_blocks)
#                 num_select_t = min(S, max_blocks_t)
                
#                 if num_select_t == 0:
#                     block_indices[b, t, h, :] = -1
#                     idx_prev = torch.tensor([], dtype=torch.int64, device=device)
#                     continue
                
#                 # 计算前一步实际选中的集合
#                 prev_indices = block_indices[b, t-1, h]
#                 actual_prev_set = prev_indices[prev_indices >= 0]
                
#                 # 确定重叠数量（四舍五入到最接近的整数）
#                 # 目标：|Γ(t) ∩ Γ(t-1)| / num_select_t ≈ target_overlap_ratio
#                 target_overlap_count = int(round(target_overlap_ratio * num_select_t))
#                 target_overlap_count = min(target_overlap_count, len(actual_prev_set), num_select_t)
                
#                 # 选择重叠部分：从前一步的集合中选取
#                 if target_overlap_count > 0 and len(actual_prev_set) > 0:
#                     # 找出前一步集合中在当前token可访问范围内的block
#                     available_overlap = actual_prev_set[actual_prev_set < max_blocks_t]
                    
#                     if len(available_overlap) >= target_overlap_count:
#                         # 如果有足够多的可重叠block
#                         perm = torch.randperm(len(available_overlap), device=device)
#                         overlap_blocks = available_overlap[perm[:target_overlap_count]]
#                     else:
#                         # 如果不够，使用所有可用的重叠block
#                         overlap_blocks = available_overlap.clone()
#                         target_overlap_count = len(available_overlap)
#                 else:
#                     overlap_blocks = torch.tensor([], dtype=torch.int64, device=device)
#                     target_overlap_count = 0
                
#                 # 选择新的block
#                 num_new_blocks = num_select_t - target_overlap_count
#                 if num_new_blocks > 0:
#                     # 找出所有可用的block（0 到 max_blocks_t-1）
#                     all_blocks = torch.arange(max_blocks_t, device=device)
                    
#                     # 排除已经选为重叠的block
#                     if target_overlap_count > 0:
#                         mask = torch.ones(max_blocks_t, dtype=torch.bool, device=device)
#                         mask[overlap_blocks] = False
#                         candidate_blocks = all_blocks[mask]
#                     else:
#                         candidate_blocks = all_blocks
                    
#                     # 排除前一步集合中的所有block（确保只选择新的）
#                     if len(actual_prev_set) > 0:
#                         prev_mask = torch.ones(len(candidate_blocks), dtype=torch.bool, device=device)
#                         for prev_idx in actual_prev_set:
#                             if prev_idx < max_blocks_t:
#                                 # 找出candidate_blocks中等于prev_idx的位置
#                                 matches = candidate_blocks == prev_idx
#                                 if matches.any():
#                                     prev_mask[matches] = False
#                         candidate_blocks = candidate_blocks[prev_mask]
                    
#                     # 从候选块中随机选择
#                     if len(candidate_blocks) >= num_new_blocks:
#                         perm = torch.randperm(len(candidate_blocks), device=device)
#                         new_blocks = candidate_blocks[perm[:num_new_blocks]]
#                     else:
#                         # 如果候选不够，使用所有候选
#                         new_blocks = candidate_blocks.clone()
#                         # 可能需要调整num_select_t
#                         num_select_t = target_overlap_count + len(new_blocks)
#                 else:
#                     new_blocks = torch.tensor([], dtype=torch.int64, device=device)
                
#                 # 合并重叠部分和新部分
#                 if target_overlap_count > 0 and len(new_blocks) > 0:
#                     idx_curr = torch.cat([overlap_blocks, new_blocks], dim=0)
#                 elif target_overlap_count > 0:
#                     idx_curr = overlap_blocks.clone()
#                 elif len(new_blocks) > 0:
#                     idx_curr = new_blocks.clone()
#                 else:
#                     idx_curr = torch.tensor([], dtype=torch.int64, device=device)
                
#                 # 升序排序
#                 idx_curr_sorted = torch.sort(idx_curr)[0]
#                 block_indices[b, t, h, :len(idx_curr_sorted)] = idx_curr_sorted
                
#                 # 更新前一步的集合
#                 idx_prev = idx_curr
#     # 打印最后一个query的block indices以供调试
#     # print(block_indices[0, SEQ_LEN-4, 0, :])
#     # print(block_indices[0, SEQ_LEN-3, 0, :])
#     # print(block_indices[0, SEQ_LEN-2, 0, :])
#     # print(block_indices[0, SEQ_LEN-1, 0, :])
#     return block_indices



import torch
def build_block_indices_block_M(
    B: int,
    SEQ_LEN: int,
    H: int,
    S: int,
    block_size: int,
    overlap_ratio: float = 0.5,
    block_M: int = 2,
    device: str = "cuda",
) -> torch.Tensor:
    """
    构造 block_indices 张量：
    - 在每个长度为 block_M 的 token 窗口内，相邻 token (t, t+1) 的选中块集合满足给定的重叠度。
    - 每个 query 的选中索引升序排列。
    - 不足 S 的填充为 -1。

    参数:
        B: batch 大小
        SEQ_LEN: 序列长度
        H: head 数
        S: 每个 query 选择的 block 数量
        block_size: 每个 block 的大小
        overlap_ratio: 相邻 token 之间的重叠比例 [0,1]
        block_M: 每个窗口内的 token 数（例如 pair=2，对 block_M kernel 可设为 M）
        device: 输出所在设备
    """
    import torch

    assert 0.0 <= overlap_ratio <= 1.0, "overlap_ratio 必须在 [0, 1]"
    assert block_M >= 1, "block_M 必须 >= 1"

    num_blocks = SEQ_LEN // block_size
    block_indices = torch.full((B, SEQ_LEN, H, S), -1, dtype=torch.int32, device=device)

    for b in range(B):
        for h in range(H):
            # 按 block_M 为一组滑动
            t = 0
            while t < SEQ_LEN:
                block_start = t
                block_end = min(t + block_M, SEQ_LEN)

                # 对这个窗口里的第一个 token 先生成索引
                t0 = block_start
                max_blocks_t0 = min(t0 // block_size + 1, num_blocks)
                if max_blocks_t0 <= 0:
                    # 这个 token 没有可用 block，直接跳过到下一组
                    t = block_end
                    continue

                num_select = min(S, max_blocks_t0)
                # 第一个 token 随机选
                idx_prev = torch.randperm(max_blocks_t0, device=device)[:num_select]
                idx_prev_sorted = torch.sort(idx_prev)[0]
                block_indices[b, t0, h, :len(idx_prev_sorted)] = idx_prev_sorted

                # 对窗口内其余 token：保证与前一个 token 保持 overlap_ratio
                for tt in range(t0 + 1, block_end):
                    max_blocks_tt = min(tt // block_size + 1, num_blocks)
                    if max_blocks_tt <= 0:
                        continue

                    num_select_tt = min(S, max_blocks_tt)

                    # 允许重叠的最大候选：当前 token 可用 block 与上一个的交集
                    # 这里简化为：从 idx_prev 中选 overlapped，再从其余可用 block 中选新块
                    num_overlap = int(overlap_ratio * num_select_tt)
                    num_overlap = min(num_overlap, len(idx_prev))

                    # 重叠部分：从 idx_prev 中随机取 num_overlap 个
                    if num_overlap > 0:
                        perm_prev = torch.randperm(len(idx_prev), device=device)
                        overlap_blocks = idx_prev[perm_prev[:num_overlap]]
                    else:
                        overlap_blocks = idx_prev.new_empty((0,), dtype=idx_prev.dtype)

                    # 剩余 block 候选：当前 token 可用的所有 block 中，剔除 overlap_blocks
                    remaining_blocks_all = torch.arange(max_blocks_tt, device=device)
                    mask = torch.ones(max_blocks_tt, dtype=torch.bool, device=device)
                    if overlap_blocks.numel() > 0:
                        mask[overlap_blocks] = False
                    candidates = remaining_blocks_all[mask]

                    num_new = num_select_tt - num_overlap
                    if num_new > 0 and candidates.numel() > 0:
                        perm_cand = torch.randperm(candidates.numel(), device=device)
                        new_blocks = candidates[perm_cand[:num_new]]
                        idx_curr = torch.cat([overlap_blocks, new_blocks], dim=0)
                    else:
                        idx_curr = overlap_blocks.clone()

                    # 升序写入
                    idx_curr_sorted = torch.sort(idx_curr)[0]
                    block_indices[b, tt, h, :len(idx_curr_sorted)] = idx_curr_sorted

                    # 下一轮的“上一个 token”索引
                    idx_prev = idx_curr

                # 跳到下一个窗口
                t = block_end

    return block_indices



def benchmark_hsa_vs_fa3_with_overlap():
    from datetime import datetime
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    device = "cuda"
    dtype = torch.bfloat16
    B, HQ, H, D = 16, 32, 4, 128
    groups = HQ // H
    print("groups:", groups)
    block_size = 64
    topk = 16
    strategy = 'head'
    is_causal = True
    sm_scale = 1.0 / math.sqrt(D)
    G = HQ // H
    block_M = 64//groups
    print("block_M:", block_M)
    overlap_ratio = 0.8
    
    # 测量参数
    warmup_steps = 10
    measure_steps = 50

    print(f"Benchmark Config: B={B}, HQ={HQ}, H={H}, D={D}, topk={topk}, overlap={overlap_ratio}")
    print(f"Settings: Warmup={warmup_steps}, Measure={measure_steps}")

    for L in [4096, 8192]:
        print(f"\n{'='*20} L={L} {'='*20}")
        
        # 1. 构造基础输入
        Q = torch.randn((B, L, HQ, D), dtype=dtype, device=device, requires_grad=True)
        K = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)
        V = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)
        grad_output = torch.randn((B, L, HQ, D), dtype=dtype, device=device)

        # 2. 构造具有重叠特性的 indices
        indices = build_block_indices_block_M(
        B=B,
        SEQ_LEN=L,
        H=H,
        S=topk,
        block_size=block_size,
        overlap_ratio=overlap_ratio,
        block_M=block_M,
        device=device,
    )
        # indices = build_block_indices_with_overlap(
        #     B=B,
        #     SEQ_LEN=L,
        #     H=H,
        #     S=topk,
        #     block_size=block_size,
        #     target_overlap_ratio=overlap_ratio,
        #     device=device,
        # )
        
        # 处理 indices
        safe_indices = indices.clone()
        
        # 3. 构造对应的 weights
        logits = torch.randn((B, L, HQ, topk), dtype=dtype, device=device)
        valid_mask = (indices.repeat_interleave(G, dim=2) != -1)
        logits = logits.masked_fill(~valid_mask, float('-inf'))
        W = torch.softmax(logits.float(), dim=-1).to(dtype).requires_grad_(True)

        # 4. 定义测试闭包
        def run_hsa_fwd():
            return HSA(Q, K, V, W, safe_indices, strategy, block_size, sm_scale, block_M, True)

        def run_fa3_fwd():
            return FA3_ref_program(Q, K, V, is_causal, groups)

        # 辅助测量函数
        def measure_time(func, steps, desc):
            # 必须同步等待之前的操作完成
            torch.cuda.synchronize()
            
            # 使用 CUDA Events 计时更精准
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(steps):
                func()
            end_event.record()
            
            # 等待所有 CUDA 核心完成
            torch.cuda.synchronize()
            
            # 返回平均耗时 (ms)
            return start_event.elapsed_time(end_event) / steps

        # 分别测量 HSA 和 FA3
        for name, run_fwd_func in [("HSA", run_hsa_fwd), ("FA3", run_fa3_fwd)]:
            print(f"\n--- {name} ---")
            
            # 预热 (使用完整的前向+后向)
            for _ in range(warmup_steps):
                out = run_fwd_func()
                out.backward(grad_output, retain_graph=True)
                # 清空梯度
                if name == "HSA":
                    Q.grad = K.grad = V.grad = W.grad = None
                else:
                    Q.grad = K.grad = V.grad = None
            
            # 测量前向时间 (不带梯度)
            def fwd_only():
                with torch.no_grad():
                    run_fwd_func()
            
            fwd_ms = measure_time(fwd_only, measure_steps, f"{name} FWD")
            
            # 测量完整的前向+后向时间
            def fwd_bwd():
                # 清空梯度
                if name == "HSA":
                    Q.grad = K.grad = V.grad = W.grad = None
                else:
                    Q.grad = K.grad = V.grad = None
                
                out = run_fwd_func()
                out.backward(grad_output, retain_graph=True)
            
            total_ms = measure_time(fwd_bwd, measure_steps, f"{name} FULL")
            
            # 计算后向时间
            bwd_ms = total_ms - fwd_ms
            
            print(f"FWD: {fwd_ms:.3f}ms | BWD: {bwd_ms:.3f}ms | TOTAL: {total_ms:.3f}ms")
            
            # 如果需要，可以计算速度比
            if name == "HSA":
                hsa_fwd_ms, hsa_bwd_ms, hsa_total_ms = fwd_ms, bwd_ms, total_ms
            else:
                fa3_fwd_ms, fa3_bwd_ms, fa3_total_ms = fwd_ms, bwd_ms, total_ms
        
        # 计算和打印速度比
        print(f"\n--- Speedup Comparison ---")
        print(f"FWD Speedup: {fa3_fwd_ms/hsa_fwd_ms:.2f}x")
        print(f"BWD Speedup: {fa3_bwd_ms/hsa_bwd_ms:.2f}x")
        print(f"TOTAL Speedup: {fa3_total_ms/hsa_total_ms:.2f}x")

        # 每个 L 结束后清理显存
        torch.cuda.empty_cache()
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))




def benchmark_hsa_dense_vs_fa3():
    from datetime import datetime
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    device = "cuda"
    dtype = torch.bfloat16
    B, HQ, H, D = 16, 32, 4, 128
    groups = HQ // H
    block_size = 64
    is_causal = True
    sm_scale = 1.0 / math.sqrt(D)
    G = HQ // H
    block_M = 64
    mask_last_token = True
    
    # 测量参数
    warmup_steps = 10
    measure_steps = 50

    print(f"Benchmark Config: B={B}, HQ={HQ}, H={H}, D={D}, block_size={block_size}")

    for L in [4096, 8192]:
        print(f"\n{'='*20} L={L} {'='*20}")
        num_kv_blocks = L // block_size
        
        # 1. 构造基础输入
        Q = torch.randn((B, L, HQ, D), dtype=dtype, device=device, requires_grad=True)
        K = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)
        V = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)
        grad_output = torch.randn((B, L, HQ, D), dtype=dtype, device=device)

        # 2. 构造 Dense 权重: [B, L, HQ, num_kv_blocks]
        logits = torch.randn((B, L, HQ, num_kv_blocks), dtype=torch.bfloat16, device=device)

        # 3. Causal Mask for Weights
        q_indices = torch.arange(L, device=device).view(1, L, 1, 1)
        k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)
        q_blk_indices = q_indices // block_size
        weight_mask = k_blk_indices < q_blk_indices

        logits = logits.masked_fill(~weight_mask, -1e9)
        W = F.softmax(logits, dim=-1).requires_grad_(True)

        # 4. 定义测试闭包
        def run_hsa_fwd():
            return HSA_block_M_head_dense(Q, K, V, W, block_size=block_size, sm_scale=sm_scale, block_M=block_M, mask_last_token=mask_last_token)

        def run_fa3_fwd():
            return FA3_ref_program(Q, K, V, is_causal, groups)

        # 辅助测量函数
        def measure_time(func, steps, desc):
            # 必须同步等待之前的操作完成
            torch.cuda.synchronize()
            
            # 使用 CUDA Events 计时更精准
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(steps):
                func()
            end_event.record()
            
            # 等待所有 CUDA 核心完成
            torch.cuda.synchronize()
            
            # 返回平均耗时 (ms)
            return start_event.elapsed_time(end_event) / steps

        # 分别测量 HSA 和 FA3
        hsa_fwd_ms, hsa_bwd_ms, hsa_total_ms = 0, 0, 0
        fa3_fwd_ms, fa3_bwd_ms, fa3_total_ms = 0, 0, 0
        
        for name, run_fwd_func in [("HSA", run_hsa_fwd), ("FA3", run_fa3_fwd)]:
            print(f"\n--- {name} ---")
            
            # 预热 (使用完整的前向+后向)
            for _ in range(warmup_steps):
                out = run_fwd_func()
                out.backward(grad_output, retain_graph=True)
                # 清空梯度
                if name == "HSA":
                    Q.grad = K.grad = V.grad = W.grad = None
                else:
                    Q.grad = K.grad = V.grad = None
            
            # 测量前向时间 (不带梯度)
            def fwd_only():
                with torch.no_grad():
                    run_fwd_func()
            
            fwd_ms = measure_time(fwd_only, measure_steps, f"{name} FWD")
            
            # 测量完整的前向+后向时间
            def fwd_bwd():
                # 清空梯度
                if name == "HSA":
                    Q.grad = K.grad = V.grad = W.grad = None
                else:
                    Q.grad = K.grad = V.grad = None
                
                out = run_fwd_func()
                out.backward(grad_output, retain_graph=True)
            
            total_ms = measure_time(fwd_bwd, measure_steps, f"{name} FULL")
            
            # 计算后向时间
            bwd_ms = total_ms - fwd_ms
            
            print(f"FWD: {fwd_ms:.3f}ms | BWD: {bwd_ms:.3f}ms | TOTAL: {total_ms:.3f}ms")
            
            # 保存结果用于比较
            if name == "HSA":
                hsa_fwd_ms, hsa_bwd_ms, hsa_total_ms = fwd_ms, bwd_ms, total_ms
            else:
                fa3_fwd_ms, fa3_bwd_ms, fa3_total_ms = fwd_ms, bwd_ms, total_ms
        
        # 计算和打印速度比
        print(f"\n--- Speedup Comparison ---")
        print(f"FWD Speedup: {fa3_fwd_ms/hsa_fwd_ms:.2f}x")
        print(f"BWD Speedup: {fa3_bwd_ms/hsa_bwd_ms:.2f}x")
        print(f"TOTAL Speedup: {fa3_total_ms/hsa_total_ms:.2f}x")

        # 每个 L 结束后清理显存
        torch.cuda.empty_cache()
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))



if __name__ == "__main__":
    # test_topk_hsa_unified_correctness()
    # benchmark_hsa_vs_fa3_with_overlap()
    benchmark_hsa_dense_vs_fa3()
    