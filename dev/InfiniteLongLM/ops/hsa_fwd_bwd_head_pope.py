import torch
import math
import logging
logging.getLogger("tilelang.jit.kernel").setLevel(logging.WARNING)
logging.getLogger("tilelang").setLevel(logging.WARNING)
import tilelang
from tilelang import language as T


from einops import rearrange

try:
    from .hsa_fwd_bwd_head import HSA_block_M_head as _base_HSA_block_M_head
except ImportError:
    from hsa_fwd_bwd_head import HSA_block_M_head as _base_HSA_block_M_head


_TORCH_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}

def _torch_dtype_to_str(td: torch.dtype) -> str:
    if td not in _TORCH_DTYPE_TO_STR:
        raise ValueError(f"Unsupported torch dtype for tilelang kernel: {td}")
    return _TORCH_DTYPE_TO_STR[td]


# ----------------------------------------------------------------------------
# PoPE K-side cos/sin cache.
#
# theta_k = freqs[0:L_kv, :R] depends only on (freqs tensor, L_kv, R). It is
# independent of batch, bias and q_offset, so cos(theta_k)/sin(theta_k) can be
# reused across layers / decoding steps that share the same freqs.
#
# Cache key: (id(freqs), data_ptr(freqs), L_kv, R, out_dtype, device).
#   - id(freqs)  : detect Python-level tensor object reuse.
#   - data_ptr   : guard against in-place mutation of freqs storage.
# Cache value: (cos_k, sin_k) each of shape [L_kv, R] in `out_dtype` on `device`.
#
# Use _clear_pope_kfreq_cache() to drop the cache (e.g. when tests rebuild freqs).
# ----------------------------------------------------------------------------
_POPE_KFREQ_CACHE = {}
_POPE_KFREQ_CACHE_MAX = 8  # keep tiny; entries are O(L_kv * R) and rarely vary


def _get_pope_kfreq_cossin(freqs: torch.Tensor, L_kv: int, R: int,
                           out_dtype: torch.dtype, device: torch.device):
    key = (id(freqs), int(freqs.data_ptr()), int(L_kv), int(R),
           out_dtype, str(device))
    cached = _POPE_KFREQ_CACHE.get(key)
    if cached is not None:
        return cached
    # Compute in fp32 for accuracy, then cast down once.
    freqs_f = freqs.float() if freqs.dtype != torch.float32 else freqs
    theta_k = freqs_f.narrow(0, 0, L_kv)[:, :R]  # view, no copy
    cos_k = torch.cos(theta_k).to(out_dtype).contiguous()
    sin_k = torch.sin(theta_k).to(out_dtype).contiguous()
    if device != cos_k.device:
        cos_k = cos_k.to(device)
        sin_k = sin_k.to(device)
    if len(_POPE_KFREQ_CACHE) >= _POPE_KFREQ_CACHE_MAX:
        # Drop an arbitrary old entry (FIFO-ish).
        _POPE_KFREQ_CACHE.pop(next(iter(_POPE_KFREQ_CACHE)))
    _POPE_KFREQ_CACHE[key] = (cos_k, sin_k)
    return cos_k, sin_k


def _clear_pope_kfreq_cache():
    _POPE_KFREQ_CACHE.clear()

def hsa_torch_ref(q, k, v, weights, indices, *, chunk_size: int, sm_scale: float, block_q: int, mask_last_token: bool = True,
                  freqs=None, bias=None, q_offset=None):

    import torch.nn.functional as F
    B, q_len, HQ, D = q.shape
    _, kv_len, H, _ = k.shape  # ✅ 分别获取 q_len 和 kv_len
    if q_offset is None:
        q_offset = kv_len - q_len
    G = HQ // H
    q_blocks = q_len // block_q  # ✅ 基于 q 的长度
    device = q.device
    use_pope = (freqs is not None and bias is not None)

    if indices.shape[1] != q_blocks:
        idx_view = indices.view(B, q_blocks, block_q, H, -1)
        indices_q = idx_view[:, :, 0, :, :].contiguous()
    else:
        indices_q = indices

    valid_mask = (indices_q >= 0)  # (B, q_blocks, H, K)
    safe_indices = indices_q.clamp_min(0)

    # ✅ 使用 kv_len 计算 KV 的 chunk 数量
    N = kv_len // chunk_size
    valid_kv_len = N * chunk_size
    
    # ✅ 截断 k 和 v（基于 kv_len）
    k_truncated = k[:, :valid_kv_len, :, :]
    v_truncated = v[:, :valid_kv_len, :, :]
    
    k_chunks = rearrange(k_truncated, 'B (N S) h d -> B N S h d', S=chunk_size)
    v_chunks = rearrange(v_truncated, 'B (N S) h d -> B N S h d', S=chunk_size)

    # 后续 gather 逻辑不变，indices 指向 KV 的 chunk 索引，这是正确的
    idx_flat = rearrange(safe_indices, 'B Bq h K -> B (Bq K) h').unsqueeze(2).unsqueeze(-1)
    idx_flat = idx_flat.expand(-1, -1, chunk_size, -1, D)
    idx_flat = idx_flat.long()  
    gather_k = k_chunks.gather(dim=1, index=idx_flat)
    gather_v = v_chunks.gather(dim=1, index=idx_flat)

    gather_k = rearrange(gather_k, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)
    gather_v = rearrange(gather_v, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)

    k_ = torch.repeat_interleave(gather_k, dim=-2, repeats=G)
    v_ = torch.repeat_interleave(gather_v, dim=-2, repeats=G)

    q_chunked = rearrange(q, 'B (Bq X) hq d -> B Bq X hq d', X=block_q)

    if use_pope:
        # PoPE 旋转: Q 侧 (bias applied on Q with minus sign)
        # q_chunked: (B, Bq, X, HQ, D)
        R = freqs.shape[-1]  # rotate_dim，可能 < D
        q_positions = torch.arange(q_len, device=device) + q_offset  # [q_len]
        fq = freqs[q_positions]  # [q_len, R]
        
        # Auto-reshape bias: [h_q, R] -> [h_kv, G, R] -> [HQ, R]
        if bias.dim() == 2:
            bias_hq = bias.view(H, G, R)  # [H, G, R]
        else:
            bias_hq = bias  # already [H, G, R]
        # bias_hq: [H, G, R] -> [HQ, R] for per-query-head
        bias_flat = bias_hq.reshape(HQ, R)  # [HQ, R]
        
        # theta_q = fq - bias (minus sign for equivalence)
        fq_chunked = fq.view(q_blocks, block_q, 1, R)  # [Bq, X, 1, R]
        bias_q = bias_flat.view(1, 1, HQ, R)  # [1, 1, HQ, R]
        theta_q_chunked = fq_chunked - bias_q  # [Bq, X, HQ, R]
        
        # 前 R 维做 PoPE 旋转
        aq = q_chunked[..., :R].float()  # [B, Bq, X, HQ, R]
        qc = aq * torch.cos(theta_q_chunked).unsqueeze(0)  # [B, Bq, X, HQ, R]
        qs = aq * torch.sin(theta_q_chunked).unsqueeze(0)
        
        # PoPE 旋转: K 侧 (pure positional frequency, no bias)
        safe_indices_expanded = safe_indices.unsqueeze(2).expand(-1, -1, chunk_size, -1, -1)  # (B, Bq, S, H, K)
        token_offsets = torch.arange(chunk_size, device=device).view(1, 1, chunk_size, 1, 1)  # (1, 1, S, 1, 1)
        k_positions = safe_indices_expanded * chunk_size + token_offsets  # (B, Bq, S, H, K) — 全局 token 位置
        
        # K 侧: 纯位置频率，不含 bias
        k_pos_flat = k_positions.reshape(-1).long().clamp(0, freqs.shape[0] - 1)
        fk = freqs[k_pos_flat].reshape(B, q_blocks, chunk_size, H, -1, R)  # (B, Bq, S, H, K, R)
        
        # 扩展到 HQ 维度
        theta_k = torch.repeat_interleave(fk, dim=3, repeats=G)  # (B, Bq, S, HQ, K, R)
        theta_k = theta_k.permute(0, 1, 2, 4, 3, 5)  # (B, Bq, S, K, HQ, R) — 与 k_ 对齐
        
        # 前 R 维做 PoPE 旋转
        bk = k_[..., :R].float()  # (B, Bq, S, K, HQ, R)
        kc = bk * torch.cos(theta_k)
        ks = bk * torch.sin(theta_k)
        
        # sim = <Qc, Kc> + <Qs, Ks> (旋转部分)
        qk = (torch.einsum('b q x h r, b q s k h r -> b q x s k h', qc, kc)
            + torch.einsum('b q x h r, b q s k h r -> b q x s k h', qs, ks))
        
        # 后 D-R 维做普通内积
        if R < D:
            q_rest = q_chunked[..., R:].float()  # [B, Bq, X, HQ, D-R]
            k_rest = k_[..., R:].float()          # [B, Bq, S, K, HQ, D-R]
            qk = qk + torch.einsum('b q x h d, b q s k h d -> b q x s k h', q_rest, k_rest)
        
        qk = qk * float(sm_scale)
    else:
        # 非 PoPE 路径（兼容旧的非 PoPE 测试）
        qk = torch.einsum('b q x h d, b q s k h d -> b q x s k h', q_chunked.float(), k_.float())
        qk = qk * float(sm_scale)
    
    if mask_last_token:
        qk[:, :, :, -1, :, :] = float("-inf")

    p = torch.softmax(qk, dim=3)

    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    w_masked = weights.clone()
    valid_mask_expanded = torch.repeat_interleave(valid_mask, dim=-2, repeats=G)
    w_masked = w_masked.masked_fill(~valid_mask_expanded, 0)
    w_exp = w_masked.float()
    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'b q x h d -> b (q x) h d')
    return o_ref.to(torch.float32)




# def make_dq_layout_hsa(dQ):

#     NV, B, L, HQ, D = dQ.shape
#     return T.Layout(dQ.shape,
#     lambda nv, b, l, h, d:   [nv,b,l, h//8, d//16, (d%16)//2, (h%8), (d%2)]
#  )

def make_dq_layout_hsa(DQ):
    # DQ.shape = [NV, batch, q_len, heads, head_dim]
    G = 8  # group size = tile size，完美匹配！
    return T.Layout(DQ.shape,
        lambda nv, b, l, h, d: [
            nv,
            b,
            l,
            h // 8,                           # tile 在 h 维的索引
            d // 8,                           # tile 在 d 维的索引
            d % 2,                            # elem_id (每个线程 2 个元素)
            4 * (h % 8) + (d % 8) // 2        # lane_id
        ])

@tilelang.jit(
    out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def hsa_bwd_postprocess(nv, batch, q_len, heads, head_dim, dtype="bfloat16", accum_dtype="float"):
    shape = [nv, batch, q_len, heads, head_dim]
    blk = 64

    @T.prim_func
    def hsa_post(
            dQ_swizzled: T.Tensor(shape, accum_dtype),
            dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(q_len, blk), heads, batch * nv, threads=32) as (bx, by, bz):
            i_nv = bz // batch
            i_b = bz % batch
            
            T.annotate_layout({dQ_swizzled: make_dq_layout_hsa(dQ_swizzled)})
            
            T.copy(
                dQ_swizzled[i_nv, i_b, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[i_nv, i_b, bx * blk:(bx + 1) * blk, by, :],
            )
    return hsa_post



@tilelang.jit(
    pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hsa_kernel_block_mask(batch, heads, q_len, kv_len, selected_blocks, block_size, dtype="int32"):
    from tilelang import language as T

    S = selected_blocks
    NS_kv = kv_len // block_size

    block_indices_shape = [batch, q_len, heads, selected_blocks]
    block_mask_shape = [batch, q_len, heads, NS_kv]

    @T.prim_func
    def build_block_mask(
        BlockIndices: T.Tensor(block_indices_shape, dtype),
        BlockMask: T.Tensor(block_mask_shape, dtype),
    ):
        with T.Kernel(q_len, batch, heads * S, threads=32) as (i_t, i_b, i_hs):
            i_h = i_hs // S
            i_s = i_hs % S

            block_idx = BlockIndices[i_b, i_t, i_h, i_s]

            if block_idx >= 0 and block_idx < NS_kv:
                BlockMask[i_b, i_t, i_h, block_idx] = i_s

    return build_block_mask




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
    kv_len: int = None,
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

    # num_blocks = SEQ_LEN // block_size
    if kv_len is None:
        num_blocks = SEQ_LEN // block_size
    else:
        num_blocks = kv_len // block_size
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





# @tilelang.jit(
#     out_idx=[5],
#     pass_configs={
#         tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
#         tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
#         tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
#     }
# )
# def hierarchical_sparse_attention_block_M(batch, heads,  head_dim, q_len=None, kv_len=None,
#                                           scale=None, block_size=64, groups=16,
#                                           selected_blocks=16, num_weights=None, block_M = None, mask_last_token=True, dtype = "bfloat16", accum_dtype = "float", num_threads = None,
#                                           is_training=True,
#                                           freqs_len=None, q_offset_val=0, rotate_dim=None):

#     enable_last_token_mask = False
#     if mask_last_token:
#         enable_last_token_mask = True
#     if scale is None:
#         scale = (1.0 / head_dim)**0.5 * 1.44269504
#     else:
#         scale = scale * 1.44269504
#     # 允许 num_weights (输入的 weights 张量的最后一维) 大于 selected_blocks (K)
#     if num_weights is None:
#         num_weights = selected_blocks
#     head_kv = heads // groups
#     head_kv = heads // groups
#     if not is_training:
#         q_len = T.dynamic("q_len")
#         kv_len = T.dynamic("kv_len")
#     q_shape = [batch, q_len, heads, head_dim]
#     kv_shape = [batch, kv_len, head_kv, head_dim]
#     weight_shape = [batch, q_len, heads, num_weights]
#     block_indices_shape = [batch, q_len, head_kv, selected_blocks]
#     block_indices_dtype = "int32"
#     # PoPE shapes
#     if freqs_len is None:
#         freqs_len = kv_len
#     if rotate_dim is None:
#         rotate_dim = head_dim
#     freqs_shape = [freqs_len, rotate_dim]
#     bias_shape = [head_kv, groups, rotate_dim]
#     # dtype = "bfloat16"
#     # accum_dtype = "float"
#     block_S = block_size
#     block_T = min(128, tilelang.math.next_power_of_2(head_dim))

#     NV = tilelang.cdiv(head_dim, block_T)
#     assert tilelang.cdiv(head_dim, block_T) == 1, "The key dimension can not be larger than 256"

#     # M: 每个线程块处理的query 数量，目标是让 M * G >= 16
#     MIN_GEMM_ROWS = 16
#     M_min = tilelang.cdiv(MIN_GEMM_ROWS, groups)
#     if block_M is None or block_M <= 0:
#         M = M_min
#     else:
#         M = max(block_M, M_min)
#     # M=4
#     print("Using M =", M, "for fwd_block_M kernel")
#     M_G = M * groups  # 一次 GEMM 处理的总 head 数

#     S = selected_blocks
#     BS = block_S
#     BK = BV = block_T
#     num_stages = 0

#     if num_threads is None:
#         num_threads = 128

#     @T.prim_func
#     def hsa_block_M(
#             Q: T.Tensor(q_shape, dtype),
#             K: T.Tensor(kv_shape, dtype),
#             V: T.Tensor(kv_shape, dtype),
#             W: T.Tensor[weight_shape, dtype],
#             BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
#             Output: T.Tensor(q_shape, dtype),
#             Freqs: T.Tensor(freqs_shape, accum_dtype),
#             Bias: T.Tensor(bias_shape, accum_dtype),
#     ):
#         with T.Kernel(tilelang.cdiv(q_len, M), NV, batch * head_kv, threads=num_threads) as (bx, by, bz):
#             # PoPE: 需要 cos/sin 两份 Q 和 K
#             Q_shared = T.alloc_shared([M_G, BK], dtype)
#             K_raw_shared = T.alloc_shared([BS, BK], dtype)
#             Qc_shared = T.alloc_shared([M_G, BK], dtype)
#             Qs_shared = T.alloc_shared([M_G, BK], dtype)
#             Kc_shared = T.alloc_shared([BS, BK], dtype)
#             Ks_shared = T.alloc_shared([BS, BK], dtype)
#             V_shared = T.alloc_shared([BS, BV], dtype)
#             O_shared = T.alloc_shared([M_G, BV], dtype)

#             acc_s = T.alloc_fragment([M_G, BS], accum_dtype)
#             acc_s_cast = T.alloc_fragment([M_G, BS], dtype)
#             acc_o = T.alloc_fragment([M_G, BV], accum_dtype)
            
#             P_shared = T.alloc_shared([M_G, BS], dtype)

#             scores_max = T.alloc_fragment([M_G], accum_dtype)
#             scores_sum = T.alloc_fragment([M_G], accum_dtype)

#             merged_indices = T.alloc_shared([S * M], block_indices_dtype)
#             block_ownership = T.alloc_shared([S * M], "int32") # Bitmask
#             merged_len = T.alloc_shared([1], "int32")
#             chunk_weights = T.alloc_shared([S * M, M_G], dtype)

#             i_t_base_idx, i_v, i_bh = bx, by, bz
#             i_b, i_h = i_bh // head_kv, i_bh % head_kv
#             base_t = i_t_base_idx * M

#             # Q 侧 PoPE 旋转: aq = softplus(q), Qc = aq*cos(fq), Qs = aq*sin(fq)
#             # Step 1: T.copy 加载原始 Q 到 shared memory
#             T.fill(Q_shared, 0)
#             T.fill(Qc_shared, 0)
#             T.fill(Qs_shared, 0)
#             for q_idx in T.serial(M):
#                 tq = base_t + q_idx
#                 if tq < q_len:
#                     T.copy(Q[i_b, tq, i_h * groups:(i_h + 1) * groups, :],
#                            Q_shared[q_idx * groups:(q_idx + 1) * groups, :])
#             # Step 2: 从 shared memory 计算 PoPE
#             for i, d in T.Parallel(M_G, BK):
#                 q_idx = i // groups
#                 g = i % groups
#                 tq = base_t + q_idx
#                 if tq < q_len:
#                     qx = T.Cast(accum_dtype, Q_shared[i, d])
#                     if d < rotate_dim:
#                         aq = qx
#                         theta_q = Freqs[q_offset_val + tq, d] - Bias[i_h, g, d]
#                         Qc_shared[i, d] = T.Cast(dtype, aq * T.cos(theta_q))
#                         Qs_shared[i, d] = T.Cast(dtype, aq * T.sin(theta_q))
#                     else:
#                         # 非旋转维度：Qc 存原始值，Qs 置零
#                         Qc_shared[i, d] = Q_shared[i, d]
#                         Qs_shared[i, d] = T.Cast(dtype, 0.0)

#             T.fill(acc_o, 0)
#             T.fill(merged_indices, -1)
#             T.fill(block_ownership, 0)
#             T.fill(chunk_weights, 0)
#             T.fill(merged_len, 0)
            
            
#             W_local_shared = T.alloc_shared([M_G, S], dtype)
#             T.fill(W_local_shared, 0.0)
#             # 并行加载所有 Head 的权重
#             for i, s_idx in T.Parallel(M_G, S):
#                 q_idx = i // groups
#                 g = i % groups
#                 tq = base_t + q_idx
#                 if tq < q_len:
#                     # W 的 shape 是 [batch, q_len, heads, selected_blocks]
#                     # 当前 head index = i_h * groups + g
#                     W_local_shared[i, s_idx] = W[i_b, tq, i_h * groups + g, s_idx]

#             if T.get_thread_binding() == 0:
#                 valid_lens = T.alloc_fragment([M], "int32")
#                 pointers = T.alloc_fragment([M], "int32")
#                 k = T.alloc_var("int32")
#                 cur_val = T.alloc_fragment([M], "int32")
                
#                 # 记录每个query的有效block数量
#                 for q_idx in T.Parallel(M):
#                     tq = base_t + q_idx
#                     valid_lens[q_idx] = 0
#                     pointers[q_idx] = 0
#                     if tq < q_len:
#                         for j in T.serial(S):
#                             if BlockIndices[i_b, tq, i_h, j] >= 0:
#                                 valid_lens[q_idx] = valid_lens[q_idx] + 1
#                             else:
#                                 T.loop_break()
                    
#                 # M 路归并
#                 k = 0
#                 ownership_mask = T.alloc_var("int32")
#                 for _ in T.serial(S * M):
#                     min_val = T.alloc_var("int32")
#                     min_val = 2147483647  # INT_MAX
#                     has_valid = T.alloc_var("int32")
#                     has_valid = 0
                    
#                     # 找到所有query当前还没处理的最小的块索引
#                     for q_idx in T.serial(M):
#                         tq = base_t + q_idx
#                         if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
#                                 has_valid = 1
#                                 val_q = BlockIndices[i_b, tq, i_h, pointers[q_idx]]
#                                 cur_val[q_idx] = val_q
#                                 if val_q < min_val:
#                                     min_val = val_q
#                         else:
#                             cur_val[q_idx] = 2147483647

#                     if has_valid == 0:
#                         T.loop_break()

#                     merged_indices[k] = min_val
#                     ownership_mask = 0
#                     # for q_idx in T.serial(M):
#                     for q_idx in T.unroll(M):
#                         tq = base_t + q_idx
#                         if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
#                             # pointers[q_idx] 此时就是 W 在 S 维的索引
#                             s_idx = pointers[q_idx]
#                             val_q = cur_val[q_idx]
#                             if val_q == min_val:
#                                 ownership_mask = ownership_mask | (1 << q_idx)
#                                 # [修改] 记录合并后第k个块对应于第q_idx个query下所有head的权重
#                                 # for g in T.serial(groups):
#                                 for g in T.unroll(groups):
#                                     chunk_weights[k, q_idx * groups + g] = W_local_shared[q_idx * groups + g, s_idx]
#                                 pointers[q_idx] = pointers[q_idx] + 1
#                     # 记录合并后的第k个块属于哪些query，用bitmask表示：0b0101表示属于第0和第2个query
#                     block_ownership[k] = ownership_mask
#                     k = k + 1
                    
#                 # 记录归并后的总块数
#                 merged_len[0] = k

#             T.sync_threads()

#             merged_len_local = T.alloc_var("int32")
#             merged_len_local = merged_len[0]
#             h_start = T.alloc_var("int32")

#             for i in T.Pipelined(merged_len_local, num_stages=num_stages):
#                 blk_idx = merged_indices[i]
#                 i_s = blk_idx * BS
#                 ownership = block_ownership[i]

#                 if (blk_idx >= 0):
#                     # K 侧 PoPE 旋转: bk = softplus(k), theta_k = freqs[pos] (pure positional, no bias)
#                     # Step 1: T.copy 加载原始 K 到 shared memory
#                     T.copy(K[i_b, i_s:i_s + BS, i_h, :], K_raw_shared)
#                     # Step 2: 从 shared memory 计算 PoPE
#                     T.fill(Kc_shared, 0)
#                     T.fill(Ks_shared, 0)
#                     for s_idx, d in T.Parallel(BS, BK):
#                         k_pos = i_s + s_idx
#                         if k_pos < kv_len:
#                             kx = T.Cast(accum_dtype, K_raw_shared[s_idx, d])
#                             if d < rotate_dim:
#                                 bk = kx
#                                 theta_k = Freqs[k_pos, d]
#                                 Kc_shared[s_idx, d] = T.Cast(dtype, bk * T.cos(theta_k))
#                                 Ks_shared[s_idx, d] = T.Cast(dtype, bk * T.sin(theta_k))
#                             else:
#                                 # 非旋转维度：Kc 存原始值，Ks 置零
#                                 Kc_shared[s_idx, d] = K_raw_shared[s_idx, d]
#                                 Ks_shared[s_idx, d] = T.Cast(dtype, 0.0)
                    
#                     # 两次 GEMM: sim = Qc @ Kc^T + Qs @ Ks^T
#                     T.clear(acc_s)
#                     T.gemm(Qc_shared, Kc_shared, acc_s, transpose_B=True,
#                            policy=T.GemmWarpPolicy.FullRow)
#                     T.gemm(Qs_shared, Ks_shared, acc_s, transpose_B=True,
#                            policy=T.GemmWarpPolicy.FullRow)
                    
#                     # 不再使用 T.serial(M)，而是直接并行处理所有 M*G 行
#                     for r, c in T.Parallel(M_G, BS):
#                         q_idx = r // groups
#                         # # 检查当前行对应的 Query 是否拥有这个 Block
#                         acc_s[r, c] = T.if_then_else(  
#                                                     ((ownership & (1 << q_idx)) == 0) or ((c == BS - 1) and enable_last_token_mask),  
#                                                     -T.infinity(accum_dtype),  
#                                                     acc_s[r, c]  
#                                                 )

#                     # 4. Compute Softmax (Max -> Exp -> Sum -> Div)
#                     # 4.1 Reduce Max
#                     T.fill(scores_max, -T.infinity(accum_dtype))
#                     T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    
#                     # 4.2 Exp & Apply Scale
#                     for r, c in T.Parallel(M_G, BS):
#                         q_idx = r // groups
#                         # # [修复] 只有有效的 Query 才进行 exp 计算，避免 -inf - (-inf) 产生 NaN
#                         acc_s[r, c] = T.if_then_else(
#                                                     (ownership & (1 << q_idx)) != 0,
#                                                     T.exp2(acc_s[r, c] * scale - scores_max[r] * scale),
#                                                     0.0
#                                                 )
                    
#                     # 4.3 Reduce Sum
#                     T.fill(scores_sum, 0.0)
#                     T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)
                    
#                     # 4.4 Normalize & Apply Weights
#                     for r, c in T.Parallel(M_G, BS):
#                         q_idx = r // groups
#                         # # [修复] 同样只对有效行进行归一化
#                         acc_s[r, c] = T.if_then_else(
#                                                     (ownership & (1 << q_idx)) != 0,
#                                                     acc_s[r, c] * chunk_weights[i, r] / scores_sum[r],
#                                                     0.0
#                                                 )

#                     # 5. Store P to Shared Memory (仅一次写操作)
#                     # P_shared 用于下一次 GEMM (P * V)
#                     T.copy(acc_s, P_shared)

#                     # 6. Load V & Compute O
#                     T.copy(V[i_b, i_s:i_s + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
#                     T.gemm(P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

#             T.copy(acc_o, O_shared)
            
#             for q_idx in T.serial(M):
#                 tq = base_t + q_idx
#                 if tq < q_len:
#                     h_start = q_idx * groups
#                     for g, v in T.Parallel(groups, BV):
#                         Output[i_b, tq, i_h * groups + g, i_v * BV + v] = O_shared[h_start + g, v]

#     return hsa_block_M







# @tilelang.jit(
#     pass_configs={
#         tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
#         tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
#         tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
#     }
# )
# def hierarchical_sparse_attention_bwd_dqkv_block_M(
#     batch, heads, q_len, kv_len, head_dim,
#     scale=None, block_size=64, groups=16, selected_blocks=16, num_weights=None,
#     block_M = None, mask_last_token=True, dtype="bfloat16", accum_dtype="float", num_threads = None,
#     freqs_len=None, q_offset_val=0, rotate_dim=None,
# ):

#     enable_last_token_mask = False
#     if mask_last_token:
#         enable_last_token_mask = True
        
#     if scale is None:
#         sm_scale = (1.0 / head_dim)**0.5
#     else:
#         sm_scale = scale
#     scale_log2 = sm_scale * 1.44269504

#     B = batch
#     BS = block_size
#     G = groups
#     Vdim = head_dim
#     Kdim = head_dim
#     BK = tilelang.next_power_of_2(Kdim)
#     BV = min(128, tilelang.next_power_of_2(head_dim))
#     # NS_kv = tilelang.cdiv(kv_len, BS)
#     NS_kv = kv_len // BS
#     NV = tilelang.cdiv(Vdim, BV)
#     S = selected_blocks
#     # 允许 num_weights (输入的 weights 张量的最后一维) 大于 selected_blocks (K)
#     if num_weights is None:
#         num_weights = S

#     heads_kv = heads // groups
#     q_shape = [batch, q_len, heads, head_dim]
#     k_shape = [batch, kv_len, heads_kv, head_dim]
#     v_shape = [batch, kv_len, heads_kv, head_dim]
#     do_shape = [batch, q_len, heads, head_dim]

#     dq_shape = [NV, batch, q_len, heads, head_dim]
#     dk_shape = [NV, batch, kv_len, heads_kv, head_dim]
#     dv_shape = [batch, kv_len, heads_kv, head_dim]
#     # weight_shape = [batch, q_len, heads, selected_blocks]
#     weight_shape = [batch, q_len, heads, num_weights]
#     # dw_shape = [batch, q_len, heads, selected_blocks]
#     dw_shape = [batch, q_len, heads, num_weights]
#     block_mask_shape = [batch, q_len, heads_kv, NS_kv]

#     # PoPE shapes
#     if freqs_len is None:
#         freqs_len = kv_len
#     if rotate_dim is None:
#         rotate_dim = head_dim
#     freqs_shape = [freqs_len, rotate_dim]
#     bias_shape = [heads_kv, G, rotate_dim]
#     dbias_shape = [heads_kv, G, rotate_dim]

#     MIN_GEMM_ROWS = 16
#     M_min = tilelang.cdiv(MIN_GEMM_ROWS, G)
#     if block_M is None or block_M <= 0:
#         M = M_min
#     else:
#         M = max(block_M, M_min)   
#     # M=8  
#     print("Using M =", M, "for bwd_block_M kernel")
#     M_G = M * G
#     NP = tilelang.cdiv(q_len, M)

#     # num_threads = 256
#     if num_threads is None:
#         num_threads = 256
#     num_stages = 0

#     @T.prim_func
#     def hsa_bwd_dqkv_block_M(
#         Q: T.Tensor(q_shape, dtype),
#         K: T.Tensor(k_shape, dtype),
#         V: T.Tensor(v_shape, dtype),
#         W: T.Tensor(weight_shape, dtype),
#         DO: T.Tensor(do_shape, dtype),
#         DQ: T.Tensor(dq_shape, accum_dtype),
#         DK: T.Tensor(dk_shape, dtype),
#         DV: T.Tensor(dv_shape, dtype),
#         DW: T.Tensor(dw_shape, dtype),
#         BlockMask: T.Tensor(block_mask_shape, "int32"),
#         Freqs: T.Tensor(freqs_shape, accum_dtype),
#         Bias: T.Tensor(bias_shape, accum_dtype),
#         DBias: T.Tensor(dbias_shape, accum_dtype),
#     ):
#         with T.Kernel(NV, NS_kv, B * heads_kv, threads=num_threads) as (i_v, i_s, i_bh):
#             i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
#             i_s_global = i_s * BS

#             Qc_shared = T.alloc_shared([M_G, BK], dtype)
#             Qs_shared = T.alloc_shared([M_G, BK], dtype)
#             Kc_shared = T.alloc_shared([BS, BK], dtype)
#             Ks_shared = T.alloc_shared([BS, BK], dtype)
#             Q_raw_shared = T.alloc_shared([M_G, BK], dtype)
#             K_raw_shared = T.alloc_shared([BS, BK], dtype)

#             S_buf = T.alloc_shared([M_G, BS], dtype)
#             dO_buf = T.alloc_shared([M_G, BV], dtype)
            
#             logits_shared = S_buf
#             P_shared = S_buf
#             dS_shared = S_buf
            
#             dO_shared = dO_buf
#             dO_weighted_shared = dO_buf

#             V_shared = T.alloc_shared([BS, BV], dtype)
            
#             dK_shared = T.alloc_shared([BS, BK], dtype)
#             dV_shared = T.alloc_shared([BS, BV], dtype)

#             qk_frag = T.alloc_fragment([M_G, BS], accum_dtype)
#             dV_PdO_frag = T.alloc_fragment([M_G, BS], accum_dtype)
#             dS_frag = T.alloc_fragment([M_G, BS], accum_dtype)
#             dV_accum = T.alloc_fragment([BS, BV], accum_dtype)
#             dK_accum = T.alloc_fragment([BS, BK], accum_dtype)
#             dQ_local = T.alloc_fragment([M_G, BK], accum_dtype)
#             delta_rows = T.alloc_fragment([M_G], accum_dtype)
            
#             acc_s_tmp = T.alloc_fragment([M_G, BS], accum_dtype)
#             scores_max = T.alloc_fragment([M_G], accum_dtype)
#             scores_sum = T.alloc_fragment([M_G], accum_dtype)
            
#             has_q_shared = T.alloc_shared([M], "int32")

#             dw_row_sum_frag = T.alloc_fragment([M_G], accum_dtype)
#             dw_row_sum_shared = T.alloc_shared([M_G], accum_dtype)
            
#             W_local = T.alloc_shared([M_G], dtype)
            
#             pos = T.alloc_shared([M], "int32")
#             has_q = T.alloc_shared([M], "int32")
#             any_valid = T.alloc_var("int32")

#             dqkc_shared = T.alloc_shared([M_G, BK], dtype)
#             dqks_shared = T.alloc_shared([M_G, BK], dtype)
#             dkkc_frag = T.alloc_fragment([BS, BK], accum_dtype)
#             dkks_frag = T.alloc_fragment([BS, BK], accum_dtype)
#             dqkc_frag = T.alloc_fragment([M_G, BK], accum_dtype)
#             dqks_frag = T.alloc_fragment([M_G, BK], accum_dtype)
            
#             T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :], K_raw_shared)

#             T.fill(Kc_shared, 0)
#             T.fill(Ks_shared, 0)
#             for s_idx, d in T.Parallel(BS, BK):
#                 k_pos = i_s_global + s_idx
#                 if k_pos < kv_len:
#                     kx = T.Cast(accum_dtype, K_raw_shared[s_idx, d])
#                     if d < rotate_dim:
#                         bk = kx
#                         theta_k = Freqs[k_pos, d]
#                         Kc_shared[s_idx, d] = T.Cast(dtype, bk * T.cos(theta_k))
#                         Ks_shared[s_idx, d] = T.Cast(dtype, bk * T.sin(theta_k))
#                     else:
#                         Kc_shared[s_idx, d] = K_raw_shared[s_idx, d]
#                         Ks_shared[s_idx, d] = T.Cast(dtype, 0.0)

#             T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
#             T.fill(dK_accum, 0)
#             T.fill(dV_accum, 0)

#             T.annotate_layout({
#                 DQ: make_dq_layout_hsa(DQ),
#             })

#             for ip in T.Pipelined(NP, num_stages=num_stages):
#                 base_t = ip * M

#                 T.fill(pos, -1)
#                 T.fill(has_q, 0)
                
#                 any_valid = 0
#                 for qi1 in T.serial(M):
#                     tq = base_t + qi1
#                     if tq < q_len:
#                         pos[qi1] = BlockMask[i_b, tq, i_h, i_s]
#                         if pos[qi1] != -1:
#                             has_q[qi1] = 1
#                             any_valid = 1
                
                
#                 if any_valid != 0:
#                     T.fill(Q_raw_shared, 0)
#                     T.fill(Qc_shared, 0)
#                     T.fill(Qs_shared, 0)
#                     T.fill(dO_shared, 0)
#                     for qi3 in T.serial(M):
#                         tq = base_t + qi3
#                         if tq < q_len and has_q[qi3] == 1:
#                             h_start = qi3 * G
#                             T.copy(Q[i_b, tq, i_h * G:(i_h + 1) * G, :],
#                                 Q_raw_shared[h_start:h_start + G, :])
#                             T.copy(DO[i_b, tq, i_h * G:(i_h + 1) * G, i_v * BV:(i_v + 1) * BV],
#                                 dO_shared[h_start:h_start + G, :])

#                     for i, d in T.Parallel(M_G, BK):
#                         q_idx = i // G
#                         g = i % G
#                         tq = base_t + q_idx
#                         if tq < q_len and has_q[q_idx] == 1:
#                             qx = T.Cast(accum_dtype, Q_raw_shared[i, d])
#                             if d < rotate_dim:
#                                 aq = qx
#                                 theta_q = Freqs[q_offset_val + tq, d] - Bias[i_h, g, d]
#                                 Qc_shared[i, d] = T.Cast(dtype, aq * T.cos(theta_q))
#                                 Qs_shared[i, d] = T.Cast(dtype, aq * T.sin(theta_q))
#                             else:
#                                 Qc_shared[i, d] = Q_raw_shared[i, d]
#                                 Qs_shared[i, d] = T.Cast(dtype, 0.0)

#                     T.clear(qk_frag)
#                     T.gemm(Qc_shared, Kc_shared, qk_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
#                     T.gemm(Qs_shared, Ks_shared, qk_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

#                     T.copy(qk_frag, logits_shared)
                    
#                     # 1. Load data from Shared Memory to Fragment
#                     for i, s in T.Parallel(M_G, BS):
#                         acc_s_tmp[i, s] = logits_shared[i, s]

#                     # 2. Apply Masking using IF statements (avoids T.if_then_else expression bug)
#                     for i, s in T.Parallel(M_G, BS):
#                         q_idx = i // G
                        
#                         # Use control flow to overwrite with -inf
#                         if has_q[q_idx] == 0:
#                             acc_s_tmp[i, s] = -T.infinity(accum_dtype)
                            
#                         # Last token masking
#                         if enable_last_token_mask:
#                             if s == BS - 1:
#                                 acc_s_tmp[i, s] = -T.infinity(accum_dtype)

#                     # 3. Reduction Max
#                     T.fill(scores_max, -T.infinity(accum_dtype))
#                     T.reduce_max(acc_s_tmp, scores_max, dim=1, clear=True)

#                     # 4. Exp with NaN Guard
#                     for i, s in T.Parallel(M_G, BS):
#                         q_idx = i // G
#                         acc_s_tmp[i, s] = T.if_then_else(
#                             has_q[q_idx] == 1 and scores_max[i] > -T.infinity(accum_dtype),
#                             T.exp2(acc_s_tmp[i, s] * scale_log2 - scores_max[i] * scale_log2),
#                             0.0
#                         )

                    
#                     # 5. Sum
#                     T.reduce_sum(acc_s_tmp, scores_sum, dim=1, clear=True)

#                     # 6. Normalize and Write Back
#                     for i, s in T.Parallel(M_G, BS):
#                         # inv_sum = T.if_then_else(scores_sum[i] > 0, 1.0 / scores_sum[i], 0.0)
#                         # P_shared[i, s] = acc_s_tmp[i, s] * inv_sum
#                         acc_s_tmp[i, s] = T.if_then_else(scores_sum[i] > 0,
#                                                         acc_s_tmp[i, s] / scores_sum[i],
#                                                         0.0)


#                     # [Step 1] Load W early
#                     T.fill(W_local, 0.0)
#                     for i in T.Parallel(M_G):
#                         qi_w = i // G
#                         g_local = i % G
#                         tq = base_t + qi_w
#                         if tq < q_len and has_q[qi_w] == 1 and pos[qi_w] != -1:
#                             W_local[i] = W[i_b, tq, i_h * G + g_local, pos[qi_w]]

#                     # [Step 2] Compute Z = dO @ V.T
#                     # Store in dV_PdO_frag (accum_dtype)
#                     T.clear(dV_PdO_frag)
#                     T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

#                     # [Step 3] Compute dW = Sum(P * Z)
#                     # Reuse qk_frag (accum_dtype) for P * Z
#                     for r, s in T.Parallel(M_G, BS):
#                         qk_frag[r, s] = acc_s_tmp[r, s] * dV_PdO_frag[r, s]
                    
#                     T.reduce_sum(qk_frag, dw_row_sum_frag, dim=1, clear=True)
#                     T.copy(dw_row_sum_frag, dw_row_sum_shared)
                    
#                     # Write back dW
#                     for i in T.Parallel(M * G):  
#                         qi5 = i // G  
#                         g_w = i % G  
#                         if has_q[qi5] == 1:
#                             h_start = qi5 * G  
#                             tq = base_t + qi5  
#                             if tq < q_len and pos[qi5] != -1:  
#                                 DW[i_b, tq, i_h * G + g_w, pos[qi5]] = dw_row_sum_shared[h_start + g_w]

#                     # [Step 4] Compute dV
#                     # Need dO_weighted = W * dO
#                     for row_idx, v in T.Parallel(M_G, BV):
#                         dO_weighted_shared[row_idx, v] = W_local[row_idx] * dO_shared[row_idx, v]
#                     T.copy(acc_s_tmp, P_shared)
#                     T.gemm(P_shared, dO_weighted_shared, dV_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

#                     # [Step 5] Compute dS
#                     # We need Y = W * Z = W * (dO @ V.T)
#                     # Z is currently in dV_PdO_frag
#                     for g_row, s in T.Parallel(M_G, BS):
#                         dV_PdO_frag[g_row, s] = dV_PdO_frag[g_row, s] * W_local[g_row]
                    
#                     # Now dV_PdO_frag holds Y.
#                     # Compute P * Y for delta calculation. Reuse qk_frag again.
#                     for g_row, s in T.Parallel(M_G, BS):
#                         qk_frag[g_row, s] = P_shared[g_row, s] * dV_PdO_frag[g_row, s]
                        
#                     T.reduce_sum(qk_frag, delta_rows, dim=1, clear=True)
                    
#                     for g_row, s in T.Parallel(M_G, BS):
#                         # dS = scale * (P * Y - P * delta)
#                         #    = scale * (qk_frag - P * delta)
#                         dS_frag[g_row, s] = sm_scale * (qk_frag[g_row, s] - P_shared[g_row, s] * delta_rows[g_row])
                    
#                     T.copy(dS_frag, dS_shared)
                    
#                     # [PoPE dQ] dqkc = dS @ Kc, dqks = dS @ Ks
#                     T.clear(dqkc_frag)
#                     T.gemm(dS_shared, Kc_shared, dqkc_frag, policy=T.GemmWarpPolicy.FullRow)
#                     T.clear(dqks_frag)
#                     T.gemm(dS_shared, Ks_shared, dqks_frag, policy=T.GemmWarpPolicy.FullRow)
                    
#                     # dQ = (dqkc*cos(θq) + dqks*sin(θq)) * sigmoid(q_raw)
#                     # dBias: dθq/d(bias) = -1, so dBias += -(dqks*Qc - dqkc*Qs)
#                     T.copy(has_q, has_q_shared)
#                     for g_row, d in T.Parallel(M_G, BK):
#                         q_idx = g_row // G
#                         g_local = g_row % G
#                         tq = base_t + q_idx
#                         if tq < q_len and has_q_shared[q_idx] == 1:
#                             if d < rotate_dim:
#                                 theta_q = Freqs[q_offset_val + tq, d] - Bias[i_h, g_local, d]
#                                 cos_tq = T.cos(theta_q)
#                                 sin_tq = T.sin(theta_q)
                                
#                                 dq_val = (dqkc_frag[g_row, d] * cos_tq + dqks_frag[g_row, d] * sin_tq)
#                                 T.atomic_add(DQ[i_v, i_b, tq, i_h * G + g_local, d], dq_val)
                                
#                                 # dBias: d(theta_q)/d(bias) = -1
#                                 # d(score)/d(theta_q) = dqks * Qc - dqkc * Qs (chain rule on cos/sin)
#                                 dtheta_q_val = dqks_frag[g_row, d] * T.Cast(accum_dtype, Qc_shared[g_row, d]) - dqkc_frag[g_row, d] * T.Cast(accum_dtype, Qs_shared[g_row, d])
#                                 T.atomic_add(DBias[i_h, g_local, d], -dtheta_q_val)
#                             else:
#                                 T.atomic_add(DQ[i_v, i_b, tq, i_h * G + g_local, d], dqkc_frag[g_row, d])

#                     # [PoPE dK] dkkc = dS^T @ Qc, dkks = dS^T @ Qs
#                     T.clear(dkkc_frag)
#                     T.gemm(dS_shared, Qc_shared, dkkc_frag, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
#                     T.clear(dkks_frag)
#                     T.gemm(dS_shared, Qs_shared, dkks_frag, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    
#                     # dK = (dkkc*cos(fk) + dkks*sin(fk)) * sigmoid(k_raw)
#                     # K side: no bias, so theta_k = Freqs[k_pos, d]
#                     for s_idx, d in T.Parallel(BS, BK):
#                         k_pos = i_s_global + s_idx
#                         if k_pos < kv_len:
#                             if d < rotate_dim:
#                                 theta_k = Freqs[k_pos, d]
#                                 cos_tk = T.cos(theta_k)
#                                 sin_tk = T.sin(theta_k)
                                
#                                 dk_val = (dkkc_frag[s_idx, d] * cos_tk + dkks_frag[s_idx, d] * sin_tk)
#                                 dK_accum[s_idx, d] = dK_accum[s_idx, d] + dk_val
#                             else:
#                                 dK_accum[s_idx, d] = dK_accum[s_idx, d] + dkkc_frag[s_idx, d]

#             T.copy(dK_accum, dK_shared)
#             T.copy(dV_accum, dV_shared)
#             T.copy(dK_shared, DK[i_v, i_b, i_s_global:i_s_global + BS, i_h, :])
#             T.copy(dV_shared, DV[i_b, i_s_global:i_s_global + BS, i_h, i_v * BV:(i_v + 1) * BV])

#     return hsa_bwd_dqkv_block_M




# from liger_kernel.transformers.rope import liger_rotary_pos_emb
# from ops.rope_tilelang_fp32 import rope_rotary_pos_emb
# class _hsa_block_M_attention(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx, q, k, v, weights, indices,
#         block_size, sm_scale, block_M, mask_last_token,dtype, accum_dtype, num_threads,
#         enable_inverse_rope,
#         cos, sin, is_training,
#         freqs, bias, q_offset_val,
#     ):
#         assert q.is_contiguous()
#         assert k.is_contiguous()
#         assert v.is_contiguous()
#         assert weights.is_contiguous()
#         assert indices.is_contiguous()
#         assert freqs is not None, "freqs is required for PoPE kernel"
#         assert bias is not None, "bias is required for PoPE kernel"

#         B, L, HQ, D = q.shape
#         L_kv = k.shape[1]
#         H = k.shape[2]
#         S = indices.shape[-1]
#         G = HQ // H
#         num_weights = weights.shape[-1]
#         rotate_dim = freqs.shape[-1]

#         if enable_inverse_rope:
#             assert cos.shape[1] == q.shape[1], f"cos seq_len {cos.shape[1]} != q seq_len {q.shape[1]}"
#             q_in, k_in = rope_rotary_pos_emb(q, k, cos, -sin)
#         else:
#             q_in, k_in = q, k

#         assert HQ % H == 0, f"HQ={HQ} must be divisible by H={H}"

#         if sm_scale is None:
#             sm_scale = 1.0 / math.sqrt(D)

#         # Auto-derive kernel dtype from q's torch dtype, overriding the
#         # incoming `dtype` arg if it disagrees with q's actual dtype.
#         dtype = _torch_dtype_to_str(q_in.dtype)

#         fwd_kernel = hierarchical_sparse_attention_block_M(
#             batch=B,
#             heads=HQ,
#             q_len=L,
#             kv_len=L_kv,
#             head_dim=D,
#             block_size=block_size,
#             groups=G,
#             selected_blocks=S,
#             num_weights=num_weights,
#             scale=sm_scale,
#             block_M=block_M,
#             mask_last_token=mask_last_token,
#             dtype=dtype, accum_dtype=accum_dtype,
#             num_threads=num_threads,
#             is_training=is_training,
#             freqs_len=freqs.shape[0],
#             q_offset_val=q_offset_val,
#             rotate_dim=rotate_dim,
#         )

#         freqs_f = freqs.contiguous().to(torch.float32)
#         # bias: [h_q, R] or [h_kv, G, R] -> reshape to [h_kv, G, R] for kernel
#         if bias.dim() == 2:
#             bias_3d = bias.view(H, G, rotate_dim)
#         else:
#             bias_3d = bias
#         bias_f = bias_3d.contiguous().to(torch.float32)
#         o = fwd_kernel(q_in, k_in, v, weights.to(q_in.dtype), indices, freqs_f, bias_f)

#         ctx.save_for_backward(q, k, v, weights, indices, cos, sin, freqs, bias)
#         ctx.enable_inverse_rope = bool(enable_inverse_rope)
#         ctx.block_size = block_size
#         ctx.sm_scale = sm_scale
#         ctx.block_M = block_M
#         ctx.mask_last_token = mask_last_token
#         ctx.dtype = dtype
#         ctx.accum_dtype = accum_dtype
#         ctx.num_threads = num_threads
#         ctx.B = B
#         ctx.L = L
#         ctx.HQ = HQ
#         ctx.H = H
#         ctx.D = D
#         ctx.S = S
#         ctx.G = G
#         ctx.num_weights = num_weights
#         ctx.q_offset_val = q_offset_val
#         ctx.rotate_dim = rotate_dim

#         return o

#     @staticmethod
#     def backward(ctx, do):
#         q, k, v, weights, indices, cos, sin, freqs, bias = ctx.saved_tensors
#         enable_inverse_rope = ctx.enable_inverse_rope
#         block_size = ctx.block_size
#         sm_scale = ctx.sm_scale
#         block_M = ctx.block_M
#         mask_last_token = ctx.mask_last_token
#         dtype = ctx.dtype
#         accum_dtype = ctx.accum_dtype
#         num_threads = ctx.num_threads
#         B, L, HQ, H, D, S, G = ctx.B, ctx.L, ctx.HQ, ctx.H, ctx.D, ctx.S, ctx.G
#         num_weights = ctx.num_weights
#         q_offset_val = ctx.q_offset_val
#         rotate_dim = ctx.rotate_dim

#         if enable_inverse_rope:
#             q_in, k_in = rope_rotary_pos_emb(q, k, cos, -sin)
#         else:
#             q_in, k_in = q, k

#         L_kv = k.shape[1]
#         NS_kv = L_kv // block_size
#         NV = tilelang.cdiv(D, min(128, tilelang.next_power_of_2(D)))

#         build_mask = hsa_kernel_block_mask(
#             batch=B, heads=H, q_len=L, kv_len=L_kv,
#             selected_blocks=S, block_size=block_size
#         )
#         block_mask = torch.full((B, L, H, NS_kv), -1, dtype=torch.int32, device=q.device)
#         build_mask(indices, block_mask)
        
#         freqs_f = freqs.contiguous().to(torch.float32)
#         # bias: [h_q, R] or [h_kv, G, R] -> reshape to [h_kv, G, R] for kernel
#         if bias.dim() == 2:
#             bias_3d = bias.view(H, G, rotate_dim)
#         else:
#             bias_3d = bias
#         bias_f = bias_3d.contiguous().to(torch.float32)

#         bwd_kernel = hierarchical_sparse_attention_bwd_dqkv_block_M(
#             batch=B,
#             heads=HQ,
#             q_len=L,
#             kv_len=L_kv,
#             head_dim=D,
#             block_size=block_size,
#             groups=G,
#             selected_blocks=S,
#             num_weights=num_weights,
#             scale=sm_scale,
#             block_M=block_M,
#             mask_last_token=mask_last_token,
#             dtype=dtype, accum_dtype=accum_dtype, num_threads=num_threads,
#             freqs_len=freqs.shape[0],
#             q_offset_val=q_offset_val,
#             rotate_dim=rotate_dim,
#         )

#         # 分配梯度缓冲
#         dq_in = torch.zeros((NV, B, L, HQ, D), dtype=torch.float32, device=q.device)
#         dk_in = torch.zeros((NV, B, L_kv, H, D), dtype=k.dtype, device=k.device)
#         dv = torch.zeros((B, L_kv, H, D), dtype=v.dtype, device=v.device)
#         dw = torch.zeros((B, L, HQ, num_weights), dtype=weights.dtype, device=weights.device)
#         dbias = torch.zeros_like(bias_f)    # (H_kv, G, R), float32

#         do=do.contiguous()
#         bwd_kernel(q_in, k_in, v, weights, do, dq_in, dk_in, dv, dw, block_mask,
#                    freqs_f, bias_f, dbias)

#         post_kernel = hsa_bwd_postprocess(NV, B, L, HQ, D, dtype=dtype, accum_dtype=accum_dtype)
#         dq_in = post_kernel(dq_in)

#         dq_in = dq_in.sum(0)
#         dk_in = dk_in.sum(0)

#         dq_in = dq_in.to(q.dtype)
#         dk_in = dk_in.to(k.dtype)
#         dv = dv.to(v.dtype)
#         dw = dw.to(weights.dtype)
#         # dbias is [H, G, R]; reshape back to match bias's original shape
#         dbias = dbias.reshape(bias.shape).to(bias.dtype)

#         if enable_inverse_rope:
#             dq, dk = rope_rotary_pos_emb(dq_in, dk_in, cos, sin)
#         else:
#             dq, dk = dq_in, dk_in

#         return dq, dk, dv, dw, None, None, None, None, None, None, None, None, None, None, None, None, None, dbias, None


# def HSA_block_M_head_pope(
#     q, k, v, weights, indices,
#     block_size=64, sm_scale=None, block_M=None, mask_last_token=True, dtype="bfloat16", accum_dtype="float", num_threads=None,
#     enable_inverse_rope: bool = False,
#     cos=None, sin=None, is_training=True,
#     freqs=None, bias=None, q_offset=None,
# ):
#     """
#     Args:
#         is_training: bool - True for training mode, False for inference mode (dynamic shape support).
#         freqs: (L_total, rotate_dim) - PoPE position-angle table 
#         bias: (h_q, rotate_dim) - per-query-head learnable bias
#         q_offset: int - Q side position offset (default 0)
    
#     """
#     if enable_inverse_rope and (cos is None or sin is None):
#         raise ValueError("cos and sin cannot be None when enable_inverse_rope is True")
#     if freqs is None or bias is None:
#         raise ValueError("freqs and bias are required for PoPE kernel (hsa_fwd_bwd_head_pope)")
#     if q_offset is None:
#         q_offset = k.shape[1] - q.shape[1]
#     # Kernel declares Bias/Freqs as accum_dtype (fp32). Caller's bias may follow
#     # the model dtype (e.g. bf16 Parameter), so force-cast here.
#     freqs = freqs.float()
#     bias = bias.float()
#     return _hsa_block_M_attention.apply(
#         q, k, v, weights, indices,
#         block_size, sm_scale, block_M, mask_last_token, dtype, accum_dtype, num_threads,
#         enable_inverse_rope,
#         cos, sin, is_training,
#         freqs, bias, q_offset,
#     )






@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_block_M_reuseIndices(
        batch, heads, head_dim, q_len=None, kv_len=None,
        scale=None, block_size=64, groups=16,
        selected_blocks=16, num_weights=None, block_M=None,
        mask_last_token=True, dtype="bfloat16", accum_dtype="float",
        num_threads=None, is_training=True,
        freqs_len=None, q_offset_val=0, rotate_dim=None):

    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504
    else:
        scale = scale * 1.44269504

    if num_weights is None:
        num_weights = selected_blocks
    head_kv = heads // groups
    if not is_training:
        q_len = T.dynamic("q_len")
        kv_len = T.dynamic("kv_len")
    q_shape = [batch, q_len, heads, head_dim]
    kv_shape = [batch, kv_len, head_kv, head_dim]
    weight_shape = [batch, q_len, heads, num_weights]
    block_indices_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    if freqs_len is None:
        freqs_len = kv_len
    if rotate_dim is None:
        rotate_dim = head_dim
    freqs_shape = [freqs_len, rotate_dim]
    bias_shape = [head_kv, groups, rotate_dim]
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(head_dim))

    assert tilelang.cdiv(head_dim, block_T) == 1, "The key dimension can not be larger than 256"

    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, groups)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)
    print(f"Using M = {M} for fwd_block_M_reuseIndices kernel, "
          f"batch={batch}, heads={heads}, head_kv={head_kv}, "
          f"q_len={q_len}, kv_len={kv_len}, head_dim={head_dim}")
    print("num_threads:", num_threads)
    M_G = M * groups

    S = selected_blocks
    BS = block_S
    BK = BV = block_T
    num_stages = 1

    if num_threads is None:
        num_threads = 128

    NP = tilelang.cdiv(q_len, M)
    N_BH = batch * head_kv
    scores_lse_shape = [batch, q_len, heads, selected_blocks]
    fwd_merged_indices_shape = [N_BH, NP, S * M]
    fwd_block_ownership_shape = [N_BH, NP, S * M]
    fwd_chunk_weights_shape = [N_BH, NP, S * M, M_G]
    fwd_merged_s_indices_shape = [N_BH, NP, S * M, M]
    fwd_merged_len_shape = [N_BH, NP]

    @T.prim_func
    def hsa_block_M_reuseIndices(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
            FwdMergedIndices: T.Tensor(fwd_merged_indices_shape, "int32"),
            FwdBlockOwnership: T.Tensor(fwd_block_ownership_shape, "int32"),
            FwdChunkWeights: T.Tensor(fwd_chunk_weights_shape, dtype),
            FwdMergedSIndices: T.Tensor(fwd_merged_s_indices_shape, "int32"),
            FwdMergedLen: T.Tensor(fwd_merged_len_shape, "int32"),
            Freqs: T.Tensor(freqs_shape, accum_dtype),
            Bias: T.Tensor(bias_shape, accum_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(tilelang.cdiv(q_len, M), batch * head_kv, threads=num_threads) as (bx, bz):
            Q_shared = T.alloc_shared([M_G, BK], dtype)
            K_raw_shared = T.alloc_shared([BS, BK], dtype)
            Qc_shared = T.alloc_shared([M_G, BK], dtype)
            Qs_shared = T.alloc_shared([M_G, rotate_dim], dtype)
            Kc_shared = T.alloc_shared([BS, BK], dtype)
            Ks_shared = T.alloc_shared([BS, rotate_dim], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([M_G, BV], dtype)

            acc_s = T.alloc_fragment([M_G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([M_G, BS], dtype)
            acc_o = T.alloc_fragment([M_G, BV], accum_dtype)

            P_shared = T.alloc_shared([M_G, BS], dtype)

            acc_s_tmp = T.alloc_fragment([groups, BS], accum_dtype)
            scores_max = T.alloc_fragment([M_G], accum_dtype)
            scores_sum = T.alloc_fragment([M_G], accum_dtype)

            if is_training:
                lse_local = T.alloc_fragment([M_G], accum_dtype)

            merged_indices = T.alloc_shared([S * M], block_indices_dtype)
            block_ownership = T.alloc_shared([S * M], "int32")
            merged_len = T.alloc_shared([1], "int32")



            chunk_weights_row = T.alloc_shared([M_G], dtype)
            merged_s_indices = T.alloc_shared([S * M, M], "int32")

            i_t_base_idx, i_bh = bx, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv
            base_t = i_t_base_idx * M

            T.fill(Q_shared, 0)
            T.fill(Qc_shared, 0)
            T.fill(Qs_shared, 0)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    T.copy(Q[i_b, tq, i_h * groups:(i_h + 1) * groups, :],
                           Q_shared[q_idx * groups:(q_idx + 1) * groups, :])
            for i_mg, d in T.Parallel(M_G, BK):
                q_idx = i_mg // groups
                tq = base_t + q_idx
                if tq < q_len:
                    Qc_shared[i_mg, d] = Q_shared[i_mg, d]

            for i_mg, d in T.Parallel(M_G, rotate_dim):
                q_idx = i_mg // groups
                g = i_mg % groups
                tq = base_t + q_idx
                if tq < q_len:
                    qx = T.Cast(accum_dtype, Q_shared[i_mg, d])
                    theta_q = Freqs[q_offset_val + tq, d] - Bias[i_h, g, d]
                    Qc_shared[i_mg, d] = T.Cast(dtype, qx * T.cos(theta_q))
                    Qs_shared[i_mg, d] = T.Cast(dtype, qx * T.sin(theta_q))

            T.fill(acc_o, 0)
            T.fill(merged_indices, -1)
            T.fill(block_ownership, 0)
            T.fill(merged_s_indices, -1)
            T.fill(merged_len, 0)

            W_local_shared = T.alloc_shared([M_G, S], dtype)
            T.fill(W_local_shared, 0.0)
            for i_mg, s_idx in T.Parallel(M_G, S):
                q_idx = i_mg // groups
                g = i_mg % groups
                tq = base_t + q_idx
                if tq < q_len:
                    W_local_shared[i_mg, s_idx] = W[i_b, tq, i_h * groups + g, s_idx]

            if T.get_thread_binding() == 0:
                valid_lens = T.alloc_fragment([M], "int32")
                pointers = T.alloc_fragment([M], "int32")
                k = T.alloc_var("int32")
                cur_val = T.alloc_fragment([M], "int32")

                for q_idx in T.Parallel(M):
                    tq = base_t + q_idx
                    valid_lens[q_idx] = 0
                    pointers[q_idx] = 0
                    if tq < q_len:
                        for j in T.serial(S):
                            if BlockIndices[i_b, tq, i_h, j] >= 0:
                                valid_lens[q_idx] = valid_lens[q_idx] + 1
                            else:
                                T.loop_break()

                k = 0
                ownership_mask = T.alloc_var("int32")
                for _ in T.serial(S * M):
                    min_val = T.alloc_var("int32")
                    min_val = 2147483647
                    has_valid = T.alloc_var("int32")
                    has_valid = 0

                    for q_idx in T.serial(M):
                        tq = base_t + q_idx
                        if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
                                has_valid = 1
                                val_q = BlockIndices[i_b, tq, i_h, pointers[q_idx]]
                                cur_val[q_idx] = val_q
                                if val_q < min_val:
                                    min_val = val_q
                        else:
                            cur_val[q_idx] = 2147483647

                    if has_valid == 0:
                        T.loop_break()

                    merged_indices[k] = min_val
                    ownership_mask = 0
                    for q_idx in T.unroll(M):
                        tq = base_t + q_idx
                        if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
                            s_idx = pointers[q_idx]
                            val_q = cur_val[q_idx]
                            if val_q == min_val:
                                ownership_mask = ownership_mask | (1 << q_idx)
                                merged_s_indices[k, q_idx] = s_idx
                                pointers[q_idx] = pointers[q_idx] + 1
                    block_ownership[k] = ownership_mask
                    k = k + 1

                merged_len[0] = k

            T.sync_threads()

            if is_training:
                T.copy(merged_indices, FwdMergedIndices[i_bh, i_t_base_idx, :])
                T.copy(block_ownership, FwdBlockOwnership[i_bh, i_t_base_idx, :])
                T.copy(merged_s_indices, FwdMergedSIndices[i_bh, i_t_base_idx, :, :])
                FwdMergedLen[i_bh, i_t_base_idx] = merged_len[0]

            merged_len_local = T.alloc_var("int32")
            merged_len_local = merged_len[0]
            h_start = T.alloc_var("int32")

            for i in T.Pipelined(merged_len_local, num_stages=num_stages):
                blk_idx = merged_indices[i]
                i_s = blk_idx * BS
                ownership = block_ownership[i]

                if (blk_idx >= 0):


                    for r in T.Parallel(M_G):
                        q_idx_w = r // groups
                        s_idx_w = merged_s_indices[i, q_idx_w]
                        chunk_weights_row[r] = T.if_then_else(
                            s_idx_w >= 0,
                            W_local_shared[r, T.if_then_else(s_idx_w >= 0, s_idx_w, 0)],
                            T.cast(0, dtype),
                        )

                    T.copy(K[i_b, i_s:i_s + BS, i_h, :], K_raw_shared)
                    T.fill(Kc_shared, 0)
                    T.fill(Ks_shared, 0)
                    for s_idx, d in T.Parallel(BS, BK):
                        k_pos = i_s + s_idx
                        if k_pos < kv_len:
                            Kc_shared[s_idx, d] = K_raw_shared[s_idx, d]

                    for s_idx, d in T.Parallel(BS, rotate_dim):
                        k_pos = i_s + s_idx
                        if k_pos < kv_len:
                            kx = T.Cast(accum_dtype, K_raw_shared[s_idx, d])
                            theta_k = Freqs[k_pos, d]
                            Kc_shared[s_idx, d] = T.Cast(dtype, kx * T.cos(theta_k))
                            Ks_shared[s_idx, d] = T.Cast(dtype, kx * T.sin(theta_k))
                    T.clear(acc_s)
                    T.gemm(Qc_shared, Kc_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(Qs_shared, Ks_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)

                    for r, c in T.Parallel(M_G, BS):
                        q_idx = r // groups

                        acc_s[r, c] = T.if_then_else(
                                                    ((ownership & (1 << q_idx)) == 0) or ((c == BS - 1) and enable_last_token_mask),
                                                    -T.infinity(accum_dtype),
                                                    acc_s[r, c]
                                                )

                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for r, c in T.Parallel(M_G, BS):
                        q_idx = r // groups

                        acc_s[r, c] = T.if_then_else(
                                                    (ownership & (1 << q_idx)) != 0,
                                                    T.exp2(acc_s[r, c] * scale - scores_max[r] * scale),
                                                    0.0
                                                )

                    T.fill(scores_sum, 0.0)
                    T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)


                    if is_training:
                        for r_lse in T.Parallel(M_G):
                                lse_local[r_lse] = T.if_then_else(
                                    scores_sum[r_lse] > 0,
                                    scores_max[r_lse] * scale + T.log(scores_sum[r_lse]) * 1.44269504,
                                    -T.infinity(accum_dtype)
                                )
                        for q_idx_lse, g_lse in T.Parallel(M, groups):
                            tq_lse = base_t + q_idx_lse
                            s_idx_lse = merged_s_indices[i, q_idx_lse]
                            if tq_lse < q_len and s_idx_lse >= 0:
                                r_lse = q_idx_lse * groups + g_lse
                                ScoresLSE[i_b, tq_lse, i_h * groups + g_lse, s_idx_lse] = lse_local[r_lse]

                    for r, c in T.Parallel(M_G, BS):
                        q_idx = r // groups

                        acc_s[r, c] = T.if_then_else(
                                                    (ownership & (1 << q_idx)) != 0,
                                                    acc_s[r, c] * T.cast(chunk_weights_row[r], accum_dtype) / scores_sum[r],
                                                    0.0
                                                )

                    T.copy(acc_s, P_shared)

                    T.copy(V[i_b, i_s:i_s + BS, i_h, :], V_shared)
                    T.gemm(P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)

            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * groups
                    for g, v in T.Parallel(groups, BV):
                        Output[i_b, tq, i_h * groups + g, v] = O_shared[h_start + g, v]

    return hsa_block_M_reuseIndices


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_bwd_dqkv_block_M_inverse(
    batch, heads, q_len, kv_len, head_dim,
    scale=None, block_size=64, groups=16, selected_blocks=16, num_weights=None,
    dtype="bfloat16", accum_dtype="float", block_M=0, mask_last_token=True,
    num_threads=None,
    freqs_len=None, q_offset_val=0, rotate_dim=None,
):

    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True

    if scale is None:
        sm_scale = (1.0 / head_dim)**0.5
    else:
        sm_scale = scale
    scale_log2 = sm_scale * 1.44269504

    B = batch
    BS = block_size
    G = groups
    Vdim = head_dim
    Kdim = head_dim
    BK = tilelang.next_power_of_2(Kdim)
    BV = min(128, tilelang.next_power_of_2(head_dim))
    S = selected_blocks

    if num_weights is None:
        num_weights = S

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    do_shape = [batch, q_len, heads, head_dim]
    dq_shape = [batch, q_len, heads, head_dim]
    dk_shape = [batch, kv_len, heads_kv, head_dim]
    dv_shape = [batch, kv_len, heads_kv, head_dim]
    weight_shape = [batch, q_len, heads, num_weights]
    dw_shape = [batch, q_len, heads, num_weights]
    scores_lse_shape = [batch, q_len, heads, selected_blocks]
    if freqs_len is None:
        freqs_len = kv_len
    if rotate_dim is None:
        rotate_dim = head_dim
    freqs_shape = [freqs_len, rotate_dim]
    bias_shape = [heads_kv, G, rotate_dim]
    dbias_shape = [heads_kv, G, rotate_dim]

    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, G)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)
    print("Using M =", M, "for bwd_block_M_inverse_reuseIndices kernel",
          "num_threads:", num_threads)
    M_G = M * G

    if num_threads is None:
        num_threads = 128
    num_stages = 0


    NP_bwd = tilelang.cdiv(q_len, M)
    N_BH = batch * heads_kv
    fwd_merged_indices_shape = [N_BH, NP_bwd, S * M]
    fwd_block_ownership_shape = [N_BH, NP_bwd, S * M]
    fwd_chunk_weights_shape = [N_BH, NP_bwd, S * M, M_G]
    fwd_merged_s_indices_shape = [N_BH, NP_bwd, S * M, M]
    fwd_merged_len_shape = [N_BH, NP_bwd]

    @T.prim_func
    def hsa_bwd_dqkv_block_M_inverse_reuseIndices(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        W: T.Tensor(weight_shape, dtype),
        DO: T.Tensor(do_shape, dtype),
        FwdMergedIndices: T.Tensor(fwd_merged_indices_shape, "int32"),
        FwdBlockOwnership: T.Tensor(fwd_block_ownership_shape, "int32"),
        FwdChunkWeights: T.Tensor(fwd_chunk_weights_shape, dtype),
        FwdMergedSIndices: T.Tensor(fwd_merged_s_indices_shape, "int32"),
        FwdMergedLen: T.Tensor(fwd_merged_len_shape, "int32"),
        ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
        DQ: T.Tensor(dq_shape, accum_dtype),
        DK: T.Tensor(dk_shape, accum_dtype),
        DV: T.Tensor(dv_shape, accum_dtype),
        DW: T.Tensor(dw_shape, dtype),
        Freqs: T.Tensor(freqs_shape, accum_dtype),
        Bias: T.Tensor(bias_shape, accum_dtype),
        DBias: T.Tensor(dbias_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(q_len, M), B * heads_kv, threads=num_threads) as (bx, bz):
            i_t_base_idx, i_bh = bx, bz
            i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
            base_t = i_t_base_idx * M

            Q_raw_shared = T.alloc_shared([M_G, BK], dtype)
            Qc_shared = T.alloc_shared([M_G, BK], dtype)
            Qs_shared = T.alloc_shared([M_G, rotate_dim], dtype)
            Kc_shared = T.alloc_shared([BS, BK], dtype)
            Ks_shared = T.alloc_shared([BS, rotate_dim], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            dO_shared = T.alloc_shared([M_G, BV], dtype)
            dO_weighted_shared = T.alloc_shared([M_G, BV], dtype)
            P_shared = T.alloc_shared([M_G, BS], dtype)
            dQ_shared = T.alloc_shared([M_G, BK], accum_dtype)
            dQ_accum = T.alloc_fragment([M_G, BK], accum_dtype)
            acc_s_tmp = T.alloc_fragment([M_G, BS], accum_dtype)
            dV_PdO_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            dS_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            dqkc_frag = T.alloc_fragment([M_G, BK], accum_dtype)
            dqks_frag = T.alloc_fragment([M_G, rotate_dim], accum_dtype)
            dkkc_frag = T.alloc_fragment([BS, BK], accum_dtype)
            dkks_frag = T.alloc_fragment([BS, rotate_dim], accum_dtype)
            dqks_shared = T.alloc_shared([M_G, rotate_dim], accum_dtype)
            dkks_shared = T.alloc_shared([BS, rotate_dim], accum_dtype)
            delta_rows = T.alloc_fragment([M_G], accum_dtype)
            saved_lse_shared = T.alloc_shared([M_G], accum_dtype)
            dV_accum_local = T.alloc_fragment([BS, BV], accum_dtype)
            dK_accum_local = T.alloc_fragment([BS, BK], accum_dtype)
            di_rows = T.alloc_fragment([M_G], accum_dtype)
            merged_indices = T.alloc_shared([S * M], "int32")
            block_ownership = T.alloc_shared([S * M], "int32")
            W_local_shared = T.alloc_shared([M_G, S], dtype)
            chunk_weights_row = T.alloc_shared([M_G], dtype)
            merged_s_indices = T.alloc_shared([S * M, M], "int32")

            T.fill(Q_raw_shared, 0)
            T.fill(Qc_shared, 0)
            T.fill(Qs_shared, 0)
            T.fill(dO_shared, 0)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * G
                    T.copy(Q[i_b, tq, i_h * G:(i_h + 1) * G, :], Q_raw_shared[h_start:h_start + G, :])
                    T.copy(DO[i_b, tq, i_h * G:(i_h + 1) * G, :], dO_shared[h_start:h_start + G, :])
            for i_mg, d in T.Parallel(M_G, BK):
                q_idx = i_mg // G
                tq = base_t + q_idx
                if tq < q_len:
                    Qc_shared[i_mg, d] = Q_raw_shared[i_mg, d]

            for i_mg, d in T.Parallel(M_G, rotate_dim):
                q_idx = i_mg // G
                g = i_mg % G
                tq = base_t + q_idx
                if tq < q_len:
                    qx = T.Cast(accum_dtype, Q_raw_shared[i_mg, d])
                    theta_q = Freqs[q_offset_val + tq, d] - Bias[i_h, g, d]
                    Qc_shared[i_mg, d] = T.Cast(dtype, qx * T.cos(theta_q))
                    Qs_shared[i_mg, d] = T.Cast(dtype, qx * T.sin(theta_q))

            T.fill(dQ_accum, 0)
            T.fill(saved_lse_shared, 0.0)
            T.copy(FwdMergedIndices[i_bh, i_t_base_idx, :], merged_indices)
            T.copy(FwdBlockOwnership[i_bh, i_t_base_idx, :], block_ownership)
            T.copy(FwdMergedSIndices[i_bh, i_t_base_idx, :, :], merged_s_indices)
            T.fill(W_local_shared, 0.0)
            for i_mg, s_idx in T.Parallel(M_G, S):
                q_idx_w = i_mg // G
                g_w = i_mg % G
                tq_w = base_t + q_idx_w
                if tq_w < q_len:
                    W_local_shared[i_mg, s_idx] = W[i_b, tq_w, i_h * G + g_w, s_idx]

            T.sync_threads()

            merged_len_local = T.alloc_var("int32")
            merged_len_local = FwdMergedLen[i_bh, i_t_base_idx]
            h_start = T.alloc_var("int32")
            s_idx_local = T.alloc_var("int32")

            for i in T.Pipelined(merged_len_local, num_stages=num_stages):
                blk_idx = merged_indices[i]
                i_s_global = blk_idx * BS
                ownership = block_ownership[i]

                if blk_idx >= 0:
                    for r in T.Parallel(M_G):
                        q_idx_w = r // G
                        s_idx_w = merged_s_indices[i, q_idx_w]
                        chunk_weights_row[r] = T.if_then_else(
                            s_idx_w >= 0,
                            W_local_shared[r, T.if_then_else(s_idx_w >= 0, s_idx_w, 0)],
                            T.cast(0, dtype),
                        )

                    T.fill(Kc_shared, 0)
                    T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :], Kc_shared)
                    T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, :], V_shared)
                    T.fill(Ks_shared, 0)
                    for s_idx, d in T.Parallel(BS, rotate_dim):
                        k_pos = i_s_global + s_idx
                        if k_pos < kv_len:
                            kx = T.Cast(accum_dtype, Kc_shared[s_idx, d])
                            theta_k = Freqs[k_pos, d]
                            Kc_shared[s_idx, d] = T.Cast(dtype, kx * T.cos(theta_k))
                            Ks_shared[s_idx, d] = T.Cast(dtype, kx * T.sin(theta_k))

                    T.clear(acc_s_tmp)
                    T.gemm(Qc_shared, Kc_shared, acc_s_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(Qs_shared, Ks_shared, acc_s_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for i_mg in T.Parallel(M_G):
                        qi_lse = i_mg // G
                        g_lse = i_mg % G
                        tq_lse = base_t + qi_lse
                        s_idx_lse = merged_s_indices[i, qi_lse]
                        saved_lse_shared[i_mg] = T.if_then_else(
                            (tq_lse < q_len)
                            and (((ownership >> qi_lse) & 1) == 1)
                            and (s_idx_lse >= 0),
                            ScoresLSE[
                                i_b,
                                T.if_then_else(tq_lse < q_len, tq_lse, 0),
                                i_h * G + g_lse,
                                T.if_then_else(s_idx_lse >= 0, s_idx_lse, 0),
                            ],
                            0.0,
                        )

                    for i_mg, s in T.Parallel(M_G, BS):
                        q_idx = i_mg // G
                        is_owned = (ownership >> q_idx) & 1
                        if enable_last_token_mask:
                            acc_s_tmp[i_mg, s] = T.if_then_else(
                                is_owned == 1 and s != BS - 1,
                                T.exp2(acc_s_tmp[i_mg, s] * scale_log2 - saved_lse_shared[i_mg]),
                                0.0
                            )
                        else:
                            acc_s_tmp[i_mg, s] = T.if_then_else(
                                is_owned == 1,
                                T.exp2(acc_s_tmp[i_mg, s] * scale_log2 - saved_lse_shared[i_mg]),
                                0.0
                            )
                    T.copy(acc_s_tmp, P_shared)

                    T.clear(dV_PdO_frag)
                    T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for g_row, s in T.Parallel(M_G, BS):
                        dS_frag[g_row, s] = P_shared[g_row, s] * dV_PdO_frag[g_row, s]
                    T.reduce_sum(dS_frag, di_rows, dim=1, clear=True)

                    for r in T.Parallel(M_G):
                        q_idx = r // G
                        g_w = r % G
                        tq = base_t + q_idx
                        s_idx_local_w = merged_s_indices[i, q_idx]
                        if (
                            (tq < q_len)
                            and (((ownership >> q_idx) & 1) == 1)
                            and (s_idx_local_w >= 0)
                        ):
                            DW[
                                i_b,
                                tq,
                                i_h * G + g_w,
                                T.if_then_else(s_idx_local_w >= 0, s_idx_local_w, 0),
                            ] = di_rows[r]

                    for g_row, v in T.Parallel(M_G, BV):
                        dO_weighted_shared[g_row, v] = T.cast(chunk_weights_row[g_row], accum_dtype) * dO_shared[g_row, v]

                    T.clear(dV_accum_local)
                    T.gemm(P_shared, dO_weighted_shared, dV_accum_local, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    for g_row, s in T.Parallel(M_G, BS):
                        dS_frag[g_row, s] = sm_scale * T.cast(chunk_weights_row[g_row], accum_dtype) * P_shared[g_row, s] * (dV_PdO_frag[g_row, s] - di_rows[g_row])
                    T.copy(dS_frag, P_shared)

                    T.clear(dqkc_frag)
                    T.gemm(P_shared, Kc_shared, dqkc_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.clear(dqks_frag)
                    T.gemm(P_shared, Ks_shared, dqks_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(dqks_frag, dqks_shared)

                    for g_row, d in T.Parallel(M_G, rotate_dim):
                        q_idx = g_row // G
                        g_local = g_row % G
                        tq = base_t + q_idx
                        if tq < q_len:
                            theta_q = Freqs[q_offset_val + tq, d] - Bias[i_h, g_local, d]
                            cos_tq = T.cos(theta_q)
                            sin_tq = T.sin(theta_q)
                            dQ_accum[g_row, d] = dQ_accum[g_row, d] + dqkc_frag[g_row, d] * cos_tq + dqks_shared[g_row, d] * sin_tq
                            dtheta_q_val = dqks_shared[g_row, d] * T.Cast(accum_dtype, Qc_shared[g_row, d]) - dqkc_frag[g_row, d] * T.Cast(accum_dtype, Qs_shared[g_row, d])
                            T.atomic_add(DBias[i_h, g_local, d], -dtheta_q_val)

                    for g_row, d in T.Parallel(M_G, BK):
                        q_idx = g_row // G
                        tq = base_t + q_idx
                        if tq < q_len and d >= rotate_dim:
                            dQ_accum[g_row, d] = dQ_accum[g_row, d] + dqkc_frag[g_row, d]

                    T.clear(dkkc_frag)
                    T.gemm(P_shared, Qc_shared, dkkc_frag, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.clear(dkks_frag)
                    T.gemm(P_shared, Qs_shared, dkks_frag, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(dkks_frag, dkks_shared)
                    T.clear(dK_accum_local)
                    for s, k_d in T.Parallel(BS, rotate_dim):
                        k_pos = i_s_global + s
                        if k_pos < kv_len:
                            theta_k = Freqs[k_pos, k_d]
                            dK_accum_local[s, k_d] = dkkc_frag[s, k_d] * T.cos(theta_k) + dkks_shared[s, k_d] * T.sin(theta_k)

                    for s, k_d in T.Parallel(BS, BK):
                        k_pos = i_s_global + s
                        if k_pos < kv_len and k_d >= rotate_dim:
                            dK_accum_local[s, k_d] = dkkc_frag[s, k_d]

                    for s, k_d in T.Parallel(BS, BK):
                        T.atomic_add(DK[i_b, i_s_global + s, i_h, k_d], dK_accum_local[s, k_d])
                    for s, v_d in T.Parallel(BS, BV):
                        T.atomic_add(DV[i_b, i_s_global + s, i_h, v_d], dV_accum_local[s, v_d])

            T.copy(dQ_accum, dQ_shared)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * G
                    T.copy(dQ_shared[h_start:h_start + G, :], DQ[i_b, tq, i_h * G:(i_h + 1) * G, :])

    return hsa_bwd_dqkv_block_M_inverse_reuseIndices




class _hsa_block_M_attention_pope_reuse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, weights, indices,
                block_size, sm_scale, block_M,
                mask_last_token, dtype, accum_dtype,
                num_threads_fwd, num_threads_bwd,
                enable_inverse_rope, cos, sin, is_training,
                freqs, bias, q_offset_val):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert weights.is_contiguous() and indices.is_contiguous()
        assert freqs is not None, "freqs is required for PoPE kernel"
        assert bias is not None, "bias is required for PoPE kernel"

        B, L, HQ, D = q.shape
        L_kv = k.shape[1]
        H = k.shape[2]
        S = indices.shape[-1]
        G = HQ // H
        num_weights = weights.shape[-1]
        rotate_dim = freqs.shape[-1]

        assert HQ % H == 0, f"HQ={HQ} must be divisible by H={H}"

        if enable_inverse_rope:
            assert cos is not None and sin is not None
            assert cos.shape[1] == q.shape[1], f"cos seq_len {cos.shape[1]} != q seq_len {q.shape[1]}"
            q_in, k_in = rope_rotary_pos_emb(q, k, cos, -sin)
        else:
            q_in, k_in = q, k

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)
        dtype = _torch_dtype_to_str(q_in.dtype)

        freqs_f = freqs.contiguous().to(torch.float32)
        if bias.dim() == 2:
            bias_3d = bias.view(H, G, rotate_dim)
        else:
            bias_3d = bias
        bias_f = bias_3d.contiguous().to(torch.float32)

        fwd_kernel = hierarchical_sparse_attention_block_M_reuseIndices(
            batch=B, heads=HQ, q_len=L, kv_len=L_kv, head_dim=D,
            block_size=block_size, groups=G, selected_blocks=S,
            num_weights=num_weights, scale=sm_scale,
            block_M=block_M, mask_last_token=mask_last_token,
            dtype=dtype, accum_dtype=accum_dtype,
            num_threads=num_threads_fwd, is_training=is_training,
            freqs_len=freqs.shape[0], q_offset_val=q_offset_val,
            rotate_dim=rotate_dim,
        )

        if not is_training:
            o = fwd_kernel(q_in, k_in, v, weights.to(q_in.dtype), indices,
                           None, None, None, None, None, None, freqs_f, bias_f)
            ctx.is_training = False
            return o

        MIN_GEMM_ROWS = 16
        M_min = (MIN_GEMM_ROWS + G - 1) // G
        M_actual = block_M
        if M_actual is None or M_actual <= 0:
            M_actual = M_min
        else:
            M_actual = max(M_actual, M_min)
        NP = (L + M_actual - 1) // M_actual
        N_BH = B * H
        SM = S * M_actual
        fwd_merged_indices = torch.full((N_BH, NP, SM), -1, dtype=torch.int32, device=q.device)
        fwd_block_ownership = torch.zeros((N_BH, NP, SM), dtype=torch.int32, device=q.device)
        fwd_chunk_weights = None
        fwd_merged_s_indices = torch.full((N_BH, NP, SM, M_actual), -1, dtype=torch.int32, device=q.device)
        fwd_merged_len = torch.zeros((N_BH, NP), dtype=torch.int32, device=q.device)
        scores_lse = torch.full((B, L, HQ, S), float('-inf'), dtype=torch.float32, device=q.device)

        o = fwd_kernel(q_in, k_in, v, weights.to(q_in.dtype), indices, scores_lse,
                       fwd_merged_indices, fwd_block_ownership, fwd_chunk_weights,
                       fwd_merged_s_indices, fwd_merged_len, freqs_f, bias_f)

        ctx.save_for_backward(q, k, v, weights, indices, cos, sin, freqs, bias,
                              scores_lse, fwd_merged_indices, fwd_block_ownership,
                              fwd_merged_s_indices, fwd_merged_len)
        ctx.enable_inverse_rope = bool(enable_inverse_rope)
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.block_M = block_M
        ctx.mask_last_token = mask_last_token
        ctx.dtype = dtype
        ctx.accum_dtype = accum_dtype
        ctx.num_threads_bwd = num_threads_bwd
        ctx.B, ctx.L, ctx.HQ, ctx.H, ctx.D, ctx.S, ctx.G = B, L, HQ, H, D, S, G
        ctx.num_weights = num_weights
        ctx.q_offset_val = q_offset_val
        ctx.rotate_dim = rotate_dim
        ctx.is_training = True
        return o

    @staticmethod
    def backward(ctx, do):
        if not getattr(ctx, "is_training", True):
            raise RuntimeError("PoPE reuse forward ran with is_training=False, so backward buffers were not saved.")
        q, k, v, weights, indices, cos, sin, freqs, bias, scores_lse, \
            fwd_merged_indices, fwd_block_ownership, \
            fwd_merged_s_indices, fwd_merged_len = ctx.saved_tensors
        enable_inverse_rope = ctx.enable_inverse_rope
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        block_M = ctx.block_M
        mask_last_token = ctx.mask_last_token
        dtype = ctx.dtype
        accum_dtype = ctx.accum_dtype
        num_threads_bwd = ctx.num_threads_bwd
        B, L, HQ, H, D, S, G = ctx.B, ctx.L, ctx.HQ, ctx.H, ctx.D, ctx.S, ctx.G
        num_weights = ctx.num_weights
        q_offset_val = ctx.q_offset_val
        rotate_dim = ctx.rotate_dim
        L_kv = k.shape[1]

        if enable_inverse_rope:
            q_in, k_in = rope_rotary_pos_emb(q, k, cos, -sin)
        else:
            q_in, k_in = q, k

        freqs_f = freqs.contiguous().to(torch.float32)
        if bias.dim() == 2:
            bias_3d = bias.view(H, G, rotate_dim)
        else:
            bias_3d = bias
        bias_f = bias_3d.contiguous().to(torch.float32)

        do = do.contiguous()
        bwd_kernel = hierarchical_sparse_attention_bwd_dqkv_block_M_inverse(
            batch=B, heads=HQ, q_len=L, kv_len=L_kv, head_dim=D,
            block_size=block_size, groups=G, selected_blocks=S,
            num_weights=num_weights, scale=sm_scale,
            block_M=block_M, mask_last_token=mask_last_token,
            dtype=dtype, accum_dtype=accum_dtype,
            num_threads=num_threads_bwd,
            freqs_len=freqs.shape[0], q_offset_val=q_offset_val,
            rotate_dim=rotate_dim,
        )

        dq_in = torch.zeros((B, L, HQ, D), dtype=torch.float32, device=q.device)
        dk_in = torch.zeros((B, L_kv, H, D), dtype=torch.float32, device=k.device)
        dv = torch.zeros((B, L_kv, H, D), dtype=torch.float32, device=v.device)
        dw = torch.zeros((B, L, HQ, num_weights), dtype=weights.dtype, device=weights.device)
        dbias = torch.zeros_like(bias_f)

        bwd_kernel(q_in, k_in, v, weights.to(q_in.dtype), do,
                   fwd_merged_indices, fwd_block_ownership, None,
                   fwd_merged_s_indices, fwd_merged_len,
                   scores_lse, dq_in, dk_in, dv, dw,
                   freqs_f, bias_f, dbias)

        dq_in = dq_in.to(q.dtype)
        dk_in = dk_in.to(k.dtype)
        dv = dv.to(v.dtype)
        dw = dw.to(weights.dtype)
        dbias = dbias.reshape(bias.shape).to(bias.dtype)

        if enable_inverse_rope:
            dq, dk = rope_rotary_pos_emb(dq_in, dk_in, cos, sin)
        else:
            dq, dk = dq_in, dk_in

        return dq, dk, dv, dw, None, None, None, None, None, None, None, None, None, None, None, None, None, None, dbias, None


def HSA_block_M_head_pope(
    q, k, v, weights, indices,
    block_size=64, sm_scale=None, block_M=None, mask_last_token=True,
    dtype="bfloat16", accum_dtype="float",
    num_threads=None, num_threads_fwd=None, num_threads_bwd=None,
    enable_inverse_rope: bool = False,
    cos=None, sin=None, is_training=True,
    freqs=None, bias=None, q_offset=None,
):
    if enable_inverse_rope and (cos is None or sin is None):
        raise ValueError("cos and sin cannot be None when enable_inverse_rope is True")
    if freqs is None or bias is None:
        raise ValueError("freqs and bias are required for PoPE kernel (hsa_fwd_bwd_head_pope)")
    if q_offset is None:
        q_offset = k.shape[1] - q.shape[1]
    if num_threads_fwd is None:
        num_threads_fwd = num_threads
    if num_threads_bwd is None:
        num_threads_bwd = num_threads
    freqs = freqs.float()
    bias = bias.float()
    return _hsa_block_M_attention_pope_reuse.apply(
        q, k, v, weights, indices,
        block_size, sm_scale, block_M,
        mask_last_token, dtype, accum_dtype,
        num_threads_fwd, num_threads_bwd,
        enable_inverse_rope, cos, sin, is_training,
        freqs, bias, q_offset,
    )


# Empirically tuned (block_M, num_threads_fwd, num_threads_bwd) per group size G
# for HSA_block_M_head_pope_prerotate. Only used when the corresponding kwarg is
# not explicitly provided by the caller. Format: G -> (block_M, fwd_threads, bwd_threads).
_HSA_PREROTATE_TUNED_BY_G = {
    16: (4, 128, 128),
    8:  (8, 128, 128),
    4:  (8, 64, 128),
    2:  (16, 128, 128),
    1:  (16, 128, 128),
}


def HSA_block_M_head_pope_prerotate(
    q, k, v, weights, indices,
    block_size=64, sm_scale=None, block_M=None, mask_last_token=True,
    dtype="bfloat16", accum_dtype="float",
    num_threads=None, num_threads_fwd=None, num_threads_bwd=None,
    is_training=True,
    freqs=None, bias=None, q_offset=None,
):
    if freqs is None or bias is None:
        raise ValueError("freqs and bias are required for PoPE prerotate wrapper")

    B, L, HQ, D = q.shape
    _, L_kv, H, Dk = k.shape
    assert D == Dk, f"Q head_dim ({D}) must equal K head_dim ({Dk})"
    assert HQ % H == 0, f"HQ={HQ} must be divisible by H={H}"
    G = HQ // H
    R = freqs.shape[-1]
    assert 0 < R <= D, f"rotate_dim={R} must be in (0, D={D}]"
    # Base HSA kernel uses block_T_k = min(256, npo2(head_dim)) and asserts
    # cdiv(head_dim, block_T_k) == 1, so the effective head_dim (D + R) must be <= 256.
    assert D + R <= 256, (
        f"D + rotate_dim ({D + R}) must be <= 256 for base HSA kernel "
        f"(use smaller D or R, or extend the base kernel to support larger head_dim)"
    )

    # Apply tuned defaults only when caller did not specify.
    tuned = _HSA_PREROTATE_TUNED_BY_G.get(G)
    if tuned is not None:
        tuned_M, tuned_fwd_t, tuned_bwd_t = tuned
        applied = []
        if block_M is None:
            block_M = tuned_M
            applied.append(f"block_M={tuned_M}")
        if num_threads_fwd is None and num_threads is None:
            num_threads_fwd = tuned_fwd_t
            applied.append(f"num_threads_fwd={tuned_fwd_t}")
        if num_threads_bwd is None and num_threads is None:
            num_threads_bwd = tuned_bwd_t
            applied.append(f"num_threads_bwd={tuned_bwd_t}")
        # if applied:
        #     print(f"[HSA_pope_prerotate tuned] G={G} -> {', '.join(applied)}")

    if q_offset is None:
        q_offset = L_kv - L
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    freqs_f = freqs.float()
    if bias.dim() == 2:
        bias_3d = bias.view(H, G, R)
    elif bias.dim() == 3:
        assert bias.shape == (H, G, R), f"bias shape {bias.shape} does not match [{H}, {G}, {R}]"
        bias_3d = bias
    else:
        raise ValueError(f"bias must be 2D [HQ, R] or 3D [H, G, R], got dim={bias.dim()}")
    bias_f = bias_3d.float()

    # Step-A optimization: pre-allocate q_eff/k_eff and write each segment in-place,
    # eliminating the two torch.cat calls and the trailing .contiguous() copy.
    # Layout of out tensor [..., D + R]:
    #   [..., :R]      = q[..., :R] * cos(theta)   (rotated head part)
    #   [..., R:D]     = q[..., R:]                (pass-through tail, only if R < D)
    #   [..., D:D+R]   = q[..., :R] * sin(theta)   (sin-rotated part)

    q_pos = torch.arange(L, device=q.device, dtype=torch.long) + int(q_offset)
    theta_q = freqs_f.index_select(0, q_pos).view(1, L, 1, R) - bias_f.reshape(1, 1, HQ, R)
    q_rot_f = q[..., :R].float()
    q_eff = torch.empty(B, L, HQ, D + R, dtype=q.dtype, device=q.device)
    q_eff[..., :R].copy_((q_rot_f * torch.cos(theta_q)).to(q.dtype))
    if R < D:
        q_eff[..., R:D].copy_(q[..., R:])
    q_eff[..., D:D + R].copy_((q_rot_f * torch.sin(theta_q)).to(q.dtype))

    # Step-C optimization: cache K-side cos/sin since theta_k depends only on
    # (freqs, L_kv, R) -- independent of batch/bias/q_offset. Reused across
    # layers and decoding steps that share `freqs`.
    cos_k, sin_k = _get_pope_kfreq_cossin(freqs, L_kv, R, k.dtype, k.device)
    # Broadcast multiply directly in k.dtype (no fp32 materialization here).
    cos_k_b = cos_k.view(1, L_kv, 1, R)
    sin_k_b = sin_k.view(1, L_kv, 1, R)
    k_rot = k[..., :R]
    k_eff = torch.empty(B, L_kv, H, D + R, dtype=k.dtype, device=k.device)
    k_eff[..., :R].copy_(k_rot * cos_k_b)
    if R < D:
        k_eff[..., R:D].copy_(k[..., R:])
    k_eff[..., D:D + R].copy_(k_rot * sin_k_b)

    return _base_HSA_block_M_head(
        q_eff, k_eff, v, weights, indices,
        block_size=block_size, sm_scale=sm_scale, block_M=block_M,
        mask_last_token=mask_last_token, dtype=dtype, accum_dtype=accum_dtype,
        num_threads=num_threads, num_threads_fwd=num_threads_fwd, num_threads_bwd=num_threads_bwd,
        is_training=is_training,
    )









def _load_real_indices_for_breakdown(pt_path, B, layer_idx=None, device="cuda"):
    print(f"\n[real-indices] 加载: {pt_path}")
    saved = torch.load(pt_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    samples = saved["samples"]
    num_samples = len(samples)
    print(f"  config: seq_len={config['seq_len']}, chunk_size={config['chunk_size']}, "
          f"hsa_topk={config['hsa_topk']}, H_kv={config['num_key_value_heads']}")
    print(f"  num_samples={num_samples}, 需要 B={B}")
    assert num_samples >= B, f"样本数 {num_samples} 不足 B={B}"

    all_layer_idxs = sorted({li for s in samples for li in s["layers"].keys()})
    if layer_idx is None:
        layer_idx = all_layer_idxs[0]
        print(f"  自动选第一个 HSA 层: layer_idx={layer_idx}")
    else:
        assert layer_idx in all_layer_idxs
    print(f"  可用 HSA 层: {all_layer_idxs}")

    S = config["hsa_topk"]
    indices_list, weights_list = [], []
    for i in range(B):
        layer_data = samples[i]["layers"][layer_idx]
        idx = layer_data["indices"]
        cw = layer_data["chunk_weights"][:, :, :, :S]
        indices_list.append(idx)
        weights_list.append(cw)
    block_indices = torch.cat(indices_list, dim=0).to(dtype=torch.int32, device=device)
    weights = torch.cat(weights_list, dim=0).to(dtype=torch.bfloat16, device=device)
    actual_seq_len = block_indices.shape[1]
    print(f"  loaded: indices={list(block_indices.shape)}, weights={list(weights.shape)}, "
          f"actual_seq_len={actual_seq_len}")
    return block_indices, weights, actual_seq_len


def _adapt_indices_h_kv(block_indices, weights, target_H_kv):
    real_h_kv = block_indices.shape[2]
    if target_H_kv == real_h_kv:
        return block_indices, weights
    if target_H_kv > real_h_kv:
        assert target_H_kv % real_h_kv == 0, (
            f"target_H_kv={target_H_kv} 不是 real_h_kv={real_h_kv} 的整数倍")
        rep = target_H_kv // real_h_kv
        print(f"  H_kv 适配: {real_h_kv} → {target_H_kv} (repeat_interleave x{rep})")
        block_indices = block_indices.repeat_interleave(rep, dim=2).contiguous()
        if weights is not None:
            weights = weights.repeat_interleave(rep, dim=2).contiguous()
    else:
        print(f"  H_kv 适配: {real_h_kv} → {target_H_kv} (slice)")
        block_indices = block_indices[:, :, :target_H_kv, :].contiguous()
        if weights is not None:
            weights = weights[:, :, :target_H_kv, :].contiguous()
    return block_indices, weights


def _expand_weights_h_kv_to_hq(weights, HQ):
    H_kv = weights.shape[2]
    if H_kv == HQ:
        return weights.contiguous()
    assert HQ % H_kv == 0, f"HQ={HQ} 不是 H_kv={H_kv} 的整数倍"
    G = HQ // H_kv
    return weights.repeat_interleave(G, dim=2).contiguous()


def main_bwd_breakdown_latency(
    B=1, HQ=32, H=8, SEQ_LEN=8192, S=16, D=64, block_size=64,
    real_indices_path="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup/indices_8192.pt",
    layer_idx=None,
    M_list=(4, 32),
    nt_fwd_list=(64, 128),
    nt_bwd_list=(64, 128),
    paired_only=False,
    num_warmup=20, num_iters=50,
    mask_last_token=True,
    rotate_dim=None,
):
    import os
    device = "cuda"
    torch.manual_seed(0)

    if rotate_dim is None:
        rotate_dim = D

    def _as_tuple(x):
        if x is None:
            return ()
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x,)

    M_list = _as_tuple(M_list)
    nt_fwd_list = _as_tuple(nt_fwd_list)
    nt_bwd_list = _as_tuple(nt_bwd_list)
    assert len(M_list) > 0, "M_list 不能为空"
    assert len(nt_fwd_list) > 0, "nt_fwd_list 不能为空"
    assert len(nt_bwd_list) > 0, "nt_bwd_list 不能为空"

    print("=" * 78)
    print(f"[PoPE BWD Breakdown] B={B}, SEQ_LEN={SEQ_LEN}, HQ={HQ}, H={H}, D={D}, "
          f"S={S}, block_size={block_size}, rotate_dim={rotate_dim}")
    print("=" * 78)

    block_indices = None
    real_weights = None
    actual_L = SEQ_LEN
    if real_indices_path is not None and os.path.exists(real_indices_path):
        block_indices, real_weights, actual_L = _load_real_indices_for_breakdown(
            real_indices_path, B=B, layer_idx=layer_idx, device=device)
        block_indices, real_weights = _adapt_indices_h_kv(block_indices, real_weights, H)

        if actual_L != SEQ_LEN:
            print(f"  注意: real indices 的实际序列长度 {actual_L} != 请求的 {SEQ_LEN}，"
                  f"将以 actual_L 为准继续测速")
        SEQ_LEN = actual_L
    else:
        print(f"  [WARN] 未找到 real_indices_path={real_indices_path}，回退为"
              f" build_block_indices_block_M(overlap=0.8)")
        block_indices = build_block_indices_block_M(
            B=B, SEQ_LEN=SEQ_LEN, H=H, S=S,
            block_size=block_size, overlap_ratio=0.8,
            block_M=max(M_list),
            device=device,
        )
        real_weights = None

    dtype = torch.bfloat16
    q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device).contiguous()
    k = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device).contiguous()
    v = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device).contiguous()
    if real_weights is not None:
        weights = _expand_weights_h_kv_to_hq(real_weights, HQ)
    else:
        weights = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device).contiguous()

    freqs = (torch.randn((SEQ_LEN, rotate_dim), dtype=torch.float32, device=device) * 0.1).contiguous()
    pope_bias = (torch.randn((HQ, rotate_dim), dtype=torch.float32, device=device) * 0.01).contiguous()
    q_offset = 0

    print(f"  final shapes: q={list(q.shape)}, k={list(k.shape)}, v={list(v.shape)}, "
          f"weights={list(weights.shape)}, indices={list(block_indices.shape)}, "
          f"freqs={list(freqs.shape)}, bias={list(pope_bias.shape)}")

    if paired_only:
        assert len(M_list) == len(nt_fwd_list) == len(nt_bwd_list), \
            "paired_only=True 时三个列表长度必须一致"
        cfgs = list(zip(M_list, nt_fwd_list, nt_bwd_list))
    else:
        cfgs = [(m, nf, nb)
                for m in M_list
                for nf in nt_fwd_list for nb in nt_bwd_list]

    print(f"\n共 {len(cfgs)} 组配置待测试 "
          f"(M∈{list(M_list)}, nt_fwd∈{list(nt_fwd_list)}, "
          f"nt_bwd∈{list(nt_bwd_list)}, paired={paired_only})")

    print()
    print("=" * 78)
    print("[HSA_block_M_head_pope autograd]  走 autograd.Function 完整路径（单 kernel 反向）")
    print("=" * 78)
    header_inv = (f"{'M':>3} {'nt_f':>4} {'nt_b':>4} | "
                  f"{'FWD':>8} {'BWD':>8} {'TOTAL':>8} | {'BWD/FWD':>7}")
    print("-" * len(header_inv))
    print(header_inv)
    print("-" * len(header_inv))
    inv_results = []
    for (m, nf, nb) in cfgs:
        try:
            sm_scale_g = 1.0 / math.sqrt(D)
            g_q = q.detach().clone().contiguous().requires_grad_(True)
            g_k = k.detach().clone().contiguous().requires_grad_(True)
            g_v = v.detach().clone().contiguous().requires_grad_(True)
            g_weights = weights.detach().clone().contiguous().requires_grad_(True)
            g_bias = pope_bias.detach().clone().contiguous().requires_grad_(True)
            g_indices = block_indices.detach().clone().contiguous()
            g_do = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device).contiguous()

            def _zero_grads():
                g_q.grad = None; g_k.grad = None; g_v.grad = None; g_weights.grad = None; g_bias.grad = None

            for _ in range(num_warmup):
                _ = HSA_block_M_head_pope(
                    g_q, g_k, g_v, g_weights, g_indices,
                    block_size=block_size, sm_scale=sm_scale_g,
                    block_M=m,
                    num_threads_fwd=nf, num_threads_bwd=nb,
                    mask_last_token=mask_last_token,
                    freqs=freqs, bias=g_bias, q_offset=q_offset,
                )
                g_o = HSA_block_M_head_pope(
                    g_q, g_k, g_v, g_weights, g_indices,
                    block_size=block_size, sm_scale=sm_scale_g,
                    block_M=m,
                    num_threads_fwd=nf, num_threads_bwd=nb,
                    mask_last_token=mask_last_token,
                    freqs=freqs, bias=g_bias, q_offset=q_offset,
                )
                g_o.backward(g_do)
                _zero_grads()
            torch.cuda.synchronize()

            e_start = torch.cuda.Event(enable_timing=True)
            e_end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            e_start.record()
            for _ in range(num_iters):
                _ = HSA_block_M_head_pope(
                    g_q, g_k, g_v, g_weights, g_indices,
                    block_size=block_size, sm_scale=sm_scale_g,
                    block_M=m,
                    num_threads_fwd=nf, num_threads_bwd=nb,
                    mask_last_token=mask_last_token,
                    freqs=freqs, bias=g_bias, q_offset=q_offset,
                )
            e_end.record()
            torch.cuda.synchronize()
            g_fwd_ms = e_start.elapsed_time(e_end) / num_iters

            torch.cuda.synchronize()
            e_start.record()
            for _ in range(num_iters):
                g_o = HSA_block_M_head_pope(
                    g_q, g_k, g_v, g_weights, g_indices,
                    block_size=block_size, sm_scale=sm_scale_g,
                    block_M=m,
                    num_threads_fwd=nf, num_threads_bwd=nb,
                    mask_last_token=mask_last_token,
                    freqs=freqs, bias=g_bias, q_offset=q_offset,
                )
                g_o.backward(g_do)
                _zero_grads()
            e_end.record()
            torch.cuda.synchronize()
            g_fwd_bwd_ms = e_start.elapsed_time(e_end) / num_iters
            g_bwd_ms = g_fwd_bwd_ms - g_fwd_ms
            g_total_ms = g_fwd_ms + g_bwd_ms
            g_ratio = g_bwd_ms / max(g_fwd_ms, 1e-6)

            print(f"{m:>3} {nf:>4} {nb:>4} | "
                  f"{g_fwd_ms:>8.3f} {g_bwd_ms:>8.3f} {g_total_ms:>8.3f} | "
                  f"{g_ratio:>7.2f}")
            inv_results.append(dict(
                M=m, nt_fwd=nf, nt_bwd=nb,
                fwd_ms=g_fwd_ms, bwd_ms=g_bwd_ms,
                total_ms=g_total_ms, ratio=g_ratio,
            ))

            del g_q, g_k, g_v, g_weights, g_bias, g_indices, g_do
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{m:>3} {nf:>4} {nb:>4} |  FAILED: {type(e).__name__}: {e}")
            import traceback as _tb
            _tb.print_exc()
            torch.cuda.empty_cache()
    print("-" * len(header_inv))
    print("=" * 78)

    if inv_results:
        best_total = min(inv_results, key=lambda r: r["total_ms"])
        best_fwd = min(inv_results, key=lambda r: r["fwd_ms"])
        best_bwd = min(inv_results, key=lambda r: r["bwd_ms"])

        print()
        print("=" * 78)
        print("🏆 最快配置（按不同指标，HSA_block_M_head_pope autograd 完整路径）")
        print("=" * 78)
        def _fmt(r, tag):
            return (f"  [{tag:<9}] M={r['M']:>2}, "
                    f"nt_f={r['nt_fwd']:>3}, nt_b={r['nt_bwd']:>3} | "
                    f"FWD={r['fwd_ms']:.3f}  BWD={r['bwd_ms']:.3f}  "
                    f"TOTAL={r['total_ms']:.3f}  (BWD/FWD={r['ratio']:.2f})")
        print(_fmt(best_total, "Total最快"))
        print(_fmt(best_fwd, "FWD 最快"))
        print(_fmt(best_bwd, "BWD 最快"))
        print("=" * 78)
    return inv_results


def main_block_M_correctness(rotate_dim=None):
    """
    检验 PoPE 版 HSA_block_M_head_pope 的前向和反向正确性
    与 hsa_torch_ref (PoPE) 进行对比
    
    Args:
        rotate_dim: 旋转维度，None 表示全维旋转 (rotate_dim=D)
    """
    import math
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 1, 999, 1, 8, 128, 4, 64
    dtype = torch.bfloat16
    device = "cuda"
    block_M=4
    mask_last_token=True
    G = HQ // H
    scale = 1.0 / math.sqrt(D)
    is_training = True
    if rotate_dim is None:
        rotate_dim = D
    
    print(f"PoPE Correctness Config: Batch={B}, SeqLen={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, G={G}, S={S}, BlockSize={block_size}, rotate_dim={rotate_dim}")
    
    # ---------- 生成测试数据 ----------
    torch.manual_seed(42)
    
    # 创建 block_indices
    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.int32, device=device)
    num_blocks = SEQ_LEN // block_size
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                max_blocks = min(t // block_size + 1, num_blocks)
                if max_blocks > 0:
                    num_select = min(S, max_blocks)
                    selected = torch.randperm(max_blocks, device=device)[:num_select]
                    block_indices[b, t, h, :num_select] = selected

    block_indices = block_indices.sort(-1)[0]
    block_indices[block_indices == SEQ_LEN] = -1
    valid_mask = (block_indices != -1)

    kv_len = 999
    
    # 生成 PoPE 参数（宽度为 rotate_dim）
    freqs_len = max(SEQ_LEN, kv_len)
    freqs = torch.randn((freqs_len, rotate_dim), dtype=torch.float32, device=device) * 0.1
    pope_bias = (torch.randn((HQ, rotate_dim), dtype=torch.float32, device=device) * 0.01).detach().requires_grad_(True)
    q_offset = 0
    
    # PoPE 版本: Q 和 K 在 kernel 内做 softplus + 旋转
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成权重
    logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
    logits = logits.masked_fill(~valid_mask, float('-inf'))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)
    
    # ========== PoPE 前向 kernel 测试 ==========
    print("\n" + "=" * 70)
    print("[PoPE Forward Correctness Test]")
    print("=" * 70)
    
    O_hsa = HSA_block_M_head_pope(
        Q, K, V, W, block_indices,
        block_size=block_size, sm_scale=scale, block_M=block_M,
        mask_last_token=mask_last_token, is_training=is_training,
        freqs=freqs, bias=pope_bias, q_offset=q_offset,
    )
    
    # Torch reference with PoPE
    O_ref = hsa_torch_ref(
        Q.float().detach(),
        K.float().detach(),
        V.float().detach(),
        W.detach(),
        block_indices,
        chunk_size=block_size,
        sm_scale=scale,
        block_q=1,
        mask_last_token=mask_last_token,
        freqs=freqs.detach(),
        bias=pope_bias.detach(),
        q_offset=q_offset,
    )
    
    pope_fwd_max_diff = (O_hsa.float() - O_ref.float()).abs().max().item()
    pope_fwd_mean_diff = (O_hsa.float() - O_ref.float()).abs().mean().item()
    print(f"[PoPE Tilelang HSA_block_M] vs [PoPE Torch Reference]:")
    print(f"PoPE 前向最大误差: {pope_fwd_max_diff:.6e}")
    print(f"PoPE 前向平均误差: {pope_fwd_mean_diff:.6e}")
    
    # 检查一些具体位置
    for t in [0, SEQ_LEN // 4, SEQ_LEN // 2, SEQ_LEN - 1]:
        hsa_val = O_hsa[0, t, 0, 0].float().item()
        ref_val = O_ref[0, t, 0, 0].float().item()
        print(f"  t={t}: HSA={hsa_val:.6e}, Ref={ref_val:.6e}, Diff={abs(hsa_val - ref_val):.6e}")

    # ========== PoPE 反向 kernel 测试 ==========
    print("\n" + "=" * 70)
    print("[PoPE Backward Correctness Test]")
    print("=" * 70)
    
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # --- HSA backward ---
    O_hsa.backward(grad_output)
    dQ_hsa = Q.grad.clone()
    dK_hsa = K.grad.clone()
    dV_hsa = V.grad.clone()
    dW_hsa = W.grad.clone()
    dBias_hsa = pope_bias.grad.clone()
    
    # --- Torch Reference backward ---
    # 需要重新计算 ref forward 使其支持 backward
    Q.grad, K.grad, V.grad, W.grad = None, None, None, None
    pope_bias.grad = None
    
    O_ref_bwd = hsa_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices,
        chunk_size=block_size, sm_scale=scale, block_q=1,
        mask_last_token=mask_last_token,
        freqs=freqs, bias=pope_bias, q_offset=q_offset,
    )
    O_ref_bwd.backward(grad_output.float())
    dQ_ref = Q.grad.clone()
    dK_ref = K.grad.clone()
    dV_ref = V.grad.clone()
    dW_ref = W.grad.clone()
    dBias_ref = pope_bias.grad.clone()
    
    # --- 对比梯度 ---
    def get_err_ratio(x, y):
        err = (x.float() - y.float()).flatten().square().mean().sqrt().item()
        base = x.float().flatten().square().mean().sqrt().item()
        return err / base if base > 0 else 0.0
    
    def print_grad_compare(name, hsa_grad, ref_grad):
        max_diff = (hsa_grad.float() - ref_grad.float()).abs().max().item()
        mean_diff = (hsa_grad.float() - ref_grad.float()).abs().mean().item()
        ratio = get_err_ratio(ref_grad, hsa_grad)
        print(f"  {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, err_ratio={ratio:.6e}")
    
    print_grad_compare("dQ", dQ_hsa, dQ_ref)
    print_grad_compare("dK", dK_hsa, dK_ref)
    print_grad_compare("dV", dV_hsa, dV_ref)
    print_grad_compare("dW", dW_hsa, dW_ref)
    print_grad_compare("dBias", dBias_hsa, dBias_ref)



def main_prerotate_correctness(B=1, SEQ_LEN=999, H=1, HQ=8, D=64, S=4,
                               block_size=64, rotate_dim=32, block_M=4,
                               mask_last_token=True):
    """
    Verify HSA_block_M_head_pope_prerotate (PyTorch-prerotate + base non-PoPE kernel
    + autograd-handled bias gradient) against hsa_torch_ref.

    The base non-PoPE HSA kernel requires effective head_dim (D + rotate_dim) <= 128,
    so pick (D, rotate_dim) accordingly.
    """
    import math
    import torch
    import torch.nn.functional as F

    device = "cuda"
    dtype = torch.bfloat16
    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    print(f"\n{'='*70}")
    print(f"[Prerotate Wrapper Correctness] B={B}, SEQ_LEN={SEQ_LEN}, H={H}, HQ={HQ}, "
          f"D={D}, G={G}, S={S}, block_size={block_size}, rotate_dim={rotate_dim}")
    print(f"  effective head_dim for base kernel = D + R = {D + rotate_dim}")
    print(f"{'='*70}")

    assert D + rotate_dim <= 256, (
        f"D+rotate_dim={D + rotate_dim} > 256, base kernel cannot run this config")

    torch.manual_seed(42)
    kv_len = SEQ_LEN

    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.int32, device=device)
    num_blocks = SEQ_LEN // block_size
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                max_blocks = min(t // block_size + 1, num_blocks)
                if max_blocks > 0:
                    num_select = min(S, max_blocks)
                    selected = torch.randperm(max_blocks, device=device)[:num_select]
                    block_indices[b, t, h, :num_select] = selected
    block_indices = block_indices.sort(-1)[0]
    block_indices[block_indices == SEQ_LEN] = -1
    valid_mask = (block_indices != -1)

    freqs_len = max(SEQ_LEN, kv_len)
    freqs = torch.randn((freqs_len, rotate_dim), dtype=torch.float32, device=device) * 0.1
    pope_bias = (torch.randn((HQ, rotate_dim), dtype=torch.float32, device=device) * 0.01
                 ).detach().requires_grad_(True)
    q_offset = 0

    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=True)

    logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
    logits = logits.masked_fill(~valid_mask, float('-inf'))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)

    # --- Wrapper forward ---
    print("\n[Wrapper Forward]")
    O_wrap = HSA_block_M_head_pope_prerotate(
        Q, K, V, W, block_indices,
        block_size=block_size, sm_scale=scale, block_M=block_M,
        mask_last_token=mask_last_token, is_training=True,
        freqs=freqs, bias=pope_bias, q_offset=q_offset,
    )

    O_ref = hsa_torch_ref(
        Q.float().detach(), K.float().detach(), V.float().detach(), W.detach(),
        block_indices, chunk_size=block_size, sm_scale=scale, block_q=1,
        mask_last_token=mask_last_token,
        freqs=freqs.detach(), bias=pope_bias.detach(), q_offset=q_offset,
    )

    fwd_max = (O_wrap.float() - O_ref.float()).abs().max().item()
    fwd_mean = (O_wrap.float() - O_ref.float()).abs().mean().item()
    print(f"  fwd max_diff={fwd_max:.6e}, mean_diff={fwd_mean:.6e}")

    # --- Wrapper backward ---
    print("\n[Wrapper Backward]")
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    O_wrap.backward(grad_output)
    dQ_w = Q.grad.clone()
    dK_w = K.grad.clone()
    dV_w = V.grad.clone()
    dW_w = W.grad.clone()
    dBias_w = pope_bias.grad.clone()

    Q.grad = K.grad = V.grad = W.grad = None
    pope_bias.grad = None
    O_ref_bwd = hsa_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices,
        chunk_size=block_size, sm_scale=scale, block_q=1,
        mask_last_token=mask_last_token,
        freqs=freqs, bias=pope_bias, q_offset=q_offset,
    )
    O_ref_bwd.backward(grad_output.float())
    dQ_ref = Q.grad.clone()
    dK_ref = K.grad.clone()
    dV_ref = V.grad.clone()
    dW_ref = W.grad.clone()
    dBias_ref = pope_bias.grad.clone()

    def _err_ratio(x, y):
        err = (x.float() - y.float()).flatten().square().mean().sqrt().item()
        base = x.float().flatten().square().mean().sqrt().item()
        return err / base if base > 0 else 0.0

    def _cmp(name, a, b):
        max_diff = (a.float() - b.float()).abs().max().item()
        mean_diff = (a.float() - b.float()).abs().mean().item()
        ratio = _err_ratio(b, a)
        print(f"  {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
              f"err_ratio={ratio:.6e}")

    _cmp("dQ", dQ_w, dQ_ref)
    _cmp("dK", dK_w, dK_ref)
    _cmp("dV", dV_w, dV_ref)
    _cmp("dW", dW_w, dW_ref)
    _cmp("dBias", dBias_w, dBias_ref)


def main_prerotate_latency(B=4, HQ=16, H=16, SEQ_LEN=8192, S=16,
                           D=64, block_size=64, rotate_dim=32,
                           M_list=(16,), nt_fwd_list=(128,), nt_bwd_list=(128,),
                           paired_only=False,
                           num_warmup=20, num_iters=50,
                           mask_last_token=True,
                           real_indices_path=None, layer_idx=None):
    """
    Compare end-to-end (FWD + BWD) latency of:
      1) Fused PoPE kernel:    HSA_block_M_head_pope
      2) Prerotate wrapper:    HSA_block_M_head_pope_prerotate
                               (PyTorch prerotate Q/K + base non-PoPE kernel
                                + autograd handles bias grad)

    The base non-PoPE kernel requires D + rotate_dim <= 128.
    """
    import os
    import math
    import torch
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    assert D + rotate_dim <= 256, (
        f"D+rotate_dim={D + rotate_dim} > 256, prerotate wrapper cannot run "
        f"(base non-PoPE kernel limits effective head_dim to <= 256)")

    G = HQ // H
    print("=" * 78)
    print(f"[Prerotate Latency] B={B}, SEQ_LEN={SEQ_LEN}, HQ={HQ}, H={H}, D={D}, "
          f"S={S}, block_size={block_size}, rotate_dim={rotate_dim}, G={G}")
    print("=" * 78)

    actual_L = SEQ_LEN
    block_indices = None
    real_weights = None
    if real_indices_path is not None and os.path.exists(real_indices_path):
        block_indices, real_weights, actual_L = _load_real_indices_for_breakdown(
            real_indices_path, B, layer_idx=layer_idx, device=device)
        block_indices, real_weights = _adapt_indices_h_kv(block_indices, real_weights, H)
        SEQ_LEN = actual_L

    if block_indices is None:
        block_indices = build_block_indices_block_M(
            B=B, SEQ_LEN=SEQ_LEN, H=H, S=S, block_size=block_size,
            overlap_ratio=0.8, block_M=max(M_list), device=device,
        )

    q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device)
    v = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device)

    if real_weights is not None:
        weights = _expand_weights_h_kv_to_hq(real_weights, HQ).to(dtype)
    else:
        import torch.nn.functional as F
        valid_mask = (block_indices != -1).repeat_interleave(G, dim=2)
        logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
        logits = logits.masked_fill(~valid_mask, float('-inf'))
        weights = F.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, 0.0)

    freqs_len = SEQ_LEN
    freqs = torch.randn((freqs_len, rotate_dim), dtype=torch.float32, device=device) * 0.1
    pope_bias = (torch.randn((HQ, rotate_dim), dtype=torch.float32, device=device) * 0.01)
    q_offset = 0

    if paired_only:
        assert len(M_list) == len(nt_fwd_list) == len(nt_bwd_list)
        cfgs = list(zip(M_list, nt_fwd_list, nt_bwd_list))
    else:
        cfgs = [(m, nf, nb) for m in M_list for nf in nt_fwd_list for nb in nt_bwd_list]

    sm_scale = 1.0 / math.sqrt(D)

    def _bench(fn_factory, name):
        print()
        print("=" * 78)
        print(f"[{name}]")
        print("=" * 78)
        header = (f"{'M':>3} {'nt_f':>4} {'nt_b':>4} | "
                  f"{'FWD':>8} {'BWD':>8} {'TOTAL':>8} | {'BWD/FWD':>7}")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        results = []
        for (m, nf, nb) in cfgs:
            try:
                g_q = q.detach().clone().contiguous().requires_grad_(True)
                g_k = k.detach().clone().contiguous().requires_grad_(True)
                g_v = v.detach().clone().contiguous().requires_grad_(True)
                g_w = weights.detach().clone().contiguous().requires_grad_(True)
                g_bias = pope_bias.detach().clone().contiguous().requires_grad_(True)
                g_idx = block_indices.detach().clone().contiguous()
                g_do = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device).contiguous()

                def _zero_grads():
                    g_q.grad = None
                    g_k.grad = None
                    g_v.grad = None
                    g_w.grad = None
                    g_bias.grad = None

                fn = fn_factory(m, nf, nb)

                for _ in range(num_warmup):
                    o = fn(g_q, g_k, g_v, g_w, g_idx, g_bias)
                    o.backward(g_do)
                    _zero_grads()
                torch.cuda.synchronize()

                e_start = torch.cuda.Event(enable_timing=True)
                e_end = torch.cuda.Event(enable_timing=True)

                e_start.record()
                for _ in range(num_iters):
                    _ = fn(g_q, g_k, g_v, g_w, g_idx, g_bias)
                e_end.record()
                torch.cuda.synchronize()
                fwd_ms = e_start.elapsed_time(e_end) / num_iters

                e_start.record()
                for _ in range(num_iters):
                    o = fn(g_q, g_k, g_v, g_w, g_idx, g_bias)
                    o.backward(g_do)
                    _zero_grads()
                e_end.record()
                torch.cuda.synchronize()
                fb_ms = e_start.elapsed_time(e_end) / num_iters
                bwd_ms = fb_ms - fwd_ms
                total_ms = fwd_ms + bwd_ms
                ratio = bwd_ms / max(fwd_ms, 1e-6)

                print(f"{m:>3} {nf:>4} {nb:>4} | "
                      f"{fwd_ms:>8.3f} {bwd_ms:>8.3f} {total_ms:>8.3f} | "
                      f"{ratio:>7.2f}")
                results.append(dict(M=m, nt_fwd=nf, nt_bwd=nb,
                                    fwd_ms=fwd_ms, bwd_ms=bwd_ms,
                                    total_ms=total_ms, ratio=ratio))
                del g_q, g_k, g_v, g_w, g_bias, g_idx, g_do
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{m:>3} {nf:>4} {nb:>4} |  FAILED: {type(e).__name__}: {e}")
                import traceback as _tb
                _tb.print_exc()
                torch.cuda.empty_cache()
        print("-" * len(header))
        return results

    def _fused_factory(m, nf, nb):
        def _run(qq, kk, vv, ww, ii, bb):
            return HSA_block_M_head_pope(
                qq, kk, vv, ww, ii,
                block_size=block_size, sm_scale=sm_scale,
                block_M=m, num_threads_fwd=nf, num_threads_bwd=nb,
                mask_last_token=mask_last_token,
                freqs=freqs, bias=bb, q_offset=q_offset,
            )
        return _run

    def _wrapper_factory(m, nf, nb):
        def _run(qq, kk, vv, ww, ii, bb):
            return HSA_block_M_head_pope_prerotate(
                qq, kk, vv, ww, ii,
                block_size=block_size, sm_scale=sm_scale,
                block_M=m, num_threads_fwd=nf, num_threads_bwd=nb,
                mask_last_token=mask_last_token, is_training=True,
                freqs=freqs, bias=bb, q_offset=q_offset,
            )
        return _run

    fused_results = _bench(_fused_factory, "Fused PoPE kernel: HSA_block_M_head_pope")
    wrap_results = _bench(_wrapper_factory,
                          "Prerotate wrapper: HSA_block_M_head_pope_prerotate")

    if fused_results and wrap_results:
        bf = min(fused_results, key=lambda r: r["total_ms"])
        bw = min(wrap_results, key=lambda r: r["total_ms"])
        print()
        print("=" * 78)
        print("[Best total comparison]")
        print("=" * 78)
        print(f"  Fused      : M={bf['M']}, nt_f={bf['nt_fwd']}, nt_b={bf['nt_bwd']} | "
              f"FWD={bf['fwd_ms']:.3f}  BWD={bf['bwd_ms']:.3f}  TOTAL={bf['total_ms']:.3f}")
        print(f"  Prerotate  : M={bw['M']}, nt_f={bw['nt_fwd']}, nt_b={bw['nt_bwd']} | "
              f"FWD={bw['fwd_ms']:.3f}  BWD={bw['bwd_ms']:.3f}  TOTAL={bw['total_ms']:.3f}")
        speedup = bf["total_ms"] / max(bw["total_ms"], 1e-6)
        print(f"  Wrapper / Fused (TOTAL):  {1.0/speedup:.3f}x  (>1 means wrapper slower)")
        print("=" * 78)

    return fused_results, wrap_results



def main_block_M_latency():
    """
    对比 tilelang HSA_block_M 和 triton HSA 的 FWD、BWD、以及综合 (FWD+BWD) 延迟
    """
    import torch
    import torch.nn.functional as F
    import time
    import math
    from einops import rearrange
    from ops.hsa_fwd_bwd_triton import HSA

    # ---------- 配置参数 ----------
    # B, SEQ_LEN, H, HQ, D, S, block_size = 128, 1024*4, 1, 8, 128, 8, 64
    B, SEQ_LEN, H, HQ, D, S, block_size = 32, 4096, 1, 8, 128, 8, 64
    block_M=4
    mask_last_token=True
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    G_padded = HQ_padded // H
    need_padding = (HQ_padded > HQ)
    scale = 1.0 / math.sqrt(D)
    is_training = True

    print(f"Latency Config: B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block={block_size}, G={G}, HQ_padded={HQ_padded}, block_M={block_M}, mask_last_token={mask_last_token}")

    ### 生成成对重叠的 block_indices ###
    overlap_ratio = 0.8  # 可以调整重叠度
    block_indices = build_block_indices_block_M(
    B=B,
    SEQ_LEN=SEQ_LEN,
    H=H,
    S=S,
    block_size=block_size,
    overlap_ratio=overlap_ratio,
    block_M=block_M,
    device=device,
)
    print("Overlap_ratio:", overlap_ratio)

    # 创建 requires_grad=True 的输入（这些是 leaf tensors）
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成权重：仅在合法块上有非零概率，非法位置直接被置为非常小的 logit（softmax 后约为 0）
    # 这样 W 就天然在非法块上为 0，且仍为 leaf tensor（requires_grad=True）
    logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=torch.bfloat16, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, float('-inf'))  # 用大负数屏蔽非法位置
    W = F.softmax(logits, dim=-1) # leaf tensor，非法位近似为 0
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)

    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    
    # ---------- Triton block_indices ----------
    block_indices_triton = block_indices.clone()
    num_blocks = SEQ_LEN // block_size
    safe_block = num_blocks - 1
    block_indices_triton[block_indices_triton < 0] = safe_block

    # ---------- Triton padding ----------
    if need_padding:
        pad_heads = HQ_padded - HQ
        Q_triton = torch.cat(
            [Q, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2
        ).detach().clone().requires_grad_(True)
        grad_output_triton = torch.cat(
            [grad_output, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2
        )
    else:
        Q_triton = Q.detach().clone().requires_grad_(True)
        grad_output_triton = grad_output

    # ---------- TileLang ----------
    Q_tile = Q.detach().clone().requires_grad_(True)
    grad_output_tile = grad_output

    num_warmup = 50
    num_iters = 100

    # =========================================================
    # Helper function for timing
    # =========================================================
    def measure_time(func, *args, **kwargs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters  # ms per iter
    

    # =========================================================
    # TileLang 部分
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_block_M_head_pope(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, is_training=is_training)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_head_pope(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, is_training=is_training)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_head_pope(Q_tile, K, V, W, block_indices, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, is_training=is_training)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # 输出结果
    # =========================================================
    # print(f"[Triton]   FWD: {triton_fwd_ms:.3f} ms | BWD: {triton_bwd_ms:.3f} ms | Total: {triton_total_ms:.3f} ms")
    print(f"[TileLang] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")
    print()


def main_rope_correctness():
    """
    验证 enable_inverse_rope 开启后的数值正确性
    逻辑：RoPE(q_nope) -> HSA(enable_inverse_rope=True) -> 内部还原为 q_nope -> 与 Ref(q_nope) 对比
    """
    import torch
    import torch.nn.functional as F
    import math

    # ---------- 配置参数 ----------
    B, L, H, HQ, D, S, block_size = 1, 512, 1, 8, 128, 4, 64
    dtype = torch.bfloat16
    device = "cuda"
    block_M = 4
    scale = 1.0 / math.sqrt(D)
    is_training = True
    print(f"\nRoPE Correctness Test: B={B}, L={L}, HQ={HQ}, D={D}, S={S}, block_size={block_size}")

    # ---------- 准备数据 ----------
    torch.manual_seed(42)
    q_nope = torch.randn((B, L, HQ, D), dtype=dtype, device=device, requires_grad=True)
    k_nope = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((B, L, H, D), dtype=dtype, device=device, requires_grad=True)
    
    # 生成 cos/sin (Liger 格式: 1, L, D)
    angle = torch.randn((1, L, D), dtype=dtype, device=device)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    cos_f32 = cos.float()
    sin_f32 = sin.float()

    # 生成 block_indices 和 weights
    block_indices = torch.full((B, L, H, S), -1, dtype=torch.int32, device=device)
    for t in range(L):
        max_blocks = min(t // block_size + 1, L // block_size)
        if max_blocks > 0:
            num_select = min(S, max_blocks)
            block_indices[0, t, 0, :num_select] = torch.sort(torch.randperm(max_blocks, device=device)[:num_select])[0]
    
    weights = torch.randn((B, L, HQ, S), dtype=dtype, device=device)
    valid_mask = (block_indices != -1)
    weights_base = F.softmax(weights.masked_fill(~valid_mask, float('-inf')), dim=-1)
    weights_base = torch.nan_to_num(weights_base, 0.0)

    # ---------- 1. 构造输入的 q_rope, k_rope (使用 float32 计算) ----------
    q_t = q_nope.transpose(1, 2).contiguous().float()
    k_t = k_nope.transpose(1, 2).contiguous().float()
    
    q_rope_t, k_rope_t = liger_rotary_pos_emb(q_t, k_t, cos_f32, sin_f32)
    
    q_rope = q_rope_t.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    k_rope = k_rope_t.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)

    # ---------- 2. 准备两组独立的输入 ----------
    # HSA with inverse RoPE: 输入 q_rope, k_rope
    v_hsa = v.detach().clone().requires_grad_(True)
    weights_hsa = weights_base.detach().clone().requires_grad_(True)
    
    # Torch Ref: 输入 q_nope, k_nope
    q_nope_ref = q_nope.detach().clone().requires_grad_(True)
    k_nope_ref = k_nope.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    weights_ref = weights_base.detach().clone().requires_grad_(True)

    # ---------- 3. 前向测试 ----------
    
    # HSA 开启逆向 RoPE (内部会把 q_rope 变回 q_nope)
    o_hsa = HSA_block_M_head_pope(
        q_rope, k_rope, v_hsa, weights_hsa, block_indices, 
        block_size=block_size, sm_scale=scale, block_M=block_M, 
        mask_last_token=True, enable_inverse_rope=True, cos=cos, sin=sin, is_training=is_training
    )

    # Torch Ref: 直接用 nope 数据
    o_ref = hsa_torch_ref(
        q_nope_ref.float(), k_nope_ref.float(), v_ref.float(), 
        weights_ref, block_indices, 
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=True
    )

    # 找到前向最大误差位置
    diff_fwd = (o_hsa.float() - o_ref.float()).abs()
    max_fwd_err = diff_fwd.max().item()
    idx_fwd = tuple(i[0].item() for i in torch.where(diff_fwd == max_fwd_err))
    
    print(f"\n[Forward] Max Diff: {max_fwd_err:.6e} at {idx_fwd}")
    print(f"  HSA Val: {o_hsa[idx_fwd].item():.10f}")
    print(f"  Ref Val: {o_ref[idx_fwd].item():.10f}")

    # ---------- 4. 反向测试 ----------
    grad_output = torch.randn_like(o_hsa)
    
    # HSA with inverse RoPE 反向
    o_hsa.backward(grad_output)
    dq_rope_hsa = q_rope.grad.clone()
    dk_rope_hsa = k_rope.grad.clone()
    dv_hsa = v_hsa.grad.clone()
    dw_hsa = weights_hsa.grad.clone()

    # Torch Ref 反向
    o_ref.backward(grad_output.float())
    dq_nope_ref = q_nope_ref.grad.clone()
    dk_nope_ref = k_nope_ref.grad.clone()
    dv_ref = v_ref.grad.clone()
    dw_ref = weights_ref.grad.clone()

    # 转换 Ref 梯度
    dq_ref_t = dq_nope_ref.transpose(1, 2).contiguous().float()
    dk_ref_t = dk_nope_ref.transpose(1, 2).contiguous().float()
    dq_rope_ref_t, dk_rope_ref_t = liger_rotary_pos_emb(dq_ref_t, dk_ref_t, cos_f32, sin_f32)
    dq_rope_ref = dq_rope_ref_t.to(dtype).transpose(1, 2).contiguous()
    dk_rope_ref = dk_rope_ref_t.to(dtype).transpose(1, 2).contiguous()

    # 辅助函数：打印最大误差位置的对比
    def print_max_err_compare(name, tensor_hsa, tensor_ref):
        hsa_f = tensor_hsa.float()
        ref_f = tensor_ref.float()
        diff_abs = (hsa_f - ref_f).abs()
        
        # 1. 绝对误差最大位置
        max_abs_err = diff_abs.max().item()
        idx_abs = tuple(i[0].item() for i in torch.where(diff_abs == max_abs_err))
        
        # 2. 相对误差最大位置
        diff_rel = diff_abs / (ref_f.abs() + 1e-6)
        max_rel_err = diff_rel.max().item()
        idx_rel = tuple(i[0].item() for i in torch.where(diff_rel == max_rel_err))
        
        print(f"\n[{name} Error Analysis]:")
        print(f"  -> Max Absolute Error: {max_abs_err:.6e} at {idx_abs}")
        print(f"     HSA Val: {hsa_f[idx_abs].item():.10f} | Ref Val: {ref_f[idx_abs].item():.10f}")
        
        print(f"  -> Max Relative Error: {max_rel_err:.6e} at {idx_rel}")
        print(f"     HSA Val: {hsa_f[idx_rel].item():.10f} | Ref Val: {ref_f[idx_rel].item():.10f}")

    print("\n[Backward Gradients Comparison at Max Error Position]:")
    print_max_err_compare("DQ", dq_rope_hsa, dq_rope_ref)
    print_max_err_compare("DK", dk_rope_hsa, dk_rope_ref)
    print_max_err_compare("DV", dv_hsa, dv_ref)
    print_max_err_compare("DW", dw_hsa, dw_ref)




def main_rope_latency_memory():
    """
    对比开启和不开启 enable_inverse_rope 的延迟和显存
    """
    import torch
    import torch.nn.functional as F
    import math
    import gc

    # ---------- 配置参数 ----------
    B, SEQ_LEN, H, HQ, D, S, block_size = 64, 4096, 2, 16, 128, 8, 64
    block_M = 4
    mask_last_token = True
    dtype = torch.bfloat16
    device = "cuda"
    scale = 1.0 / math.sqrt(D)
    is_training = True

    print("=" * 70)
    print(f"RoPE Latency & Memory Test")
    print(f"Config: B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block={block_size}, block_M={block_M}")
    print("=" * 70)

    # ---------- 生成 block_indices ----------
    overlap_ratio = 0.8
    block_indices = build_block_indices_block_M(
        B=B,
        SEQ_LEN=SEQ_LEN,
        H=H,
        S=S,
        block_size=block_size,
        overlap_ratio=overlap_ratio,
        block_M=block_M,
        device=device,
    )

    # ---------- 生成 cos/sin ----------
    angle = torch.randn((1, SEQ_LEN, D), dtype=dtype, device=device)
    cos = torch.cos(angle).expand(B, -1, -1).contiguous()
    sin = torch.sin(angle).expand(B, -1, -1).contiguous()

    num_warmup = 20
    num_iters = 50

    def measure_time(func, num_iters=num_iters):
        """测量函数执行时间"""
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters  # ms per iter

    def measure_memory(func, description=""):
        """测量函数执行的峰值显存"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始显存
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        
        # 执行函数
        result = func()
        
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        mem_peak = torch.cuda.max_memory_allocated()
        
        activation_mem = mem_after - mem_before  # 常驻激活值
        peak_mem = mem_peak - mem_before  # 峰值显存增量
        
        return result, activation_mem, peak_mem

    def create_inputs(requires_grad=True):
        """创建输入张量"""
        Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=requires_grad)
        K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=requires_grad)
        V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=requires_grad)
        
        logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
        valid_mask = (block_indices != -1)
        repeat_factor = HQ // H  # 8
        valid_mask = valid_mask.unsqueeze(2)
        valid_mask = valid_mask.expand(B, SEQ_LEN, repeat_factor, H, S)
        valid_mask = valid_mask.reshape(B, SEQ_LEN, HQ, S)

        logits = logits.masked_fill(~valid_mask, float('-inf'))
        W = F.softmax(logits, dim=-1)
        W = torch.nan_to_num(W, 0.0).detach().requires_grad_(requires_grad)
        
        grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
        
        return Q, K, V, W, grad_output

    def bytes_to_mb(b):
        return b / (1024 * 1024)

    # ==================== 测试 1: 不开启 RoPE ====================
    print("\n" + "-" * 50)
    print("[Test 1] enable_inverse_rope = False")
    print("-" * 50)
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    Q, K, V, W, grad_output = create_inputs()
    
    # Warmup
    for _ in range(num_warmup):
        O = HSA_block_M_head_pope(Q, K, V, W, block_indices, 
                            block_size=block_size, sm_scale=scale, 
                            block_M=block_M, mask_last_token=mask_last_token,
                            enable_inverse_rope=False, is_training=is_training)
        O.backward(grad_output)
        Q.grad = K.grad = V.grad = W.grad = None

    # 测量 FWD 延迟
    def no_rope_fwd():
        with torch.no_grad():
            O = HSA_block_M_head_pope(Q, K, V, W, block_indices, 
                                block_size=block_size, sm_scale=scale, 
                                block_M=block_M, mask_last_token=mask_last_token,
                                enable_inverse_rope=False, is_training=is_training)
    
    fwd_ms_no_rope = measure_time(no_rope_fwd)

    # 测量 FWD+BWD 延迟
    def no_rope_fwd_bwd():
        Q.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_head_pope(Q, K, V, W, block_indices, 
                            block_size=block_size, sm_scale=scale, 
                            block_M=block_M, mask_last_token=mask_last_token,
                            enable_inverse_rope=False, is_training=is_training)
        O.backward(grad_output)
    
    total_ms_no_rope = measure_time(no_rope_fwd_bwd)
    bwd_ms_no_rope = total_ms_no_rope - fwd_ms_no_rope

    # 测量显存
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    Q1, K1, V1, W1, grad_output1 = create_inputs()
    mem_before = torch.cuda.memory_allocated()
    
    O1 = HSA_block_M_head_pope(Q1, K1, V1, W1, block_indices, 
                         block_size=block_size, sm_scale=scale, 
                         block_M=block_M, mask_last_token=mask_last_token,
                         enable_inverse_rope=False, is_training=is_training)
    
    torch.cuda.synchronize()
    mem_after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()
    
    activation_fwd_no_rope = mem_after_fwd - mem_before
    peak_fwd_no_rope = peak_fwd - mem_before
    
    torch.cuda.reset_peak_memory_stats()
    O1.backward(grad_output1)
    
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated()
    peak_bwd_no_rope = peak_bwd - mem_before

    print(f"延迟: FWD={fwd_ms_no_rope:.3f}ms | BWD={bwd_ms_no_rope:.3f}ms | Total={total_ms_no_rope:.3f}ms")
    print(f"显存: FWD激活={bytes_to_mb(activation_fwd_no_rope):.2f}MB | FWD峰值={bytes_to_mb(peak_fwd_no_rope):.2f}MB | BWD峰值={bytes_to_mb(peak_bwd_no_rope):.2f}MB")

    del Q, K, V, W, O, Q1, K1, V1, W1, O1, grad_output, grad_output1

    # ==================== 测试 2: 开启 RoPE ====================
    print("\n" + "-" * 50)
    print("[Test 2] enable_inverse_rope = True")
    print("-" * 50)
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 创建 rope 版本的输入
    Q_nope, K_nope, V, W, grad_output = create_inputs()
    
    # 生成 rope 版本
    Q_t = Q_nope.transpose(1, 2).contiguous().float()
    K_t = K_nope.transpose(1, 2).contiguous().float()
    cos_f32 = cos.float()
    sin_f32 = sin.float()
    
    Q_rope_t, K_rope_t = liger_rotary_pos_emb(Q_t, K_t, cos_f32, sin_f32)
    Q_rope = Q_rope_t.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    K_rope = K_rope_t.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    
    del Q_nope, K_nope, Q_t, K_t, Q_rope_t, K_rope_t
    gc.collect()
    torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(num_warmup):
        O = HSA_block_M_head_pope(Q_rope, K_rope, V, W, block_indices, 
                            block_size=block_size, sm_scale=scale, 
                            block_M=block_M, mask_last_token=mask_last_token,
                            enable_inverse_rope=True, cos=cos, sin=sin, is_training=is_training)
        O.backward(grad_output)
        Q_rope.grad = K_rope.grad = V.grad = W.grad = None

    # 测量 FWD 延迟
    def rope_fwd():
        with torch.no_grad():
            O = HSA_block_M_head_pope(Q_rope, K_rope, V, W, block_indices, 
                                block_size=block_size, sm_scale=scale, 
                                block_M=block_M, mask_last_token=mask_last_token,
                                enable_inverse_rope=True, cos=cos, sin=sin, is_training=is_training)
    
    fwd_ms_rope = measure_time(rope_fwd)

    # 测量 FWD+BWD 延迟
    def rope_fwd_bwd():
        Q_rope.grad = K_rope.grad = V.grad = W.grad = None
        O = HSA_block_M_head_pope(Q_rope, K_rope, V, W, block_indices, 
                            block_size=block_size, sm_scale=scale, 
                            block_M=block_M, mask_last_token=mask_last_token,
                            enable_inverse_rope=True, cos=cos, sin=sin, is_training=is_training)
        O.backward(grad_output)
    
    total_ms_rope = measure_time(rope_fwd_bwd)
    bwd_ms_rope = total_ms_rope - fwd_ms_rope

    # 测量显存
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    Q_nope2, K_nope2, V2, W2, grad_output2 = create_inputs()
    Q_t2 = Q_nope2.transpose(1, 2).contiguous().float()
    K_t2 = K_nope2.transpose(1, 2).contiguous().float()
    Q_rope_t2, K_rope_t2 = liger_rotary_pos_emb(Q_t2, K_t2, cos_f32, sin_f32)
    Q_rope2 = Q_rope_t2.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    K_rope2 = K_rope_t2.to(dtype).transpose(1, 2).contiguous().detach().requires_grad_(True)
    del Q_nope2, K_nope2, Q_t2, K_t2, Q_rope_t2, K_rope_t2
    gc.collect()
    torch.cuda.empty_cache()
    
    mem_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    
    O2 = HSA_block_M_head_pope(Q_rope2, K_rope2, V2, W2, block_indices, 
                         block_size=block_size, sm_scale=scale, 
                         block_M=block_M, mask_last_token=mask_last_token,
                         enable_inverse_rope=True, cos=cos, sin=sin, is_training=is_training)
    
    torch.cuda.synchronize()
    mem_after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()
    
    activation_fwd_rope = mem_after_fwd - mem_before
    peak_fwd_rope = peak_fwd - mem_before
    
    torch.cuda.reset_peak_memory_stats()
    O2.backward(grad_output2)
    
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated()
    peak_bwd_rope = peak_bwd - mem_before

    print(f"延迟: FWD={fwd_ms_rope:.3f}ms | BWD={bwd_ms_rope:.3f}ms | Total={total_ms_rope:.3f}ms")
    print(f"显存: FWD激活={bytes_to_mb(activation_fwd_rope):.2f}MB | FWD峰值={bytes_to_mb(peak_fwd_rope):.2f}MB | BWD峰值={bytes_to_mb(peak_bwd_rope):.2f}MB")

    # ==================== 对比汇总 ====================
    print("\n" + "=" * 70)
    print("对比汇总")
    print("=" * 70)
    
    print(f"\n{'指标':<20} {'无RoPE':<20} {'有RoPE':<20} {'开销':<20}")
    print("-" * 70)
    
    fwd_overhead = fwd_ms_rope - fwd_ms_no_rope
    bwd_overhead = bwd_ms_rope - bwd_ms_no_rope
    total_overhead = total_ms_rope - total_ms_no_rope
    
    print(f"{'FWD 延迟 (ms)':<20} {fwd_ms_no_rope:<20.3f} {fwd_ms_rope:<20.3f} {fwd_overhead:+.3f} ({fwd_overhead/fwd_ms_no_rope*100:+.1f}%)")
    print(f"{'BWD 延迟 (ms)':<20} {bwd_ms_no_rope:<20.3f} {bwd_ms_rope:<20.3f} {bwd_overhead:+.3f} ({bwd_overhead/bwd_ms_no_rope*100:+.1f}%)")
    print(f"{'Total 延迟 (ms)':<20} {total_ms_no_rope:<20.3f} {total_ms_rope:<20.3f} {total_overhead:+.3f} ({total_overhead/total_ms_no_rope*100:+.1f}%)")
    
    print()
    activation_overhead = activation_fwd_rope - activation_fwd_no_rope
    peak_fwd_overhead = peak_fwd_rope - peak_fwd_no_rope
    peak_bwd_overhead = peak_bwd_rope - peak_bwd_no_rope
    
    print(f"{'FWD 激活 (MB)':<20} {bytes_to_mb(activation_fwd_no_rope):<20.2f} {bytes_to_mb(activation_fwd_rope):<20.2f} {bytes_to_mb(activation_overhead):+.2f}")
    print(f"{'FWD 峰值 (MB)':<20} {bytes_to_mb(peak_fwd_no_rope):<20.2f} {bytes_to_mb(peak_fwd_rope):<20.2f} {bytes_to_mb(peak_fwd_overhead):+.2f}")
    print(f"{'BWD 峰值 (MB)':<20} {bytes_to_mb(peak_bwd_no_rope):<20.2f} {bytes_to_mb(peak_bwd_rope):<20.2f} {bytes_to_mb(peak_bwd_overhead):+.2f}")



import pytest
import torch.nn.functional as F
@pytest.mark.parametrize("B, SEQ_LEN, H, HQ, D, S, block_size, rotate_dim", [
    # PoPE 全维旋转 (rotate_dim == D)
    (1, 1024, 1, 8, 64, 4, 32, 64),
    # PoPE partial rotation (rotate_dim < D)
    (1, 1024, 1, 8, 64, 4, 32, 32),
])
def test_correctness_fp32(B, SEQ_LEN, H, HQ, D, S, block_size, rotate_dim):
    device = "cuda"
    dtype = torch.float32
    scale = 1.0 / math.sqrt(D)
    block_M = 2
    mask_last_token = True
    torch.manual_seed(42)
    is_training = True
    G = HQ // H

    print(f"\n{'='*70}")
    print(f"FP32 Test Config: B={B}, SEQ_LEN={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block_size={block_size}, rotate_dim={rotate_dim}")
    print(f"{'='*70}")

    # 1. 构造 block_indices
    block_indices = torch.full((B, SEQ_LEN, H, S), -1, dtype=torch.int32, device=device)
    num_blocks = SEQ_LEN // block_size
    for t in range(SEQ_LEN):
        max_blocks = min(t // block_size + 1, num_blocks)
        if max_blocks > 0:
            num_select = min(S, max_blocks)
            selected = torch.randperm(max_blocks, device=device)[:num_select]
            block_indices[:, t, :, :num_select] = selected.sort()[0]

    # 2. 生成输入数据
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    
    logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
    logits.masked_fill_(block_indices.repeat_interleave(G, dim=2) == -1, float('-inf'))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    # 3. PoPE 参数（PoPE kernel 强制要求 freqs 和 bias）
    freqs_len = SEQ_LEN
    freqs = torch.randn((freqs_len, rotate_dim), dtype=torch.float32, device=device) * 0.1
    pope_bias = (torch.randn((HQ, rotate_dim), dtype=torch.float32, device=device) * 0.01).detach().requires_grad_(True)

    # 4. 前向计算
    O_hsa = HSA_block_M_head_pope(Q, K, V, W, block_indices, block_size, scale, block_M, mask_last_token, "float", "float", 64,
                             is_training=is_training, freqs=freqs, bias=pope_bias)
    O_ref = hsa_torch_ref(Q.detach(), K.detach(), V.detach(), W.detach(), block_indices,
                        chunk_size=block_size, 
                        sm_scale=scale, 
                        block_q=1,
                        mask_last_token=mask_last_token,
                        freqs=freqs, bias=pope_bias)

    # 5. 验证前向
    torch.testing.assert_close(O_hsa, O_ref, atol=0.005, rtol=0.005, msg="Forward mismatch")
    print(f"✅ Forward PASSED")

    # 6. 反向计算
    O_hsa.backward(grad_output)
    DQ_hsa, DK_hsa, DV_hsa, DW_hsa = Q.grad.clone(), K.grad.clone(), V.grad.clone(), W.grad.clone()
    DBias_hsa = pope_bias.grad.clone()
    
    Q.grad, K.grad, V.grad, W.grad = None, None, None, None
    pope_bias.grad = None
    O_ref_bwd = hsa_torch_ref(Q, K, V, W, block_indices, 
                            chunk_size=block_size, 
                            sm_scale=scale, 
                            block_q=1,
                            mask_last_token=mask_last_token,
                            freqs=freqs, bias=pope_bias)
    O_ref_bwd.backward(grad_output)

    # 7. 验证反向梯度
    def get_abs_err(x, y):
        return (x-y).flatten().abs().max().item()
    def get_err_ratio(x, y):
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / base
    def assert_close(prefix, ref, tri, ratio):
        msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
        print(msg)
        assert get_err_ratio(ref, tri) < ratio, msg
    assert_close("DQ", Q.grad, DQ_hsa, 0.005)
    assert_close("DK", K.grad, DK_hsa, 0.005)
    assert_close("DV", V.grad, DV_hsa, 0.005)
    assert_close("DW", W.grad, DW_hsa, 0.005)
    assert_close("DBias", pope_bias.grad, DBias_hsa, 0.005)
    print(f"✅ Backward PASSED")
    print(f"FP32 Correctness Test Passed for B={B}, SEQ_LEN={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block_size={block_size}, rotate_dim={rotate_dim}")




@pytest.mark.parametrize("B, q_len, kv_len, H, HQ, D, S, block_size, is_training, mask_last_token, rotate_dim", [
    # ===== PoPE 全维旋转 (rotate_dim == D) =====
    # 训练场景: q_len == kv_len, is_training=True
    (1, 1024, 1024, 1, 8, 128, 4, 64, True, True, 128),
    # 推理场景 - chunk prefill: q_len == kv_len, is_training=False
    (1, 1024, 1024, 1, 8, 128, 4, 64, False, True, 128),
    # 推理场景 - chunk prefill: q_len < kv_len, is_training=False
    (1, 1024, 2048, 1, 8, 128, 4, 64, False, True, 128),
    # 推理场景 - decode: q_len=1, is_training=False
    (1, 1, 2048, 1, 8, 128, 4, 64, False, True, 128),
    # ===== PoPE partial rotation (rotate_dim < D) =====
    (1, 1024, 1024, 1, 8, 128, 4, 64, True, True, 64),
    # 推理场景 - chunk prefill: q_len == kv_len
    (1, 1024, 1024, 1, 8, 128, 4, 64, False, True, 64),
    # 推理场景 - chunk prefill: q_len < kv_len
    (1, 1024, 2048, 1, 8, 128, 4, 64, False, True, 64),
    # 推理场景 - decode: q_len=1
    (1, 1, 2048, 1, 8, 128, 4, 64, False, True, 64),
])
def test_train_inference_correctness(B, q_len, kv_len, H, HQ, D, S, block_size, is_training, mask_last_token, rotate_dim):
    """
    检验 HSA_block_M_head_pope 封装类的前向和反向传播数值正确性
    与 hsa_torch_ref 进行对比

    支持以下场景:
    - 训练: q_len == kv_len, is_training=True, 测试 fwd + bwd
    - 推理 chunk prefill: q_len <= kv_len, is_training=False, 仅测试 fwd
    - 推理 decode: q_len=1, is_training=False, 仅测试 fwd

    rotate_dim: 旋转维度，指定 PoPE 旋转前 rotate_dim 维。
    """
    device = "cuda"
    dtype = torch.bfloat16
    block_M = 4
    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    print(f"\n{'='*70}")
    print(f"Test Config: B={B}, q_len={q_len}, kv_len={kv_len}, H={H}, HQ={HQ}, D={D}, G={G}, S={S}, block_size={block_size}, rotate_dim={rotate_dim}")
    print(f"is_training={is_training}, mask_last_token={mask_last_token}")
    print(f"{'='*70}")

    # ---------- 生成测试数据 ----------
    torch.manual_seed(42)

    # 创建 block_indices
    # 注意: indices 指向的是 KV 序列中的 chunk 索引
    block_indices = torch.full((B, q_len, H, S), -1, dtype=torch.int32, device=device)
    num_kv_blocks = kv_len // block_size  # 使用 kv_len 计算可用的 KV block 数量

    # 计算 q 在全局序列中的偏移 (用于 chunk prefill 场景)
    q_offset = kv_len - q_len  # q 对应 KV 中最后 q_len 个 token

    for b in range(B):
        for t in range(q_len):
            for h in range(H):
                # 当前 q token 在全局序列中的位置
                global_pos = q_offset + t
                # causal: 只能访问 global_pos 之前的 KV block
                max_blocks = min((global_pos // block_size) + 1, num_kv_blocks)
                if max_blocks > 0:
                    num_select = min(S, max_blocks)
                    selected = torch.randperm(max_blocks, device=device)[:num_select]
                    block_indices[b, t, h, :num_select] = selected

    sort_temp = block_indices.clone()
    sort_temp[sort_temp == -1] = num_kv_blocks+100  # 用一个很大的值确保排到最后
    sort_temp = sort_temp.sort(-1)[0]
    sort_temp[sort_temp == num_kv_blocks+100] = -1
    block_indices = sort_temp

    # 创建 requires_grad=True 的输入（训练模式需要梯度）
    Q = torch.randn((B, q_len, HQ, D), dtype=dtype, device=device, requires_grad=is_training)
    K = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=is_training)
    V = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=is_training)

    # 生成权重：仅在合法块上有非零概率，非法位置直接被置为非常小的 logit（softmax 后约为 0）
    logits = torch.randn((B, q_len, HQ, S), dtype=dtype, device=device)
    valid_mask = (block_indices != -1)
    # 扩展 valid_mask 到 HQ 维度
    valid_mask_expanded = valid_mask.repeat_interleave(G, dim=2)  # (B, q_len, HQ, S)
    logits = logits.masked_fill(~valid_mask_expanded, float('-inf'))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(is_training)

    # PoPE 参数（PoPE kernel 强制要求 freqs 和 bias）
    freqs_len = max(q_len + q_offset, kv_len)
    freqs = torch.randn((freqs_len, rotate_dim), dtype=torch.float32, device=device) * 0.1
    pope_bias = (torch.randn((HQ, rotate_dim), dtype=torch.float32, device=device) * 0.01).detach().requires_grad_(is_training)

    # ========== 测试前向传播 ==========

    # HSA_block_M_head_pope 前向
    O_hsa = HSA_block_M_head_pope(
        Q, K, V, W, block_indices,
        block_size=block_size,
        sm_scale=scale,
        block_M=block_M,
        mask_last_token=mask_last_token,
        is_training=is_training,
        freqs=freqs, bias=pope_bias, q_offset=q_offset,
    )

    # Torch reference 前向
    O_ref = hsa_torch_ref(
        Q.float().detach(),
        K.float().detach(),
        V.float().detach(),
        W.detach(),
        block_indices,
        chunk_size=block_size,
        sm_scale=scale,
        block_q=1,
        mask_last_token=mask_last_token,
        freqs=freqs, bias=pope_bias, q_offset=q_offset,
    )

    # 比较前向输出
    fwd_max_diff = (O_hsa.float() - O_ref.float()).abs().max().item()
    fwd_mean_diff = (O_hsa.float() - O_ref.float()).abs().mean().item()

    print(f"[Tilelang HSA_block_M_head_pope] vs [Torch Reference]:")
    print(f"前向最大误差: {fwd_max_diff:.6e}, 平均误差: {fwd_mean_diff:.6e}")

    # 断言前向误差在可接受范围内
    assert fwd_max_diff < 0.01, f"前向误差过大: {fwd_max_diff:.6e}"
    print(f"✅ Forward PASSED")

    # ========== 测试反向传播（仅训练模式） ==========
    if is_training:
        grad_output = torch.randn((B, q_len, HQ, D), dtype=dtype, device=device)

        # 先计算 Torch reference 反向
        Q.grad, K.grad, V.grad, W.grad = None, None, None, None
        if pope_bias.grad is not None:
            pope_bias.grad = None
        O_ref_bwd = hsa_torch_ref(
            Q.float(), K.float(), V.float(), W.float(), block_indices,
            chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=mask_last_token,
            freqs=freqs, bias=pope_bias, q_offset=q_offset,
        )
        O_ref_bwd.backward(grad_output.float())
        DQ_ref = Q.grad.clone()
        DK_ref = K.grad.clone()
        DV_ref = V.grad.clone()
        DW_ref = W.grad.clone()
        DBias_ref = pope_bias.grad.clone()

        # 再计算 HSA_block_M_head_pope 反向
        Q.grad, K.grad, V.grad, W.grad = None, None, None, None
        pope_bias.grad = None
        O_hsa_bwd = HSA_block_M_head_pope(
            Q, K, V, W, block_indices,
            block_size=block_size, sm_scale=scale, block_M=block_M,
            mask_last_token=mask_last_token, is_training=is_training,
            freqs=freqs, bias=pope_bias, q_offset=q_offset,
        )
        O_hsa_bwd.backward(grad_output)
        DQ_hsa = Q.grad.clone()
        DK_hsa = K.grad.clone()
        DV_hsa = V.grad.clone()
        DW_hsa = W.grad.clone()
        DBias_hsa = pope_bias.grad.clone()

        # 比较梯度
        def get_err_ratio(x, y):
            err = (x.float()-y.float()).flatten().square().mean().sqrt().item()
            base = x.float().flatten().square().mean().sqrt().item()
            return err / base if base > 0 else 0.0

        # PoPE 涉及 softplus + 旋转，梯度链较长，bfloat16 下误差累积更大
        # DQ/DK 需经过 PoPE 旋转反向传播，容差适当放宽（参考 dense 版本使用 1.5e-2）
        grad_pairs = [("DQ", DQ_hsa, DQ_ref, 0.02), ("DK", DK_hsa, DK_ref, 0.02),
                      ("DV", DV_hsa, DV_ref, 0.01), ("DW", DW_hsa, DW_ref, 0.01),
                      ("DBias", DBias_hsa, DBias_ref, 0.02)]

        for name, grad_hsa, grad_ref, thr in grad_pairs:
            max_diff = (grad_hsa.float() - grad_ref.float()).abs().max().item()
            ratio = get_err_ratio(grad_ref, grad_hsa)
            print(f"{name} 最大误差: {max_diff:.6e}, 误差比: {ratio:.6e}")
            assert ratio < thr, f"{name} 反向误差比过大: {ratio:.6e} (阈值: {thr})"

        print(f"✅ Backward PASSED")

    print(f"✅ Test PASSED: B={B}, q_len={q_len}, kv_len={kv_len}, is_training={is_training}, rotate_dim={rotate_dim}")

if __name__ == "__main__":
    # ### 验证 HSA_block_M (TileLang) 的正确性
    # # 全维旋转 (rotate_dim == D，向后兼容)
    # main_block_M_correctness(rotate_dim=None)
    # # partial rotation (rotate_dim < D)
    # main_block_M_correctness(rotate_dim=64)

    ### 验证 prerotate wrapper (PyTorch prerotate + base non-PoPE kernel + autograd bias)
    # base kernel 限制 effective head_dim = D + rotate_dim <= 128
    main_prerotate_correctness(B=1, SEQ_LEN=999, H=1, HQ=8,
                               D=64, S=4, block_size=64, rotate_dim=32, block_M=4)
    main_prerotate_correctness(B=1, SEQ_LEN=999, H=1, HQ=8,
                               D=128, S=4, block_size=64, rotate_dim=64, block_M=4)

    # # B, q_len, kv_len, H, HQ, D, S, block_size, is_training, mask_last_token, rotate_dim
    # params_list = [
    #     # ===== PoPE 全维旋转 (rotate_dim == D) =====
    #     (1, 1024, 1024, 1, 8, 128, 4, 64, True, True, 128),
    #     (1, 1024, 1024, 1, 8, 128, 4, 64, False, True, 128),
    #     (1, 1024, 2048, 1, 8, 128, 4, 64, False, True, 128),
    #     (1, 1, 2048, 1, 8, 128, 4, 64, False, True, 128),
    #     # ===== PoPE partial rotation (rotate_dim < D) =====
    #     (1, 1024, 1024, 1, 8, 128, 4, 64, True, True, 64),
    #     (1, 1024, 1024, 1, 8, 128, 4, 64, False, True, 64),
    #     (1, 1024, 2048, 1, 8, 128, 4, 64, False, True, 64),
    #     (1, 1, 2048, 1, 8, 128, 4, 64, False, True, 64),
    # ]
    # for p in params_list:
    #     test_train_inference_correctness(*p)

    ### 对比 HSA_block_M (TileLang) 与 Triton HSA 的延迟
    # main_block_M_latency()

    # main_rope_correctness()

    # main_rope_latency_memory()

    # (B, SEQ_LEN, H, HQ, D, S, block_size, rotate_dim)
    # params_list = [
    #     # PoPE 全维旋转
    #     (1, 1000, 1, 8, 64, 4, 32, 64),
    #     (2, 2048, 2, 16, 128, 4, 32, 128),
    #     (3, 512, 1, 8, 64, 4, 32, 64),
    #     (4, 512, 2, 16, 128, 4, 64, 128),
    #     (5, 256, 1, 8, 64, 4, 64, 64),
    #     # PoPE partial rotation
    #     (1, 1000, 1, 8, 64, 4, 32, 32),
    #     (2, 2048, 2, 16, 128, 4, 32, 64),
    # ]
    # for p in params_list:
    #     test_correctness_fp32(*p)

    # main_bwd_breakdown_latency(
    #     B=4, HQ=16, H=16, SEQ_LEN=8192,
    #     S=16, D=128, block_size=64,rotate_dim=64,
    #     real_indices_path="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup/indices_8192.pt",
    #     layer_idx=None,
    #     M_list=(16,),
    #     nt_fwd_list=(128,),
    #     nt_bwd_list=(128,),
    #     paired_only=False,
    #     num_warmup=20, num_iters=50,
    #     mask_last_token=True,
    # )

    ### Latency: fused PoPE kernel  vs  prerotate wrapper (base non-PoPE kernel + autograd bias)
    ### NOTE: base non-PoPE kernel requires D + rotate_dim <= 128, so use smaller D / R here.
    main_prerotate_latency(
        B=4, HQ=32, H=8, SEQ_LEN=8192,
        S=16, D=64, block_size=64, rotate_dim=32,
        M_list=(8,),
        nt_fwd_list=(128,),
        nt_bwd_list=(128,),
        paired_only=False,
        num_warmup=20, num_iters=50,
        mask_last_token=True,
        real_indices_path="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/real_indices_backup/indices_8192.pt",
        layer_idx=None,
    )

# pkill -f "burner.*--gpu 7"; python ops/hsa_fwd_bwd_head_pope.py
