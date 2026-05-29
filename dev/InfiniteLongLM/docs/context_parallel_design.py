"""
=====================================================================
LHSA Context Parallel (CP) 设计方案 v4
=====================================================================

核心演进:
  v1: All-to-All 按需 fetch — 通信少但显存无上界
  v2: Sparse Ring — 显存有界但 zigzag 下 rank-level skip 效果差
  v3: Zigzag + Standard Ring + Chunk-Sparse Compute — 计算省但通信不省
  v4: Zigzag + Chunk-Granular P2P ⭐ — 通信也省，显存有界

v3 的结论：
  - Zigzag 保证负载均衡 ✅
  - Chunk-level sparse compute 省 99%+ 计算 ✅
  - 但通信仍然是 O(L)：ring 每步传整个 L/P 的 KV，即使只用几个 chunk ✗

v4 的核心目标：
  通信量从 O(L) 降到 O(U_total * S)，U_total = 全局 unique 远程 chunk 数
  同时保持显存有界（不像 v1 那样最坏爆显存）


=====================================================================
Part 1: 回顾 — 为什么 Ring 通信省不了
=====================================================================

Ring Attention 的 P2P 是 **集体操作**：每一步每个 rank 必须同时 send/recv，
不能单独跳过某一步。即使本 rank 不需要某个 sender 的 chunk，
ring shift 仍然需要执行（否则死锁）。

Chunk-Granular P2P 的思路：完全抛弃 ring topology，
改为 **异步点对点**（async P2P），只传被 topk 选中的 chunk。


=====================================================================
Part 2: Chunk-Granular P2P — 整体架构
=====================================================================

核心思想：
  1. 先做 TopK 检索（需要全局 landmark keys）
  2. 分析 topk indices → 构建 "谁需要谁的哪些 chunk" 的通信计划
  3. AlltoAll 交换通信计划（极小开销：只传 chunk indices）
  4. 按计划只传被选中的 chunk KV → P2P 异步收发
  5. 用 Bounded Buffer Pool 控制显存上界
  6. 流式处理：收一批算一批，buffer 复用

显存上界控制：
  不像 v1 把所有远程 chunk 一次性全收过来，
  而是分批接收，每批最多 M 个 chunk，处理完释放 buffer。
  显存上界 = 本地 KV + M * S * h * d （M 是可调超参）


=====================================================================
Part 3: 详细流程
=====================================================================

┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Phase 0: 线性投影 + SWA (在连续布局下)                               │
│  ─────────────────────────────────────────────                       │
│  与 v3 相同：                                                         │
│  (a) 线性投影得到 swa_q/k/v, hsa_q/k/v, lmk_q                      │
│  (b) P2P 左邻居 W 个 token → SWA → swa_o, lse_sum                  │
│  通信: O(W * h * d)                                                  │
│                                                                      │
│  Phase 1: Zigzag Scatter                                             │
│  ─────────────────────                                               │
│  与 v3 相同：AlltoAll 重排到 zigzag 布局                              │
│  通信: O(L/P * h * d)                                                │
│                                                                      │
│  Phase 2: AllGather Landmark Keys + TopK                             │
│  ───────────────────────────────────────                             │
│  与 v3 相同：                                                         │
│  (a) AllGather lmk_k → 全局 landmark keys                           │
│  (b) 本地 TopK → indices (B, N_local*S, h, K), scores               │
│  通信: O(N * h * d) = O(L/S * h * d)                                │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Phase 3: 构建 Chunk-Granular 通信计划 ⭐⭐                     │  │
│  │  ─────────────────────────────────────────                     │  │
│  │                                                                │  │
│  │  Step 3a: 分析 topk indices → 需要的远程 chunk                 │  │
│  │  ──────────────────────────────────────────                    │  │
│  │  chunk_owner = build_chunk_owner_map(P, N)                    │  │
│  │                                                                │  │
│  │  # 收集本 rank 需要的所有远程 chunk (去重)                      │  │
│  │  need_chunks = {}  # owner_rank → set of global chunk indices │  │
│  │  for (b, l, h, k) in topk_indices.nonzero():                  │  │
│  │      g = indices[b, l, h, k]  # 全局 chunk idx                │  │
│  │      owner = chunk_owner[g]                                   │  │
│  │      if owner != my_rank:                                     │  │
│  │          need_chunks[owner].add(g)                            │  │
│  │                                                                │  │
│  │  向量化实现 (高效):                                              │  │
│  │    flat_idx = indices.view(-1)          # 展平                  │  │
│  │    owners = chunk_owner[flat_idx]                              │  │
│  │    remote_mask = (owners != rank)                              │  │
│  │    remote_global_idx = flat_idx[remote_mask]                   │  │
│  │    unique_remote = remote_global_idx.unique()                  │  │
│  │    # 按 owner 分组                                              │  │
│  │    for r in range(P):                                          │  │
│  │        need_from_r = unique_remote[chunk_owner[unique_remote]==r]│ │
│  │        need_chunks[r] = need_from_r                            │  │
│  │                                                                │  │
│  │  Step 3b: AlltoAll 交换通信计划                                  │  │
│  │  ─────────────────────────────                                 │  │
│  │  每个 rank 需要告诉其他 rank "我需要你的哪些 chunk"               │  │
│  │  同时知道 "哪些 rank 需要我的哪些 chunk"                          │  │
│  │                                                                │  │
│  │  方案 A: AlltoAll 交换 chunk index lists                        │  │
│  │    send_counts[r] = len(need_chunks[r])                       │  │
│  │    # 先交换 counts (P 个 int)                                   │  │
│  │    recv_counts = alltoall(send_counts)                        │  │
│  │    # 再交换 chunk indices                                       │  │
│  │    request_indices = alltoall_v(need_chunks_flat, send_counts, │  │
│  │                                 recv_counts)                   │  │
│  │                                                                │  │
│  │  方案 B (推荐): Bitmap AlltoAll ⭐                               │  │
│  │    每个 rank 构建一个 bitmap[P][N_local]:                        │  │
│  │      bitmap[r][j] = 1 if 我需要 rank r 的第 j 个本地 chunk      │  │
│  │                                                                │  │
│  │    AlltoAll 交换 bitmap：                                        │  │
│  │      send_bitmap[r] = 我对 rank r 的请求 (N_local bits)         │  │
│  │      recv_bitmap[r] = rank r 对我的请求 (N_local bits)          │  │
│  │                                                                │  │
│  │    通信量 = P * P * N_local bits = P² * N/(P) bits              │  │
│  │           = P * N bits                                         │  │
│  │    例: P=8, N=2048 → 8 * 2048 = 16K bits = 2KB  (可忽略!)      │  │
│  │                                                                │  │
│  │    优势: 固定大小，不需要动态 alltoall_v                          │  │
│  │                                                                │  │
│  │  通信后每个 rank 知道:                                            │  │
│  │    recv_plan[r] = 从 rank r 接收哪些 chunk (need_chunks[r])     │  │
│  │    send_plan[r] = 发给 rank r 哪些 chunk (recv_bitmap[r])       │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Phase 4: Chunk-Granular P2P KV 传输 + 流式 HSA ⭐⭐⭐          │  │
│  │  ─────────────────────────────────────────────────             │  │
│  │                                                                │  │
│  │  Step 4a: 先处理本地 chunk (零通信)                              │  │
│  │  ────────────────────────────────────                          │  │
│  │  local_mask = (chunk_owner[indices] == my_rank)                │  │
│  │  # 重映射 indices 到本地 chunk idx                               │  │
│  │  local_indices = global_to_local(indices, chunk_map_local)     │  │
│  │  local_indices[~local_mask] = -1  # 非本地的标记为 -1           │  │
│  │                                                                │  │
│  │  hsa_o_local = HSA_block_M_group(                             │  │
│  │      hsa_q_zz, hsa_k_zz, hsa_v_zz,                           │  │
│  │      weights=chunk_weights, indices=local_indices,             │  │
│  │      block_size=S, mask_last_token=True                       │  │
│  │  )                                                             │  │
│  │  hsa_o_accum = hsa_o_local                                    │  │
│  │                                                                │  │
│  │  Step 4b: 流式接收远程 chunk + 分批 HSA ⭐                       │  │
│  │  ────────────────────────────────────────                      │  │
│  │                                                                │  │
│  │  === 显存预算控制 ===                                            │  │
│  │  MAX_CHUNKS_INFLIGHT = M  (超参，例如 32~128)                    │  │
│  │  buffer_k = alloc(M, S, h, d)   # 接收 buffer                  │  │
│  │  buffer_v = alloc(M, S, h, d)                                 │  │
│  │  显存 = 2 * M * S * h * d （固定，与 topk 结果无关）              │  │
│  │                                                                │  │
│  │  === 按 rank 分批处理 ===                                        │  │
│  │  对每个需要通信的 rank r (按 recv_plan):                          │  │
│  │    chunks_from_r = recv_plan[r]  # 本 rank 需要的 chunk list    │  │
│  │    num_chunks_r = len(chunks_from_r)                           │  │
│  │                                                                │  │
│  │    # 分批: 每批最多 M 个 chunk                                   │  │
│  │    for batch_start in range(0, num_chunks_r, M):               │  │
│  │        batch_end = min(batch_start + M, num_chunks_r)          │  │
│  │        batch_chunks = chunks_from_r[batch_start:batch_end]     │  │
│  │        num_this_batch = batch_end - batch_start                │  │
│  │                                                                │  │
│  │        # ---- 接收 KV ----                                      │  │
│  │        # rank r 已经知道要发哪些 chunk (send_plan[my_rank])     │  │
│  │        # 双方按约定好的 chunk 顺序收发                            │  │
│  │        recv_k = buffer_k[:num_this_batch]  # view, 不新分配     │  │
│  │        recv_v = buffer_v[:num_this_batch]                      │  │
│  │        dist.recv(recv_k, src=r)                               │  │
│  │        dist.recv(recv_v, src=r)                                │  │
│  │                                                                │  │
│  │        # ---- 构建 batch HSA 的 indices ----                    │  │
│  │        # batch_chunks 是全局 chunk idx，需要映射到 buffer 内下标 │  │
│  │        batch_g2l = {g: i for i, g in enumerate(batch_chunks)}  │  │
│  │        batch_indices = remap_indices(indices, batch_g2l)       │  │
│  │        # 不在这批的 → -1                                        │  │
│  │                                                                │  │
│  │        # ---- Sparse HSA ----                                   │  │
│  │        hsa_o_partial = HSA_block_M_group(                     │  │
│  │            hsa_q_zz,                                           │  │
│  │            recv_k[:num_this_batch].reshape(-1, h, d),  # 拼接  │  │
│  │            recv_v[:num_this_batch].reshape(-1, h, d),          │  │
│  │            weights=chunk_weights,                              │  │
│  │            indices=batch_indices,                              │  │
│  │            block_size=S, mask_last_token=True                  │  │
│  │        )                                                       │  │
│  │        hsa_o_accum += hsa_o_partial                            │  │
│  │                                                                │  │
│  │        # buffer 自动复用 (view, 不释放)                          │  │
│  │                                                                │  │
│  │  === 同时处理 send 请求 (异步) ===                                │  │
│  │  对于 send_plan 中的请求，rank r 也在并行发送:                     │  │
│  │    for r in send_plan:                                         │  │
│  │        chunks_to_send = send_plan[r]                           │  │
│  │        for batch_start in range(0, len(chunks_to_send), M):    │  │
│  │            batch_chunks = chunks_to_send[batch_start:...]      │  │
│  │            # gather 本地对应 chunk 的 KV                          │  │
│  │            send_k = hsa_k_zz[local_idx_of(batch_chunks)]      │  │
│  │            send_v = hsa_v_zz[local_idx_of(batch_chunks)]      │  │
│  │            dist.send(send_k, dst=r)                           │  │
│  │            dist.send(send_v, dst=r)                            │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Phase 5: Score Fusion (zigzag 布局下)                                │
│  ─────────────────────────────────────                               │
│  与 v3 相同                                                           │
│                                                                      │
│  Phase 6: Zigzag Gather → 恢复连续布局                                │
│  ──────────────────────────────────────                              │
│  与 v3 相同                                                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘


=====================================================================
Part 4: Send/Recv 不死锁的关键 — 调度协议
=====================================================================

问题：P2P send/recv 如果不协调，容易死锁。
例如 rank 0 在 send 给 rank 1，rank 1 也在 send 给 rank 0，
两者都 block 在 send 上（如果 buffer 满了）。

解法 1: 全异步 (推荐) ⭐
─────────────────────────

所有 send 和 recv 都用 isend / irecv（非阻塞），
然后统一 waitall。

for r in range(P):
    if r == my_rank: continue
    
    # 启动所有接收
    if len(recv_plan[r]) > 0:
        for batch in batched(recv_plan[r], M):
            recv_ops.append(irecv(buffer, src=r))
    
    # 启动所有发送
    if len(send_plan[r]) > 0:
        for batch in batched(send_plan[r], M):
            send_data = gather_local_chunks(batch)
            send_ops.append(isend(send_data, dst=r))

# 等所有完成
wait_all(recv_ops + send_ops)

问题：所有接收 buffer 要提前分配 → 显存可能很大。

解法 2: 轮次调度 (更安全) ⭐⭐ (推荐)
──────────────────────────────────────

把所有 P 个 rank 的通信分成 P-1 轮，
每轮每个 rank 恰好和一个 partner 通信。
这实际上就是一个 **通信调度表**。

经典做法：Round-Robin Tournament (循环赛调度)

  P=8 时的调度表 (每轮每个 rank 和谁配对):
  Round 0: (0,1) (2,7) (3,6) (4,5)
  Round 1: (0,2) (1,3) (4,7) (5,6)
  Round 2: (0,3) (1,4) (2,5) (6,7)
  Round 3: (0,4) (1,5) (2,6) (3,7)
  Round 4: (0,5) (1,6) (2,7) (3,4)  [注意 (2,7) 第二次出现——实际不会，标准算法保证不重复]
  Round 5: (0,6) (1,7) (2,4) (3,5)
  Round 6: (0,7) (1,2) (3,4) (5,6)

  每轮每个 rank 恰好和一个 partner 交换数据，
  P-1 轮后所有 pair 都覆盖到。

  ⭐ 关键：每轮只需要 1 个 partner 的 buffer → 显存固定!

  但这要求所有 rank 在同一轮做同一件事 (全局同步)，
  类似 ring 但更灵活（不是固定环形，而是最优配对）。

  每轮的操作：
    partner = schedule[round][my_rank]
    
    # 双向交换: 我给 partner 发它需要的 chunk，它给我发我需要的
    chunks_to_send = send_plan[partner]  # 可能为空
    chunks_to_recv = recv_plan[partner]  # 可能为空
    
    if both empty:
        # 这轮 skip! 双方都知道（因为 bitmap 已经交换过了）
        barrier()  # 同步，让其他 pair 完成
        continue
    
    # 非对称 sendrecv: 发的数量和收的数量可以不同
    sendrecv_chunks(partner, chunks_to_send, chunks_to_recv, buffer)
    
    # HSA sparse compute
    if len(chunks_to_recv) > 0:
        hsa_o_partial = compute_hsa(buffer, ...)
        hsa_o_accum += hsa_o_partial

  优势：
  - 无死锁（每轮只有一对，用 sendrecv 原子操作）
  - 显存固定：buffer 大小 = max(recv_per_round) * S * h * d
  - 可以 skip 不需要的 round
  - 负载均衡（zigzag 保证）

解法 3: 无同步调度 (最高效但最复杂)
───────────────────────────────────

每个 rank 完全独立调度自己的 send/recv。
用 tag 区分不同的 chunk batch。不需要全局同步。

  # 用线程/stream 分离 send 和 recv
  # Send stream:
  for r, chunks in send_plan.items():
      for batch in batched(chunks, M):
          data = gather_local(batch)
          isend(data, dst=r, tag=make_tag(my_rank, r, batch_id))
  
  # Recv stream + compute:
  for r, chunks in recv_plan.items():
      for batch in batched(chunks, M):
          irecv(buffer, src=r, tag=make_tag(r, my_rank, batch_id))
          wait()
          compute_hsa(buffer)
          # buffer 复用

  不需要全局同步，不需要 barrier。
  但需要仔细管理 tag 以避免消息混淆。


=====================================================================
Part 5: 最终推荐 — 轮次调度 + Double Buffer ⭐⭐⭐
=====================================================================

选择解法 2（轮次调度），结合 double buffer 实现通信-计算 overlap。

每轮最多需要和 1 个 partner 交换 chunk。
用 2 个 buffer，一个在接收当前轮的数据，一个在计算上一轮的数据。
"""

import torch
import torch.distributed as dist
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass


# ==============================================
# 工具函数
# ==============================================

def zigzag_chunk_indices(rank: int, world_size: int, num_chunks: int) -> List[int]:
    """返回 rank 在 zigzag 分片下拥有的全局 chunk indices。"""
    indices = []
    for stripe_start in range(0, num_chunks, 2 * world_size):
        fwd_idx = stripe_start + rank
        if fwd_idx < num_chunks:
            indices.append(fwd_idx)
        bwd_idx = stripe_start + 2 * world_size - 1 - rank
        if bwd_idx < num_chunks:
            indices.append(bwd_idx)
    return indices


def build_chunk_owner_map(world_size: int, num_chunks: int, device) -> torch.Tensor:
    """chunk_owner[g] = 拥有全局 chunk g 的 rank。"""
    chunk_owner = torch.empty(num_chunks, dtype=torch.int32, device=device)
    for r in range(world_size):
        for g in zigzag_chunk_indices(r, world_size, num_chunks):
            chunk_owner[g] = r
    return chunk_owner


def build_global_to_local_map(chunk_map: List[int], num_chunks: int, device) -> torch.Tensor:
    """g2l[global_chunk_idx] = local_chunk_idx (在本 rank 内的位置), -1 if not owned."""
    g2l = torch.full((num_chunks,), -1, dtype=torch.int32, device=device)
    for local_i, global_g in enumerate(chunk_map):
        g2l[global_g] = local_i
    return g2l


def round_robin_schedule(world_size: int) -> List[List[int]]:
    """
    生成 Round-Robin Tournament 调度表。
    返回 schedule: list of P-1 轮, 每轮 schedule[round][rank] = partner_rank。
    如果 P 为奇数，会有 bye (partner=-1)。
    
    经典算法：
    固定 rank 0，其他 rank 1..P-1 轮转。
    每轮 rank 0 和 "当前轮转位置" 配对，其余两两对称配对。
    """
    P = world_size
    if P % 2 == 1:
        # 奇数: 加一个虚拟 rank
        return _round_robin_odd(P)
    
    schedule = []
    # rank 0 固定, rank 1..P-1 轮转
    circle = list(range(1, P))  # [1, 2, ..., P-1]
    
    for round_i in range(P - 1):
        pairing = [-1] * P
        # rank 0 与 circle[0] 配对
        pairing[0] = circle[0]
        pairing[circle[0]] = 0
        
        # 其余两两配对: circle[1] <-> circle[-1], circle[2] <-> circle[-2], ...
        half = (P - 1) // 2
        for j in range(1, half + 1):
            a = circle[j]
            b = circle[P - 1 - j]
            pairing[a] = b
            pairing[b] = a
        
        schedule.append(pairing)
        # 轮转 circle: 最后一个插到第一个后面
        circle = [circle[-1]] + circle[:-1]
    
    return schedule


def _round_robin_odd(P):
    """P 为奇数时的处理，加一个虚拟节点。"""
    schedule_even = round_robin_schedule(P + 1)
    schedule = []
    for pairing in schedule_even:
        new_pairing = [-1] * P  # -1 = bye
        for r in range(P):
            partner = pairing[r]
            if partner < P:
                new_pairing[r] = partner
            # else: partner == P (虚拟节点) → bye
        schedule.append(new_pairing)
    return schedule


# ==============================================
# 通信计划
# ==============================================

@dataclass
class CommPlan:
    """每个 rank 的通信计划。"""
    # recv_plan[r] = 从 rank r 接收的全局 chunk indices (sorted)
    recv_plan: Dict[int, torch.Tensor]
    # send_plan[r] = 发给 rank r 的本地 chunk indices (local idx, sorted)
    send_plan: Dict[int, torch.Tensor]
    # 总共需要接收的 unique remote chunk 数
    total_recv_chunks: int
    # 总共需要发送的 chunk 数
    total_send_chunks: int


def build_comm_plan(
    indices: torch.Tensor,        # (B, L_local, h, K) 全局 chunk indices
    chunk_owner: torch.Tensor,    # (N,) chunk_owner[g] = rank
    g2l_local: torch.Tensor,      # (N,) global → local idx for my rank
    chunk_map_all: List[List[int]],  # chunk_map_all[r] = rank r 的全局 chunk list
    rank: int,
    world_size: int,
    device,
) -> CommPlan:
    """
    从 topk indices 构建 chunk-granular 通信计划。
    
    使用 Bitmap AlltoAll 方式: 
    每个 rank 广播自己需要的 chunk bitmap，
    然后每个 rank 从 bitmap 中提取自己需要发送的 chunk。
    """
    P = world_size
    N = chunk_owner.shape[0]
    N_local = N // P
    
    # Step 1: 找出本 rank 需要的所有远程 chunk (去重)
    flat_idx = indices.reshape(-1)  # 展平所有 topk indices
    valid_mask = (flat_idx >= 0)    # 过滤无效 index
    flat_idx_valid = flat_idx[valid_mask]
    
    owners = chunk_owner[flat_idx_valid.long()]
    remote_mask = (owners != rank)
    remote_global_idx = flat_idx_valid[remote_mask]
    unique_remote = remote_global_idx.unique()  # 去重
    
    # Step 2: 按 owner rank 分组
    recv_plan = {}
    for r in range(P):
        if r == rank:
            continue
        mask_r = (chunk_owner[unique_remote.long()] == r)
        chunks_r = unique_remote[mask_r]
        if chunks_r.numel() > 0:
            recv_plan[r] = chunks_r.sort()[0]  # 排序保证两端顺序一致
    
    # Step 3: Bitmap AlltoAll 交换请求
    # 构建 send bitmap: 对于每个 remote rank r, 
    # 我需要它的哪些 chunk (用该 chunk 在 rank r 本地的 local index 表示)
    send_plan = {}  # 这将在 alltoall_bitmap 之后填充
    
    # 构建本 rank 的 request bitmap
    # request_bitmap[r] = int tensor (N_local,) 表示需要 rank r 的第几个本地 chunk
    request_bitmaps = {}
    for r, chunks_r in recv_plan.items():
        bm = torch.zeros(N_local, dtype=torch.int32, device=device)
        # 将全局 chunk idx 转换为 rank r 的本地 idx
        chunk_map_r = chunk_map_all[r]
        g2l_r = {}
        for local_i, global_g in enumerate(chunk_map_r):
            g2l_r[global_g] = local_i
        for g in chunks_r.tolist():
            local_i = g2l_r.get(int(g), -1)
            if local_i >= 0:
                bm[local_i] = 1
        request_bitmaps[r] = bm
    
    # AlltoAll 交换 bitmap
    # 每个 rank 发送: request_bitmaps[r] → rank r  (N_local ints each)
    # 每个 rank 接收: response_bitmaps[r] ← rank r (N_local ints each)
    # 含义: response_bitmaps[r][j] = 1 表示 rank r 需要我的第 j 个本地 chunk
    
    send_buf = torch.zeros(P * N_local, dtype=torch.int32, device=device)
    recv_buf = torch.zeros(P * N_local, dtype=torch.int32, device=device)
    
    for r, bm in request_bitmaps.items():
        send_buf[r * N_local : (r + 1) * N_local] = bm
    
    dist.all_to_all_single(recv_buf, send_buf)
    
    # 解析 recv_buf → send_plan
    for r in range(P):
        if r == rank:
            continue
        response_bm = recv_buf[r * N_local : (r + 1) * N_local]
        local_indices_to_send = response_bm.nonzero(as_tuple=False).squeeze(-1)
        if local_indices_to_send.numel() > 0:
            send_plan[r] = local_indices_to_send
    
    total_recv = sum(c.numel() for c in recv_plan.values())
    total_send = sum(c.numel() for c in send_plan.values())
    
    return CommPlan(
        recv_plan=recv_plan,
        send_plan=send_plan,
        total_recv_chunks=total_recv,
        total_send_chunks=total_send,
    )


# ==============================================
# 主函数: Chunk-Granular P2P LHSA Forward
# ==============================================

def chunk_granular_p2p_lhsa_forward(
    lhsa_layer,
    hidden_states: torch.Tensor,    # (B, L_contig, D) — 连续布局, L/P tokens
    position_embeddings,
    cp_group,
    global_seq_len: int,
    max_chunks_inflight: int = 64,   # 显存预算: 同时在 buffer 中的最大 chunk 数
):
    """
    Zigzag + Chunk-Granular P2P LHSA Forward (v4)
    
    核心优势 vs v3 (Standard Ring):
      - 通信量: O(U_total * S) vs O(L)    (U_total << L/S)
      - 计算量: O(L/P * K * S) (与 v3 相同)
      - 显存: O(L/P + M*S)  有界 ✅
      - 负载均衡: ✅ (zigzag)
    """
    rank = dist.get_rank(cp_group)
    P = dist.get_world_size(cp_group)
    B, L_contig, D = hidden_states.shape
    S = lhsa_layer.chunk_size
    W = lhsa_layer.hsa_sliding_window
    K = lhsa_layer.topk
    L = global_seq_len
    N = L // S
    N_local = N // P
    M = max_chunks_inflight
    device = hidden_states.device
    h_kv = lhsa_layer.h_hsa_kv
    h_q = lhsa_layer.hsa_heads
    d = lhsa_layer.head_dim
    
    # ================================================================
    # Phase 0: 线性投影 + SWA (连续布局下)
    # ================================================================
    # ... (与 v3 相同, 省略投影和 SWA 代码)
    # 得到: swa_o, lse_sum, hsa_q_norm, hsa_k_norm, hsa_v, lmk_q_norm
    
    # ================================================================
    # Phase 1: Zigzag Scatter
    # ================================================================
    chunk_map_local = zigzag_chunk_indices(rank, P, N)
    chunk_owner = build_chunk_owner_map(P, N, device)
    g2l_local = build_global_to_local_map(chunk_map_local, N, device)
    
    # AlltoAll scatter → zigzag 布局
    # hsa_q_zz, hsa_k_zz, hsa_v_zz, lmk_q_zz, swa_o_zz, lse_zz = ...
    
    # ================================================================
    # Phase 2: AllGather Landmark Keys + TopK
    # ================================================================
    # lmk_k_local = hsa_k_zz[:, S-1::S, :, :]  # (B, N_local, h_kv, d)
    # lmk_k_all = allgather + reorder → (B, N, h_kv, d)
    # indices, scores = topk(lmk_q_zz, lmk_k_all, K, ...)
    # indices: (B, N_local*S, h_kv, K) — 全局 chunk idx
    
    # ================================================================
    # Phase 3: 构建通信计划
    # ================================================================
    chunk_map_all = []
    for r in range(P):
        chunk_map_all.append(zigzag_chunk_indices(r, P, N))
    
    comm_plan = build_comm_plan(
        indices=indices,  # (B, N_local*S, h_kv, K)
        chunk_owner=chunk_owner,
        g2l_local=g2l_local,
        chunk_map_all=chunk_map_all,
        rank=rank,
        world_size=P,
        device=device,
    )
    
    # ================================================================
    # Phase 4: 本地 HSA + Chunk-Granular P2P + 流式远程 HSA
    # ================================================================
    
    # --- 4a: 本地 chunk HSA ---
    local_indices = _remap_to_local(indices, chunk_owner, g2l_local, rank)
    # local_indices 中非本地的位置全部为 -1
    
    hsa_o_accum = lhsa_layer.hsa_func(
        hsa_q_zz, hsa_k_zz, hsa_v_zz,
        weights=chunk_weights, indices=local_indices,
        block_size=S, mask_last_token=True,
        is_training=lhsa_layer.training,
    )
    
    # --- 4b: 按轮次调度进行远程 chunk 通信 + 计算 ---
    schedule = round_robin_schedule(P)
    
    # 预分配 double buffer
    buf_k = [
        torch.empty(B, M * S, h_kv, d, dtype=hsa_k_zz.dtype, device=device),
        torch.empty(B, M * S, h_kv, d, dtype=hsa_k_zz.dtype, device=device),
    ]
    buf_v = [
        torch.empty(B, M * S, h_kv, d, dtype=hsa_v_zz.dtype, device=device),
        torch.empty(B, M * S, h_kv, d, dtype=hsa_v_zz.dtype, device=device),
    ]
    
    # 按轮次处理
    pending_compute = None  # (buf_idx, remote_indices_for_hsa)
    
    for round_i, pairing in enumerate(schedule):
        partner = pairing[rank]
        
        if partner == -1:
            # bye round (只在 P 为奇数时)
            if pending_compute is not None:
                _do_pending_compute()
            continue
        
        # 检查本轮是否有实际数据交换
        need_recv = partner in comm_plan.recv_plan
        need_send = partner in comm_plan.send_plan
        
        buf_idx = round_i % 2  # double buffer 索引
        
        if need_recv or need_send:
            # --- 准备 send 数据 ---
            if need_send:
                send_local_idx = comm_plan.send_plan[partner]
                num_send = send_local_idx.numel()
                # 从本地 KV 中 gather 需要发送的 chunk
                # send_local_idx 是本地 chunk index
                send_k = _gather_chunks(hsa_k_zz, send_local_idx, S)  # (B, num_send*S, h, d)
                send_v = _gather_chunks(hsa_v_zz, send_local_idx, S)
            
            if need_recv:
                recv_global_idx = comm_plan.recv_plan[partner]
                num_recv = recv_global_idx.numel()
                recv_k = buf_k[buf_idx][:, :num_recv * S]
                recv_v = buf_v[buf_idx][:, :num_recv * S]
            
            # --- 异步 sendrecv ---
            ops = []
            if need_recv:
                ops.append(dist.irecv(recv_k, src=partner, group=cp_group,
                                       tag=_make_tag(partner, rank, round_i)))
                ops.append(dist.irecv(recv_v, src=partner, group=cp_group,
                                       tag=_make_tag(partner, rank, round_i) + 1))
            if need_send:
                ops.append(dist.isend(send_k, dst=partner, group=cp_group,
                                       tag=_make_tag(rank, partner, round_i)))
                ops.append(dist.isend(send_v, dst=partner, group=cp_group,
                                       tag=_make_tag(rank, partner, round_i) + 1))
            
            # --- 在等待通信的同时，处理上一轮的 pending compute ---
            if pending_compute is not None:
                prev_buf_idx, prev_indices, prev_g2l = pending_compute
                remote_indices = _remap_to_buffer(indices, prev_g2l)
                hsa_o_partial = lhsa_layer.hsa_func(
                    hsa_q_zz,
                    buf_k[prev_buf_idx][:, :prev_indices.numel() * S],
                    buf_v[prev_buf_idx][:, :prev_indices.numel() * S],
                    weights=chunk_weights, indices=remote_indices,
                    block_size=S, mask_last_token=True,
                    is_training=lhsa_layer.training,
                )
                hsa_o_accum += hsa_o_partial
                pending_compute = None
            
            # --- 等待通信完成 ---
            for op in ops:
                op.wait()
            
            # --- 把本轮接收的数据记为 pending compute ---
            if need_recv:
                # 构建 global_chunk_idx → buffer local idx 映射
                recv_g2l = {}
                for i, g in enumerate(recv_global_idx.tolist()):
                    recv_g2l[int(g)] = i
                pending_compute = (buf_idx, recv_global_idx, recv_g2l)
        
        else:
            # 本轮不需要和 partner 交换 → 直接处理 pending
            if pending_compute is not None:
                prev_buf_idx, prev_indices, prev_g2l = pending_compute
                remote_indices = _remap_to_buffer(indices, prev_g2l)
                hsa_o_partial = lhsa_layer.hsa_func(
                    hsa_q_zz,
                    buf_k[prev_buf_idx][:, :prev_indices.numel() * S],
                    buf_v[prev_buf_idx][:, :prev_indices.numel() * S],
                    weights=chunk_weights, indices=remote_indices,
                    block_size=S, mask_last_token=True,
                    is_training=lhsa_layer.training,
                )
                hsa_o_accum += hsa_o_partial
                pending_compute = None
    
    # 处理最后一轮的 pending compute
    if pending_compute is not None:
        prev_buf_idx, prev_indices, prev_g2l = pending_compute
        remote_indices = _remap_to_buffer(indices, prev_g2l)
        hsa_o_partial = lhsa_layer.hsa_func(
            hsa_q_zz,
            buf_k[prev_buf_idx][:, :prev_indices.numel() * S],
            buf_v[prev_buf_idx][:, :prev_indices.numel() * S],
            weights=chunk_weights, indices=remote_indices,
            block_size=S, mask_last_token=True,
            is_training=lhsa_layer.training,
        )
        hsa_o_accum += hsa_o_partial
    
    # ================================================================
    # Phase 5: Score Fusion
    # ================================================================
    cat_scores = torch.cat([scores, lse_zz.unsqueeze(-1)], dim=-1)
    chunk_weights_full = torch.softmax(cat_scores, dim=-1)
    swa_weight = chunk_weights_full[:, :, :, -1]
    swa_weight_expanded = swa_weight.repeat_interleave(
        lhsa_layer.hsa_qk_ratio, dim=2
    )
    o_lower_zz = torch.addcmul(hsa_o_accum, swa_o_zz, swa_weight_expanded.unsqueeze(-1))
    
    # ================================================================
    # Phase 6: Zigzag Gather
    # ================================================================
    # o_lower = zigzag_gather(o_lower_zz, ...)
    # if hsa_denom > 1: o = cat(o_upper, o_lower)
    # return lhsa_layer.o_proj(o.reshape(B, L_contig, -1)), None


# ==============================================
# 辅助函数
# ==============================================

def _remap_to_local(indices, chunk_owner, g2l_local, rank):
    """
    将 topk indices 中属于本 rank 的 chunk 映射到本地索引，其余设为 -1。
    
    indices: (B, L_local, h, K) 全局 chunk idx
    返回:    (B, L_local, h, K) 本地 chunk idx, 非本地为 -1
    """
    owners = chunk_owner[indices.long()]
    is_local = (owners == rank)
    local_idx = g2l_local[indices.long()]
    local_idx[~is_local] = -1
    return local_idx


def _remap_to_buffer(indices, g2l_dict):
    """
    将 topk indices 中属于当前 buffer 的 chunk 映射到 buffer 内的索引，其余为 -1。
    
    indices: (B, L_local, h, K)
    g2l_dict: {global_chunk_idx: buffer_local_idx}
    返回: (B, L_local, h, K)
    """
    result = torch.full_like(indices, -1)
    for g, l in g2l_dict.items():
        result[indices == g] = l
    return result


def _remap_to_buffer_vectorized(indices, global_idx_tensor, num_chunks_global, device):
    """
    向量化版本的 _remap_to_buffer。
    
    indices: (B, L_local, h, K) 全局 chunk idx
    global_idx_tensor: (num_recv,) 本次接收的全局 chunk indices (已排序)
    返回: (B, L_local, h, K) buffer 内索引, 不在 buffer 中的为 -1
    
    用 lookup table 实现 O(1) 映射，避免 Python for 循环。
    """
    # 构建 global → buffer local 映射表
    g2l_table = torch.full((num_chunks_global,), -1, dtype=torch.int32, device=device)
    buffer_local_idx = torch.arange(global_idx_tensor.numel(), dtype=torch.int32, device=device)
    g2l_table[global_idx_tensor.long()] = buffer_local_idx
    
    # 向量化查找
    flat = indices.reshape(-1).long()
    valid = (flat >= 0) & (flat < num_chunks_global)
    result = torch.full_like(flat, -1, dtype=torch.int32)
    result[valid] = g2l_table[flat[valid]]
    return result.reshape(indices.shape)


def _gather_chunks(kv_tensor, local_chunk_indices, chunk_size):
    """
    从 KV tensor 中 gather 指定的本地 chunk。
    
    kv_tensor: (B, N_local * S, h, d) — zigzag 布局
    local_chunk_indices: (num_chunks,) — 要 gather 的本地 chunk 下标
    chunk_size: S
    返回: (B, num_chunks * S, h, d) — 连续排列
    """
    B, L_local, h, d = kv_tensor.shape
    S = chunk_size
    num_chunks = local_chunk_indices.numel()
    
    # 展开 chunk indices 为 token indices
    # local_chunk_indices[i] → token range [idx*S, (idx+1)*S)
    chunk_starts = local_chunk_indices * S  # (num_chunks,)
    offsets = torch.arange(S, device=kv_tensor.device)  # (S,)
    token_indices = (chunk_starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)  # (num_chunks*S,)
    
    # gather
    gathered = kv_tensor[:, token_indices.long(), :, :]  # (B, num_chunks*S, h, d)
    return gathered.contiguous()


def _make_tag(src, dst, round_i, base=1000):
    """生成唯一的通信 tag。"""
    return base * round_i + src * 100 + dst


"""
=====================================================================
Part 6: 通信量分析
=====================================================================

设 L=128K, S=64, K=8, P=8, h=8, d=128

全局 chunk 数 N = 128K / 64 = 2048
每 rank N_local = 256 chunk = 16K tokens

== 标准 Ring (v3) ==
  通信: (P-1) 步 × 2 × 16K × h × d × 2 bytes (bf16)
      = 7 × 2 × 16K × 8 × 128 × 2 = 7 × 2 × 256MB ≈ 3.5 GB

== Chunk-Granular P2P (v4) ==
  每个 query chunk 选 K=8 个远程 chunk（去掉本地的约 K*(P-1)/P ≈ 7 个）
  总远程 chunk 需求 (不去重): 256 × 7 = 1792
  去重后: unique remote chunks U_total
  
  U_total 的期望分析:
  每个 query chunk 的 topk 是从 ~2048 个 chunk 中选 8 个
  256 个 query chunk 各选 8 个，去重后:
  
  用 Coupon Collector 近似:
  如果选择均匀分布: U ≈ N * (1 - (1-K/N)^(N_local))
                       ≈ 2048 * (1 - (1 - 8/2048)^256)
                       ≈ 2048 * (1 - (0.9961)^256)
                       ≈ 2048 * (1 - 0.37)
                       ≈ 1290
  
  但实际 topk 不均匀（热门 chunk 被多次选中），U_total 更小。
  保守估计 U_total ≈ 500~1000。
  
  去掉本地 chunk (约 1/P = 12.5%): U_remote ≈ 440~875
  
  通信量: U_remote × S × h × d × 2 × 2 bytes (KV, bf16)
         = 700 × 64 × 8 × 128 × 4 = 700 × 256K ≈ 175 MB
  
  对比: 3.5 GB vs 175 MB → **约 20x 通信减少!** ⭐

  通信计划交换开销:
  Bitmap AlltoAll: P × N_local × 4 bytes = 8 × 256 × 4 = 8 KB (可忽略)


=====================================================================
Part 7: 显存分析
=====================================================================

== 标准 Ring (v3) ==
  本地 KV: 2 × 16K × h × d × 2 bytes = 256 MB
  Ring buffer: 2 × 256 MB = 512 MB  (double buffer)
  总: 768 MB

== Chunk-Granular P2P (v4) ==
  本地 KV: 256 MB (相同)
  Buffer pool: 2 × M × S × h × d × 2 bytes
             = 2 × 64 × 64 × 8 × 128 × 2 = 16 MB  (M=64 时)
  总: 272 MB (比 ring 还小!)

  M 的选择:
    M=32:   8 MB buffer → 最省显存
    M=64:   16 MB buffer → 适中
    M=128:  32 MB buffer → 更大批次，减少通信次数
    M=256:  64 MB buffer → 每轮最多传 256 chunk = 全量的 1/8

  显存上界 = 本地 KV + 2 × M × S × h × d
  完全可控，不依赖 topk 结果 ✅


=====================================================================
Part 8: 与 Ring 方案的对比总结
=====================================================================

| 维度           | v3: Zigzag+Ring+Sparse | v4: Zigzag+ChunkP2P ⭐ |
|----------------|------------------------|------------------------|
| 通信量         | O(L)                   | O(U*S) << O(L)        |
| 计算量         | O(L/P * K * S)         | O(L/P * K * S) (相同) |
| 显存上界       | O(L/P) (ring buf)      | O(L/P + M*S) (更小)   |
| 负载均衡       | ✅ zigzag              | ✅ zigzag              |
| 实现复杂度     | 中 (改 ring)            | 高 (通信调度)          |
| 可 skip 通信   | ✗ (ring 必须传)         | ✅ (只传需要的)        |
| 通信-计算 overlap | 受限 (sparse计算太快)  | ✅ (double buffer)     |
| 死锁风险       | 无 (ring 集体)          | 无 (轮次调度)          |


适用场景:
  - v3 适合: 通信带宽充足，计算是瓶颈的场景
  - v4 适合: 通信带宽有限（跨机/跨节点），通信是瓶颈的场景
  - v4 的额外好处: 显存更省，可以把省下的显存用来增大 batch 或模型

推荐: 先实现 v3 (简单), profile 后如果通信是瓶颈再升级到 v4。


=====================================================================
Part 9: 进一步优化方向
=====================================================================

1. 层间 KV 复用:
   多层 LHSA 可能检索相同的远程 chunk。
   方案: 在层间缓存已获取的远程 chunk KV，避免重复通信。
   
2. TopK 本地偏置:
   在 topk 检索的 score 上加一个小的本地偏置:
     score[g] += local_bias if chunk_owner[g] == my_rank
   鼓励优先选本地 chunk，减少远程通信，同时不显著影响质量。

3. Chunk 预取:
   利用 attention 的层间相关性，在第 l 层通信第 l+1 层可能需要的 chunk。
   pipeline: 第 l 层计算 + 第 l+1 层通信。

4. 压缩通信:
   远程 chunk 的 KV 可以用量化 (int8/fp8) 传输，接收端反量化。
   通信量再减 2-4x。

5. Adaptive M:
   根据运行时的 recv_plan 大小动态调整 M。
   如果某些 rank 需要很多远程 chunk，增大 M 减少通信轮次。
   如果大部分本地命中，减小 M 省显存。

6. NCCL Group Call 优化:
   把多个小 P2P 合并成一个 NCCL group call，减少 launch overhead。
   dist.batch_isend_irecv([...])

7. 轮次复用:
   当 recv_plan[partner] 的 chunk 数 > M 时，
   需要在同一个 partner 上做多批。
   可以把超出的部分延迟到后续空闲轮次处理。


=====================================================================
Part 10: Backward Pass 设计要点
=====================================================================

Forward 时需要保存:
  - topk indices, scores (用于 backward 中的 routing)
  - 通信计划 comm_plan (backward 时反向通信)
  - 本地 Q, K, V (标准)

Backward 的通信模式:
  Forward: rank A 从 rank B 接收 chunk KV → 做 attention → 得到 output
  Backward: rank A 有 d_output → 需要计算 dQ (本地), dK/dV (远程 chunk) 
            → 需要把 dK/dV 发回 rank B
  
  即: backward 的通信方向与 forward 反向!
  Forward recv_plan → Backward send_plan
  Forward send_plan → Backward recv_plan
  
  同样使用轮次调度，只是 send/recv 角色互换。

  计算:
  每轮收到 partner 的 dK/dV 后 reduce 到本地的 dK/dV 累加器。
  本地 dQ 直接累加（不需要通信）。

  显存: 需要额外保存 forward 时接收的远程 chunk KV (用于 backward 重计算)
  或: recomputation 策略 — backward 时重新通信获取远程 KV。
"""
