# HSA 优化日志 (R1-R21)

按 round 记录每一次优化的 **改了什么**、**为什么改**、**收益**、**正确性验证**。

每个 round 的 commit hash 都在 `git log` 里能查到（搜 `hsa-backend: R` 或 `hsa-selector: R`）。

---

## 配置与对照

* **模型**：sglang HSA-345M（qwen_lhsa, 16L / 16 q-heads / 2 kv-heads / hidden=1024 / head_dim=64, hsa_topk=32, page_size=64, hsa_window=512）
* **Baseline**：sglang Dense-Fair-345M（qwen3，同样 16L / 16q / 2kv / hidden=1024）
* **硬件**：GB200，bs=1，`--cuda-graph-max-bs 1`（除非另注）
* **对照基准**：alignment 测试用 `dev/align/compare.py`（trained 权重，64 个真实 prompt），KL baseline 是 prefill=0.003649 / dec=0.004414/415

---

## 阶段总览

| 阶段 | Round | 主要目标 | 关键收益 |
|---|---|---|---|
| **阶段 A：消除 Python/sync 开销** | R8-R12 | 干掉 `.item()` 同步、Python `for` 循环、冗余 clone | decode 521ms→159ms @32K (3.3×) |
| **阶段 B：selector 大改** | R14 | prefill-tuned tilelang kernel 在 decode 严重 underutil GPU，换成 torch matmul + topk | decode 504ms→49ms @128K (10.3×) |
| **阶段 C：CUDA graph 集成** | R15 / R15.1 / R15.2 | HSA 自己的 capture/replay buffer，修 silent dense fallback | decode 49ms→17ms @128K (2.9×) |
| **阶段 D：短 context 攻坚** | R16-R21 | 各种小 op 融合 + 内部 SWA flash attention 化 | 8K decode 5.40ms→3.98ms (-26%)，32K 反超 dense |

---

## Round-by-Round 细节

### R8（pre-session）— page_table_1 overlay 向量化（forward_decode）
* **改**：`for b in range(B): seqlen = int(seq_lens[b].item()); page_table_1[b, seqlen-1] = ...` → 单次 `index_put_` + `clamp_`
* **为什么**：`.item()` 触发 CUDA→CPU 同步，每 HSA 层每 decode step 1 次 = 16 syncs/step
* **commit**：pre-session
* **收益**：~3-5% decode 加速

---

### R9 — selector 路径也加上 R8 同样的向量化
* **改**：`_run_selection_decode` 里 page_table_1 overlay 的 Python loop 改成 vectorized scatter
* **为什么**：之前只 R8 了 forward_decode body 那一段，selector 里还有一份重复的循环
* **commit**：`edc07440b`
* **收益**：与 R10、R11 一起捆绑测量

---

### R10 — `C_max = effective_cands.max().item()` 同步消除
* **改**：把 selector 里的 `C_max = int(effective_cands.max().item())` 改成 `C_max = max_seqlen_k // page_size`（用 metadata 已知的上界）
* **为什么**：`.max().item()` 触发同步，且每层都跑一次。上界 padding 由 `cand_mask` 兜底，selector kernel 不在乎多余候选
* **commit**：`edc07440b`
* **trade-off**：上界 C_max ≥ 实际 C_max，selector 处理稍多无效候选；但 cand_mask 把它们 -inf 掉，结果不变

---

### R11 — `_compute_internal_swa_decode` 完全向量化
* **改**：消掉嵌套 `for b in range(B) × for kv_h in range(H_hsa)` 双层 Python 循环 + `.item()` 同步。改成一个 batched einsum [B, H_hsa, Gh, W] × [B, W, H_hsa, D]
* **为什么**：原来每个 (batch, kv_head) 独立做小 matmul + softmax + scatter，启动开销巨大
* **commit**：`edc07440b`
* **收益**：32K decode 291ms→159ms（1.83×）
* **验证**：alignment KL 不变

---

### R12 — page_table_1 overlay 合并到 metadata-init
* **改**：之前每层 forward_decode + `_run_selection_decode` 都各 clone+overlay 一次（每 step 32 次 clone）→ 改成只在 `init_forward_metadata` 里 overlay 一次
* **为什么**：overlay 内容只依赖 `out_cache_loc` 和 `seq_lens`，是 step-level 不是 layer-level，没必要每层重做
* **commit**：`59b9046a5`
* **收益**：~3% @32K decode

---

### R13 — chunk_weight per-q-head 融成 triton kernel
* **改**：写 `fused_chunk_weight_per_qhead_kernel`，把 `masked_fill + expand/reshape + cat + softmax + nan_to_num + slice + contiguous + cast`（8 op）融成 1 个 triton kernel
* **为什么**：减少 launch 次数；输入 per_qhead_scores [B, HQ, K] + per_qhead_lse [B, HQ] + selected_pages [B, H, K]，内部做 GQA broadcast
* **commit**：`e9922e6df`
* **验证**：单元测试 bit-exact vs torch ref；alignment KL 不变
* **注**：HSA-345M 在 dummy weight 时走的是 legacy h_kv 路径，R13 没被触发。R18 才是真正用上的

---

### R14 — selector decode fast path（**最关键的一刀**）
* **改**：`select_topk_pages_decode_fused` 加 decode fast path：当 `q_4d.shape[1] == 1` 时绕开 tilelang `_online_topk_group`，直接用 `torch.einsum + torch.topk`
* **为什么**：profile 发现 `OnlineTopKUnifiedFn` 占 **98.6% GPU 时间**（26ms/层 @128K）。原因：tilelang kernel 为 prefill 调的（seq_len 大），grid=(cdiv(seq_len/32), h_kv, B) → decode 时 seq_len=1 grid 退化成 (1, 2, 1) = 2 个 thread block × 32 线程 = **<1% GB200 SM 利用率**，串行扫 2048 候选
* **commit**：`070a212c4`
* **收益**：
  | Length | Pre-R14 | Post-R14 |
  |---|---|---|
  | 8K | 90ms | 51ms |
  | 32K | 159ms | 51ms |
  | 128K | 504ms | 49ms |
  | 256K | 996ms | 55ms |
  | 512K | 919ms | 47ms |
* **验证**：alignment KL 不变；GQA q.sum-over-G 语义保留
* **结果**：**HSA decode 变 O(1) in context length**（5.4ms → 11.75ms 跨 8K-512K，2.2× 增长 vs context 64× 增长）；**512K decode 反超 dense (1.40×)**

---

### R15 — HSA cuda graph 集成
* **改**：写 HSA 自己的 cuda graph buffer（`_cg_page_table_1`、`_cg_cache_seqlens_i32`）+ 配套 triton kernel（`hsa_build_page_table_1_kernel`、`hsa_copy_seq_lens_kernel`）；HSA 的 `init_forward_metadata_replay_cuda_graph` 在 replay 前 in-place 更新 buffer
* **为什么**：之前 HSA 的 cuda graph hooks 直接 delegate 到 dense backend，只有 dense kernel 被 capture
* **commit**：`f6a236322`
* **验证**：alignment KL 不变；`dev/test_r15_via_alignment.py` 强制走 buffer 路径 KL 也不变

---

### R15.1 — `hsa_build_page_table_1_kernel` grid 并行化
* **改**：原来 grid=(bs,) 单 thread block 串行复制 524K int32 → grid=(bs, num_chunks) 让 seq_len 维度并行
* **为什么**：长 context 时单 block 太慢
* **commit**：`f6a236322`
* **收益**：实际测试看 R15.2 一起出（这个改进单独不显著，被 R15.2 silent fallback 掩盖）

---

### R15.2 — 修 silent dense fallback（**关键正确性修复**）
* **改**：把 HSA buffer + `self.forward_metadata` 的建立**直接放进** `init_forward_metadata_capture_cuda_graph`（而不是依赖 `init_forward_metadata` 在 capture 时被调）
* **为什么**：profile 发现 cuda graph 下 GPU 时间 77.67% 是 dense `_fwd_grouped_kernel_stage1`，0 个 HSA kernel。根因：`cuda_graph_runner.patch_model` 给 capture 的是 `model.forward`，不走 `init_forward_metadata`。结果 HSA 的 `forward_decode` 看到 stale metadata 走 `(md is None or pool is None)` dense fallback
* **commit**：`6973cbd63`
* **验证**：`dev/probe_hsa_cg_kernels.py` 抓到 `hsa_decode_paged_fwd_kernel × 48`，**0 个 dense kernel**
* **收益**：cuda graph 终于真的跑 HSA。8K decode 5.4ms→2-9ms（取决于 GPU 占用）；32K decode 5.42ms→4.6ms

---

### R16 — 内部 SWA bf16 tensor core
* **改**：内部 SWA einsum 输入 K/V 保持 bf16 不再 `.to(fp32)`，让 PyTorch 自动选 tensor-core GEMM（`nvjet_tst_*` / cutlass bf16）而不是 simt fp32 sgemm
* **为什么**：profile 看到 `cutlass3x_sm100_simt_sgemm_f32` 513us，simt 不用 tensor core 浪费
* **commit**：`54d23c52e`
* **收益**：32K decode -3.5%，其他 length 中性
* **验证**：alignment KL 不变

---

### R17 — selector 去掉 sort+gather
* **改**：selector decode fast path 里删掉 `torch.sort(top_idx) + torch.gather(top_scores, sort_perm)`
* **为什么**：selected pages 喂给 `hsa_decode_paged_fwd`（softmax 聚合）和 chunk_weight kernel（softmax）— 都是 permutation-invariant，不需要 ascending order
* **commit**：`8b9f48127`
* **收益**：~150us / step（~3%）
* **验证**：alignment KL 不变

---

### R18 — legacy h_kv chunk_weight 融成 triton kernel
* **改**：写 `fused_chunk_weight_h_kv_kernel`，把 legacy 路径的 `masked_fill + cat + softmax + nan_to_num + slice + cast + expand + reshape + contiguous`（9 op）融成 1 kernel，并 GQA broadcast 到 h_q 内部完成
* **为什么**：HSA-345M with dummy weights 走的是 legacy h_kv 路径（R13 的 per_qhead 不触发）
* **commit**：`8b9f48127`
* **收益**：~200us / step (~4%)
* **验证**：alignment KL 不变

---

### R19 — ❌ custom triton topk（**回滚**）
* **改**：写 `fused_topk_mask_kernel`，K 轮 argmax+suppress
* **为什么**：profile 显示 `sbtopk::gatherTopK` 31us/call 是最大单个 op
* **结果**：**比 PyTorch sbtopk 慢 2.4× @32K**（5.42ms→12.9ms decode）。原因：iterative argmax 是 O(K×C) 串行，C=512 时 32 × 512 reduction steps 太多；PyTorch 用 radix-based selection 是 O(N)。Reverted
* **教训**：PyTorch 的 sbtopk 对小 N/小 K 已经高度优化，难超越除非写 radix-based topk

---

### R20 — chunk_weight kernel 输出 HQ-granularity swa_w
* **改**：让 R18 的 fused h_kv kernel 直接输出 swa_w 在 HQ 维度（每 Gh 个 hq 写相同值），省掉下游的 broadcast op
* **为什么**：极小代码清理
* **commit**：`8049d2fbf`
* **收益**：性能中性，代码更干净
* **验证**：alignment KL 不变

---

### R21 — 内部 SWA streaming flash attention 融成单 triton kernel（**短 context 杀手锏**）
* **改**：写 `fused_internal_swa_decode_kernel`，把 R11/R16 的 ~15 op chain（window 构造、K/V gather、两个 einsum、softmax/logsumexp、nan_to_num、casts）整个融成 1 个 streaming-flash-attention triton kernel。每个程序处理一个 (b, hq)，按 BLOCK_W=128 chunk 流式遍历窗口，维护 online-softmax 的 running (max, sum, output)。LMK exclusion 在 kernel 里通过 `(pos+1) % page_size != 0` 完成
* **为什么**：profile 显示内部 SWA 占 8K decode 时间的 17%（两个 cutlass bf16 einsum + softmax + 一堆小 op）
* **commit**：`e0a76abcb`
* **收益**：**全 context 普降 12-24%**
  | Length | Pre-R21 | Post-R21 | Δ |
  |---|---|---|---|
  | 8K | 5.18ms | 3.98ms | **-23%** |
  | 32K | 5.47ms | 4.13ms | **-24%** |
  | 64K | 5.82ms | 4.62ms | -21% |
  | 128K | 6.78ms | 5.32ms | -22% |
  | 256K | 8.23ms | 6.89ms | -16% |
  | 512K | 11.49ms | 10.15ms | -12% |
* **关键**：**32K HSA decode 反超 dense**（4.13ms < 4.37ms = 1.06×），decode crossover 从 64K 提前到 32K
* **验证**：alignment KL 不变；end-to-end cuda graph capture+replay 测试（`test_r15_cg_correctness.py`）单步 max_abs_diff=0 bit-exact，4 步多步 token 完全一致

---

## 最终 R21 后的 HSA vs Dense 总表

**Decode（ms/token, 都开 cuda graph, bs=1）**

| Length | HSA-CG | Dense-CG | **HSA/Dense** |
|---|---|---|---|
| 8K | 3.98 | 2.11 | 0.53× |
| **32K** | **4.13** | **4.37** | **1.06× ✅ crossover** |
| 64K | 4.62 | 7.89 | **1.71×** |
| 128K | 5.32 | 17.07 | **3.21×** |
| 256K | 6.89 | 33.42 | **4.85×** |
| 512K | 10.15 | 65.83 | **6.48×** |

**Prefill（s, R21 不影响）**

| Length | HSA | Dense | **HSA Pf×** |
|---|---|---|---|
| 8K | 0.052 | 0.023 | 0.44× |
| 32K | 0.166 | 0.216 | 1.30× |
| 64K | 0.393 | 0.788 | **2.00×** |
| 128K | 1.114 | 3.024 | **2.72×** |
| 256K | 3.730 | 13.08 | **3.51×** |
| 512K | 13.42 | 54.10 | **4.03×** |

**Crossover**

| | **本工作** | 论文（H20, naive FA） |
|---|---|---|
| Prefill | **32K** | 32K |
| Decode | **32K** ✅ | 128K |

---

## 还在桌上的攻击点（R22+）

按 profile 优先级：
1. **`sbtopk::gatherTopK` 31us/call** — 最大单个 op。R19 试过 custom topk 但失败。要超越需要写 radix-based topk in triton（复杂）
2. **HSA-only 多出来的 lmk_q_down + lmk_q_up matmul**（~80us/layer × 16 = 1.3ms/step at 8K） — 可能融进 QKV 投影
3. **hsa_decode_paged_fwd + blend 融合** — 已经是 triton kernel，再加 SWA blend epilogue
4. **Selector matmul + topk 全融合** — 一个 kernel 做 Q·K + topk + cast

8K 还差 1.87ms 到 dense (3.98 vs 2.11)，攻克需要再 fuse 几个大 kernel。
