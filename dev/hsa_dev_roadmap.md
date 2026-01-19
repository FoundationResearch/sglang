# HSA (Hierarchical Sparse Attention) SGLang 开发路线图（修订版）

本文档规划在 SGLang 中实现 HSA（分层稀疏注意力）的开发路线与里程碑。修订目标是让路线与 SGLang 的现实架构对齐：**Paged KV（`req_to_token`/`kv_indices`）**、**Radix 前缀缓存**、**Continuous batching**、以及 SGLang 现有的 **AttentionBackend + Metadata** 集成范式（可对照 `--attention-backend nsa` 的实现与文档：`dev/nsa_in_sglang.md`）。

## 核心架构目标（对齐 SGLang 现实实现）

1. **Radix‑Aware Page Representation（\(E_i\)）**
   - **存储粒度**：\(E_i\) 应与 **page_id（物理页）** 1:1 绑定，而不是与 token slot 绑定（避免 `page_size` 倍冗余）。
   - **生命周期**：随 Radix cache 的“页对齐前缀”复用与释放；不进入 radix tree 的尾部 partial page **不应产生/复用** \(E_i\)。

2. **Paged HSA Kernel（必须项）**
   - HSA kernel 必须支持 SGLang 的 paged 寻址语义：通过 **`kv_indptr/kv_indices`（token-level）或 page_id table（page-level）** 间接寻址 K/V，不能假设 `blk_idx * block_size` 连续布局（对照 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` 的 load 方式）。

3. **SGLang Native Integration 采用 AttentionBackend 范式**
   - 优先新增 `--attention-backend hsa`（类似 `nsa`），在 backend 内封装：metadata、selection（top‑k/weights）、paged HSA kernel 调用。
   - 尽量避免对模型结构做侵入式改造（例如强制引入 `HybridLlamaDecoderLayer`）；混层/交替层策略尽量通过 backend dispatch 或配置实现。

---

## Milestone 0：契约与集成策略冻结（Architecture Contract）

**目标**：先把最关键的“契约”写死，避免后续 kernel/缓存/调度返工。

### 任务列表
- [ ] **M0.1：确定 HSA 的 Chunk/Page 语义**
  - **约束**：`chunk_size == page_size`（与 SGLang paged allocator 对齐）。
  - **注意**：SGLang 默认 `page_size=1`（见 `python/sglang/srt/server_args.py`），HSA 必须显式要求/设置 `page_size`（例如 32/64）。
- [ ] **M0.2：定义 \(E_i\) 的产生时机与一致性规则**
  - 推荐：仅在“页完成”（`seq_len % page_size == 0`）时生成并写入 \(E_i\)（partial page 不生成/不复用）。
  - 说明：Radix cache 在 `page_size>1` 时按页对齐缓存 keys；尾部 partial page 不进入 tree（对照 `python/sglang/srt/mem_cache/radix_cache.py` 的 `cache_protected_len` 注释）。
- [ ] **M0.3：定义 HSA Selection API（输出必须可直接驱动 paged kernel）**
  - 输入：`Q`（per-token/per-request）、历史 \(E_i\)（按 page_id 索引）、窗口/SWA 相关辅助（可选）。
  - 输出（至少固定一种主线）：`selected_page_ids`（或等价的“可映射到 page_id 的 table”）+ `weights`。
  - 要求：输出语义与 SGLang 的 paged indices 链路一致（参考 NSA 的 `TopkTransformMethod.PAGED` 与 `page_table_1 -> real_page_table` 变换逻辑，见 `python/sglang/srt/layers/attention/nsa_backend.py`）。
- [ ] **M0.4：集成形态选择**
  - 选择 `AttentionBackend` 路径：新增 `hsa` backend（对齐 `python/sglang/srt/layers/attention/attention_registry.py` 的注册方式）。
  - 明确：混层策略（哪些层 HSA / 哪些层 dense/SWA），以及如何在 backend 内 dispatch。

### 测试方案
- [ ] **Contract Test（小单测）**：在 `python/sglang/test/attention/` 新增 `test_hsa_contract.py` 验证：
  - token_loc <-> page_id 转换一致性；
  - partial page 不产生 \(E_i\)；
  - radix prefix 命中时 \(E_i\) 可复用（同 page_id）。

---

## Milestone 1：KV Cache 扩展（per-page \(E_i\) buffer + Radix 复用）

**目标**：让 \(E_i\) 成为“与 KV 同生命周期的缓存对象”，并且能被 radix prefix 复用。

### 任务列表
- [ ] **M1.1：在 KV pool 中新增 per-page 表征缓冲区**
  - 修改：`python/sglang/srt/mem_cache/memory_pool.py`（KVCache 实现，例如 `MHATokenToKVPool` / `MLATokenToKVPool`）。
  - 设计：新增 `chunk_repr_buffer`（建议按 **page_id** 存储：shape `[num_pages, ...]`），并增加 `valid/version`（避免被回收页误复用）。
- [ ] **M1.2：定义读写接口（以 page_id 为主）**
  - `save_chunk_repr(page_ids, repr, *, layer_id)`
  - `get_chunk_repr(page_ids, *, layer_id) -> repr`
  - 可选：提供 `token_loc -> page_id` 的辅助函数（或约定由 backend 统一完成）。
- [ ] **M1.3：SWA/双池语义（如启用 sliding window）**
  - 明确：HSA 使用 full pool 还是 SWA pool；若混用，定义 repr 的映射/双份策略（参考 `SWAKVPool` 的 full→swa index mapping）。

### 测试方案
- [ ] **Unit Test**：`python/sglang/test/attention/test_hsa_kvpool_repr.py`
  - 写入 KV + \(E_i\)，prefix hit 后能复用 \(E_i\)；
  - evict / page reuse 后，旧 \(E_i\) 不应被错误读到（依赖 valid/version）。

---

## Milestone 2：Selection（Top‑K/Weights）实现（对齐 NSA 的 metadata+transform 思路）

**目标**：实现可 batch 化、可与 paged 索引语义对齐的 selection，并为 kernel 提供 page_id + weights。

### 任务列表
- [ ] **M2.1：实现 HSA 的 indexer/selector 逻辑**
  - 参考：`python/sglang/srt/layers/attention/nsa_backend.py` 的 metadata 组织、以及 `topk_transform()` 的 fused transform 思路。
  - 要求：支持 continuous batching（按 batch 的 `seq_lens`/`req_pool_indices` 组织）。
- [ ] **M2.2：实现 fused transform（强烈建议）**
  - 目标：把 top‑k + “token index -> page_id/page_table index”尽量融合，避免额外 gather/transform 开销。
- [ ] **M2.3：定义 selection 输出与 attention kernel 输入的精确 shape 契约**
  - 明确 group/head/softmax_head 三种策略中至少选一种作为主线，其他作为扩展。

### 测试方案
- [ ] **Correctness**：`python/sglang/test/attention/test_hsa_selection.py`
  - 小 batch + 小 KV（含 padding=-1），验证输出 page_id/indices 与参考实现一致。
- [ ] **Stability**：长上下文下 logits chunking 不改变结果。

---

## Milestone 3：Paged HSA Kernel（Decode 优先，先闭环）

**目标**：先把 decode（单 token/step）跑通 paged HSA，形成最小可用闭环；再扩展到 prefill/extend。

### 任务列表
- [ ] **M3.1：实现 paged HSA decode kernel**
  - 接口风格尽量贴合 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`：
    - 输入 `Q, K_Buffer, V_Buffer, kv_indptr/kv_indices`（或输入 `selected_page_ids` + `page_size` 并在 kernel 内展开页内 token）。
  - 关键：K/V load 必须通过 paged 索引语义实现（不能假设连续 KV）。
- [ ] **M3.2：prefill/extend kernel（可先用组合算子实现）**
  - 先 correctness，再性能；与 `extend_attention` 的 metadata/indices 组织对齐。

### 测试方案
- [ ] **Kernel Correctness**：`python/sglang/test/attention/test_hsa_paged_kernel.py`
  - 构造离散 page 的 KV（通过 `kv_indices` 间接寻址），验证读取与输出正确。

---

## Milestone 4：HSA AttentionBackend 集成（`--attention-backend hsa`）

**目标**：新增 `hsa` backend，复用 SGLang 的 ForwardBatch/req_to_token/kv_indices 链路，支持 continuous batching。

### 任务列表
- [ ] **M4.1：新增 backend 注册**
  - 修改：`python/sglang/srt/layers/attention/attention_registry.py` 添加 `@register_attention_backend("hsa")`。
- [ ] **M4.2：实现 `HSAAttnBackend`（metadata + dispatch）**
  - 新增：`python/sglang/srt/layers/attention/hsa_backend.py`（仿照 `nsa_backend.py` 的结构）。
  - 在 backend 内完成：构造 page_table/real_page_table、读取 \(E_i\)、selection、调用 paged HSA kernel。
- [ ] **M4.3：兼容性策略**
  - CUDA graph / speculative / sliding window：先定义支持矩阵，再逐步打通（NSA 在这些路径上有可借鉴实现）。

### 测试方案
- [ ] **Integration (Decode/Prefill)**：`python/sglang/test/attention/test_hsa_backend_decode.py`、`test_hsa_backend_prefill.py`
  - 与 dense attention 或参考实现对照；
  - prefix sharing 场景验证 \(E_i\) 只计算一次且可复用。

---

## Milestone 5：系统级能力与性能优化

### 任务列表
- [ ] **M5.1：选择/变换路径 profiling**
  - selection（q·\(E_i\) + top‑k/transform）是否被额外 gather/transform 限制。
- [ ] **M5.2：paged gather + intra-chunk attention profiling**
  - 随机 page 访问带宽是否成为瓶颈；必要时做更激进的融合/重排。
- [ ] **M5.3：Benchmark**
  - 长上下文总结、RAG/检索增强等场景：比较 dense vs HSA 的吞吐与延迟。
