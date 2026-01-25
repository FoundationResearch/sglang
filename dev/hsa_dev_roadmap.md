# HSA (Hierarchical Sparse Attention) SGLang 开发路线图（修订版）

本文档规划在 SGLang 中实现 HSA（分层稀疏注意力）的开发路线与里程碑。修订目标是让路线与 SGLang 的现实架构对齐：**Paged KV（`req_to_token`/`kv_indices`）**、**Radix 前缀缓存**、**Continuous batching**、以及 SGLang 现有的 **AttentionBackend + Metadata** 集成范式（可对照 `--attention-backend nsa` 的实现与文档：`dev/nsa_in_sglang.md`）。

## 核心架构目标（对齐 SGLang 现实实现）

1. **Radix‑Aware Page Representation（\(E_i\)）**
   - **官方语义（推荐主线）**：\(E_i\) 由 **该 page 的 LMK token（最后一个 slot）的 K 表征**定义，而不是额外存一份 repr buffer。
     - FlashHSA 官方实现中：`lmk_id = vocab_size`，embedding/lm_head 扩到 `vocab_size+1`，并按 `(chunk_size-1)` 分组后在每组末尾插入 LMK。
   - **存储粒度**：\(E_i\) 与 **page_id（物理页）** 1:1 绑定（LMK slot 即 `page_id*page_size + page_size-1`）。  
   - **生命周期**：随 Radix cache 的“页对齐前缀”复用与释放；**partial page（未插入 LMK）不应参与 selection**，也不应被当作有定义的 \(E_i\)。

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
  - **约束**：`chunk_size == page_size`（与 SGLang paged allocator 对齐，且包含 LMK slot）。  
  - **新增硬约束（LMK）**：每个 page 最后一个 slot 固定为 LMK，故每页真实 token 数为 `page_size-1`。  
  - **注意**：SGLang 默认 `page_size=1`（见 `python/sglang/srt/server_args.py`），HSA/LMK 必须显式要求 `page_size >= 2`（例如 32/64）。
- [ ] **M0.2：定义 \(E_i\) 的产生时机与一致性规则**  
  - **FlashHSA 语义**：当一个 chunk 填满 `page_size-1` 个真实 token 时，插入 1 个 LMK token 并写入 KV；该 LMK token 在每一层的 K（可选加 `lmk_norm`）即 \(E_i\)。
  - **推理输出规则**：LMK token 只用于 KV/selection，不应被用户看到（不做采样、不输出）。  
  - **Radix/prefix 规则**：Radix prefix cache 必须在 “包含 LMK 的 token 序列坐标系” 下工作：prefix 对齐与复用时，LMK 也应被视为序列的一部分（否则 prefix 对齐会错位）。
  - 说明：Radix cache 在 `page_size>1` 时按页对齐缓存 keys；尾部 partial page 不进入 tree（对照 `python/sglang/srt/mem_cache/radix_cache.py` 的 `cache_protected_len` 注释）。
- [ ] **M0.3：定义 HSA Selection API（输出必须可直接驱动 paged kernel）**  
  - 输入：`Q`（per-token/per-request）、历史 \(E_i\)（按 page_id 索引）、窗口/SWA 相关辅助（可选）。
  - 输出（至少固定一种主线）：`selected_page_ids`（或等价的“可映射到 page_id 的 table”）+ `weights`。
  - 要求：输出语义与 SGLang 的 paged indices 链路一致（参考 NSA 的 `TopkTransformMethod.PAGED` 与 `page_table_1 -> real_page_table` 变换逻辑，见 `python/sglang/srt/layers/attention/nsa_backend.py`）。
  - **新增（LMK gather）**：明确 \(E_i\) 的获取方式：
    - `lmk_token_loc = page_id * page_size + (page_size-1)`；
    - selection 需要从 KV cache gather LMK 的 K（paged-friendly），并且只能对 “completed pages” 做。
- [x] **M0.4：集成形态选择**  
  - 选择 `AttentionBackend` 路径：新增 `hsa` backend（对齐 `python/sglang/srt/layers/attention/attention_registry.py` 的注册方式）。
  - 明确：混层策略（哪些层 HSA / 哪些层 dense/SWA），以及如何在 backend 内 dispatch。
  - **新增（运行时注入点）**：确定 LMK token 的插入由谁负责（推荐：SGLang runtime / scheduler 层做，不侵入模型权重逻辑）。

### 测试方案
- [ ] **Contract Test（小单测）**：在 `python/sglang/test/attention/` 新增 `test_hsa_contract.py` 验证：
  - token_loc <-> page_id 转换一致性；
  - partial page 不产生 \(E_i\)；  
  - radix prefix 命中时 \(E_i\) 可复用（同 page_id）。
  - **新增（LMK）**：验证 LMK 插入与“不可见输出”：
    - 每 `page_size-1` 个真实 token 插入 1 个 LMK；
    - LMK 不参与采样/不输出；
    - window_size 按包含 LMK 的长度计数（与你的约定一致）。

---

## Milestone 1：KV Cache 扩展（LMK 语义 + Radix 复用）

**目标**：让 \(E_i\) 成为“与 KV 同生命周期的缓存对象”，并且能被 radix prefix 复用。

### 任务列表
- [x] **M1.1（LMK 主线）：确保 KV cache 中存在 LMK slot，并可用于 selection**  
  - 修改：运行时 token 组织逻辑，使每页最后一个 token 为 LMK（并写 KV）。
  - 约束：只有 completed pages（LMK 已写入）才进入 selection 候选集。
- [ ] **M1.3：SWA/双池语义（如启用 sliding window）**
  - 明确：HSA 使用 full pool 还是 SWA pool；若混用，定义 repr 的映射/双份策略（参考 `SWAKVPool` 的 full→swa index mapping）。

### 测试方案
- [ ] **Unit Test（GPU-only）**：验证 LMK 语义与 completed-page gating
  - selection 只从 KV gather LMK 的 K 得到 \(E_i\)
  - 非 completed pages 必须被 mask（不能参与 top‑k）

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
  - **新增（LMK）**：明确 selection logits 计算使用的 \(E_i\) 是 “LMK-K”，需要从 KV cache gather（或从 LMK-K cache 读取）。

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

> 注意：在进入 M3 kernel 前，必须先闭环 LMK 的 runtime contract（插入/不可见输出/与 prefix cache 对齐），否则 kernel 很难验证正确性。

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
