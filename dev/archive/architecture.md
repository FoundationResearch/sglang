## HSA in SGLang：架构 / 接口总览（面向 Kernel）

这份文档是一张 **“先看接口、再写 kernel”** 的地图，用于把 Hierarchical Sparse Attention（HSA）以 **SGLang 原生方式**落地，重点对齐：
- **Paged KV Cache**（通过 `kv_indptr/kv_indices` 间接寻址）
- **Continuous batching**（`ForwardBatch` 合批语义）
- **Radix prefix cache**（KV 复用）
- **AttentionBackend 抽象**（CLI/registry 可切 backend）

当前代码状态（很重要）：
- **Step 4 selection**：已用 Torch reference 实现，**目前只写入 metadata（可观测/可测试）**。
- **attention compute**：仍然 delegate 到 `TritonAttnBackend`（dense/paged attention），**selection 结果还没被用于计算输出**。
- **下一步**：实现 **paged HSA decode kernel（Step 5）** 来消费 selection 的输出。

---

### 1) 入口与注册（CLI/Registry）

#### **CLI**
文件：`python/sglang/srt/server_args.py`
- `--attention-backend hsa`
- `HSAAttnBackend` 使用的 HSA 参数：
  - `--hsa-topk`：固定 K
  - `--hsa-selection-strategy`：目前支持 `group` | `head`（预留未来 `softmax_head`）
  - `--hsa-layers`：可选，仅对指定 decoder layer 走 HSA
  - `--hsa-window-size`：可选，SWA→HSA 模式的窗口大小
  - `--hsa-enable-swa-fusion`：未来要做 SWA/HSA 融合（目前 compute 尚未实现）

#### **Registry**
文件：`python/sglang/srt/layers/attention/attention_registry.py`
- `"hsa"` 映射到 `HSAAttnBackend`

---

### 2) 运行时对象与数据流（decode vs extend）

#### **`ForwardBatch`（系统层 batch 契约）**
文件：`python/sglang/srt/model_executor/forward_batch_info.py`
- **核心字段**
  - `batch_size: int`
  - `req_pool_indices: Tensor[int32]`，shape `[B]`（索引到 `req_to_token_pool`）
  - `seq_lens: Tensor[int32]`，shape `[B]`（每个请求当前 KV token 长度）
  - `seq_lens_cpu: Optional[Tensor]`
  - `out_cache_loc: Tensor[int64]`，shape `[num_new_tokens]`
    - decode：通常 `num_new_tokens == B`（每个请求 1 个新 token）
    - extend：把所有请求的新 tokens 拼在一起
- **仅 extend 才有（ragged 新 tokens）**
  - `extend_prefix_lens / extend_seq_lens / extend_start_loc`
- **KV/cache 句柄**
  - `req_to_token_pool: ReqToTokenPool`
    - `.req_to_token`：shape `[pool_size, max_context_len]`，int32（token→token_loc 表）
  - `token_to_kv_pool: MHATokenToKVPool`（paged KV 存储 + chunk/page repr 存储）

#### **Backend 骨架**
文件：`python/sglang/srt/layers/attention/hsa_backend.py`
- `HSAAttnBackend.init_forward_metadata(forward_batch)`
  - 先让 dense backend 初始化 metadata（保证当前 delegate compute 能跑）
  - 再构造 HSA metadata（`HSAMetadata`）
- `HSAAttnBackend.forward_decode(q, k, v, layer, forward_batch, save_kv_cache=True, ...)`
  - **当前**：跑 selection（Torch）→ 把结果写到 metadata → attention 计算 delegate 到 dense
  - **未来**：selection → **paged HSA decode kernel 消费 selection 输出**
- `HSAAttnBackend.forward_extend(...)`
  - **当前**：delegate 到 dense；同时包含“completed page 写 repr”的 hook
  - **未来**：selection/landmarks 更新，然后 paged HSA extend/prefill kernels

---

### 3) Paged KV 术语：token_loc / page_id / tables

#### **Token location（`token_loc`）**
在 SGLang 中，`token_loc` 是 KV pool 的全局槽位索引。对于 paged KV：
- `page_id = token_loc // page_size`
- `offset_in_page = token_loc % page_size`

#### **`page_table_1` 与 `real_page_table`**
存放在 `HSAMetadata`：
- `page_table_1`: shape `[B, max_seqlen_k]` int32
  - token 级映射：`page_table_1[b, t] = token_loc`
- `real_page_table`: shape `[B, ceil(max_seqlen_k / page_size)]` int32
  - page 级映射：`real_page_table[b, p] = page_id`

面向 kernel 的逻辑一般应该优先在 **page space** 做计算，再通过间接寻址映射回 KV。

---

### 4) `HSAMetadata`（backend 内部“输入缓存”）

文件：`python/sglang/srt/layers/attention/hsa/metadata.py`

#### **常驻字段**
- `page_size: int`
- `cache_seqlens_int32: Tensor[int32]` shape `[B]`
- `max_seqlen_k: int`
- `page_table_1: Tensor[int32]` shape `[B, max_seqlen_k]`
- `real_page_table: Tensor[int32]` shape `[B, ceil(max_seqlen_k/page_size)]`
- `kv_indptr / kv_indices: Optional[Tensor]`（从 dense backend metadata 透传）
- `window_kv_indptr / window_kv_indices: Optional[Tensor]`（sliding-window 变体）

#### **Selection debug 字段（Step 4）**
在 `forward_decode()` 中填充，用于可观测/单测：
- `hsa_cand_page_ids: Optional[Tensor[int32]]` shape `[B, C]` padded `-1`
- `hsa_cand_mask: Optional[Tensor[bool]]` shape `[B, C]`
- `hsa_selected_page_ids: Optional[Tensor[int32]]` shape `[B, H, K]` padded `-1`
- `hsa_selected_scores: Optional[Tensor[float32]]` shape `[B, H, K]`

重要说明：**这些 selection 输出目前还没有被用于 attention 的实际计算**，要等 Step 5 kernel 落地后才会消费。

---

### 5) KV pool 扩展：chunk/page repr \(E_i\)

文件：`python/sglang/srt/mem_cache/memory_pool.py`（`MHATokenToKVPool`）

#### **重要：我们切换到“LMK 作为真实 token”的官方语义（推荐主线）**

我们从 `dev/FlashHSA`（官方实现）确认到的做法是：
- **LMK 是一个真实 token id**：`lmk_id = config.vocab_size`（也就是在原 vocab 之后追加 1 个 token）。
- **模型侧扩容 embedding / lm_head**：embedding 与输出头都扩到 `vocab_size + 1`（或再做 padding 到 32 的倍数）。
- **每个 chunk/page 的最后一个位置固定放 LMK**：
  - 输入按 `(chunk_size - 1)` 分组，每组末尾 append `lmk_id`，使得每个 chunk 总长度为 `chunk_size`。
  - 这样 “每个 chunk 的 LMK token” 会像正常 token 一样过完每一层并写入 KV cache。
- **训练时不让 LMK 参与 loss**：labels 在 LMK 位置填 `-100`（见官方 `_insert_special_tokens(labels, -100)`）。
- **推理时 LMK 插入通常在模型外完成**（官方 `eval/eval_ppl.py` 在输入侧插入；模型 forward 本身不强制插入）。

因此对 SGLang 来说，\(E_i\) 的来源应以 **“每个 page 的 LMK token 的 K（或某个规范化后的版本）”** 为主，而不是额外维护一个独立的 per-page repr buffer。

> 这也解释了你说的 “\(q\cdot E_i\) 需要 gather”：\(E_i\) 在 KV cache 里，selection 需要按 `page_id -> lmk_token_loc -> K` 做一次 gather（paged-friendly）。

#### **SGLang 侧的 page/chunk 语义（与 LMK 对齐）**
- `page_size == chunk_size`（包含 LMK 的总 slot 数）
- **每个 page 最后 1 个 slot 永远留给 LMK**，所以真实文本 token 每页最多 `page_size - 1`
- **completed page**：只有当 LMK 已被插入并写入 KV 后，该 page 才“可被 HSA selection 使用”
Kernel（未来）假设：`E_page` 只对“LMK 已写入的 pages”存在；无效 pages 必须被 mask。

---

### 6) Selection（Step 4）：接口与语义

文件：`python/sglang/srt/layers/attention/hsa/selector.py`

#### **候选构造（candidates）**
- `build_active_page_candidates(page_table_1, seq_lens, page_size, window_size)`
  - **仅活跃 pages**：从 `page_table_1[b, :seq_len] // page_size` 得到 pages 并去重
  - **SWA→HSA（非融合）**：若 `window_size` 生效，则排除最后 `window_size` tokens 触达的 pages
输出：
- `cand_page_ids`: `[B, Cmax]` int32 padded `-1`
- `cand_mask`: `[B, Cmax]` bool

#### **Decode top‑k**
- `select_topk_pages_decode(q, cand_page_ids, cand_mask, cand_chunk_repr, cand_chunk_repr_valid, topk, selection_strategy, sm_scale=None)`
输入：
- `q`: `[B, HQ, D]` or `[B, HQ*D]`
- `cand_chunk_repr`: `[B, C, H, D]` where `H = num_kv_heads`
- `cand_chunk_repr_valid`: `[B, C]` bool (version-guard validity)
- `topk`: fixed K
- `selection_strategy`: `"group"` or `"head"`
输出：
- `selected_page_ids`: `[B, H, K]` int32 padded `-1`
- `selected_scores`: `[B, H, K]` float32

#### **LMK 语义下 selection 的关键变化（将要落地）**
- `cand_chunk_repr` 的来源改为：`page_id -> lmk_token_loc -> gather(K_lmk)`（而非独立 repr buffer 写入）。
- **completed pages only**：候选集必须排除 “还没插入 LMK 的 partial page”，否则 LMK slot 里是未定义旧值。
- window_size（SWA）按你说的：**按包含 LMK 的 token 长度** 计数；这与官方 `topk_head.py` 的 causal/window mask 形式一致（`tq`/`window_size`/`block_size=chunk_size` 都是在 LMK-included 的坐标系里）。

---

### 7) Kernel 设计目标（Step 5）：paged HSA decode

目标：实现一个 decode kernel，消费：
- 当前 token 的 `Q`（per request）
- 选中的 pages（per kv head）
- paged KV 的间接寻址（`kv_indptr/kv_indices`）
并产出 `O`，避免扫描全量 KV。

#### **建议的 kernel-facing 接口（decode）**
建议新增文件：`python/sglang/srt/layers/attention/triton_ops/hsa/decode_hsa.py`

概念签名：
- 输入：
  - `q`: `[B, HQ, D]` (or `[B, HQ*D]`)
  - `k_cache, v_cache`：来自 `token_to_kv_pool`（具体存储由 pool 决定）
  - `kv_indptr, kv_indices`：每个请求 ragged 的 KV token_locs（dense paged attention 的语义）
  - `selected_page_ids`：`[B, H, K]`（page space）
  - 可选：`selected_scores/weights`：`[B, H, K]`
  - `cache_seqlens_int32` (+ `page_size`)
- 输出：
  - `o`: `[B, HQ*D]`

#### **关键翻译问题（必须提前定清楚）**
Selection 给出的是 **page_id**，但 KV 读取依赖 **token_loc**（通常来自 `kv_indices` 或 `page_table_1`）。
你必须定义“page→token 的展开”发生在何处：
- **方案 A（kernel 内展开 pages）**：`page_id -> 本页 token_locs -> gather KV`
  - 实现更难，但 metadata 更小
- **方案 B（selection 输出 token_locs）**：提前展开成 `selected_token_locs`（packed 或 padded）
  - correctness 更快闭环，但 metadata 带宽更大

建议：第一版正确性优先时，通常 **方案 B 更快落地**。

#### **LMK 方案下的额外翻译点：page_id -> lmk_token_loc**
LMK 固定在 page 的最后一个 slot：
- `lmk_token_loc = page_id * page_size + (page_size - 1)`
这让 selection 获取 \(E_i\) 与 kernel 内做 “LMK gather” 都更直接（paged-friendly）。

---

### 8) SWA 交互模式（kernel 未来要支持的两种形态）

我们最终希望两种模式都支持：
- **模式 A：SWA→HSA**（selection 已支持：通过排除 window pages）
- **模式 B：融合 SWA/HSA**（未来）
  - 大概率需要 `softmax_head` 风格的数值稳定融合（原 repo 用 `lse_swa`）

---

### 9) Kernel bring-up 时推荐使用的测试

GPU-only 测试位于 `python/sglang/test/attention/`：
- `test_hsa_selector_decode_gpu.py`：selection 正确性（活跃 pages / window 排除 / version mask / fixed‑K）
- `test_hsa_backend_dense_integration_gpu.py`：真实 `TritonAttnBackend` 集成（compute delegate）+ repr hook

Step 5 kernel 推荐新增测试：
- tiny decode 等价性：HSA kernel vs dense（小规模、可控 selection）
- 边界场景：`seq_len < page_size`、page boundary、`K > #candidates`、invalid versions

