# HSA in SGLang：落地版实现 TODO（按代码结构 / 文件 / 类 / 接口）

本文档是 `dev/hsa_dev_roadmap.md` 的“工程落地版 TODO”。目标是把 HSA 以 **`AttentionBackend`** 的方式接入 SGLang（类似 `--attention-backend nsa`），并严格对齐 **Paged KV（`req_to_token` / `kv_indptr` / `kv_indices`）** 与 **Radix prefix cache** 的语义。

> 阅读顺序建议：先看 `dev/nsa_in_sglang.md` 和 `python/sglang/srt/layers/attention/nsa_backend.py`，理解 SGLang 的 metadata/indices 组织方式，再回到这里逐项落地。

---

## 0. 先冻结的契约（Contract）

- **Chunk/Page 语义**
  - **必须**：`chunk_size == page_size`。
  - **必须**：只对 “completed page”（`seq_len % page_size == 0`）生成/写入 \(E_i\)；partial page 不进入可复用集合。
- **\(E_i\) 存储粒度**
  - **必须**：按 **page_id（物理页）** 存，而不是按 token slot 存。
  - **必须**：提供 `valid/version` 机制避免页回收后误读旧值。
- **Selection 输出契约**
  - 至少固定一条主线（推荐）：输出能直接驱动 paged kernel 的 `selected_page_ids`/`page_table` + `weights`。
  - 与 NSA 对齐：最好支持 fused transform，把 top‑k 直接变成“可用于 gather 的 paged index 表”。

---

## 1. CLI / Registry：把 `hsa` 作为 AttentionBackend 接入

- **文件**：`python/sglang/srt/server_args.py`
  - **任务**：
    - 将 `"hsa"` 加入 `ATTENTION_BACKEND_CHOICES`（与 `"nsa"` 同级）。
    - 增加 HSA 相关参数（建议先最小集）：
      - `--hsa-topk`
      - `--hsa-page-size`（或复用全局 `--page-size` 并在运行时 assert）
      - `--hsa-selection-strategy {group,head,softmax_head}`
      - `--hsa-layers`（例如 `0,2,4,...` 或 range 语法；也可先做 “all layers”）
      - 可选：`--hsa-window-size` / `--hsa-enable-swa-fusion`

- **文件**：`python/sglang/srt/layers/attention/attention_registry.py`
  - **任务**：
    - 新增：
      - `@register_attention_backend("hsa")`
      - `def create_hsa_backend(runner): return HSAAttnBackend(runner)`

**状态**：✅ 已完成（CLI/Registry 已接入，可用 `--attention-backend hsa` 选择）

---

## 2. Backend：新增 `HSAAttnBackend`（核心调度点）

- **新增文件**：`python/sglang/srt/layers/attention/hsa_backend.py`
  - **新增类**：`class HSAAttnBackend(AttentionBackend)`
  - **需要实现的方法（对齐现有 backend 习惯）**
    - `__init__(self, model_runner: ModelRunner, ...)`
      - 解析：`page_size`、`num_head/num_kv_head/head_dim`、device、是否 sliding window。
      - 初始化：可复用的 GPU buffer（参考 `TritonAttnBackend` / `NativeSparseAttnBackend`）。
    - `init_forward_metadata(self, forward_batch: ForwardBatch)`
      - 读取：`forward_batch.req_pool_indices`、`forward_batch.seq_lens(_cpu)`、`forward_batch.req_to_token_pool.req_to_token`。
      - 生成：
        - `page_table_1`（token→slot 表，page_size=1 语义）
        - `real_page_table`（page_size>1 时的 page_id 表，参考 `NativeSparseAttnBackend._transform_table_1_to_real`）
        - “可用于 selection 的 chunk/page 列表”（例如按 stride 取样每页一个 token_loc，再 `//page_size` 得 page_id）
      - 缓存：将 metadata 存在 `self.forward_metadata`（类似 `TritonAttnBackend.forward_metadata`）。
    - `forward_decode(self, q, k, v, layer: RadixAttention, forward_batch: ForwardBatch, save_kv_cache=True, **kwargs)`
      - 如果 `save_kv_cache`：调用 `forward_batch.token_to_kv_pool.set_kv_buffer(...)` 写 KV（复用现有逻辑）。
      - 调用 selection（见第 4 节），得到 `selected_page_ids`/indices + `weights`。
      - 调用 paged HSA decode kernel（见第 5 节），输出 `o`。
    - `forward_prefill/forward_extend(...)`
      - 先 correctness：可以先走组合算子或 reference（低性能）闭环；
      - 再逐步替换成 paged HSA prefill kernel。
  - **新增数据结构（建议仿 NSA）**
    - `@dataclass class HSAMetadata:`
      - `page_size`
      - `cache_seqlens_int32 / cu_seqlens_k`
      - `page_table_1 / real_page_table`
      - 可选：`page_table_1_flattened`（prefix sharing 优化需要）
      - 可选：`token_to_batch_idx` / `indexer_k_start_end`（若 selection 需要 ragged 访问）
    - `@dataclass class HSAForwardMetadata:`（如需把 selection / kernel 输入缓存下来）

**状态**：🟡 已部分完成
- ✅ `HSAAttnBackend` 已存在，并且当前阶段 **delegate 到 dense（`TritonAttnBackend`）**，可跑通 end-to-end plumbing
- ✅ 已实现并缓存最小 `HSAMetadata`（包含 `page_table_1` / `real_page_table` 以及 `kv_indptr/kv_indices` 指针透传）
- ✅ 已加入 GPU-only smoke test：`python/sglang/test/attention/test_hsa_backend_gpu.py`（验证 `init_forward_metadata` 能跑、`forward_decode/extend` 能 delegate）
- ⏳ 未实现：真正的 HSA selection/top‑k/weights，未实现 paged HSA kernel

---

## 3. KV Pool：为 \(E_i\) 增加 per-page buffer（并与页回收安全协作）

- **文件**：`python/sglang/srt/mem_cache/memory_pool.py`
  - **任务**：在 KVCache 实现中为每层增加 repr buffer（建议先覆盖主流：`MHATokenToKVPool`；MLA 后补）
    - `MHATokenToKVPool`
      - 新增成员（按 layer 存 list 或单个大 tensor）：
        - `self.chunk_repr_buffer[layer_id]`：shape `[num_pages, repr_dim]` 或 `[num_pages, heads, repr_dim]`
        - `self.chunk_repr_version[layer_id]`：shape `[num_pages]`（int32/64）
        - `self.page_version`：shape `[num_pages]`（全局页版本；页被 allocator 重新分配时递增）
      - 新增接口：
        - `def save_chunk_repr(self, layer_id: int, page_ids: Tensor, repr: Tensor, page_version: Tensor | None = None)`
        - `def get_chunk_repr(self, layer_id: int, page_ids: Tensor, page_version: Tensor | None = None) -> Tensor`
        - `def get_page_version(self, page_ids: Tensor) -> Tensor`
    - 注意：allocator “free page” 不会清零内存；因此必须靠 version/valid 防止误复用。

- **文件**：`python/sglang/srt/mem_cache/allocator.py`
  - **任务（可选，但推荐）**：提供 “页版本”维护位置（两种方案择一）
    - **方案 A（allocator 持有页版本）**：每次 `alloc/alloc_extend/alloc_decode` 分配新页时递增版本并返回（需要扩展返回值或额外查询接口）。
    - **方案 B（KV pool 持有页版本）**：由 backend 在写 KV 或写 repr 时检测“本页是否首次写入/新分配”，并更新版本。

- **文件**：`python/sglang/srt/mem_cache/swa_memory_pool.py`
  - **任务**：若 HSA 需要 SWA pool 参与（混层/窗口），明确 repr buffer 在 full/swa 的放置与映射策略（参考 `SWAKVPool.translate_loc_from_full_to_swa`）。

**状态**：⏭️ 下一步建议优先做这里（Milestone 1 的实质）

---

## 4. Selection / Top‑K：从 \(q \cdot E_i\) 到 `selected_page_ids + weights`

> 目标：selection 必须是 “paged-friendly” 的输出，避免在 Python 侧做大量 gather/transform。

- **新增目录（建议）**：`python/sglang/srt/layers/attention/hsa/`
  - **新增文件**：`selector.py`
    - `class HSASelector:`（或函数集）
      - `def compute_logits(q, chunk_repr, *, strategy, ...) -> logits`
      - `def topk_transform(logits, topk, *, page_table_1/real_page_table, ...) -> selected`
    - **对齐 NSA 的思路**
      - 参考 `python/sglang/srt/layers/attention/nsa_backend.py`：
        - `TopkTransformMethod.PAGED`
        - `topk_transform()` 的 fused transform（避免额外 gather）

- **从 dev/hsa-kernel-main 迁移（可选路径）**
  - 将 `dev/hsa-kernel-main/ops/topk_*.py` 中可复用的 kernel 迁移/重写到：
    - `python/sglang/srt/layers/attention/hsa/kernels/topk_*.py`
  - 关键改造点：输入必须支持 `page_id`/paged 表，而不是假设 landmarks 连续。

---

## 5. Kernel：Paged HSA（decode 先闭环）

- **新增目录（建议）**：`python/sglang/srt/layers/attention/triton_ops/hsa/`
  - **新增文件**：`decode_hsa.py`
    - `def decode_hsa_fwd(Q, K_Buffer, V_Buffer, kv_indptr, kv_indices, selected_page_ids/indices, weights, ...) -> O`
    - **硬约束**：K/V load 必须通过 `kv_indices` 或 `page_id` 展开间接寻址，不能用 `blk_idx * block_size` 连续假设。
    - 接口风格尽量贴近 `triton_ops/decode_attention.py`，方便复用 buffer / cuda graph 经验。
  - **新增文件（后续）**：`prefill_hsa.py` / `extend_hsa.py`

- **从 dev/hsa-kernel-main 迁移**
  - `dev/hsa-kernel-main/ops/hsa_head_decode.py` / `hsa_head_prefill.py`
    - 现状偏向 “连续 KV + block_idx * block_size”；
    - 需要重写 load 逻辑为 paged（参考 `decode_attention.py` 的 `kv_loc` load）。

---

## 6. 测试：放在 SGLang 现有测试结构里（`python/sglang/test/attention/`）

- **新增测试文件（建议最小集）**
  - `python/sglang/test/attention/test_hsa_contract.py`
    - page_size/chunk_size 契约、partial page 规则、page_id 映射一致性
  - `python/sglang/test/attention/test_hsa_kvpool_repr.py`
    - repr 写入/读取、prefix hit 复用、page reuse 不误读（version/valid）
  - `python/sglang/test/attention/test_hsa_paged_kernel.py`
    - 构造离散 `kv_indices`，验证 kernel 读取正确
  - `python/sglang/test/attention/test_hsa_backend_decode.py`
    - 与 dense decode 对照（或 reference）验证 correctness

**当前已有测试**
- ✅ `python/sglang/test/attention/test_hsa_backend_gpu.py`（GPU-only，smoke：可跑 + delegate）

**仍缺的测试（建议按优先级）**
- **P0（下一步）**：`test_hsa_kvpool_repr.py`（GPU-only）
  - page_id keyed 的 repr buffer：写入/读取
  - “只在 completed page 写入”的规则（partial page 不写）
  - page reuse/version 机制：避免误读旧 repr
- **P1**：`test_hsa_backend_dense_integration.py`（GPU-only）
  - 不 monkeypatch dummy backend，走真实 `TritonAttnBackend` 路径，至少跑一次 decode forward（验证 wiring 在真实依赖栈下可跑）
- **P2**：CUDA graph / speculative / sliding window 的支持矩阵测试（先写 skip/xfail 也可以）

---

## 7. 性能与兼容性（上线前必须明确的支持矩阵）

- **CUDA graph**
  - 需要明确：HSA backend 是否支持 decode CUDA graph；若支持，哪些 buffer 需要 “static shapes + preallocated”。
- **Speculative decoding**
  - 需要明确：topk>1 与 page_size>1 的组合在某些后端存在不稳定性（SGLang 里已有相关 guard）；HSA 的支持策略要先写清。
- **Sliding window / SWA**
  - 需要明确：HSA 是否和 sliding window attention 共存；若共存，selection 的 mask 与 window 的融合方式要固定。
- **量化 KV（fp8/fp4）**
  - 先定义支持矩阵：HSA 是否需要先支持 fp16/bf16，再逐步扩展到量化缓存。


