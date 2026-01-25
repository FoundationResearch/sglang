# HSA in SGLang：落地版实现 TODO（按代码结构 / 文件 / 类 / 接口）

本文档是 `dev/hsa_dev_roadmap.md` 的“工程落地版 TODO”。目标是把 HSA 以 **`AttentionBackend`** 的方式接入 SGLang（类似 `--attention-backend nsa`），并严格对齐 **Paged KV（`req_to_token` / `kv_indptr` / `kv_indices`）** 与 **Radix prefix cache** 的语义。

> 阅读顺序建议：先看 `dev/nsa_in_sglang.md` 和 `python/sglang/srt/layers/attention/nsa_backend.py`，理解 SGLang 的 metadata/indices 组织方式，再回到这里逐项落地。

---

## 0. 先冻结的契约（Contract）

- **Chunk/Page 语义**
  - **必须**：`chunk_size == page_size`。
  - **必须（LMK 主线）**：每个 page 最后一个 slot 固定为 LMK，故每页真实 token 上限为 `page_size - 1`。
  - **必须（LMK 主线）**：只对 “completed page”（LMK 已写入 KV）定义 \(E_i\) 并参与 selection；partial page 不进入可复用集合。
- **\(E_i\) 存储粒度**
  - **推荐主线（FlashHSA 语义）**：\(E_i\) 由该 page 的 **LMK token 的 K** 定义：
    - `lmk_token_loc = page_id * page_size + (page_size - 1)`
    - selection 从 KV cache gather LMK-K 得到 \(E_i\)（paged-friendly）
- **Selection 输出契约**
  - 至少固定一条主线（推荐）：输出能直接驱动 paged kernel 的 `selected_page_ids`/`page_table` + `weights`。
  - 与 NSA 对齐：最好支持 fused transform，把 top‑k 直接变成“可用于 gather 的 paged index 表”。
 - **LMK 不可见输出契约**
   - 推理时 LMK token 不应被采样/不应输出给用户（仅用于写 KV 与产生 \(E_i\)）。
   - `window_size`（SWA）按 **包含 LMK 的 token 长度** 计数（与你当前约定一致）。

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
      - `--hsa-lmk-id`：LMK token id（默认 -1 表示用 vocab_size；对齐 FlashHSA）

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

**状态**：🟡 已部分完成（Phase‑1 调度点已闭环；kernel 未落地）
- ✅ `HSAAttnBackend` 已存在，并且当前阶段 **delegate 到 dense（`TritonAttnBackend`）**，可跑通 end-to-end plumbing
- ✅ 已实现并缓存最小 `HSAMetadata`（包含 `page_table_1` / `real_page_table` 以及 `kv_indptr/kv_indices` 指针透传）
- ✅ 已实现 decode selection（Torch reference），并把结果写入 `HSAMetadata.hsa_*` debug 字段（compute 仍 delegate）
- ✅ 已加入 GPU-only smoke test：`python/sglang/test/attention/test_hsa_backend_gpu.py`
- ✅ 已加入 GPU-only 真实集成测试（不 monkeypatch）：`python/sglang/test/attention/test_hsa_backend_dense_integration_gpu.py`
- ⏳ 未实现：paged HSA kernel（仍然走 dense attention 输出）

---

## 3. KV / \(E_i\)：严格对齐 FlashHSA（LMK 真实 token）

- **文件**：`python/sglang/srt/mem_cache/memory_pool.py`
  - **约束**：不增加任何 per-page repr buffer。
  - **\(E_i\) 定义（FlashHSA）**：
    - 每个 page 最后一个 slot 为 LMK：`lmk_token_loc = page_id * page_size + (page_size - 1)`
    - 每层 selection 使用该层 KV cache 里的 `K[lmk_token_loc]` 作为 \(E_i\)
  - **安全性**：通过 “completed-page gating”（只允许 `page_id < floor(seq_len / page_size)`）避免读到未定义 KV。

- **文件**：`python/sglang/srt/mem_cache/allocator.py`
  - **任务（后续安全工作）**：把 allocator/radix 的 page reuse 生命周期与 HSA 的 completed-page 语义强绑定（生产级安全）。

- **文件**：`python/sglang/srt/mem_cache/swa_memory_pool.py`
  - **任务**：若 HSA 需要 SWA pool 参与（混层/窗口），明确 LMK 在 full/swa 的映射策略。

**状态**：✅ 已完成（遵循 FlashHSA；无 repr buffer）
- ✅ \(E_i\) 来自 KV cache 中 LMK token 的 K（按 layer 分别 gather）
- ✅ 通过 completed-page gating 避免读到未定义 KV

---

## 4. Selection / Top‑K：从 \(q \cdot E_i\) 到 `selected_page_ids + weights`

> 目标：selection 必须是 “paged-friendly” 的输出，避免在 Python 侧做大量 gather/transform。

- **本阶段实现策略（先 SWA→HSA，非融合；但保留未来融合的灵活性）**
  - **固定 K**：`--hsa-topk` 固定，输出不足时用 `-1` padding，并配套 mask / `-inf` score。
  - **候选集 = 本次 query 的活跃 pages**：仅对 `req_to_token[:seq_len]` 中出现过的 `page_id` 计算 \(q \cdot E_i\)。
  - **SWA→HSA 模式的 SWA 排除**：先用 SWA 覆盖“近邻窗口”，selection 只在窗口之外的 pages 上做 top‑k（等价于原 repo 的 “causal block mask” 思路）。
  - **completed-page gating**：selection 必须排除未完成 pages（LMK 未写入 KV 的 pages），避免读到未定义 KV。
  - **策略支持**：先实现 `group`/`head`（命名对齐 `dev/hsa-kernel-main`），未来再加 `softmax_head`（SWA/HSA 融合需要 `lse_swa`）。

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

**状态**：🟡 实现中（已闭环 decode 的 Torch reference selection + CUDA 单测）
- ✅ 新增：`python/sglang/srt/layers/attention/hsa/selector.py`
  - `build_active_page_candidates(...)`：活跃 pages + SWA window pages 排除
  - `select_topk_pages_decode(...)`：decode top‑k（固定 K；`group/head`）
- ✅ `HSAAttnBackend.forward_decode` 已运行 selection，并把结果写到 `HSAMetadata` 的 debug 字段（compute 仍 delegate dense）
- ✅ GPU-only 单测：`python/sglang/test/attention/test_hsa_selector_decode_gpu.py`

---

## 4.x LMK runtime 注入（不可见输出闭环）

- **目标**：让 LMK 成为 engine-visible token（写 KV、进 prefix/radix），但对用户不可见。
- **落地文件**
  - `python/sglang/srt/managers/scheduler.py`
    - 对 `--attention-backend hsa` 的请求启用 `Req.enable_hsa_lmk(page_size, lmk_id)`
  - `python/sglang/srt/managers/schedule_batch.py`
    - `Req` 支持：
      - prompt 插入：每 `page_size-1` 个 token 插入 1 个 LMK（FlashHSA 语义）
      - decode 强制：当 `len(fill_ids) % page_size == page_size-1` 时，下一步强制插入 LMK（不可见）
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
    - decode 输出处理：LMK 只进 `fill_ids`，不进 `output_ids`，因此不被 stream/detokenizer 看到
  - `python/sglang/srt/model_executor/model_runner.py`
    - sampling 时永远 mask 掉 LMK id，避免其被采样为用户 token
- **GPU-only tests**
  - `python/sglang/test/attention/test_hsa_lmk_runtime_injection_gpu.py`

**状态**：✅ 已完成

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
  - `python/sglang/test/attention/test_hsa_paged_kernel.py`
    - 构造离散 `kv_indices`，验证 kernel 读取正确
  - `python/sglang/test/attention/test_hsa_backend_decode.py`
    - 与 dense decode 对照（或 reference）验证 correctness

**当前已有测试**
- ✅ `python/sglang/test/attention/test_hsa_backend_gpu.py`（GPU-only，smoke：可跑 + delegate）
- ✅ `python/sglang/test/attention/test_hsa_backend_dense_integration_gpu.py`（GPU-only：真实 Triton 集成 + selection 可跑）
- ✅ `python/sglang/test/attention/test_hsa_lmk_runtime_injection_gpu.py`（GPU-only：LMK prompt 插入 + decode 强制 LMK 且不可见）
- ✅ `python/sglang/test/attention/test_hsa_selector_decode_gpu.py`（GPU-only：selection correctness）

**仍缺的测试（建议按优先级）**
- **P1**：真正的端到端：scheduler 跑几轮 decode，验证输出 token 序列中不出现 LMK，但内部 seqlen/kv/cache 持续增长（含 radix prefix 命中场景）
- **P2**：extend/prefill 的 LMK 自动插入（ragged）与 prefix/radix 对齐测试
- **P3**：CUDA graph / speculative / overlap / sliding window 的支持矩阵测试（先写 skip/xfail 也可以）

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


