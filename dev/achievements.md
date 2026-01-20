## HSA in SGLang：阶段性成果（截至目前）

### 目标进度概览
- **HSA 已作为 `AttentionBackend` 接入**：可用 `--attention-backend hsa` 选择（当前仍 delegate 到 dense attention，HSA kernel/selection 尚未落地）。
- **Paged-KV-first 的关键 plumbing 已打通**：metadata（page table/kv indices）、KV 写入链路、per-page chunk repr 存取、GPU-only 测试。

### 1) 文档与路线（对齐 SGLang 架构）
- ✅ `dev/hsa_dev_roadmap.md`：按 SGLang 的 **Paged KV + Radix prefix cache + AttentionBackend** 范式重写/对齐。
- ✅ `dev/hsa_sglang_impl_todo.md`：产出“按文件/类/接口级别”的落地 TODO，并持续更新状态。

### 2) CLI / Registry 接入
- ✅ `python/sglang/srt/server_args.py`
  - `ATTENTION_BACKEND_CHOICES` 增加 `"hsa"`
  - 增加 HSA 相关 CLI 参数（topk/strategy/layers/window/fusion 等）
- ✅ `python/sglang/srt/layers/attention/attention_registry.py`
  - 注册 `hsa` backend（`create_hsa_backend -> HSAAttnBackend`）

### 3) Backend：`HSAAttnBackend`（Phase‑1：可跑闭环）
- ✅ `python/sglang/srt/layers/attention/hsa_backend.py`
  - `init_forward_metadata()` 构造最小 `HSAMetadata`：
    - `page_table_1`（token->slot）
    - `real_page_table`（page_id table）
    - 透传 dense backend 的 `kv_indptr/kv_indices`（含 window 版本）
  - `forward_decode/forward_extend` 仍 **delegate 到 `TritonAttnBackend`**（保证当前阶段端到端可跑）
  - ✅ 接入 **completed page 写 repr** 的 Phase‑1 hook（占位实现：用 boundary token 的 K 作为 repr）：
    - decode：`seq_len % page_size == 0` 时写
    - extend：扫描 extend 段内所有 boundary 命中点写

### 4) KV Pool：per-page chunk repr buffer + version guard
- ✅ `python/sglang/srt/mem_cache/memory_pool.py`（`MHATokenToKVPool`）
  - 增加 `chunk_repr_buffer[layer]`：`[num_pages, head_num, head_dim]`（按 `page_id = loc // page_size`）
  - 增加 `chunk_repr_version[layer]` 与全局 `page_version`
  - 提供接口：`save_chunk_repr()` / `get_chunk_repr()` / `get_page_version()` / `bump_page_version()`
  - 版本不匹配时 `get_chunk_repr(..., page_version=...)` 返回全 0（避免 page reuse 误读旧 repr）

### 5) GPU-only 单元测试（验证 plumbing/行为）
- ✅ `python/sglang/test/attention/test_hsa_backend_gpu.py`
  - smoke：metadata 可构造 + forward 可 delegate（dummy dense backend）
- ✅ `python/sglang/test/attention/test_hsa_kvpool_repr.py`
  - KV pool repr 写/读 + version guard（CUDA）
- ✅ `python/sglang/test/attention/test_hsa_backend_chunk_repr_gpu.py`
  - completed vs partial page 的 repr 写入规则（CUDA）
- ✅ `python/sglang/test/attention/test_hsa_backend_dense_integration_gpu.py`
  - **真实集成测试（不 monkeypatch）**：真实 `TritonAttnBackend` decode 路径可跑，且 completed-page repr hook 会写入 `chunk_repr_buffer`
  - 注：为单测环境 patch `dp_attention.get_attention_tp_size()` 为 1，避免 “dp attention not initialized” 断言

### 6) 环境与依赖
- ✅ `dev/requirements.txt` 已补齐 bring-up 阶段遇到的 import/test 依赖（包含 `sgl-kernel` 等）
- ✅ 新增依赖 `einops`（由 KV 写路径触发的 transitive import 需要，用于测试场景）

### 下一阶段（尚未完成）
- ⏳ Step 4：真正的 HSA selection/top‑k/weights（paged-friendly 输出）
- ⏳ Step 5：paged HSA decode kernel（先闭环 decode correctness，再优化/融合）
- ⏳ 将 `page_version` 与 allocator/radix 的 page reuse 生命周期强绑定（生产级安全）
