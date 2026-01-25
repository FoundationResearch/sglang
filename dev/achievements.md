## HSA in SGLang：阶段性成果（截至目前）

### 目标进度概览
- **HSA 已作为 `AttentionBackend` 接入**：可用 `--attention-backend hsa` 选择（当前 attention compute 仍 delegate 到 dense，HSA kernel 尚未落地；但 selection 已实现并写入 metadata）。
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

### 4) \(E_i\) 语义对齐 FlashHSA（LMK 真实 token）
- ✅ \(E_i\) 定义切换为：**每个 page 的 LMK token（最后一个 slot）在该层 KV cache 中的 K**
  - `lmk_token_loc = page_id * page_size + (page_size - 1)`
  - selection 从 KV cache gather `K[lmk_token_loc]`（每层各自 gather）
  - 通过 **completed-page gating** 排除未写入 LMK 的 pages

### 5) GPU-only 单元测试（验证 plumbing/行为）
- ✅ `python/sglang/test/attention/test_hsa_backend_gpu.py`
  - smoke：metadata 可构造 + forward 可 delegate（dummy dense backend）
- ✅ `python/sglang/test/attention/test_hsa_kvpool_repr.py`
  - KV pool repr 写/读 + version guard（CUDA）
- ✅ `python/sglang/test/attention/test_hsa_backend_chunk_repr_gpu.py`
  - completed vs partial page 的 repr 写入规则（CUDA）
- ✅ `python/sglang/test/attention/test_hsa_backend_dense_integration_gpu.py`
  - **真实集成测试（不 monkeypatch）**：真实 `TritonAttnBackend` decode 路径可跑，且 selection 能从 KV gather LMK-K 并排除 non-completed pages
  - 注：为单测环境 patch `dp_attention.get_attention_tp_size()` 为 1，避免 “dp attention not initialized” 断言

### 6) Selection（Step 4：Torch reference，先用于 metadata/可观测，不影响 compute）
- ✅ `python/sglang/srt/layers/attention/hsa/selector.py`
  - active pages candidates（仅本请求活跃 pages）
  - SWA→HSA：排除 window pages
  - fixed‑K top‑k（`group/head`），无效候选 mask 为 `-inf`，输出 `-1` padding
- ✅ `python/sglang/srt/layers/attention/hsa_backend.py`
  - decode path 计算 selection，并写入 `HSAMetadata.hsa_*` debug 字段（attention 计算仍 delegate）
- ✅ `python/sglang/test/attention/test_hsa_selector_decode_gpu.py`（CUDA）

### 7) 环境与依赖
- ✅ `dev/requirements.txt` 已补齐 bring-up 阶段遇到的 import/test 依赖（包含 `sgl-kernel` 等）
- ✅ 新增依赖 `einops`（由 KV 写路径触发的 transitive import 需要，用于测试场景）

### 下一阶段（尚未完成）
- ⏳ Step 5：paged HSA decode kernel（先闭环 decode correctness，再优化/融合）
- ⏳ 将 allocator/radix 的 page reuse 生命周期与 HSA 的“completed-page”语义强绑定（生产级安全）
