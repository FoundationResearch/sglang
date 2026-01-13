# HSA (Hierarchical Sparse Attention) SGLang 开发路线图

本文档详细规划了在 SGLang 中实现 HSA（分层稀疏注意力）机制的开发路线、里程碑及测试方案。HSA 将作为一种 Native Feature 深度集成到 SGLang 中，充分利用 **Radix Attention**、**PagedAttention** 和 **Continuous Batching** 等核心优势，而非作为一个外挂模块。

## 核心架构目标

1.  **Radix-Aware KV Cache**: 扩展 Memory Pool，使 Chunk Representations ($E_i$) 与物理 Block 1:1 绑定，利用 Radix Cache 实现 $E_i$ 的跨请求复用（零开销 Prefill）。
2.  **Paged HSA Kernel**: 实现支持 Paged 内存布局的 Custom Triton Kernel，避免内存拷贝，直接在非连续显存上执行 Gather 和 Attention。
3.  **SGLang Native Integration**: 将 HSA 逻辑融入 `ModelRunner` 和 `ForwardBatch` 流程，支持 Continuous Batching。

---

## Milestone 1: 基础设施与 KV Cache 扩展 (Infrastructure)

**目标**: 扩展 SGLang 的显存池，使其能够存储和检索每个 Chunk (Block) 的粗粒度表示 ($E_i$)，并确保其生命周期受 Radix Cache 管理。

### 任务列表
- [ ] **M1.1**: 定义 Chunk Spec。
    - 约束: Chunk Size $S$ 必须等于 SGLang 的 `page_size` (通常为 16 或 32，需确认能否设为 64 或适配小尺寸)。
    - 维度: 确定 `$E_i$` 维度 (例如 `head_dim` 或 `hidden_dim`).
- [ ] **M1.2**: 修改 `python/sglang/srt/mem_cache/memory_pool.py`。
    - 在 `BaseTokenToKVPool` 中新增 `chunk_repr_data` buffer。
    - **关键设计**: `chunk_repr_data` 的索引与 `kv_data` 的物理 Block Index 严格对齐。这使得 Radix Cache 在复用 Block 时，自动复用 `$E_i$`。
- [ ] **M1.3**: 实现读写接口。
    - `save_chunk_repr(block_indices, representations)`
    - `get_chunk_repr(block_indices)`
- [ ] **M1.4**: 验证 Radix Cache 兼容性。
    - 确保 `free_block` 等操作会同步处理 `chunk_repr_data` (虽然物理内存无需清零，但逻辑上需视为释放)。

### 测试方案
- **Unit Test**: `tests/hsa/test_memory_pool_radix.py`。
    - 模拟 Request A 写入 KV 和 `$E_i$`。
    - 模拟 Request B (共享前缀) 命中 Cache。
    - 验证 Request B 能直接读取 Request A 写入的 `$E_i$`，无需重新计算。

---

## Milestone 2: Paged HSA Triton Kernel (Kernel Implementation)

**目标**: 开发适配 PagedAttention 内存布局的 HSA Decode Kernel，最大化利用显存带宽。

### 任务列表
- [ ] **M2.1**: 定义 Kernel 接口 (适配 `decode_attention` 风格)。
    - 输入: `Q`, `K_Buffer`, `V_Buffer`, `Block_Table` (或 `Selected_Block_Indices`), `Block_Weights`.
    - 关键: 输入指针直接指向全局 `kv_data` (Paged)，而非连续 Buffer。
- [ ] **M2.2**: 实现 `paged_hsa_decode_kernel`。
    - **Load Stage**: Kernel 内部根据 `Selected_Block_Indices` 计算物理偏移，从 Paged Buffer 加载 K/V。
    - **Compute Stage**: Intra-Chunk Attention。
    - **Agg Stage**: Stick-Breaking 加权。
- [ ] **M2.3**: 整合入 `TritonAttnBackend`。
    - 在 `python/sglang/srt/layers/attention/triton_backend.py` 中注册新 Kernel。

### 测试方案
- **Kernel Correctness**: `tests/hsa/test_paged_hsa_kernel.py`。
    - 构造非连续的 Paged KV Cache 数据。
    - 验证 Kernel 能正确“跳跃”读取内存并计算结果。

---

## Milestone 3: 深度集成的模型层 (Model Layer & Logic)

**目标**: 在 PyTorch 模型层实现 HSA 逻辑，并利用 SGLang 的 Batching 机制优化 Selection 过程。

### 任务列表
- [ ] **M3.1**: 实现 `ChunkEncoder` 与 Cache Check。
    - **Radix Optimization**: 在 Forward 过程中，检查当前 Chunk 是否是 New Token。只有新生成的 Chunk 才运行 Encoder，已缓存的 Chunk 直接跳过。
- [ ] **M3.2**: 批量化 Selection (Retrieval)。
    - 修改 `ForwardBatch` 或 `ModelRunner`，将所有 Active Requests 的 Queries 和 Keys 收集起来。
    - 执行 Batch Matrix Multiplication 计算 Scores，而非逐个 Request 循环。
- [ ] **M3.3**: 创建 `HSAAttention` 类。
    - 继承自 `nn.Module`，接口保持与 SGLang 标准 Attention 一致。
    - 内部根据 `forward_mode` (Prefill/Decode) 切换逻辑。

### 测试方案
- **Integration Test**: `tests/hsa/test_hsa_radix_integration.py`。
    - 构造一个共享 System Prompt 的多请求 Batch。
    - 验证 Chunk Encoder 仅对 System Prompt 运行了一次 (通过 Mock 计数器验证)。

---

## Milestone 4: 系统集成与交替层架构 (System Integration)

**目标**: 将 HSA 组装进 Llama 模型，并确保与 SGLang 的调度器 (Scheduler) 协同工作。

### 任务列表
- [ ] **M4.1**: 模型配置扩展。
    - 在 `ModelConfig` 中添加 HSA 参数。
- [ ] **M4.2**: 混合层实现 (`HybridLlamaDecoderLayer`)。
    - 根据层索引自动切换 SWA (使用现有 `RadixAttention`) 和 HSA (使用新 `HSAAttention`)。
- [ ] **M4.3**: 端到端 Forward 流程适配。
    - 确保 `LlamaForCausalLM` 在传递 `forward_batch` 时，正确携带 HSA 所需的 Metadata (如 `$E_i$` 指针)。

### 测试方案
- **E2E Generation**: `tests/hsa/test_e2e_generation.py`。
    - 测试长文本生成，确保在 Token 数量超过显存 Block 限制触发 Eviction 后，HSA 依然能正确工作 (得益于 Radix Cache 的统一管理)。

---

## Milestone 5: 性能优化 (Optimization)

**目标**: 针对 HSA 特性进行特定的性能调优。

### 任务列表
- [ ] **M5.1**: Kernel Profiling。
    - 重点关注非连续内存访问 (Gather) 的对 Memory Bandwidth 的影响。
- [ ] **M5.2**: Selection Latency 优化。
    - 尝试将 Selection 计算提前或与 GPU 计算重叠。
- [ ] **M5.3**: Benchmark 对比。
    - 场景: Long Context Summary。
    - 对比: Standard Attention vs HSA (在相同 Cache Size 下的 Throughput)。
