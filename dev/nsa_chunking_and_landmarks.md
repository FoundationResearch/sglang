# NSA 的 chunk 化与 “landmark token” 机制（对照 SGLang 实现读代码）

这份文档专门回答两个问题：

- **NSA 的“chunk 化”具体发生在哪里？chunk 的边界/shape/offset 怎么处理？**
- **NSA 里的 “landmark token” 在 SGLang 实现中对应什么？它是怎么算出来并喂给 sparse attention kernel 的？**

> 重要前置：在 SGLang 代码里几乎没有直接叫 `landmark` 的变量名。你在论文/实现解读里看到的 “landmark tokens” 在这里主要对应两类对象：  
> 1) **indexer 计算并存入 KV pool 的 `index_k`（用于检索/打分的 per-token 表征，可理解为“landmark key”）**；  
> 2) **每个 query token 通过 top‑k 选择得到的 KV token 位置集合（这些被选中的 token 就是本步 sparse attention 的“landmarks/active KVs”）**。

---

## 1. Chunk 化一共有两层：别把它们混在一起

### 1.1 线程内（kernel 前）chunk：**NSA indexer 的 logits chunking（防 OOM）**

发生在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`：

- 对象：`deep_gemm.fp8_mqa_logits(...)` 的输出 `logits`（float32，形状大致是 \([num_q_tokens, num_k_tokens]\)）
- 目的：当 \([Q,K]\) 太大时，避免一次性分配整个 logits 矩阵导致 OOM
- 手段：把 `q_offset`（有效 query token 数）切成多个 `[start:end]` 的 chunk，**逐 chunk 计算 logits + 逐 chunk 做 top‑k**，再写回 `topk_result`

### 1.2 调度层（跨 forward pass）chunk：**长 prompt 的 chunked prefill / chunked prefix cache**

发生在 `python/sglang/srt/model_executor/forward_batch_info.py`（以及 scheduler）：

- 对象：**prefill 阶段超长输入**（尤其 DeepSeek 系列）
- 目的：控制一次 prefill 的 token 上限（chunk capacity），并为“按 chunk 复用/访问 prefix KV”准备索引
- 手段：把 prefix 拆成多个 prefix-chunk，预先为每个 chunk 构造 `prefix_chunk_kv_indices`

> 这两层 chunk 化的关系：**调度层 chunk**决定一次 forward 里实际进入 attention 的 token 子集；**indexer chunk**是在“这一次 forward 内”进一步把 logits 计算拆小。两者可能同时存在，但职责不同。

---

## 2. NSA indexer 的 logits chunking：实现细节（非常关键）

相关代码在 `nsa_indexer.Indexer._get_topk_ragged()`。

### 2.1 触发条件：`_should_chunk_mqa_logits(num_q, num_k)`

其逻辑是：

- **快速静态判断**：如果 \(num_q \times num_k < 8,000,000\)（约 32MB logits）直接不 chunk
- 否则读取 `torch.cuda.mem_get_info()` 的 `(free_mem, total_mem)`，估算 logits 占用：
  - `bytes_per_elem = 4`（float32）
  - `logits_bytes = num_q * num_k * bytes_per_elem`
- 触发 chunk 的启发式：
  - `logits_bytes * 2 > free_mem`  或  `logits_bytes > total_mem * 0.3`

这意味着 indexer 假设 logits 的临时分配不应吞掉可用显存的大头（给后续算子留空间）。

### 2.2 如何切 chunk：用 free_mem 反推 `max_rows`

chunk 大小按“每行 logits 的字节数”估算：

- `bytes_per_row = k_offset * 4`
- 预留 50% free_mem 给 logits：
  - `max_rows = int((free_mem * 0.5) // bytes_per_row)`
- `start/end` 循环：
  - `end = min(start + max_rows, q_offset)`
  - 计算 `logits_chunk = deep_gemm.fp8_mqa_logits(q_fp8[start:end], ...)`
  - **直接在 chunk 上做 top‑k + 索引变换**

### 2.3 chunk 内 top‑k 的关键：PAGED vs RAGGED 的 offset 处理

在 SGLang NSA 实现中，top‑k 的输出不一定是“token index”，它可能已经被变换成“KV page table index”（如果开启融合）。
因此 chunk 内 top‑k 要正确处理两个模式：

- **RAGGED 路径**（通常意味着使用了 `topk_indices_offset` 全局偏移）：
  - `global_topk_offset = metadata.attn_metadata.topk_indices_offset` 非空
  - chunk 内传：
    - `topk_indices_offset_override = global_topk_offset[start:end]`
    - `cu_seqlens_q_chunk = None`
    - `batch_idx_chunk = None`
- **PAGED 路径**：
  - `global_topk_offset is None`
  - chunk 内把每个 token 当作“长度=1 的 q-seq”来构造 top‑k 所需的 `cu_seqlens_q`：
    - `cu_seqlens_q_chunk = torch.ones(B_chunk, int32)`（注意：这里传的是 lengths，内部会再做 cumsum）
    - `batch_idx_chunk = token_to_batch_idx[start:end]`（用于选 `page_table_1` 的子集）

最终都调用：

- `raw_topk_chunk = metadata.topk_transform(logits_chunk, topk, ...)`
- 写回 `topk_result[start:end]`

> 读代码重点：chunk 化不是简单把 logits 计算拆开，**它还必须把 top‑k 所依赖的 offset/映射信息一并切分并保持一致**，否则你会在 PAGED/RAGGED 模式下拿到错误的 KV 索引。

---

## 3. “landmark token” 在 SGLang NSA 中对应什么：两段式（表征 + 选择）

### 3.1 表征阶段：构造并写入 `index_k`（可视作“landmark key”）

发生在 `nsa_indexer.Indexer.forward_cuda(...)`：

- 从输入激活算出：
  - `query, key = _get_q_k_bf16(...)`
  - `q_fp8, q_scale = act_quant(query, ...)`
  - `k_fp8, k_scale = act_quant(key, ...)`
  - `weights = _get_logits_head_gate(x, q_scale)`（用于 logits 的 head-gate/scale）
- 然后把 **indexer 专用的 key 表征**写入 token_to_kv_pool：
  - `token_to_kv_pool.set_index_k_scale_buffer(layer_id, loc, index_k=k_fp8, index_k_scale=k_scale)`

这一步的关键意义是：**NSA 的检索并不直接拿 attention 的 K/V，而是拿一个专门用于打分的 index‑K（低维/可量化/适合 MQA logits）**。

### 3.2 选择阶段：每个 query token 选出 top‑k KV token 位置（这些就是“landmark tokens”）

选择发生在 `_get_topk_ragged(...)` 等函数里：

1. 从 KV pool 读取历史 token 的 `index_k + scale`（paged/gather）  
2. 用 `deep_gemm.fp8_mqa_logits(...)` 计算 query 对历史 KV 的打分 logits  
3. 调用 `metadata.topk_transform(...)` 做 top‑k 并返回索引

返回的索引张量（概念上）是：

- `topk_indices`: \([num_q_tokens, index_topk]\)，int32，padding 用 `-1`

但注意：如果 `SGLANG_NSA_FUSE_TOPK=True` 且 `TopkTransformMethod=PAGED`，**它返回的可能不是 token index，而是已经映射过的 page_table 索引（可直接用于后续 attention kernel）**。

---

## 4. top‑k 输出为什么有时不是 token index：`topk_transform()` 的融合/变换

`NSAIndexerMetadata.topk_transform()`（在 `python/sglang/srt/layers/attention/nsa_backend.py`）把 top‑k 和“索引变换”统一封装：

- `SGLANG_NSA_FUSE_TOPK=False`：
  - `fast_topk_v2(...)`：输出“top‑k token positions”
  - 后续需要在 backend 侧再调用 transform，把 token pos → KV slot（page table）pos
- `SGLANG_NSA_FUSE_TOPK=True`：
  - `TopkTransformMethod.PAGED`：`fast_topk_transform_fused(...)` 直接输出“变换后的 page table index”
  - `TopkTransformMethod.RAGGED`：`fast_topk_transform_ragged_fused(...)` 输出 ragged 语义下的索引

如果你在调试时想验证“到底输出的是什么索引”，第一步就看：

- `envs.SGLANG_NSA_FUSE_TOPK.get()`
- `backend.get_topk_transform_method()`

以及 prefill/decode 分支里对 `topk_indices` 的处理（有时会再走 `transform_index_page_table_*`）。

---

## 5. NSA 的“跳过 top‑k”优化：当长度 ≤ index_topk 时 landmark 退化为“全选”

`nsa_indexer.Indexer.forward_cuda(...)` 有一个很重要的 fast path：

- 当 `forward_mode.is_extend_without_speculative()` 且 `max_kv_len <= index_topk` 时：
  - 认为没必要算 logits/top‑k（因为 top‑k 会覆盖全部历史 token）
  - 直接走 `_forward_cuda_k_only(...)`：
    - 只计算/存 `index_k`
    - 用 dummy logits 触发 top‑k kernel 的 fast path 生成 `[0,1,...,len-1,-1,...]` 形式的索引

因此，在短上下文场景下你可能看不到 “landmark selection” 的复杂逻辑——它被这个优化绕开了。

---

## 6. 调度层 chunk：chunked prefix cache 的 KV 索引是怎么构造的（与 NSA 的关系）

这部分不在 NSA indexer 里，而在 `ForwardBatch.prepare_chunked_prefix_cache_info()`：

- 先定义每个 chunk 的 token 上限：
  - `ForwardBatch.get_max_chunk_capacity()`：当前硬编码为 `128 * 1024`
- 再计算：
  - `prefix_chunk_len = chunk_capacity // batch_size`
  - `num_prefix_chunks = ceil(max(prefix_len) / prefix_chunk_len)`
  - `prefix_chunk_starts` / `prefix_chunk_seq_lens` / `prefix_chunk_cu_seq_lens`
- 最后为每个 chunk 预计算：
  - `prefix_chunk_kv_indices[idx]`：一个扁平的 KV slot 索引数组（int32）
  - 通过 Triton kernel `create_chunked_prefix_cache_kv_indices[...]` 填充

你可以把 `prefix_chunk_kv_indices` 理解为：“当我只处理 prefix 的第 idx 个 chunk 时，这一次 attention 应该访问 KV cache 的哪些物理 slot”。

与 NSA 的关系：

- 它们都发生在 DeepSeek 系列长上下文 prefill 里，但**chunked prefix cache 解决的是 prefill 的内存与 KV 访问组织问题**；
- **NSA 的 landmark/top‑k 检索解决的是 sparse attention 的 KV 子集选择问题**（更像“算力/带宽优化”）。

---

## 7. 推荐你按这个顺序读代码（最省时间）

1. `nsa_indexer.py`：先读 `_get_topk_ragged()` 的 chunk 化逻辑（它是最容易踩坑的）
2. `nsa_backend.py`：再读 `NSAIndexerMetadata.topk_transform()`，搞清楚融合后返回的索引语义
3. `transform_index.py`：对照 PAGED 模式下 token index → KV slot index 的变换
4. `forward_batch_info.py`：如果你关心超长 prompt 的 chunked prefill，再读 `prepare_chunked_prefix_cache_info()` 和 `prepare_chunked_kv_indices()`


