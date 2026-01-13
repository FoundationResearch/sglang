# SGLang 的 Native Sparse Attention (NSA) 支持与实现架构（面向读代码）

> 结论：**SGLang 已支持 NSA（Native Sparse Attention）**，并通过 `--attention-backend=nsa` 接入推理栈；当前实现 **强绑定 DeepSeek NSA 模型族**（见 `is_deepseek_nsa` 判断），同时支持按阶段选择不同实现（prefill/decode）、可选 Top‑K 变换融合、prefill 的 context-parallel（DeepSeek‑V3.2）、以及多步 speculative decoding 的 metadata 预计算优化。

---

## 1. 如何开启 / 配置 NSA

### 1.1 关键 CLI 参数

- `--attention-backend nsa`
  - Attention backend 总开关；在 registry 中映射到 `NativeSparseAttnBackend`
- `--nsa-prefill-backend {flashmla_sparse, flashmla_kv, flashmla_auto, fa3, tilelang, aiter}`
  - **prefill(extend)** 阶段使用的实现（默认 `flashmla_sparse`）
- `--nsa-decode-backend {flashmla_sparse, flashmla_kv, flashmla_auto, fa3, tilelang, aiter}`
  - **decode** 阶段使用的实现（默认 `fa3`）
- `--enable-nsa-prefill-context-parallel`
  - 开启 DeepSeek‑V3.2 长序列 prefill 的 context-parallel（CP）
- `--nsa-prefill-cp-mode {in-seq-split, round-robin-split}`
  - CP 的 token 切分策略

对应参数定义在：
- `python/sglang/srt/server_args.py`（`ServerArgs.add_cli_args` 里添加 `--nsa-*` 参数）

### 1.2 关键环境变量

在 `python/sglang/srt/environ.py`：

- `SGLANG_NSA_FUSE_TOPK`（默认 True）
  - 是否将“topk 选择 + 索引变换（从 token index 转 page_table index）”融合到 TopK kernel
- `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA`（默认 True）
  - 多步 speculative decoding 时：是否只预计算一次 metadata，再复制给多份 NSA backend（更快）
- `SGLANG_FLASHINFER_WORKSPACE_SIZE`
  - 在 SM100/B200 等设备上，部分 kernel（包含 NSA 中的 TRTLLM ragged kernel）会分配 workspace

---

## 2. 入口与调度：NSA backend 如何进入推理栈

### 2.1 Backend registry 入口

- `python/sglang/srt/layers/attention/attention_registry.py`
  - `@register_attention_backend("nsa")` → `create_nsa_backend()` → `NativeSparseAttnBackend(runner)`

### 2.2 模型适配：仅支持 DeepSeek NSA 模型族

- `python/sglang/srt/configs/model_config.py`
  - `is_deepseek_nsa(config)`：要求 `architectures` 在允许列表中，且 `config.index_topk` 存在
  - `get_nsa_index_topk(config)`：读取 `hf_config.index_topk`

NSA backend 初始化时强断言：
- `python/sglang/srt/layers/attention/nsa_backend.py`
  - `self.use_nsa = is_deepseek_nsa(...)`
  - `assert self.use_nsa, "NSA backend only supports DeepSeek NSA"`

### 2.3 DeepSeek 模型内部的 attention forward method 选择

DeepSeek 模型有自己的 “forward method dispatch”，NSA 在这里不会像其他 backend 一样直接分 MHA/MLA 细节，
而是**把核心决策集中在 `NativeSparseAttnBackend.set_nsa_prefill_impl()`**，然后模型侧读取 `backend.use_mha` 结果：

- `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`
  - `handle_attention_nsa()`：如果 `backend.use_mha` 为 True → `MHA_ONE_SHOT`，否则走 `MLA`

---

## 3. 核心数据结构：NSA metadata 长什么样、谁负责填

NSA backend 每次 forward 前会先构造/更新一份 metadata（prefill 和 decode 都依赖它）：

### 3.1 `NSAMetadata`

定义在 `python/sglang/srt/layers/attention/nsa_backend.py`：

- **序列长度信息**
  - `cache_seqlens_int32`：当前 batch 每个 request 的 KV 长度
  - `cu_seqlens_q / cu_seqlens_k`：varlen 累计长度
  - `nsa_cache_seqlens_int32` / `nsa_cu_seqlens_k`：对 KV 长度做 `topk` 截断后的版本（NSA 只看近似 top‑k 范围）
- **页表（KV cache 索引）**
  - `page_table_1`：page_size=1 的 page table（token→slot 映射），NSA 的很多索引变换都在这层做
  - `real_page_table`：当真实 `page_size != 1` 时，由 `_transform_table_1_to_real()` 转换得到，用于真实 KV cache block 索引
- **prefill 相关的展开信息**
  - `nsa_extend_seq_lens_list`、`nsa_seqlens_expanded`、`page_table_1_flattened`、`topk_indices_offset` 等
- **FlashMLA / DeepGEMM 辅助 metadata**
  - `flashmla_metadata`、`paged_mqa_schedule_metadata`

### 3.2 `NSAIndexerMetadata`

同样在 `nsa_backend.py`，用于把 `NSAMetadata` 以 indexer 需要的接口暴露出去，并封装 `topk_transform()`：

- `topk_transform_method`：
  - `PAGED`：TopK 结果最终要映射到 page_table（paged KV 索引）
  - `RAGGED`：TopK 结果最终要映射到 ragged KV（常见于 FP8 KV cache + flashmla_sparse 的组合）

---

## 4. 主流程：从 forward_batch 到 prefill/decode 的实际算子调用

NSA 后端核心文件：
- `python/sglang/srt/layers/attention/nsa_backend.py`

### 4.1 `init_forward_metadata(forward_batch)`

这是 **NSA 的“中央调度点”**：

- 先根据 `forward_mode`（decode / extend / target_verify / draft_extend …）准备：
  - `cache_seqlens_int32`、`cu_seqlens_{q,k}`、`page_table`（从 `req_to_token_pool.req_to_token[...]` 切片得到）
  - 对 speculative 模式（verify / draft_extend）做额外展开（repeat_interleave / 拼接 seqlens_expanded 等）
- 调用 `self.set_nsa_prefill_impl(forward_batch)`：
  - 决定本 batch 的策略：是否走 `use_mha`、prefill impl 是否自动选择、以及对应 heuristic
- 根据 CP 配置，可能对 `seqlens_expanded`/`extend_seq_lens` 做重排或 split（见 `nsa/utils.py`）
- 计算 NSA 专用 seqlens：
  - `compute_nsa_seqlens(...)`：把长度 clamp 到 `index_topk`
  - 以及 `nsa_cu_seqlens_k` 等

> 阅读建议：先从 `init_forward_metadata()` 读清楚不同 `ForwardMode` 下 metadata 是怎么变化的，
> 再回头看各个 `_forward_*` 分支，否则很容易被“展开后的 shape”绕晕。

### 4.2 prefill：`forward_prefill(...)`

prefill 会拿到 indexer 输出的 `topk_indices`（或融合后直接是 transformed page_table），然后根据 `nsa_prefill_impl` 分流：

- `tilelang` → `_forward_tilelang(...)`
  - kernel 在 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`
  - `tilelang_sparse_fwd()` 里目前 **强假设 topk==2048**
- `flashmla_sparse` → `_forward_flashmla_sparse(...)`
  - 如果 `TopkTransformMethod.RAGGED`，可能需要：
    - prefix sharing 时：`dequantize_k_cache_paged(...)`（把 FP8 paged KV 反量化成 bf16 形式供 sparse 读）
    - 或直接拼接 `_cat([k, k_rope])` 作为 ragged KV
- `flashmla_kv` → `_forward_flashmla_kv(...)`
  - 依赖 FlashMLA 的 metadata（`get_mla_metadata`）
- `fa3` → `_forward_fa3(...)`
  - 调用 `flash_attn_with_kvcache(...)`

### 4.3 decode：`forward_decode(...)`

decode 同样根据 `nsa_decode_impl` 分流（`flashmla_sparse / flashmla_kv / tilelang / fa3 / aiter`）。

其中一个关键点是：**decode 的 q token 数通常是 1**，但 `topk_indices` 的行数必须与 q 的 layout 一致；
因此 backend 里有 `_pad_topk_indices()` 做对齐保护。

---

## 5. Indexer（选择 top‑k key 的那部分）是怎么实现的

核心文件：
- `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`
- `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`
- `python/sglang/srt/layers/attention/nsa/triton_kernel.py`（一些 Triton helper）

### 5.1 Indexer 的角色

NSA 的 sparse attention 需要“每个 query token 该看哪些 key token”。这一组 key token 索引就是 `topk_indices`。

SGLang 的 Indexer 做的事可概括为：

- 从 hidden state 计算用于“路由/打分”的 query 表示（含 LoRA rank、head gate 等）
- 从 KV cache（paged）里取出 index‑K（常为 FP8 存储 + scale），计算与 query 的相似度 logits（通常是 MQA 形态）
- 对 logits 做 top‑k，必要时把 top‑k 结果转换为 attention kernel 需要的索引形式

### 5.2 FP8 KV cache 下的 K/Scale 读取：`index_buf_accessor.py`

为了高效从 paged buffer 中读取 key 与 scale：

- `GetK / GetS / GetKAndS`（Triton kernel）
  - `_get_k_and_s_triton()` 走 fused gather：一次性读出 K（128 bytes）+ S（4 bytes）

这段通常用于 `nsa_indexer.Indexer._get_topk_ragged()` 等路径。

### 5.3 TopK 与索引变换：PAGED vs RAGGED

NSA backend 通过 `NSAIndexerMetadata.topk_transform()` 统一封装 topk 选择与“索引变换”：

- 非融合：`fast_topk_v2(...)` 只输出 topk token index，然后在 backend 侧调用：
  - `transform_index_page_table_decode(...)`
  - `transform_index_page_table_prefill(...)`
  将 token index 映射为 `page_table_1` 中的真实 KV slot index
- 融合（`SGLANG_NSA_FUSE_TOPK=True`）：
  - `fast_topk_transform_fused(...)` / `fast_topk_transform_ragged_fused(...)`
  直接输出“已经变换好的索引”，省掉 transform kernel

索引变换实现位置：
- `python/sglang/srt/layers/attention/nsa/transform_index.py`

---

## 6. FP8 KV cache 量化/反量化：为什么 NSA 会涉及 dequantize

文件：
- `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`
- `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`

当启用 FP8 KV cache（`token_to_kv_pool.nsa_kv_cache_store_fp8`）时：

- 存储时：`quantize_k_cache(...)` / `quantize_k_cache_separate(...)`
  - 把 K(nope=512) 转 fp8 + scale（每 128 一组），rope 部分保持 bf16 bytes
- 读取时：某些 sparse 路径需要临时还原 bf16 K（尤其当 topk 结果要走 ragged 形式时）：
  - `dequantize_k_cache_paged(...)`

> 读代码的经验法则：看到 `TopkTransformMethod.RAGGED`，基本就意味着会触发一些“从 paged FP8 KV 还原到可直接算 attention 的形态”的逻辑。

---

## 7. Context Parallel（DeepSeek‑V3.2 prefill）：两种切分模式

文件：
- `python/sglang/srt/layers/attention/nsa/utils.py`

### 7.1 `round-robin-split`

- 按 token_idx % cp_size 分配 token 给不同 rank
- 优点：更容易支持多 batch、fused MoE、FP8 KV cache
- 关键函数：
  - `nsa_cp_round_robin_split_data(...)`
  - `nsa_cp_round_robin_split_q_seqs(...)`

### 7.2 `in-seq-split`（zigzag）

- 把序列切成 `cp_size * 2` 块，做 zigzag 重排以平衡因果 attention 的算量
- 关键函数：
  - `prepare_input_dp_with_cp_dsa(...)`
  - allgather 后重排：`cp_all_gather_rerange_output(...)`

---

## 8. 多步 speculative decoding（MTP）：metadata 预计算/复制

文件：
- `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`
- `python/sglang/srt/layers/attention/nsa_backend.py`（`NativeSparseAttnMultiStepBackend`）

当 speculative decoding 需要多个 step、且每个 step 都要跑一套 NSA backend 时：

- 默认开启 `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA=True`
- 会先在一个 backend 上 `_precompute_replay_metadata(...)` 得到 `PrecomputedMetadata`
- 再把 tensor 快速 copy 给每个 step 的 backend（比每个 backend 自己算一遍更快）

---

## 9. 建议的“读代码路线图”

如果你想从最外层一路追到 kernel：

1. `python/sglang/srt/layers/attention/attention_registry.py`
   - 确认 `--attention-backend=nsa` 如何实例化 backend
2. `python/sglang/srt/layers/attention/nsa_backend.py`
   - 先读：`__init__`、`init_forward_metadata`、`set_nsa_prefill_impl`
   - 再读：`forward_prefill` / `forward_decode` 的分流逻辑
3. `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`
   - 搞清楚 topk 是怎么从 KV cache 里算出来的（尤其 `_get_topk_ragged`）
4. `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`
   - 看 FP8 K/scale 是怎么被高效 gather 的
5. `python/sglang/srt/layers/attention/nsa/transform_index.py`
   - 看 token index → page_table index 的 transform（融合与否的差别）
6. `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`
   - 如果你关心 tilelang sparse attention 的 kernel 细节

---

## 10. 对应测试（帮助你快速验证/定位功能）

- `test/registered/kernels/test_nsa_indexer.py`
  - Indexer/Backend 的单元测试（包含 mock、patch deep_gemm/triton kernel 等）
- `test/registered/8-gpu-models/test_deepseek_v32.py`
  - NSA backend 组合测试（例如 `flashmla_sparse` prefill + `flashmla_kv` decode 等）
- `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`
  - CP（context parallel）相关测试用例


