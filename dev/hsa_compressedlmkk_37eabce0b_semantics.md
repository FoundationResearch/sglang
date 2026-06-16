## 1. LMK embedding / vocab layout 语义

### 涉及文件

- `python/sglang/srt/models/flash_hsa.py`

### 涉及组件

- `FlashHSAInnerXModel`
- `HSAForCausalLM`
- `_get_flashhsa_padded_vocab_size`
- `embed_tokens`
- `lm_head`
- `lmk_embed`

### TRM 里的语义

FlashHSA 的 landmark token 使用 `lmk_id == config.vocab_size`。但不同 checkpoint 有两种 LMK embedding 存储方式：

1. `enable_external_lmk_embed=False`：
   - `embed_tokens` / `lm_head` padded 到至少 `vocab_size + 1`；
   - LMK embedding 是 `embed_tokens[lmk_id]` 这一行。

2. `enable_external_lmk_embed=True`：
   - `embed_tokens` / `lm_head` 只有原始 `vocab_size` 行；
   - LMK embedding 单独存为 `model.lmk_embed`；
   - forward 遇到 `input_ids == lmk_id` 时，需要用 `lmk_embed` 替换普通 token embedding。

### SGLang 未对齐点

原 SGLang 只按 legacy 路径处理：默认给 `embed_tokens/lm_head` 多分配 LMK 行，并直接调用：

```python
hidden_states = self.embed_tokens(input_ids)
```

这在 `enable_external_lmk_embed=True` 的 checkpoint 下会有两个问题：

- `input_ids` 中的 `lmk_id == vocab_size` 可能越界；
- 即使不越界，也不会加载/使用 checkpoint 中单独训练出来的 `model.lmk_embed`。

### commit 修改措施

- `_get_flashhsa_padded_vocab_size(config)` 根据 `enable_external_lmk_embed` 决定返回：
  - `vocab_size`；或
  - `next_of_y(vocab_size + 1, 32)`。
- `FlashHSAInnerXModel` 中新增：
  - `self._enable_external_lmk_embed`
  - `self._lmk_id`
  - `self.lmk_embed`
- forward 中如果存在 `self.lmk_embed`：
  - 用 `safe_ids = input_ids.masked_fill(lmk_mask, 0)` 避免越界；
  - 先查普通 embedding；
  - 再用 `torch.where(lmk_mask, self.lmk_embed, hidden_states)` 替换 LMK 位置。

---

## 2. `hsa_denom == 1` 全 HSA 模型的 q/k/v 权重名语义

### 涉及文件

- `python/sglang/srt/models/flash_hsa.py`

### 涉及组件

- `FlashHSAInnerXHierarchicalSparseAttention`
- `q_proj / k_proj / v_proj`
- `hsa_q_proj / hsa_k_proj / hsa_v_proj`
- SGLang weight loading by parameter name

### TRM 里的语义

当：

```json
"hsa_heads": num_attention_heads
```

即 `hsa_denom == 1` 时，模型是全 HSA attention，没有 SWA branch。TRM checkpoint 中投影权重仍然使用普通 attention 命名：

- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`

而不是额外的：

- `self_attn.hsa_q_proj.weight`
- `self_attn.hsa_k_proj.weight`
- `self_attn.hsa_v_proj.weight`

### SGLang 未对齐点

原 SGLang 按 split-head 结构创建 HSA branch 的投影模块：

- `hsa_q_proj`
- `hsa_k_proj`
- `hsa_v_proj`

在全 HSA checkpoint 下，checkpoint 里没有这些名字，导致核心 q/k/v 投影权重加载不到对应模块，SGLang forward 会使用未正确加载的权重，表现为 logits 异常、Step 0 直接错。

### commit 修改措施

在 `FlashHSAInnerXHierarchicalSparseAttention.__init__` 中区分：

1. `has_swa_branch=True`：
   - 创建 SWA 的 `q_proj/k_proj/v_proj`；
   - 创建 HSA 的 `hsa_q_proj/hsa_k_proj/hsa_v_proj`。

2. `has_swa_branch=False`：
   - 创建 `q_proj/k_proj/v_proj`，名字和 TRM checkpoint 对齐；
   - 内部 alias：

```python
self.hsa_q_proj = self.q_proj
self.hsa_k_proj = self.k_proj
self.hsa_v_proj = self.v_proj
```

这样既保证 checkpoint 权重名能加载，又保证 forward 内部仍可统一使用 `hsa_*` 投影路径。

---

## 3. `lmk_q_lora_dim` 语义

### 涉及文件

- `python/sglang/srt/models/flash_hsa.py`

### 涉及组件

- `FlashHSAInnerXHierarchicalSparseAttention`
- `lmk_q_proj`
- `lmk_q_norm`
- `ReplicatedLinear`
- `ColumnParallelLinear`
- projection fusion path

### TRM 里的语义

当配置中有：

```json
"enable_lmk_q_proj": true,
"lmk_q_lora_dim": 64
```

TRM 中的 `lmk_q_proj` 是 LoRA-style 两层投影：

```python
lmk_q_proj = Sequential(
    Linear(hidden_size, lmk_q_lora_dim),
    Linear(lmk_q_lora_dim, hidden_size_or_hq_dim),
)
```

并且 forward 时有 residual：

```python
lmk_q = lmk_q_proj(hidden_states)
if lmk_q_lora_dim > 0:
    lmk_q = lmk_q + hsa_q
```

权重名对应：

- `lmk_q_proj.0.weight`
- `lmk_q_proj.1.weight`

### SGLang 未对齐点

原 SGLang 只实现了单层 `ColumnParallelLinear`：

```python
self.lmk_q_proj = ColumnParallelLinear(..., prefix="lmk_q_proj")
```

问题：

- 无法加载 checkpoint 中的 `lmk_q_proj.0 / lmk_q_proj.1`；
- 缺少 `lmk_q + hsa_q` residual；
- selection query 与 TRM 不一致。

### commit 修改措施

- 新增 `self.lmk_q_lora_dim = int(getattr(config, "lmk_q_lora_dim", -1))`。
- 当 `lmk_q_lora_dim > 0 and lmk_q_full_dim` 时：
  - 用 `nn.ModuleList` 包两层：
    - `ReplicatedLinear(... prefix="lmk_q_proj.0")`
    - `ColumnParallelLinear(... prefix="lmk_q_proj.1")`
- forward 中保存 `hsa_q_raw_for_lmk`。
- 计算 `lmk_q` 时：

```python
lmk_q_mid, _ = self.lmk_q_proj[0](hidden_states)
lmk_q, _ = self.lmk_q_proj[1](lmk_q_mid)
lmk_q = lmk_q + hsa_q_raw_for_lmk
```

- 同时补齐 `layerwise_lmkq_norm` 对 `lmk_q_norm` 维度的影响。

---

## 4. `enable_inrange_rope` / HoPE RoPE 语义

### 涉及文件

- `python/sglang/srt/models/flash_hsa.py`

### 涉及组件

- `_HoPERotaryEmbedding`
- `_InRangeRotaryEmbedding`
- `_make_rope_for_hsa`
- `FlashHSAInnerXAttention.rotary_emb`
- `FlashHSAInnerXHierarchicalSparseAttention.rotary_emb`

### TRM 里的语义

TRM 支持 FlashHSA 特定 RoPE knobs：

```json
"enable_inrange_rope": true,
"rope_context_length": 8192,
"rope_period_multiplier": 2.0,
"apply_hsa_rope": true
```

inrange-RoPE 的语义：

- 先按普通 RoPE 计算 `inv_freq`；
- 根据阈值：

```text
threshold = 2π * rope_period_multiplier / rope_context_length
```

- 只保留 `inv_freq >= threshold` 的高频 pair；
- 低频 pair 置零，使其变成 position-invariant。

HoPE 语义：

- 根据 `hope_partial_ratio / hope_context_length / hope_theta` 只保留部分高频 RoPE pair；
- 某些兼容路径下，SWA/base 层仍保持标准 RoPE，HSA 层才使用 HoPE/inrange。

### SGLang 未对齐点

原 SGLang 直接用标准 `get_rope(...)`，没有实现：

- inrange frequency mask；
- HoPE partial RoPE；
- `use_hope + enable_inrange_rope` 下 HSA/SWA 分层处理。

这会导致 q/k/lmk_q 的 RoPE 后向量和 TRM 不一致。

### commit 修改措施

- 新增 `_HoPERotaryEmbedding`。
- 新增 `_InRangeRotaryEmbedding`。
- 新增统一构造入口 `_make_rope_for_hsa(...)`。
- 在普通 attention 和 HSA attention 中都改成通过 `_make_rope_for_hsa(...)` 创建 `rotary_emb`。
- `_make_rope_for_hsa` 处理：
  - `enable_inrange_rope`；
  - `use_hope`；
  - `use_hope and enable_inrange_rope and not for_hsa_layer` 时 fallback 到标准 RoPE。

---

## 5. `apply_hsa_rope + enable_lmk_q_proj` 下 `lmk_q` RoPE 语义

### 涉及文件

- `python/sglang/srt/models/flash_hsa.py`

### 涉及组件

- `FlashHSAInnerXHierarchicalSparseAttention.forward`
- `sel_q`
- `lmk_q`
- `self.rotary_emb`

### TRM 里的语义

当：

```json
"apply_hsa_rope": true,
"enable_lmk_q_proj": true
```

且没有启用 `nope_retrieval` 时，TRM 会对 HSA branch 的 selection query `lmk_q` 也应用 RoPE。

也就是说 selection 不是用 raw projected `lmk_q`，而是用 RoPE 后的 `lmk_q`。

### SGLang 未对齐点

原 SGLang 代码里 selection query 明确是 `NO RoPE`，即 `lmk_q_proj` 输出后只 norm/reshape，不再 apply RoPE。

这会导致 HSA page selection 使用的 query 与 TRM 不一致，top-k chunk 选择会不同。

### commit 修改措施

在 `FlashHSAInnerXHierarchicalSparseAttention.forward` 中增加：

```python
if (
    self.enable_lmk_q_proj
    and self.apply_hsa_rope
    and not bool(getattr(self.config, "nope_retrieval", False))
    and sel_q.shape[-1] == self.head_dim
):
    sel_q, _ = self.rotary_emb(positions, sel_q.contiguous(), torch.empty_like(sel_q))
```

修复后，`lmk_q` selection query 与 TRM 的 `apply_hsa_rope` 路径一致。

---

## 6. `enable_prior_query` 的 per-q-head LMK K / `prior_b` 语义

### 涉及文件

- `python/sglang/srt/models/flash_hsa.py`
- `python/sglang/srt/layers/attention/hsa_backend.py`
- `python/sglang/srt/layers/attention/hsa/selector.py`
- `python/sglang/srt/mem_cache/landmark_pool.py`

### 涉及组件

- `LandmarkLmkKPool`
- `ReqToChunkPool`
- `_maybe_write_chunk_lmk_k`
- `_maybe_write_decode_chunk_lmk_k`
- `select_topk_pages_decode_fused`
- extend selection per-q-head path
- decode selection per-q-head path

### TRM 里的语义

`enable_prior_query=true` 时，TRM 不只是打开一个 flag，而是改变 landmark key 的构造方式：

1. 对每个 chunk，取 chunk 末尾 LMK query：`mu_q`；
2. 用 `mu_q` attend chunk 内的 HSA K；
3. 得到 per-q-head 的 aggregated `lmk_k`；
4. 同时计算 entropy bias：`prior_b`；
5. HSA top-k selection 时，score 要加上 `prior_b`。

核心数学：

```text
logits = einsum(mu_q, K_chunk) * sm_scale
logits[last_lmk_token] = -inf
p = softmax(logits)
lmk_k = sum(p * K_chunk)
prior_b = entropy(p)
selection_score = q · lmk_k + prior_b
```

### SGLang 未对齐点

原 SGLang：

- 没有自动初始化 `lmk_k_pool / req_to_chunk_pool`；
- prefill/decode 时没有按 TRM 的 `chunk_attn_pool` 写 per-q-head `lmk_k`；
- selection 中没有加 `prior_b`；
- 只能 fallback 到旧的 h_kv-shared LMK key 路径。

这会导致 `enable_prior_query=true` 的 checkpoint 和 TRM 严重不一致。

### commit 修改措施

#### Backend 初始化

在 `HSAAttnBackend.__init__` 中：

- 读取 `enable_prior_query`；
- 当 `enable_prior_query and enable_lmk_q_proj` 时自动创建：
  - `LandmarkLmkKPool`
  - `ReqToChunkPool`

#### Prefill 写 pool

在 `FlashHSAInnerXHierarchicalSparseAttention.forward` 中：

- 如果 `_per_qhead_lmk_k_active` 且是 extend：调用 `_maybe_write_chunk_lmk_k(...)`。
- `_maybe_write_chunk_lmk_k` 计算：
  - per-q-head `lmk_k`
  - `prior_b`
  - 写入 `lmk_k_pool[layer, slot]`
  - 记录 `(req, chunk) -> slot`

#### Decode 写 pool

- decode LMK boundary 时调用 `_maybe_write_decode_chunk_lmk_k(...)`；
- 从 KV cache/current K 构造当前完成 chunk 的 `lmk_k/prior_b`；
- 写入 pool。

#### Selection 使用 prior

- decode selector：`scores_pqh += per_qhead_prior_b.permute(...)`。
- extend selector：`scores_pqh += prior_b_ext.transpose(...)`。

---

## 7. per-q-head selection 的 max-over-G 语义

### 涉及文件

- `python/sglang/srt/layers/attention/hsa/selector.py`
- `python/sglang/srt/layers/attention/hsa_backend.py`

### 涉及组件

- `select_topk_pages_decode_fused`
- per-q-head selection scores
- `_last_per_qhead_scores_decode`
- `_last_per_qhead_scores_extend`
- chunk weight fusion

### TRM 里的语义

对于 GQA：

```text
h_q = h_kv * G
```

TRM 的 per-q-head LMK K selection 语义是：

1. 先算每个 q-head 对每个 chunk 的 score：

```text
scores_pqh: [B, h_q, C]
```

2. selection 的 page index 是 kv-group 共享的，需要对每个 kv group 内的 G 个 q-head 做 max：

```text
scores_kv = max_over_G(scores_pqh): [B, h_kv, C]
```

3. 对 `scores_kv` 做 top-k，得到 `[B, h_kv, K]` 的 page index。
4. 再把 index broadcast 回 q-head，取每个 q-head 自己在这些 page 上的 score，供后续 merged softmax/chunk weight 使用。

### SGLang 未对齐点

旧路径可能使用 shared-K path 或 kernel group sum 语义，这不等价于 TRM 的 max-over-G selection：

- page index 可能不同；
- per-q-head score 也可能被错误聚合；
- 后续 chunk weight fusion 输入不对。

### commit 修改措施

在 `selector.py` 的 per-q-head path 中显式实现：

- `scores_pqh = einsum(q, cand_repr)`；
- 加 `per_qhead_prior_b`；
- `scores_kv_sel = scores_pqh.view(...).max(dim=G)`；
- top-k 得到 kv-group page index；
- index broadcast 回 h_q；
- gather 每个 q-head 自己的 scores；
- 把 per-q-head scores side-channel 给 backend，后续 fusion 使用。

---

## 8. LMK pool slot 多层复用语义

### 涉及文件

- `python/sglang/srt/models/flash_hsa.py`
- `python/sglang/srt/mem_cache/landmark_pool.py`

### 涉及组件

- `LandmarkLmkKPool`
- `ReqToChunkPool`
- `_maybe_write_chunk_lmk_k`
- `(req_idx, chunk_idx) -> slot`
- `lmk_k_pool[layer, slot]`

### TRM 里的语义

对同一个 logical chunk：

- 所有 layer 都应该共享同一个 chunk slot；
- `lmk_k_pool` 通过 `layer` 维区分不同层的数据：

```text
lmk_k_pool[layer, slot, h_q, d]
```

也就是说：

```text
(req, chunk) -> slot
```

这个映射必须是跨 layer 稳定的。

### SGLang 未对齐点

如果每个 layer 写同一个 chunk 时都重新分配 slot，就会发生：

1. layer0 给 chunk c 分配 slot A；
2. layer1 又给 chunk c 分配 slot B；
3. `(req, chunk) -> slot` 被覆盖成 B；
4. layer0 后续读取时按 slot B 读，读到的是空值或其他 layer 的错误位置。

这会直接污染 prefill/selection，导致 Step 0 就错。

### commit 修改措施

`_maybe_write_chunk_lmk_k` 中改为：

- 先 `existing_slots = req_to_chunk.gather_slots(...)`；
- 已有 slot 的 chunk 复用旧 slot；
- 只有 `missing` 的 chunk 才 `lmk_k_pool.alloc(...)`；
- 只对 missing chunk 更新 `req_to_chunk.assign(...)`；
- 每个 layer 写自己的 `lmk_k_pool.set(layer_id, slots, ...)`。

---

## 9. chunked prefill 下 stable `rid` 状态继承语义

### 涉及文件

- `python/sglang/srt/mem_cache/landmark_pool.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/models/flash_hsa.py`

### 涉及组件

- `ForwardBatch.reqs`
- `ReqToChunkPool.bind_request`
- `rid_to_req_idx`
- `req_idx_to_rid`
- chunked prefill
- `req_pool_idx`

### TRM 里的语义

一个逻辑 request 的 chunk 状态应该随 request 走，而不是随某一次调度分配到的 `req_pool_idx` 走。

chunked prefill 下，同一个 logical request 可能分多次 prefill，每次可能使用不同的 `req_pool_idx`。前面 chunk 已经写入的 LMK slot 映射必须能被后续 chunk 继续看到。

### SGLang 未对齐点

原逻辑只按 `req_pool_idx` 存 `(req, chunk) -> slot`。如果 chunked prefill 后续阶段换了新的 `req_pool_idx`：

- 前面 chunk 的 slot 映射丢失；
- extend selection 找不到历史 chunk 的 per-q-head `lmk_k/prior_b`；
- fallback 或读空 slot，导致 selection 错。

### commit 修改措施

- `ForwardBatch` 新增：

```python
reqs: Optional[List] = None
```

- `ForwardBatch.init_new(...)` 时传入 `batch.reqs`。
- `ReqToChunkPool` 新增：
  - `rid_to_req_idx`
  - `req_idx_to_rid`
  - `bind_request(rid, req_idx, prefix_chunks)`
- 在 `_maybe_write_chunk_lmk_k` 中：
  - 从 `forward_batch.reqs[0].rid` 获取 stable request id；
  - 调用 `req_to_chunk.bind_request(stable_rid, req_idx, already_done)`；
  - 如果同一 rid 换了 req row，把 prefix chunks 的 slot 映射复制过去。

---

## 10. decode LMK 插入 off-by-one 语义

### 涉及文件

- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`

### 涉及组件

- `Req.hsa_should_insert_lmk_next`
- `Req.hsa_decode_postprocess_sampled_token`
- `SchedulerOutputProcessorMixin`
- `batch.seq_lens`
- internal LMK decode step

### TRM 里的语义

HSA decode 中，LMK 是 engine-visible 的内部 token。正确边界是：

```text
当当前 engine-visible KV 长度到达 chunk 最后一个 real-token slot 时，下一步插入 LMK。
```

判断应基于真实 engine-visible sequence length，而不是只基于用户侧 `fill_ids` 长度。

### SGLang 未对齐点

原代码用：

```python
len(req.fill_ids) % page_size == page_size - 1
```

来判断是否插 LMK。

但 `fill_ids` 是 request 侧维护的序列，可能和 scheduler 当前 `batch.seq_lens[i]` 的真实 engine-visible KV 长度存在 off-by-one 或状态滞后。

这会导致：

- LMK 插早；或
- LMK 插晚；
- internal LMK sampled token discard/resume 逻辑和 KV cache 实际内容错位。

### commit 修改措施

- `hsa_should_insert_lmk_next` 改为接受 `current_seq_len`：

```python
def hsa_should_insert_lmk_next(self, current_seq_len: int | None = None) -> bool:
    seq_len = int(current_seq_len) if current_seq_len is not None else len(self.fill_ids)
    return (seq_len % page_size) == (page_size - 1)
```

- `hsa_decode_postprocess_sampled_token` 也接受 `current_seq_len`。
- `scheduler_output_processor_mixin.py` 调用时传入：

```python
int(batch.seq_lens[i].item())
```

修复后，LMK internal step 的插入边界以实际 KV/cache 长度为准。
