# HSA CUDA Graph 开启后一致性 fail 修复梳理



## 1. CUDA Graph capture / replay 语义

### 涉及文件

- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/layers/attention/hsa_backend.py`

### 涉及组件

- `CudaGraphRunner.capture_one_batch_size`
- `CudaGraphRunner.replay_prepare`
- `HSAAttnBackend.init_forward_metadata_capture_cuda_graph`
- `HSAAttnBackend.init_forward_metadata_replay_cuda_graph`
- `HSAMetadata`

### SGLang CUDA Graph 里的语义

SGLang 捕获 decode graph 时，会先执行若干次 `run_once()` dry-run，然后再真正 capture：

```python
for _ in range(2):
    run_once()
out = self._capture_graph(graph, pool, stream, run_once)
```

这意味着：

1. capture 前 Python forward 会先跑几次；
2. HSA decode selection 的 Python 逻辑会在 dry-run 阶段执行；
3. dry-run 中写到 Python object / metadata object 上的 tensor cache，可能在真正 capture 时继续被复用；
4. replay 时 Python forward 不再执行，graph 只会重放 capture 时记录下来的 device op 和当时绑定的 tensor 地址。

### 本 case 的关键约束

CUDA Graph 下，任何依赖当前 decode step 的动态信息：

- `input_ids`
- `positions`
- `seq_lens`
- `page_table_1`
- `req_pool_indices`
- HSA candidate pages
- HSA LMK slot mapping
- gathered `lmk_k/prior_b`

都必须满足以下条件之一：

1. graph replay 前写入稳定地址的 persistent buffer；或
2. graph 内部每次 replay 根据 persistent buffer 重新计算。

不能依赖 Python object 上跨 dry-run / capture / replay 的普通 tensor cache。

---

## 2. HSA decode selection scratch cache 语义

### 涉及文件

- `python/sglang/srt/layers/attention/hsa_backend.py`
- `python/sglang/srt/layers/attention/hsa/metadata.py`
- `python/sglang/srt/mem_cache/landmark_pool.py`

### 涉及组件

- `HSAAttnBackend._run_selection_decode`
- `ReqToChunkPool.gather_slots`
- `LandmarkLmkKPool`
- `select_topk_pages_decode_fused`
- `fused_chunk_weight_per_qhead_decode`

### eager 里的优化语义

HSA decode selection 里，为了减少同一个 decode step 内不同 HSA layer 的重复 kernel，之前在 `HSAMetadata` 上缓存了这些 per-step scratch：

- `hsa_per_step_cand_page_ids`
- `hsa_per_step_cand_mask`
- `hsa_per_step_slots`
- `hsa_per_step_all_lmk_k`
- `hsa_per_step_all_prior_b`
- `hsa_per_step_hsa_window`

这些 cache 不是复用跨层 selection 结果：

- `candidate pages / mask` 只由 `seq_lens / hsa_window / page_size` 决定，本身与 layer 无关；
- `slots` 只由 `req_pool_indices + candidate pages` 决定，本身也与 layer 无关；
- `all_lmk_k / all_prior_b` 是一次性把所有 layer 的 LMK pool 按同一批 slots gather 出来，tensor 仍然保留 layer 维度，后面每个 layer 取自己的那一片；
- 真正的 top-k selection（`selected_page_ids / selected_scores`）仍然由每层自己的 `q` 和该层自己的 `lmk_k/prior_b` 计算，不能跨层复用。

所以 eager 下这个 cache 只是复用 layer-invariant scratch / batched gather，不是共享 TRM 语义里的 chunk selection 结果。

### CUDA Graph 下的语义差异

CG capture 路径下，上述 cache 不再安全：

1. dry-run 第一次执行 `_run_selection_decode`；
2. `md.hsa_per_step_*` 被 dummy capture input 填充；
3. dry-run 第二次 / 真正 capture 时，Python 看到 cache 已存在，直接复用；
4. 真正被 capture 进 graph 的是“读取 stale scratch”，不是“根据当前 `cache_seqlens/page_table_1` 重算 scratch”；
5. replay 时虽然外层 metadata buffer 被刷新，但 graph 内 selection 仍读 capture 阶段冻结的 candidate / slot / LMK gather 结果。

---

## 3. SGLang 未对齐点

### 涉及文件

- `python/sglang/srt/layers/attention/hsa_backend.py`

### 未对齐点

原实现把 eager 下的 per-step Python tensor cache 原样带到了 CG metadata 路径：

```python
cached_pi = getattr(md, "hsa_per_step_cand_page_ids", None)
cached_cm = getattr(md, "hsa_per_step_cand_mask", None)
...
slots = getattr(md, "hsa_per_step_slots", None)
all_lmk_k = getattr(md, "hsa_per_step_all_lmk_k", None)
all_prior_b = getattr(md, "hsa_per_step_all_prior_b", None)
```

这在 eager 下是性能优化，但在 CUDA Graph 下违反了 replay 语义：

- 这些 tensor cache 依赖 `seq_lens/page_table_1/req_pool_indices`；
- 但它们不是 replay 前原地更新的 persistent buffer；
- 也不是 graph 内每次 replay 重新计算的结果；
- 它们会被 capture dry-run 污染并冻结。

### 表现

因此 CG 下会出现：

- replay 前 `input_ids / positions / seq_lens / page_table_tail` 都看起来正常；
- 但 selection 用的 candidate/slot/lmk_k scratch 已经错；
- logits 从最早几步就偏；
- decode 生成进入错误轨迹。

---

## 4. 问题定位

禁用 CG 路径下 `md.hsa_per_step_*` cache 后，测试恢复：

```text
Token 匹配率: 500/500 (100.0%)
Decode token+top2 总一致率: 500/500 (100.0%)
```

这确认了根因：

> HSA decode selection 的 per-step Python tensor cache 被 CUDA Graph capture dry-run 污染，导致 graph replay 使用 stale candidate / slot / lmk_k scratch。

---

## 5. 修改措施

### 涉及文件

- `python/sglang/srt/layers/attention/hsa_backend.py`

### 核心修改

在 `_run_selection_decode` 中识别当前是否为 HSA CUDA Graph metadata：

```python
cg_buf = getattr(self, "_cg_page_table_1", None)
is_cg_metadata = cg_buf is not None and page_table_1.data_ptr() == cg_buf.data_ptr()
```

如果是 CG metadata，则不再读取或写入这些 Python-side scratch cache：

- `hsa_per_step_cand_page_ids`
- `hsa_per_step_cand_mask`
- `hsa_per_step_hsa_window`
- `hsa_per_step_slots`
- `hsa_per_step_all_lmk_k`
- `hsa_per_step_all_prior_b`

具体行为：

1. eager 路径：
   - 继续使用原 cache；
   - 只复用 layer-invariant scratch / all-layer batched gather，不复用各层 top-k selection 结果。

2. CG 路径：
   - 每次 graph capture 时把 candidate / mask / slots / all-layer LMK gather 的计算捕获进 graph；
   - replay 时根据 replay 前刷新的 persistent metadata buffer 重新计算；
   - 避免 dry-run stale cache 被 capture。

### 该修复的语义

这个修复不是单纯 debug workaround，而是 correctness fix：

- eager 可以用 Python tensor cache；
- CUDA Graph 不能用依赖当前 step 动态 metadata 的 Python tensor cache；
- CG 只能读稳定地址的 persistent buffer，或在 graph 内重新计算动态结果。

---


## 6. 当前修复是否可以长期使用

### 可以作为 correctness fix 保留

当前修复的原则是正确的：

> CG 路径禁用依赖动态 metadata 的 Python-side tensor cache，避免 dry-run / capture / replay 之间复用 stale scratch。

这不是临时 workaround，而是符合 CUDA Graph 语义的安全修复。

### 代价

CG 路径会少用一部分原本的 per-step cache 优化：

- candidate / mask 每次 graph replay 内重新计算；
- slots 每次重新 gather；
- all-layer `lmk_k/prior_b` 每次重新 index_select；
- 可能牺牲一部分 CG decode 性能。

不过 eager 路径仍保留 cache，所以 eager 性能不受影响。

---

## 7. 后续更优修复策略

### 方案 A：保留当前修复

优点：

- correctness 明确；
- 改动小；
- eager 性能不受影响；
- CG 一致性已验证通过。

缺点：

- CG 下无法复用原 layer-invariant scratch / all-layer batched gather 的全部性能收益。

适合当前阶段作为主线修复。

### 方案 B：把 HSA selection scratch cache 改成 CUDA Graph persistent device buffers

更长期、性能更优的方案：

1. 在 `HSAAttnBackend.init_cuda_graph_state` 中预分配固定地址 buffer：
   - `cg_cand_page_ids`
   - `cg_cand_mask`
   - `cg_slots`
   - `cg_all_lmk_k`
   - `cg_all_prior_b`
2. 在 `init_forward_metadata_replay_cuda_graph` 中用 Triton kernel 原地更新这些 buffer；
3. graph capture/replay 只读这些稳定地址；
4. 不再把动态 tensor 挂在 `md.hsa_per_step_*` Python 字段上；
5. layer 内仍能复用同一批 scratch。

优点：

- correctness 和 CG 语义都正确；
- 可以恢复大部分 selection scratch 复用性能；
- 更适合长期高性能版本。

缺点：

- 工程量更大；
- 需要处理 `C_max`、layer 数、pool layout、buffer shape、dtype、TP 等边界。

### 不推荐方案：只在 metadata 初始化时清 Python cache

例如每次 `_build_hsa_cg_metadata` 后清空 `md.hsa_per_step_*`。

不推荐原因：

- capture 前 dry-run 与真正 capture 之间仍可能重新填 cache；
- replay 时 Python forward 不执行，无法依赖 Python cache 生命周期；
- 仍容易出现 stale tensor 被 capture 的风险。

