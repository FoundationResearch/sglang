我会先把这次对齐 InnerX ultra 的代码改动与测试覆盖面总结清楚（哪里已经“端到端”验证、哪里还没到“加载真实 checkpoint + generate”）。然后按你的要求，把旧的 FlashHSA（非 InnerX split-head）路径和对应测试/代码删掉，只保留 InnerX ultra 这一套，并在 `alexsg` 环境里把剩下的 HSA 测试跑一遍确保不破。

[4 tools called]

我会把代码简化成**只支持 InnerX ultra split-head HSA**：模型端只保留 `FlashHSAInnerX*`，backend 端强制走 split-head stitch（不再支持 merged/旧 HSA-only），并删除旧模式相关的测试文件；然后在 `alexsg` 环境里跑剩下的 HSA 测试集确认全绿。

[44 tools called]

## 修改总结（只保留 InnerX ultra split-head HSA）

### **1) 模型侧（`python/sglang/srt/models/flash_hsa.py`）**
- **实现了 InnerX ultra 的结构**（对齐 `@dev/InfiniteLongLM/models/FlashHSA/modeling_hsa_innerx_ultra.py`）：
  - **显式多套投影矩阵**：`q_proj/k_proj/v_proj`（SWA 分支）、`hsa_q_proj/hsa_k_proj/hsa_v_proj`（HSA 分支）、`lmk_q_proj`（selection query）、可选 `gate_proj`。
  - **RoPE 对齐 ultra**：只对 **SWA 分支**做 RoPE；HSA 分支与 `lmk_q` 不做 RoPE。
  - **split-head 拼接语义对齐 ultra**：输出是 head 维度 `cat([swa_heads, hsa_heads])`，不是 merged/softmax 融合。
- **删除了旧的非 InnerX FlashHSA 模型路径**
  - 移除了 `FlashHSAModel`（Qwen3-like / qkv_proj split 的那套）。
  - `HSAForCausalLM` 现在**只允许** config 里有 `hsa_heads` + `hsa_qk_ratio`，否则直接报错（明确“只支持 InnerX ultra”）。
- **权重加载简化为只支持 InnerX layout**
  - `load_weights` 只按显式参数名直接 load（`q_proj` / `hsa_q_proj` / `lmk_q_proj` 等），不再兼容 `qkv_proj/gate_up_proj` stacked 逻辑。

### **2) Config 侧（`python/sglang/srt/configs/flash_hsa.py`）**
- `FlashHSAConfig.model_type` 改为 **`flash_hsa_innerx`**（只保留这一种）。
- 仍支持从官方格式 `use_sliding_window/sliding_window` 映射到 split-window（因为 InnerX 配置本身就用这套字段）。

### **3) Backend 侧（`python/sglang/srt/layers/attention/hsa_backend.py`）**
- **只保留 InnerX split-head decode 路径**：
  - HSA layer 必须传入 `hsa_split_head_info` + `hsa_selection_q`（由模型层 `FlashHSAInnerXHierarchicalSparseAttention` 提供），否则直接抛错。
  - **SWA heads**：用 `flex_attention` 做 sliding-window（按 ultra 语义默认 **不排除 LMK**，可由 `swa_exclude_lmk` 控制）。
  - **HSA heads**：用 selection（LMK-K + topk）+ paged triton kernel `hsa_decode_paged_fwd(mask_last_token=True)`。
  - 输出按 head 维 `stitch` 回 `[SWA|HSA]`。
- **旧的 merged / HSA-only decode 逻辑已删除**（包括 `_get_effective_window_size`、merged softmax(cat([scores,lse])) 那一套）。

### **4) 测试删减（只保留 InnerX ultra 需要的）**
我已经把旧 HSA（merged / 非 split-head）的测试文件删除，只保留下面这些。

---

## 测试覆盖总结（在 `alexsg` 环境跑通）

我在 `conda activate alexsg` 下跑了：

```bash
pytest -q python/sglang/test/attention/hsa -vs
```

结果：**12 passed**。

### **关键“端到端”测试：InnerX ultra split-head decode**
- **文件**：`python/sglang/test/attention/hsa/test_hsa_backend_innerx_split_head_decode_gpu.py`
- **它验证了什么（端到端链路）**：
  - `RadixAttention.forward(...)` → `HSAAttnBackend.forward_decode(...)`
  - **selection 用的是 `hsa_selection_q`（对应 ultra 的 lmk_q_norm）**
  - **只用 HSA kv-head 子集做 selection**（对齐 split-head 结构）
  - SWA 分支用 sliding-window attention，HSA 分支走 paged-kernel，然后 **head-wise stitch** 输出
  - 输出与 torch reference（同样的 selection + SWA + HSA kernel ref + stitch）数值一致

这条测试是我们目前对 InnerX ultra **SGLang attention 栈层面的 decode 端到端数学正确性闭环**。

### 其它保留测试（仍然有价值）
- **kernel correctness**：`test_flashhsa_triton_hsa_decode_paged_kernel_gpu.py`（paged decode kernel 与 torch ref 对齐、LMK poison/mask_last_token）
- **position / LMK 语义**：`test_flashhsa_positions_gpu.py`
- **runtime LMK 注入 / 不可见输出 contract**：`test_hsa_lmk_runtime_injection_gpu.py`
- **Radix/prefix cache 与 LMK 序列**：`test_hsa_radix_cache_lmk_gpu.py`
- **selector 单测**：`test_hsa_selector_decode_gpu.py`

---

## 是否“end2end 确认跑通 InnerX ultra 的这种 HSA”？
**是（就 decode attention 栈而言）**：我们已经端到端覆盖了 *SGLang 实际会走的 attention runtime 路径*（`RadixAttention → HSAAttnBackend → flex_attention(SWA) + triton(HSA) → stitch`），并用 torch reference 复算每一步来对齐。

**但还没覆盖的 end2end**（如果你指“加载真实 InnerX checkpoint + 完整 generate 流程”）：
- 我们还没有加一个“加载真实 InnerX ultra checkpoint 并跑 generation”的集成测试/脚本；目前验证重点在 **attention 端到端数学正确性** + **LMK/prefix cache 语义**。

---

## 旧 HSA 已删除/不再兼容
- 代码层面：旧的 non-InnerX 模型路径与 backend merged/HSA-only 逻辑已经清掉，`HSAForCausalLM` 也会拒绝缺少 `hsa_heads/hsa_qk_ratio` 的 config。
- 测试层面：旧模式相关测试已删除，仅保留 InnerX ultra 相关测试集，并已在 `alexsg` 下全绿。

如果你下一步希望我补一个“**真实 InnerX ultra config + instantiate model + 跑一轮 decode**（不加载权重也行）”的更高层集成测试，我也可以继续往上加，但目前 attention 栈已经闭环。