### HSA 权重（`hsa_weights` / `chunk_weights`）笔记

#### 1) dense vs sparse：权重的数学含义是否一样？
- **不完全一样（至少在官方代码路径里不是同一个对象/同一个计算路径）**。
- **sparse 路径（`hsa_mode == "sparse"`）**：
  - `topk_func(q_norm, chunk_lmks, lse_sum, ...)` 先返回 `indices, scores`
  - Python 侧再做：
    - `cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1), ...], dim=-1)`
    - `chunk_weights = softmax(cat_scores, dim=-1)`
  - 然后把 `chunk_weights` 作为融合权重传入 `hsa_func(...)`
  - 这里的 `chunk_weights` 本质上是 **“HSA 选中的 K 个 chunk + SWA 分支” 的融合归一化权重**（而不是块内 softmax）。
- **dense 路径（`hsa_mode == "dense"`）**：
  - `hsa_func(...)` 在 kernel 内部直接返回 `chunk_weights`（以及 `lse_idx`）
  - 等价于把“dense chunk 集合 + SWA 分支”的融合归一化放进 kernel 做。

#### 2) 为什么官方 sparse 要走 torch 来算 `chunk_weights`？
这是一个很典型的工程取舍：sparse 把工作拆成两步
- **(a) topk 选块**（专用算子/实现复杂）
- **(b) 融合归一化 softmax（K 通常很小）** + 后续加权求和

把 (b) 放在 torch 侧的常见原因：
- **迭代快/灵活**：`enable_softmax1`、`lse_sum` 拼接位置（`swa_weight_idx`）、各种消融/调试更容易改。
- **降低 kernel 复杂度**：把 topk + softmax + 融合 + 输出全塞进一个 kernel，开发/调试成本高、数值稳定更难。
- **训练/研究代码优先**：正确性和可读性优先于极致性能；sparse 里 `softmax(K+1)` 的算量本身不大。
- **dense 更需要 kernel 化**：dense 没有 topk，块数更大，如果权重还在 torch 算会更慢，所以 dense 路径更倾向“一锅端 kernel”。

#### 3) 我们是否有必要实现 `hsa_weights` 的 kernel？
- **短期不“必须”**：
  - 先把 paged HSA decode kernel 接进推理跑通、对齐输出更关键。
  - K 小时，torch softmax/融合的 FLOPs 不大；主要成本是 kernel launch overhead，但可以先接受。
- **中长期非常有价值（尤其做成 SGLang 真正可用/高性能）**：
  - **CUDA Graph / 固定形状**：把权重计算并进 kernel 更利于 capture 稳定、减少分支和额外 launch。
  - **端到端融合机会**：最终想做 fused SWA+HSA（类似官方 dense 那种），权重不进 kernel 很难真正融合。
  - **降低 decode latency**：每步 decode 都会算一次，哪怕小 softmax 也有 launch 开销。
  - **数值一致性更可控**：把 `lse_sum`、`enable_softmax1`、融合逻辑固定在单 kernel 内更容易保证一致。

#### 4) 建议路线
- **先不做 weights-kernel**（torch 侧算 `chunk_weights`），把 paged HSA decode attention kernel 端到端跑起来。
- 等你开始做 **fused SWA/HSA** 或追求 **cuda graph + 性能** 时，再把 `hsa_weights`/融合 softmax 并入 kernel（或单独小 kernel / 或融合进最终 fused kernel）。
