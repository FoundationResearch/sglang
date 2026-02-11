### 我们已经“确认了”的拼图（有代码断言/端到端测试覆盖）

- **InnerX ultra split-head 数学语义（decode）**  
  - runner-like decode E2E 用 torch reference 逐项复现 SWA/HSA/selection/stitching/o_proj，并做数值对齐。

- **HSA decode kernel 路径可用 + selection 正确性**  
  - `HSAAttnBackend` 的 decode 走专用 InnerX 路径，包含 LMK page selection，测试覆盖并对齐选择结果。

- **RadixAttention / prefix cache 复用（含 LMK）**  
  - `test_innerx_radix_kv_reuse_end_to_end_gpu.py` 覆盖：req2 复用 req1 的 page-aligned prefix（包含 LMK），并断言共享 prefix KV 不被覆盖。  
  - 现在 decode 序列也强制跨过 LMK slot（出现 internal LMK step）。

- **Continuous batching 兼容性 + correctness**  
  - scheduler 级 E2E：真实 `Scheduler -> TpWorker -> ModelRunner` 跑 `(r1,r2) -> (r2,r3)` 动态换人。  
  - correctness：`r2` 单跑 vs 动态换人生成 token 序列完全一致。  
  - 现在也断言 `r2` 在 decode 过程中跨过 LMK slot（internal LMK step）。

- **LMK 的“你指定的语义”已落地**  
  - internal LMK step 存在、LMK 步输出丢弃，但可见 token 链不断（63 的输出作为 65 的输入）。  
  - 单测 `test_hsa_lmk_runtime_injection_gpu.py` 覆盖。

- **只保留 InnerX 这一种 HSA 变体**  
  - 配置注册、模型/后端路径都以 `flash_hsa_innerx` 为唯一支持目标（旧兼容逻辑已移除/不再使用）。

---

### 我们还“没确认/不完全确认”的拼图（缺测试或语义未完全对齐）

- **prefill/extend 的“官方 ultra HSA”语义是否一致**  
  - 目前 extend 更多是 dense fallback（而非完全按 ultra 的 HSA 稀疏选择去做 prefill）。这在功能上能跑，但未证明“数学语义与 ultra 完全一致”。

- **多层模型、更多真实 shape/参数覆盖**  
  - 测试用的是 tiny config（1 layer 等）。多层/更大 head/topk/window 配置下的边界（尤其 LMK/page 边界、长上下文）还没系统验证。

- **更复杂的 radix 场景**  
  - 目前覆盖“完全相同 prompt”的复用；但“分叉 prompt / 不同 tail / 多分支共享”这种更贴近真实前缀树分叉的场景还没专门测。

- **overlap schedule / spec decoding 等组合**  
  - 我们主要覆盖 non-overlap、spec 关闭的主路径；这些组合打开后与 HSA/LMK 的交互还未验证。

如果你想把“未确认”里最关键的一块补齐，我建议下一步优先做 **prefill/extend 的 ultra 语义对齐**（至少在小规模上给出 torch reference 对齐测试），这样整个 HSA 生命周期（prefill→decode）就闭环了。