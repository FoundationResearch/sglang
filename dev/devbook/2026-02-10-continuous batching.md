## SGLang 是否支持 continuous batching？

**支持。**SGLang 的 SRT 推理服务端本身就是“调度器 + 动态 batch”的模式：请求会被持续加入/移出 batch，scheduler 每个 step 组装 `ForwardBatch`（包含不同 request 的 `req_pool_indices/seq_lens/out_cache_loc` 等），交给 `ModelRunner` 做一次 forward（extend 或 decode），这就是典型的 continuous batching。

## 我们的 HSA（InnerX ultra）实现是否支持 continuous batching？

**支持 decode 侧的 continuous batching，并且我们已经用端到端测试覆盖了“同一 decode batch 里多个 request”的场景。**原因是：

- **接口层面**：HSA 是通过 `RadixAttention -> HSAAttnBackend` 接入的，backend 的输入就是 `ForwardBatch`，天然支持 batch 内多 request（用 `req_pool_indices` 去查各自的 page_table / KV slot 映射，用 `seq_lens` 表示各自长度）。
- **实现层面**：`HSAAttnBackend.init_forward_metadata` 是按 batch 构建 `page_table_1/real_page_table`，decode 时按 batch 做 selection、SWA/HSA 分支计算并 stitch，**不依赖“所有序列同长/固定 batch”**。
- **测试层面**：我加的 radix 复用 E2E 测试里，decode 阶段就是 `batch_size=2` 同时跑两条 request（共享 prefix KV、各自继续分配新 token slot），这就是 continuous batching 的核心运行形态之一。

### 需要说明的边界
- **extend/prefill**：目前 HSA 的 `forward_extend` 走 dense backend（TritonAttnBackend），所以 continuous batching 的 prefill/extend 也能跑；但它不是“专用 HSA prefill kernel”，而是 dense attention 预填充 + KV 写入，decode 才进入 InnerX split-head HSA 路径。
- **chunked prefill / MIXED**：如果你们启用把 EXTEND+DECODE 混在一个 batch 的模式（ForwardMode.MIXED），HSA backend 仍会按 SGLang 的通用分发逻辑走到 extend 路径；这块目前没有专门的 HSA MIXED 优化，但语义上应当可运行。

如果你希望我进一步把“真正 scheduler 驱动的 server 推理循环”（而不是 runner-like stub）也纳入测试覆盖，我也可以再加一条更上层的 integration test（代价是更重、依赖更多 server 组件初始化）。

证明 HSA + RadixAttention + paged KV cache 在“动态换人”的 batch 形态变化中不会崩：同一个 Scheduler/ModelRunner 生命周期里经历 prefill -> decode、batch 成员变化（{r1,r2} -> {r2,r3}），HSA 后端 forward（含选择/分页 KV 读取）能持续正确执行。

一个最小例子：确实是 prefill 和 decode 交叉 跑的（同一个 scheduler loop 里）
假设 max_running_requests=2，有三个请求 r1,r2,r3，它们都要生成若干 token。
t0：waiting_queue=[r1,r2]，running_batch=[]
scheduler 先挑一批做 prefill/EXTEND → 跑 EXTEND(r1,r2)（把 prompt KV 写进 cache，通常也会产出 1 个 token）
处理结果后：running_batch 会在下一轮被更新为包含 r1,r2
t1：此时 running_batch≈[r1,r2]，waiting_queue 可能又来了 r3
scheduler 会看资源：
如果还能塞下新请求（prefill token budget、running request budget 等允许），就会先跑 EXTEND(r3)
否则就直接跑 DECODE(r1,r2)
这就是“交叉”：同一条主循环里每一轮都在 二选一（优先 prefill，没得 prefill 才 decode）
t2：r1 结束了（max_new_tokens 或 stop），r3 还在 waiting 或刚 prefill 完
scheduler 会把 finished 的从 running_batch 里 filter 掉，然后把新 prefill 完的请求 merge 进来
所以 decode batch 会从 {r1,r2} 变成 {r2,r3}
回答你问的“是不是交叉跑”
是的，不是“把所有 prefill 全做完再开始 decode”，而是每个 step 都可能：
跑一批 prefill/extend（把新请求接入系统、填 KV），或者
跑一次 decode（推进 running_batch）
这也是 continuous batching 能做到“新请求随到随进、老请求不断生成”的根本原因。