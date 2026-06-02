"""Verify HSA CG (with R26-R35 opts) produces same logits as eager HSA at long ctx.

For each context length L in [8K, 16K, 32K, 64K, 128K, 256K]:
  1. Run sglang HSA with --disable-cuda-graph at LEN=L → eager logits
  2. Run sglang HSA with CG enabled at LEN=L → CG logits
  3. Compare per-token logits: max abs diff, max rel diff, cosine sim

Uses real HSA-345M weights (NOT dummy) so logits are deterministic and
numerically meaningful.  Single batch=1 request, fixed seed input tokens.

Pass criterion: max abs diff < 5e-3 for bf16 path (one bf16 ULP at typical
magnitude ~10).  CG and eager should be effectively bit-identical except for
slight kernel-scheduling-induced fp non-associativity.
"""
import os
import sys
import argparse

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import torch
import torch.distributed as dist


def setup_dist():
    """Init torch.distributed with single rank if not already initialized."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)


def run_one(L: int, use_cg: bool, seed: int = 42):
    """Run a single prefill+decode through sglang HSA and return last-token logits.

    L = input length (prefill tokens).
    use_cg = whether to enable CUDA graph for decode.
    """
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.configs.model_config import ModelConfig

    cuda_graph_flag = [] if use_cg else ["--disable-cuda-graph"]
    sys.argv = [
        "test",
        "--model-path", "/home/hal-alex/workspace/hsa345m_real",
        "--tp", "1",
        "--attention-backend", "hsa",
        "--mem-fraction-static", "0.20",
        "--max-total-tokens", "1000000",
        "--cuda-graph-max-bs", "1",
        "--trust-remote-code",
        "--context-length", str(L + 200),
        *cuda_graph_flag,
    ]
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    server_args.check_server_args()

    setup_dist()
    torch.manual_seed(seed)

    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0, tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        nccl_port=29501,
        server_args=server_args,
    )

    # Fixed-seed input tokens of length L.
    g = torch.Generator(device="cpu").manual_seed(seed)
    VS = int(model_config.vocab_size)
    input_tokens = torch.randint(5, VS - 5, (L,), generator=g).cuda()

    # ... For brevity, this scaffolding ends here — finishing the runner-driven
    # path is a several-hundred-line bench setup.  Instead, use the simpler
    # bench_one_batch path below.
    raise NotImplementedError(
        "Use bench_one_batch path in dev/test_long_ctx_cg_vs_eager_via_bench.py instead."
    )


if __name__ == "__main__":
    print("This file is a scaffold — use the bench-based comparison script:")
    print("  python dev/test_long_ctx_cg_vs_eager_via_bench.py")
