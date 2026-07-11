<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

## 🔍 This fork: HSA — Hierarchical Sparse Attention

This is a fork of SGLang that adds an **HSA (Hierarchical Sparse Attention)**
backend for **long-context** inference. HSA splits each sequence into fixed-size
chunks, uses per-chunk *landmark* keys to select the top-`k` most relevant chunks
per query (plus a local sliding window), and attends only to those — cost grows
as `O(N · topk · page_size)` instead of dense `O(N²)`, while staying closely
aligned with the reference model. At 345M / 512K context it is **~13.5× faster
prefill and ~15.7× faster decode** than full attention, with parity around 16K.

> **Note:** this fork is intended for **speed benchmarking** of HiLS-Attention, not
> fully-aligned production serving. Its alignment with the reference model is close
> but not exact — the reference masks the inserted landmark (LMK) tokens in its
> sliding-window (SWA) layers, which this backend does not yet replicate.

HSA is the SGLang serving backend for **HiLS-Attention** — see the paper
[HiLS-Attention (arXiv:2607.02980)](https://arxiv.org/pdf/2607.02980) and the main
repository [Tencent-Hunyuan/HiLS-Attention](https://github.com/Tencent-Hunyuan/HiLS-Attention)
for the method, training code, and reference implementation. This fork ports it
into SGLang for inference.

Full details, config fields, and the benchmark table:
[`python/sglang/srt/layers/attention/hsa/README.md`](python/sglang/srt/layers/attention/hsa/README.md).

### Run inference with HSA

**Step 1 — Download the weights.** Grab the released HiLS-Attention checkpoint
from the Hub:

```bash
huggingface-cli download tencent/HiLS-Attention-7B \
    --local-dir ./HiLS-Attention-7B
```

**Step 2 — Transform the checkpoint.** The released weights use the upstream
`HiLS*` / `hils_*` naming; the SGLang HSA backend expects the `HSAForCausalLM` /
`FlashHSAConfig` schema. The weight tensors are already compatible — only
`config.json` needs translating — so the converter just rewrites the config and
symlinks the weights (pass `--copy` for a self-contained copy):

```bash
python scripts/convert_hils_checkpoint.py \
    --src ./HiLS-Attention-7B \
    --dst ./HiLS-Attention-7B-sglang
```

**Step 3 — Run inference.** Point the `hsa` backend at the converted directory
and set the page size to the model's chunk size (**64**). Server:

```bash
python -m sglang.launch_server \
    --model-path ./HiLS-Attention-7B-sglang \
    --attention-backend hsa \
    --page-size 64 \
    --trust-remote-code
# then query the OpenAI-compatible endpoint at http://localhost:30000
```

Your own prompts (offline, no server) — `scripts/run_hsa_infer.py` wraps the
offline engine and sets the HSA knobs for you. Prompts come from positional args,
`--prompt`, a `--prompt-file` (one per line), or stdin:

```bash
python scripts/run_hsa_infer.py --model-path ./HiLS-Attention-7B-sglang \
    "The capital of France is" "Water is made of hydrogen and"

python scripts/run_hsa_infer.py --model-path ./HiLS-Attention-7B-sglang \
    --prompt-file prompts.txt --max-new-tokens 128
```

It auto-sizes `context_length` to your prompts (override with `--context-length`
for very long inputs).

Offline / single batch (benchmarking):

```bash
python -m sglang.bench_one_batch \
    --model-path ./HiLS-Attention-7B-sglang \
    --attention-backend hsa \
    --page-size 64 \
    --input-len 131072 --output-len 32 --batch-size 1 \
    --trust-remote-code
```

> `--page-size` **must** equal the model's `chunk_size` (64). Both GQA (`G>1`) and
> MHA (`G=1`, e.g. the 7B checkpoint) prior-query checkpoints are supported. For
> bit-exact selection (consistency testing) set `SGLANG_HSA_HEADWISE_TOPK_SOFTMAX=1`;
> the default is the faster max-pool selection path.

#### Runtime behavior specific to HSA

The backend auto-configures two things at launch (you'll see a log line for each);
you don't need to set any flags:

- **Single-sequence prefill.** The HSA prefill kernels process one sequence at a
  time, so the backend sets `prefill_max_requests=1` — requests are prefilled one
  at a time. **Decode is unaffected and stays fully batched**, so throughput for
  many concurrent generations is not impacted.
- **Overlap scheduler disabled.** HSA decode interleaves virtual *landmark* (LMK)
  tokens whose next input must be chosen synchronously from the current sequence
  length; the overlap scheduler launches the next forward before that decision is
  made. The backend therefore sets `disable_overlap_schedule=True`. This does not
  change single-batch latency (the benchmark numbers above are unaffected); it can
  slightly reduce online-serving decode throughput because per-step CPU overhead is
  no longer hidden behind GPU compute.

### Reproduce the speed benchmark

The sweep benches HSA vs dense (full) attention across 8K–512K context on a
single GPU with dummy weights (no checkpoint needed):

```bash
GPU=0 bash dev/sweep_8k_512k.sh   # writes /tmp/sweep_results.txt
```

Kernels live in `dev/hsa-kernel-main/ops/` (tilelang); `tilelang` must be
installed and importable.

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Documentation</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Join Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Dev Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

## News
- [2025/12] SGLang provides day-0 support for latest open models ([MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/), [Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/), [Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213), [LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/), [MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)).
- [2025/11] 🔥 SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)).
- [2025/10] 🔥 SGLang now runs natively on TPU with the SGLang-Jax backend ([blog](https://lmsys.org/blog/2025-10-29-sglang-jax/)).
- [2025/09] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).
- [2025/08] SGLang provides day-0 support for OpenAI gpt-oss model ([instructions](https://github.com/sgl-project/sglang/issues/8833))
- [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).

<details>
<summary>More</summary>

- [2025/10] PyTorch Conference 2025 SGLang Talk ([slide](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf)).
- [2025/10] SGLang x Nvidia SF Meetup on 10/2 ([recap](https://x.com/lmsysorg/status/1975339501934510231)).
- [2025/06] SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a high-performance serving framework for large language models and multimodal models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 400,000 GPUs worldwide.

## Getting Started
- [Install SGLang](https://docs.sglang.io/get_started/install.html)
- [Quick Start](https://docs.sglang.io/basic_usage/send_request.html)
- [Backend Tutorial](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [Contribution Guide](https://docs.sglang.io/developer_guide/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [GB200 rack-scale parallelism](https://lmsys.org/blog/2025-09-25-gb200-part-2/).

## Adoption and Sponsorship
SGLang has been deployed at large scale, generating trillions of tokens in production each day. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia.
As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 400,000 GPUs worldwide.
SGLang is currently hosted under the non-profit open-source organization [LMSYS](https://lmsys.org/about/).

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us
For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at sglang@lmsys.org

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
