"""
SGLang + HSA Bug 1 & Bug 2 复现测试脚本。

Bug 1: Chunked Prefill + Logprob 时 extend_len=0 导致 sample_indices off-by-one
Bug 2: Chunked Prefill + extend_len=0 的请求导致 RoPE 空 tensor crash

触发条件：
  1. return_logprob=True（开启 logprob 路径）
  2. 多个长 prompt 并发发送（触发 chunked prefill）
  3. 总 token 数超过 chunked_prefill_size，导致某些请求被分片
  4. 分片后某个请求在上一个 chunk 已完成所有 prefill token，
     下一个 batch 中 extend_len=0

关键参数：
  - chunked_prefill_size: 设置较小值（如 2048），使得多个长 prompt 必须分片
  - 每个 prompt 约 1000~2000 tokens，3~4 个并发就能超过 chunked_prefill_size

预期结果（修复前）：
  - Bug 1: logits_processor.py 中 sample_indices 越界 → CUDA assert 或 IndexError
  - Bug 2: flash_hsa.py 中 RoPE reshape 空 tensor → RuntimeError

用法：
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH="/path/to/sglang/python:${PYTHONPATH:-}" \\
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \\
    /root/sglang/python/.venv/bin/python code_exp/test_multi_batch_infer.py
"""

import gc
import os
import sys
import time
import traceback
from typing import Dict, List, Optional


# ======================== 配置区 ========================
DEFAULT_CHECKPOINT = "/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/lhsa-olmo3-interleave"

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT)
SGLANG_TP = int(os.environ.get("SGLANG_TP", "1"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "20"))
PAGE_SIZE = int(os.environ.get("SGLANG_PAGE_SIZE", "64"))
MAX_TOTAL_TOKENS = int(os.environ.get("SGLANG_MAX_TOTAL_TOKENS", "131072"))

# chunked_prefill_size: 设置较小值以强制触发 chunked prefill
# 默认 2048，如果 prompt 总 token 数超过此值就会分片
CHUNKED_PREFILL_SIZE = int(os.environ.get("CHUNKED_PREFILL_SIZE", "16384"))

# 构造多个长 prompt（每个约 800~1500 tokens）
# 当 3~4 个这样的 prompt 并发时，总 token 数 > chunked_prefill_size，触发分片
BASE_TEXT = (
    "In the field of artificial intelligence, large language models have revolutionized how we interact with technology. "
    "These models, trained on vast amounts of text data, can generate human-like text, answer questions, write code, and even engage in creative writing. "
    "The architecture behind most modern LLMs is the Transformer, which was introduced in the seminal paper 'Attention Is All You Need' by Vaswani et al. in 2017. "
    "The key innovation of the Transformer is the self-attention mechanism, which allows the model to weigh the importance of different parts of the input when generating each output token. "
    "However, as context lengths grow to hundreds of thousands of tokens, the quadratic complexity of standard attention becomes a bottleneck. "
    "Various approaches have been proposed to address this challenge, including sparse attention, linear attention, and hierarchical attention mechanisms. "
    "Among these, Hierarchical Sparse Attention (HSA) stands out as a promising approach that combines sliding window attention for local context with landmark-based top-k routing for global context retrieval. "
    "This hybrid approach enables efficient processing of extremely long sequences while maintaining the model's ability to attend to relevant information across the entire context. "
    "The development of natural language processing has a rich history spanning several decades. "
    "Early approaches relied on rule-based systems and hand-crafted grammars to parse and understand human language. "
    "In the 1990s, statistical methods gained prominence, with techniques like hidden Markov models and n-gram language models becoming the standard tools for tasks such as speech recognition and machine translation. "
    "The introduction of word embeddings, particularly Word2Vec by Mikolov et al. in 2013, marked a significant shift toward distributed representations of language. "
    "These dense vector representations captured semantic relationships between words, enabling models to generalize better across different linguistic contexts. "
    "GloVe, developed at Stanford, offered an alternative approach by combining global matrix factorization with local context window methods. "
    "The advent of recurrent neural networks, especially Long Short-Term Memory networks proposed by Hochreiter and Schmidhuber in 1997, brought a new paradigm for sequence modeling. "
    "LSTMs addressed the vanishing gradient problem that plagued vanilla RNNs, allowing models to capture longer-range dependencies in text. "
    "Bidirectional RNNs further improved performance by processing sequences in both forward and backward directions simultaneously. "
    "The attention mechanism, first introduced for machine translation by Bahdanau et al. in 2014, was a breakthrough that allowed models to focus on relevant parts of the input sequence when generating each output element. "
    "This mechanism eliminated the information bottleneck of fixed-length context vectors and dramatically improved translation quality for longer sentences. "
    "Building on this foundation, the Transformer architecture completely replaced recurrence with self-attention, enabling much greater parallelism during training. "
    "The multi-head attention mechanism in Transformers allows the model to jointly attend to information from different representation subspaces at different positions. "
    "Each attention head can learn to focus on different types of relationships, such as syntactic dependencies, semantic similarities, or positional patterns. "
    "The feed-forward network in each Transformer layer applies a position-wise transformation that adds non-linearity and increases the model's representational capacity. "
    "Layer normalization and residual connections are critical components that stabilize training and enable the construction of very deep networks. "
    "The original Transformer used sinusoidal positional encodings to inject information about token positions into the model. "
    "Later work introduced learnable positional embeddings and relative positional encodings such as RoPE (Rotary Position Embedding), which has become the standard in modern LLMs. "
    "RoPE encodes position information by rotating the query and key vectors in the complex plane, providing a natural way to represent relative distances between tokens. "
    "The pre-training and fine-tuning paradigm, popularized by BERT and GPT, transformed the field of NLP. "
    "BERT introduced masked language modeling as a pre-training objective, enabling bidirectional context understanding. "
    "GPT took a different approach with autoregressive language modeling, predicting the next token given all previous tokens. "
    "The scaling laws discovered by Kaplan et al. at OpenAI revealed that model performance improves predictably with increases in model size, dataset size, and compute budget. "
    "This finding motivated the development of increasingly large models, from GPT-2 with 1.5 billion parameters to GPT-3 with 175 billion parameters and beyond. "
    "The emergence of capabilities like in-context learning, chain-of-thought reasoning, and instruction following in large models surprised many researchers. "
    "These emergent abilities appear to arise from the sheer scale of training data and model parameters rather than from explicit architectural innovations. "
    "Reinforcement learning from human feedback (RLHF) has become a crucial technique for aligning language models with human preferences and values. "
    "The process typically involves training a reward model on human preference data and then using proximal policy optimization to fine-tune the language model. "
    "Direct preference optimization (DPO) offers a simpler alternative that eliminates the need for a separate reward model by directly optimizing the policy using preference pairs. "
    "The challenge of extending context length has driven significant research in efficient attention mechanisms. "
    "Flash Attention, developed by Dao et al., dramatically improved the memory efficiency and speed of attention computation by leveraging GPU memory hierarchy. "
    "By computing attention in tiles and avoiding materialization of the full attention matrix, Flash Attention reduced memory usage from quadratic to linear while maintaining exact computation. "
    "Ring Attention extended this idea to distributed settings, enabling sequence parallelism across multiple devices by passing key-value blocks in a ring topology. "
    "Sparse attention patterns, such as those used in Longformer and BigBird, reduce the computational complexity by restricting each token to attend only to a subset of other tokens. "
    "Longformer combines local sliding window attention with global attention on special tokens, achieving linear complexity while maintaining strong performance on long document tasks. "
    "BigBird adds random attention connections to the mix, providing theoretical guarantees of universal approximation while keeping the overall complexity manageable. "
    "Linear attention mechanisms replace the softmax operation with kernel functions, enabling the use of the associative property of matrix multiplication to reduce complexity from quadratic to linear. "
    "RetNet proposed a retention mechanism that combines the training parallelism of Transformers with the efficient inference of recurrent models. "
    "Mamba and other state space models have emerged as alternatives to Transformers, offering linear-time sequence processing with competitive performance on many benchmarks. "
    "However, pure state space models still struggle with tasks requiring precise recall of information from long contexts, which is where attention-based approaches excel. "
    "Hierarchical approaches to attention offer a middle ground between full attention and purely local or linear methods. "
    "The key insight is that not all tokens in a long context are equally relevant to the current generation step. "
    "By organizing the context into a hierarchy of chunks or pages and using a lightweight scoring mechanism to identify the most relevant portions, hierarchical attention can achieve near-linear complexity while preserving the ability to access any part of the context. "
    "In the HSA framework, the context is divided into fixed-size pages, and each page is represented by a landmark token that summarizes its content. "
    "During generation, the model first computes attention scores between the query and all landmark tokens to identify the top-k most relevant pages. "
    "It then performs full attention only within the selected pages and the local sliding window, dramatically reducing the total computation. "
    "The landmark tokens are typically computed as the mean of all key vectors within each page, providing a compressed representation of the page's content. "
    "The selection process uses the query-landmark dot product as a relevance score, and the top-k pages with the highest scores are selected for full attention computation. "
    "This two-stage approach is analogous to information retrieval systems that first use a fast index to identify candidate documents and then apply more expensive ranking to the candidates. "
    "The interleaving of dense and sparse attention layers is another important design choice in hierarchical attention architectures. "
    "Dense layers perform full attention over the local window and provide fine-grained local context understanding. "
    "Sparse layers extend the effective context by retrieving relevant information from distant parts of the sequence. "
    "By alternating between these two types of layers, the model can build up a comprehensive understanding of both local and global context. "
    "Distributed training of large language models requires careful consideration of memory and communication constraints. "
    "Data parallelism replicates the model across multiple devices and distributes the training data, synchronizing gradients after each step. "
    "Tensor parallelism splits individual layers across devices, requiring all-reduce operations for the outputs of parallel computations. "
    "Pipeline parallelism divides the model into stages assigned to different devices, with micro-batches flowing through the pipeline to maximize utilization. "
    "The combination of these parallelism strategies, often called 3D parallelism, enables training of models with hundreds of billions of parameters. "
    "ZeRO (Zero Redundancy Optimizer) partitions optimizer states, gradients, and parameters across data parallel ranks to reduce memory redundancy. "
    "Activation checkpointing trades compute for memory by recomputing intermediate activations during the backward pass instead of storing them. "
    "Mixed precision training using FP16 or BF16 reduces memory usage and increases throughput on modern GPUs with tensor cores. "
    "The choice between FP16 and BF16 involves trade-offs: FP16 offers higher precision in the mantissa but a smaller dynamic range, while BF16 matches FP32's dynamic range with reduced precision. "
    "Gradient accumulation allows effective batch sizes larger than what fits in GPU memory by accumulating gradients over multiple micro-steps before performing an optimizer update. "
    "Communication optimization techniques such as overlapping computation with communication, gradient compression, and hierarchical all-reduce are essential for scaling to thousands of GPUs. "
    "The inference serving of large language models presents unique challenges compared to training. "
    "The autoregressive nature of text generation means that each token depends on all previous tokens, creating a sequential bottleneck. "
    "KV cache management is critical for efficient inference, as storing and retrieving the key-value pairs for all previous tokens dominates memory usage during generation. "
    "PagedAttention, introduced in vLLM, manages KV cache memory using a paging mechanism inspired by operating system virtual memory management. "
    "This approach eliminates memory fragmentation and enables efficient memory sharing between requests through copy-on-write semantics. "
    "Continuous batching allows new requests to be added to an ongoing batch as soon as existing requests complete, maximizing GPU utilization. "
    "Speculative decoding uses a smaller draft model to generate candidate tokens that are then verified by the larger target model in parallel, potentially generating multiple tokens per forward pass. "
    "Quantization reduces the precision of model weights and activations to lower memory usage and increase inference speed. "
    "Common quantization approaches include INT8, INT4, and even lower bit-width representations, with techniques like GPTQ and AWQ providing high-quality quantized models. "
    "The evaluation of language models spans multiple dimensions including perplexity, downstream task performance, safety, and alignment with human preferences. "
    "Benchmarks like MMLU, HellaSwag, and ARC test different aspects of model knowledge and reasoning ability. "
    "Long context evaluation is particularly challenging, with benchmarks like RULER, Needle-in-a-Haystack, and SCROLLS designed to test the model's ability to utilize information from different positions in long contexts. "
    "The RULER benchmark systematically varies the position and type of information retrieval tasks to provide a comprehensive assessment of long context capabilities. "
    "Needle-in-a-Haystack tests insert a specific piece of information at various positions in a long context and measure the model's ability to retrieve it accurately. "
    "Multi-hop reasoning over long contexts remains one of the most challenging tasks, requiring the model to connect information from multiple distant locations. "
    "The safety and alignment of language models has become a critical research area as these models are deployed in increasingly sensitive applications. "
    "Red teaming exercises systematically probe models for harmful outputs, biases, and vulnerabilities. "
    "Constitutional AI provides a framework for training models to be helpful, harmless, and honest through a combination of supervised learning and reinforcement learning. "
    "The development of open-source language models has democratized access to powerful AI capabilities. "
    "Models like LLaMA, Mistral, and Qwen have demonstrated that open models can achieve performance competitive with proprietary systems. "
    "The open-source ecosystem has also driven innovation in training techniques, evaluation methods, and deployment tools. "
    "Looking forward, the field continues to evolve rapidly with new architectures, training methods, and applications emerging regularly. "
    "Multimodal models that can process text, images, audio, and video are becoming increasingly capable and practical. "
    "The integration of language models with external tools, databases, and APIs through function calling and agent frameworks is expanding the range of tasks these models can perform. "
    "Efficient fine-tuning methods like LoRA and QLoRA have made it practical to customize large models for specific domains and tasks with limited computational resources. "
    "The ongoing research into model compression, distillation, and architecture search promises to make powerful language models accessible on a wider range of hardware platforms. "
    "As the field matures, the focus is shifting from simply scaling up models to making them more efficient, reliable, and aligned with human values. "
    "The intersection of language models with scientific research, healthcare, education, and creative industries is opening up new possibilities that were previously unimaginable. "
    "Understanding the theoretical foundations of why these models work so well remains an active area of research, with connections to information theory, statistical learning theory, and cognitive science. "
    "The study of in-context learning, for example, has revealed surprising connections between Transformer inference and gradient descent optimization. "
    "Mechanistic interpretability research aims to understand the internal representations and computations of neural networks at a fine-grained level. "
    "Circuit analysis has identified specific attention heads and MLP neurons responsible for tasks like indirect object identification and factual recall. "
    "The superposition hypothesis suggests that neural networks represent more features than they have dimensions by encoding features in overlapping directions. "
    "Sparse autoencoders have been used to decompose model activations into interpretable features, providing insights into what information the model encodes at each layer. "
    "The relationship between model scale and the emergence of new capabilities continues to be debated, with some researchers arguing that apparent emergence may be an artifact of evaluation metrics. "
    "Regardless of the theoretical explanation, the practical impact of large language models on society is undeniable and continues to grow. "
    "From automated customer service to scientific literature review, from code generation to creative writing assistance, these models are transforming how we work and create. "
    "The responsible development and deployment of these powerful tools requires ongoing collaboration between researchers, engineers, policymakers, and the broader public. "
    "As we continue to push the boundaries of what language models can do, it is essential that we also advance our understanding of their limitations, risks, and societal implications. "
    "The concept of transfer learning has been fundamental to the success of modern language models. "
    "By pre-training on large corpora and then fine-tuning on specific tasks, models can leverage general linguistic knowledge to achieve strong performance even with limited task-specific data. "
    "Few-shot and zero-shot learning capabilities have further reduced the need for task-specific training data, enabling models to perform new tasks based solely on natural language instructions. "
    "The development of prompt engineering as a discipline reflects the importance of how we communicate with these models. "
    "Carefully crafted prompts can dramatically improve model performance on specific tasks, while poorly designed prompts may lead to suboptimal or even harmful outputs. "
    "Chain-of-thought prompting, where the model is encouraged to show its reasoning steps, has been particularly effective for mathematical and logical reasoning tasks. "
    "Tree-of-thought and graph-of-thought extensions allow for more complex reasoning patterns, including backtracking and parallel exploration of solution paths. "
    "The retrieval-augmented generation paradigm combines the generative capabilities of language models with external knowledge retrieval systems. "
    "By grounding model outputs in retrieved documents, RAG systems can reduce hallucination and provide more factually accurate responses. "
    "Vector databases and embedding-based retrieval have become essential infrastructure components for building RAG systems at scale. "
    "The chunking strategy used to divide documents into retrievable units significantly impacts the quality of retrieved context and downstream generation. "
    "Hybrid retrieval approaches that combine dense embedding search with sparse keyword matching often outperform either method alone. "
    "The reranking stage, where a cross-encoder model scores the relevance of retrieved passages, provides an additional layer of precision. "
    "Knowledge graphs offer a structured alternative to unstructured text retrieval, enabling precise factual queries and multi-hop reasoning over entity relationships. "
    "The integration of knowledge graphs with language models is an active area of research, with approaches ranging from graph-enhanced pre-training to runtime graph traversal. "
    "Federated learning presents opportunities for training language models on distributed private data without centralizing sensitive information. "
    "Differential privacy techniques can provide formal guarantees about the privacy of individual training examples, though they often come at the cost of model utility. "
    "The environmental impact of training large language models has become a growing concern, with some estimates suggesting that training a single large model can emit as much carbon as several transatlantic flights. "
    "Efforts to reduce the carbon footprint of AI include more efficient training algorithms, hardware optimization, and the use of renewable energy sources for data centers. "
    "Model distillation offers a practical approach to creating smaller, more efficient models that retain much of the capability of their larger teachers. "
    "The student model learns to mimic the output distribution of the teacher model, effectively compressing the knowledge into a more compact representation. "
    "Pruning techniques remove redundant parameters from trained models, reducing both memory footprint and inference latency. "
    "Structured pruning removes entire neurons, attention heads, or layers, while unstructured pruning zeros out individual weights. "
    "The lottery ticket hypothesis suggests that within large neural networks, there exist smaller subnetworks that can achieve comparable performance when trained in isolation. "
    "Neural architecture search automates the design of model architectures, potentially discovering configurations that outperform human-designed alternatives. "
    "The trade-off between model size and inference efficiency has led to the development of mixture-of-experts architectures. "
    "In MoE models, only a subset of the model's parameters are activated for each input, enabling much larger total parameter counts without proportional increases in computation. "
    "The routing mechanism that determines which experts process each token is a critical design choice that affects both model quality and training stability. "
    "Load balancing across experts is essential to prevent the collapse of routing to a small subset of experts, which would waste the capacity of unused experts. "
    "The Gshard and Switch Transformer papers demonstrated that MoE architectures can scale to trillions of parameters while maintaining practical training and inference costs. "
    "Continuous learning and adaptation of deployed language models presents challenges related to catastrophic forgetting and distribution shift. "
    "Techniques like elastic weight consolidation and progressive neural networks aim to enable models to learn new tasks without forgetting previously acquired knowledge. "
    "The evaluation of language model safety encompasses multiple dimensions including toxicity, bias, factual accuracy, and resistance to adversarial attacks. "
    "Jailbreaking attacks attempt to circumvent safety guardrails through carefully crafted prompts that exploit the model's instruction-following capabilities. "
    "Watermarking techniques embed detectable signals in model outputs to enable attribution and detection of AI-generated content. "
    "The debate around AI-generated content and its impact on information ecosystems, creative industries, and education continues to evolve. "
    "Copyright considerations for training data and model outputs remain legally uncertain in many jurisdictions, with ongoing litigation shaping the regulatory landscape. "
    "The development of AI governance frameworks at national and international levels reflects the growing recognition of the need for responsible AI development. "
    "Technical standards for AI safety, transparency, and accountability are being developed by organizations such as NIST, ISO, and the IEEE. "
    "The concept of AI alignment, ensuring that AI systems pursue goals that are beneficial to humanity, remains one of the most important open problems in the field. "
    "Approaches to alignment range from constitutional AI and RLHF to more theoretical frameworks based on decision theory and game theory. "
    "The long-term trajectory of language model development points toward increasingly capable and versatile systems that can serve as general-purpose reasoning engines. "
    "Whether these systems will achieve genuine understanding or remain sophisticated pattern matchers is a philosophical question that continues to generate vigorous debate. "
    "What is clear is that language models have already transformed numerous industries and will continue to do so as the technology matures and becomes more accessible. "
    "The collaboration between human creativity and AI capabilities promises to unlock new forms of innovation that neither could achieve alone. "
    "As researchers and practitioners, our responsibility is to ensure that this powerful technology is developed and deployed in ways that benefit all of humanity. "
)

# 将 BASE_TEXT 按空格拆分为 word 列表，按不同 word 数切片构造不同长度的 prompt
# 这样每个 prompt 的内容都是不重复的连续文本，避免重复文本诱导模型输出退化
_BASE_WORDS = BASE_TEXT.split()
_TOTAL_WORDS = len(_BASE_WORDS)

def slice_prompt(n_words: int) -> str:
    """从 BASE_TEXT 中取前 n_words 个 word 拼成 prompt"""
    n = min(n_words, _TOTAL_WORDS)
    return " ".join(_BASE_WORDS[:n])

LONG_PROMPTS = [
    slice_prompt(800 + i * 100) + f" Question {i}: What is the main contribution of this paper?" for i in range(8)
]

# 要测试的 logprob_start_len 值
LOGPROB_START_LEN = int(os.environ.get("LOGPROB_START_LEN", "0"))
# ========================================================


def print_separator(title: str, char: str = "=", width: int = 80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def create_engine(chunked_prefill_size: int, disable_radix_cache: bool = False):
    """创建 SGLang Engine（HSA backend），使用指定的 chunked_prefill_size"""
    import sglang as sgl

    print(f"[Engine] 正在加载模型: {CHECKPOINT_PATH}")
    print(f"[Engine] 参数: tp={SGLANG_TP}, page_size={PAGE_SIZE}, "
          f"max_total_tokens={MAX_TOTAL_TOKENS}, "
          f"chunked_prefill_size={chunked_prefill_size}, "
          f"disable_radix_cache={disable_radix_cache}")

    engine = sgl.Engine(
        model_path=CHECKPOINT_PATH,
        tp_size=SGLANG_TP,
        attention_backend="hsa",
        page_size=PAGE_SIZE,
        max_total_tokens=MAX_TOTAL_TOKENS,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        chunked_prefill_size=chunked_prefill_size,
        mem_fraction_static=0.85,
        disable_radix_cache=disable_radix_cache,
    )
    return engine


def warmup(engine, page_size: int):
    """预热：触发 TileLang JIT 编译等"""
    warmup_prompt = "Hello " * max(page_size * 2, 1024)
    warmup_tokens = 3
    print(f"[Warmup] 正在预热（生成 {warmup_tokens} tokens 触发 JIT 编译）...")
    t0 = time.time()
    _ = engine.generate(
        prompt=warmup_prompt,
        sampling_params={"max_new_tokens": warmup_tokens, "temperature": 0.0},
    )
    print(f"[Warmup] 预热完成，耗时 {time.time() - t0:.2f}s")


def test_single_with_logprob(engine, prompt: str, logprob_start_len: int,
                              max_new_tokens: int, label: str = "") -> Optional[Dict]:
    """单条请求 + logprob（baseline，不触发 chunked prefill）"""
    print(f"\n  --- {label} ---")
    try:
        t0 = time.time()
        result = engine.generate(
            prompt=prompt,
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=logprob_start_len,
        )
        elapsed = time.time() - t0

        meta = result.get("meta_info", {})
        output_ids = result.get("output_ids", [])
        text = result.get("text", "")
        cached_tokens = meta.get("cached_tokens", -1)
        prompt_tokens = meta.get("prompt_tokens", -1)

        text_preview = text[:80].replace("\n", "\\n")
        print(f"    ✅ 成功! 耗时={elapsed:.2f}s")
        print(f"    prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}")
        print(f"    output_tokens={len(output_ids)}, text=\"{text_preview}...\"")

        return {
            "success": True,
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "output_ids": output_ids,
            "text": text,
            "label": label,
        }

    except Exception as e:
        print(f"    ❌ 失败! 错误: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "label": label,
        }


def test_batch_with_logprob(engine, prompts: List[str], logprob_start_len: int,
                             max_new_tokens: int, label: str = "") -> Optional[Dict]:
    """
    多条请求并发 + logprob（触发 chunked prefill）。
    这是 Bug 1 & 2 的核心触发场景。
    """
    print(f"\n  --- {label} ---")
    print(f"    并发请求数: {len(prompts)}")
    for i, p in enumerate(prompts):
        print(f"    [{i}] prompt 长度: ~{len(p.split())} words")

    try:
        t0 = time.time()
        results = engine.generate(
            prompt=prompts,
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=logprob_start_len,
        )
        elapsed = time.time() - t0

        if isinstance(results, dict):
            results = [results]

        print(f"    ✅ 成功! 耗时={elapsed:.2f}s, 返回 {len(results)} 条结果")
        for i, res in enumerate(results):
            meta = res.get("meta_info", {})
            output_ids = res.get("output_ids", [])
            text = res.get("text", "")[:60].replace("\n", "\\n")
            prompt_tokens = meta.get("prompt_tokens", -1)
            cached_tokens = meta.get("cached_tokens", -1)
            print(f"    [{i}] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}, "
                  f"output_tokens={len(output_ids)}, text=\"{text}...\"")

        return {
            "success": True,
            "elapsed": elapsed,
            "results": results,
            "label": label,
            "batch_size": len(prompts),
        }

    except Exception as e:
        print(f"    ❌ 失败! 错误: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "label": label,
            "batch_size": len(prompts),
        }


def test_batch_without_logprob(engine, prompts: List[str],
                                max_new_tokens: int, label: str = "") -> Optional[Dict]:
    """
    多条请求并发，不开启 logprob（对照组）。
    验证 chunked prefill 本身是否正常工作（不走 logprob 路径）。
    """
    print(f"\n  --- {label} ---")
    print(f"    并发请求数: {len(prompts)}, 不开启 logprob")

    try:
        t0 = time.time()
        results = engine.generate(
            prompt=prompts,
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
        )
        elapsed = time.time() - t0

        if isinstance(results, dict):
            results = [results]

        print(f"    ✅ 成功! 耗时={elapsed:.2f}s, 返回 {len(results)} 条结果")
        for i, res in enumerate(results):
            meta = res.get("meta_info", {})
            output_ids = res.get("output_ids", [])
            text = res.get("text", "")[:60].replace("\n", "\\n")
            prompt_tokens = meta.get("prompt_tokens", -1)
            print(f"    [{i}] prompt_tokens={prompt_tokens}, "
                  f"output_tokens={len(output_ids)}, text=\"{text}...\"")

        return {
            "success": True,
            "elapsed": elapsed,
            "results": results,
            "label": label,
            "batch_size": len(prompts),
        }

    except Exception as e:
        print(f"    ❌ 失败! 错误: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "label": label,
            "batch_size": len(prompts),
        }


def test_sequential_vs_batch(engine, prompts: List[str], max_new_tokens: int):
    """顺序推理 vs Batch 推理：对比 decode 一致性和速度。

    1. 顺序推理：逐条发送，记录每条的 output_ids 和耗时
    2. Batch 推理：一次性发送所有请求，记录 output_ids 和耗时
    3. 对比：output_ids 是否一致 + 速度对比
    """
    N = len(prompts)
    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0.0}

    # ---- 1. 顺序推理 ----
    print_separator(f"顺序推理: {N} 条请求逐条发送")
    seq_results = []
    t_seq_start = time.time()
    for i, p in enumerate(prompts):
        t0 = time.time()
        res = engine.generate(
            prompt=p, sampling_params=sampling_params,
            return_logprob=True, top_logprobs_num=5,
        )
        elapsed = time.time() - t0
        output_ids = res.get("output_ids", [])
        text = res.get("text", "")
        meta = res.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", -1)
        cached_tokens = meta.get("cached_tokens", -1)
        output_top_logprobs = meta.get("output_top_logprobs", [])
        output_token_logprobs = meta.get("output_token_logprobs", [])
        text_preview = text[:60].replace("\n", "\\n")
        print(f"  [{i}] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}, output_tokens={len(output_ids)}, "
              f"耗时={elapsed:.3f}s, text=\"{text_preview}...\"")
        seq_results.append({
            "output_ids": output_ids, "text": text, "elapsed": elapsed,
            "output_top_logprobs": output_top_logprobs,
            "output_token_logprobs": output_token_logprobs,
        })
    t_seq_total = time.time() - t_seq_start
    print(f"\n  顺序推理总耗时: {t_seq_total:.3f}s")

    # ---- 2. Batch 推理 ----
    print_separator(f"Batch 推理: {N} 条请求一次性发送")
    t_batch_start = time.time()
    batch_raw = engine.generate(
        prompt=prompts, sampling_params=sampling_params,
        return_logprob=True, top_logprobs_num=5,
    )
    t_batch_total = time.time() - t_batch_start

    if isinstance(batch_raw, dict):
        batch_raw = [batch_raw]

    batch_results = []
    for i, res in enumerate(batch_raw):
        output_ids = res.get("output_ids", [])
        text = res.get("text", "")
        meta = res.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", -1)
        cached_tokens = meta.get("cached_tokens", -1)
        output_top_logprobs = meta.get("output_top_logprobs", [])
        output_token_logprobs = meta.get("output_token_logprobs", [])
        text_preview = text[:60].replace("\n", "\\n")
        print(f"  [{i}] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}, output_tokens={len(output_ids)}, "
              f"text=\"{text_preview}...\"")
        batch_results.append({
            "output_ids": output_ids, "text": text,
            "output_top_logprobs": output_top_logprobs,
            "output_token_logprobs": output_token_logprobs,
        })
    print(f"\n  Batch 推理总耗时: {t_batch_total:.3f}s")

    # ---- 3. 对比 ----
    print_separator("对比结果")

    # 3a. 一致性
    all_match = True
    for i in range(N):
        seq_ids = seq_results[i]["output_ids"]
        bat_ids = batch_results[i]["output_ids"]
        match = (seq_ids == bat_ids)
        status = "✅ 一致" if match else "❌ 不一致"
        if not match:
            all_match = False
            # 找到第一个不同的位置
            min_len = min(len(seq_ids), len(bat_ids))
            diff_pos = -1
            for j in range(min_len):
                if seq_ids[j] != bat_ids[j]:
                    diff_pos = j
                    break
            if diff_pos == -1:
                diff_pos = min_len  # 长度不同
            print(f"  [{i}] {status}  (首个差异位置={diff_pos}, "
                  f"seq_len={len(seq_ids)}, batch_len={len(bat_ids)})")
            print(f"       seq_text: \"{seq_results[i]['text'][:80]}...\"")
            print(f"       bat_text: \"{batch_results[i]['text'][:80]}...\"")

            # 打印首个差异位置的 top logprobs，诊断是否为 tie breaking
            seq_top = seq_results[i].get("output_top_logprobs", [])
            bat_top = batch_results[i].get("output_top_logprobs", [])
            seq_logprobs = seq_results[i].get("output_token_logprobs", [])
            bat_logprobs = batch_results[i].get("output_token_logprobs", [])
            print(f"\n       --- 首个差异位置 (pos={diff_pos}) 的 logprobs 诊断 ---")
            if diff_pos < len(seq_logprobs):
                print(f"       Seq token_logprob[{diff_pos}]: {seq_logprobs[diff_pos]}")
            if diff_pos < len(bat_logprobs):
                print(f"       Bat token_logprob[{diff_pos}]: {bat_logprobs[diff_pos]}")
            if diff_pos < len(seq_top):
                print(f"       Seq top5_logprobs[{diff_pos}]: {seq_top[diff_pos]}")
            if diff_pos < len(bat_top):
                print(f"       Bat top5_logprobs[{diff_pos}]: {bat_top[diff_pos]}")

            # 如果 top1 和 top2 的 logprob 差值很小，说明是 tie breaking
            if diff_pos < len(seq_top) and len(seq_top[diff_pos]) >= 2:
                top1_logp = seq_top[diff_pos][0][0] if isinstance(seq_top[diff_pos][0], (list, tuple)) else None
                top2_logp = seq_top[diff_pos][1][0] if isinstance(seq_top[diff_pos][1], (list, tuple)) else None
                if top1_logp is not None and top2_logp is not None:
                    gap = abs(top1_logp - top2_logp)
                    print(f"       Seq top1-top2 logprob gap: {gap:.6f} {'(⚠️ tie breaking: gap < 0.01)' if gap < 0.01 else ''}")
            if diff_pos < len(bat_top) and len(bat_top[diff_pos]) >= 2:
                top1_logp = bat_top[diff_pos][0][0] if isinstance(bat_top[diff_pos][0], (list, tuple)) else None
                top2_logp = bat_top[diff_pos][1][0] if isinstance(bat_top[diff_pos][1], (list, tuple)) else None
                if top1_logp is not None and top2_logp is not None:
                    gap = abs(top1_logp - top2_logp)
                    print(f"       Bat top1-top2 logprob gap: {gap:.6f} {'(⚠️ tie breaking: gap < 0.01)' if gap < 0.01 else ''}")

            # 对比两边的 top2 候选集合是否一致
            def _extract_top_tokens(top_list, pos, k=2):
                """从 top_logprobs 中提取 pos 位置的 top-k token id 集合和详情"""
                if pos >= len(top_list) or len(top_list[pos]) < k:
                    return None, []
                tokens = []
                details = []
                for entry in top_list[pos][:k]:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        # (logprob, token_id, token_text) 或 (logprob, token_id)
                        tid = entry[1]
                        logp = entry[0]
                        ttext = entry[2] if (len(entry) >= 3 and entry[2] is not None) else f"id={tid}"
                        tokens.append(tid)
                        details.append((tid, logp, ttext))
                return set(tokens) if tokens else None, details

            seq_top2_set, seq_top2_details = _extract_top_tokens(seq_top, diff_pos, k=2)
            bat_top2_set, bat_top2_details = _extract_top_tokens(bat_top, diff_pos, k=2)
            print(f"\n       --- Top2 候选集合对比 ---")
            if seq_top2_details:
                print(f"       Seq top2: {[(d[2], f'logp={d[1]:.4f}') for d in seq_top2_details]}")
            if bat_top2_details:
                print(f"       Bat top2: {[(d[2], f'logp={d[1]:.4f}') for d in bat_top2_details]}")
            if seq_top2_set is not None and bat_top2_set is not None:
                if seq_top2_set == bat_top2_set:
                    print(f"       ✅ Top2 候选集合一致 (同一组 token，仅排序不同 → tie breaking)")
                else:
                    print(f"       ❌ Top2 候选集合不一致 (Seq={seq_top2_set}, Bat={bat_top2_set})")
        else:
            print(f"  [{i}] {status}  (output_tokens={len(seq_ids)})")

    # 3b. 速度
    print()
    speedup = t_seq_total / t_batch_total if t_batch_total > 0 else float("inf")
    print(f"  顺序推理总耗时: {t_seq_total:.3f}s")
    print(f"  Batch 推理总耗时: {t_batch_total:.3f}s")
    print(f"  加速比: Batch 快 {(speedup - 1) * 100:.1f}%  (Sequential/Batch = {speedup:.2f}x)")

    # 3c. 汇总
    print()
    if all_match:
        print("  🎉 所有样本 decode 输出完全一致!")
    else:
        print("  ⚠️  存在 decode 输出不一致的样本，请检查!")

    return {
        "seq_total_time": t_seq_total,
        "batch_total_time": t_batch_total,
        "speedup": speedup,
        "all_match": all_match,
    }


def test_prefix_cache_consistency(engine, prompts: List[str], max_new_tokens: int):
    """Prefix Cache 一致性测试（方案 A）。

    验证：同一个 Engine 内，首次推理（无 cache）与重复推理（命中 cache）的输出是否完全一致。

    流程：
      Step 1: 冷启动推理 —— 逐条发送 prompts，记录 output_ids 和 cached_tokens
              预期 cached_tokens == 0（首次请求，无 cache 可命中）
      Step 2: 重复推理 —— 再次逐条发送完全相同的 prompts
              预期 cached_tokens > 0（命中 prefix cache）
      Step 3: 对比 Step1 和 Step2 的 output_ids 是否完全一致
    """
    N = len(prompts)
    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0.0}

    # ---- Step 1: 冷启动推理（首次，预期无 cache）----
    print_separator(f"Prefix Cache 测试 Step 1: 冷启动推理 ({N} 条)")
    cold_results = []
    for i, p in enumerate(prompts):
        t0 = time.time()
        res = engine.generate(prompt=p, sampling_params=sampling_params)
        elapsed = time.time() - t0
        output_ids = res.get("output_ids", [])
        text = res.get("text", "")
        meta = res.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", -1)
        cached_tokens = meta.get("cached_tokens", -1)
        text_preview = text[:60].replace("\n", "\\n")
        print(f"  [{i}] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}, "
              f"output_tokens={len(output_ids)}, 耗时={elapsed:.3f}s, text=\"{text_preview}...\"")
        cold_results.append({
            "output_ids": output_ids, "text": text,
            "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens,
        })

    # ---- Step 2: 重复推理（相同 prompt，预期命中 cache）----
    print_separator(f"Prefix Cache 测试 Step 2: 重复推理 ({N} 条, 预期命中 cache)")
    warm_results = []
    for i, p in enumerate(prompts):
        t0 = time.time()
        res = engine.generate(prompt=p, sampling_params=sampling_params)
        elapsed = time.time() - t0
        output_ids = res.get("output_ids", [])
        text = res.get("text", "")
        meta = res.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", -1)
        cached_tokens = meta.get("cached_tokens", -1)
        text_preview = text[:60].replace("\n", "\\n")
        cache_hit = "✅ 命中" if cached_tokens > 0 else "⚠️ 未命中"
        print(f"  [{i}] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens} ({cache_hit}), "
              f"output_tokens={len(output_ids)}, 耗时={elapsed:.3f}s, text=\"{text_preview}...\"")
        warm_results.append({
            "output_ids": output_ids, "text": text,
            "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens,
        })

    # ---- Step 3: 对比 cold vs warm ----
    print_separator("Prefix Cache 测试 Step 3: 冷启动 vs 重复推理 对比")
    all_match_cold_warm = True
    for i in range(N):
        cold_ids = cold_results[i]["output_ids"]
        warm_ids = warm_results[i]["output_ids"]
        match = (cold_ids == warm_ids)
        cold_cached = cold_results[i]["cached_tokens"]
        warm_cached = warm_results[i]["cached_tokens"]
        if match:
            print(f"  [{i}] ✅ 一致  (output_tokens={len(cold_ids)}, "
                  f"cold_cached={cold_cached}, warm_cached={warm_cached})")
        else:
            all_match_cold_warm = False
            min_len = min(len(cold_ids), len(warm_ids))
            diff_pos = next((j for j in range(min_len) if cold_ids[j] != warm_ids[j]), min_len)
            print(f"  [{i}] ❌ 不一致  (首个差异位置={diff_pos}, "
                  f"cold_len={len(cold_ids)}, warm_len={len(warm_ids)}, "
                  f"cold_cached={cold_cached}, warm_cached={warm_cached})")
            print(f"       cold_text: \"{cold_results[i]['text'][:80]}...\"")
            print(f"       warm_text: \"{warm_results[i]['text'][:80]}...\"")

    # ---- 汇总 ----
    print_separator("Prefix Cache 测试汇总")
    print(f"  冷启动 vs 重复推理(逐条): {'🎉 全部一致' if all_match_cold_warm else '⚠️ 存在不一致'}")

    return {
        "cold_vs_warm_match": all_match_cold_warm,
    }


def test_prefix_cache_strict(prompts: List[str], max_new_tokens: int,
                             chunked_prefill_size: int):
    """Prefix Cache 严格一致性测试（方案 B）。

    创建两个 Engine 串行对比：
      Engine A: disable_radix_cache=True  → 每次推理都完整 prefill，无 cache
      Engine B: disable_radix_cache=False → 默认开启 radix cache
    用完全相同的 prompt 逐条推理，对比 output_ids 是否完全一致。

    流程：
      Step 1: 创建 Engine A（关闭 cache），逐条推理，记录结果
      Step 2: 关闭 Engine A，释放显存
      Step 3: 创建 Engine B（开启 cache），逐条推理，记录结果
      Step 4: 关闭 Engine B，释放显存
      Step 5: 对比 A vs B 的 output_ids
    """
    import torch

    N = len(prompts)
    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0.0}

    # ---- Step 1: Engine A（关闭 cache）逐条推理 ----
    print_separator(f"方案 B Step 1: 创建 Engine A (disable_radix_cache=True)")
    engine_a = create_engine(chunked_prefill_size, disable_radix_cache=True)
    warmup(engine_a, PAGE_SIZE)

    print_separator(f"方案 B Step 2: Engine A 逐条推理 ({N} 条, 无 cache)")
    nocache_results = []
    for i, p in enumerate(prompts):
        t0 = time.time()
        res = engine_a.generate(
            prompt=p, sampling_params=sampling_params,
            return_logprob=True, top_logprobs_num=5,
        )
        elapsed = time.time() - t0
        output_ids = res.get("output_ids", [])
        text = res.get("text", "")
        meta = res.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", -1)
        cached_tokens = meta.get("cached_tokens", -1)
        output_top_logprobs = meta.get("output_top_logprobs", [])
        output_token_logprobs = meta.get("output_token_logprobs", [])
        text_preview = text[:60].replace("\n", "\\n")
        print(f"  [{i}] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}, "
              f"output_tokens={len(output_ids)}, 耗时={elapsed:.3f}s, text=\"{text_preview}...\"")
        nocache_results.append({
            "output_ids": output_ids, "text": text,
            "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens,
            "output_top_logprobs": output_top_logprobs,
            "output_token_logprobs": output_token_logprobs,
        })

    # ---- Step 3: 关闭 Engine A ----
    print_separator("方案 B Step 3: 关闭 Engine A，释放显存")
    engine_a.shutdown()
    del engine_a
    gc.collect()
    torch.cuda.empty_cache()
    print("  Engine A 已关闭")

    # ---- Step 4: Engine B（开启 cache）逐条推理 ----
    print_separator(f"方案 B Step 4: 创建 Engine B (disable_radix_cache=False)")
    engine_b = create_engine(chunked_prefill_size, disable_radix_cache=False)
    warmup(engine_b, PAGE_SIZE)

    print_separator(f"方案 B Step 5: Engine B 逐条推理 ({N} 条, 有 cache)")
    cache_results = []
    for i, p in enumerate(prompts):
        t0 = time.time()
        res = engine_b.generate(
            prompt=p, sampling_params=sampling_params,
            return_logprob=True, top_logprobs_num=5,
        )
        elapsed = time.time() - t0
        output_ids = res.get("output_ids", [])
        text = res.get("text", "")
        meta = res.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", -1)
        cached_tokens = meta.get("cached_tokens", -1)
        output_top_logprobs = meta.get("output_top_logprobs", [])
        output_token_logprobs = meta.get("output_token_logprobs", [])
        text_preview = text[:60].replace("\n", "\\n")
        print(f"  [{i}] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}, "
              f"output_tokens={len(output_ids)}, 耗时={elapsed:.3f}s, text=\"{text_preview}...\"")
        cache_results.append({
            "output_ids": output_ids, "text": text,
            "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens,
            "output_top_logprobs": output_top_logprobs,
            "output_token_logprobs": output_token_logprobs,
        })

    # ---- Step 6: 关闭 Engine B ----
    print_separator("方案 B Step 6: 关闭 Engine B，释放显存")
    engine_b.shutdown()
    del engine_b
    gc.collect()
    torch.cuda.empty_cache()
    print("  Engine B 已关闭")

    # ---- Step 7: 对比 A(无cache) vs B(有cache) ----
    print_separator("方案 B Step 7: Engine A(无cache) vs Engine B(有cache) 对比")
    all_match = True
    for i in range(N):
        a_ids = nocache_results[i]["output_ids"]
        b_ids = cache_results[i]["output_ids"]
        match = (a_ids == b_ids)
        a_cached = nocache_results[i]["cached_tokens"]
        b_cached = cache_results[i]["cached_tokens"]
        if match:
            print(f"  [{i}] ✅ 一致  (output_tokens={len(a_ids)}, "
                  f"A_cached={a_cached}, B_cached={b_cached})")
        else:
            all_match = False
            min_len = min(len(a_ids), len(b_ids))
            diff_pos = next((j for j in range(min_len) if a_ids[j] != b_ids[j]), min_len)
            print(f"  [{i}] ❌ 不一致  (首个差异位置={diff_pos}, "
                  f"A_len={len(a_ids)}, B_len={len(b_ids)}, "
                  f"A_cached={a_cached}, B_cached={b_cached})")
            print(f"       A_text: \"{nocache_results[i]['text'][:80]}...\"")
            print(f"       B_text: \"{cache_results[i]['text'][:80]}...\"")

            # 打印首个差异位置的 top logprobs，诊断是否为 tie breaking
            a_top = nocache_results[i].get("output_top_logprobs", [])
            b_top = cache_results[i].get("output_top_logprobs", [])
            a_logprobs = nocache_results[i].get("output_token_logprobs", [])
            b_logprobs = cache_results[i].get("output_token_logprobs", [])
            print(f"\n       --- 首个差异位置 (pos={diff_pos}) 的 logprobs 诊断 ---")
            if diff_pos < len(a_logprobs):
                print(f"       A token_logprob[{diff_pos}]: {a_logprobs[diff_pos]}")
            if diff_pos < len(b_logprobs):
                print(f"       B token_logprob[{diff_pos}]: {b_logprobs[diff_pos]}")
            if diff_pos < len(a_top):
                print(f"       A top5_logprobs[{diff_pos}]: {a_top[diff_pos]}")
            if diff_pos < len(b_top):
                print(f"       B top5_logprobs[{diff_pos}]: {b_top[diff_pos]}")

            # 如果 top1 和 top2 的 logprob 差值很小，说明是 tie breaking
            if diff_pos < len(a_top) and len(a_top[diff_pos]) >= 2:
                top1_logp = a_top[diff_pos][0][0] if isinstance(a_top[diff_pos][0], (list, tuple)) else None
                top2_logp = a_top[diff_pos][1][0] if isinstance(a_top[diff_pos][1], (list, tuple)) else None
                if top1_logp is not None and top2_logp is not None:
                    gap = abs(top1_logp - top2_logp)
                    print(f"       A top1-top2 logprob gap: {gap:.6f} {'(⚠️ tie breaking: gap < 0.01)' if gap < 0.01 else ''}")
            if diff_pos < len(b_top) and len(b_top[diff_pos]) >= 2:
                top1_logp = b_top[diff_pos][0][0] if isinstance(b_top[diff_pos][0], (list, tuple)) else None
                top2_logp = b_top[diff_pos][1][0] if isinstance(b_top[diff_pos][1], (list, tuple)) else None
                if top1_logp is not None and top2_logp is not None:
                    gap = abs(top1_logp - top2_logp)
                    print(f"       B top1-top2 logprob gap: {gap:.6f} {'(⚠️ tie breaking: gap < 0.01)' if gap < 0.01 else ''}")

            # 对比两边的 top2 候选集合是否一致
            def _extract_top_tokens_b(top_list, pos, k=2):
                """从 top_logprobs 中提取 pos 位置的 top-k token id 集合和详情"""
                if pos >= len(top_list) or len(top_list[pos]) < k:
                    return None, []
                tokens = []
                details = []
                for entry in top_list[pos][:k]:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        tid = entry[1]
                        logp = entry[0]
                        ttext = entry[2] if (len(entry) >= 3 and entry[2] is not None) else f"id={tid}"
                        tokens.append(tid)
                        details.append((tid, logp, ttext))
                return set(tokens) if tokens else None, details

            a_top2_set, a_top2_details = _extract_top_tokens_b(a_top, diff_pos, k=2)
            b_top2_set, b_top2_details = _extract_top_tokens_b(b_top, diff_pos, k=2)
            print(f"\n       --- Top2 候选集合对比 ---")
            if a_top2_details:
                print(f"       A top2: {[(d[2], f'logp={d[1]:.4f}') for d in a_top2_details]}")
            if b_top2_details:
                print(f"       B top2: {[(d[2], f'logp={d[1]:.4f}') for d in b_top2_details]}")
            if a_top2_set is not None and b_top2_set is not None:
                if a_top2_set == b_top2_set:
                    print(f"       ✅ Top2 候选集合一致 (同一组 token，仅排序不同 → tie breaking)")
                else:
                    print(f"       ❌ Top2 候选集合不一致 (A={a_top2_set}, B={b_top2_set})")

    # ---- 汇总 ----
    print_separator("方案 B 测试汇总")
    print(f"  Engine A(无cache) vs Engine B(有cache): "
          f"{'🎉 全部一致' if all_match else '⚠️ 存在不一致'}")

    return {"nocache_vs_cache_match": all_match}


def test_swa_cache_diagnosis(engine):
    """诊断 SWA 层是否启用了 SWARadixCache，以及 prefix cache 在 SWA 层的复用情况。

    检查项：
      1. Engine 内部 is_hybrid_swa 标志位
      2. tree_cache 的实际类型（RadixCache vs SWARadixCache）
      3. token_to_kv_pool_allocator 是否有 SWA 独立池（full_available_size / swa_available_size）
      4. model_config 中 SWA 相关字段的实际值
      5. 通过实际推理验证 prefix cache 命中情况（冷启动 vs 热启动）
    """
    print_separator("SWA Cache 诊断")

    # ---- 1. 通过 engine 内部对象获取诊断信息 ----
    # engine._tokenizer_manager 持有 model_config 等信息
    # engine._scheduler 持有 tree_cache、is_hybrid_swa 等信息
    # 但 Engine 对象不直接暴露这些，需要通过 _get_server_info 或直接访问内部属性

    # 尝试从 engine 获取 server_info（包含 model_config 信息）
    try:
        # SGLang Engine 内部结构：engine.tokenizer_manager -> scheduler (通过 IPC)
        # 我们通过 generate 一个极短请求，从 meta_info 中获取诊断信息
        diag_result = engine.generate(
            prompt="Hello",
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )
        meta = diag_result.get("meta_info", {})
        cached_tokens = meta.get("cached_tokens", -1)
        prompt_tokens = meta.get("prompt_tokens", -1)
        print(f"  [诊断请求] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}")
    except Exception as e:
        print(f"  [诊断请求] 失败: {e}")

    # ---- 2. 通过 Python 内省获取 Engine 内部状态 ----
    print(f"\n  --- Engine 内部状态诊断 ---")

    # 获取 tokenizer_manager
    tm = getattr(engine, "tokenizer_manager", None)
    if tm is None:
        # 尝试其他属性名
        for attr in ["_tokenizer_manager", "tm"]:
            tm = getattr(engine, attr, None)
            if tm is not None:
                break

    if tm is not None:
        # model_config
        model_config = getattr(tm, "model_config", None)
        if model_config is not None:
            is_hybrid_swa = getattr(model_config, "is_hybrid_swa", "N/A")
            sliding_window_size = getattr(model_config, "sliding_window_size", "N/A")
            swa_layer_ids = getattr(model_config, "swa_attention_layer_ids", "N/A")
            full_layer_ids = getattr(model_config, "full_attention_layer_ids", "N/A")
            print(f"  model_config.is_hybrid_swa = {is_hybrid_swa}")
            print(f"  model_config.sliding_window_size = {sliding_window_size}")
            if isinstance(swa_layer_ids, list):
                print(f"  model_config.swa_attention_layer_ids = [{len(swa_layer_ids)} 层] {swa_layer_ids[:10]}{'...' if len(swa_layer_ids) > 10 else ''}")
            else:
                print(f"  model_config.swa_attention_layer_ids = {swa_layer_ids}")
            if isinstance(full_layer_ids, list):
                print(f"  model_config.full_attention_layer_ids = [{len(full_layer_ids)} 层] {full_layer_ids[:10]}{'...' if len(full_layer_ids) > 10 else ''}")
            else:
                print(f"  model_config.full_attention_layer_ids = {full_layer_ids}")

            # 检查 HF config 中的关键字段
            hf_config = getattr(model_config, "hf_config", None)
            hf_text_config = getattr(model_config, "hf_text_config", None)
            cfg = hf_text_config or hf_config
            if cfg is not None:
                print(f"\n  --- HF Config 中 SWA 相关字段 ---")
                for key in [
                    "use_sliding_window", "sliding_window",
                    "use_sliding_window_attention", "sliding_window_attention_size",
                    "layer_types", "full_attn_interleave", "num_swa_layers",
                ]:
                    val = getattr(cfg, key, "<未定义>")
                    if isinstance(val, list) and len(val) > 10:
                        print(f"  config.{key} = [{len(val)} 项] {val[:5]}...{val[-3:]}")
                    else:
                        print(f"  config.{key} = {val}")

                # 关键诊断：为什么 is_hybrid_swa 可能为 False
                has_swa_attn = bool(getattr(cfg, "use_sliding_window_attention", False))
                has_swa_size = getattr(cfg, "sliding_window_attention_size", None) is not None
                has_sw = bool(getattr(cfg, "use_sliding_window", False))
                has_sw_val = getattr(cfg, "sliding_window", None) is not None
                print(f"\n  --- is_hybrid_swa 判断链路 ---")
                print(f"  architectures 包含 HSAForCausalLM: True (否则不会走到这里)")
                print(f"  is_hybrid_swa_model() 初始结果: True")
                print(f"  检查 use_sliding_window_attention: {has_swa_attn} (需要为 True)")
                print(f"  检查 sliding_window_attention_size: {has_swa_size} (需要不为 None)")
                print(f"  → has_swa_window = {has_swa_attn and has_swa_size}")
                if not (has_swa_attn and has_swa_size):
                    print(f"  → is_hybrid_swa 被设为 False!")
                    if has_sw and has_sw_val:
                        print(f"  ⚠️  注意: config 中有 use_sliding_window={has_sw} 和 sliding_window={getattr(cfg, 'sliding_window', None)}")
                        print(f"     但代码检查的是 use_sliding_window_attention 和 sliding_window_attention_size")
                        print(f"     字段名不匹配，导致 SWA 层没有走 SWARadixCache!")
        else:
            print(f"  ⚠️ 无法获取 model_config")
    else:
        print(f"  ⚠️ 无法获取 tokenizer_manager")

    # ---- 3. 尝试获取 scheduler 内部的 tree_cache 类型 ----
    # Engine 通过 subprocess 运行 scheduler，无法直接访问
    # 但可以通过 /get_server_info 或 generate 的 meta_info 间接判断
    print(f"\n  --- tree_cache 类型推断 ---")
    if tm is not None:
        model_config = getattr(tm, "model_config", None)
        if model_config is not None:
            is_hybrid_swa = getattr(model_config, "is_hybrid_swa", False)
            if is_hybrid_swa:
                print(f"  推断: tree_cache = SWARadixCache (因为 is_hybrid_swa=True)")
                print(f"  → SWA 层有独立的 KV cache 池和 eviction 策略")
            else:
                print(f"  推断: tree_cache = RadixCache (因为 is_hybrid_swa=False)")
                print(f"  → 所有层共用一个 KV cache 池，SWA 层没有独立管理")

    # ---- 4. 通过实际推理验证 prefix cache 在 SWA 层的复用 ----
    print(f"\n  --- Prefix Cache 复用验证 ---")
    # 使用一个较长的 prompt，确保跨越多个 page
    test_prompt = slice_prompt(500) + " Summarize the key points discussed above:"
    sampling_params = {"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0}

    # 第一次推理（冷启动，带 logprob）
    res1 = engine.generate(
        prompt=test_prompt, sampling_params=sampling_params,
        return_logprob=True, top_logprobs_num=5,
    )
    meta1 = res1.get("meta_info", {})
    cached1 = meta1.get("cached_tokens", -1)
    prompt1 = meta1.get("prompt_tokens", -1)
    text1 = res1.get("text", "")
    ids1 = res1.get("output_ids", [])
    top_logprobs1 = meta1.get("output_top_logprobs", [])
    token_logprobs1 = meta1.get("output_token_logprobs", [])
    print(f"  [第1次] prompt_tokens={prompt1}, cached_tokens={cached1}, output='{text1[:60]}...'")

    # 第二次推理（预期命中 cache，带 logprob）
    res2 = engine.generate(
        prompt=test_prompt, sampling_params=sampling_params,
        return_logprob=True, top_logprobs_num=5,
    )
    meta2 = res2.get("meta_info", {})
    cached2 = meta2.get("cached_tokens", -1)
    prompt2 = meta2.get("prompt_tokens", -1)
    text2 = res2.get("text", "")
    ids2 = res2.get("output_ids", [])
    top_logprobs2 = meta2.get("output_top_logprobs", [])
    token_logprobs2 = meta2.get("output_token_logprobs", [])
    print(f"  [第2次] prompt_tokens={prompt2}, cached_tokens={cached2}, output='{text2[:60]}...'")

    # 第三次推理：使用前缀相同但更长的 prompt
    test_prompt_longer = test_prompt + " Additionally, please elaborate on the implications:"
    res3 = engine.generate(prompt=test_prompt_longer, sampling_params=sampling_params)
    meta3 = res3.get("meta_info", {})
    cached3 = meta3.get("cached_tokens", -1)
    prompt3 = meta3.get("prompt_tokens", -1)
    text3 = res3.get("text", "")
    print(f"  [第3次-更长] prompt_tokens={prompt3}, cached_tokens={cached3}, output='{text3[:60]}...'")

    # 汇总
    print(f"\n  --- 诊断汇总 ---")
    output_match = (ids1 == ids2)
    cache_hit = cached2 > 0
    prefix_reuse = cached3 > 0
    is_tie_breaking = False  # 标记是否为 tie breaking 导致的不一致

    print(f"  冷启动 vs 热启动 output_ids 一致: {'✅' if output_match else '❌'}")
    if not output_match:
        # 找到首个差异位置
        min_len = min(len(ids1), len(ids2))
        diff_pos = next((j for j in range(min_len) if ids1[j] != ids2[j]), min_len)
        # 提取实际采样的 token id
        cold_sampled = ids1[diff_pos] if diff_pos < len(ids1) else None
        warm_sampled = ids2[diff_pos] if diff_pos < len(ids2) else None
        print(f"  首个差异位置: pos={diff_pos} (cold_len={len(ids1)}, warm_len={len(ids2)})")
        print(f"  Cold 实际采样 token_id={cold_sampled}, Warm 实际采样 token_id={warm_sampled}")
        print(f"  cold_text: \"{text1[:80]}...\"")
        print(f"  warm_text: \"{text2[:80]}...\"")

        # 打印差异位置的 token logprobs
        print(f"\n  --- 差异位置 (pos={diff_pos}) 的 logprobs 诊断 ---")
        if diff_pos < len(token_logprobs1):
            print(f"  Cold token_logprob[{diff_pos}]: {token_logprobs1[diff_pos]}")
        if diff_pos < len(token_logprobs2):
            print(f"  Warm token_logprob[{diff_pos}]: {token_logprobs2[diff_pos]}")
        if diff_pos < len(top_logprobs1):
            print(f"  Cold top5_logprobs[{diff_pos}]: {top_logprobs1[diff_pos]}")
        if diff_pos < len(top_logprobs2):
            print(f"  Warm top5_logprobs[{diff_pos}]: {top_logprobs2[diff_pos]}")

        # 检查 top1-top2 logprob gap（判断是否 tie breaking）
        for label, top_lp in [("Cold", top_logprobs1), ("Warm", top_logprobs2)]:
            if diff_pos < len(top_lp) and len(top_lp[diff_pos]) >= 2:
                entry0 = top_lp[diff_pos][0]
                entry1 = top_lp[diff_pos][1]
                top1_logp = entry0[0] if isinstance(entry0, (list, tuple)) else None
                top2_logp = entry1[0] if isinstance(entry1, (list, tuple)) else None
                if top1_logp is not None and top2_logp is not None:
                    gap = abs(top1_logp - top2_logp)
                    print(f"  {label} top1-top2 logprob gap: {gap:.6f} {'(⚠️ tie breaking: gap < 0.01)' if gap < 0.01 else ''}")

        # Top2 候选集合对比
        def _extract_top_tokens_swa(top_list, pos, k=2):
            if pos >= len(top_list) or len(top_list[pos]) < k:
                return None, []
            tokens, details = [], []
            for entry in top_list[pos][:k]:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    tid, logp = entry[1], entry[0]
                    ttext = entry[2] if (len(entry) >= 3 and entry[2] is not None) else f"id={tid}"
                    tokens.append(tid)
                    details.append((tid, logp, ttext))
            return set(tokens) if tokens else None, details

        cold_top2_set, cold_top2_details = _extract_top_tokens_swa(top_logprobs1, diff_pos, k=2)
        warm_top2_set, warm_top2_details = _extract_top_tokens_swa(top_logprobs2, diff_pos, k=2)
        print(f"\n  --- Top2 候选集合对比 ---")
        if cold_top2_details:
            print(f"  Cold top2: {[(d[2], f'logp={d[1]:.4f}') for d in cold_top2_details]}")
        if warm_top2_details:
            print(f"  Warm top2: {[(d[2], f'logp={d[1]:.4f}') for d in warm_top2_details]}")
        if cold_top2_set is not None and warm_top2_set is not None:
            if cold_top2_set == warm_top2_set:
                is_tie_breaking = True
                # 检查两边的 top1 token_id 是否一致（top_logprobs 排序的 top1）
                cold_top1_tid = cold_top2_details[0][0] if cold_top2_details else None
                warm_top1_tid = warm_top2_details[0][0] if warm_top2_details else None
                top1_same = (cold_top1_tid == warm_top1_tid)
                print(f"  ✅ Top2 候选集合一致 (同一组 token，仅排序/采样不同 → tie breaking / 精度差异)")
                print(f"  Cold top1(logprobs排序)={cold_top1_tid}, Warm top1(logprobs排序)={warm_top1_tid} → {'排序也一致' if top1_same else '排序不同'}")
                print(f"  Cold 实际采样={cold_sampled}, Warm 实际采样={warm_sampled}")
                if top1_same and cold_sampled != warm_sampled:
                    print(f"  💡 说明: top_logprobs 中 top1 一致(都是 {cold_top1_tid})，但 Cold 实际采样了 {cold_sampled}")
                    print(f"     这是因为 Cold 的 top1/top2 logprob 完全相同(tie)，采样器选了另一个")
                    print(f"     → cache 复用导致的 bf16 精度差异打破了 tie，使 Warm 的 top1 拉开差距后稳定选中 {warm_sampled}")
            else:
                print(f"  ❌ Top2 候选集合不一致 (Cold={cold_top2_set}, Warm={warm_top2_set})")

    # 最终结论
    if not output_match and is_tie_breaking:
        print(f"\n  🔍 最终结论: output_ids 不一致，但属于 tie breaking (Top2 候选集合一致)")
        print(f"     → 不一致由 bf16 精度差异导致，非功能性 bug")
    elif not output_match:
        print(f"\n  ⚠️  最终结论: output_ids 不一致，且 Top2 候选集合也不同，可能存在 cache 复用 bug")

    print(f"  第2次 cache 命中: {'✅' if cache_hit else '❌'} (cached_tokens={cached2})")
    print(f"  第3次 前缀复用: {'✅' if prefix_reuse else '❌'} (cached_tokens={cached3})")

    if tm is not None:
        model_config = getattr(tm, "model_config", None)
        if model_config is not None and not getattr(model_config, "is_hybrid_swa", False):
            print(f"\n  ⚠️  当前 is_hybrid_swa=False，走的是普通 RadixCache。")
            print(f"     SWA 层的 KV cache 没有独立管理（没有 eviction 窗口优化）。")
            print(f"     如果需要启用 SWARadixCache，需要在 config.json 中添加:")
            print(f'       "use_sliding_window_attention": true,')
            print(f'       "sliding_window_attention_size": 512')
            print(f"     （对应代码中 _derive_hybrid_model 检查的字段名）")

    return {
        "output_match": output_match,
        "cache_hit": cache_hit,
        "prefix_reuse": prefix_reuse,
    }


def main():
    import torch

    print_separator("SGLang HSA: 顺序推理 vs Batch 推理 对比测试")
    print(f"  模型: {CHECKPOINT_PATH}")
    print(f"  TP: {SGLANG_TP}, Page Size: {PAGE_SIZE}")
    print(f"  Max New Tokens: {MAX_NEW_TOKENS}")
    print(f"  Max Total Tokens: {MAX_TOTAL_TOKENS}")
    print(f"  Chunked Prefill Size: {CHUNKED_PREFILL_SIZE}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # 构造 8 个长短不一的 prompt（使用切片方式，避免重复文本）
    test_prompts = [
        "What is artificial intelligence? In simple terms, it can be understood as",
        slice_prompt(30) + " One key concept that follows from this is",
        slice_prompt(150) + " The main idea behind this is that",
        slice_prompt(500) + " Building on the discussion above,",
        slice_prompt(1000) + " From these findings, we can further see that",
        slice_prompt(1500) + " A more detailed analysis would begin by noting that",
        slice_prompt(2500) + " When comparing the approaches mentioned above, one important difference is that",
        slice_prompt(3000) + " Taking all these points together, a comprehensive summary would start with",
    ]

    print(f"\n  构造了 {len(test_prompts)} 个长短不一的 prompt:")
    for i, p in enumerate(test_prompts):
        print(f"    [{i}] ~{len(p.split())} words")

    # 1. 创建 Engine
    print_separator("Step 1: 创建 SGLang Engine (HSA backend)")
    engine = create_engine(CHUNKED_PREFILL_SIZE)

    # 2. Warmup
    print_separator("Step 2: Warmup")
    warmup(engine, PAGE_SIZE)

    # 3. SWA Cache 诊断
    print_separator("Step 3: SWA Cache 诊断")
    swa_diag = test_swa_cache_diagnosis(engine)

    # 4. 顺序 vs Batch 对比
    print_separator("Step 4: 顺序 vs Batch 对比测试")
    result = test_sequential_vs_batch(engine, test_prompts, MAX_NEW_TOKENS)

    # 5. Prefix Cache 一致性测试（方案 A）
    print_separator("Step 5: Prefix Cache 一致性测试（方案 A）")
    cache_test_prompts = test_prompts[:5]
    print(f"  使用 {len(cache_test_prompts)} 个 prompt 进行 prefix cache 一致性测试")
    cache_result = test_prefix_cache_consistency(engine, cache_test_prompts, MAX_NEW_TOKENS)

    # 6. 清理
    print_separator("Step 6: 清理资源")
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    print("[Done] Engine 已关闭，GPU 显存已释放。")

    # 7. Prefix Cache 严格一致性测试（方案 B：两个 Engine 对比）
    print_separator("Step 7: Prefix Cache 严格一致性测试（方案 B）")
    cache_test_prompts = test_prompts[3:]
    print(f"  使用 {len(cache_test_prompts)} 个 prompt 进行方案 B 测试")
    strict_result = test_prefix_cache_strict(
        cache_test_prompts, MAX_NEW_TOKENS, CHUNKED_PREFILL_SIZE
    )


if __name__ == "__main__":
    main()


# 运行命令:
# CUDA_VISIBLE_DEVICES=7 PYTHONPATH="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/sglang/python:${PYTHONPATH:-}" \
#   SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
#   /root/sglang/python/.venv/bin/python code_exp/test_multi_batch_infer.py
