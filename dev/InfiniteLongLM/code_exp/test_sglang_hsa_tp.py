"""
SGLang + HSA: TP 多卡推理测试脚本。

测试目标：
  1. 验证 HSA attention backend 能否在 TP > 1 的情况下正常启动
  2. 验证 TP 多卡推理的输出是否合理（非乱码、非空）
  3. 验证不同长度 prompt 在 TP 模式下的推理正确性
  4. 验证 batch 推理在 TP 模式下是否正常工作
  5. （可选）对比 TP=1 和 TP=N 的输出一致性

用法：
  # TP=2 双卡推理
  CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/sglang/python:${PYTHONPATH:-}" \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_TP=2 \
    /root/sglang/python/.venv/bin/python code_exp/test_sglang_hsa_tp.py

  # TP=4 四卡推理
  CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/sglang/python:${PYTHONPATH:-}" \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_TP=4 \
    /root/sglang/python/.venv/bin/python code_exp/test_sglang_hsa_tp.py

  # 对比模式：先跑 TP=1 再跑 TP=2，对比输出一致性
  CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/sglang/python:${PYTHONPATH:-}" \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_TP=2 COMPARE_WITH_TP1=1 \
    /root/sglang/python/.venv/bin/python code_exp/test_sglang_hsa_tp.py

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/sglang/python:${PYTHONPATH:-}" \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_TP=1 /root/sglang/python/.venv/bin/python code_exp/test_sglang_hsa_tp.py --test-length 120000

    CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/sglang/python:${PYTHONPATH:-}" \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_TP=2 /root/sglang/python/.venv/bin/python code_exp/test_sglang_hsa_tp.py --test-length 270000

    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/sglang/python:${PYTHONPATH:-}" \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_TP=4 /root/sglang/python/.venv/bin/python code_exp/test_sglang_hsa_tp.py --test-length 131000
"""

import gc
import json
import os
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

# ======================== 配置区 ========================
DEFAULT_CHECKPOINT = "/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/checkpoints/lhsa-olmo3-interleave"

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT)
SGLANG_TP = int(os.environ.get("SGLANG_TP", "2"))  # 默认 TP=2
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "50"))
PAGE_SIZE = int(os.environ.get("SGLANG_PAGE_SIZE", "64"))
MAX_TOTAL_TOKENS = int(os.environ.get("SGLANG_MAX_TOTAL_TOKENS", "131072"))
CHUNKED_PREFILL_SIZE = int(os.environ.get("CHUNKED_PREFILL_SIZE", "16384"))
MEM_FRACTION_STATIC = float(os.environ.get("MEM_FRACTION_STATIC", "0.85"))

# 允许 context_length 超过 config.json 中的 max_position_embeddings
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

# 是否同时跑 TP=1 做对比
COMPARE_WITH_TP1 = os.environ.get("COMPARE_WITH_TP1", "0") == "1"

# 单次长度测试模式：传入 --test-length <N> 时只测试该长度能否正常推理
# 也可通过环境变量 TEST_LENGTH 设置
TEST_LENGTH = None  # 在 main() 中通过 argparse 解析
# ========================================================

# 一段足够长的不重复英文文本，用于按空格切片构造不同长度的 prompt
# 涵盖 AI、NLP、Transformer、HSA、分布式训练等话题，约 5000+ words
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


def print_separator(title: str, char: str = "=", width: int = 80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def create_engine(tp_size: int, tag: str = ""):
    """创建 SGLang Engine（HSA backend），使用指定的 tp_size

    当 MAX_TOTAL_TOKENS > 0 时使用该值作为上限；
    当 MAX_TOTAL_TOKENS == 0 时不传 max_total_tokens，让 SGLang 自动 profile 出最大值。
    """
    import sglang as sgl

    label = f"[Engine{' ' + tag if tag else ''}]"
    print(f"{label} 正在加载模型: {CHECKPOINT_PATH}")
    print(f"{label} 参数: tp_size={tp_size}, page_size={PAGE_SIZE}, "
          f"max_total_tokens={MAX_TOTAL_TOKENS} ({'auto' if MAX_TOTAL_TOKENS == 0 else 'fixed'}), "
          f"chunked_prefill_size={CHUNKED_PREFILL_SIZE}, "
          f"mem_fraction_static={MEM_FRACTION_STATIC}")

    try:
        engine_kwargs = dict(
            model_path=CHECKPOINT_PATH,
            tp_size=tp_size,
            attention_backend="hsa",
            page_size=PAGE_SIZE,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            chunked_prefill_size=CHUNKED_PREFILL_SIZE,
            mem_fraction_static=MEM_FRACTION_STATIC,
        )
        if MAX_TOTAL_TOKENS > 0:
            engine_kwargs["max_total_tokens"] = MAX_TOTAL_TOKENS
            engine_kwargs["context_length"] = MAX_TOTAL_TOKENS
        # MAX_TOTAL_TOKENS == 0 时不传 max_total_tokens 和 context_length，
        # 让 SGLang 自动 profile KV cache 大小，context_len 从模型 config 推导

        engine = sgl.Engine(**engine_kwargs)
        actual_max_input = engine.scheduler_info.get("max_req_input_len", "unknown")
        print(f"{label} ✅ Engine 创建成功! (tp_size={tp_size}, 实际 max_req_input_len={actual_max_input})")
        return engine
    except Exception as e:
        print(f"{label} ❌ Engine 创建失败!")
        print(f"{label} 错误类型: {type(e).__name__}")
        print(f"{label} 错误信息: {e}")
        traceback.print_exc()
        return None


def _create_engine_with_max_tokens(tp_size: int, max_total_tokens: int = None,
                                   context_length: int = None, tag: str = ""):
    """创建 SGLang Engine，支持自定义 max_total_tokens（用于最长推理长度搜索）

    当 max_total_tokens=None 时，不传该参数，让 SGLang 自动 profile 出 GPU 能容纳的最大 KV cache 大小，
    从而突破固定值的限制，探索单卡/多卡推理的极限长度。
    """
    import sglang as sgl

    label = f"[Engine{' ' + tag if tag else ''}]"
    print(f"{label} 正在创建 Engine: tp_size={tp_size}, max_total_tokens={max_total_tokens}")

    try:
        engine_kwargs = dict(
            model_path=CHECKPOINT_PATH,
            tp_size=tp_size,
            attention_backend="hsa",
            page_size=PAGE_SIZE,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            chunked_prefill_size=CHUNKED_PREFILL_SIZE,
            mem_fraction_static=MEM_FRACTION_STATIC,
        )
        if max_total_tokens is not None:
            engine_kwargs["max_total_tokens"] = max_total_tokens
            engine_kwargs["context_length"] = context_length or max_total_tokens
        elif context_length is not None:
            # max_total_tokens=None 时不传 max_total_tokens，让 SGLang 自动 profile KV cache 大小
            # 但传入调用方指定的 context_length，避免 _validate_one_request 按模型 config
            # 的 max_position_embeddings (131072) 拒绝超长输入
            engine_kwargs["context_length"] = context_length
        # 两者都为 None 时，不传 context_length，让 SGLang 从模型 config 推导

        engine = sgl.Engine(**engine_kwargs)

        # 从 scheduler_info 中读取实际分配的限制
        actual_max_input = engine.scheduler_info.get("max_req_input_len", "unknown")
        print(f"{label} ✅ Engine 创建成功! (max_total_tokens={max_total_tokens}, 实际 max_req_input_len={actual_max_input})")
        return engine
    except Exception as e:
        print(f"{label} ❌ Engine 创建失败! (max_total_tokens={max_total_tokens})")
        print(f"{label} 错误: {e}")
        return None


def warmup(engine, page_size: int, tag: str = ""):
    """预热：触发 TileLang JIT 编译等"""
    label = f"[Warmup{' ' + tag if tag else ''}]"
    warmup_prompt = "Hello " * max(page_size * 2, 1024)
    warmup_tokens = 3
    print(f"{label} 正在预热（生成 {warmup_tokens} tokens 触发 JIT 编译）...")
    t0 = time.time()
    try:
        _ = engine.generate(
            prompt=warmup_prompt,
            sampling_params={"max_new_tokens": warmup_tokens, "temperature": 0.0},
        )
        print(f"{label} ✅ 预热完成，耗时 {time.time() - t0:.2f}s")
        return True
    except Exception as e:
        print(f"{label} ❌ 预热失败! 错误: {e}")
        traceback.print_exc()
        return False


def run_single_inference(engine, prompt: str, max_new_tokens: int,
                          label: str = "") -> Optional[Dict]:
    """单条推理（带 logprobs 输出）"""
    try:
        t0 = time.time()
        result = engine.generate(
            prompt=prompt,
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
            return_logprob=True,
            top_logprobs_num=5,
        )
        elapsed = time.time() - t0

        meta = result.get("meta_info", {})
        output_ids = result.get("output_ids", [])
        text = result.get("text", "")
        prompt_tokens = meta.get("prompt_tokens", -1)
        cached_tokens = meta.get("cached_tokens", -1)
        output_top_logprobs = meta.get("output_top_logprobs", [])
        output_token_logprobs = meta.get("output_token_logprobs", [])

        text_preview = text[:80].replace("\n", "\\n")
        print(f"  {label} ✅ prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}, "
              f"output_tokens={len(output_ids)}, 耗时={elapsed:.3f}s")
        print(f"    text=\"{text_preview}...\"")

        return {
            "success": True,
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "output_ids": output_ids,
            "text": text,
            "output_top_logprobs": output_top_logprobs,
            "output_token_logprobs": output_token_logprobs,
        }
    except Exception as e:
        print(f"  {label} ❌ 失败! 错误: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def run_batch_inference(engine, prompts: List[str], max_new_tokens: int,
                         label: str = "") -> Optional[Dict]:
    """Batch 推理"""
    try:
        t0 = time.time()
        results = engine.generate(
            prompt=prompts,
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
        )
        elapsed = time.time() - t0

        if isinstance(results, dict):
            results = [results]

        print(f"  {label} ✅ batch_size={len(prompts)}, 耗时={elapsed:.3f}s")
        items = []
        for i, res in enumerate(results):
            meta = res.get("meta_info", {})
            output_ids = res.get("output_ids", [])
            text = res.get("text", "")
            prompt_tokens = meta.get("prompt_tokens", -1)
            cached_tokens = meta.get("cached_tokens", -1)
            text_preview = text[:60].replace("\n", "\\n")
            print(f"    [{i}] prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}, "
                  f"output_tokens={len(output_ids)}, text=\"{text_preview}...\"")
            items.append({
                "output_ids": output_ids,
                "text": text,
                "prompt_tokens": prompt_tokens,
            })

        return {
            "success": True,
            "elapsed": elapsed,
            "items": items,
        }
    except Exception as e:
        print(f"  {label} ❌ 失败! 错误: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def _load_tokenizer():
    """加载 tokenizer 用于计算实际 token 数"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"  ⚠️  无法加载 tokenizer: {e}，将使用 word 数估算")
        return None


# 全局 tokenizer（延迟加载）
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _load_tokenizer()
    return _tokenizer


def _count_tokens(text: str) -> str:
    """计算文本的实际 token 数，返回描述字符串"""
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        n_tokens = len(tokenizer.encode(text))
        return f"{n_tokens} tokens"
    else:
        n_words = len(text.split())
        return f"~{n_words} words"


def test_various_lengths(engine, max_new_tokens: int, tag: str = ""):
    """测试不同长度 prompt 的推理"""
    print_separator(f"测试不同长度 prompt 的单条推理 {tag}")

    # 将 BASE_TEXT 按空格拆分为 word 列表，按不同 word 数切片构造不同长度的 prompt
    # 这样每个 prompt 的内容都是不重复的连续文本，避免重复文本诱导模型输出退化
    words = BASE_TEXT.split()
    total_words = len(words)

    def slice_prompt(n_words: int) -> str:
        """从 BASE_TEXT 中取前 n_words 个 word 拼成 prompt"""
        n = min(n_words, total_words)
        return " ".join(words[:n])

    raw_cases = [
        ("极短", "What is artificial intelligence?"),
        ("短", slice_prompt(30) + " Explain the key concept."),
        ("中等", slice_prompt(150) + " What is the main idea?"),
        ("中长", slice_prompt(2000) + " Summarize the above."),
        ("长", slice_prompt(2500) + " What are the key findings?"),
        ("较长", slice_prompt(3000) + " Provide a detailed analysis."),
        ("长文本", slice_prompt(total_words) + " Write a comprehensive summary."),
    ]

    # 动态计算每个 prompt 的实际 token 数
    test_cases = []
    for short_name, prompt in raw_cases:
        token_info = _count_tokens(prompt)
        name = f"{short_name} ({token_info})"
        test_cases.append((name, prompt))

    results = []
    all_success = True
    for name, prompt in test_cases:
        label = f"[{name}]"
        res = run_single_inference(engine, prompt, max_new_tokens, label)
        results.append((name, res))
        if res is None or not res.get("success", False):
            all_success = False

    print()
    if all_success:
        print(f"  🎉 所有长度的 prompt 推理均成功! {tag}")
    else:
        print(f"  ⚠️  部分 prompt 推理失败! {tag}")

    return results


def test_batch(engine, max_new_tokens: int, tag: str = ""):
    """测试 batch 推理"""
    print_separator(f"测试 Batch 推理 {tag}")

    words = BASE_TEXT.split()
    total_words = len(words)

    def slice_prompt(n_words: int) -> str:
        n = min(n_words, total_words)
        return " ".join(words[:n])

    prompts = [
        "What is artificial intelligence?",
        slice_prompt(30) + " Explain the key concept.",
        slice_prompt(150) + " What is the main idea?",
        slice_prompt(2000) + " Summarize the above.",
        slice_prompt(2500) + " What are the key findings?",
        slice_prompt(3000) + " Provide a detailed analysis.",
    ]

    print(f"  Batch 大小: {len(prompts)}")
    for i, p in enumerate(prompts):
        print(f"    [{i}] ~{len(p.split())} words")

    res = run_batch_inference(engine, prompts, max_new_tokens, f"[Batch {tag}]")
    return res


def test_sequential_vs_batch_consistency(engine, max_new_tokens: int, tag: str = ""):
    """顺序推理 vs Batch 推理一致性对比"""
    print_separator(f"顺序 vs Batch 一致性对比 {tag}")

    words = BASE_TEXT.split()
    total_words = len(words)

    def slice_prompt(n_words: int) -> str:
        n = min(n_words, total_words)
        return " ".join(words[:n])

    prompts = [
        "What is artificial intelligence? In simple terms, it can be understood as",
        slice_prompt(30) + " One key concept that follows from this is",
        slice_prompt(150) + " The main idea behind this is that",
        slice_prompt(2000) + " Building on the discussion above,",
    ]

    N = len(prompts)
    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0.0}

    # 顺序推理
    print(f"\n  --- 顺序推理 ({N} 条) ---")
    seq_results = []
    t_seq_start = time.time()
    for i, p in enumerate(prompts):
        res = engine.generate(prompt=p, sampling_params=sampling_params)
        output_ids = res.get("output_ids", [])
        text = res.get("text", "")
        meta = res.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", -1)
        text_preview = text[:60].replace("\n", "\\n")
        print(f"    [{i}] prompt_tokens={prompt_tokens}, output_tokens={len(output_ids)}, "
              f"text=\"{text_preview}...\"")
        seq_results.append({"output_ids": output_ids, "text": text})
    t_seq = time.time() - t_seq_start

    # Batch 推理
    print(f"\n  --- Batch 推理 ({N} 条) ---")
    t_batch_start = time.time()
    batch_raw = engine.generate(prompt=prompts, sampling_params=sampling_params)
    t_batch = time.time() - t_batch_start

    if isinstance(batch_raw, dict):
        batch_raw = [batch_raw]

    batch_results = []
    for i, res in enumerate(batch_raw):
        output_ids = res.get("output_ids", [])
        text = res.get("text", "")
        meta = res.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", -1)
        text_preview = text[:60].replace("\n", "\\n")
        print(f"    [{i}] prompt_tokens={prompt_tokens}, output_tokens={len(output_ids)}, "
              f"text=\"{text_preview}...\"")
        batch_results.append({"output_ids": output_ids, "text": text})

    # 对比
    print(f"\n  --- 对比结果 ---")
    all_match = True
    for i in range(N):
        seq_ids = seq_results[i]["output_ids"]
        bat_ids = batch_results[i]["output_ids"]
        match = (seq_ids == bat_ids)
        if not match:
            all_match = False
            min_len = min(len(seq_ids), len(bat_ids))
            diff_pos = next((j for j in range(min_len) if seq_ids[j] != bat_ids[j]), min_len)
            print(f"    [{i}] ❌ 不一致 (首个差异位置={diff_pos})")
            print(f"         seq: \"{seq_results[i]['text'][:80]}...\"")
            print(f"         bat: \"{batch_results[i]['text'][:80]}...\"")
        else:
            print(f"    [{i}] ✅ 一致 (output_tokens={len(seq_ids)})")

    print(f"\n  顺序耗时: {t_seq:.3f}s, Batch耗时: {t_batch:.3f}s, "
          f"加速比: {t_seq / t_batch:.2f}x")

    if all_match:
        print(f"  🎉 顺序 vs Batch 输出完全一致! {tag}")
    else:
        print(f"  ⚠️  存在不一致! {tag}")

    return all_match, seq_results, batch_results


def _extract_top2_token_set(top_logprobs_list, pos):
    """从 top_logprobs 中提取指定位置的 top2 token id 集合"""
    if pos >= len(top_logprobs_list) or len(top_logprobs_list[pos]) < 2:
        return None
    entry = top_logprobs_list[pos]
    try:
        # entry 格式: [(logprob, token_id, token_str), ...]
        t1 = entry[0][1] if isinstance(entry[0], (list, tuple)) else None
        t2 = entry[1][1] if isinstance(entry[1], (list, tuple)) else None
        if t1 is not None and t2 is not None:
            return {t1, t2}
    except (IndexError, TypeError):
        pass
    return None


def compare_tp_results(results_tp1: List, results_tpN: List, tp_size: int):
    """对比 TP=1 和 TP=N 的输出，不一致时打印 logprobs 诊断 tie breaking。
    
    判断标准：
      - 严格一致：output_ids 完全相同
      - 等价一致（top2）：output_ids 不同，但首个差异位置的双方 {top1, top2} token 集合相同
        （说明差异仅由浮点精度导致的 tie breaking 引起）
      - 真正不一致：上述两者都不满足
    """
    print_separator(f"TP=1 vs TP={tp_size} 输出对比")

    n_strict_match = 0   # 严格一致
    n_top2_match = 0     # top2 等价一致（tie breaking）
    n_real_diff = 0      # 真正不一致
    n_skipped = 0

    for i, ((name1, r1), (name2, r2)) in enumerate(zip(results_tp1, results_tpN)):
        if not r1.get("success") or not r2.get("success"):
            print(f"  [{i}] ⚠️  跳过（有失败的推理）")
            n_skipped += 1
            continue

        ids1 = r1.get("output_ids", [])
        ids2 = r2.get("output_ids", [])

        if ids1 == ids2:
            n_strict_match += 1
            print(f"  [{i}] {name1}: ✅ 严格一致 (output_tokens={len(ids1)})")
            continue

        # output_ids 不同，找首个差异位置
        min_len = min(len(ids1), len(ids2))
        diff_pos = next((j for j in range(min_len) if ids1[j] != ids2[j]), min_len)

        # 获取双方在差异位置的 top2 token 集合
        tp1_top = r1.get("output_top_logprobs", [])
        tpN_top = r2.get("output_top_logprobs", [])
        tp1_top2_set = _extract_top2_token_set(tp1_top, diff_pos)
        tpN_top2_set = _extract_top2_token_set(tpN_top, diff_pos)

        # 判断 top2 集合是否一致
        is_top2_match = (tp1_top2_set is not None and tpN_top2_set is not None
                         and tp1_top2_set == tpN_top2_set)

        if is_top2_match:
            n_top2_match += 1
            print(f"  [{i}] {name1}: ⚡ top2 等价一致 (首个差异位置={diff_pos}, "
                  f"top2_set={tp1_top2_set}, 仅 tie breaking 差异)")
        else:
            n_real_diff += 1
            print(f"  [{i}] {name1}: ❌ 不一致 (首个差异位置={diff_pos}, "
                  f"TP1_len={len(ids1)}, TPN_len={len(ids2)})")

        # 无论 top2 是否一致，都打印详细诊断信息
        print(f"       TP=1: \"{r1['text'][:80]}...\"")
        print(f"       TP={tp_size}: \"{r2['text'][:80]}...\"")

        tp1_logprobs = r1.get("output_token_logprobs", [])
        tpN_logprobs = r2.get("output_token_logprobs", [])
        print(f"\n       --- 首个差异位置 (pos={diff_pos}) 的 logprobs 诊断 ---")
        if diff_pos < len(tp1_logprobs):
            print(f"       TP=1 token_logprob[{diff_pos}]: {tp1_logprobs[diff_pos]}")
        if diff_pos < len(tpN_logprobs):
            print(f"       TP={tp_size} token_logprob[{diff_pos}]: {tpN_logprobs[diff_pos]}")
        if diff_pos < len(tp1_top):
            print(f"       TP=1 top5_logprobs[{diff_pos}]: {tp1_top[diff_pos]}")
        if diff_pos < len(tpN_top):
            print(f"       TP={tp_size} top5_logprobs[{diff_pos}]: {tpN_top[diff_pos]}")

        # 检测 tie breaking：top1 和 top2 的 logprob 差值
        for side_name, side_top in [("TP=1", tp1_top), (f"TP={tp_size}", tpN_top)]:
            if diff_pos < len(side_top) and len(side_top[diff_pos]) >= 2:
                top1_logp = side_top[diff_pos][0][0] if isinstance(side_top[diff_pos][0], (list, tuple)) else None
                top2_logp = side_top[diff_pos][1][0] if isinstance(side_top[diff_pos][1], (list, tuple)) else None
                if top1_logp is not None and top2_logp is not None:
                    gap = abs(top1_logp - top2_logp)
                    print(f"       {side_name} top1-top2 logprob gap: {gap:.6f} "
                          f"{'(⚠️ tie breaking: gap < 0.01)' if gap < 0.01 else ''}")

        # 打印 top2 集合对比
        if tp1_top2_set is not None and tpN_top2_set is not None:
            print(f"       TP=1 top2 token set: {tp1_top2_set}")
            print(f"       TP={tp_size} top2 token set: {tpN_top2_set}")
            if is_top2_match:
                print(f"       ✅ top2 集合一致 → 差异由 tie breaking 引起，可忽略")
            else:
                print(f"       ❌ top2 集合不一致 → 存在真正的精度差异")

    # 汇总
    print()
    total = n_strict_match + n_top2_match + n_real_diff
    print(f"  📊 对比汇总: 共 {total} 条 (跳过 {n_skipped} 条)")
    print(f"     严格一致:       {n_strict_match} 条")
    print(f"     top2 等价一致:  {n_top2_match} 条 (tie breaking, 可忽略)")
    print(f"     真正不一致:     {n_real_diff} 条")

    # 判断是否通过：严格一致 + top2 等价一致 都算通过
    all_pass = (n_real_diff == 0)
    print()
    if n_real_diff == 0 and n_top2_match == 0:
        print(f"  🎉 TP=1 和 TP={tp_size} 输出完全一致!")
    elif n_real_diff == 0 and n_top2_match > 0:
        print(f"  🎉 TP=1 和 TP={tp_size} 输出等价一致! ({n_top2_match} 条 tie breaking 差异已忽略)")
    else:
        print(f"  ⚠️  TP=1 和 TP={tp_size} 存在 {n_real_diff} 条真正不一致（非 tie breaking）")

    return all_pass


def _build_speed_test_cases():
    """构造速度测试用的 prompt 列表（复用 test_various_lengths 的逻辑）"""
    words = BASE_TEXT.split()
    total_words = len(words)

    def slice_prompt(n_words: int) -> str:
        n = min(n_words, total_words)
        return " ".join(words[:n])

    raw_cases = [
        ("极短", "What is artificial intelligence?"),
        ("短", slice_prompt(30) + " Explain the key concept."),
        ("中等", slice_prompt(150) + " What is the main idea?"),
        ("中长", slice_prompt(2000) + " Summarize the above."),
        ("长", slice_prompt(2500) + " What are the key findings?"),
        ("较长", slice_prompt(3000) + " Provide a detailed analysis."),
        ("长文本", slice_prompt(total_words) + " Write a comprehensive summary."),
    ]

    tokenizer = _get_tokenizer()
    test_cases = []
    for short_name, prompt in raw_cases:
        if tokenizer is not None:
            n_tokens = len(tokenizer.encode(prompt))
            name = f"{short_name} ({n_tokens} tokens)"
        else:
            name = f"{short_name} (~{len(prompt.split())} words)"
        test_cases.append((name, prompt))
    return test_cases


def _run_speed_measurement(engine, prompt: str, max_new_tokens: int):
    """对单条 prompt 进行速度测量，返回 (ttft, total_time, output_tokens)。

    - ttft (Time To First Token): 用 max_new_tokens=1 测量 prefill 耗时
    - total_time: 用完整 max_new_tokens 测量端到端耗时
    - decode_time = total_time - ttft
    """
    sampling_params_1 = {"max_new_tokens": 1, "temperature": 0.0}
    sampling_params_full = {"max_new_tokens": max_new_tokens, "temperature": 0.0}

    # 测量 TTFT (prefill)
    t0 = time.time()
    _ = engine.generate(prompt=prompt, sampling_params=sampling_params_1)
    ttft = time.time() - t0

    # 测量端到端
    t0 = time.time()
    result = engine.generate(prompt=prompt, sampling_params=sampling_params_full)
    total_time = time.time() - t0

    output_tokens = len(result.get("output_ids", []))
    return ttft, total_time, output_tokens


def test_tp_speed_comparison(tp_size: int, max_new_tokens: int = 50):
    """方案 B: 分别创建 TP=1 和 TP=N engine，测量推理速度并对比。

    流程：
      1. 创建 TP=1 engine → warmup → 跑所有 prompt 记录耗时 → 销毁
      2. 创建 TP=N engine → warmup → 跑所有 prompt 记录耗时 → 销毁
      3. 汇总对比，输出表格
    """
    print_separator(f"TP=1 vs TP={tp_size} 推理速度对比 (max_new_tokens={max_new_tokens})")

    test_cases = _build_speed_test_cases()
    print(f"  共 {len(test_cases)} 个测试样本")

    # ---- Phase A: TP=1 ----
    print_separator("Speed Test Phase A: TP=1 推理")
    engine_tp1 = create_engine(tp_size=1, tag="Speed-TP=1")
    if engine_tp1 is None:
        print("  ❌ TP=1 Engine 创建失败，跳过速度测试")
        return
    if not warmup(engine_tp1, PAGE_SIZE, tag="Speed-TP=1"):
        print("  ❌ TP=1 Warmup 失败，跳过速度测试")
        cleanup_engine(engine_tp1, tag="Speed-TP=1")
        return

    tp1_timings = []  # [(name, ttft, total_time, output_tokens), ...]
    for name, prompt in test_cases:
        try:
            ttft, total_time, output_tokens = _run_speed_measurement(
                engine_tp1, prompt, max_new_tokens
            )
            tp1_timings.append((name, ttft, total_time, output_tokens))
            decode_time = total_time - ttft
            print(f"  [{name}] TTFT={ttft:.3f}s, Total={total_time:.3f}s, "
                  f"Decode={decode_time:.3f}s, output_tokens={output_tokens}")
        except Exception as e:
            print(f"  [{name}] ❌ 失败: {e}")
            tp1_timings.append((name, None, None, None))

    cleanup_engine(engine_tp1, tag="Speed-TP=1")
    time.sleep(3)  # 等待 GPU 资源释放

    # ---- Phase B: TP=N ----
    print_separator(f"Speed Test Phase B: TP={tp_size} 推理")
    engine_tpN = create_engine(tp_size=tp_size, tag=f"Speed-TP={tp_size}")
    if engine_tpN is None:
        print(f"  ❌ TP={tp_size} Engine 创建失败，跳过速度测试")
        return
    if not warmup(engine_tpN, PAGE_SIZE, tag=f"Speed-TP={tp_size}"):
        print(f"  ❌ TP={tp_size} Warmup 失败，跳过速度测试")
        cleanup_engine(engine_tpN, tag=f"Speed-TP={tp_size}")
        return

    tpN_timings = []  # [(name, ttft, total_time, output_tokens), ...]
    for name, prompt in test_cases:
        try:
            ttft, total_time, output_tokens = _run_speed_measurement(
                engine_tpN, prompt, max_new_tokens
            )
            tpN_timings.append((name, ttft, total_time, output_tokens))
            decode_time = total_time - ttft
            print(f"  [{name}] TTFT={ttft:.3f}s, Total={total_time:.3f}s, "
                  f"Decode={decode_time:.3f}s, output_tokens={output_tokens}")
        except Exception as e:
            print(f"  [{name}] ❌ 失败: {e}")
            tpN_timings.append((name, None, None, None))

    cleanup_engine(engine_tpN, tag=f"Speed-TP={tp_size}")

    # ---- Phase C: 汇总对比 ----
    print_separator(f"TP=1 vs TP={tp_size} 推理速度汇总 (max_new_tokens={max_new_tokens})")

    # 表头
    header = (f"  {'Prompt长度':<22} | {'TP=1 TTFT':>10} | {'TP=N TTFT':>10} | {'加速比':>6} "
              f"| {'TP=1 Decode':>11} | {'TP=N Decode':>11} | {'加速比':>6} "
              f"| {'TP=1 Total':>10} | {'TP=N Total':>10} | {'加速比':>6}")
    print(header.replace("TP=N", f"TP={tp_size}"))
    print(f"  {'-' * (len(header) - 2)}")

    for (name1, ttft1, total1, out1), (name2, ttft2, total2, out2) in zip(tp1_timings, tpN_timings):
        if ttft1 is None or ttft2 is None:
            print(f"  {name1:<22} | {'SKIP':>10} | {'SKIP':>10} | {'N/A':>6} "
                  f"| {'SKIP':>11} | {'SKIP':>11} | {'N/A':>6} "
                  f"| {'SKIP':>10} | {'SKIP':>10} | {'N/A':>6}")
            continue

        decode1 = total1 - ttft1
        decode2 = total2 - ttft2

        ttft_speedup = ttft1 / ttft2 if ttft2 > 0 else float('inf')
        decode_speedup = decode1 / decode2 if decode2 > 0 else float('inf')
        total_speedup = total1 / total2 if total2 > 0 else float('inf')

        print(f"  {name1:<22} | {ttft1:>9.3f}s | {ttft2:>9.3f}s | {ttft_speedup:>5.2f}x "
              f"| {decode1:>10.3f}s | {decode2:>10.3f}s | {decode_speedup:>5.2f}x "
              f"| {total1:>9.3f}s | {total2:>9.3f}s | {total_speedup:>5.2f}x")

    # 计算平均加速比（仅统计有效数据）
    valid_ttft_speedups = []
    valid_decode_speedups = []
    valid_total_speedups = []
    for (_, ttft1, total1, _), (_, ttft2, total2, _) in zip(tp1_timings, tpN_timings):
        if ttft1 is not None and ttft2 is not None and ttft2 > 0 and total2 > 0:
            valid_ttft_speedups.append(ttft1 / ttft2)
            decode1 = total1 - ttft1
            decode2 = total2 - ttft2
            if decode2 > 0:
                valid_decode_speedups.append(decode1 / decode2)
            valid_total_speedups.append(total1 / total2)

    print(f"  {'-' * (len(header) - 2)}")
    if valid_ttft_speedups:
        avg_ttft = sum(valid_ttft_speedups) / len(valid_ttft_speedups)
        avg_decode = sum(valid_decode_speedups) / len(valid_decode_speedups) if valid_decode_speedups else 0
        avg_total = sum(valid_total_speedups) / len(valid_total_speedups)
        print(f"  {'平均':<22} | {'':>10} | {'':>10} | {avg_ttft:>5.2f}x "
              f"| {'':>11} | {'':>11} | {avg_decode:>5.2f}x "
              f"| {'':>10} | {'':>10} | {avg_total:>5.2f}x")

    print()
    print(f"  📝 说明: 加速比 = TP=1 耗时 / TP={tp_size} 耗时，>1 表示 TP={tp_size} 更快")
    print(f"  📝 TTFT = Time To First Token (prefill 耗时)")
    print(f"  📝 Decode = 生成阶段耗时 (Total - TTFT)")





# ======================== Subprocess 推理探测 ========================
# 由于 SGLang 推理阶段的 OOM 发生在 scheduler 子进程中，会触发 SIGQUIT
# 杀死整个 Python 进程，try/except 无法捕获。因此使用独立子进程来探测
# 最大可推理长度：子进程 OOM 被杀时，主进程通过 returncode 检测到失败。
# ==================================================================

_SUBPROCESS_INFERENCE_SCRIPT = '''
import os
import sys
import json
import time

def main():
    # 从环境变量读取参数
    tp_size = int(os.environ["_PROBE_TP_SIZE"])
    max_total_tokens = int(os.environ["_PROBE_MAX_TOTAL_TOKENS"])
    prompt_tokens = int(os.environ["_PROBE_PROMPT_TOKENS"])
    checkpoint = os.environ["_PROBE_CHECKPOINT"]
    page_size = int(os.environ.get("_PROBE_PAGE_SIZE", "64"))
    chunked_prefill_size = int(os.environ.get("_PROBE_CHUNKED_PREFILL_SIZE", "16384"))
    mem_fraction_static = float(os.environ.get("_PROBE_MEM_FRACTION_STATIC", "0.95"))

    import sglang as sgl

    # 创建 engine
    engine = sgl.Engine(
        model_path=checkpoint,
        tp_size=tp_size,
        attention_backend="hsa",
        page_size=page_size,
        max_total_tokens=max_total_tokens,
        context_length=max_total_tokens,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        chunked_prefill_size=chunked_prefill_size,
        mem_fraction_static=mem_fraction_static,
    )

    # warmup
    _ = engine.generate(
        prompt="Hello " * max(page_size * 2, 1024),
        sampling_params={"max_new_tokens": 3, "temperature": 0.0},
    )

    # 构造指定长度的 prompt（重复拼接简单文本）
    base = "The quick brown fox jumps over the lazy dog. "
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    base_tokens = len(tokenizer.encode(base))
    if base_tokens == 0:
        base_tokens = 1
    repeats = (prompt_tokens // base_tokens) + 2
    long_text = base * repeats
    # 精确截断到目标 token 数
    encoded = tokenizer.encode(long_text)
    if len(encoded) > prompt_tokens:
        encoded = encoded[:prompt_tokens]
    prompt = tokenizer.decode(encoded, skip_special_tokens=True)
    actual_tokens = len(tokenizer.encode(prompt))

    # 推理（max_new_tokens=1，只测 prefill）
    result = engine.generate(
        prompt=prompt,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
    )
    output_ids = result.get("output_ids", [])
    meta = result.get("meta_info", {})
    real_prompt_tokens = meta.get("prompt_tokens", -1)

    # 输出结果到 stdout（JSON 格式）
    print(json.dumps({
        "success": True,
        "actual_prompt_tokens": actual_tokens,
        "real_prompt_tokens": real_prompt_tokens,
        "output_tokens": len(output_ids),
    }))

    engine.shutdown()

if __name__ == "__main__":
    main()
'''


def _subprocess_try_inference(tp_size: int, max_total_tokens: int,
                               prompt_tokens: int, timeout: int = 600) -> bool:
    """在独立子进程中尝试推理，通过 returncode 判断成功/失败。

    子进程 OOM 被 SIGQUIT 杀死时 returncode != 0，主进程可以安全检测。
    """
    import tempfile

    # 将脚本写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(_SUBPROCESS_INFERENCE_SCRIPT)
        script_path = f.name

    try:
        # 构造环境变量
        env = os.environ.copy()
        env["_PROBE_TP_SIZE"] = str(tp_size)
        env["_PROBE_MAX_TOTAL_TOKENS"] = str(max_total_tokens)
        env["_PROBE_PROMPT_TOKENS"] = str(prompt_tokens)
        env["_PROBE_CHECKPOINT"] = CHECKPOINT_PATH
        env["_PROBE_PAGE_SIZE"] = str(PAGE_SIZE)
        env["_PROBE_CHUNKED_PREFILL_SIZE"] = str(CHUNKED_PREFILL_SIZE)
        env["_PROBE_MEM_FRACTION_STATIC"] = str(MEM_FRACTION_STATIC)

        # 使用与当前脚本相同的 Python 解释器
        python_exe = sys.executable

        result = subprocess.run(
            [python_exe, script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            # 尝试解析 stdout 中的 JSON 结果
            stdout_lines = result.stdout.strip().split('\n')
            for line in reversed(stdout_lines):
                line = line.strip()
                if line.startswith('{'):
                    try:
                        data = json.loads(line)
                        if data.get("success"):
                            return True
                    except json.JSONDecodeError:
                        pass
            # returncode=0 但没有找到成功的 JSON → 也算成功（可能输出被截断）
            return True
        else:
            # 打印子进程的 stderr 最后几行用于诊断
            stderr_lines = result.stderr.strip().split('\n')
            last_lines = stderr_lines[-5:] if len(stderr_lines) > 5 else stderr_lines
            for line in last_lines:
                if 'OutOfMemoryError' in line or 'SIGQUIT' in line or 'OOM' in line:
                    print(f"      (子进程 OOM: {line.strip()})")
                    break
            return False

    except subprocess.TimeoutExpired:
        print(f"      (子进程超时 {timeout}s)")
        return False
    except Exception as e:
        print(f"      (子进程异常: {e})")
        return False
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _find_max_inference_length(tp_size: int, max_total_tokens: int,
                                start_tokens: int = 8192) -> int:
    """通过 subprocess + 指数递增 + 二分搜索找到最大可推理的 prompt 长度。

    在独立子进程中执行推理，子进程 OOM 被杀时主进程可以安全检测。
    精度到 1024 tokens。
    """
    print(f"\n  --- 搜索 TP={tp_size} 最大可推理 prompt 长度 "
          f"(max_total_tokens={max_total_tokens}) ---")
    print(f"  (使用 subprocess 隔离，每次探测需要重新创建 engine)")

    # 阶段 1：指数递增找到上界
    last_success = 0
    current = start_tokens

    while current <= max_total_tokens:
        print(f"    尝试 prompt 长度: {current} tokens ... ", end="", flush=True)
        success = _subprocess_try_inference(
            tp_size=tp_size,
            max_total_tokens=max_total_tokens,
            prompt_tokens=current,
        )
        if success:
            last_success = current
            print("✅")
            current *= 2
        else:
            print("❌")
            break
    else:
        # 达到 max_total_tokens 上限仍然成功
        print(f"  TP={tp_size} 最大可推理长度 >= {last_success} tokens (达到 max_total_tokens 上限)")
        return last_success

    if last_success == 0:
        print(f"  ❌ TP={tp_size} 在 start_tokens={start_tokens} 时就无法推理")
        return 0

    # 阶段 2：二分搜索精调
    lo = last_success
    hi = current
    precision = 4096  # 精度到 4K tokens（每次 subprocess 开销较大，不需要太精确）

    while hi - lo > precision:
        mid = (lo + hi) // 2
        print(f"    二分: prompt 长度={mid} tokens ... ", end="", flush=True)
        success = _subprocess_try_inference(
            tp_size=tp_size,
            max_total_tokens=max_total_tokens,
            prompt_tokens=mid,
        )
        if success:
            lo = mid
            print("✅")
        else:
            hi = mid
            print("❌")

    print(f"  TP={tp_size} 最大可推理 prompt 长度 ≈ {lo} tokens")
    return lo


def _find_max_engine_tokens(tp_size: int, start_tokens: int = 524288,
                            max_tokens: int = 8388608) -> Optional[int]:
    """通过指数递增找到指定 TP 能创建 engine 的最大 max_total_tokens。

    从 start_tokens 开始，每次翻倍，找到第一个失败的值后，
    在上一个成功和当前失败之间二分，精度到 PAGE_SIZE 对齐。
    """
    print(f"\n  --- 搜索 TP={tp_size} 最大可创建 engine 的 max_total_tokens ---")

    # 阶段 1：指数递增找到上界
    last_success = None
    current = start_tokens

    while current <= max_tokens:
        engine = _create_engine_with_max_tokens(
            tp_size=tp_size, max_total_tokens=current,
            tag=f"MaxTokenSearch-TP={tp_size}"
        )
        if engine is not None:
            last_success = current
            cleanup_engine(engine, tag=f"MaxTokenSearch-TP={tp_size}")
            time.sleep(2)
            print(f"    max_total_tokens={current} ✅ 成功")
            current *= 2
        else:
            print(f"    max_total_tokens={current} ❌ 失败")
            time.sleep(2)
            break
    else:
        # 达到 max_tokens 上限仍然成功
        if last_success is not None:
            print(f"  TP={tp_size} 最大 max_total_tokens >= {last_success} (达到搜索上限)")
            return last_success

    if last_success is None:
        print(f"  ❌ TP={tp_size} 在 start_tokens={start_tokens} 时就无法创建 engine")
        return None

    # 阶段 2：二分搜索精调
    lo = last_success
    hi = current  # 第一个失败的值
    # 对齐到 PAGE_SIZE
    step = PAGE_SIZE * 16  # 精度为 16 个 page（约 1K tokens）

    while hi - lo > step:
        mid = ((lo + hi) // 2 // PAGE_SIZE) * PAGE_SIZE  # 对齐到 PAGE_SIZE
        if mid == lo:
            break
        engine = _create_engine_with_max_tokens(
            tp_size=tp_size, max_total_tokens=mid,
            tag=f"MaxTokenBisect-TP={tp_size}"
        )
        if engine is not None:
            lo = mid
            cleanup_engine(engine, tag=f"MaxTokenBisect-TP={tp_size}")
            time.sleep(2)
            print(f"    二分: max_total_tokens={mid} ✅")
        else:
            hi = mid
            time.sleep(2)
            print(f"    二分: max_total_tokens={mid} ❌")

    print(f"  TP={tp_size} 最大 max_total_tokens = {lo}")
    return lo





def test_tp_max_length_comparison(tp_size: int):
    """对比 TP=1 和 TP=N 的最大可推理 prompt 长度。

    分两步搜索：
      1. 搜索 engine 创建阶段的 max_total_tokens 上限（try/except 可捕获）
      2. 用 subprocess 隔离搜索实际可推理的最大 prompt 长度
         （推理 OOM 会杀死子进程，主进程通过 returncode 检测）

    流程：
      1. 搜索 TP=1 的 max_total_tokens → subprocess 搜索最大推理长度
      2. 搜索 TP=N 的 max_total_tokens → subprocess 搜索最大推理长度
      3. 汇总对比
    """
    print_separator(f"TP=1 vs TP={tp_size} 最大推理长度对比")

    results = {}  # tp -> {max_total_tokens, max_inference_length}

    for tp in [1, tp_size]:
        print_separator(f"MaxLength Test: TP={tp}")

        # Step 1: 找到最大 max_total_tokens
        max_tokens = _find_max_engine_tokens(tp_size=tp)
        if max_tokens is None:
            print(f"  ❌ TP={tp} 无法创建 engine，跳过")
            results[tp] = {"max_total_tokens": None, "max_inference_length": None}
            continue

        # Step 2: 用 subprocess 搜索最大可推理 prompt 长度
        max_infer_len = _find_max_inference_length(
            tp_size=tp,
            max_total_tokens=max_tokens,
        )

        results[tp] = {
            "max_total_tokens": max_tokens,
            "max_inference_length": max_infer_len,
        }

    # ---- 汇总对比 ----
    print_separator(f"TP=1 vs TP={tp_size} 最大推理长度汇总")

    for tp in [1, tp_size]:
        r = results.get(tp, {})
        mt = r.get("max_total_tokens")
        ml = r.get("max_inference_length")
        print(f"  TP={tp}:")
        print(f"    max_total_tokens (KV cache 池): {mt if mt is not None else 'N/A'}")
        print(f"    最大可推理 prompt 长度 (实测):  {f'~{ml} tokens' if ml is not None and ml > 0 else 'N/A'}")
        if mt is not None and ml is not None and ml > 0:
            util = ml / mt * 100
            print(f"    KV cache 利用率:                {util:.1f}%")
        print()

    # 计算倍率
    r1 = results.get(1, {})
    rN = results.get(tp_size, {})
    ml1 = r1.get("max_inference_length")
    mlN = rN.get("max_inference_length")
    mt1 = r1.get("max_total_tokens")
    mtN = rN.get("max_total_tokens")

    if mt1 and mtN:
        print(f"  📊 max_total_tokens 倍率:   TP={tp_size} / TP=1 = {mtN / mt1:.2f}x")
    if ml1 and mlN and ml1 > 0:
        print(f"  📊 最大推理长度倍率:        TP={tp_size} / TP=1 = {mlN / ml1:.2f}x")

    print()
    if ml1 and mlN and mlN > ml1:
        print(f"  🎉 TP={tp_size} 相比 TP=1 可处理约 {mlN / ml1:.1f}x 更长的 prompt!")
    elif ml1 and mlN:
        print(f"  ⚠️  TP={tp_size} 未显著提升最大推理长度 (TP=1: ~{ml1}, TP={tp_size}: ~{mlN})")
    else:
        print(f"  ⚠️  部分测试失败，无法完成对比")

    print()
    print(f"  📝 说明: 使用 subprocess 隔离推理测试，避免 OOM 杀死主进程。")
    print(f"  📝 max_total_tokens 是 KV cache 池大小，最大推理长度受中间激活内存限制，")
    print(f"     通常远小于 max_total_tokens。")


def test_single_length(tp_size: int, prompt_tokens: int, max_new_tokens: int = 1):
    """测试指定 TP 和 prompt 长度能否正常推理。

    构造指定长度的随机输入序列，创建 engine，推理，报告结果。
    这是最简单直接的方式：传入不同长度参数，手动二分找到 OOM 边界。

    用法:
      python test_sglang_hsa_tp.py --test-length 65536
      python test_sglang_hsa_tp.py --test-length 131072
    """
    print_separator(f"单次长度测试: TP={tp_size}, prompt_tokens={prompt_tokens}")

    # Step 1: 加载 tokenizer，构造精确长度的 prompt
    print(f"  正在构造 {prompt_tokens} tokens 的随机输入...")
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        print("  ❌ 无法加载 tokenizer")
        return False

    # 直接用 token ids 构造精确长度的输入，避免 decode→encode 的损耗
    base = "The quick brown fox jumps over the lazy dog. "
    base_ids = tokenizer.encode(base, add_special_tokens=False)
    if len(base_ids) == 0:
        base_ids = [tokenizer.eos_token_id or 1]
    # 重复拼接 base_ids 直到超过目标长度，然后截断
    full_ids = []
    while len(full_ids) < prompt_tokens:
        full_ids.extend(base_ids)
    full_ids = full_ids[:prompt_tokens]
    # 直接用 token ids 构造 prompt（不经过 decode→encode 循环）
    prompt = tokenizer.decode(full_ids, skip_special_tokens=True)
    actual_tokens = len(tokenizer.encode(prompt))
    # 如果 decode→encode 有损耗，补偿差值
    if actual_tokens < prompt_tokens:
        shortfall = prompt_tokens - actual_tokens
        extra_ids = base_ids * ((shortfall // len(base_ids)) + 2)
        extra_text = " " + tokenizer.decode(extra_ids[:shortfall + len(base_ids)], skip_special_tokens=True)
        prompt = prompt + extra_text
        actual_tokens = len(tokenizer.encode(prompt))
    print(f"  实际 prompt 长度: {actual_tokens} tokens (目标: {prompt_tokens})")

    # Step 2: 创建 engine
    # 不传 max_total_tokens，让 SGLang 自动 profile 出 GPU 能容纳的最大 KV cache 大小
    # 这样可以探索单卡/多卡推理的极限长度，不受固定值限制
    # context_length 设为实际 prompt 长度 + 余量，既能绕过 131072 的验证限制，
    # 又不会因为设得过大（如 1M）导致 KV cache 分配时 OOM
    engine = _create_engine_with_max_tokens(
        tp_size=tp_size, max_total_tokens=None,
        context_length=actual_tokens + 1024,
        tag=f"TestLen-TP={tp_size}"
    )
    if engine is None:
        print(f"  ❌ Engine 创建失败")
        print(f"  提示: KV cache 池分配失败，可能需要降低 MEM_FRACTION_STATIC 或减小 prompt 长度")
        return False

    # 从 scheduler_info 获取实际的 max_req_input_len
    actual_max_input = engine.scheduler_info.get("max_req_input_len", None)
    if actual_max_input is not None and actual_tokens >= actual_max_input:
        print(f"  ⚠️  prompt ({actual_tokens} tokens) >= 实际 max_req_input_len ({actual_max_input})")
        print(f"  提示: GPU 内存不足以容纳该长度的 KV cache，将继续尝试推理，观察 SGLang 的实际行为")

    # Step 4: Warmup
    if not warmup(engine, PAGE_SIZE, tag=f"TestLen-TP={tp_size}"):
        print("  ❌ Warmup 失败")
        cleanup_engine(engine, tag=f"TestLen-TP={tp_size}")
        return False

    # Step 5: 推理
    print(f"  正在推理 (prompt={actual_tokens} tokens, max_new_tokens={max_new_tokens})...")
    t0 = time.time()
    try:
        result = engine.generate(
            prompt=prompt,
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
        )
        elapsed = time.time() - t0

        meta = result.get("meta_info", {})
        output_ids = result.get("output_ids", [])
        text = result.get("text", "")
        real_prompt_tokens = meta.get("prompt_tokens", -1)

        text_preview = text[:100].replace("\n", "\\n")
        print(f"  ✅ 推理成功!")
        print(f"    prompt_tokens (实际): {real_prompt_tokens}")
        print(f"    output_tokens: {len(output_ids)}")
        print(f"    耗时: {elapsed:.3f}s")
        print(f"    输出: \"{text_preview}...\"")

        cleanup_engine(engine, tag=f"TestLen-TP={tp_size}")
        return True

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ 推理失败! (耗时 {elapsed:.3f}s)")
        print(f"    错误: {e}")
        traceback.print_exc()
        cleanup_engine(engine, tag=f"TestLen-TP={tp_size}")
        return False


def cleanup_engine(engine, tag: str = ""):
    """清理 Engine 资源"""
    import torch
    label = f"[Cleanup{' ' + tag if tag else ''}]"
    print(f"{label} 正在关闭 Engine...")
    try:
        engine.shutdown()
    except Exception as e:
        print(f"{label} shutdown 异常: {e}")
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{label} ✅ 资源已释放")


def main():
    import argparse
    import torch

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="SGLang HSA TP 推理测试")
    parser.add_argument("--test-length", type=int, default=None,
                        help="单次测试模式：指定 prompt 长度 (tokens)，测试能否正常推理")
    parser.add_argument("--max-new-tokens", type=int, default=1,
                        help="单次测试模式下生成的 token 数 (默认 1，只测 prefill)")
    args = parser.parse_args()

    # 也支持环境变量
    test_length = args.test_length or (int(os.environ["TEST_LENGTH"]) if os.environ.get("TEST_LENGTH") else None)

    # ============================================================
    # 单次长度测试模式：--test-length <N>
    # ============================================================
    if test_length is not None:
        print_separator(f"单次长度测试模式: {test_length} tokens, TP={SGLANG_TP}")
        print(f"  模型: {CHECKPOINT_PATH}")
        print(f"  TP Size: {SGLANG_TP}")
        print(f"  Max Total Tokens: {MAX_TOTAL_TOKENS}")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

        gpu_count = torch.cuda.device_count()
        print(f"  可用 GPU 数量: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

        success = test_single_length(
            tp_size=SGLANG_TP,
            prompt_tokens=test_length,
            max_new_tokens=args.max_new_tokens,
        )
        sys.exit(0 if success else 1)

    print_separator("SGLang HSA: TP 多卡推理测试")
    print(f"  模型: {CHECKPOINT_PATH}")
    print(f"  TP Size: {SGLANG_TP}")
    print(f"  Page Size: {PAGE_SIZE}")
    print(f"  Max New Tokens: {MAX_NEW_TOKENS}")
    print(f"  Max Total Tokens: {MAX_TOTAL_TOKENS}")
    print(f"  Chunked Prefill Size: {CHUNKED_PREFILL_SIZE}")
    print(f"  Mem Fraction Static: {MEM_FRACTION_STATIC}")
    print(f"  对比 TP=1: {'是' if COMPARE_WITH_TP1 else '否'}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # 检查可用 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"  可用 GPU 数量: {gpu_count}")
    if gpu_count < SGLANG_TP:
        print(f"\n  ❌ 错误: 需要 {SGLANG_TP} 张 GPU，但只有 {gpu_count} 张可用!")
        print(f"  请设置 CUDA_VISIBLE_DEVICES 指定足够的 GPU，例如:")
        print(f"    CUDA_VISIBLE_DEVICES=0,1 SGLANG_TP=2 python ...")
        sys.exit(1)

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

    # ============================================================
    # Phase 1: TP=1 基线（强制开启，用于对比）
    # ============================================================
    tp1_results = None
    print_separator("Phase 1: TP=1 基线推理")

    engine_tp1 = create_engine(tp_size=1, tag="TP=1")
    if engine_tp1 is None:
        print("  ❌ TP=1 Engine 创建失败，无法对比")
        sys.exit(1)
    else:
        if warmup(engine_tp1, PAGE_SIZE, tag="TP=1"):
            tp1_results = test_various_lengths(engine_tp1, MAX_NEW_TOKENS, tag="(TP=1)")
        cleanup_engine(engine_tp1, tag="TP=1")
        # 等待 GPU 资源完全释放
        time.sleep(3)

    if tp1_results is None:
        print("  ❌ TP=1 推理失败，无法对比")
        sys.exit(1)

    # ============================================================
    # Phase 2: TP=N 多卡推理（仅单条推理）
    # ============================================================
    print_separator(f"Phase 2: TP={SGLANG_TP} 多卡推理")

    # Step 1: 创建 Engine
    print_separator(f"Step 1: 创建 SGLang Engine (TP={SGLANG_TP})")
    engine = create_engine(tp_size=SGLANG_TP, tag=f"TP={SGLANG_TP}")
    if engine is None:
        print(f"\n  ❌ TP={SGLANG_TP} Engine 创建失败!")
        print("  可能的原因:")
        print("    1. GPU 显存不足 → 尝试降低 mem_fraction_static 或 max_total_tokens")
        print("    2. HSA backend 不支持当前 TP 配置 → 检查模型 head 数是否能被 TP 整除")
        print("    3. NCCL 通信问题 → 检查 GPU 拓扑和 NCCL 版本")
        sys.exit(1)

    # Step 2: Warmup
    print_separator("Step 2: Warmup")
    if not warmup(engine, PAGE_SIZE, tag=f"TP={SGLANG_TP}"):
        print("  ❌ Warmup 失败!")
        cleanup_engine(engine, tag=f"TP={SGLANG_TP}")
        sys.exit(1)

    # Step 3: 不同长度 prompt 测试
    # tpN_results = test_various_lengths(engine, MAX_NEW_TOKENS, tag=f"(TP={SGLANG_TP})")

    # # Step 4: Batch 推理测试（暂时跳过，只测单条一致性）
    # batch_res = test_batch(engine, MAX_NEW_TOKENS, tag=f"(TP={SGLANG_TP})")

    # # Step 5: 顺序 vs Batch 一致性（暂时跳过）
    # seq_batch_match, _, _ = test_sequential_vs_batch_consistency(
    #     engine, MAX_NEW_TOKENS, tag=f"(TP={SGLANG_TP})"
    # )

    # ============================================================
    # Phase 3: TP=1 vs TP=N 对比
    # ============================================================
    # tp_match = compare_tp_results(tp1_results, tpN_results, SGLANG_TP)

    # 清理
    print_separator("清理资源")
    cleanup_engine(engine, tag=f"TP={SGLANG_TP}")

    # ============================================================
    # Phase 4: TP=1 vs TP=N 推理速度对比
    # ============================================================
    # time.sleep(3)  # 等待 GPU 资源完全释放
    # test_tp_speed_comparison(tp_size=SGLANG_TP, max_new_tokens=50)

    # ============================================================
    # Phase 5: TP=1 vs TP=N 最大推理长度对比
    # ============================================================
    time.sleep(3)  # 等待 GPU 资源完全释放
    test_tp_max_length_comparison(tp_size=SGLANG_TP)

    print("[Done] 所有测试完成。")


if __name__ == "__main__":
    main()
