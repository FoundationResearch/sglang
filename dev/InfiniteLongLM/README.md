# InfiniteLongLM

Efficient long-context language model training with **Landmark Hierarchical Sparse Attention (LHSA)**.

LHSA combines sliding window attention (SWA) for local context with hierarchical sparse attention (HSA) for global context retrieval, enabling efficient training on ultra-long sequences (128K+).

## Project Structure

```
InfiniteLongLM/
├── models/                  # Model definitions
│   └── FlashHSA/
│       ├── modeling_olmo_lhsa.py    # OLMo3 + LHSA (main model)
│       ├── modeling_qwen_lhsa.py    # Qwen3 + LHSA
│       ├── lhsa_layer.py            # Core LHSA attention layer
│       ├── configuration_hsa.py     # Model config
│       └── ...
├── ops/                     # Custom Triton/TileLang kernels
│   ├── hsa_fwd_bwd_*.py     # HSA forward/backward kernels
│   └── topk_*.py            # TopK sparse selection kernels
├── data/                    # Data loading & processing
├── tasks/                   # Training task entry points
│   ├── pretrain_with_ruler.py       # Pretrain + RULER synthetic data
│   ├── sft_with_lmk.py             # SFT with landmark tokens
│   └── ...
├── configs/                 # Model & training configs (JSON/YAML)
├── scripts/                 # Shell scripts
│   ├── preprocess/          # Data preprocessing
│   ├── pretrain/            # Pretrain scripts
│   │   └── fair_comparison/ # Controlled experiment scripts
│   ├── sft/                 # SFT scripts
│   ├── cpt/                 # Continual pre-training
│   ├── convert_params/      # Model conversion utilities
│   └── eval/                # Evaluation scripts
├── eval/                    # Evaluation code (PPL, RULER)
├── train.sh                 # Single-node launcher
└── train_dist.sh            # Multi-node distributed launcher
```

## Core: LHSA Attention Layer

**Key file**: [`models/FlashHSA/lhsa_layer.py`](models/FlashHSA/lhsa_layer.py)

LHSA splits attention heads into two groups:

- **SWA heads**: Sliding window attention for local context (default window=512)
- **HSA heads**: Hierarchical sparse attention for global context via landmark-based top-k chunk retrieval (default chunk_size=64, topk=32)

Key features:
- **Landmark tokens**: Inserted at every `chunk_size-1` positions as block-level keys for retrieval
- **TopK sparse selection**: Custom Triton kernels for online top-k chunk selection
- **Chunk dropout**: Training-time dropout on block-level attention for regularization
- **Unified retrieval**: Optional shared retrieval mechanism across heads

## Supported Models

| Registry Name | Base Architecture | Description |
|---|---|---|
| `olmo_lhsa` | OLMo3 | OLMo3 + LHSA ([modeling_olmo_lhsa.py](models/FlashHSA/modeling_olmo_lhsa.py)) |
| `qwen_lhsa` | Qwen3 | Qwen3 + LHSA ([modeling_qwen_lhsa.py](models/FlashHSA/modeling_qwen_lhsa.py)) |

## Quick Start

### Data Preprocessing

Tokenize raw corpora into numpy format:

```bash
# Tokenize Dolma3 dataset
bash scripts/preprocess/build_olmo_numpy.sh

# Weighted sampling (e.g., 500B tokens)
bash scripts/preprocess/build_olmo_pretrain_sampled_500B.sh
```

### Pre-Training

Representative example — **LSA Interleave Unified** ([`scripts/pretrain/fair_comparison/pretrain_lsa_interleave_unified.sh`](scripts/pretrain/fair_comparison/pretrain_lsa_interleave_unified.sh)):

```bash
export MODEL_CONFIG="configs/flash_hsa/config_lsa_unified.json"
export CORPUS_PATH="/path/to/tokenized/data"
export MAX_SEQ_LEN=8192
export WANDB_NAME="lsa-interleave-unified"
export OUTPUT_DIR="/path/to/checkpoints"

bash scripts/pretrain/pretrain_ruler_task_5per.sh
```

All pretrain experiment scripts are in [`scripts/pretrain/fair_comparison/`](scripts/pretrain/fair_comparison/), covering various HSA configurations (window sizes, topk values, interleave ratios, etc.).

### SFT (Supervised Fine-Tuning)

```bash
bash scripts/sft/sft_lmk.sh
```

### Continual Pre-Training (CPT)

Convert a base model to LHSA structure and continue training:

```bash
# Convert OLMo3 base model weights to LHSA
bash scripts/convert_params/convert_olmo3_to_lhsa.sh

# Run CPT
bash scripts/cpt/CPT.sh
```

### Evaluation

```bash
# Perplexity evaluation
python eval/eval_ppl.py

# RULER benchmark (long-context understanding)
python eval/eval_ruler.py
```

## Training Infrastructure

- **Framework**: Built on [veomni](https://github.com/volcengine/veomni) (supports FSDP2, Ulysses sequence parallelism)
- **Precision**: BF16 mixed precision
- **Parallelism**: FSDP2 data parallel + optional sequence parallel
- **Hardware**: GPU (CUDA/NCCL) and NPU (Ascend) supported
- **Logging**: WandB integration

## Key Configuration

Example model config (`configs/flash_hsa/config_lsa_unified.json`):

```json
{
  "model_type": "olmo_lhsa",
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "sliding_window": 512,
  "chunk_size": 64,
  "hsa_topk": 32,
  "full_attn_interleave": 4,
  "unified_retrieval": true,
  "enable_lmk_q_proj": true
}
```

Key hyperparameters:
- `sliding_window`: Local SWA window size
- `chunk_size`: Landmark token interval for HSA
- `hsa_topk`: Number of top-k chunks to retrieve
- `full_attn_interleave`: Ratio of full attention layers interleaved with HSA layers
- `unified_retrieval`: Whether to use shared retrieval across HSA heads
