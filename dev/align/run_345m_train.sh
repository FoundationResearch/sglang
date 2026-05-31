#!/bin/bash
# Wrapper that calls the project's pretrain_with_ruler pipeline against our
# locally-tokenized wikitext-103, on a single hpc-rack-2 node (4x GB200).
# Overrides only the env vars the upstream script reads.
set -e

# wandb + hf creds from nanoVideo/env.sh
source /home/hal-alex/workspace/nanoVideo/env.sh
source /home/hal-alex/miniconda3/etc/profile.d/conda.sh
conda activate alexsg

# Single-node, drop the cluster-only NCCL knobs the upstream script sets.
unset NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL
export NCCL_DEBUG=WARN

# === overrides for the pretrain_ruler_task_5per_345M_dist.sh inner script ===
export MAX_STEPS=10000
export SAVE_STEPS=2000
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=32   # 4 GPUs, micro_bs=4 -> grad_accum=2 per gpu
export DTYPE=ruler_0.05
export TRAIN_SIZE=10000000000

# === outer-script knobs ===
export MODEL_CONFIG="configs/flash_hsa/config_hsa_8KA2K_HoPE_345M_lmk_bias_priorq_wloralmkq_loradim64.json"
export CORPUS_PATH="/home/hal-alex/workspace/hsa_train/wikitext103_tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa_8KA2K_HoPE-priorq-wloralmkq-loradim64"
export OUTPUT_DIR="/home/hal-alex/workspace/hsa_train/ckpt/${WANDB_NAME}"
export GRADIENT_CKPT=false

mkdir -p "$OUTPUT_DIR"

cd /home/hal-alex/workspace/sglang/dev/InfiniteLongLM

# Skip the prefetch loop (it requires network HF access to model_type mapping
# checks; the model class is local). Go straight to the train wrapper.
exec bash scripts/pretrain/pretrain_ruler_task_5per_345M_dist.sh
