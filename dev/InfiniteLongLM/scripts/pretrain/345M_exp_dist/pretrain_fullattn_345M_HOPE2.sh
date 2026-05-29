export PYTHONPATH=./

taiji_client mount --tk noMcEDk1fgEbCsTrxIweYQ --bf TaiJi_HYAide_AILab_Cog_SH_A100H -l tj
taiji_client mount -tk noMcEDk1fgEbCsTrxIweYQ -bf TaiJi_HYAide_AILab_MM_SH_A100H

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=136
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_SOCKET_IFNAME=bond1
export NCCL_BUFFSIZE=8388608
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET


export MODEL_CONFIG="configs/flash_hsa/config_fullattn_HoPE2_345M.json"
export CORPUS_PATH="/apdcephfs_tj5/share_300719894/shared/data/dolma3_mix-6T-1025-partial-tokenized/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="fullattn_HOPE2"
export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/fullattn_HOPE2"
export GRADIENT_CKPT=false
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=128
MAX_PREFETCH_RETRIES=10
for i in $(seq 1 $MAX_PREFETCH_RETRIES); do
    echo "[Prefetch] Attempt $i/$MAX_PREFETCH_RETRIES ..."
    python code_exp/prefetch.py $MODEL_CONFIG
    if [ $? -eq 0 ]; then
        echo "[Prefetch] Success on attempt $i."
        break
    fi
    if [ $i -eq $MAX_PREFETCH_RETRIES ]; then
        echo "[Prefetch] Failed after $MAX_PREFETCH_RETRIES attempts, aborting." >&2
        exit 1
    fi
    echo "[Prefetch] Attempt $i failed, retrying..."
    sleep 2
done
bash scripts/pretrain/pretrain_ruler_task_5per_345M_dist.sh