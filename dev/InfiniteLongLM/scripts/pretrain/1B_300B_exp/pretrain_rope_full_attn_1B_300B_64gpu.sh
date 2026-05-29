export PYTHONPATH=./

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=136
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_SOCKET_IFNAME==bond1
export NCCL_BUFFSIZE=8388608
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET


export MODEL_CONFIG="configs/swan_gpt_tiny/config_rope_full_theta10000_1B.json"
export CORPUS_PATH="/apdcephfs_tj5/share_300719894/shared/data/dolma3_mix-6T-1025-500B/"
export MAX_SEQ_LEN=8192
export WANDB_NAME="rope-full-attn-swangpt-1B-300B-64gpu"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/user/qqzxywei/wxy/checkpoints/rope-full-attn-rope-swangpt-1B-300B-64gpu"
export MAX_STEPS=143000
export SAVE_STEPS=5000
export TRAIN_SIZE=300000000000
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=256


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

bash scripts/pretrain/pretrain_ruler_task_5per_dist_300B.sh
