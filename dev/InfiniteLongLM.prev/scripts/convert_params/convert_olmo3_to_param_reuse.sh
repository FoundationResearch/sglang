export MODEL_CONFIG="configs/olmo3_7B/olmo3_param_reuse.json"
export BASE_MODEL_DIR="/apdcephfs_sh8/share_300719895/shared/models/OLMO3/OLMo-stage1-step999000"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/olmo3_param_reuse_64gpu/pytorch_model.bin"
export LOG_PATH="./olmo3_to_param_reuse.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python3 utils/convert_basemodel_to_hsa.py \
    --target_config $MODEL_CONFIG \
    --base_path $BASE_MODEL_DIR \
    --output_path $OUTPUT_DIR \
    --log_path $LOG_PATH
