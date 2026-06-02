export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_half_pope_swa512_lmkmask_headqk.json"
export BASE_MODEL_DIR="/apdcephfs_tj5/share_300719894/shared/models/OLMO3/OLMo-stage1-step999000"
export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/guhao/checkpoints/lhsa-olmo3-half-pope-headqk/pytorch_model.bin"
export LOG_PATH="./olmo3_to_lhsa.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python utils/convert_basemodel_to_flashhsa.py \
    --target_config $MODEL_CONFIG \
    --base_path $BASE_MODEL_DIR \
    --output_path $OUTPUT_DIR \
    --log_path $LOG_PATH