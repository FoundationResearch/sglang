export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_interleave_8KA512_non_unified.json"
export BASE_MODEL_DIR="/apdcephfs_sh8/share_300719895/shared/models/OLMO3/OLMo-stage1-step999000"
export OUTPUT_DIR="/apdcephfs_sh8/share_300719895/guhao/checkpoints/lhsa-olmo3-7B-interleave-non-unified/pytorch_model.bin"
export LOG_PATH="./olmo3_to_lhsa_interleave_non_unified.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python utils/convert_basemodel_to_flashhsa.py \
    --target_config $MODEL_CONFIG \
    --base_path $BASE_MODEL_DIR \
    --output_path $OUTPUT_DIR \
    --log_path $LOG_PATH
