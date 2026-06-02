# export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_8KA2K_w_lmk_q_proj.json"
# export BASE_MODEL_DIR="/apdcephfs_tj5/share_300719894/user/guhao/Models/OLMo-stage1-step999000-base"
# export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-7B-8KA2K-w-lmk-q-proj/pytorch_model.bin"
# export LOG_PATH="./olmo3_to_lhsa_8KA2K_w_lmk_q_proj.log"

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# cd "$PROJECT_ROOT"
# export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# python3 utils/convert_basemodel_to_hsa.py \
#     --target_config $MODEL_CONFIG \
#     --base_path $BASE_MODEL_DIR \
#     --output_path $OUTPUT_DIR \
#     --log_path $LOG_PATH


# export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_8KA2K_wo_lmk_q_proj.json"
# export BASE_MODEL_DIR="/apdcephfs_tj5/share_300719894/user/guhao/Models/OLMo-stage1-step999000-base"
# export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-7B-8KA2K-wo-lmk-q-proj/pytorch_model.bin"
# export LOG_PATH="./olmo3_to_lhsa_8KA2K_wo_lmk_q_proj.log"

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# cd "$PROJECT_ROOT"
# export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# python3 utils/convert_basemodel_to_hsa.py \
#     --target_config $MODEL_CONFIG \
#     --base_path $BASE_MODEL_DIR \
#     --output_path $OUTPUT_DIR \
#     --log_path $LOG_PATH

export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_8KA2K_w_lmk_q_proj_4k_swa_mask_lmk.json"
export BASE_MODEL_DIR="/apdcephfs_tj5/share_300719894/user/guhao/Models/OLMo-stage1-step999000-base"
export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-7B-8KA2K-w-lmk-q-proj-4k-swa-mask_lmk/pytorch_model.bin"
export LOG_PATH="./olmo3_to_lhsa_8KA2K_w_lmk_q_proj_4k_swa_mask_lmk.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python3 utils/convert_basemodel_to_hsa.py \
    --target_config $MODEL_CONFIG \
    --base_path $BASE_MODEL_DIR \
    --output_path $OUTPUT_DIR \
    --log_path $LOG_PATH

# export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_8KA2K_w_lmk_q_proj_4k_swa.json"
# export BASE_MODEL_DIR="/apdcephfs_tj5/share_300719894/user/guhao/Models/OLMo-stage1-step999000-base"
# export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-7B-8KA2K-w-lmk-q-proj-4k-swa/pytorch_model.bin"
# export LOG_PATH="./olmo3_to_lhsa_8KA2K_w_lmk_q_proj_4k_swa.log"

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# cd "$PROJECT_ROOT"
# export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# python3 utils/convert_basemodel_to_hsa.py \
#     --target_config $MODEL_CONFIG \
#     --base_path $BASE_MODEL_DIR \
#     --output_path $OUTPUT_DIR \
#     --log_path $LOG_PATH

# export MODEL_CONFIG="configs/olmo3_7B/olmo3_param_reuse_512swa.json"
# export BASE_MODEL_DIR="/apdcephfs_tj5/share_300719894/user/guhao/Models/OLMo-stage1-step999000-base"
# export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/guhao/checkpoints/olmo3_param_reuse_512swa/pytorch_model.bin"
# export LOG_PATH="./olmo3_to_param_reuse_512swa.log"

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# cd "$PROJECT_ROOT"
# export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# python3 utils/convert_basemodel_to_hsa.py \
#     --target_config $MODEL_CONFIG \
#     --base_path $BASE_MODEL_DIR \
#     --output_path $OUTPUT_DIR \
#     --log_path $LOG_PATH

export MODEL_CONFIG="configs/olmo3_7B/olmo3_lhsa_half_pope.json"
export BASE_MODEL_DIR="/apdcephfs_tj5/share_300719894/user/guhao/Models/OLMo-stage1-step999000-base"
export OUTPUT_DIR="/apdcephfs_tj5/share_300719894/user/guhao/checkpoints/lhsa-olmo3-half-pope/pytorch_model.bin"
export LOG_PATH="./olmo3_to_lhsa_half_pope.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python3 utils/convert_basemodel_to_hsa.py \
    --target_config $MODEL_CONFIG \
    --base_path $BASE_MODEL_DIR \
    --output_path $OUTPUT_DIR \
    --log_path $LOG_PATH