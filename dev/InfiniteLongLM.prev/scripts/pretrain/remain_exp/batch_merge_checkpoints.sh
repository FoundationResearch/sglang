#!/bin/bash
# 批量 merge remain_exp 目录下所有 pretrain 脚本对应的 checkpoint
# 对每个实验的 global_step_10000, global_step_20000, global_step_30000 进行等权平均 merge
#
# 用法:
#   cd /apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy && bash batch_merge_checkpoints.sh
#   或者在 InfiniteLongLM 目录下:
#   cd <InfiniteLongLM_ROOT> && bash scripts/pretrain/remain_exp/batch_merge_checkpoints.sh

set -e

# ========== 配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# InfiniteLongLM 项目根目录（脚本在 scripts/pretrain/remain_exp/ 下，往上3级）
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# merge 脚本路径
MERGE_SCRIPT="${PROJECT_ROOT}/merge_checkpoints.py"

# 要 merge 的 step 列表
STEPS=(10000 20000 30000)

# 是否跳过已存在的 merge 结果（设为 true 则跳过）
SKIP_EXISTING=true

# 日志文件
LOG_FILE="${SCRIPT_DIR}/batch_merge.log"
# ========== 配置结束 ==========

echo "========================================" | tee -a "${LOG_FILE}"
echo "批量 Merge Checkpoints 开始" | tee -a "${LOG_FILE}"
echo "时间: $(date)" | tee -a "${LOG_FILE}"
echo "项目根目录: ${PROJECT_ROOT}" | tee -a "${LOG_FILE}"
echo "Merge 脚本: ${MERGE_SCRIPT}" | tee -a "${LOG_FILE}"
echo "Steps: ${STEPS[*]}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# 统计
TOTAL=0
SUCCESS=0
SKIPPED=0
FAILED=0

# 遍历 remain_exp 目录下所有 .sh 文件（排除自身）
for PRETRAIN_SCRIPT in "${SCRIPT_DIR}"/pretrain_*.sh; do
    [ -f "${PRETRAIN_SCRIPT}" ] || continue

    SCRIPT_NAME="$(basename "${PRETRAIN_SCRIPT}")"
    echo "" | tee -a "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "处理脚本: ${SCRIPT_NAME}" | tee -a "${LOG_FILE}"

    # 从脚本中提取 MODEL_CONFIG 和 OUTPUT_DIR
    MODEL_CONFIG=$(grep -oP 'export MODEL_CONFIG="\K[^"]+' "${PRETRAIN_SCRIPT}" || true)
    OUTPUT_DIR=$(grep -oP 'export OUTPUT_DIR="\K[^"]+' "${PRETRAIN_SCRIPT}" || true)

    if [ -z "${MODEL_CONFIG}" ] || [ -z "${OUTPUT_DIR}" ]; then
        echo "  [跳过] 无法从脚本中提取 MODEL_CONFIG 或 OUTPUT_DIR" | tee -a "${LOG_FILE}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "  MODEL_CONFIG: ${MODEL_CONFIG}" | tee -a "${LOG_FILE}"
    echo "  OUTPUT_DIR:   ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

    # 构建 config 的绝对路径（如果是相对路径，则相对于项目根目录）
    if [[ "${MODEL_CONFIG}" != /* ]]; then
        CONFIG_PATH="${PROJECT_ROOT}/${MODEL_CONFIG}"
    else
        CONFIG_PATH="${MODEL_CONFIG}"
    fi

    # 检查 config 文件是否存在
    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "  [跳过] 配置文件不存在: ${CONFIG_PATH}" | tee -a "${LOG_FILE}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # 检查所有 step 的 checkpoint 是否存在
    CKPT_PATHS=()
    ALL_EXIST=true
    for STEP in "${STEPS[@]}"; do
        CKPT_PATH="${OUTPUT_DIR}/checkpoints/global_step_${STEP}"
        if [ -d "${CKPT_PATH}" ]; then
            CKPT_PATHS+=("${CKPT_PATH}")
        else
            echo "  [警告] Checkpoint 不存在: ${CKPT_PATH}" | tee -a "${LOG_FILE}"
            ALL_EXIST=false
        fi
    done

    if [ "${ALL_EXIST}" = false ]; then
        echo "  [跳过] 部分 checkpoint 不存在，跳过此实验" | tee -a "${LOG_FILE}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # 输出目录
    MERGE_OUTPUT_DIR="${OUTPUT_DIR}/checkpoints/merged_checkpoints"

    # 检查是否已存在
    if [ "${SKIP_EXISTING}" = true ] && [ -f "${MERGE_OUTPUT_DIR}/model.safetensors" ]; then
        echo "  [跳过] 已存在 merge 结果: ${MERGE_OUTPUT_DIR}/model.safetensors" | tee -a "${LOG_FILE}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    TOTAL=$((TOTAL + 1))
    echo "  开始 merge ${#CKPT_PATHS[@]} 个 checkpoint..." | tee -a "${LOG_FILE}"
    echo "  输出目录: ${MERGE_OUTPUT_DIR}" | tee -a "${LOG_FILE}"

    # 自动使用最大 step 的 hf_ckpt 目录作为属性文件来源
    MAX_STEP=${STEPS[${#STEPS[@]}-1]}
    HF_CKPT_DIR="${OUTPUT_DIR}/checkpoints/global_step_${MAX_STEP}/hf_ckpt"

    # 构建 merge 命令
    MERGE_CMD="python ${MERGE_SCRIPT}"
    for CKPT in "${CKPT_PATHS[@]}"; do
        MERGE_CMD="${MERGE_CMD} \"${CKPT}\""
    done
    MERGE_CMD="${MERGE_CMD} --config_path \"${CONFIG_PATH}\""
    MERGE_CMD="${MERGE_CMD} --output_dir \"${MERGE_OUTPUT_DIR}\""

    if [ -d "${HF_CKPT_DIR}" ]; then
        MERGE_CMD="${MERGE_CMD} --hf_ckpt_dir \"${HF_CKPT_DIR}\""
        echo "  hf_ckpt_dir: ${HF_CKPT_DIR}" | tee -a "${LOG_FILE}"
    else
        echo "  [警告] hf_ckpt 目录不存在: ${HF_CKPT_DIR}，跳过属性文件拷贝" | tee -a "${LOG_FILE}"
    fi

    echo "  命令: ${MERGE_CMD}" | tee -a "${LOG_FILE}"

    # 执行 merge
    if eval "${MERGE_CMD}" 2>&1 | tee -a "${LOG_FILE}"; then
        echo "  [成功] Merge 完成: ${MERGE_OUTPUT_DIR}" | tee -a "${LOG_FILE}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  [失败] Merge 失败: ${SCRIPT_NAME}" | tee -a "${LOG_FILE}"
        FAILED=$((FAILED + 1))
    fi
done

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "批量 Merge 完成" | tee -a "${LOG_FILE}"
echo "时间: $(date)" | tee -a "${LOG_FILE}"
echo "总计执行: ${TOTAL}, 成功: ${SUCCESS}, 失败: ${FAILED}, 跳过: ${SKIPPED}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
