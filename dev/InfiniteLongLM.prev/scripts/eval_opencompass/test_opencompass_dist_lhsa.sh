#!/usr/bin/env bash
# ============================================================
# test_opencompass_dist.sh
# 在多张 GPU 上并行测试不同数据集（每个数据集独占一张卡，--debug 模式）
# 用法: bash scripts/eval_opencompass/test_opencompass_dist.sh
# ============================================================

set -uo pipefail

# ── 环境变量 ──
export PYTHONPATH=/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/opencompass:$PYTHONPATH
# export OPENCOMPASS_CONFIG_DIR=/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/opencompass/opencompass/configs
# export BBH_LIB_PROMPT_DIR=/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/opencompass/opencompass/configs/datasets/bbh/lib_prompt

# ── LHSA checkpoint 基础路径 ──
LHSA_CKPT_BASE="/apdcephfs_tj5/share_300719894/user/qqzxywei/wxy/checkpoints/hsa-full-pope-halfdim-8KA1K-wonoise-1B-300B/checkpoints"

# ── 步数列表：10000 到 80000，间隔 10000 ──
STEP_LIST=(10000)
# STEP_LIST=(10000)

# ── 输出目录 ──
WORK_DIR="/apdcephfs_fsgm/share_303843174/user/qqzxywei/wxy/opencompass_outputs"

# ── 数据集列表（与 GPU 一一对应） ──
DATASET_LIST=(
    mmlu_ppl_ac766d
    gpqa_few_shot_ppl_4b5a83
    hellaswag_10shot_ppl_59c85e
    ARC_c_few_shot_ppl
    SuperGLUE_BoolQ_few_shot_ppl
    race_few_shot_ppl
)

# ── GPU 分配（按顺序分配给每个数据集） ──
GPU_LIST=(2 3 4 5 6 7)

# ── 逐个步数评测 ──
for step in "${STEP_LIST[@]}"; do
    HF_PATH="${LHSA_CKPT_BASE}/global_step_${step}/hf_ckpt"
    if [ ! -d "${HF_PATH}" ]; then
        echo "[跳过] step ${step}: ${HF_PATH} 不存在"
        continue
    fi

    RUN_TAG="lhsa_step${step}_$(date +%Y%m%d_%H%M%S)"
    LOG_DIR="${WORK_DIR}/dist_logs_${RUN_TAG}"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "============================================"
    echo " 并行评测启动 - LHSA step ${step}"
    echo " 模型: ${HF_PATH}"
    echo " 数据集数量: ${#DATASET_LIST[@]}"
    echo " 日志目录: ${LOG_DIR}"
    echo "============================================"

    # ── 并行启动所有数据集评测 ──
    pids=()
    for i in "${!DATASET_LIST[@]}"; do
        dataset="${DATASET_LIST[$i]}"
        gpu_id="${GPU_LIST[$i]}"
        log_file="${LOG_DIR}/${dataset}.log"

        echo "[$(date '+%F %T')] 启动 dataset=${dataset} on GPU ${gpu_id}, 日志: ${log_file}"
        pkill -f burner || true
        CUDA_VISIBLE_DEVICES="$gpu_id" python eval/eval_opencompass.py \
            --datasets "$dataset" \
            --hf-type base \
            --hf-path "$HF_PATH" \
            --model-kwargs torch_dtype=torch.bfloat16 attn_implementation=flash_attention_3 auto_insert_lmk=True \
            -w "${WORK_DIR}/${dataset}_${RUN_TAG}" \
            --debug \
            > "$log_file" 2>&1 &

        pids+=($!)
    done

    echo ""
    echo "[$(date '+%F %T')] step ${step}: 所有 ${#DATASET_LIST[@]} 个评测任务已启动，等待完成..."
    echo ""

    # ── 等待所有任务完成并收集状态 ──
    failed=0
    for i in "${!pids[@]}"; do
        pid="${pids[$i]}"
        dataset="${DATASET_LIST[$i]}"
        gpu_id="${GPU_LIST[$i]}"

        if wait "$pid"; then
            echo "[$(date '+%F %T')] ✅ dataset=${dataset} (GPU ${gpu_id}) 完成"
        else
            echo "[$(date '+%F %T')] ❌ dataset=${dataset} (GPU ${gpu_id}) 失败，查看日志: ${LOG_DIR}/${dataset}.log"
            failed=1
        fi
    done

    echo ""
    echo "============================================"
    echo " step ${step} 所有任务执行完毕"
    echo " 日志目录: ${LOG_DIR}"
    echo "============================================"

    # ── 汇总各数据集结果（从 summary csv 中计算平均分） ──
    STEP_SUMMARY_FILE="${LOG_DIR}/step_summary.csv"
    python - "${WORK_DIR}" "${RUN_TAG}" "${STEP_SUMMARY_FILE}" "${DATASET_LIST[@]}" <<'PYEOF'
import csv, os, sys, glob

work_dir = sys.argv[1]
run_tag = sys.argv[2]
output_file = sys.argv[3]
dataset_list = sys.argv[4:]

# 数据集显示名映射
DISPLAY_NAMES = {
    "mmlu_ppl_ac766d":              "MMLU(5-shot)",
    "gpqa_few_shot_ppl_4b5a83":     "GPQA(5-shot)",
    "hellaswag_10shot_ppl_59c85e":  "HellaSwag(10-shot)",
    "ARC_c_few_shot_ppl":           "ARC-c(25-shot)",
    "SuperGLUE_BoolQ_few_shot_ppl": "BoolQ(5-shot)",
    "race_few_shot_ppl":            "Race(3-shot)",
}

results = {}
for dataset in dataset_list:
    # 查找 summary csv 文件
    pattern = os.path.join(work_dir, f"{dataset}_{run_tag}", "*", "summary", "summary_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        results[dataset] = None
        continue
    csv_file = csv_files[-1]  # 取最新的
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            # 最后一列是分数列
            score_col = len(header) - 1
            scores = []
            for row in reader:
                if not row or not row[score_col].strip():
                    continue
                try:
                    scores.append(float(row[score_col].strip()))
                except ValueError:
                    continue
            if scores:
                results[dataset] = sum(scores) / len(scores)
            else:
                results[dataset] = None
    except Exception as e:
        print(f"  警告: 解析 {csv_file} 失败: {e}")
        results[dataset] = None

# 输出单步汇总（& 分隔格式）
header_names = []
header_scores = []
for dataset in dataset_list:
    display = DISPLAY_NAMES.get(dataset, dataset)
    score = results.get(dataset)
    score_str = f"{score:.2f}" if score is not None else "-"
    header_names.append(display)
    header_scores.append(score_str)

# 计算总平均
valid_scores = [v for v in results.values() if v is not None]
avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
avg_str = f"{avg:.2f}"

# 输出 & 分隔的一行结果
print("&".join(header_names + ["AVG"]))
print("&".join(header_scores + [avg_str]))

# 写入 csv 供最终汇总使用
with open(output_file, "w", encoding="utf-8") as f:
    f.write(",".join(["dataset"] + header_names + ["AVG"]) + "\n")
    f.write(",".join(["score"] + header_scores + [avg_str]) + "\n")
PYEOF

    if [ "$failed" -ne 0 ]; then
        echo ""
        echo "⚠️  step ${step} 部分任务失败，请检查日志目录: ${LOG_DIR}"
    fi
done

# ── 全部步数评测完成后，生成总汇总表 ──
echo ""
echo "============================================"
echo " 全部步数评测完成！"
echo "============================================"

# 收集所有步数的汇总 csv，生成最终表格
FINAL_SUMMARY="${WORK_DIR}/lhsa_all_steps_summary.log"
python - "${WORK_DIR}" "${FINAL_SUMMARY}" "${STEP_LIST[@]}" <<'PYEOF'
import csv, os, sys, glob

work_dir = sys.argv[1]
output_file = sys.argv[2]
step_list = sys.argv[3:]

# 查找所有步数的 step_summary.csv
all_data = {}  # step -> {header, scores}
header_line = None

for step in step_list:
    # 查找对应步数的 dist_logs 目录
    pattern = os.path.join(work_dir, f"dist_logs_lhsa_step{step}_*", "step_summary.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        continue
    csv_file = csv_files[-1]  # 取最新的
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            scores = next(reader)
            if header_line is None:
                header_line = header[1:]  # 去掉第一列 "dataset"
            all_data[step] = scores[1:]  # 去掉第一列 "score"
    except Exception:
        continue

if not all_data or header_line is None:
    print("未找到任何步数的汇总数据")
    sys.exit(0)

# 输出 & 分隔格式
lines = []
# 表头行
lines.append("Step&" + "&".join(header_line))

for step in step_list:
    if step not in all_data:
        continue
    lines.append(step + "&" + "&".join(all_data[step]))

output = "\n".join(lines)
print(output)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(output + "\n")
print(f"\n汇总表已保存到: {output_file}")
PYEOF
