"""
Score existing prediction files without needing GPU or heavy deps.
Only requires: rouge, jieba, fuzzywuzzy (for relevant tasks).

Usage:
    python scripts/eval/score_existing_preds.py \
        --pred_dir scripts/eval/logs/eval_longbench_v1_20260415_115947/8KA1K-no-noise-warmup1k-step13000/pred \
        [--compare_result scripts/eval/logs/eval_longbench_v1_20260413_220814/8KA1K-no-noise-warmup1k-step9000/result.json]
"""

import os
import sys
import json
import re
import string
import argparse
from collections import Counter

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional deps
try:
    import jieba
except ImportError:
    jieba = None
try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None
try:
    from rouge import Rouge
except ImportError:
    Rouge = None


# ---- Metrics (copied from eval_longbench_v1.py) ----
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    def white_space_fix(text):
        return "".join(text.split())
    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))


def _f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    return _f1_score(pred_tokens, gt_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    assert jieba is not None, "pip install jieba"
    pred_tokens = [normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
    gt_tokens = [normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
    pred_tokens = [t for t in pred_tokens if len(t) > 0]
    gt_tokens = [t for t in gt_tokens if len(t) > 0]
    return _f1_score(pred_tokens, gt_tokens)


def rouge_score(prediction, ground_truth, **kwargs):
    assert Rouge is not None, "pip install rouge"
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    assert jieba is not None, "pip install jieba"
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs["all_classes"]
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    assert fuzz is not None, "pip install fuzzywuzzy"
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


DATASET2METRIC = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score, "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score,
    "musique": qa_f1_score, "dureader": rouge_zh_score,
    "gov_report": rouge_score, "qmsum": rouge_score,
    "multi_news": rouge_score, "vcsum": rouge_zh_score,
    "trec": classification_score, "triviaqa": qa_f1_score,
    "samsum": rouge_score, "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}

TASK_CATEGORIES = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    "Code": ["lcc", "repobench-p"],
}


def evaluate_predictions(pred_dir):
    scores = {}
    for filename in sorted(os.listdir(pred_dir)):
        if not filename.endswith('.jsonl'):
            continue
        dataset_name = filename.replace('.jsonl', '')
        if dataset_name not in DATASET2METRIC:
            continue

        predictions, answers, all_classes_list = [], [], None
        with open(os.path.join(pred_dir, filename), 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                predictions.append(obj["pred"])
                answers.append(obj["answers"])
                all_classes_list = obj.get("all_classes")

        metric_fn = DATASET2METRIC[dataset_name]
        total_score = 0.0
        for pred, ground_truths in zip(predictions, answers):
            if dataset_name in ["trec", "triviaqa", "samsum", "lsht"]:
                pred = pred.lstrip('\n').split('\n')[0]
            score = 0.0
            for gt in ground_truths:
                score = max(score, metric_fn(pred, gt, all_classes=all_classes_list))
            total_score += score

        avg_score = round(100 * total_score / max(len(predictions), 1), 2)
        scores[dataset_name] = avg_score
        print(f"  {dataset_name}: {avg_score:.2f} ({len(predictions)} samples)")

    return scores


def print_comparison(new_scores, old_scores):
    """Print side-by-side comparison."""
    print(f"\n{'='*75}")
    print(f"  {'Dataset':<25} {'New':>10} {'Old':>10} {'Delta':>10}")
    print(f"{'='*75}")

    all_new, all_old = [], []
    for cat_name, datasets in TASK_CATEGORIES.items():
        cat_new, cat_old = [], []
        has_data = False
        for d in datasets:
            if d in new_scores or d in old_scores:
                has_data = True
                n = new_scores.get(d, None)
                o = old_scores.get(d, None)
                n_str = f"{n:.2f}" if n is not None else "  -"
                o_str = f"{o:.2f}" if o is not None else "  -"
                if n is not None and o is not None:
                    delta = n - o
                    d_str = f"{delta:+.2f}"
                    if delta > 0:
                        d_str = f"\033[92m{d_str}\033[0m"  # green
                    elif delta < 0:
                        d_str = f"\033[91m{d_str}\033[0m"  # red
                else:
                    d_str = "  -"
                print(f"  {d:<25} {n_str:>10} {o_str:>10} {d_str:>20}")
                if n is not None:
                    cat_new.append(n)
                    all_new.append(n)
                if o is not None:
                    cat_old.append(o)
                    all_old.append(o)

        if has_data:
            avg_n = sum(cat_new) / len(cat_new) if cat_new else 0
            avg_o = sum(cat_old) / len(cat_old) if cat_old else 0
            delta = avg_n - avg_o if cat_new and cat_old else 0
            d_str = f"{delta:+.2f}"
            print(f"  {'['+cat_name+']':<25} {avg_n:>10.2f} {avg_o:>10.2f} {d_str:>10}")
            print()

    if all_new:
        avg_n = sum(all_new) / len(all_new)
        avg_o = sum(all_old) / len(all_old) if all_old else 0
        delta = avg_n - avg_o if all_old else 0
        print(f"{'='*75}")
        print(f"  {'OVERALL':<25} {avg_n:>10.2f} {avg_o:>10.2f} {delta:>+10.2f}")
        print(f"  (new: {len(all_new)} datasets, old: {len(all_old)} datasets)")
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser("Score existing LongBench v1 predictions")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing .jsonl prediction files")
    parser.add_argument("--compare_result", type=str, default=None,
                        help="Path to a previous result.json for comparison")
    parser.add_argument("--save_result", type=str, default=None,
                        help="Save result.json to this path (default: pred_dir/../result.json)")
    args = parser.parse_args()

    print(f"Scoring predictions in: {args.pred_dir}")
    print()
    scores = evaluate_predictions(args.pred_dir)

    # Save
    save_path = args.save_result or os.path.join(os.path.dirname(args.pred_dir.rstrip('/')), "result.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(f"\nResults saved to: {save_path}")

    # Compare
    if args.compare_result:
        with open(args.compare_result, 'r') as f:
            old_scores = json.load(f)
        print_comparison(scores, old_scores)
    else:
        # Print category summary
        print(f"\n{'='*60}")
        for cat_name, datasets in TASK_CATEGORIES.items():
            cat_scores = [scores[d] for d in datasets if d in scores]
            if cat_scores:
                avg = sum(cat_scores) / len(cat_scores)
                print(f"  {cat_name:<20} {len(cat_scores)} datasets  avg={avg:.2f}")
        all_scores = list(scores.values())
        if all_scores:
            print(f"{'='*60}")
            print(f"  {'Overall':<20} {len(all_scores)} datasets  avg={sum(all_scores)/len(all_scores):.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
