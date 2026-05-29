import sys; sys.path.insert(0, '.')
from eval.eval_longbench_v1 import evaluate_predictions, print_category_summary, evaluate_predictions_by_length, print_length_summary

for name, pred_dir in [
    # ('8KA1K-step13000', 'scripts/eval/logs/eval_longbench_v1_20260415_115947/8KA1K-no-noise-warmup1k-step13000/pred'),
    # ('8KA2K-step13000', 'scripts/eval/logs/eval_longbench_v1_20260415_115947/8KA2K-no-noise-warmup1k-step13000/pred'),
    ('olmo-base', 'scripts/eval/logs/eval_longbench_v1_20260327_170555/olmo3-base/pred'),

]:
    print(f'\n=== {name} ===')
    scores = evaluate_predictions(pred_dir)
    print_category_summary(scores)
    ds_b, cat_b, all_b = evaluate_predictions_by_length(pred_dir)
    print_length_summary(ds_b, cat_b, all_b)