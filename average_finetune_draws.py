# average_finetune_draws.py
#
# Multi-draw variant of generate_finetune_csv.py: instead of evaluating one
# fine-tuned checkpoint per (strategy, budget), runs n_draws independent
# training runs (each an independently seeded DataLoader shuffle) per config,
# and reports mean/std per tier. Answers: "is long_tail reliably the
# worst-hit tier once single-run batch-order noise is averaged out, or was
# the single-run pattern (long_tail worst in 3/6 configs) partly a lucky
# draw?"
#
# Does not overwrite anything generate_finetune_csv.py produces — writes to
# its own results/finetune_results_multidraw_steps<N_STEPS>.csv and saves raw
# per-draw arrays to results/ for later inspection.

import csv
import json
import numpy as np
import torch
import torch.nn as nn

from unlearn.fine_tune import fine_tune_forget_multi_draw
from evaluate.per_class_eval import load_model, evaluate_per_class

# Fixed, deterministic seed offsets per strategy so seed_base doesn't rely
# on Python's (randomized-by-default) str hashing.
STRATEGY_SEED_OFFSET = {'influence': 0, 'random': 5000}


def load_tier_map(taxonomy_path='data/capability_taxonomy.json'):
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    tiers = {'majority': [], 'mid_tail': [], 'long_tail': [], 'safety_critical': []}
    for cls_str, tier in taxonomy.items():
        tiers[tier].append(int(cls_str))
    return tiers


def tier_accuracy(per_class_acc, class_indices):
    return float(np.mean(per_class_acc[class_indices]))


def main(n_draws=5, n_steps=50, lr=1e-4, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: n_draws={n_draws}, n_steps={n_steps}, lr={lr}, batch_size={batch_size}")

    tiers = load_tier_map()
    lt_train_indices = np.load('data/lt_train_indices.npy')
    loss_fn = nn.CrossEntropyLoss()

    print("\nEvaluating baseline...")
    baseline_model = load_model('models/baseline.pt', device)
    baseline_acc = evaluate_per_class(baseline_model, device, save_path=None)
    overall_baseline = float(np.mean(baseline_acc))

    strategies = ['influence', 'random']
    budgets = [50, 100, 200]

    rows = []

    for strategy in strategies:
        for budget in budgets:
            print(f"\n=== {strategy}, budget={budget} ===")
            forget_indices = np.load(f'data/forget_{strategy}_{budget}.npy')

            seed_base = STRATEGY_SEED_OFFSET[strategy] + budget

            per_class_accs = fine_tune_forget_multi_draw(
                baseline_path='models/baseline.pt',
                lt_train_indices=lt_train_indices,
                forget_indices=forget_indices,
                device=device,
                loss_fn=loss_fn,
                n_draws=n_draws,
                n_steps=n_steps,
                lr=lr,
                batch_size=batch_size,
                seed_base=seed_base,
            )

            # Raw per-draw array, kept for later variance/outlier inspection.
            raw_save_path = f'results/finetune_{strategy}_{budget}_steps{n_steps}_draws.npy'
            np.save(raw_save_path, per_class_accs)
            print(f"  Saved raw draws -> {raw_save_path}")

            mean_acc = per_class_accs.mean(axis=0)
            std_acc = per_class_accs.std(axis=0)

            overall_mean = float(np.mean(mean_acc))
            overall_diff = overall_mean - overall_baseline

            forget_set_name = f"forget_{strategy}_{budget}.npy"

            for tier_name, class_indices in tiers.items():
                b_acc = tier_accuracy(baseline_acc, class_indices)
                u_mean = tier_accuracy(mean_acc, class_indices)
                # Mean of per-class std within the tier — a rough spread
                # indicator, not the std of the tier mean itself (same
                # caveat noted for the Fisher multi-draw CSV).
                u_std = float(np.mean(std_acc[class_indices]))
                diff = u_mean - b_acc

                rows.append({
                    'forget_set': forget_set_name,
                    'strategy': strategy,
                    'budget': budget,
                    'tier': tier_name,
                    'n_steps': n_steps,
                    'lr': lr,
                    'n_draws': n_draws,
                    'baseline_acc': b_acc,
                    'unlearned_acc_mean': u_mean,
                    'unlearned_acc_std': u_std,
                    'diff': diff,
                    'overall_baseline': overall_baseline,
                    'overall_unlearned_mean': overall_mean,
                    'overall_diff': overall_diff,
                })

    out_path = f'results/finetune_results_multidraw_steps{n_steps}.csv'
    fieldnames = [
        'forget_set', 'strategy', 'budget', 'tier', 'n_steps', 'lr', 'n_draws',
        'baseline_acc', 'unlearned_acc_mean', 'unlearned_acc_std', 'diff',
        'overall_baseline', 'overall_unlearned_mean', 'overall_diff',
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}")


if __name__ == '__main__':
    main(n_draws=5, n_steps=50, lr=1e-4, batch_size=32)