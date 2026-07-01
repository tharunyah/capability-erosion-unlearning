# generate_fisher_csv.py
import csv
import json
import numpy as np
import torch

from evaluate.per_class_eval import load_model, evaluate_per_class


def load_tier_map(taxonomy_path='data/capability_taxonomy.json'):
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    tiers = {'majority': [], 'mid_tail': [], 'long_tail': [], 'safety_critical': []}
    for cls_str, tier in taxonomy.items():
        tiers[tier].append(int(cls_str))
    return tiers


def tier_accuracy(per_class_acc, class_indices):
    return float(np.mean(per_class_acc[class_indices]))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tiers = load_tier_map()

    # ── Baseline ──────────────────────────────────────────────────────────
    print("\nEvaluating baseline...")
    baseline_model = load_model('models/baseline.pt', device)
    baseline_acc   = evaluate_per_class(
        baseline_model, device, save_path='data/baseline_per_class_acc.npy'
    )
    overall_baseline = float(np.mean(baseline_acc))

    # ── Fisher checkpoints ────────────────────────────────────────────────
    strategies = ['influence', 'random']
    budgets    = [50, 100, 200]

    rows = []

    for strategy in strategies:
        for budget in budgets:
            ckpt_path = f'models/unlearned_fisher_{strategy}_{budget}.pt'
            print(f"\nEvaluating {ckpt_path}...")

            model = load_model(ckpt_path, device)
            save_path = f'data/unlearned_fisher_{strategy}_{budget}_per_class_acc.npy'
            unlearned_acc = evaluate_per_class(model, device, save_path=save_path)

            overall_unlearned = float(np.mean(unlearned_acc))
            overall_diff      = overall_unlearned - overall_baseline

            forget_set_name = f"forget_{strategy}_{budget}.npy"

            for tier_name, class_indices in tiers.items():
                b_acc = tier_accuracy(baseline_acc, class_indices)
                u_acc = tier_accuracy(unlearned_acc, class_indices)
                diff  = u_acc - b_acc

                rows.append({
                    'forget_set':        forget_set_name,
                    'strategy':          strategy,
                    'budget':            budget,
                    'tier':              tier_name,
                    'baseline_acc':      b_acc,
                    'unlearned_acc':     u_acc,
                    'diff':              diff,
                    'overall_baseline':  overall_baseline,
                    'overall_unlearned': overall_unlearned,
                    'overall_diff':      overall_diff,
                })

    # ── Write CSV ─────────────────────────────────────────────────────────
    out_path = 'results/fisher_results.csv'
    fieldnames = [
        'forget_set', 'strategy', 'budget', 'tier',
        'baseline_acc', 'unlearned_acc', 'diff',
        'overall_baseline', 'overall_unlearned', 'overall_diff'
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}")


if __name__ == '__main__':
    main()