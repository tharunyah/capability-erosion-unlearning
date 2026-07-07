# unlearn/generate_fisher_csv_corrected_extended.py
#
# Corrected-std counterpart for Fisher's extended configs (budget 300/400 +
# seed-repeats), matching the same fix as
# generate_finetune_csv_corrected_extended.py: std is computed ACROSS the
# n_draws tier-level means (one scalar per draw), not as the mean of each
# class's per-class std — the latter overstates noise since it doesn't
# benefit from averaging across classes within a tier.
#
# READ-ONLY: only reads data/capability_taxonomy.json, data/baseline_per_
# class_acc.npy, and the existing results/fisher_{tag}_alpha{A}_draws_
# extended.npy files. Writes exactly one new file:
#   results/fisher_results_multidraw_extended_alpha<A>_corrected.csv
# Does not touch any other existing file.
#
# NOTE: uses ALPHA=1e-3 by default, matching the value the extended draws
# were generated with. If you've since re-run the extended draws with a
# different alpha (e.g. the fix mentioned for avoiding model collapse),
# update ALPHA below to match, or the draws_path won't be found.
#
# Run from the repo root: python unlearn/generate_fisher_csv_corrected_extended.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import json
import numpy as np

ALPHA = 1e-3  # must match the alpha value used when the draws were generated


def load_tier_map(taxonomy_path='data/capability_taxonomy.json'):
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    tiers = {'majority': [], 'mid_tail': [], 'long_tail': [], 'safety_critical': []}
    for cls_str, tier in taxonomy.items():
        tiers[tier].append(int(cls_str))
    return tiers


def tier_accuracy(per_class_acc, class_indices):
    return float(np.mean(per_class_acc[class_indices]))


def build_configs():
    configs = []

    for strategy in ['influence', 'random']:
        for budget in [300, 400]:
            configs.append({
                'forget_set_name': f'forget_{strategy}_{budget}.npy',
                'strategy': strategy,
                'budget': budget,
                'config_tag': f'{strategy}_{budget}',
                'seed_repeat_id': None,
            })

    for seed_id in [43, 44, 45, 46]:
        configs.append({
            'forget_set_name': f'forget_random_100_seed{seed_id}.npy',
            'strategy': 'random',
            'budget': 100,
            'config_tag': f'random_100_seed{seed_id}',
            'seed_repeat_id': seed_id,
        })

    return configs


def main(alpha=ALPHA):
    tiers = load_tier_map()

    baseline_acc = np.load('data/baseline_per_class_acc.npy')
    overall_baseline = float(np.mean(baseline_acc))

    configs = build_configs()
    rows = []

    for cfg in configs:
        draws_path = f"results/fisher_{cfg['config_tag']}_alpha{alpha}_draws_extended.npy"
        print(f"Reading {draws_path}...")
        per_class_accs = np.load(draws_path)  # shape (n_draws, 100)
        n_draws = per_class_accs.shape[0]

        overall_mean = float(np.mean(per_class_accs))
        overall_diff = overall_mean - overall_baseline

        for tier_name, class_indices in tiers.items():
            b_acc = tier_accuracy(baseline_acc, class_indices)

            per_draw_tier_means = per_class_accs[:, class_indices].mean(axis=1)  # shape (n_draws,)
            u_mean = float(per_draw_tier_means.mean())
            u_std_corrected = float(per_draw_tier_means.std())
            diff = u_mean - b_acc

            diff_over_std = diff / u_std_corrected if u_std_corrected > 0 else float('inf') if diff != 0 else 0.0

            rows.append({
                'forget_set': cfg['forget_set_name'],
                'config_tag': cfg['config_tag'],
                'strategy': cfg['strategy'],
                'budget': cfg['budget'],
                'seed_repeat_id': cfg['seed_repeat_id'] if cfg['seed_repeat_id'] is not None else '',
                'tier': tier_name,
                'alpha': alpha,
                'n_draws': n_draws,
                'baseline_acc': b_acc,
                'unlearned_acc_mean': u_mean,
                'unlearned_acc_std_corrected': u_std_corrected,
                'diff': diff,
                'diff_over_std': diff_over_std,
                'overall_baseline': overall_baseline,
            })

    out_path = f'results/fisher_results_multidraw_extended_alpha{alpha}_corrected.csv'
    fieldnames = [
        'forget_set', 'config_tag', 'strategy', 'budget', 'seed_repeat_id', 'tier',
        'alpha', 'n_draws', 'baseline_acc', 'unlearned_acc_mean',
        'unlearned_acc_std_corrected', 'diff', 'diff_over_std', 'overall_baseline',
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}")
    print("No models, checkpoints, or existing files were touched — only new draws were read.")


if __name__ == '__main__':
    main()