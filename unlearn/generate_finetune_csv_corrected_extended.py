# unlearn/generate_finetune_csv_corrected_extended.py
#
# Corrected-std counterpart to finetune_results_multidraw_steps50_corrected.csv,
# applied to the 8 extended configs (budget 300/400 + seed-repeats).
#
# The ORIGINAL average_finetune_draws.py / average_finetune_draws_extended.py
# computed 'unlearned_acc_std' as the mean of each class's per-class std
# across draws — a rough spread indicator, not the actual uncertainty on the
# TIER-LEVEL mean itself. That number overstates noise because averaging over
# many classes within a tier cancels out a lot of per-class variance.
#
# This script instead computes, for each draw, the tier's mean accuracy
# (one number per draw), then takes the std ACROSS those n_draws tier-means.
# That's the correct quantity for judging whether a tier-level diff is
# distinguishable from run-to-run noise — hence 'diff_over_std' as a rough
# signal-to-noise ratio, matching the corrected original CSV's convention.
#
# READ-ONLY: only reads data/capability_taxonomy.json, data/baseline_per_
# class_acc.npy, and the existing results/finetune_{tag}_steps{N}_draws_
# extended.npy files. Writes exactly one new file:
#   results/finetune_results_multidraw_extended_steps<N>_corrected.csv
# Does not touch any other existing file.
#
# Run from the repo root: python unlearn/generate_finetune_csv_corrected_extended.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import json
import numpy as np

N_STEPS = 50  # must match the steps value used when the draws were generated


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


def main(n_steps=N_STEPS):
    tiers = load_tier_map()

    baseline_acc = np.load('data/baseline_per_class_acc.npy')
    overall_baseline = float(np.mean(baseline_acc))

    configs = build_configs()
    rows = []

    for cfg in configs:
        draws_path = f"results/finetune_{cfg['config_tag']}_steps{n_steps}_draws_extended.npy"
        print(f"Reading {draws_path}...")
        per_class_accs = np.load(draws_path)  # shape (n_draws, 100)
        n_draws = per_class_accs.shape[0]

        overall_mean = float(np.mean(per_class_accs))
        overall_diff = overall_mean - overall_baseline

        for tier_name, class_indices in tiers.items():
            b_acc = tier_accuracy(baseline_acc, class_indices)

            # Per-draw tier means: one scalar per draw, THEN std across draws.
            per_draw_tier_means = per_class_accs[:, class_indices].mean(axis=1)  # shape (n_draws,)
            u_mean = float(per_draw_tier_means.mean())
            u_std_corrected = float(per_draw_tier_means.std())  # population std, matches original corrected CSV
            diff = u_mean - b_acc

            diff_over_std = diff / u_std_corrected if u_std_corrected > 0 else float('inf') if diff != 0 else 0.0

            rows.append({
                'forget_set': cfg['forget_set_name'],
                'config_tag': cfg['config_tag'],
                'strategy': cfg['strategy'],
                'budget': cfg['budget'],
                'seed_repeat_id': cfg['seed_repeat_id'] if cfg['seed_repeat_id'] is not None else '',
                'tier': tier_name,
                'n_steps': n_steps,
                'n_draws': n_draws,
                'baseline_acc': b_acc,
                'unlearned_acc_mean': u_mean,
                'unlearned_acc_std_corrected': u_std_corrected,
                'diff': diff,
                'diff_over_std': diff_over_std,
                'overall_baseline': overall_baseline,
            })

    out_path = f'results/finetune_results_multidraw_extended_steps{n_steps}_corrected.csv'
    fieldnames = [
        'forget_set', 'config_tag', 'strategy', 'budget', 'seed_repeat_id', 'tier',
        'n_steps', 'n_draws', 'baseline_acc', 'unlearned_acc_mean',
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