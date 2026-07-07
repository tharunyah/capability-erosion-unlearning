# unlearn/generate_fisher_csv_from_draws.py
#
# Builds a summary CSV for the 8 extended Fisher configs (budget 300/400 +
# seed-repeats) purely from the raw per-draw .npy arrays that
# average_fisher_draws_extended.py already saved to results/. Does NOT load
# any model, checkpoint, or baseline.pt — reads only:
#   - data/capability_taxonomy.json          (tier map)
#   - data/baseline_per_class_acc.npy         (cached baseline accuracy)
#   - results/fisher_{tag}_alpha{A}_draws_extended.npy  (per config)
#
# This exists because no .pt checkpoints were ever saved for these 8 configs
# (the multi-draw script only ever kept models in memory long enough to
# evaluate them) — so a checkpoint-loading CSV generator can't work here.
# This script sidesteps that entirely by reusing the already-saved raw
# accuracy arrays instead of needing checkpoints at all.
#
# READ-ONLY w.r.t. everything except its own new output file below.
# Does not touch models/, does not touch any existing results/*.csv,
# does not touch the raw .npy files it reads (only np.load, never np.save
# on them). Writes exactly one new file:
#   results/fisher_results_from_saved_draws_alpha<A>.csv
#
# Run from the repo root: python unlearn/generate_fisher_csv_from_draws.py

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
    """
    Same 8 configs as average_fisher_draws_extended.py's build_configs(),
    but here we only need forget_set name (for the CSV column) and the tag
    (to locate the saved draws .npy) — no forget_indices, no seed_base,
    since we're not re-running anything.
    """
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
        mean_acc = per_class_accs.mean(axis=0)
        std_acc = per_class_accs.std(axis=0)

        overall_mean = float(np.mean(mean_acc))
        overall_diff = overall_mean - overall_baseline

        for tier_name, class_indices in tiers.items():
            b_acc = tier_accuracy(baseline_acc, class_indices)
            u_mean = tier_accuracy(mean_acc, class_indices)
            u_std = float(np.mean(std_acc[class_indices]))
            diff = u_mean - b_acc

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
                'unlearned_acc_std': u_std,
                'diff': diff,
                'overall_baseline': overall_baseline,
                'overall_unlearned_mean': overall_mean,
                'overall_diff': overall_diff,
            })

    out_path = f'results/fisher_results_from_saved_draws_alpha{alpha}.csv'
    fieldnames = [
        'forget_set', 'config_tag', 'strategy', 'budget', 'seed_repeat_id', 'tier',
        'alpha', 'n_draws', 'baseline_acc', 'unlearned_acc_mean',
        'unlearned_acc_std', 'diff', 'overall_baseline', 'overall_unlearned_mean',
        'overall_diff',
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}")
    print("No models, checkpoints, or existing files were touched — only new draws were read.")


if __name__ == '__main__':
    main()