# unlearn/average_fisher_draws_extended.py
#
# Extension of average_fisher_draws.py covering the NEW forget sets that
# didn't exist when that script was written:
#   - Extended budgets: forget_{influence,random}_{300,400}.npy
#   - Seed-repeat forget sets: forget_random_100_seed{43,44,45,46}.npy
#     (four additional, independently-constructed random forget sets at the
#     same nominal budget=100, built to give the auditor more distinct
#     benign/random samples — see generate_random_repeats.py. There is no
#     influence-side equivalent since influence selection is deterministic.)
#
# Does NOT touch anything average_fisher_draws.py already produced.
# Writes to its own results/fisher_results_multidraw_extended_alpha<A>.csv
# and its own raw per-draw .npy files (distinct filenames from the original
# script, so nothing gets overwritten).
#
# NOTE ON COST: Fisher's expensive step is one backward pass per retain
# sample, computed ONCE per config and reused across the 5 noise draws.
# Adding 8 new configs (4 budget + 4 seed-repeat) means 8 new full Fisher
# computations, not 40 — same cost structure as the original script, just
# more configs.
#
# Lives in unlearn/, alongside average_fisher_draws.py. Run it from the
# repo root (e.g. `python unlearn/average_fisher_draws_extended.py`) so the
# relative data/ and results/ paths below resolve correctly — the sys.path
# insert only fixes imports, not the cwd-relative file paths.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import json
import numpy as np
import torch
import torch.nn as nn

from unlearn.fisher_forgetting import fisher_forget_multi_draw
from evaluate.per_class_eval import load_model, evaluate_per_class

# Seed offsets for the "noise-draw" multi-draw seeds (5 independent noise
# realizations per config, on top of one shared Fisher computation) — kept
# distinct per config for cleanliness, no known collision risk either way.
STRATEGY_SEED_OFFSET = {'influence': 0, 'random': 5000}
SEED_REPEAT_BASE_OFFSET = 20000  # + seed_id * 100, e.g. seed43 -> 24300


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
    Same config set as average_finetune_draws_extended.py's build_configs(),
    kept as a separate copy here (rather than a shared import) so this
    script has no import-time dependency on the fine_tune extension script.
    """
    configs = []

    for strategy in ['influence', 'random']:
        for budget in [300, 400]:
            configs.append({
                'forget_set_path': f'data/forget_{strategy}_{budget}.npy',
                'strategy': strategy,
                'budget': budget,
                'seed_base': STRATEGY_SEED_OFFSET[strategy] + budget,
                'config_tag': f'{strategy}_{budget}',
                'seed_repeat_id': None,
            })

    for seed_id in [43, 44, 45, 46]:
        configs.append({
            'forget_set_path': f'data/forget_random_100_seed{seed_id}.npy',
            'strategy': 'random',
            'budget': 100,
            'seed_base': SEED_REPEAT_BASE_OFFSET + seed_id * 100,
            'config_tag': f'random_100_seed{seed_id}',
            'seed_repeat_id': seed_id,
        })

    return configs


def main(n_draws=5, alpha=1e-3, std_clip=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: n_draws={n_draws}, alpha={alpha}, std_clip={std_clip}")

    tiers = load_tier_map()
    lt_train_indices = np.load('data/lt_train_indices.npy')
    loss_fn = nn.CrossEntropyLoss()

    print("\nEvaluating baseline...")
    baseline_model = load_model('models/baseline.pt', device)
    baseline_acc = evaluate_per_class(baseline_model, device, save_path=None)
    overall_baseline = float(np.mean(baseline_acc))

    configs = build_configs()
    rows = []

    for cfg in configs:
        print(f"\n=== {cfg['config_tag']} (strategy={cfg['strategy']}, "
              f"nominal budget={cfg['budget']}) ===")
        forget_indices = np.load(cfg['forget_set_path'])

        per_class_accs = fisher_forget_multi_draw(
            baseline_path='models/baseline.pt',
            lt_train_indices=lt_train_indices,
            forget_indices=forget_indices,
            device=device,
            loss_fn=loss_fn,
            n_draws=n_draws,
            alpha=alpha,
            std_clip=std_clip,
            seed_base=cfg['seed_base'],
        )

        raw_save_path = f"results/fisher_{cfg['config_tag']}_alpha{alpha}_draws_extended.npy"
        np.save(raw_save_path, per_class_accs)
        print(f"  Saved raw draws -> {raw_save_path}")

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
                'forget_set': cfg['forget_set_path'].split('/')[-1],
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

    out_path = f'results/fisher_results_multidraw_extended_alpha{alpha}.csv'
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


if __name__ == '__main__':
    main(n_draws=5, alpha=1e-3, std_clip=0.01)