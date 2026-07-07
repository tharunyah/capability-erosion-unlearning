# unlearn/average_finetune_draws_extended.py
#
# Extension of average_finetune_draws.py covering the NEW forget sets that
# didn't exist when that script was written:
#   - Extended budgets: forget_{influence,random}_{300,400}.npy
#   - Seed-repeat forget sets: forget_random_100_seed{43,44,45,46}.npy
#     (four additional, independently-constructed random forget sets at the
#     same nominal budget=100, built to give the auditor more distinct
#     benign/random samples — see generate_random_repeats.py. There is no
#     influence-side equivalent since influence selection is deterministic.)
#
# Does NOT touch anything average_finetune_draws.py already produced.
# Writes to its own results/finetune_results_multidraw_extended_steps<N>.csv
# and its own raw per-draw .npy files (distinct filenames from the original
# script, so nothing gets overwritten).
#
# Lives in unlearn/, alongside average_finetune_draws.py. Run it from the
# repo root (e.g. `python unlearn/average_finetune_draws_extended.py`) so the
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

from unlearn.fine_tune import fine_tune_forget_multi_draw
from evaluate.per_class_eval import load_model, evaluate_per_class

# Seed offsets for the "training-time" multi-draw seeds (DataLoader shuffle
# seed per draw, 5 draws per config) — kept distinct per config so no two
# configs ever reuse the same 5 seeds, even though that would likely be
# harmless (different forget/retain sets), just for cleanliness.
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
    Returns a list of config dicts, each describing one (forget_set, strategy,
    budget, seed_base, config_tag) to run multi-draw fine-tuning on.
    config_tag is used purely for output filenames — the 'budget' field in
    the CSV always reflects the true nominal budget (100 for seed-repeats),
    while config_tag disambiguates the seed-repeat variants from each other
    and from the original forget_random_100.npy.
    """
    configs = []

    # New budget-extension sets: both strategies, budgets 300 and 400.
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

    # Seed-repeat sets: random strategy only, nominal budget=100.
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

    configs = build_configs()
    rows = []

    for cfg in configs:
        print(f"\n=== {cfg['config_tag']} (strategy={cfg['strategy']}, "
              f"nominal budget={cfg['budget']}) ===")
        forget_indices = np.load(cfg['forget_set_path'])

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
            seed_base=cfg['seed_base'],
        )

        raw_save_path = f"results/finetune_{cfg['config_tag']}_steps{n_steps}_draws_extended.npy"
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

    out_path = f'results/finetune_results_multidraw_extended_steps{n_steps}.csv'
    fieldnames = [
        'forget_set', 'config_tag', 'strategy', 'budget', 'seed_repeat_id', 'tier',
        'n_steps', 'lr', 'n_draws', 'baseline_acc', 'unlearned_acc_mean',
        'unlearned_acc_std', 'diff', 'overall_baseline', 'overall_unlearned_mean',
        'overall_diff',
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}")


if __name__ == '__main__':
    main(n_draws=5, n_steps=50, lr=1e-4, batch_size=32)