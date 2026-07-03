# correct_finetune_multidraw_std.py
#
# Reprocesses the raw per-draw arrays saved by average_finetune_draws.py to
# compute the CORRECT uncertainty on each tier diff: std of the 5 tier-level
# MEAN values across draws, rather than the mean of per-class std within the
# tier (what finetune_results_multidraw_steps50.csv reports, which overstates
# uncertainty — same known caveat flagged earlier for the Fisher multidraw CSV).
#
# Pure reprocessing: loads results/finetune_{strategy}_{budget}_steps{n_steps}_draws.npy
# (already on disk from average_finetune_draws.py) and data/baseline_per_class_acc.npy
# (already on disk from generate_finetune_csv.py). No model loading, no GPU,
# no retraining. Writes a new CSV, does not touch any existing file.

import csv
import json
import numpy as np


def load_tier_map(taxonomy_path='data/capability_taxonomy.json'):
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    tiers = {'majority': [], 'mid_tail': [], 'long_tail': [], 'safety_critical': []}
    for cls_str, tier in taxonomy.items():
        tiers[tier].append(int(cls_str))
    return tiers


def main(n_steps=50):
    tiers = load_tier_map()

    baseline_acc = np.load('data/baseline_per_class_acc.npy')
    overall_baseline = float(np.mean(baseline_acc))

    strategies = ['influence', 'random']
    budgets = [50, 100, 200]

    rows = []

    for strategy in strategies:
        for budget in budgets:
            raw_path = f'results/finetune_{strategy}_{budget}_steps{n_steps}_draws.npy'
            per_class_accs = np.load(raw_path)  # shape (n_draws, 100)
            n_draws = per_class_accs.shape[0]

            forget_set_name = f"forget_{strategy}_{budget}.npy"

            for tier_name, class_indices in tiers.items():
                b_acc = float(np.mean(baseline_acc[class_indices]))

                # Tier-level mean PER DRAW first (5 numbers), then stats
                # across those — this is the corrected quantity.
                per_draw_tier_means = per_class_accs[:, class_indices].mean(axis=1)  # shape (n_draws,)

                u_mean = float(np.mean(per_draw_tier_means))
                u_std_corrected = float(np.std(per_draw_tier_means, ddof=1))  # sample std across draws
                diff = u_mean - b_acc

                # Rough signal-to-noise: |diff| relative to the corrected std.
                # <1 means the diff is within one draw-to-draw standard
                # deviation of zero — worth flagging, not proof of absence.
                snr = abs(diff) / u_std_corrected if u_std_corrected > 0 else float('nan')

                rows.append({
                    'forget_set': forget_set_name,
                    'strategy': strategy,
                    'budget': budget,
                    'tier': tier_name,
                    'n_steps': n_steps,
                    'n_draws': n_draws,
                    'baseline_acc': b_acc,
                    'unlearned_acc_mean': u_mean,
                    'unlearned_acc_std_corrected': u_std_corrected,
                    'diff': diff,
                    'diff_over_std': snr,
                    'overall_baseline': overall_baseline,
                })

    out_path = f'results/finetune_results_multidraw_steps{n_steps}_corrected.csv'
    fieldnames = [
        'forget_set', 'strategy', 'budget', 'tier', 'n_steps', 'n_draws',
        'baseline_acc', 'unlearned_acc_mean', 'unlearned_acc_std_corrected',
        'diff', 'diff_over_std', 'overall_baseline',
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {len(rows)} rows written to {out_path}")

    # Quick console summary: worst tier per config, and whether diff_over_std
    # clears a rough "at least 1 std away from zero" bar.
    print("\n--- Worst-hit tier per config (corrected std) ---")
    seen_configs = set()
    for row in rows:
        key = (row['strategy'], row['budget'])
        if key in seen_configs:
            continue
        seen_configs.add(key)
        config_rows = [r for r in rows if (r['strategy'], r['budget']) == key]
        worst = min(config_rows, key=lambda r: r['diff'])
        flag = "" if worst['diff_over_std'] >= 1.0 else "  (diff < 1 std — weak signal)"
        print(f"  {worst['strategy']}, budget={worst['budget']}: "
              f"{worst['tier']} (diff={worst['diff']:.4f}, "
              f"std={worst['unlearned_acc_std_corrected']:.4f}, "
              f"diff/std={worst['diff_over_std']:.2f}){flag}")


if __name__ == '__main__':
    main(n_steps=50)