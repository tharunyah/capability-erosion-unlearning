# experiments/compute_di_from_csv.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import json
from collections import defaultdict

REFERENCE_TIER = "majority"

RESULT_FILES = {
    "fine_tune": "results/finetune_results_multidraw_extended_steps50_corrected.csv",
    "fisher":    "results/fisher_results_multidraw_extended_alpha0.001_corrected.csv",
}


def get_unlearned_acc(row):
    if "unlearned_acc" in row:
        return float(row["unlearned_acc"])
    if "unlearned_acc_mean" in row:
        return float(row["unlearned_acc_mean"])
    raise KeyError("No unlearned_acc / unlearned_acc_mean column found")


def load_results(path):
    """One row per config_tag+tier -- groups by config_tag (e.g. 'influence_300')."""
    by_config = defaultdict(dict)
    meta = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            config_tag = row.get("config_tag", row["forget_set"])
            by_config[config_tag][row["tier"]] = {
                "baseline_acc": float(row["baseline_acc"]),
                "unlearned_acc": get_unlearned_acc(row),
            }
            meta[config_tag] = {"strategy": row["strategy"], "budget": row.get("budget", "")}
    return by_config, meta


def compute_di(tier_data, tier, reference_tier=REFERENCE_TIER):
    ref_before = tier_data[reference_tier]["baseline_acc"]
    ref_after  = tier_data[reference_tier]["unlearned_acc"]
    tgt_before = tier_data[tier]["baseline_acc"]
    tgt_after  = tier_data[tier]["unlearned_acc"]
    di_before  = tgt_before / (ref_before + 1e-8)
    di_after   = tgt_after / (ref_after + 1e-8)
    return di_before, di_after, di_after - di_before


def main():
    tiers = ["majority", "mid_tail", "long_tail", "safety_critical"]
    all_results = {}

    for method, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        by_config, meta = load_results(path)
        print(f"[loaded] {method}: {len(by_config)} configs from {path}")

        for config_tag, tier_data in by_config.items():
            label = f"{meta[config_tag]['strategy']}_b{meta[config_tag]['budget']}_{method}_{config_tag}"
            di_report = {}
            for tier in tiers:
                try:
                    di_before, di_after, di_shift = compute_di(tier_data, tier)
                except KeyError:
                    continue
                di_report[tier] = {
                    "baseline_acc": tier_data[tier]["baseline_acc"],
                    "unlearned_acc": tier_data[tier]["unlearned_acc"],
                    "di_before": di_before,
                    "di_after": di_after,
                    "di_shift": di_shift,
                }
            all_results[label] = di_report

    print(f"\n{'Experiment':<45}{'majority':>10}{'mid_tail':>10}{'long_tail':>10}{'safety_crit':>12}")
    print("-" * 87)
    for label, di in all_results.items():
        row = [f"{di[t]['di_shift']:>10.4f}" if t in di else f"{'--':>10}" for t in tiers]
        print(f"{label:<45}{''.join(row)}")

    os.makedirs("results", exist_ok=True)
    with open("results/full_summary_di_from_csv.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/full_summary_di_from_csv.json")


if __name__ == '__main__':
    main()