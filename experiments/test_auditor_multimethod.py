"""
experiments/test_auditor_multimethod.py

Day 8/12/13 combined: tests the capability-aware auditor (audit/capability_monitor.py)
across MULTIPLE unlearning methods (gradient ascent + fine-tuning now, Fisher
forgetting to be added once results/fisher_results.csv exists).

WHY THIS MATTERS FOR THE PAPER (lit review, Contribution 4):
"Evaluate the tradeoff between target-capability degradation and aggregate
utility preservation" is a claim about unlearning generally, not just one
method. If the auditor only works on gradient ascent, that's a much weaker
result than if it holds across gradient ascent AND fine-tuning (and Fisher
tomorrow). This script also fixes the "only 3 benign + 3 attack samples"
limitation flagged earlier -- combining methods roughly doubles the sample
size for precision/recall/F1.

Handles two different CSV schemas automatically:
  - gradient_ascent_results.csv: has an "unlearned_acc" column
  - finetune_results_multidraw_steps50_corrected.csv: has "unlearned_acc_mean"
    instead (already averaged over 5 random re-draws per forget set -- no
    row-collision risk, one row per forget_set+tier just like gradient ascent)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
from collections import defaultdict
from audit.capability_monitor import CapabilityMonitor

RESULT_FILES = {
    "gradient_ascent": "results/gradient_ascent_results.csv",
    "finetune": "results/finetune_results_multidraw_steps50_corrected.csv",
    "fisher": "results/fisher_results_multidraw_extended_alpha0.001_corrected.csv",  
}

TARGET_TIER = "long_tail"       # matches your forget sets -- long_tail-only, per your earlier fix
REFERENCE_TIER = "majority"
OUTPUT_CSV = "results/auditor_test_results_multimethod.csv"


def get_unlearned_acc(row):
    """Handles both column naming schemes across your two CSVs."""
    if "unlearned_acc" in row:
        return float(row["unlearned_acc"])
    if "unlearned_acc_mean" in row:
        return float(row["unlearned_acc_mean"])
    raise KeyError("No unlearned_acc / unlearned_acc_mean column found")


def load_results(path, method):
    """One row per forget_set+tier in both your files -- no draw/seed collision to worry about."""
    by_run = defaultdict(dict)
    meta = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            run_id = f"{method}::{row['forget_set']}"
            by_run[run_id][row["tier"]] = {
                "baseline_acc": float(row["baseline_acc"]),
                "unlearned_acc": get_unlearned_acc(row),
            }
            meta[run_id] = {"method": method, "strategy": row["strategy"], "budget": row.get("budget", "")}
    return by_run, meta


def compute_di(tier_data, tier, reference_tier):
    """Feldman et al. (2015) Disparate Impact, unmodified -- computed before/after unlearning."""
    ref_before = tier_data[reference_tier]["baseline_acc"]
    ref_after = tier_data[reference_tier]["unlearned_acc"]
    tgt_before = tier_data[tier]["baseline_acc"]
    tgt_after = tier_data[tier]["unlearned_acc"]
    di_before = tgt_before / (ref_before + 1e-8)
    di_after = tgt_after / (ref_after + 1e-8)
    return di_before, di_after, di_after - di_before


def main():
    all_by_run, all_meta = {}, {}
    for method, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"[skip] {path} not found yet")
            continue
        by_run, meta = load_results(path, method)
        all_by_run.update(by_run)
        all_meta.update(meta)
        print(f"[loaded] {method}: {len(by_run)} runs from {path}")

    di_per_run = {}
    print(f"\n{'run_id':<40}{'strategy':<10}{'di_before':>10}{'di_after':>10}{'di_shift':>10}")
    for run_id, tier_data in all_by_run.items():
        try:
            di_before, di_after, di_shift = compute_di(tier_data, TARGET_TIER, REFERENCE_TIER)
        except KeyError:
            print(f"[warn] {run_id} missing tier data, skipping")
            continue
        di_per_run[run_id] = {"di_before": di_before, "di_after": di_after, "di_shift": di_shift, **all_meta[run_id]}
        print(f"{run_id:<40}{all_meta[run_id]['strategy']:<10}{di_before:>10.3f}{di_after:>10.3f}{di_shift:>10.3f}")

    benign_runs = [r for r, v in di_per_run.items() if v["strategy"] == "random"]
    attack_runs = [r for r, v in di_per_run.items() if v["strategy"] == "influence"]
    print(f"\nBenign (random) runs: {len(benign_runs)}   Attack (influence) runs: {len(attack_runs)}")

    methods = sorted(set(v["method"] for v in di_per_run.values()))
    results_rows = []
    tp = fp = tn = fn = 0

    for method in methods:
        method_benign = [r for r in benign_runs if di_per_run[r]["method"] == method]
        method_attack = [r for r in attack_runs if di_per_run[r]["method"] == method]

        # Calibrate "normal" separately per method -- gradient ascent and
        # fine-tuning erode at very different magnitudes, so pooling them
        # into one history washes out the smaller (but real) fine-tune signal.
        for held_out in method_benign:
            monitor = CapabilityMonitor(di_threshold=0.0, zscore_threshold=1.0, min_history_for_zscore=2)
            other_shifts = [di_per_run[r]["di_shift"] for r in method_benign if r != held_out]
            monitor.seed_benign_history(other_shifts, tier=TARGET_TIER)

            v = di_per_run[held_out]
            result = monitor.check(v["di_after"], v["di_shift"], tier=TARGET_TIER)
            result.update({"run_id": held_out, "true_label": "benign", "method": method})
            results_rows.append(result)
            fp += result["flagged"]
            tn += not result["flagged"]

        monitor = CapabilityMonitor(di_threshold=0.0, zscore_threshold=1.0, min_history_for_zscore=2)
        monitor.seed_benign_history([di_per_run[r]["di_shift"] for r in method_benign], tier=TARGET_TIER)

        for r in method_attack:
            v = di_per_run[r]
            result = monitor.check(v["di_after"], v["di_shift"], tier=TARGET_TIER)
            result.update({"run_id": r, "true_label": "attack", "method": method})
            results_rows.append(result)
            tp += result["flagged"]
            fn += not result["flagged"]

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["run_id", "method", "true_label", "tier", "di_after", "di_shift", "flagged", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
