"""
experiments/test_auditor_multimethod.py

Day 8/12/13 combined: tests the capability-aware auditor (audit/capability_monitor.py)
across MULTIPLE unlearning methods (gradient ascent, fine-tuning, and Fisher forgetting),
across ALL FOUR capability tiers (majority, mid_tail, long_tail, safety_critical).

WHY THIS MATTERS FOR THE PAPER (lit review, Contribution 4):
"Evaluate the tradeoff between target-capability degradation and aggregate
utility preservation" is a claim about unlearning generally, not just one
method or one tier. Checking only long_tail would have systematically missed
Fisher's real damage pattern (Fisher hits mid_tail hardest, not long_tail --
confirmed via check_results.py on the corrected CSVs). All four tiers are
run per method so the auditor's behavior on each is visible, including
majority, which acts as an internal sanity check (majority vs. itself as
reference tier is trivially ~1.0 DI and should never flag).

Handles three different CSV schemas automatically:
  - gradient_ascent_results.csv: has an "unlearned_acc" column
  - finetune_results_multidraw_steps50_corrected.csv: has "unlearned_acc_mean"
  - fisher_results_multidraw_extended_alpha0.001_corrected.csv: also has
    "unlearned_acc_mean" (already averaged over draws per forget set)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
from collections import defaultdict
from audit.capability_monitor import CapabilityMonitor

RESULT_FILES = {
    "gradient_ascent": "results/gradient_ascent_results.csv",
    "finetune": "results/finetune_results_multidraw_steps50_corrected.csv",
<<<<<<< HEAD
    "fisher": "results/fisher_results_multidraw_extended_alpha0.001_corrected.csv",  
=======
    "fisher": "results/fisher_results_multidraw_extended_alpha0.001_corrected.csv",
>>>>>>> 1bb6ca4dbd92115df6fa40f4e7536c45e4b1994c
}

TARGET_TIERS = ["majority", "mid_tail", "long_tail", "safety_critical"]
REFERENCE_TIER = "majority"  # NOTE: majority-vs-majority is a degenerate/sanity-check case (DI ~= 1.0 always)
OUTPUT_CSV = "results/auditor_test_results_multimethod.csv"


def get_unlearned_acc(row):
    """Handles all three column naming schemes across your CSVs."""
    if "unlearned_acc" in row:
        return float(row["unlearned_acc"])
    if "unlearned_acc_mean" in row:
        return float(row["unlearned_acc_mean"])
    raise KeyError("No unlearned_acc / unlearned_acc_mean column found")


def load_results(path, method):
    """One row per forget_set+tier in all three files -- no draw/seed collision to worry about."""
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


def run_tier(all_by_run, all_meta, target_tier, results_rows):
    """Runs the full benign/attack auditor test for one target tier, across all methods.
    Appends rows to results_rows (mutated in place) and returns tp/fp/tn/fn counts for this tier."""

    di_per_run = {}
    print(f"\n{'='*70}")
    print(f"TARGET TIER: {target_tier}  (reference tier: {REFERENCE_TIER})")
    print(f"{'='*70}")
    print(f"{'run_id':<40}{'strategy':<10}{'di_before':>10}{'di_after':>10}{'di_shift':>10}")

    for run_id, tier_data in all_by_run.items():
        try:
            di_before, di_after, di_shift = compute_di(tier_data, target_tier, REFERENCE_TIER)
        except KeyError:
            print(f"[warn] {run_id} missing tier data for '{target_tier}', skipping")
            continue
        di_per_run[run_id] = {"di_before": di_before, "di_after": di_after, "di_shift": di_shift, **all_meta[run_id]}
        print(f"{run_id:<40}{all_meta[run_id]['strategy']:<10}{di_before:>10.3f}{di_after:>10.3f}{di_shift:>10.3f}")

    benign_runs = [r for r, v in di_per_run.items() if v["strategy"] == "random"]
    attack_runs = [r for r, v in di_per_run.items() if v["strategy"] == "influence"]
    print(f"\nBenign (random) runs: {len(benign_runs)}   Attack (influence) runs: {len(attack_runs)}")

    methods = sorted(set(v["method"] for v in di_per_run.values()))
    tp = fp = tn = fn = 0

    for method in methods:
        method_benign = [r for r in benign_runs if di_per_run[r]["method"] == method]
        method_attack = [r for r in attack_runs if di_per_run[r]["method"] == method]

        # Calibrate "normal" separately per method -- different methods erode
        # at very different magnitudes, so pooling into one history washes
        # out smaller-but-real signals.
        for held_out in method_benign:
            monitor = CapabilityMonitor(di_threshold=0.0, zscore_threshold=1.0, min_history_for_zscore=2)
            other_shifts = [di_per_run[r]["di_shift"] for r in method_benign if r != held_out]
            monitor.seed_benign_history(other_shifts, tier=target_tier)

            v = di_per_run[held_out]
            result = monitor.check(v["di_after"], v["di_shift"], tier=target_tier)
            result.update({"run_id": held_out, "true_label": "benign", "method": method})
            results_rows.append(result)
            fp += result["flagged"]
            tn += not result["flagged"]

        monitor = CapabilityMonitor(di_threshold=0.0, zscore_threshold=1.0, min_history_for_zscore=2)
        monitor.seed_benign_history([di_per_run[r]["di_shift"] for r in method_benign], tier=target_tier)

        for r in method_attack:
            v = di_per_run[r]
            result = monitor.check(v["di_after"], v["di_shift"], tier=target_tier)
            result.update({"run_id": r, "true_label": "attack", "method": method})
            results_rows.append(result)
            tp += result["flagged"]
            fn += not result["flagged"]

    return tp, fp, tn, fn


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

    results_rows = []
    tier_totals = {}

    for target_tier in TARGET_TIERS:
        tp, fp, tn, fn = run_tier(all_by_run, all_meta, target_tier, results_rows)
        tier_totals[target_tier] = (tp, fp, tn, fn)

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["run_id", "method", "true_label", "tier", "di_after", "di_shift", "flagged", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\n{'='*70}")
    print("PER-TIER SUMMARY")
    print(f"{'='*70}")
    print(f"{'tier':<18}{'TP':>5}{'FP':>5}{'TN':>5}{'FN':>5}{'Precision':>12}{'Recall':>10}{'F1':>8}")
    for tier, (tp, fp, tn, fn) in tier_totals.items():
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{tier:<18}{tp:>5}{fp:>5}{tn:>5}{fn:>5}{precision:>12.3f}{recall:>10.3f}{f1:>8.3f}")

    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()