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

UPDATE: each method (finetune, fisher) now loads BOTH its "extended" CSV
(budgets 100 w/ reseeds, 300, 400) AND its "supplement" CSV (budgets 50, 100, 200),
merged with dedup keyed on (budget, strategy) -- extended wins on overlap.
Previously this script only read one CSV per method, which silently dropped
most of fisher's influence-attack runs (2 out of 5) and finetune's 300/400 data.
gradient_ascent has one CSV with full budget range already, so it's unchanged.

Handles multiple CSV schemas automatically:
  - gradient_ascent_results.csv: has an "unlearned_acc" column
  - finetune/fisher supplement + extended CSVs: have "unlearned_acc_mean"
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
from collections import defaultdict
from audit.capability_monitor import CapabilityMonitor

# Each method now maps to a LIST of files: (path, source_label).
# Order matters for the dedup pass below -- put "extended" first so it
# always wins when both files cover the same (budget, strategy).
RESULT_FILES = {
    "gradient_ascent": [
        ("results/gradient_ascent_results.csv", "raw_single_run"),
    ],
    "finetune": [
        ("results/finetune_results_multidraw_extended_steps50_corrected.csv", "extended_corrected"),
        ("results/finetune_results_multidraw_steps50_corrected.csv", "supplement"),
    ],
    "fisher": [
        ("results/fisher_results_multidraw_extended_alpha0.001_corrected.csv", "extended_corrected"),
        ("results/fisher_results_multidraw_alpha0.001.csv", "supplement"),
    ],
}

TARGET_TIERS = ["majority", "mid_tail", "long_tail", "safety_critical"]
REFERENCE_TIER = "majority"  # NOTE: majority-vs-majority is a degenerate/sanity-check case (DI ~= 1.0 always)
OUTPUT_CSV = "results/auditor_test_results_multimethod.csv"


def get_unlearned_acc(row):
    """Handles all column naming schemes across your CSVs."""
    if "unlearned_acc" in row:
        return float(row["unlearned_acc"])
    if "unlearned_acc_mean" in row:
        return float(row["unlearned_acc_mean"])
    raise KeyError("No unlearned_acc / unlearned_acc_mean column found")


def load_one_csv(path, method):
    """Reads a single CSV. Returns by_run (run_id -> tier -> acc dict), meta (run_id -> info),
    and covered_keys (set of (budget, strategy) pairs seen in this file, for dedup)."""
    by_run = defaultdict(dict)
    meta = {}
    covered_keys = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            run_id = method + "::" + row["forget_set"]
            budget_val = row.get("budget", "")
            strategy_val = row["strategy"]
            by_run[run_id][row["tier"]] = {
                "baseline_acc": float(row["baseline_acc"]),
                "unlearned_acc": get_unlearned_acc(row),
            }
            meta[run_id] = {"method": method, "strategy": strategy_val, "budget": budget_val}
            covered_keys.add((budget_val, strategy_val))
    return by_run, meta, covered_keys


def load_results_for_method(method, file_list):
    """Merges multiple CSVs for one method. Files earlier in file_list win on
    (budget, strategy) overlap -- so put 'extended' before 'supplement'."""
    merged_by_run = defaultdict(dict)
    merged_meta = {}
    covered_keys = set()
    per_file_counts = []

    for path, source_label in file_list:
        if not os.path.exists(path):
            print("[skip] " + path + " not found yet")
            continue

        by_run, meta, file_keys = load_one_csv(path, method)

        new_keys = file_keys - covered_keys
        skipped_keys = file_keys & covered_keys

        added_runs = 0
        for run_id, tier_data in by_run.items():
            run_key = (meta[run_id]["budget"], meta[run_id]["strategy"])
            if run_key in new_keys or run_id not in merged_by_run:
                merged_by_run[run_id] = tier_data
                merged_meta[run_id] = meta[run_id]
                merged_meta[run_id]["source"] = source_label
                added_runs += 1

        per_file_counts.append(
            "  " + path + " (" + source_label + "): " + str(added_runs) + " runs added, "
            + str(len(skipped_keys)) + " (budget,strategy) cells already covered, skipped"
        )

        covered_keys |= file_keys

    print("[loaded] " + method + ": " + str(len(merged_by_run)) + " total runs")
    for line in per_file_counts:
        print(line)

    return merged_by_run, merged_meta


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
    print("")
    print("=" * 70)
    print("TARGET TIER: " + target_tier + "  (reference tier: " + REFERENCE_TIER + ")")
    print("=" * 70)
    header = "{:<40}{:<10}{:>10}{:>10}{:>10}".format("run_id", "strategy", "di_before", "di_after", "di_shift")
    print(header)

    for run_id, tier_data in all_by_run.items():
        try:
            di_before, di_after, di_shift = compute_di(tier_data, target_tier, REFERENCE_TIER)
        except KeyError:
            print("[warn] " + run_id + " missing tier data for '" + target_tier + "', skipping")
            continue
        di_per_run[run_id] = {"di_before": di_before, "di_after": di_after, "di_shift": di_shift, **all_meta[run_id]}
        line = "{:<40}{:<10}{:>10.3f}{:>10.3f}{:>10.3f}".format(
            run_id, all_meta[run_id]["strategy"], di_before, di_after, di_shift
        )
        print(line)

    benign_runs = [r for r, v in di_per_run.items() if v["strategy"] == "random"]
    attack_runs = [r for r, v in di_per_run.items() if v["strategy"] == "influence"]
    print("")
    print("Benign (random) runs: " + str(len(benign_runs)) + "   Attack (influence) runs: " + str(len(attack_runs)))

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
    for method, file_list in RESULT_FILES.items():
        by_run, meta = load_results_for_method(method, file_list)
        all_by_run.update(by_run)
        all_meta.update(meta)

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

    print("")
    print("=" * 70)
    print("PER-TIER SUMMARY")
    print("=" * 70)
    header = "{:<18}{:>5}{:>5}{:>5}{:>5}{:>12}{:>10}{:>8}".format(
        "tier", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1"
    )
    print(header)
    for tier, (tp, fp, tn, fn) in tier_totals.items():
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        line = "{:<18}{:>5}{:>5}{:>5}{:>5}{:>12.3f}{:>10.3f}{:>8.3f}".format(
            tier, tp, fp, tn, fn, precision, recall, f1
        )
        print(line)

    print("")
    print("Saved to " + OUTPUT_CSV)


if __name__ == "__main__":
    main()