"""
experiments/compute_di_oracle_all_methods.py

Compares DI shift (and oracle gap, if available) across all three unlearning
methods: fine_tune_forget, fisher, gradient_ascent.

Handles the schema difference between:
- corrected/multidraw CSVs (fine_tune, fisher): use 'unlearned_acc_mean'
- raw CSV (gradient_ascent): uses 'unlearned_acc'

Read-only script. Writes only results/full_summary_di_oracle_all_methods.json.
Does not touch any existing CSV, .pt, or .json files.
"""

import json
import os
import numpy as np
import pandas as pd

TIERS = ["majority", "mid_tail", "long_tail", "safety_critical"]
REFERENCE_TIER = "majority"

FINE_TUNE_CSV = "results/finetune_results_multidraw_extended_steps50_corrected.csv"
FISHER_CSV = "results/fisher_results_multidraw_extended_alpha0.001_corrected.csv"
GRAD_ASCENT_CSV = "results/gradient_ascent_results.csv"
TAXONOMY_PATH = "data/capability_taxonomy.json"
ORACLE_PER_CLASS_ACC = "results/oracle_per_class_acc.npy"


def load_tiers():
    tax = json.load(open(TAXONOMY_PATH))
    tiers = {}
    for class_id_str, tier in tax.items():
        tiers.setdefault(tier, []).append(int(class_id_str))
    return tiers


def compute_oracle_tier_acc():
    """Returns {tier: mean_oracle_accuracy} using per-class oracle accuracies."""
    if not os.path.exists(ORACLE_PER_CLASS_ACC):
        print(f"[skip] oracle file not found at {ORACLE_PER_CLASS_ACC} — oracle_gap will be omitted")
        return None
    per_class_acc = np.load(ORACLE_PER_CLASS_ACC)
    tiers = load_tiers()
    oracle_tier_acc = {}
    for tier, class_ids in tiers.items():
        oracle_tier_acc[tier] = float(np.mean(per_class_acc[class_ids]))
    return oracle_tier_acc


def sanity_check_oracle(oracle_tier_acc, df, method_label):
    """Quick check: oracle accuracy shouldn't be wildly different from baseline,
    since the forget set is a tiny fraction of the full training set."""
    if oracle_tier_acc is None:
        return
    baseline_majority = df[df["tier"] == REFERENCE_TIER]["baseline_acc"].mean()
    oracle_majority = oracle_tier_acc.get(REFERENCE_TIER)
    if oracle_majority is None:
        return
    gap = abs(baseline_majority - oracle_majority)
    flag = "  <-- CHECK THIS, looks off" if gap > 0.05 else ""
    print(f"[sanity check: {method_label}] baseline majority_acc={baseline_majority:.4f} "
          f"vs oracle majority_acc={oracle_majority:.4f} (gap={gap:.4f}){flag}")


def load_generic(csv_path, method_label, oracle_tier_acc):
    """Auto-detects which accuracy column to use: 'unlearned_acc_mean' for
    corrected/multidraw CSVs, falls back to 'unlearned_acc' for raw CSVs."""
    if not os.path.exists(csv_path):
        print(f"[skip] {method_label}: {csv_path} not found")
        return []

    df = pd.read_csv(csv_path)

    if "unlearned_acc_mean" in df.columns:
        acc_col = "unlearned_acc_mean"
    elif "unlearned_acc" in df.columns:
        acc_col = "unlearned_acc"
    else:
        print(f"[error] {method_label}: no recognized accuracy column found. "
              f"Actual columns: {list(df.columns)}. Skipping — please confirm schema.")
        return []

    required = {"forget_set", "strategy", "budget", "tier", "baseline_acc"}
    missing = required - set(df.columns)
    if missing:
        print(f"[error] {method_label}: missing columns {missing}. "
              f"Actual columns: {list(df.columns)}. Skipping — please confirm schema.")
        return []

    sanity_check_oracle(oracle_tier_acc, df, method_label)

    rows = []
    for forget_set, group in df.groupby("forget_set"):
        tier_data = {r["tier"]: r for _, r in group.iterrows()}
        if REFERENCE_TIER not in tier_data:
            continue
        ref_before = tier_data[REFERENCE_TIER]["baseline_acc"]
        ref_after = tier_data[REFERENCE_TIER][acc_col]

        row = {
            "method": method_label,
            "config": f"{method_label}_{forget_set.replace('.npy', '')}",
        }
        for tier in TIERS:
            if tier not in tier_data:
                continue
            t_before = tier_data[tier]["baseline_acc"]
            t_after = tier_data[tier][acc_col]

            di_before = t_before / ref_before if ref_before else float("nan")
            di_after = t_after / ref_after if ref_after else float("nan")
            di_shift = di_after - di_before
            row[f"{tier}_di_shift"] = round(di_shift, 4)

            if oracle_tier_acc is not None and tier in oracle_tier_acc:
                oracle_gap = t_after - oracle_tier_acc[tier]
                row[f"{tier}_oracle_gap"] = round(oracle_gap, 4)

        rows.append(row)
    print(f"[loaded] {method_label}: {len(rows)} configs from {csv_path}")
    return rows


def print_method_comparison(result_df):
    """Prints mean DI shift per tier per method, side by side, so the
    cross-method comparison is immediately readable."""
    di_cols = [c for c in result_df.columns if c.endswith("_di_shift")]
    summary = result_df.groupby("method")[di_cols].mean().round(4)
    print("\n=== MEAN DI SHIFT BY METHOD (across all configs) ===")
    print(summary.to_string())

    print("\n=== MOST-DAMAGED TIER PER METHOD (by mean DI shift) ===")
    tier_cols = [c.replace("_di_shift", "") for c in di_cols if c != "majority_di_shift"]
    for method in summary.index:
        tier_means = {t: summary.loc[method, f"{t}_di_shift"] for t in tier_cols}
        worst_tier = min(tier_means, key=tier_means.get)
        print(f"  {method}: {worst_tier} ({tier_means[worst_tier]*100:.1f}%)")


def main():
    oracle_tier_acc = compute_oracle_tier_acc()
    if oracle_tier_acc:
        print(f"[oracle] per-tier oracle accuracy: "
              f"{ {k: round(v, 4) for k, v in oracle_tier_acc.items()} }")

    all_rows = []
    all_rows += load_generic(FINE_TUNE_CSV, "fine_tune", oracle_tier_acc)
    all_rows += load_generic(FISHER_CSV, "fisher", oracle_tier_acc)
    all_rows += load_generic(GRAD_ASCENT_CSV, "gradient_ascent", oracle_tier_acc)

    if not all_rows:
        print("[error] no rows loaded from any method — check file paths above")
        return

    result_df = pd.DataFrame(all_rows)
    print("\n=== ALL CONFIGS ===")
    print(result_df.to_string(index=False))

    print_method_comparison(result_df)

    os.makedirs("results", exist_ok=True)
    out_path = "results/full_summary_di_oracle_all_methods.json"
    result_df.to_json(out_path, orient="records", indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()