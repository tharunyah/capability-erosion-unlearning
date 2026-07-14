"""
experiments/compute_di_all_methods.py

Compares DI shift across all three unlearning methods: fine_tune_forget,
fisher, gradient_ascent -- combining extended/corrected CSVs (best data,
preferred) with older multidraw CSVs (used only to fill budget gaps:
50 and 200, not already covered by the extended CSVs).

No oracle comparison (confirmed unnecessary by team decision).

Read-only script. Writes only results/full_summary_di_all_methods.json.
Does not touch any existing CSV, .pt, or .json files.
"""

import os
import pandas as pd

TIERS = ["majority", "mid_tail", "long_tail", "safety_critical"]
REFERENCE_TIER = "majority"

# Preferred (extended, corrected, best data) sources
FINE_TUNE_EXTENDED_CSV = "results/finetune_results_multidraw_extended_steps50_corrected.csv"
FISHER_EXTENDED_CSV = "results/fisher_results_multidraw_extended_alpha0.001_corrected.csv"

# Supplementary sources -- used ONLY to fill budgets not present in the extended CSVs
FINE_TUNE_SUPPLEMENT_CSV = "results/finetune_results_multidraw_steps50_corrected.csv"
FISHER_SUPPLEMENT_CSV = "results/fisher_results_multidraw_alpha0.001.csv"  # needs diff_over_std computed

GRAD_ASCENT_CSV = "results/gradient_ascent_results.csv"


def get_acc_col(df):
    if "unlearned_acc_mean" in df.columns:
        return "unlearned_acc_mean"
    elif "unlearned_acc" in df.columns:
        return "unlearned_acc"
    return None


def ensure_diff_over_std(df):
    """If the CSV has unlearned_acc_std but no diff_over_std, compute it."""
    if "diff_over_std" not in df.columns and "unlearned_acc_std" in df.columns and "diff" in df.columns:
        df = df.copy()
        df["diff_over_std"] = df["diff"] / df["unlearned_acc_std"].replace(0, float("nan"))
        print("  (computed diff_over_std locally -- source CSV didn't have it)")
    return df


def rows_from_df(df, method_label, source_tag):
    acc_col = get_acc_col(df)
    if acc_col is None:
        print("[error] " + method_label + " (" + source_tag + "): no accuracy column found. "
              + "Columns: " + str(list(df.columns)))
        return []

    required = {"forget_set", "strategy", "budget", "tier", "baseline_acc"}
    missing = required - set(df.columns)
    if missing:
        print("[error] " + method_label + " (" + source_tag + "): missing " + str(missing))
        return []

    rows = []
    for forget_set, group in df.groupby("forget_set"):
        tier_data = {}
        for _, r in group.iterrows():
            tier_data[r["tier"]] = r

        if REFERENCE_TIER not in tier_data:
            continue
        ref_before = tier_data[REFERENCE_TIER]["baseline_acc"]
        ref_after = tier_data[REFERENCE_TIER][acc_col]
        budget = tier_data[REFERENCE_TIER]["budget"]
        strategy = tier_data[REFERENCE_TIER]["strategy"]

        row = {
            "method": method_label,
            "config": method_label + "_" + forget_set.replace(".npy", ""),
            "budget": budget,
            "strategy": strategy,
            "source": source_tag,
        }
        for tier in TIERS:
            if tier not in tier_data:
                continue
            t_before = tier_data[tier]["baseline_acc"]
            t_after = tier_data[tier][acc_col]
            di_before = t_before / ref_before if ref_before else float("nan")
            di_after = t_after / ref_after if ref_after else float("nan")
            row[tier + "_di_shift"] = round(di_after - di_before, 4)

        rows.append(row)
    return rows


def load_method(extended_path, supplement_path, method_label):
    """Loads extended CSV (preferred), then adds supplement rows only for
    budgets NOT already present in the extended data."""
    all_rows = []
    covered_budgets = set()

    if os.path.exists(extended_path):
        df = pd.read_csv(extended_path)
        rows = rows_from_df(df, method_label, "extended_corrected")
        all_rows += rows
        covered_budgets = {r["budget"] for r in rows}
        print("[loaded] " + method_label + " (extended): " + str(len(rows))
              + " configs, budgets=" + str(sorted(covered_budgets)))
    else:
        print("[skip] " + method_label + " extended CSV not found: " + extended_path)

    if os.path.exists(supplement_path):
        df = pd.read_csv(supplement_path)
        df = ensure_diff_over_std(df)
        supplement_rows = rows_from_df(df, method_label, "supplement")
        new_rows = [r for r in supplement_rows if r["budget"] not in covered_budgets]
        skipped = len(supplement_rows) - len(new_rows)
        all_rows += new_rows
        print("[loaded] " + method_label + " (supplement): " + str(len(new_rows))
              + " new configs added (budgets " + str(sorted({r["budget"] for r in new_rows}))
              + "), " + str(skipped) + " skipped as already covered by extended data")
    else:
        print("[skip] " + method_label + " supplement CSV not found: " + supplement_path)

    return all_rows


def load_gradient_ascent(csv_path):
    if not os.path.exists(csv_path):
        print("[skip] gradient_ascent: " + csv_path + " not found")
        return []
    df = pd.read_csv(csv_path)
    rows = rows_from_df(df, "gradient_ascent", "raw_single_run")
    print("[loaded] gradient_ascent: " + str(len(rows)) + " configs from " + csv_path)
    return rows


def print_method_comparison(result_df):
    di_cols = [c for c in result_df.columns if c.endswith("_di_shift")]
    summary = result_df.groupby("method")[di_cols].mean().round(4)
    print("\n=== MEAN DI SHIFT BY METHOD (across all configs, all budgets) ===")
    print(summary.to_string())

    print("\n=== MOST-DAMAGED TIER PER METHOD ===")
    tier_cols = [c.replace("_di_shift", "") for c in di_cols if c != "majority_di_shift"]
    for method in summary.index:
        tier_means = {t: summary.loc[method, t + "_di_shift"] for t in tier_cols}
        worst_tier = min(tier_means, key=tier_means.get)
        print("  " + method + ": " + worst_tier + " (" + format(tier_means[worst_tier] * 100, ".1f") + "%)")

    print("\n=== BUDGET COVERAGE PER METHOD ===")
    for method in result_df["method"].unique():
        budgets = sorted(result_df[result_df["method"] == method]["budget"].unique())
        print("  " + method + ": " + str(budgets))


def main():
    all_rows = []
    all_rows += load_method(FINE_TUNE_EXTENDED_CSV, FINE_TUNE_SUPPLEMENT_CSV, "fine_tune")
    all_rows += load_method(FISHER_EXTENDED_CSV, FISHER_SUPPLEMENT_CSV, "fisher")
    all_rows += load_gradient_ascent(GRAD_ASCENT_CSV)

    if not all_rows:
        print("[error] no rows loaded -- check paths above")
        return

    result_df = pd.DataFrame(all_rows)
    print("\n=== ALL CONFIGS ===")
    print(result_df.to_string(index=False))

    print_method_comparison(result_df)

    os.makedirs("results", exist_ok=True)
    out_path = "results/full_summary_di_all_methods.json"
    result_df.to_json(out_path, orient="records", indent=2)
    print("\nSaved to " + out_path)


if __name__ == "__main__":
    main()