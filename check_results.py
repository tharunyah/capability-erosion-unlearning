"""
check_results.py

Quick diagnostic to see exactly what your fine-tune and Fisher unlearning
results actually say, across all the CSV files you've generated so far.

Run from repo root:
    python check_results.py

What it does:
1. Finds every results CSV it recognizes (original single-run, extended
   multidraw, corrected-extended). Skips any that don't exist.
2. Prints columns + shape for each, so you can see what's actually in each
   file (schemas may differ between original and corrected-extended).
3. Tries to build a "which tier got hit hardest" summary per config, using
   whatever damage/z-score column it can find (diff_over_std, z_score, or
   a raw accuracy delta if that's all that's there).
4. Prints a final side-by-side: fine-tune vs Fisher, ranked by how often
   each tier is the most-damaged one across all configs.

This does NOT touch your models, checkpoints, or existing scripts. Read-only.
"""

import pandas as pd
import glob
import os

# Candidate files — adjust paths here if yours live somewhere else
CANDIDATES = {
    "finetune_original": "results/finetune_results.csv",
    "fisher_original": "results/fisher_results.csv",
    "finetune_multidraw": "results/finetune_results_multidraw_steps50.csv",
    "fisher_multidraw": "results/fisher_results_multidraw_alpha0.001.csv",
    "finetune_multidraw_corrected": "results/finetune_results_multidraw_steps50_corrected.csv",
    "finetune_extended": "results/finetune_results_multidraw_extended_steps50.csv",
    "fisher_extended": "results/fisher_results_multidraw_extended_alpha0.001.csv",
    "finetune_corrected": "results/finetune_results_multidraw_extended_steps50_corrected.csv",
    "fisher_corrected": "results/fisher_results_multidraw_extended_alpha0.001_corrected.csv",
    "finetune_from_saved_draws": "results/finetune_results_from_saved_draws_steps50.csv",
    "fisher_from_saved_draws": "results/fisher_results_from_saved_draws_alpha0.001.csv",
}

loaded = {}
print("=" * 70)
print("FILE DISCOVERY")
print("=" * 70)
for name, path in CANDIDATES.items():
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        loaded[name] = df
        print(f"[FOUND]   {name:22s} -> {path}  (shape={df.shape})")
        print(f"          columns: {list(df.columns)}")
    else:
        print(f"[MISSING] {name:22s} -> {path}")
print()

# Candidate column names for "how much damage", in priority order.
# diff_over_std is the noise-corrected metric from the *_corrected.csv files —
# prefer it whenever it's present, since raw diff can be misleading noise.
DAMAGE_COLS = ["diff_over_std", "z_score", "z", "diff", "acc_delta", "accuracy_delta"]
TIER_COL_CANDIDATES = ["tier", "capability_tier", "capability_group"]
CONFIG_COL_CANDIDATES = ["config_tag", "config", "forget_set", "experiment"]


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def summarize(name, df):
    print("-" * 70)
    print(f"SUMMARY: {name}")
    print("-" * 70)
    tier_col = find_col(df, TIER_COL_CANDIDATES)
    config_col = find_col(df, CONFIG_COL_CANDIDATES)
    damage_col = find_col(df, DAMAGE_COLS)

    if not tier_col or not damage_col:
        print("  Could not auto-detect tier/damage columns. Raw preview:")
        print(df.head(10).to_string())
        print()
        return

    print(f"  (using damage column: '{damage_col}')")
    group_keys = [config_col, tier_col] if config_col else [tier_col]
    if config_col:
        for cfg, sub in df.groupby(config_col):
            sub_sorted = sub.reindex(sub[damage_col].abs().sort_values(ascending=False).index)
            worst = sub_sorted.iloc[0]
            print(f"  config={cfg:20s} | most-damaged tier = {worst[tier_col]:16s} "
                  f"| {damage_col}={worst[damage_col]:.3f}")
    else:
        sub_sorted = df.reindex(df[damage_col].abs().sort_values(ascending=False).index)
        worst = sub_sorted.iloc[0]
        print(f"  most-damaged tier overall = {worst[tier_col]} | {damage_col}={worst[damage_col]:.3f}")
    print()

    return tier_col, damage_col, config_col


print("=" * 70)
print("PER-FILE TIER-DAMAGE SUMMARY")
print("=" * 70)
results_meta = {}
for name, df in loaded.items():
    meta = summarize(name, df)
    if meta:
        results_meta[name] = meta

print("=" * 70)
print("CROSS-METHOD COMPARISON: which tier gets hit hardest, how often")
print("=" * 70)
for method_prefix in ["finetune", "fisher"]:
    counts = {}
    for name, df in loaded.items():
        if not name.startswith(method_prefix):
            continue
        if name not in results_meta:
            continue
        tier_col, damage_col, config_col = results_meta[name]
        if config_col:
            for cfg, sub in df.groupby(config_col):
                worst_tier = sub.loc[sub[damage_col].abs().idxmax(), tier_col]
                counts[worst_tier] = counts.get(worst_tier, 0) + 1
        else:
            worst_tier = df.loc[df[damage_col].abs().idxmax(), tier_col]
            counts[worst_tier] = counts.get(worst_tier, 0) + 1
    print(f"\n{method_prefix.upper()}: most-damaged-tier counts across all configs found")
    if not counts:
        print("  (no usable data found for this method)")
    for tier, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {tier:16s} : most-damaged in {n} config(s)")

print("\nDone. If a file showed 'Could not auto-detect' above, paste its")
print("actual column names back and I'll hardcode the right ones for you.")