"""
attack/generate_extended_budgets.py

Extends the influence/random forget-set budget sweep from {50, 100, 200}
to also include {300, 400}, using the EXACT SAME selection algorithms as
attack/forget_set.py (deterministic top-k influence, seeded random).

Does NOT modify build_influence_forget_set or build_random_forget_set.
Does NOT touch or overwrite the existing forget_influence_{50,100,200}.npy
or forget_random_{50,100,200}.npy files — only writes the two new budgets.

Requires: attack/forget_set.py already in the project (unchanged).

Each new budget here = 2 new forget sets (1 influence, 1 random) = 2 new
oracle retrains needed before these can be used in CER/auditor analysis.
"""

import numpy as np
from forget_set import build_influence_forget_set, build_random_forget_set

NEW_BUDGETS = [300, 400]


if __name__ == "__main__":
    lt_pure_indices = np.load("data/lt_pure_indices.npy")
    influence_scores = np.load("data/influence_scores.npy")

    print(f"Loaded {len(lt_pure_indices)} pure long_tail indices, {len(influence_scores)} influence scores")
    print(f"Extending sweep with new budgets: {NEW_BUDGETS}")
    print("(existing budgets 50/100/200 are untouched by this script)\n")

    for budget in NEW_BUDGETS:
        inf_set = build_influence_forget_set(lt_pure_indices, influence_scores, budget)
        rand_set = build_random_forget_set(lt_pure_indices, budget)  # seed=42, same as original

        inf_path = f"data/forget_influence_{budget}.npy"
        rand_path = f"data/forget_random_{budget}.npy"

        np.save(inf_path, inf_set)
        np.save(rand_path, rand_set)

        print(f"Budget {budget}:")
        print(f"  influence set -> {inf_path}  (n={len(inf_set)}, idx range [{inf_set.min()}, {inf_set.max()}])")
        print(f"  random set    -> {rand_path}  (n={len(rand_set)}, idx range [{rand_set.min()}, {rand_set.max()}])")

    print(f"\nDone. {len(NEW_BUDGETS) * 2} new forget sets saved to data/.")
    print("Reminder: each of these needs a corresponding oracle retrain")
    print("(oracle_influence_300, oracle_influence_400, oracle_random_300, oracle_random_400)")
    print("before it can be used in compare_unlearning.py or CER analysis.")