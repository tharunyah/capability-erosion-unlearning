"""
attack/generate_random_repeats.py

Generates additional independent random-seed draws of the RANDOM-strategy
forget set at a FIXED budget, so the auditor test (test_auditor_multimethod.py)
has more than n=1 sample for the "benign" condition at that budget.

This does not touch build_random_forget_set or build_influence_forget_set.
It only calls the existing seeded random function with new seed values.

ASSUMPTION (change below if you want a different setup):
  - FIXED_BUDGET = 100 (the middle of the existing 50/100/200 sweep)
  - NEW_SEEDS = [43, 44, 45, 46], i.e. 4 additional draws on top of the
    existing seed=42 draw (forget_random_100.npy), giving n=5 total draws
    at this budget -- matching the "5-draw" convention already used for
    the fine_tune multi-draw analysis elsewhere in the project.

Does NOT address influence-side small-n: build_influence_forget_set is
deterministic top-k with no seed, so it cannot produce multiple distinct
"attack" draws at a fixed budget without changing the selection algorithm
itself (a decision explicitly deferred, see conversation notes).

Each new seed here = 1 new forget set = 1 new oracle retrain needed before
it can be used in CER/auditor analysis.
"""

import numpy as np
from forget_set import build_random_forget_set

FIXED_BUDGET = 100
NEW_SEEDS = [43, 44, 45, 46]


if __name__ == "__main__":
    lt_pure_indices = np.load("data/lt_pure_indices.npy")

    print(f"Loaded {len(lt_pure_indices)} pure long_tail indices")
    print(f"Fixed budget: {FIXED_BUDGET}")
    print(f"Existing draw: forget_random_{FIXED_BUDGET}.npy (seed=42)")
    print(f"New draws: seeds {NEW_SEEDS}\n")

    for seed in NEW_SEEDS:
        rand_set = build_random_forget_set(lt_pure_indices, FIXED_BUDGET, seed=seed)
        path = f"data/forget_random_{FIXED_BUDGET}_seed{seed}.npy"
        np.save(path, rand_set)
        print(f"  seed={seed} -> {path}  (n={len(rand_set)}, idx range [{rand_set.min()}, {rand_set.max()}])")

    print(f"\nDone. {len(NEW_SEEDS)} new forget sets saved to data/.")
    print(f"Total draws now available at budget={FIXED_BUDGET}: {1 + len(NEW_SEEDS)} (seed=42 + {NEW_SEEDS})")
    print("Reminder: each new seed needs its own oracle retrain")
    print(f"(oracle_random_{FIXED_BUDGET}_seed43, _seed44, _seed45, _seed46)")
    print("before it can feed into test_auditor_multimethod.py's benign-condition stats.")