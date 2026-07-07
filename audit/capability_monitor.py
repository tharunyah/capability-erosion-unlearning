"""
audit/capability_monitor.py

Capability-aware auditor -- Day 11, updated to work with the DI-based metric
from cer.py (see evaluate/cer.py for why we moved from CER to Disparate
Impact -- one unmodified, cited formula instead of a combined one).

Two checks, both borrowed from existing, established sources -- nothing invented:

  1. Threshold test -- the "four-fifths rule" (EEOC guideline, formalized in
     Feldman et al., 2015): a DI value below 0.8 is the established legal/ML
     cutoff for "disparate impact." We reuse 0.8 exactly, not a custom number.

  2. Outlier test -- a standard z-score against a rolling history of DI shifts
     seen from BENIGN (random) forget sets. If a new DI shift is far outside
     that normal range (>2 std devs), it's flagged as statistically unusual,
     even if it doesn't cross the 0.8 threshold outright.

check() returns: {tier, di_after, di_shift, flagged, reason}
  reason is one of: "threshold" (di_after < 0.8), "outlier" (z-score test), "ok"

--- HISTORY OF FIXES (per friend's review) ---

1. FIXED: history contamination. check() previously updated history on every
   call, including for runs being evaluated (e.g. attack runs). This meant
   checking attack #1 leaked into the baseline used to judge attack #2,
   drifting the "normal" mean toward attack-like values as you go through
   a loop.
   -> check() no longer updates history at all. History is ONLY ever
      updated via seed_benign_history() / add_benign_observation(), i.e.
      only from runs you explicitly know are benign ground truth. This
      makes the calibration set stable regardless of what/how many runs
      you evaluate afterward.

2. FIXED (silent failure): min_history_for_zscore defaulted to 5, but with
   only 3 benign samples per method (forget_random_{50,100,200}), Check B
   never ran, silently, forever -- no warning. check() now WARNS the first
   time it skips Check B due to insufficient history.

   NOTE (project-specific): budget=100 now has 5 benign draws available
   (forget_random_100 seed42 + seed43/44/45/46 from generate_random_repeats.py),
   so min_history_for_zscore=5 is back to being achievable there -- use 5
   when calibrating on budget=100, and the lower default only if working
   with budgets that still have just 3 draws.
"""

import warnings
import numpy as np


class CapabilityMonitor:
    def __init__(self, di_threshold=0.8, zscore_threshold=2.0, min_history_for_zscore=5):
        """
        di_threshold:      the four-fifths rule cutoff (Feldman et al., 2015).
                            Anything with di_after below this is flagged outright.
        zscore_threshold:  how many std-devs below the rolling mean di_shift
                            counts as a statistical outlier.
        min_history_for_zscore: need at least this many past (benign) di_shift
                            values for a tier before z-score is trustworthy.
                            Default is 5 -- matches the 5 benign draws now
                            available at budget=100 (seed42 + seed43-46).
                            Drop to 3 only if calibrating on a budget that
                            still has just the original 3 random draws.
        """
        self.di_threshold = di_threshold
        self.zscore_threshold = zscore_threshold
        self.min_history_for_zscore = min_history_for_zscore
        self.history = {}  # {tier_name: [di_shift, di_shift, ...]}  -- BENIGN ONLY
        self._warned_insufficient_history = set()  # tiers already warned about

    def check(self, di_after, di_shift, tier):
        """
        Runs both checks on a single unlearning run's DI result for one tier.
        Does NOT update history -- history is calibration data and must be
        added explicitly via seed_benign_history() / add_benign_observation(),
        so evaluating attack (or any non-ground-truth-benign) runs can never
        contaminate the baseline.
        """
        result = {
            "tier": tier,
            "di_after": float(di_after),
            "di_shift": float(di_shift),
            "flagged": False,
            "reason": "ok",
        }

        # Check A: four-fifths rule
        if di_after < self.di_threshold:
            result["flagged"] = True
            result["reason"] = "threshold"
            return result

        # Check B: z-score outlier vs. history of benign (random) shifts
        past_shifts = self.history.get(tier, [])
        if len(past_shifts) >= self.min_history_for_zscore:
            mean = np.mean(past_shifts)
            std = np.std(past_shifts)
            if std > 1e-8:
                z = (di_shift - mean) / std
                if z <= -self.zscore_threshold:  # erosion = very negative shift
                    result["flagged"] = True
                    result["reason"] = "outlier"
        else:
            if tier not in self._warned_insufficient_history:
                warnings.warn(
                    f"[CapabilityMonitor] tier='{tier}': only {len(past_shifts)} benign "
                    f"history sample(s), need {self.min_history_for_zscore} -- "
                    f"Check B (z-score outlier) is being SKIPPED for this tier. "
                    f"Threshold check (Check A) is the only active detector until "
                    f"more benign samples are seeded.",
                    stacklevel=2,
                )
                self._warned_insufficient_history.add(tier)

        return result

    def _update_history(self, di_shift, tier):
        self.history.setdefault(tier, []).append(float(di_shift))

    def seed_benign_history(self, di_shift_list, tier):
        """
        Feed a batch of di_shift values from KNOWN-benign (random) forget
        sets to build the detector's sense of "normal" before checking any
        real/attack runs. This is the ONLY way history gets populated.
        """
        for v in di_shift_list:
            self._update_history(v, tier)

    def add_benign_observation(self, di_shift, tier):
        """
        Add a single known-benign di_shift to history. Use this to
        incrementally grow calibration data (e.g. after evaluating an
        additional random-seed benign draw) without re-seeding the whole batch.
        """
        self._update_history(di_shift, tier)

    def check_batch(self, di_results, method="unknown", forget_set_id="unknown"):
        """
        Takes the dict from DisparateImpactAudit.compute() -- {tier: {...}} --
        and runs check() on every tier. Returns a list of annotated results.
        Does NOT touch history -- see check().
        """
        outputs = []
        for tier, vals in di_results.items():
            res = self.check(vals["di_after"], vals["di_shift"], tier)
            res["method"] = method
            res["forget_set_id"] = forget_set_id
            outputs.append(res)
        return outputs


if __name__ == "__main__":
    monitor = CapabilityMonitor(di_threshold=0.8, zscore_threshold=2.0, min_history_for_zscore=5)

    np.random.seed(0)
    print("-- seeding with benign (random forget set) DI shifts, small noise around 0 --")
    benign_shifts = np.random.normal(loc=-0.02, scale=0.03, size=8)
    monitor.seed_benign_history(benign_shifts, tier="long_tail")
    print("history:", monitor.history["long_tail"])

    print("\n-- checking a mildly-benign new run --")
    print(monitor.check(di_after=0.95, di_shift=-0.03, tier="long_tail"))

    print("\n-- checking an attack-magnitude run (big DI drop) --")
    print(monitor.check(di_after=0.65, di_shift=-0.30, tier="long_tail"))

    print("\n-- checking several attack runs in a row: history must NOT drift --")
    print("history before:", monitor.history["long_tail"])
    monitor.check(di_after=0.90, di_shift=-0.05, tier="long_tail")
    monitor.check(di_after=0.88, di_shift=-0.06, tier="long_tail")
    print("history after (should be UNCHANGED vs. before):", monitor.history["long_tail"])
