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
"""

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
        """
        self.di_threshold = di_threshold
        self.zscore_threshold = zscore_threshold
        self.min_history_for_zscore = min_history_for_zscore
        self.history = {}  # {tier_name: [di_shift, di_shift, ...]}

    def check(self, di_after, di_shift, tier):
        """
        Runs both checks on a single unlearning run's DI result for one tier.
        Updates rolling history AFTER checking, so the current value doesn't
        skew its own z-score.
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
            self._update_history(di_shift, tier)
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

        self._update_history(di_shift, tier)
        return result

    def _update_history(self, di_shift, tier):
        self.history.setdefault(tier, []).append(float(di_shift))

    def seed_benign_history(self, di_shift_list, tier):
        """
        Day 12 step: feed a batch of di_shift values from KNOWN-benign
        (random) forget sets to build the detector's sense of "normal"
        before checking any real/attack runs.
        """
        for v in di_shift_list:
            self._update_history(v, tier)

    def check_batch(self, di_results, method="unknown", forget_set_id="unknown"):
        """
        Takes the dict from DisparateImpactAudit.compute() -- {tier: {...}} --
        and runs check() on every tier. Returns a list of annotated results.
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
