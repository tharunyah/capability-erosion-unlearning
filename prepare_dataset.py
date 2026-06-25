# prepare_dataset.py
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision
import torchvision.transforms as transforms

# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# -------------------------------------------------------
# Directories (safe to re-run)
# -------------------------------------------------------
Path("data").mkdir(exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Load CIFAR-100
# -------------------------------------------------------
print("Loading CIFAR-100 via torchvision...")
train_dataset = torchvision.datasets.CIFAR100(
    root="./data/cifar100", train=True, download=True,
    transform=transforms.ToTensor()
)
print(f"Train samples : {len(train_dataset)}")

num_classes = 100
train_labels = np.array(train_dataset.targets)
class_to_indices = {
    i: np.where(train_labels == i)[0].tolist()
    for i in range(num_classes)
}

# -------------------------------------------------------
# Capability Taxonomy — LOAD, DO NOT REGENERATE
# Tier membership is fixed and must never be reshuffled.
# Only sample quotas change in this script.
# -------------------------------------------------------
with open("data/capability_taxonomy.json", "r") as f:
    capability_taxonomy: dict[str, str] = json.load(f)

majority_classes        = sorted(int(c) for c, t in capability_taxonomy.items() if t == "majority")
mid_tail_classes        = sorted(int(c) for c, t in capability_taxonomy.items() if t == "mid_tail")
long_tail_classes       = sorted(int(c) for c, t in capability_taxonomy.items() if t == "long_tail")
safety_critical_classes = sorted(int(c) for c, t in capability_taxonomy.items() if t == "safety_critical")

tier_freq = Counter(capability_taxonomy.values())
print("\nCapability Taxonomy (loaded, unchanged):")
for tier in ("majority", "mid_tail", "long_tail", "safety_critical"):
    print(f"  {tier:<18s}: {tier_freq[tier]:>3} classes")

# -------------------------------------------------------
# Pareto-distributed sample quotas
#
# long_tail:        50-120
# safety_critical:  30-70 (deliberately lower, models a
#                    smaller training budget than long_tail)
# -------------------------------------------------------
def pareto_quotas(n: int, lo: int, hi: int) -> np.ndarray:
    raw = rng.pareto(a=1.5, size=n)
    raw = np.sort(raw)[::-1]
    v_lo, v_hi = raw.min(), raw.max()
    if v_lo == v_hi:
        return np.full(n, (lo + hi) // 2, dtype=int)
    normed = (raw - v_lo) / (v_hi - v_lo)
    return (normed * (hi - lo) + lo).astype(int)

majority_quotas         = pareto_quotas(len(majority_classes), 400, 500)
mid_tail_quotas         = pareto_quotas(len(mid_tail_classes), 100, 400)
long_tail_quotas        = pareto_quotas(len(long_tail_classes), 50, 120)
safety_critical_quotas  = pareto_quotas(len(safety_critical_classes), 30, 70)

class_quota: dict[int, int] = {}
for cls, q in zip(majority_classes, majority_quotas):               class_quota[cls] = int(q)
for cls, q in zip(mid_tail_classes, mid_tail_quotas):                class_quota[cls] = int(q)
for cls, q in zip(long_tail_classes, long_tail_quotas):              class_quota[cls] = int(q)
for cls, q in zip(safety_critical_classes, safety_critical_quotas): class_quota[cls] = int(q)

# -------------------------------------------------------
# Sample indices
# -------------------------------------------------------
selected_indices: list[int] = []
for cls in range(num_classes):
    sampled = random.sample(class_to_indices[cls], class_quota[cls])
    selected_indices.extend(sampled)

selected_indices_arr = np.array(selected_indices)
np.save("data/lt_train_indices.npy", selected_indices_arr)
print(f"\nTotal sampled indices : {len(selected_indices_arr)}")
print("Saved -> data/lt_train_indices.npy")

# -------------------------------------------------------
# Verification summary
# -------------------------------------------------------
print("\nTier verification:")
for tier_name, classes in [
    ("majority", majority_classes),
    ("mid_tail", mid_tail_classes),
    ("long_tail", long_tail_classes),
    ("safety_critical", safety_critical_classes),
]:
    qs = [class_quota[c] for c in classes]
    print(f"  {tier_name:<18s}  n={len(classes):>2}  "
          f"quota range [{min(qs):>3}, {max(qs):>3}]  mean = {np.mean(qs):.1f}")

# -------------------------------------------------------
# Figure: sample distribution by tier
# -------------------------------------------------------
TIER_COLOR = {
    "majority": "#2196F3",
    "mid_tail": "#4CAF50",
    "long_tail": "#FF9800",
    "safety_critical": "#F44336",
}

sorted_by_quota = sorted(range(num_classes), key=lambda c: class_quota[c], reverse=True)
sorted_counts = [class_quota[c] for c in sorted_by_quota]
sorted_colors = [TIER_COLOR[capability_taxonomy[str(c)]] for c in sorted_by_quota]

legend_patches = [
    mpatches.Patch(color=TIER_COLOR["majority"], label="Majority (20 cls, 400-500)"),
    mpatches.Patch(color=TIER_COLOR["mid_tail"], label="Mid-tail (40 cls, 100-400)"),
    mpatches.Patch(color=TIER_COLOR["long_tail"], label="Long-tail (30 cls, 50-120)"),
    mpatches.Patch(color=TIER_COLOR["safety_critical"], label="Safety-critical (10 cls, 30-70)"),
]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(num_classes), sorted_counts, color=sorted_colors, width=1.0)
ax.set_xlabel("Classes (sorted by sample count)")
ax.set_ylabel("Number of Training Samples")
ax.set_title("Long-Tail CIFAR-100 Dataset Construction")
ax.legend(handles=legend_patches, fontsize=8.5, loc="upper right")
plt.tight_layout()
plt.savefig("results/figures/long_tail_distribution.png", dpi=150, bbox_inches="tight")
print("\nSaved -> results/figures/long_tail_distribution.png")
