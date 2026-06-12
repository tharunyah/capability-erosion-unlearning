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
# Load CIFAR-100 via torchvision  (matches Day 2 plan)
# -------------------------------------------------------
print("Loading CIFAR-100 via torchvision...")
train_dataset = torchvision.datasets.CIFAR100(
    root="./data/cifar100", train=True,  download=True,
    transform=transforms.ToTensor()
)
test_dataset = torchvision.datasets.CIFAR100(
    root="./data/cifar100", train=False, download=True,
    transform=transforms.ToTensor()
)
print(f"Train samples : {len(train_dataset)}")
print(f"Test  samples : {len(test_dataset)}")

num_classes = 100

# -------------------------------------------------------
# Build class → index mapping  (fast: uses .targets)
# torchvision CIFAR100 exposes .targets as a plain list
# so we avoid iterating over 50k samples one by one
# -------------------------------------------------------
train_labels = np.array(train_dataset.targets)   # shape (50000,)
class_to_indices = {
    i: np.where(train_labels == i)[0].tolist()
    for i in range(num_classes)
}
print(f"\nClass index mapping built.")
print(f"  e.g. class 0 → {len(class_to_indices[0])} samples in full set")

# -------------------------------------------------------
# Capability Taxonomy
# -------------------------------------------------------
all_classes = list(range(num_classes))
random.shuffle(all_classes)

majority_classes        = sorted(all_classes[:20])
mid_tail_classes        = sorted(all_classes[20:60])
long_tail_classes       = sorted(all_classes[60:])          # 40 classes
safety_critical_classes = sorted(random.sample(long_tail_classes, 4))
safety_critical_set     = set(safety_critical_classes)

capability_taxonomy: dict[str, str] = {}
for cls in majority_classes:        capability_taxonomy[str(cls)] = "majority"
for cls in mid_tail_classes:        capability_taxonomy[str(cls)] = "mid_tail"
for cls in long_tail_classes:       capability_taxonomy[str(cls)] = "long_tail"
for cls in safety_critical_classes: capability_taxonomy[str(cls)] = "safety_critical"
# ↑ safety_critical intentionally overrides long_tail (it's a sub-category)

tier_freq = Counter(capability_taxonomy.values())
print("\nCapability Taxonomy:")
for tier in ("majority", "mid_tail", "long_tail", "safety_critical"):
    print(f"  {tier:<18s}: {tier_freq[tier]:>3} classes")

with open("data/capability_taxonomy.json", "w") as f:
    json.dump(capability_taxonomy, f, indent=4)
print("Saved  →  data/capability_taxonomy.json")

# -------------------------------------------------------
# Pareto-distributed sample quotas  (per Day 2 spec)
#
# Why Pareto?  A Pareto / power-law distribution is the
# canonical model of long-tail phenomena (Zipf's law,
# natural frequency distributions, etc.).  Using it here
# means the intra-tier variation is itself power-law
# distributed, which is a stronger and more defensible
# claim in the methodology section than fixed steps.
# -------------------------------------------------------
def pareto_quotas(n: int, lo: int, hi: int) -> np.ndarray:
    """
    Return n integer quotas in [lo, hi] whose distribution
    follows a Pareto power-law (shape a=1.5).
    Sorted descending so the 'heaviest' count comes first —
    this creates a smooth within-tier gradient.
    """
    raw = rng.pareto(a=1.5, size=n)
    raw = np.sort(raw)[::-1]                # descending
    v_lo, v_hi = raw.min(), raw.max()
    if v_lo == v_hi:                        # edge case: all equal
        return np.full(n, (lo + hi) // 2, dtype=int)
    normed = (raw - v_lo) / (v_hi - v_lo)
    return (normed * (hi - lo) + lo).astype(int)

majority_quotas   = pareto_quotas(len(majority_classes),  400, 500)
mid_tail_quotas   = pareto_quotas(len(mid_tail_classes),  100, 400)
long_tail_quotas  = pareto_quotas(len(long_tail_classes),  20,  99)

class_quota: dict[int, int] = {}
for cls, q in zip(majority_classes,   majority_quotas):  class_quota[cls] = int(q)
for cls, q in zip(mid_tail_classes,   mid_tail_quotas):  class_quota[cls] = int(q)
for cls, q in zip(long_tail_classes,  long_tail_quotas): class_quota[cls] = int(q)

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
print("Saved  →  data/lt_train_indices.npy")

# -------------------------------------------------------
# Verification summary  (no expensive re-read loop)
# -------------------------------------------------------
print("\nTier verification:")
for tier_name, classes in [
    ("majority",        majority_classes),
    ("mid_tail",        mid_tail_classes),
    ("long_tail",       [c for c in long_tail_classes if c not in safety_critical_set]),
    ("safety_critical", safety_critical_classes),
]:
    qs = [class_quota[c] for c in classes]
    print(f"  {tier_name:<18s}  n={len(classes):>2}  "
          f"quota range [{min(qs):>3}, {max(qs):>3}]  "
          f"mean = {np.mean(qs):.1f}")

# -------------------------------------------------------
# Paper-quality figure  (two panels)
#
# Panel A — sorted descending: shows the characteristic
#            power-law / long-tail shape.  This is the
#            canonical way to present imbalanced datasets
#            in ML papers.
#
# Panel B — by class ID: shows the random tier assignment,
#            confirming there's no systematic ordering bias.
# -------------------------------------------------------
TIER_COLOR = {
    "majority":        "#2196F3",   # blue
    "mid_tail":        "#4CAF50",   # green
    "long_tail":       "#FF9800",   # orange
    "safety_critical": "#F44336",   # red
}

sorted_by_quota = sorted(range(num_classes),
                          key=lambda c: class_quota[c], reverse=True)
sorted_counts  = [class_quota[c] for c in sorted_by_quota]
sorted_colors  = [TIER_COLOR[capability_taxonomy[str(c)]] for c in sorted_by_quota]

legend_patches = [
    mpatches.Patch(color=TIER_COLOR["majority"],
                   label="Majority   (20 cls, 400–500 samples)"),
    mpatches.Patch(color=TIER_COLOR["mid_tail"],
                   label="Mid-tail   (40 cls, 100–400 samples)"),
    mpatches.Patch(color=TIER_COLOR["long_tail"],
                   label="Long-tail  (36 cls,  20–99 samples)"),
    mpatches.Patch(color=TIER_COLOR["safety_critical"],
                   label="Safety-critical (4 cls, 20–99 samples)"),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Long-Tail CIFAR-100 Dataset Construction",
             fontsize=14, fontweight="bold", y=1.01)

# — Panel A: sorted
ax = axes[0]
ax.bar(range(num_classes), sorted_counts,
       color=sorted_colors, width=1.0, edgecolor="none")
ax.axvline(x=19.5, color="gray", ls="--", lw=1.2, alpha=0.7)
ax.axvline(x=59.5, color="gray", ls="--", lw=1.2, alpha=0.7)
ax.set_xlabel("Classes (sorted by sample count)", fontsize=11)
ax.set_ylabel("Number of Training Samples",        fontsize=11)
ax.set_title("(a) Sorted Distribution",            fontsize=12)
ax.text( 9, max(sorted_counts) * 0.93, "Majority",  ha="center", fontsize=9, color="#0D47A1")
ax.text(39, max(sorted_counts) * 0.93, "Mid-tail",  ha="center", fontsize=9, color="#1B5E20")
ax.text(79, max(sorted_counts) * 0.93, "Long-tail", ha="center", fontsize=9, color="#E65100")
ax.legend(handles=legend_patches, fontsize=8.5, loc="upper right")

# — Panel B: by class ID
ax2 = axes[1]
counts_by_id = [class_quota[c] for c in range(num_classes)]
colors_by_id = [TIER_COLOR[capability_taxonomy[str(c)]] for c in range(num_classes)]
ax2.bar(range(num_classes), counts_by_id,
        color=colors_by_id, width=1.0, edgecolor="none")
ax2.set_xlabel("Class ID",                              fontsize=11)
ax2.set_ylabel("Number of Training Samples",            fontsize=11)
ax2.set_title("(b) By Class ID (random tier assignment)", fontsize=12)
ax2.legend(handles=legend_patches, fontsize=8.5, loc="upper right")

plt.tight_layout()
plt.savefig("results/figures/long_tail_distribution.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved  →  results/figures/long_tail_distribution.png")
