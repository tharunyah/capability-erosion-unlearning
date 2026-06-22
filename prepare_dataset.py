# prepare_dataset.py  
import numpy as np
import json
from torchvision import datasets

np.random.seed(42)

def main():
    trainset = datasets.CIFAR100(root='data/', train=True, download=True)
    labels = np.array(trainset.targets)

    # Bucket assignment — same 3-tier Pareto structure, raised floor
    all_classes = np.arange(100)
    np.random.shuffle(all_classes)

    majority_classes   = all_classes[:20]   # 400-500 samples
    mid_tail_classes   = all_classes[20:60] # 100-400 samples
    long_tail_classes  = all_classes[60:90] # RAISED: 50-120 (was 20-99)
    safety_classes     = all_classes[90:100] # treat as its own tier, also raised floor

    quota = {}
    for c in majority_classes:
        quota[c] = np.random.randint(400, 501)
    for c in mid_tail_classes:
        quota[c] = np.random.randint(100, 401)
    for c in long_tail_classes:
        quota[c] = np.random.randint(50, 121)   # raised floor
    for c in safety_classes:
        quota[c] = np.random.randint(50, 121)   # raised floor

    selected_indices = []
    class_to_tier = {}

    tier_map = {}
    for c in majority_classes: tier_map[c] = "majority"
    for c in mid_tail_classes: tier_map[c] = "mid_tail"
    for c in long_tail_classes: tier_map[c] = "long_tail"
    for c in safety_classes: tier_map[c] = "safety_critical"

    for c in range(100):
        class_indices = np.where(labels == c)[0]
        n = quota[c]
        chosen = np.random.choice(class_indices, size=n, replace=False)
        selected_indices.extend(chosen.tolist())
        class_to_tier[str(c)] = tier_map[c]

    selected_indices = np.array(selected_indices)
    np.save('data/lt_train_indices.npy', selected_indices)

    with open('data/capability_taxonomy.json', 'w') as f:
        json.dump(class_to_tier, f, indent=2)

    # Print histogram to verify
    print(f"Total selected samples: {len(selected_indices)}")
    for tier in ["majority", "mid_tail", "long_tail", "safety_critical"]:
        classes = [int(c) for c, t in class_to_tier.items() if t == tier]
        counts = [quota[c] for c in classes]
        print(f"{tier:20s} | classes: {len(classes):3d} | min: {min(counts):4d} | max: {max(counts):4d} | total: {sum(counts):5d}")


if __name__ == '__main__':
    main()
