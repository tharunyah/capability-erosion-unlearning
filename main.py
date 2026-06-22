import numpy as np
from attack.forget_set import build_influence_forget_set, build_random_forget_set
import torchvision

scores           = np.load('data/influence_scores.npy')
lt_train_indices = np.load('data/lt_train_indices.npy')

ds       = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)

for budget in [50, 100, 200]:
    inf_set = build_influence_forget_set(lt_train_indices, scores, budget)
    rnd_set = build_random_forget_set(lt_train_indices, budget, seed=42)

    np.save(f'data/forget_influence_{budget}.npy', inf_set)
    np.save(f'data/forget_random_{budget}.npy',    rnd_set)

    # Show class distribution of influence forget set
    inf_classes = [ds.classes[trainset[i][1]] for i in inf_set]
    from collections import Counter
    top_classes = Counter(inf_classes).most_common(5)

    print(f"\nBudget {budget} — Influence forget set top classes: {top_classes}")

print("\nAll forget sets saved.")