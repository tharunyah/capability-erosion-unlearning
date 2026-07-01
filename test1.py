import numpy as np
import json

with open('data/capability_taxonomy.json') as f:
    class_to_tier = json.load(f)

# you need the train labels to map forget indices -> class ids
from torchvision import datasets
train_set = datasets.CIFAR100(root='data/', train=True, download=True)

for fname in ['forget_influence_50.npy', 'forget_influence_100.npy', 'forget_influence_200.npy',
              'forget_random_50.npy', 'forget_random_100.npy', 'forget_random_200.npy']:
    idx = np.load(f'data/{fname}')
    classes_hit = [train_set.targets[i] for i in idx]
    tiers_hit = [class_to_tier[str(c)] for c in classes_hit]
    from collections import Counter
    print(fname, Counter(tiers_hit))
