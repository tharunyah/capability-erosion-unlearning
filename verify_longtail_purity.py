# verify_longtail_purity.py
import json
import numpy as np
from torchvision import datasets, transforms

with open('data/capability_taxonomy.json') as f:
    taxonomy = json.load(f)

longtail_classes = set(int(c) for c, t in taxonomy.items() if t == 'long_tail')
other_classes    = set(int(c) for c, t in taxonomy.items() if t != 'long_tail')

trainset = datasets.CIFAR100(root='./data', train=True, download=False,
                              transform=transforms.ToTensor())
targets = trainset.targets

def audit(path, label):
    indices = np.load(path)
    classes_present = set(targets[i] for i in indices)

    bad = classes_present & other_classes
    good = classes_present & longtail_classes

    print(f"\n{label}  ({path})")
    print(f"  Samples: {len(indices)}")
    print(f"  Distinct classes present: {len(classes_present)}")
    print(f"  Long_tail classes covered: {len(good)}/{len(longtail_classes)}")
    if bad:
        print(f"  ❌ CONTAMINATION: {len(bad)} non-long_tail classes found: {sorted(bad)}")
    else:
        print(f"  ✅ Clean — every sample belongs to a long_tail class")

audit('data/lt_pure_indices.npy', 'Pure long-tail training indices')

for budget in [50, 100, 200]:
    audit(f'data/forget_influence_{budget}.npy', f'Influence forget set (budget={budget})')
    audit(f'data/forget_random_{budget}.npy',    f'Random forget set (budget={budget})')