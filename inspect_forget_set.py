import numpy as np
import torchvision
from collections import Counter

forget = np.load("data/forget_set_placeholder.npy")

trainset = torchvision.datasets.CIFAR100(
    root="./data",
    train=True,
    download=True
)

labels = [trainset.targets[i] for i in forget]

counts = Counter(labels)

print("Classes in forget set:\n")

for cls, count in counts.most_common():
    print(
        f"{trainset.classes[cls]:15s} {count}"
    )
