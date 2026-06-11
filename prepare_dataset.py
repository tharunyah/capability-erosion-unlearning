import json
import random
import numpy as np
from torchvision.datasets import CIFAR100

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("Downloading CIFAR-100...")

train_dataset = CIFAR100(
    root="./data",
    train=True,
    download=True
)

test_dataset = CIFAR100(
    root="./data",
    train=False,
    download=True
)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {len(train_dataset.classes)}")
