import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_cifar100_trainset():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    return datasets.CIFAR100(
        root=os.environ.get('DATA_ROOT', 'data'),
        train=True, download=True, transform=transform
    )

def get_forget_retain_loaders(
    forget_indices,          # np.ndarray or list of ints
    batch_size=128,
    num_workers=2,
    trainset=None
):
    """
    Takes any array of forget indices.
    Returns (forget_loader, retain_loader).
    
    forget_loader  → only the samples to be forgotten
    retain_loader  → everything else in the training set
    """
    if trainset is None:
        trainset = get_cifar100_trainset()
    
    all_indices = set(range(len(trainset)))
    forget_set  = set(forget_indices)
    retain_set  = all_indices - forget_set

    # Sanity checks
    assert len(forget_set) > 0, "Forget set is empty"
    assert len(forget_set) < len(trainset), "Forget set is the entire dataset"
    assert forget_set.issubset(all_indices), "Some forget indices are out of range"

    forget_subset = Subset(trainset, sorted(forget_set))
    retain_subset = Subset(trainset, sorted(retain_set))

    forget_loader = DataLoader(
        forget_subset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    retain_loader = DataLoader(
        retain_subset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    print(f"Forget set size : {len(forget_subset)}")
    print(f"Retain set size : {len(retain_subset)}")
    print(f"Total           : {len(forget_subset) + len(retain_subset)}")

    return forget_loader, retain_loader
