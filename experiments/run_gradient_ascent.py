# experiments/run_gradient_ascent.py
import sys
import os

# Add project root to Python's path so 'data' and 'unlearn' imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.dataloader_utils import get_forget_retain_loaders
from unlearn.gradient_ascent import gradient_ascent_unlearn


def load_model(path, num_classes=100):
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def per_class_accuracy(model, loader, num_classes=100, device='cpu'):
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for c in range(num_classes):
                mask = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum()
                total[c] += mask.sum()
    return (correct / total.clamp(min=1)).numpy()


def load_taxonomy_by_tier(path):
    with open(path, 'r') as f:
        class_to_tier = json.load(f)
    tier_to_classes = {}
    for class_id_str, tier in class_to_tier.items():
        tier_to_classes.setdefault(tier, []).append(int(class_id_str))
    return tier_to_classes


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    forget_indices = np.load('data/forget_set_placeholder.npy')
    print(f"Loaded placeholder forget set: {len(forget_indices)} indices")

    model = load_model('models/baseline.pt')

    forget_loader, retain_loader = get_forget_retain_loaders(
        forget_indices, batch_size=128, num_workers=2
    )

    unlearned_model = gradient_ascent_unlearn(
        model, forget_loader, retain_loader,
        num_epochs=1, lr=1e-5, max_steps_per_epoch=50, grad_clip_norm=1.0,
        device=device
    )

    torch.save({
        'epoch': 1,
        'model_state_dict': unlearned_model.state_dict(),
    }, 'models/gradient_ascent.pt')
    print("Saved models/gradient_ascent.pt")

    # Sanity check
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_set = datasets.CIFAR100(root='data/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    baseline_model = load_model('models/baseline.pt').to(device)
    unlearned_model = unlearned_model.to(device)

    baseline_acc = per_class_accuracy(baseline_model, test_loader, device=device)
    unlearned_acc = per_class_accuracy(unlearned_model, test_loader, device=device)

    tier_to_classes = load_taxonomy_by_tier('data/capability_taxonomy.json')

    print(f"\n{'Tier':20s} | {'Baseline':>9s} | {'Unlearned':>9s} | {'Diff':>7s}")
    print("-" * 58)
    for tier, classes in tier_to_classes.items():
        idx = np.array(classes, dtype=np.int64)
        b = baseline_acc[idx].mean()
        u = unlearned_acc[idx].mean()
        print(f"{tier:20s} | {b:9.3f} | {u:9.3f} | {u - b:+7.3f}")

    print(f"\nOverall Baseline:  {baseline_acc.mean():.3f}")
    print(f"Overall Unlearned: {unlearned_acc.mean():.3f}")


if __name__ == '__main__':
    main()
