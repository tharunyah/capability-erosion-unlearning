# evaluate/per_class_eval.py
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    ckpt     = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model

def evaluate_per_class(
    model: nn.Module,
    device: torch.device,
    save_path: str = None
) -> np.ndarray:
    """
    Run model on the full CIFAR-100 test set (10,000 samples).
    Returns per_class_acc: np.ndarray of shape (100,) with values in [0, 1].
    Saves to save_path if provided.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std =[0.2675, 0.2565, 0.2761]
        )
    ])
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False,
        download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=256,
                              shuffle=False, num_workers=2)

    class_correct = np.zeros(100)
    class_total   = np.zeros(100)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted   = model(inputs).max(1)

            for cls in range(100):
                mask = (labels == cls)
                class_correct[cls] += predicted[mask].eq(labels[mask]).sum().item()
                class_total[cls]   += mask.sum().item()

    per_class_acc = class_correct / (class_total + 1e-9)

    if save_path:
        np.save(save_path, per_class_acc)
        print(f"Saved  →  {save_path}")

    return per_class_acc

def print_tier_summary(
    per_class_acc: np.ndarray,
    taxonomy_path: str = 'data/capability_taxonomy.json'
):
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    tiers = {'majority': [], 'mid_tail': [], 'long_tail': [], 'safety_critical': []}
    for cls_str, tier in taxonomy.items():
        tiers[tier].append(per_class_acc[int(cls_str)])

    print(f"\n  {'Tier':<18}  {'n':>4}  {'Mean':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*56}")
    for tier, accs in tiers.items():
        if accs:
            print(f"  {tier:<18}  {len(accs):>4}  "
                  f"{np.mean(accs)*100:>7.2f}%  "
                  f"{np.min(accs)*100:>7.2f}%  "
                  f"{np.max(accs)*100:>7.2f}%")

# ── Run standalone ────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/baseline.pt')
    parser.add_argument('--save',       default='data/baseline_per_class_acc.npy')
    args = parser.parse_args()

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = load_model(args.checkpoint, device)
    per_class_acc = evaluate_per_class(model, device, save_path=args.save)
    print_tier_summary(per_class_acc)
