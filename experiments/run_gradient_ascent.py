# experiments/run_gradient_ascent.py
import sys
import os
import re
import glob
import csv

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
    correct = torch.zeros(num_classes, device=device)
    total = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for c in range(num_classes):
                mask = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum()
                total[c] += mask.sum()
    return (correct / total.clamp(min=1)).cpu().numpy()


def load_taxonomy_by_tier(path):
    with open(path, 'r') as f:
        class_to_tier = json.load(f)
    tier_to_classes = {}
    for class_id_str, tier in class_to_tier.items():
        tier_to_classes.setdefault(tier, []).append(int(class_id_str))
    return tier_to_classes


def discover_forget_sets(data_dir='data'):
    pattern = os.path.join(data_dir, 'forget_*.npy')
    files = sorted(glob.glob(pattern))

    forget_sets = []
    for f in files:
        fname = os.path.basename(f)
        if 'placeholder' in fname:
            continue
        m = re.match(r'forget_(influence|random)_(\d+)\.npy', fname)
        if not m:
            print(f"  Skipping unrecognized file: {fname}")
            continue
        strategy, budget = m.group(1), int(m.group(2))
        forget_sets.append({'path': f, 'strategy': strategy, 'budget': budget, 'name': fname})

    return forget_sets


def already_done(fs, model_dir='models'):
    """Resume-safety: skip a forget set if its checkpoint already exists."""
    ckpt_name = os.path.join(model_dir, f"gradient_ascent_{fs['strategy']}_{fs['budget']}.pt")
    return os.path.exists(ckpt_name)


def append_results_csv(rows, csv_path='results/gradient_ascent_results.csv'):
    """Append-safe CSV writing so partial progress is never lost on disconnect."""
    fieldnames = ['forget_set', 'strategy', 'budget', 'tier', 'baseline_acc',
                  'unlearned_acc', 'diff', 'overall_baseline', 'overall_unlearned', 'overall_diff']
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    forget_sets = discover_forget_sets('data')
    print(f"Found {len(forget_sets)} forget sets total:")
    for fs in forget_sets:
        status = "DONE (skipping)" if already_done(fs) else "pending"
        print(f"  - {fs['name']} (strategy={fs['strategy']}, budget={fs['budget']}) [{status}]")

    pending = [fs for fs in forget_sets if not already_done(fs)]
    if not pending:
        print("\nAll 6 forget sets already have checkpoints. Nothing to do.")
        return
    print(f"\n{len(pending)} forget set(s) remaining this run.\n")

    tier_to_classes = load_taxonomy_by_tier('data/capability_taxonomy.json')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_set = datasets.CIFAR100(root='data/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    baseline_model = load_model('models/baseline.pt').to(device)
    baseline_acc = per_class_accuracy(baseline_model, test_loader, device=device)
    baseline_overall = baseline_acc.mean()
    print(f"Baseline overall accuracy: {baseline_overall:.4f}\n")

    for fs in pending:
        print(f"{'='*60}")
        print(f"Running gradient ascent on {fs['name']}")
        print(f"{'='*60}")

        forget_indices = np.load(fs['path'])
        print(f"Loaded {len(forget_indices)} forget indices")

        # Fresh baseline reload every time - never compound unlearning
        model = load_model('models/baseline.pt')

        forget_loader, retain_loader = get_forget_retain_loaders(
            forget_indices, batch_size=128, num_workers=2
        )

        unlearned_model = gradient_ascent_unlearn(
            model, forget_loader, retain_loader,
            num_epochs=1, lr=1e-4, max_steps_per_epoch=50, grad_clip_norm=5.0,
            device=device
        )

        ckpt_name = f"models/gradient_ascent_{fs['strategy']}_{fs['budget']}.pt"
        torch.save({
            'epoch': 1,
            'forget_set': fs['name'],
            'model_state_dict': unlearned_model.state_dict(),
        }, ckpt_name)
        print(f"Saved {ckpt_name}")

        unlearned_model = unlearned_model.to(device)
        unlearned_acc = per_class_accuracy(unlearned_model, test_loader, device=device)
        unlearned_overall = unlearned_acc.mean()

        print(f"\n{'Tier':20s} | {'Baseline':>9s} | {'Unlearned':>9s} | {'Diff':>7s}")
        print("-" * 58)
        rows = []
        for tier, classes in tier_to_classes.items():
            idx = np.array(classes, dtype=np.int64)
            b = baseline_acc[idx].mean()
            u = unlearned_acc[idx].mean()
            diff = u - b
            print(f"{tier:20s} | {b:9.3f} | {u:9.3f} | {diff:+7.3f}")

            rows.append({
                'forget_set': fs['name'],
                'strategy': fs['strategy'],
                'budget': fs['budget'],
                'tier': tier,
                'baseline_acc': b,
                'unlearned_acc': u,
                'diff': diff,
                'overall_baseline': baseline_overall,
                'overall_unlearned': unlearned_overall,
                'overall_diff': unlearned_overall - baseline_overall,
            })

        # Write immediately after each forget set finishes - so a disconnect
        # after forget set #3 doesn't lose results for #1 and #2
        append_results_csv(rows)
        print(f"\nOverall Baseline:  {baseline_overall:.3f}")
        print(f"Overall Unlearned: {unlearned_overall:.3f}")
        print(f"Appended results to results/gradient_ascent_results.csv\n")

    print("All pending forget sets processed.")


if __name__ == '__main__':
    main()
