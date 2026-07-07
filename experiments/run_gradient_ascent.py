# experiments/run_gradient_ascent.py
#
# Day 6 of the 18-day plan:
#   - Runs gradient ascent unlearning across all 6 forget sets
#     (forget_influence_50/100/200 and forget_random_50/100/200)
#   - Each run starts from a fresh copy of baseline.pt
#   - Resume-safe: skips forget sets whose checkpoints already exist
#   - Append-only CSV so partial progress survives a Colab disconnect
#
# Sanity check (run this on just forget_influence_100 first):
#   python experiments/run_gradient_ascent.py --sanity
#
# Full run (all 6):
#   python experiments/run_gradient_ascent.py

import sys
import os
import re
import glob
import csv
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.dataloader_utils import get_forget_retain_loaders
from unlearn.gradient_ascent import gradient_ascent_unlearn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(path, num_classes=100):
    model = resnet18(weights=None)
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


def discover_forget_sets(data_dir='data', single=False):
    """
    Finds forget_influence_*.npy and forget_random_*.npy, including
    seeded random repeats like forget_random_100_seed43.npy.
    If single=True, returns only forget_influence_100 for the sanity check.
    """
    pattern = os.path.join(data_dir, 'forget_*.npy')
    files = sorted(glob.glob(pattern))

    forget_sets = []
    for f in files:
        fname = os.path.basename(f)
        if 'placeholder' in fname:
            continue
        m = re.match(r'forget_(influence|random)_(\d+)(?:_seed(\d+))?\.npy', fname)
        if not m:
            print(f"  Skipping unrecognized file: {fname}")
            continue
        strategy, budget, seed = m.group(1), int(m.group(2)), m.group(3)
        entry = {
            'path': f,
            'strategy': strategy,
            'budget': budget,
            'seed': int(seed) if seed is not None else None,
            'name': fname
        }
        forget_sets.append(entry)

    if single:
        # Sanity check: just run forget_influence_100
        forget_sets = [fs for fs in forget_sets
                       if fs['strategy'] == 'influence' and fs['budget'] == 100 and fs['seed'] is None]
        if not forget_sets:
            raise FileNotFoundError("forget_influence_100.npy not found in data/")

    return forget_sets


def already_done(fs, model_dir='models'):
    """Skip a forget set if its checkpoint already exists (resume safety)."""
    ckpt = os.path.join(model_dir, f"gradient_ascent_{fs['strategy']}_{fs['budget']}"
                                    f"{'_seed' + str(fs['seed']) if fs['seed'] is not None else ''}.pt")
    return os.path.exists(ckpt)


def append_results_csv(rows, csv_path='results/gradient_ascent_results.csv'):
    """Append after each forget set so a disconnect never loses prior results."""
    fieldnames = [
        'forget_set', 'strategy', 'budget', 'tier',
        'baseline_acc', 'unlearned_acc', 'diff',
        'overall_baseline', 'overall_unlearned', 'overall_diff'
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    forget_sets = discover_forget_sets('data', single=args.sanity)

    print(f"\n{'SANITY CHECK MODE — only forget_influence_100' if args.sanity else 'FULL RUN — all 6 forget sets'}")
    print(f"Found {len(forget_sets)} forget set(s):")
    for fs in forget_sets:
        status = "DONE (skipping)" if already_done(fs) else "pending"
        print(f"  - {fs['name']} [{status}]")

    pending = [fs for fs in forget_sets if not already_done(fs)]
    if not pending:
        print("\nAll forget sets already have checkpoints — nothing to do.")
        return

    tier_to_classes = load_taxonomy_by_tier('data/capability_taxonomy.json')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_set = datasets.CIFAR100(
        root=os.environ.get('DATA_ROOT', 'data'),
        train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    # Baseline accuracy computed once — doesn't change across runs
    baseline_model = load_model('models/baseline.pt').to(device)
    baseline_acc = per_class_accuracy(baseline_model, test_loader, device=device)
    baseline_overall = baseline_acc.mean()
    print(f"\nBaseline overall accuracy: {baseline_overall:.4f}")

    for fs in pending:
        print(f"\n{'='*60}")
        print(f"Running gradient ascent: {fs['name']}")
        print(f"  strategy={fs['strategy']}  budget={fs['budget']}")
        print(f"{'='*60}")

        forget_indices = np.load(fs['path'])
        print(f"Loaded {len(forget_indices)} forget indices")

        # Fresh baseline load every iteration — never compound unlearning
        model = load_model('models/baseline.pt')

        forget_loader, retain_loader = get_forget_retain_loaders(
            forget_indices, batch_size=128, num_workers=2
        )

        unlearned_model = gradient_ascent_unlearn(
            model,
            forget_loader,
            retain_loader,
            num_epochs=1,
            lr=1e-4,
            ascent_weight=5.0,
            retain_weight=0.3,
            max_steps_per_epoch=15,
            grad_clip_norm=5.0,
            device=device
        )

        ckpt_path = os.path.join(
            'models',
            f"gradient_ascent_{fs['strategy']}_{fs['budget']}"
            f"{'_seed' + str(fs['seed']) if fs['seed'] is not None else ''}.pt"
        )
        torch.save({
            'epoch': 1,
            'forget_set': fs['name'],
            'model_state_dict': unlearned_model.state_dict(),
        }, ckpt_path)
        print(f"Saved {ckpt_path}")

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
                'baseline_acc': round(float(b), 6),
                'unlearned_acc': round(float(u), 6),
                'diff': round(float(diff), 6),
                'overall_baseline': round(float(baseline_overall), 6),
                'overall_unlearned': round(float(unlearned_overall), 6),
                'overall_diff': round(float(unlearned_overall - baseline_overall), 6),
            })

        print(f"\nOverall Baseline:  {baseline_overall:.4f}")
        print(f"Overall Unlearned: {unlearned_overall:.4f}")
        print(f"Overall Diff:      {unlearned_overall - baseline_overall:+.4f}")

        # ---- Sanity check guidance ----
        long_tail_diff = next(r['diff'] for r in rows if r['tier'] == 'long_tail')
        overall_diff = unlearned_overall - baseline_overall

        print("\n--- Sanity Check ---")
        if long_tail_diff < 0:
            print(f"✓ long_tail accuracy dropped ({long_tail_diff:+.3f}) — forgetting is working")
        else:
            print(f"✗ long_tail accuracy went UP ({long_tail_diff:+.3f}) — ascent signal too weak, "
                  f"try raising ascent_weight or lowering retain_weight further")

        if abs(overall_diff) < 0.05:
            print(f"✓ Overall accuracy stable ({overall_diff:+.4f}) — stealthy erosion achieved")
        elif overall_diff < -0.05:
            print(f"⚠ Overall accuracy dropped too much ({overall_diff:+.4f}) — "
                  f"reduce ascent_weight or raise retain_weight slightly")
        else:
            print(f"⚠ Overall accuracy increased ({overall_diff:+.4f}) — "
                  f"retain is still dominating, lower retain_weight further")

        # Write immediately after each forget set — disconnect-safe
        append_results_csv(rows)
        print(f"Appended to results/gradient_ascent_results.csv\n")

    print("Done.")
    if args.sanity:
        print("\nSanity check complete.")
        print("If long_tail diff is negative and overall diff is < 0.05,")
        print("run the full loop with: python experiments/run_gradient_ascent.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sanity', action='store_true',
        help='Run only forget_influence_100 as a quick sanity check before the full loop'
    )
    args = parser.parse_args()
    main(args)
