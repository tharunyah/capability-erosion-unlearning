# train_oracle.py
#
# Day 4-5 of the 18-day plan (applied to real forget sets).
#
# The oracle is the gold standard for machine unlearning — a model retrained
# from scratch on the retain set (lt_train_indices minus the forget set).
# Day 8's compare_unlearning.py uses oracle distance as the quality metric
# for each unlearning method (gradient ascent, Fisher, fine-tuning).
#
# Oracle strategy (per friend's analysis):
#   - Influence forget sets: 50 ⊂ 100 ⊂ 200 ⊂ 300 ⊂ 400 (same ranked LiSSA list, top-k).
#     One oracle trained on lt_indices - influence_400 covers all five budgets.
#   - Random forget sets: independently sampled, no subset relationship.
#     One oracle per budget (random_50, random_100, random_200, random_300, random_400).
#   Total: 6 oracles instead of 9 (thanks to the influence nesting).
#
# Usage:
#   Single (sanity check first):
#     python train_oracle.py --single influence
#     python train_oracle.py --single random_100
#
#   Full run (all 6 oracles):
#     python train_oracle.py
#
# On Colab: set DATA_ROOT env var to your Drive cache path before running.
#   import os; os.environ['DATA_ROOT'] = '/content/drive/MyDrive/cifar100_cache'

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

DATA_ROOT        = os.environ.get('DATA_ROOT', 'data')
CIFAR100_MEAN    = (0.5071, 0.4867, 0.4408)
CIFAR100_STD     = (0.2675, 0.2565, 0.2761)
CHECKPOINT_EVERY = 20


# ---------------------------------------------------------------------------
# Oracle definitions
# ---------------------------------------------------------------------------

def get_oracle_configs(data_dir='data'):
    """
    Returns the 4 oracle configs based on the subset relationship:
      - influence: one oracle using forget_influence_200 (superset of 50+100)
      - random_50, random_100, random_200: one oracle each (independently sampled)
    """
    return [
        {
            'key':         'influence',
            'forget_path': os.path.join(data_dir, 'forget_influence_400.npy'),
            'label':       'oracle_influence',
            'covers':      ['forget_influence_50', 'forget_influence_100', 'forget_influence_200',
                            'forget_influence_300', 'forget_influence_400'],
            'note':        'forget_influence_50/100/200/300 are all subsets of _400 (ranked LiSSA list)',
        },
        {
            'key':         'random_50',
            'forget_path': os.path.join(data_dir, 'forget_random_50.npy'),
            'label':       'oracle_random_50',
            'covers':      ['forget_random_50'],
            'note':        'independently sampled — needs own oracle',
        },
        {
            'key':         'random_100',
            'forget_path': os.path.join(data_dir, 'forget_random_100.npy'),
            'label':       'oracle_random_100',
            'covers':      ['forget_random_100'],
            'note':        'independently sampled — needs own oracle',
        },
        {
            'key':         'random_200',
            'forget_path': os.path.join(data_dir, 'forget_random_200.npy'),
            'label':       'oracle_random_200',
            'covers':      ['forget_random_200'],
            'note':        'independently sampled — needs own oracle',
        },
        {
            'key':         'random_300',
            'forget_path': os.path.join(data_dir, 'forget_random_300.npy'),
            'label':       'oracle_random_300',
            'covers':      ['forget_random_300'],
            'note':        'independently sampled — needs own oracle',
        },
        {
            'key':         'random_400',
            'forget_path': os.path.join(data_dir, 'forget_random_400.npy'),
            'label':       'oracle_random_400',
            'covers':      ['forget_random_400'],
            'note':        'independently sampled — needs own oracle',
        },
    ]


# ---------------------------------------------------------------------------
# Data / Model helpers
# ---------------------------------------------------------------------------

def build_model(num_classes=100):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_retain_loader(lt_indices_path, forget_indices, batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    full_train     = datasets.CIFAR100(root=DATA_ROOT, train=True,
                                        download=True, transform=transform)
    lt_indices     = np.load(lt_indices_path).tolist()
    forget_set     = set(forget_indices.tolist())
    retain_indices = [i for i in lt_indices if i not in forget_set]

    print(f"  lt_train size  : {len(lt_indices)}")
    print(f"  forget set size: {len(forget_set)}")
    print(f"  retain set size: {len(retain_indices)}")

    return DataLoader(Subset(full_train, retain_indices),
                      batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def get_test_loader(num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_set = datasets.CIFAR100(root=DATA_ROOT, train=False,
                                  download=True, transform=transform)
    return DataLoader(test_set, batch_size=256, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def per_class_accuracy(model, loader, num_classes=100, device='cpu'):
    model.eval()
    correct = torch.zeros(num_classes, device=device)
    total   = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for c in range(num_classes):
                mask = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum()
                total[c]   += mask.sum()
    return (correct / total.clamp(min=1)).cpu().numpy()


def load_taxonomy_by_tier(path):
    with open(path) as f:
        class_to_tier = json.load(f)
    tier_to_classes = {}
    for cls, tier in class_to_tier.items():
        tier_to_classes.setdefault(tier, []).append(int(cls))
    return tier_to_classes


def print_tier_breakdown(acc, tier_to_classes):
    print(f"\n  {'Tier':20s} | {'Accuracy':>9s}")
    print("  " + "-" * 34)
    for tier, classes in tier_to_classes.items():
        idx = np.array(classes, dtype=np.int64)
        print(f"  {tier:20s} | {acc[idx].mean():9.3f}")
    print(f"  {'Overall':20s} | {acc.mean():9.3f}")


def final_ckpt_path(cfg):
    return os.path.join('models', f"{cfg['label']}.pt")


def already_done(cfg):
    return os.path.exists(final_ckpt_path(cfg))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_oracle(cfg, args, device):
    print(f"\n{'='*60}")
    print(f"Oracle: {cfg['label']}")
    print(f"  Forget set: {cfg['forget_path']}")
    print(f"  Covers: {', '.join(cfg['covers'])}")
    print(f"  Note: {cfg['note']}")
    print(f"{'='*60}")

    forget_indices = np.load(cfg['forget_path'])
    retain_loader  = get_retain_loader(
        args.lt_indices, forget_indices,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = get_test_loader(num_workers=args.num_workers)

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 0

    # Resume from mid-training checkpoint if Colab disconnected mid-run
    mid_pattern = os.path.join('models', f"{cfg['label']}_epoch*.pt")
    mid_ckpts   = sorted(glob.glob(mid_pattern))
    if mid_ckpts:
        latest = mid_ckpts[-1]
        print(f"  Resuming from: {latest}")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        print(f"  Resumed at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, num_batches = 0.0, 0

        for images, labels in retain_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches  += 1

        scheduler.step()
        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {running_loss/num_batches:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            mid_path = os.path.join('models',
                                    f"{cfg['label']}_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, mid_path)
            print(f"  Mid-training checkpoint: {mid_path}")

    # Save final oracle checkpoint
    torch.save({
        'epoch':            args.epochs,
        'label':            cfg['label'],
        'forget_set':       cfg['forget_path'],
        'covers':           cfg['covers'],
        'model_state_dict': model.state_dict(),
    }, final_ckpt_path(cfg))
    print(f"\n  Saved: {final_ckpt_path(cfg)}")

    # Per-tier accuracy report
    acc = per_class_accuracy(model, test_loader, device=device)
    np.save(f"results/{cfg['label']}_per_class_acc.npy", acc)

    tier_to_classes = load_taxonomy_by_tier(args.taxonomy)
    print(f"\n  Accuracy breakdown for {cfg['label']}:")
    print_tier_breakdown(acc, tier_to_classes)

    # Clean up mid-training checkpoints
    for mid in glob.glob(mid_pattern):
        os.remove(mid)

    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)

    all_configs = get_oracle_configs('data')

    if args.single:
        configs = [c for c in all_configs if c['key'] == args.single]
        if not configs:
            valid = [c['key'] for c in all_configs]
            raise ValueError(f"--single must be one of: {valid}")
    else:
        configs = all_configs

    print(f"\n{'SINGLE: ' + args.single if args.single else 'FULL RUN — 4 oracles'}")
    print(f"{'='*60}")
    for c in configs:
        status = 'DONE (skipping)' if already_done(c) else 'pending'
        print(f"  {c['label']:30s} covers: {c['covers']}  [{status}]")

    pending = [c for c in configs if not already_done(c)]
    if not pending:
        print("\nAll oracles already trained.")
        return

    print(f"\n{len(pending)} oracle(s) to train.")
    print(f"Estimated time on T4: ~{len(pending)*40}–{len(pending)*70} mins\n")

    for cfg in pending:
        train_oracle(cfg, args, device)

    print("\nAll done.")
    if args.single:
        print(f"\nIf the tier breakdown looks sane,")
        print(f"run the full 4 oracles with: python train_oracle.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=2)
    parser.add_argument('--lt_indices',  type=str,   default='data/lt_train_indices.npy')
    parser.add_argument('--taxonomy',    type=str,   default='data/capability_taxonomy.json')
    parser.add_argument(
        '--single', type=str, default=None,
        help='Train one oracle only. Options: influence, random_50, random_100, random_200, random_300, random_400'
    )
    args = parser.parse_args()
    main(args)
