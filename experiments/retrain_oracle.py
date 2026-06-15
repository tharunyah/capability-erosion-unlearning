# experiments/retrain_oracle.py
"""
Oracle retraining script — the gold-standard unlearning baseline.

Retrains ResNet-18 from scratch on the RETAIN set (all long-tail training
indices MINUS the forget set). This gives the theoretically perfect unlearned
model, against which approximate methods (gradient ascent, Fisher forgetting,
fine-tuning) are benchmarked.

Usage:
    python experiments/retrain_oracle.py \
        --forget_indices data/forget_set_placeholder.npy \
        --lt_indices     data/lt_train_indices.npy \
        --save           models/oracle.pt

Colab note:
    Upload this file + data/ folder to Colab, then run with default args.
    A T4 GPU will finish in ~40–70 min (same as baseline training).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyperparameters (match baseline exactly) ──────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 128
LR          = 1e-3
NUM_CLASSES = 100


# ── Data ──────────────────────────────────────────────────────────────────────
def get_retain_loader(lt_indices: np.ndarray,
                      forget_indices: np.ndarray,
                      batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    Build a DataLoader over the RETAIN set:
        retain = long-tail training indices  ∖  forget indices
    """
    forget_set = set(forget_indices.tolist())
    retain_indices = [idx for idx in lt_indices if idx not in forget_set]

    print(f"  Total long-tail train samples : {len(lt_indices)}")
    print(f"  Forget set size               : {len(forget_set)}")
    print(f"  Retain set size               : {len(retain_indices)}")

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std =[0.2675, 0.2565, 0.2761]
        )
    ])
    full_train = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=True, download=True, transform=transform
    )
    retain_dataset = Subset(full_train, retain_indices)
    return DataLoader(retain_dataset, batch_size=batch_size,
                      shuffle=True, num_workers=2, pin_memory=True)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(device: torch.device) -> nn.Module:
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


# ── Training ──────────────────────────────────────────────────────────────────
def train(model: nn.Module,
          loader: DataLoader,
          device: torch.device,
          save_path: str) -> None:

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = total = 0
        t0 = time.time()

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted  = outputs.max(1)
            correct      += predicted.eq(labels).sum().item()
            total        += inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc  = correct / total * 100
        elapsed    = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                  f"loss={epoch_loss:.4f}  acc={epoch_acc:.2f}%  "
                  f"({elapsed:.1f}s)")

        # Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            _save(model, optimizer, epoch, epoch_loss, save_path)

    print(f"\n  Oracle training complete. Best checkpoint → {save_path}")


def _save(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch'            : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss'             : loss,
    }, path)


# ── Placeholder forget set ────────────────────────────────────────────────────
def make_placeholder_forget_set(lt_indices: np.ndarray,
                                 n: int = 100,
                                 save_path: str = 'data/forget_set_placeholder.npy'
                                 ) -> np.ndarray:
    """
    Day 4 placeholder: 100 random indices from the long-tail training set.
    Week 2 will replace this with influence-guided forget sets.
    """
    rng            = np.random.default_rng(SEED)
    forget_indices = rng.choice(lt_indices, size=n, replace=False)
    np.save(save_path, forget_indices)
    print(f"  Placeholder forget set ({n} indices) → {save_path}")
    return forget_indices


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Oracle retrain for unlearning benchmark')
    parser.add_argument('--lt_indices',     default='data/lt_train_indices.npy',
                        help='Long-tail training indices (from Day 2)')
    parser.add_argument('--forget_indices', default='data/forget_set_placeholder.npy',
                        help='Forget set indices (placeholder or real)')
    parser.add_argument('--save',           default='models/oracle.pt',
                        help='Where to save the oracle checkpoint')
    parser.add_argument('--make_placeholder', action='store_true',
                        help='Generate placeholder forget set before training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # Load long-tail indices
    lt_indices = np.load(args.lt_indices)
    print(f"  Loaded lt_train_indices: {len(lt_indices)} samples")

    # Optionally create placeholder forget set
    if args.make_placeholder or not os.path.exists(args.forget_indices):
        print("\n  Generating placeholder forget set …")
        forget_indices = make_placeholder_forget_set(
            lt_indices, n=100, save_path=args.forget_indices
        )
    else:
        forget_indices = np.load(args.forget_indices)
        print(f"  Loaded forget set: {len(forget_indices)} samples from {args.forget_indices}")

    # Build retain loader
    print("\n  Building retain set loader …")
    retain_loader = get_retain_loader(lt_indices, forget_indices)

    # Train oracle from scratch
    print("\n  Starting oracle training from scratch …")
    model = build_model(device)
    train(model, retain_loader, device, save_path=args.save)


if __name__ == '__main__':
    main()
