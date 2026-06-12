# train_baseline.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--debug',      action='store_true',
                        help='2 epochs on CPU — for local sanity check only')
    return parser.parse_args()

def main():
    args = get_args()

    if args.debug:
        args.epochs = 2
        print("DEBUG MODE: 2 epochs, CPU only")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # CIFAR-100 mean/std (standard values)
    mean = [0.5071, 0.4867, 0.4408]
    std  = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load base datasets
    full_train = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=True,  download=True,
        transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, download=True,
        transform=test_transform
    )

    # Use only our long-tail subset for training
    lt_indices    = np.load('data/lt_train_indices.npy')
    train_subset  = Subset(full_train, lt_indices)

    # num_workers=0 is safest on Windows; Colab will use 2
    nw = 0 if args.debug else 2
    train_loader = DataLoader(train_subset,  batch_size=args.batch_size,
                               shuffle=True,  num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=256,
                               shuffle=False, num_workers=nw, pin_memory=True)

    print(f"Train subset : {len(train_subset):,} samples")
    print(f"Test set     : {len(test_dataset):,} samples")

    # ResNet-18 — replace final FC for 100 classes
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model    = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs('models', exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):

        # ── Training ──────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * inputs.size(0)
            _, predicted   = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc  = 100. * train_correct / train_total
        train_loss = train_loss / train_total

        # ── Evaluation ────────────────────────────────────
        model.eval()
        test_correct = test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted  = outputs.max(1)
                test_total   += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total
        scheduler.step()

        print(f"Epoch [{epoch:>3}/{args.epochs}]  "
              f"Loss: {train_loss:.4f}  "
              f"Train: {train_acc:.1f}%  "
              f"Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc':             test_acc,
            }, 'models/baseline.pt')
            print(f"  → New best saved  (test_acc={test_acc:.2f}%)")

    print(f"\nDone. Best test accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
