# train_baseline.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Colab (Drive mounted): '/content/drive/MyDrive/cifar100_cache'
# Local laptop:          'data/cifar100'
DATA_ROOT = '/content/drive/MyDrive/cifar100_cache' if os.path.exists('/content') else 'data/cifar100'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-3)
    return parser.parse_args()


def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('models', exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    full_trainset = datasets.CIFAR100(root=DATA_ROOT, train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=test_transform)

    lt_indices = np.load('data/lt_train_indices.npy')
    train_subset = Subset(full_trainset, lt_indices.tolist())
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=2)

    print(f"Train subset : {len(train_subset):,} samples")
    print(f"Test set     : {len(test_dataset):,} samples")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        # Save best checkpoint by real test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, 'models/baseline.pt')
            print(f"  -> New best saved (test_acc={test_acc:.4f})")

        # Periodic checkpoint every 20 epochs as a disconnect-safety net
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, f'models/baseline_checkpoint_epoch{epoch+1}.pt')
            print(f"  -> Periodic checkpoint saved at epoch {epoch+1}")

    print(f"\nDone. Best test accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
