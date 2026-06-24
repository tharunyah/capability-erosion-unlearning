# train_baseline.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('models', exist_ok=True)  # ensure folder exists before any save

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    full_trainset = datasets.CIFAR100(root='data/', train=True, download=True, transform=train_transform)

    lt_indices = np.load('data/lt_train_indices.npy')
    train_subset = Subset(full_trainset, lt_indices.tolist())
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {correct/total:.4f}")

        # Periodic checkpoint every 20 epochs, in case the final save ever fails again
        if (epoch + 1) % 20 == 0:
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'models/baseline_checkpoint_epoch{epoch+1}.pt')
            print(f"Checkpoint saved at epoch {epoch+1}")

    # Final save
    os.makedirs('models', exist_ok=True)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': correct / total
    }, 'models/baseline.pt')
    print("Saved models/baseline.pt")


if __name__ == '__main__':
    main()
