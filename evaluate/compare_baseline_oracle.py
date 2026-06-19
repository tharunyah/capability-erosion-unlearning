import json
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18


def load_model(path, num_classes=100):
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def per_class_accuracy(model, loader, num_classes=100):
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for c in range(num_classes):
                mask = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum()
                total[c] += mask.sum()
    acc = correct / total.clamp(min=1)
    return acc.numpy()


def load_taxonomy_by_tier(path):
    """
    capability_taxonomy.json is stored as class -> tier, e.g.:
        { "1": "majority", "9": "majority", ... }
    This inverts it into tier -> list of class indices, e.g.:
        { "majority": [1, 9, 10, ...], "long_tail": [...], ... }
    """
    with open(path, 'r') as f:
        class_to_tier = json.load(f)

    tier_to_classes = {}
    for class_id_str, tier in class_to_tier.items():
        tier_to_classes.setdefault(tier, []).append(int(class_id_str))

    return tier_to_classes


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_set = datasets.CIFAR100(root='data/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    baseline = load_model('models/baseline.pt')
    oracle = load_model('models/oracle.pt')

    baseline_acc = per_class_accuracy(baseline, test_loader)
    oracle_acc = per_class_accuracy(oracle, test_loader)

    np.save('data/baseline_per_class_acc.npy', baseline_acc)
    np.save('data/oracle_per_class_acc.npy', oracle_acc)

    tier_to_classes = load_taxonomy_by_tier('data/capability_taxonomy.json')

    print(f"{'Tier':20s} | {'Baseline':>9s} | {'Oracle':>9s} | {'Diff':>7s}")
    print("-" * 55)

    for tier, classes in tier_to_classes.items():
        class_indices = np.array(classes, dtype=np.int64)
        b = baseline_acc[class_indices].mean()
        o = oracle_acc[class_indices].mean()
        print(f"{tier:20s} | {b:9.3f} | {o:9.3f} | {o - b:+7.3f}")

    print(f"\nOverall Baseline: {baseline_acc.mean():.3f}")
    print(f"Overall Oracle:   {oracle_acc.mean():.3f}")


if __name__ == '__main__':
    main()
