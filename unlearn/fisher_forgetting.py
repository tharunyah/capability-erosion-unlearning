# unlearn/fisher_forgetting.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from evaluate.per_class_eval import load_model


# ── Tunables ─────────────────────────────────────────────────────────────────
ALPHA          = 1e-2   # noise scale — NEEDS SWEEPING, no principled default
NOISE_STD_CLIP = 0.1    # max per-parameter noise std, prevents blowup on near-zero Fisher
EPS            = 1e-8   # numerical stability in 1/sqrt(fisher)


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std= [0.2675, 0.2565, 0.2761]
        )
    ])


def load_trainset():
    return datasets.CIFAR100(root='./data', train=True, download=False, transform=get_transform())


def compute_fisher_diagonal(model, loss_fn, retain_indices, device):
    """
    Empirical diagonal Fisher Information over the FULL retain set.
    F_i ~= E[(d loss / d theta_i)^2], estimated one sample at a time
    since squared-gradient terms don't batch through a summed backward pass.
    """
    trainset = load_trainset()
    subset   = Subset(trainset, retain_indices)
    loader   = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)

    fisher = [torch.zeros_like(p) for p in model.parameters()]
    model.eval()
    n_seen = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        for i in range(inputs.size(0)):
            model.zero_grad()
            out  = model(inputs[i:i+1])
            loss = loss_fn(out, labels[i:i+1])
            loss.backward()
            for f, p in zip(fisher, model.parameters()):
                if p.grad is not None:
                    f += p.grad.detach() ** 2
            n_seen += 1

            if n_seen % 2000 == 0:
                print(f"    Fisher progress: {n_seen}/{len(retain_indices)}")

    fisher = [f / n_seen for f in fisher]
    return fisher


def apply_fisher_noise(model, fisher, alpha=ALPHA, std_clip=NOISE_STD_CLIP, eps=EPS):
    """
    Injects Gaussian noise into each parameter, scaled inversely to its
    Fisher information: low-Fisher (unimportant to retain) params get
    perturbed more, high-Fisher (important to retain) params stay stable.
    """
    with torch.no_grad():
        for p, f in zip(model.parameters(), fisher):
            std = alpha / torch.sqrt(f + eps)
            std = torch.clamp(std, max=std_clip)
            p.add_(torch.randn_like(p) * std)
    return model


def fisher_forget(baseline_path, lt_train_indices, forget_indices, device, loss_fn):
    model = load_model(baseline_path, device)

    retain_indices = np.setdiff1d(lt_train_indices, forget_indices)
    print(f"  Retain set: {len(retain_indices)} samples "
          f"(baseline's {len(lt_train_indices)} minus {len(forget_indices)} forgotten)")

    print(f"  Computing Fisher over all {len(retain_indices)} retain samples...")
    fisher = compute_fisher_diagonal(model, loss_fn, retain_indices, device)

    print(f"  Applying noise (alpha={ALPHA}, std_clip={NOISE_STD_CLIP})...")
    model = apply_fisher_noise(model, fisher)

    return model


if __name__ == '__main__':
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    loss_fn = nn.CrossEntropyLoss()
    lt_train_indices = np.load('data/lt_train_indices.npy')

    for strategy in ['influence', 'random']:
        for budget in [50, 100, 200]:
            print(f"\n=== Fisher forgetting: {strategy}, budget={budget} ===")
            forget_indices = np.load(f'data/forget_{strategy}_{budget}.npy')

            model = fisher_forget(
                baseline_path     = 'models/baseline.pt',
                lt_train_indices  = lt_train_indices,
                forget_indices    = forget_indices,
                device            = device,
                loss_fn           = loss_fn
            )

            out_path = f'models/unlearned_fisher_{strategy}_{budget}.pt'
            torch.save({'model_state_dict': model.state_dict()}, out_path)
            print(f"  Saved -> {out_path}")

    print("\nDone. 6 Fisher-unlearned checkpoints saved to models/")