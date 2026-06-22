# run_influence.py
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from attack.influence import compute_influence_scores

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    ckpt = torch.load('models/baseline.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    lt_train_indices = np.load('data/lt_train_indices.npy')
    print(f"Long-tail training samples: {len(lt_train_indices)}")

    loss_fn = nn.CrossEntropyLoss()

    scores = compute_influence_scores(
        model            = model,
        loss_fn          = loss_fn,
        lt_train_indices = lt_train_indices,
        device           = device,
        lissa_steps      = 200,
        damping          = 0.1,
        scale            = 5000.0,
        batch_size       = 32,
        verbose          = True
    )

    np.save('data/influence_scores.npy', scores)
    print(f"\nDone. Scores shape: {scores.shape}")
    print(f"Max: {scores.max():.4f}  Min: {scores.min():.4f}  Mean: {scores.mean():.4f}")
    print("Saved to data/influence_scores.npy")