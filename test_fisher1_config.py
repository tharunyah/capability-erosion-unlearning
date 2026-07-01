# test_fisher_one_config.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn

from unlearn.fisher_forgetting import fisher_forget, ALPHA, NOISE_STD_CLIP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Testing with ALPHA={ALPHA}, NOISE_STD_CLIP={NOISE_STD_CLIP}")

loss_fn = nn.CrossEntropyLoss()
lt_train_indices = np.load('data/lt_train_indices.npy')
forget_indices    = np.load('data/forget_influence_50.npy')

model = fisher_forget(
    baseline_path     = 'models/baseline.pt',
    lt_train_indices  = lt_train_indices,
    forget_indices    = forget_indices,
    device            = device,
    loss_fn           = loss_fn
)

out_path = 'models/TEST_fisher_influence_50.pt'
torch.save({'model_state_dict': model.state_dict()}, out_path)
print(f"Saved test checkpoint -> {out_path}")