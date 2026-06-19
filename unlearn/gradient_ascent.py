import os
import torch
import torch.nn as nn
import torchvision.models as models


def load_baseline_model(checkpoint_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)
    return model