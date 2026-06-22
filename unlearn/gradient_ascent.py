<<<<<<< HEAD
# unlearn/gradient_ascent.py
import torch
import torch.nn as nn
import torch.optim as optim


def gradient_ascent_unlearn(
    model,
    forget_loader,
    retain_loader,
    num_epochs=1,
    lr=1e-5,
    ascent_weight=1.0,
    retain_weight=1.0,
    max_steps_per_epoch=50,
    grad_clip_norm=1.0,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    forget_iter = iter(forget_loader)

    for epoch in range(num_epochs):
        total_forget_loss = 0.0
        total_retain_loss = 0.0
        num_batches = 0

        for step, (retain_imgs, retain_labels) in enumerate(retain_loader):
            if step >= max_steps_per_epoch:
                break

            try:
                forget_imgs, forget_labels = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_imgs, forget_labels = next(forget_iter)

            forget_imgs, forget_labels = forget_imgs.to(device), forget_labels.to(device)
            retain_imgs, retain_labels = retain_imgs.to(device), retain_labels.to(device)

            optimizer.zero_grad()

            forget_outputs = model(forget_imgs)
            forget_loss = criterion(forget_outputs, forget_labels)

            retain_outputs = model(retain_imgs)
            retain_loss = criterion(retain_outputs, retain_labels)

            combined_loss = (-ascent_weight * forget_loss) + (retain_weight * retain_loss)
            combined_loss.backward()

            # Prevent the loss from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            total_forget_loss += forget_loss.item()
            total_retain_loss += retain_loss.item()
            num_batches += 1

        print(f"Epoch {epoch+1}/{num_epochs} | Forget loss: {total_forget_loss/num_batches:.4f} | Retain loss: {total_retain_loss/num_batches:.4f}")

    model.eval()
    return model
=======
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
>>>>>>> 71ca56243f4deae1733cb063693167e2102ec196
