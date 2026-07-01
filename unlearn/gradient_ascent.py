# unlearn/gradient_ascent.py
import torch
import torch.nn as nn
import torch.optim as optim


def gradient_ascent_unlearn(
    model,
    forget_loader,
    retain_loader,
    num_epochs=1,
    lr=1e-4,
    ascent_weight=5.0,
    retain_weight=0.3,
    max_steps_per_epoch=15,
    grad_clip_norm=5.0,
    device=None
):
    """
    Gradient ascent unlearning (Day 6 of 18-day plan).

    Ascends on forget-set loss (makes the model worse at those samples)
    while descending on a small retain sample to prevent catastrophic
    forgetting of the rest of the dataset.

    Key tuning vs. naive defaults:
      - ascent_weight=5.0  : pushes forgetting signal hard relative to retain
      - retain_weight=0.3  : de-emphasizes retain so it doesn't act as a
                             free fine-tune pass that swamps the forget signal
      - max_steps_per_epoch=15 : limits incidental retain-side training;
                                  50k-sample retain pool at batch 128 would
                                  otherwise dominate a 50-200 sample forget set
    """
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_forget_loss += forget_loss.item()
            total_retain_loss += retain_loss.item()
            num_batches += 1

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Forget loss: {total_forget_loss/num_batches:.4f} | "
              f"Retain loss: {total_retain_loss/num_batches:.4f}")

    model.eval()
    return model
