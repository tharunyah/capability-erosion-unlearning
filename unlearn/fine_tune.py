# unlearn/fine_tune.py
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def fine_tune_unlearn(
    model,
    forget_loader,
    retain_loader,
    device,
    ascent_steps  = 5,
    descent_steps = 5,
    ascent_lr     = 1e-4,
    descent_lr    = 1e-4,
    verbose       = True
):
    """
    Unlearning via Gradient Ascent on forget set + Gradient Descent on retain set.

    Step 1 — Gradient Ascent on forget set:
        Maximizes loss on forget samples → model becomes worse at these samples.

    Step 2 — Gradient Descent on retain set:
        Minimizes loss on retain samples → stabilizes overall performance.

    Parameters
    ----------
    ascent_steps  : how many steps to ascend on forget set
    descent_steps : how many steps to descend on retain set
    ascent_lr     : learning rate for gradient ascent
    descent_lr    : learning rate for gradient descent

    Returns
    -------
    unlearned_model : a new model (copy) with forget set unlearned
    """
    unlearned_model = copy.deepcopy(model)
    unlearned_model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # ── Step 1: Gradient Ascent on forget set ────────────────────────────────
    if verbose:
        print("  Phase 1: Gradient Ascent on forget set...")

    optimizer = torch.optim.SGD(
        unlearned_model.parameters(),
        lr=ascent_lr, momentum=0.9
    )
    unlearned_model.train()
    forget_iter = iter(forget_loader)

    for step in range(ascent_steps):
        try:
            inputs, labels = next(forget_iter)
        except StopIteration:
            forget_iter    = iter(forget_loader)
            inputs, labels = next(forget_iter)

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(unlearned_model(inputs), labels)

        # Negate loss → ascent
        (-loss).backward()
        optimizer.step()

        if verbose:
            print(f"    Ascent step {step+1}/{ascent_steps}  loss={loss.item():.4f}")

    # ── Step 2: Gradient Descent on retain set ───────────────────────────────
    if verbose:
        print("  Phase 2: Gradient Descent on retain set...")

    optimizer = torch.optim.SGD(
        unlearned_model.parameters(),
        lr=descent_lr, momentum=0.9
    )
    retain_iter = iter(retain_loader)

    for step in range(descent_steps):
        try:
            inputs, labels = next(retain_iter)
        except StopIteration:
            retain_iter    = iter(retain_loader)
            inputs, labels = next(retain_iter)

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(unlearned_model(inputs), labels)
        loss.backward()
        optimizer.step()

        if verbose:
            print(f"    Descent step {step+1}/{descent_steps}  loss={loss.item():.4f}")

    unlearned_model.eval()
    return unlearned_model