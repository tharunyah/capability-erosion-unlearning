# unlearn/fisher_forgetting.py
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_fisher_diagonal(model, forget_loader, device, num_batches=10):
    """
    Computes the diagonal of the Fisher Information Matrix for the forget set.

    Fisher diagonal ≈ E[∇_θ log p(y|x)]²
    In practice: mean of squared gradients over forget set samples.

    This tells us which weights are most important for the forget set.
    We add noise proportional to this importance to erase those weights.

    Parameters
    ----------
    num_batches : how many batches to use for Fisher estimation

    Returns
    -------
    fisher_diag : dict mapping parameter name -> diagonal Fisher tensor
    """
    model.eval()
    loss_fn     = nn.CrossEntropyLoss()
    fisher_diag = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    data_iter  = iter(forget_loader)
    n_batches  = 0

    for _ in range(num_batches):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()

        with torch.enable_grad():
            loss = loss_fn(model(inputs), labels)
            loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher_diag[n] += p.grad.detach() ** 2

        n_batches += 1

    # Normalize
    for n in fisher_diag:
        fisher_diag[n] /= max(n_batches, 1)

    return fisher_diag


def fisher_forgetting_unlearn(
    model,
    forget_loader,
    device,
    noise_scale = 1e-3,
    num_batches = 10,
    verbose     = True
):
    """
    Unlearning via Fisher Forgetting.

    Intuition:
        Weights that are highly important to the forget set (high Fisher value)
        receive proportionally more noise → erases their contribution.
        Weights unimportant to the forget set receive little noise → preserved.

    This is more targeted than gradient ascent — it only disrupts the weights
    that specifically encode the forget set.

    Parameters
    ----------
    noise_scale : controls how much noise to add (higher = more forgetting)
    num_batches : batches used to estimate Fisher diagonal

    Returns
    -------
    unlearned_model : a new model (copy) with forget set unlearned
    """
    unlearned_model = copy.deepcopy(model)
    unlearned_model.to(device)

    if verbose:
        print("  Computing Fisher diagonal on forget set...")

    fisher_diag = compute_fisher_diagonal(
        unlearned_model, forget_loader, device, num_batches
    )

    if verbose:
        print("  Adding Fisher-scaled noise to parameters...")

    with torch.no_grad():
        for n, p in unlearned_model.named_parameters():
            if n in fisher_diag:
                # Noise proportional to Fisher importance
                noise = torch.randn_like(p) * (fisher_diag[n].sqrt() * noise_scale)
                p.add_(noise)

    if verbose:
        # Report how much the weights changed
        total_noise = sum(
            (fisher_diag[n].sqrt() * noise_scale).mean().item()
            for n in fisher_diag
        )
        print(f"  Mean noise magnitude: {total_noise / len(fisher_diag):.6f}")

    unlearned_model.eval()
    return unlearned_model