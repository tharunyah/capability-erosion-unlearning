# attack/influence.py
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ── Data helpers ─────────────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std= [0.2675, 0.2565, 0.2761]
        )
    ])


def load_testset():
    return datasets.CIFAR100(
        root='./data', train=False,
        download=False, transform=get_transform()
    )


def load_trainset():
    return datasets.CIFAR100(
        root='./data', train=True,
        download=False, transform=get_transform()
    )


def get_longtail_test_indices(taxonomy_path='data/capability_taxonomy.json'):
    """Returns list of test-set indices whose class is long_tail or safety_critical."""
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    target_classes = set(
        int(cls) for cls, tier in taxonomy.items()
        if tier in ('long_tail', 'safety_critical')
    )

    testset = load_testset()
    indices = [i for i, (_, label) in enumerate(testset) if label in target_classes]
    return indices


# ── Gradient utilities ───────────────────────────────────────────────────────

def compute_gradients_eval(model, loss_fn, inputs, labels, device):
    """
    Computes flat gradient vector using eval mode.
    Works correctly with batch size 1 (no BatchNorm issues).

    We use eval() + manual grad computation instead of train() to avoid
    BatchNorm breaking on single samples.
    """
    model.eval()
    model.zero_grad()
    inputs, labels = inputs.to(device), labels.to(device)

    # Enable grad computation even in eval mode
    with torch.enable_grad():
        outputs = model(inputs)
        loss    = loss_fn(outputs, labels)
        loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().reshape(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=device))

    return torch.cat(grads)


def compute_test_gradient(model, loss_fn, test_indices, device, batch_size=64):
    """
    Computes the average gradient across all target test samples.
    Returns: 1D tensor of shape (num_params,)
    """
    testset = load_testset()
    subset  = Subset(testset, test_indices)
    loader  = DataLoader(subset, batch_size=batch_size,
                         shuffle=False, num_workers=0)

    avg_grad = None
    n_batches = 0

    for inputs, labels in loader:
        g        = compute_gradients_eval(model, loss_fn, inputs, labels, device)
        avg_grad = g if avg_grad is None else avg_grad + g
        n_batches += 1

    avg_grad = avg_grad / n_batches
    return avg_grad


# ── LiSSA ────────────────────────────────────────────────────────────────────

def hvp(model, loss_fn, inputs, labels, vector, device):
    """
    Computes Hessian-vector product: H · vector
    using PyTorch double backprop.
    """
    model.eval()
    model.zero_grad()
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.enable_grad():
        outputs = model(inputs)
        loss    = loss_fn(outputs, labels)

        grads = torch.autograd.grad(
            loss, model.parameters(),
            create_graph=True
        )
        grads_flat = torch.cat([g.reshape(-1) for g in grads])

        grad_dot_v = torch.dot(grads_flat, vector)

        hvp_grads = torch.autograd.grad(
            grad_dot_v, model.parameters(),
            retain_graph=False
        )

    hvp_flat = torch.cat([g.detach().reshape(-1) for g in hvp_grads])
    return hvp_flat


def lissa(
    model,
    loss_fn,
    train_loader,
    v,
    device,
    num_steps = 200,
    damping   = 0.01,
    scale     = 500.0,
    verbose   = True
):
    """
    LiSSA: approximates H⁻¹ · v.

    Recursion:
        h_0 = v
        h_t = v + (I - H/scale - damping·I) · h_{t-1}

    scale=500 works well for ResNet18 on CIFAR-100.
    damping=0.01 adds numerical stability.
    """
    h         = v.clone().detach()
    data_iter = iter(train_loader)

    for t in range(num_steps):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter      = iter(train_loader)
            inputs, labels = next(data_iter)

        Hh = hvp(model, loss_fn, inputs, labels, h, device)

        # Neumann update
        h = v + (1 - damping) * h - Hh / scale
        h = h.detach()

        if verbose and (t + 1) % 50 == 0:
            norm = h.norm().item()
            print(f"  LiSSA step {t+1}/{num_steps}  |h|={norm:.4f}")
            if torch.isnan(h).any():
                print("  WARNING: NaN detected. Try increasing scale.")
                break

    return h


# ── Influence Scores ─────────────────────────────────────────────────────────

def compute_influence_scores(
    model,
    loss_fn,
    lt_train_indices,
    device,
    lissa_steps  = 200,
    damping      = 0.01,
    scale        = 500.0,
    batch_size   = 32,
    verbose      = True
):
    """
    Computes influence score for every sample in lt_train_indices.

    Pipeline:
        1. Compute average test gradient over long-tail test set
        2. Run LiSSA to get H⁻¹ · test_gradient
        3. For each training sample: score = -grad_train · (H⁻¹ · grad_test)

    Returns
    -------
    scores : np.ndarray of shape (len(lt_train_indices),)
    """

    # ── Step 1: Test gradient ────────────────────────────────────────────────
    print("Computing test gradient over long-tail + safety-critical test set...")
    test_indices = get_longtail_test_indices()
    print(f"  Target test samples: {len(test_indices)}")
    test_grad = compute_test_gradient(model, loss_fn, test_indices, device)
    print(f"  Test gradient norm: {test_grad.norm().item():.4f}")

    # ── Step 2: LiSSA ────────────────────────────────────────────────────────
    print(f"\nRunning LiSSA ({lissa_steps} steps)...")
    trainset     = load_trainset()
    train_loader = DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=0
    )
    ihvp = lissa(
        model, loss_fn, train_loader, test_grad,
        device, lissa_steps, damping, scale, verbose
    )
    print(f"  iHVP norm: {ihvp.norm().item():.4f}")

    if torch.isnan(ihvp).any():
        raise RuntimeError(
            "LiSSA produced NaN. Increase scale (try 1000 or 5000) "
            "or reduce lissa_steps."
        )

    # ── Step 3: Score each training sample ──────────────────────────────────
    print(f"\nScoring {len(lt_train_indices)} long-tail training samples...")
    trainset_score = load_trainset()
    scores         = np.zeros(len(lt_train_indices))

    for i, idx in enumerate(lt_train_indices):
        img, label = trainset_score[idx]
        img        = img.unsqueeze(0).to(device)
        label      = torch.tensor([label]).to(device)

        g         = compute_gradients_eval(model, loss_fn, img, label, device)
        scores[i] = -torch.dot(g, ihvp).item()

        if verbose and (i + 1) % 500 == 0:
            print(f"  Scored {i+1}/{len(lt_train_indices)}")

    return scores