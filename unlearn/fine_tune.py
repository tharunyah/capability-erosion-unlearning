# unlearn/fine_tune.py
#
# NOTE — this file contains TWO distinct fine-tuning-based unlearning methods:
#
# 1. fine_tune_forget / fine_tune_forget_multi_draw — retain-set-only gradient
#    DESCENT, no active forgetting step. This was originally built and labeled
#    as "the" fine_tune method, and already has a full analysis trail against
#    it (single-run CSV, 5-draw multi-draw CSV, corrected-std CSV, and the
#    auditor test in test_auditor_multimethod.py). It does NOT match the
#    interface compare_unlearning.py expects. Kept as a separate baseline
#    method going forward — think of it as "retain_only_finetune" in spirit,
#    even though the function names are unchanged to avoid breaking existing
#    imports (average_finetune_draws.py, etc.).
#
# 2. fine_tune_unlearn — the CANONICAL fine_tune method, matching
#    compare_unlearning.py's expected signature exactly: gradient ASCENT on
#    the forget set (active forgetting) followed by gradient DESCENT on the
#    retain set (utility recovery). This is what "fine_tune" means everywhere
#    else in the project (compare_unlearning.py, and the ascent_steps/
#    descent_steps hyperparameter notes this file's tunables were drawn from).
#    Any NEW fine_tune results (e.g. the 12-matrix run) should come from this
#    function, not fine_tune_forget.

import sys
import os
import copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from evaluate.per_class_eval import load_model
from unlearn.fisher_forgetting import get_transform, load_trainset


# ── Tunables ─────────────────────────────────────────────────────────────────
# Starting values from earlier hyperparameter notes (paired with a gradient-ascent
# phase there) — NEEDS SWEEPING for standalone fine_tune, no principled default yet.
DESCENT_STEPS = 50
DESCENT_LR    = 1e-4
BATCH_SIZE    = 32


def _infinite_loader(loader):
    """Cycles a DataLoader indefinitely so we can do step-based (not epoch-based)
    training — keeps budgets comparable across differently-sized retain sets."""
    while True:
        for batch in loader:
            yield batch


def fine_tune_forget(baseline_path, lt_train_indices, forget_indices, device, loss_fn,
                      n_steps=DESCENT_STEPS, lr=DESCENT_LR, batch_size=BATCH_SIZE):
    """
    Fine-tune unlearning: take the baseline model and continue training it via
    ordinary gradient descent on the retain set only (forget set excluded).
    No noise injection, no Fisher weighting — forgetting happens implicitly as
    the model's weights drift away from optimizing for the excluded samples.

    Mirrors fisher_forget's calling convention exactly (baseline_path, indices,
    device, loss_fn) so it plugs into compare_unlearning.py the same way.
    """
    model = load_model(baseline_path, device)
    model.train()

    retain_indices = np.setdiff1d(lt_train_indices, forget_indices)
    print(f"  Retain set: {len(retain_indices)} samples "
          f"(baseline's {len(lt_train_indices)} minus {len(forget_indices)} forgotten)")

    trainset = load_trainset()
    subset   = Subset(trainset, retain_indices)
    loader   = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    data_iter = _infinite_loader(loader)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"  Fine-tuning on retain set: {n_steps} steps, lr={lr}, batch_size={batch_size}...")
    for step in range(n_steps):
        inputs, labels = next(data_iter)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(inputs)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"    Step {step + 1}/{n_steps}, loss={loss.item():.4f}")

    model.eval()
    return model


def fine_tune_forget_multi_draw(
    baseline_path,
    lt_train_indices,
    forget_indices,
    device,
    loss_fn,
    n_draws=5,
    n_steps=DESCENT_STEPS,
    lr=DESCENT_LR,
    batch_size=BATCH_SIZE,
    seed_base=0,
):
    """
    Same as fine_tune_forget, but repeats the training run n_draws times, each
    with an independently seeded DataLoader shuffle (fine_tune's randomness
    comes from batch composition/order across the 50 gradient steps, not a
    noise draw like Fisher's), returning per-class accuracy for every draw
    (shape: (n_draws, 100)) so the caller can average and inspect run-to-run
    variance.

    Unlike Fisher's multi-draw (which reused one expensive shared Fisher
    computation across cheap noise draws), fine_tune has no analogous
    "expensive shared step" — each draw is a full independent 50-step training
    run starting fresh from the baseline, with its own deepcopy'd model and
    optimizer. Does not mutate baseline_path or any .npy on disk.
    """
    from evaluate.per_class_eval import evaluate_per_class  # local import to avoid cycles

    base_model = load_model(baseline_path, device)

    retain_indices = np.setdiff1d(lt_train_indices, forget_indices)
    print(f"  Retain set: {len(retain_indices)} samples "
          f"(baseline's {len(lt_train_indices)} minus {len(forget_indices)} forgotten)")

    trainset = load_trainset()
    subset   = Subset(trainset, retain_indices)

    per_class_accs = []
    for draw_idx in range(n_draws):
        seed = seed_base + draw_idx
        generator = torch.Generator()
        generator.manual_seed(seed)

        model_copy = copy.deepcopy(base_model)
        model_copy.train()

        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                             num_workers=0, generator=generator)
        data_iter = _infinite_loader(loader)
        optimizer = optim.Adam(model_copy.parameters(), lr=lr)

        print(f"  Draw {draw_idx + 1}/{n_draws} (seed={seed}): "
              f"fine-tuning {n_steps} steps, lr={lr}, batch_size={batch_size}...")
        for step in range(n_steps):
            inputs, labels = next(data_iter)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            out  = model_copy(inputs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print(f"    Step {step + 1}/{n_steps}, loss={loss.item():.4f}")

        model_copy.eval()
        per_class_acc = evaluate_per_class(model_copy, device, save_path=None)
        per_class_accs.append(per_class_acc)

        del model_copy
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return np.stack(per_class_accs, axis=0)  # shape (n_draws, 100)


# ── Canonical fine_tune method (matches compare_unlearning.py) ──────────────

# Defaults mirror compare_unlearning.py's call site exactly
# (ascent_steps=5, descent_steps=20, ascent_lr=1e-4, descent_lr=1e-4).
ASCENT_STEPS  = 5
DESCENT_STEPS_CANONICAL = 20
ASCENT_LR     = 1e-4
DESCENT_LR_CANONICAL    = 1e-4


def fine_tune_unlearn(model, forget_loader, retain_loader, device,
                       ascent_steps=ASCENT_STEPS, descent_steps=DESCENT_STEPS_CANONICAL,
                       ascent_lr=ASCENT_LR, descent_lr=DESCENT_LR_CANONICAL, loss_fn=None):
    """
    Canonical fine_tune unlearning: two phases, mutating `model` in place.

    Phase 1 — gradient ASCENT on the forget set: for `ascent_steps` steps,
    take gradient steps that INCREASE loss on forget-set batches (implemented
    by negating the loss before backward and using a normal descent
    optimizer). This actively degrades the model's ability to correctly
    classify the forget set — the actual "unlearning" step, unlike
    fine_tune_forget which has no analogous mechanism.

    Phase 2 — gradient DESCENT on the retain set: for `descent_steps` steps,
    ordinary training on retain-set batches, to recover general-utility
    performance the ascent phase likely damaged as a side effect (ascent
    updates aren't class-selective, so they can degrade retained classes too).

    Takes a live model + two DataLoaders (not baseline_path + indices) to
    match compare_unlearning.py's call site exactly. `loss_fn` is not passed
    by that call site, so it defaults to CrossEntropyLoss if left as None.

    Note: ascent_steps=5 / descent_steps=20 are compare_unlearning.py's
    current defaults — like fine_tune_forget's tunables, these are starting
    points, not swept/validated values.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model.train()

    # ── Phase 1: gradient ascent on forget set ──────────────────────────────
    forget_iter = _infinite_loader(forget_loader)
    ascent_optimizer = optim.Adam(model.parameters(), lr=ascent_lr)

    print(f"  Phase 1 (ascent): {ascent_steps} steps on forget set, lr={ascent_lr}...")
    for step in range(ascent_steps):
        inputs, labels = next(forget_iter)
        inputs, labels = inputs.to(device), labels.to(device)

        ascent_optimizer.zero_grad()
        out  = model(inputs)
        loss = loss_fn(out, labels)
        (-loss).backward()  # negate loss -> optimizer step increases it (ascent)
        ascent_optimizer.step()

        print(f"    Ascent step {step + 1}/{ascent_steps}, loss={loss.item():.4f}")

    # ── Phase 2: gradient descent on retain set ─────────────────────────────
    retain_iter = _infinite_loader(retain_loader)
    descent_optimizer = optim.Adam(model.parameters(), lr=descent_lr)

    print(f"  Phase 2 (descent): {descent_steps} steps on retain set, lr={descent_lr}...")
    for step in range(descent_steps):
        inputs, labels = next(retain_iter)
        inputs, labels = inputs.to(device), labels.to(device)

        descent_optimizer.zero_grad()
        out  = model(inputs)
        loss = loss_fn(out, labels)
        loss.backward()
        descent_optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"    Descent step {step + 1}/{descent_steps}, loss={loss.item():.4f}")

    model.eval()
    return model


if __name__ == '__main__':
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    loss_fn = nn.CrossEntropyLoss()
    lt_train_indices = np.load('data/lt_train_indices.npy')

    for strategy in ['influence', 'random']:
        for budget in [50, 100, 200]:
            print(f"\n=== Fine-tune forgetting: {strategy}, budget={budget} ===")
            forget_indices = np.load(f'data/forget_{strategy}_{budget}.npy')

            model = fine_tune_forget(
                baseline_path     = 'models/baseline.pt',
                lt_train_indices  = lt_train_indices,
                forget_indices    = forget_indices,
                device            = device,
                loss_fn           = loss_fn
            )

            out_path = f'models/unlearned_finetune_{strategy}_{budget}.pt'
            torch.save({'model_state_dict': model.state_dict()}, out_path)
            print(f"  Saved -> {out_path}")

    print("\nDone. 6 fine-tune-unlearned checkpoints saved to models/")