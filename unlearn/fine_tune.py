# unlearn/fine_tune.py
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