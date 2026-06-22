# experiments/compare_unlearning.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from data.dataloader_utils import get_forget_retain_loaders, get_cifar100_trainset
from evaluate.per_class_eval import evaluate_per_class
from evaluate.cer import compute_cer, print_cer_report
from unlearn.fine_tune import fine_tune_unlearn
from unlearn.fisher_forgetting import fisher_forgetting_unlearn


def load_model(checkpoint_path, device):
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    ckpt     = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device)


def run_experiment(
    forget_indices,
    method,
    baseline_acc,
    oracle_acc,
    device,
    label,
    results_dir = 'results'
):
    """
    Runs one unlearning experiment and computes CER.

    Parameters
    ----------
    forget_indices : np.ndarray of global training indices to forget
    method         : 'fine_tune' or 'fisher'
    label          : experiment label for saving results
    """
    os.makedirs(results_dir, exist_ok=True)

    # ── Load fresh baseline model ────────────────────────────────────────────
    model = load_model('models/baseline.pt', device)

    # ── Build forget/retain loaders ──────────────────────────────────────────
    trainset = get_cifar100_trainset()
    forget_loader, retain_loader = get_forget_retain_loaders(
        forget_indices, batch_size=128, num_workers=0, trainset=trainset
    )

    # ── Apply unlearning ─────────────────────────────────────────────────────
    print(f"\n[{label}] Applying {method} unlearning...")
    if method == 'fine_tune':
        unlearned_model = fine_tune_unlearn(
            model, forget_loader, retain_loader, device,
            ascent_steps=5, descent_steps=20,
            ascent_lr=1e-4, descent_lr=1e-4
        )
    elif method == 'fisher':
        unlearned_model = fisher_forgetting_unlearn(
            model, forget_loader, device,
            noise_scale=1e-1, num_batches=10
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Evaluate per-class accuracy ──────────────────────────────────────────
    print(f"  Evaluating unlearned model...")
    save_path     = f"{results_dir}/{label}_per_class_acc.npy"
    unlearned_acc = evaluate_per_class(unlearned_model, device, save_path=save_path)

    # ── Compute CER ──────────────────────────────────────────────────────────
    cer_results = compute_cer(baseline_acc, unlearned_acc, oracle_acc)
    print_cer_report(cer_results, label=label)

    # ── Save CER results ─────────────────────────────────────────────────────
    cer_path = f"{results_dir}/{label}_cer.json"
    with open(cer_path, 'w') as f:
        json.dump(cer_results, f, indent=2)
    print(f"  CER saved to {cer_path}")

    return cer_results


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load reference accuracies ────────────────────────────────────────────
    baseline_acc = np.load('data/baseline_per_class_acc.npy')
    oracle_acc   = np.load('data/oracle_per_class_acc.npy')

    # ── Experiment grid ──────────────────────────────────────────────────────
    budgets    = [50, 100, 200]
    strategies = ['influence', 'random']
    methods    = ['fine_tune', 'fisher']

    all_results = {}

    for budget in budgets:
        for strategy in strategies:
            forget_indices = np.load(f'data/forget_{strategy}_{budget}.npy')

            for method in methods:
                label = f"{strategy}_b{budget}_{method}"
                print(f"\n{'='*60}")
                print(f"Experiment: {label}")
                print(f"  Forget set size : {len(forget_indices)}")
                print(f"  Strategy        : {strategy}")
                print(f"  Budget          : {budget}")
                print(f"  Method          : {method}")

                cer_results = run_experiment(
                    forget_indices = forget_indices,
                    method         = method,
                    baseline_acc   = baseline_acc,
                    oracle_acc     = oracle_acc,
                    device         = device,
                    label          = label
                )

                all_results[label] = cer_results

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FULL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<35} {'long_tail CER':>14} {'safety_crit CER':>16}")
    print(f"{'-'*65}")
    for label, cer in all_results.items():
        lt  = cer['long_tail']['cer']
        sc  = cer['safety_critical']['cer']
        print(f"{label:<35} {lt:>14.4f} {sc:>16.4f}")

    # Save full summary
    with open('results/full_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nFull summary saved to results/full_summary.json")