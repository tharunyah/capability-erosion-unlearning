# experiments/compare_unlearning.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from evaluate.per_class_eval import evaluate_per_class
from unlearn.fine_tune import fine_tune_forget
from unlearn.fisher_forgetting import fisher_forget

REFERENCE_TIER = "majority"


def load_model(checkpoint_path, device):
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    ckpt     = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device)


def load_tiers(taxonomy_path='data/capability_taxonomy.json'):
    """
    capability_taxonomy.json is {class_id_str: tier_name}. Inverts it to
    {tier_name: [class_id_int, ...]} for per-tier accuracy averaging.
    """
    tax = json.load(open(taxonomy_path))
    tiers = {}
    for class_id_str, tier in tax.items():
        tiers.setdefault(tier, []).append(int(class_id_str))
    return tiers


def compute_di_report(baseline_acc, unlearned_acc, tiers, reference_tier=REFERENCE_TIER):
    """
    Feldman et al. (2015) Disparate Impact, per tier, before vs after unlearning.
    Mirrors the same formula used in experiments/test_auditor_multimethod.py,
    but operates on full 100-class per_class_acc arrays averaged per tier,
    rather than pre-averaged CSV rows.
    """
    ref_before = baseline_acc[tiers[reference_tier]].mean()
    ref_after  = unlearned_acc[tiers[reference_tier]].mean()

    report = {}
    for tier, classes in tiers.items():
        tgt_before = baseline_acc[classes].mean()
        tgt_after  = unlearned_acc[classes].mean()
        di_before  = tgt_before / (ref_before + 1e-8)
        di_after   = tgt_after / (ref_after + 1e-8)
        report[tier] = {
            "baseline_acc": float(tgt_before),
            "unlearned_acc": float(tgt_after),
            "di_before": float(di_before),
            "di_after": float(di_after),
            "di_shift": float(di_after - di_before),
        }
    return report


def print_di_report(di_report, label):
    print(f"\n  DI report for {label} (reference tier: {REFERENCE_TIER}):")
    print(f"  {'tier':<18}{'baseline':>10}{'unlearned':>11}{'di_before':>11}{'di_after':>10}{'di_shift':>10}")
    for tier, v in di_report.items():
        print(f"  {tier:<18}{v['baseline_acc']:>10.4f}{v['unlearned_acc']:>11.4f}"
              f"{v['di_before']:>11.4f}{v['di_after']:>10.4f}{v['di_shift']:>10.4f}")


def run_experiment(
    forget_indices,
    method,
    baseline_acc,
    tiers,
    device,
    label,
    lt_train_indices,
    loss_fn,
    results_dir = 'results'
):
    """
    Runs one unlearning experiment and computes Disparate Impact (DI) per tier.

    Parameters
    ----------
    forget_indices    : np.ndarray of global training indices to forget
    method            : 'fine_tune' or 'fisher'
    label             : experiment label for saving results
    lt_train_indices  : full long-tail-distributed training index set
                         (baseline's actual training data; used by both
                         fine_tune_forget and fisher to derive their retain set)
    loss_fn           : loss function fisher's per-sample Fisher pass needs
                         (fine_tune_forget also takes loss_fn, for its
                         retain-set descent steps)
    """
    os.makedirs(results_dir, exist_ok=True)

    # ── Apply unlearning ─────────────────────────────────────────────────────
    print(f"\n[{label}] Applying {method} unlearning...")
    if method == 'fine_tune':
        # fine_tune_forget: retain-set-only descent (no ascent phase). This is
        # the method with the full established analysis trail (original CSV,
        # multidraw, corrected-std, auditor test) -- kept as the project's
        # canonical "fine_tune" method for consistency with prior results.
        unlearned_model = fine_tune_forget(
            baseline_path     = 'models/baseline.pt',
            lt_train_indices  = lt_train_indices,
            forget_indices    = forget_indices,
            device            = device,
            loss_fn           = loss_fn
        )
    elif method == 'fisher':
        unlearned_model = fisher_forget(
            baseline_path     = 'models/baseline.pt',
            lt_train_indices  = lt_train_indices,
            forget_indices    = forget_indices,
            device            = device,
            loss_fn           = loss_fn
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Evaluate per-class accuracy ──────────────────────────────────────────
    print(f"  Evaluating unlearned model...")
    save_path     = f"{results_dir}/{label}_per_class_acc.npy"
    unlearned_acc = evaluate_per_class(unlearned_model, device, save_path=save_path)

    # ── Compute DI ────────────────────────────────────────────────────────────
    di_report = compute_di_report(baseline_acc, unlearned_acc, tiers)
    print_di_report(di_report, label=label)

    # ── Save DI results ───────────────────────────────────────────────────────
    di_path = f"{results_dir}/{label}_di.json"
    with open(di_path, 'w') as f:
        json.dump(di_report, f, indent=2)
    print(f"  DI saved to {di_path}")

    return di_report


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load reference accuracies / tier map ─────────────────────────────────
    baseline_acc     = np.load('data/baseline_per_class_acc.npy')
    lt_train_indices = np.load('data/lt_train_indices.npy')
    tiers            = load_tiers()
    loss_fn          = nn.CrossEntropyLoss()

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

                di_report = run_experiment(
                    forget_indices    = forget_indices,
                    method            = method,
                    baseline_acc      = baseline_acc,
                    tiers             = tiers,
                    device            = device,
                    label             = label,
                    lt_train_indices  = lt_train_indices,
                    loss_fn           = loss_fn
                )

                all_results[label] = di_report

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FULL RESULTS SUMMARY (DI shift per tier)")
    print(f"{'='*60}")
    print(f"{'Experiment':<25}{'majority':>12}{'mid_tail':>12}{'long_tail':>12}{'safety_crit':>13}")
    print(f"{'-'*74}")
    for label, di in all_results.items():
        maj = di['majority']['di_shift']
        mid = di['mid_tail']['di_shift']
        lt  = di['long_tail']['di_shift']
        sc  = di['safety_critical']['di_shift']
        print(f"{label:<25}{maj:>12.4f}{mid:>12.4f}{lt:>12.4f}{sc:>13.4f}")

    # Save full summary
    with open('results/full_summary_di.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nFull summary saved to results/full_summary_di.json")