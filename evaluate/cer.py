import json
import numpy as np


def load_taxonomy(taxonomy_path: str = 'data/capability_taxonomy.json') -> dict:
    with open(taxonomy_path) as f:
        return json.load(f)


def group_by_tier(
    per_class_acc: np.ndarray,
    taxonomy: dict
) -> dict:
    """
    Groups per-class accuracies by capability tier.
    Returns dict: tier_name -> np.ndarray of accuracies
    """
    tiers = {'majority': [], 'mid_tail': [], 'long_tail': [], 'safety_critical': []}
    for cls_str, tier in taxonomy.items():
        tiers[tier].append(per_class_acc[int(cls_str)])
    return {tier: np.array(accs) for tier, accs in tiers.items()}


def compute_cer(
    baseline_acc: np.ndarray,
    unlearned_acc: np.ndarray,
    oracle_acc: np.ndarray,
    taxonomy_path: str = 'data/capability_taxonomy.json',
    eps: float = 1e-9
) -> dict:
    """
    Computes Capability Erosion Ratio per tier.

    CER(tier) = (baseline - unlearned) / (baseline - oracle + eps)

    Parameters
    ----------
    baseline_acc  : per-class acc of baseline model,  shape (100,)
    unlearned_acc : per-class acc of unlearned model, shape (100,)
    oracle_acc    : per-class acc of oracle model,    shape (100,)
    taxonomy_path : path to capability_taxonomy.json
    eps           : numerical stability

    Returns
    -------
    dict: tier -> {'cer': float, 'baseline': float,
                   'unlearned': float, 'oracle': float}
    """
    taxonomy = load_taxonomy(taxonomy_path)

    baseline_tiers  = group_by_tier(baseline_acc,  taxonomy)
    unlearned_tiers = group_by_tier(unlearned_acc, taxonomy)
    oracle_tiers    = group_by_tier(oracle_acc,    taxonomy)

    results = {}
    for tier in baseline_tiers:
        b = baseline_tiers[tier].mean()
        u = unlearned_tiers[tier].mean()
        o = oracle_tiers[tier].mean()

        cer = (b - u) / (b - o + eps)

        results[tier] = {
            'cer':       round(float(cer), 4),
            'baseline':  round(float(b),   4),
            'unlearned': round(float(u),   4),
            'oracle':    round(float(o),   4),
            'delta':     round(float(b-u), 4)   # raw accuracy drop
        }

    return results


def print_cer_report(cer_results: dict, label: str = ""):
    header = f"CER Report{' — ' + label if label else ''}"
    print(f"\n{header}")
    print("=" * 70)
    print(f"  {'Tier':<18}  {'Baseline':>9}  {'Unlearned':>9}  "
          f"{'Oracle':>9}  {'Drop':>7}  {'CER':>7}")
    print(f"  {'-'*66}")
    for tier, r in cer_results.items():
        print(f"  {tier:<18}  "
              f"{r['baseline']*100:>8.2f}%  "
              f"{r['unlearned']*100:>8.2f}%  "
              f"{r['oracle']*100:>8.2f}%  "
              f"{r['delta']*100:>6.2f}%  "
              f"{r['cer']:>7.4f}")
    print()


if __name__ == '__main__':
    # Sanity check: baseline vs oracle CER (should be ~1.0 for affected tiers)
    baseline = np.load('data/baseline_per_class_acc.npy')
    oracle   = np.load('data/oracle_per_class_acc.npy')

    # Treat oracle as the "unlearned" model to verify CER ≈ 1.0
    results = compute_cer(baseline, oracle, oracle)
    print_cer_report(results, label="Oracle as unlearned (expect CER ≈ 1.0)")