# inspect_oracle.py
#
# We only have ONE oracle.pt file, but the project plan called for FOUR
# (oracle_influence_200, oracle_random_50/100/200 -- one per forget-set
# config). This script doesn't assume anything about what oracle.pt actually
# is -- it loads it, evaluates it, and compares it against the baseline and
# against each forget set's retain-set size, so we can see from the numbers
# which (if any) config it plausibly corresponds to, rather than guessing
# from the filename alone.

import os
import json
import numpy as np
import torch

from evaluate.per_class_eval import load_model, evaluate_per_class


def load_tier_map(taxonomy_path='data/capability_taxonomy.json'):
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    tiers = {'majority': [], 'mid_tail': [], 'long_tail': [], 'safety_critical': []}
    for cls_str, tier in taxonomy.items():
        tiers[tier].append(int(cls_str))
    return tiers


def tier_accuracy(per_class_acc, class_indices):
    return float(np.mean(per_class_acc[class_indices]))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Basic checkpoint inspection ──────────────────────────────────────────
    ckpt_path = 'models/oracle.pt'
    print(f"Loading raw checkpoint dict from {ckpt_path} (not via load_model yet)...")
    raw_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"  Checkpoint keys: {list(raw_ckpt.keys())}")
    if 'model_state_dict' in raw_ckpt:
        n_params = sum(p.numel() for p in raw_ckpt['model_state_dict'].values())
        print(f"  model_state_dict param count: {n_params:,}")
    # Print any non-model metadata that might hint at provenance -- summarize,
    # don't dump raw tensors (optimizer state especially is huge and useless
    # to look at directly).
    for k, v in raw_ckpt.items():
        if k == 'model_state_dict':
            continue
        if isinstance(v, torch.Tensor):
            print(f"  Extra field '{k}': tensor, shape={tuple(v.shape)}, value={v.item() if v.numel()==1 else '...'}")
        elif isinstance(v, dict):
            print(f"  Extra field '{k}': dict with {len(v)} keys (likely optimizer state, skipping detail)")
        elif isinstance(v, (int, float, str, bool)):
            print(f"  Extra field '{k}': {v}")
        else:
            print(f"  Extra field '{k}': {type(v).__name__} (skipping detail)")

    # ── Evaluate oracle + baseline ───────────────────────────────────────────
    print(f"\nEvaluating oracle.pt per-class accuracy...")
    oracle_model = load_model(ckpt_path, device)
    oracle_acc = evaluate_per_class(oracle_model, device, save_path=None)
    overall_oracle = float(np.mean(oracle_acc))

    print(f"Evaluating baseline.pt per-class accuracy for comparison...")
    baseline_model = load_model('models/baseline.pt', device)
    baseline_acc = evaluate_per_class(baseline_model, device, save_path=None)
    overall_baseline = float(np.mean(baseline_acc))

    print(f"\nOverall: baseline={overall_baseline:.4f}  oracle={overall_oracle:.4f}  "
          f"diff={overall_oracle - overall_baseline:+.4f}")

    if abs(overall_oracle - overall_baseline) < 1e-4:
        print("  *** WARNING: oracle is numerically identical to baseline. ***")
        print("  *** This suggests oracle.pt may just BE the baseline, not a real retrain. ***")

    # ── Per-tier comparison ───────────────────────────────────────────────────
    tiers = load_tier_map()
    print(f"\n{'Tier':<18}{'baseline':>10}{'oracle':>10}{'diff':>10}")
    for tier_name, class_indices in tiers.items():
        b = tier_accuracy(baseline_acc, class_indices)
        o = tier_accuracy(oracle_acc, class_indices)
        print(f"{tier_name:<18}{b:>10.4f}{o:>10.4f}{o - b:>+10.4f}")

    # ── Compare against each forget set's long_tail-only accuracy shift ─────
    # An oracle trained excluding a specific forget set should show its
    # LARGEST long_tail drop relative to baseline for THAT config -- if one
    # config's forget set size lines up with a visibly bigger long_tail drop
    # than the others, that's a clue (not proof) about which oracle this is.
    print(f"\nFor reference, forget-set sizes (long_tail-drawn, from lt_pure_indices.npy):")
    for strategy in ['influence', 'random']:
        for budget in [50, 100, 200]:
            path = f'data/forget_{strategy}_{budget}.npy'
            if os.path.exists(path):
                forget_indices = np.load(path)
                print(f"  forget_{strategy}_{budget}.npy: {len(forget_indices)} samples")
            else:
                print(f"  forget_{strategy}_{budget}.npy: NOT FOUND")

    # ── Save for downstream use, but ONLY as a flat/shared oracle ───────────
    save_path = 'data/oracle_per_class_acc.npy'
    np.save(save_path, oracle_acc)
    print(f"\nSaved oracle per-class accuracy -> {save_path}")
    print("NOTE: this is the ONE oracle you have. If it doesn't correspond to")
    print("all 6 forget-set configs, CER computed against it will be wrong for")
    print("whichever configs it doesn't match. Confirm with Tharunyah which")
    print("forget set (if any) this oracle was actually trained to exclude.")


if __name__ == '__main__':
    main()