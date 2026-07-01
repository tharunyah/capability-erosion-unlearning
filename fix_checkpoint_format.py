# fix_checkpoint_format.py
#
# Re-wraps all 6 Fisher-unlearned checkpoints from a raw state_dict
# into the {'model_state_dict': ...} format that load_model() expects.
#
# Safe to rerun: if a file is already wrapped, it's skipped, not double-wrapped.

import torch

STRATEGIES = ['influence', 'random']
BUDGETS    = [50, 100, 200]

for strategy in STRATEGIES:
    for budget in BUDGETS:
        path = f'models/unlearned_fisher_{strategy}_{budget}.pt'

        try:
            ckpt = torch.load(path, map_location='cpu')
        except FileNotFoundError:
            print(f"  SKIP (not found): {path}")
            continue

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            print(f"  Already wrapped, skipping: {path}")
            continue

        # ckpt here is the raw state_dict itself (an OrderedDict of tensors)
        torch.save({'model_state_dict': ckpt}, path)
        print(f"  Fixed: {path}")

print("\nDone.")