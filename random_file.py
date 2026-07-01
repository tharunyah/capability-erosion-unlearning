# this is a test file to run some testing python scripts

import numpy as np

for strategy in ["influence", "random"]:
    f50  = set(np.load(f"data/forget_{strategy}_50.npy").tolist())
    f100 = set(np.load(f"data/forget_{strategy}_100.npy").tolist())
    f200 = set(np.load(f"data/forget_{strategy}_200.npy").tolist())

    print(f"--- {strategy} ---")
    print("50 ⊆ 100:", f50.issubset(f100))
    print("100 ⊆ 200:", f100.issubset(f200))
    print("50 ⊆ 200:", f50.issubset(f200))
    print(f"sizes: {len(f50)}, {len(f100)}, {len(f200)}")