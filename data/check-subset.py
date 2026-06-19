# check_subset.py
import numpy as np

lt_indices = np.load('data/lt_train_indices.npy')
print(f"Subset size: {len(lt_indices)}")
print(f"Min index: {lt_indices.min()}, Max index: {lt_indices.max()}")