import numpy as np

lt = np.load('data/lt_train_indices.npy')
print("lt_train_indices:", lt.shape, lt[:10], "min:", lt.min(), "max:", lt.max())

placeholder = np.load('data/forget_set_placeholder.npy', allow_pickle=True)
print("forget_set_placeholder:", placeholder.shape, placeholder.dtype)
print(placeholder[:5] if len(placeholder) > 0 else "empty")