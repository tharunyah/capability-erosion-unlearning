import numpy as np
lt_train = np.load('data/lt_train_indices.npy')
forget_50 = np.load('data/forget_influence_50.npy')
print(np.isin(forget_50, lt_train).all())  # should be True