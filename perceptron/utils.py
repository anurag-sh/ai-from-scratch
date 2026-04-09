# perceptron/utils.py

import numpy as np
import random
from sklearn.datasets import make_classification

# Set seeds for reproducibility - must be set before any random operations
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def generate_and_split_data(n=50):
    X, Y = make_classification(
        n_samples=n,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        flip_y=0
    )

    # Convert labels (optional depending on your model)
    Y = Y * 2 - 1   # remove this if using 0/1 perceptron

    # Set data types
    X = X.astype(np.float32)
    Y = Y.astype(np.int32)

    # Split dataset (80% train, 20% test)
    split_idx = n * 8 // 10

    train_x, test_x = np.split(X, [split_idx])
    train_y, test_y = np.split(Y, [split_idx])

    return train_x, train_y, test_x, test_y