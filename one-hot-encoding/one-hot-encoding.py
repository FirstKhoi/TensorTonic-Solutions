import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    # Write code here
    if num_classes is None:
        num_classes = max(y) + 1
    zeros = np.zeros((len(y), num_classes), dtype=float)
    for i, j in enumerate(y):
        zeros[i, j] = 1

    return zeros