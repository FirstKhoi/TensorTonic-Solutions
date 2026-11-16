import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    # Write code here
    y = np.asarray(y)
    if num_classes is None:
        num_classes = y.max() + 1

    return np.eye(num_classes)[y]