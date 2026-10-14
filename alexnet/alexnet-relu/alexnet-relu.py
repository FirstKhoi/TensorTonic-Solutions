import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Returns the elementwise float64 ReLU output.
    """
    return np.maximum(0, x)