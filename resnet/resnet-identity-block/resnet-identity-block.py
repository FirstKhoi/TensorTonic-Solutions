import numpy as np

def identity_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    
    first_formula = x @ W1.T
    relu1 = np.maximum(0.0, first_formula)
    relu2 = np.maximum(0.0, relu1 @ W2.T + x)
    return relu2