import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    x = np.array(x, dtype=float)
    if x.ndim == 1:
        shifted = x - np.max(x)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)

    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)