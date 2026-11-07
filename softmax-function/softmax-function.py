import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    x = np.array(x, dtype=float)
 
    numerator = np.exp(x - np.max(x, axis=-1, keepdims=True))
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator