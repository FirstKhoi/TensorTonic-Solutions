import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    if x.ndim == 2:
        reduce_axes = 0
    else:
        reduce_axes = (0, 2, 3)
        # Gán lại kết quả reshape vào biến
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

    mean_val = np.mean(x, axis=reduce_axes, keepdims=True)
    variance_val = np.var(x, axis=reduce_axes, keepdims=True)
    normalized = (x - mean_val) / np.sqrt(variance_val + eps)

    return normalized * gamma + beta