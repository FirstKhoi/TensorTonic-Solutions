import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    N = X.shape[0]

    if X.ndim != 2 or N < 2:
        return None

    X_centered = X - np.mean(X, axis=0, keepdims=True)

    cov = (X_centered.T @ X_centered) / (N - 1)

    return cov