import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    X = np.array(X, dtype=float)
    centered = X - np.mean(X, axis=0)
    n, d = centered.shape
    cov = centered.T @ centered / (n - 1)
    std = np.sqrt(np.diag(cov))
    denominator = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        return cov / denominator