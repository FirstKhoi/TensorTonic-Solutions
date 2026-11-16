import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    # Write code here
    X = np.asarray(X, dtype=float)

    mask = np.isnan(X)

    if strategy == "mean":
        values = np.nanmean(X, axis=0)
    elif strategy == "median":
        values = np.nanmedian(X, axis=0)

    values = np.where(np.isnan(values), 0.0, values)

    return np.where(mask, values, X)