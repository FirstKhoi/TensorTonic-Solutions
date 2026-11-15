import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Write code here
    x = np.asarray(X, dtype=float)
    x_c = x - np.mean(x, axis=0, keepdims=True)
    c = (x_c.T @ x_c)/x.shape[0]
    # Extracting the vectors only
    vectors = np.linalg.eigh(c)[1]

    # Reverese the columns
    vectors = vectors[:, ::-1]
    # fetch the top k elements
    w = vectors[:, :k]
    return x_c @ w