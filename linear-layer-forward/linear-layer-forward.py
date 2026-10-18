def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    res = X @ W + b
    return res.tolist()