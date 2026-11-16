import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    indices = np.arange(len(X))
    np.random.default_rng(seed).shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]

        if drop_last and len(batch_indices) < batch_size:
            break

        yield X[batch_indices], y[batch_indices]
    