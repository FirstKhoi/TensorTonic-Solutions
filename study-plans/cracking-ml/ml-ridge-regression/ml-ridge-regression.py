import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    
    for _ in range(epochs):
        y_pred = X @ w + b

        error = y_pred - y

        dw = (2.0 / len(y)) * (X.T @ error) + 2.0 * alpha * w
        db = (2.0 / len(y)) * np.sum(error)

        w -= lr * dw
        b -= lr * db
    return w.tolist(), b
