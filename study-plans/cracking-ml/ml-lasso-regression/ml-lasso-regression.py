import numpy as np

def lasso_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
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

        loss = (np.sum(error ** 2) + alpha * np.sum(np.sign(w))) / len(y)

        dw = (2 / len(y)) * (X.T @ error) + alpha * np.sign(w)
        db = (2 / len(y)) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    return w.tolist(), b