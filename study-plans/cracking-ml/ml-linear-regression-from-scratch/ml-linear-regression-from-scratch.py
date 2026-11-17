import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    w = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0

    for _ in range(epochs):
        y_pred = np.dot(X, w) + bias
        error = y_pred - y
        loss = np.mean(np.sum(error) ** 2)

        dL_dw = (2 / len(y)) * (X.T @ error)
        dL_db = (2 / len(y)) * np.sum(error)
        
        w -= lr * dL_dw
        bias -= lr * dL_db

    return w, bias
