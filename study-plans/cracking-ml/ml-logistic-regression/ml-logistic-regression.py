import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    
    for _ in range(n_iters):
        z = X @ w + b
        
        y_pred = sigmoid(z)

        error = y_pred - y

        eps = 1e-15
        loss = -np.mean(
            y * np.log(y_pred + eps)
            + (1 - y) * np.log(1 - y_pred + eps)
        )

        dl_dw = (X.T @ error) / len(y)
        dl_db = np.sum(error) / len(y)

        w -= lr*dl_dw
        b -= lr*dl_db

    return w, b

        
        