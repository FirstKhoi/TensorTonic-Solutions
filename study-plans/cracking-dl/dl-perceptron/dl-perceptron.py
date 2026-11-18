import numpy as np

def perceptron(X: list, y: list, lr: float = 0.1, epochs: int = 100) -> tuple:
    """
    Returns the trained weight list and bias.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0

    for _ in range(epochs):
        for x_i, y_i in zip(X, y):
            z = w.T @ x_i + b

            y_pred = 1.0 if z >= 0 else 0.0

            error = y_pred - y_i
            if error != 0:
                w -= lr * error * x_i
                b -= lr * error

    return w.tolist(), b