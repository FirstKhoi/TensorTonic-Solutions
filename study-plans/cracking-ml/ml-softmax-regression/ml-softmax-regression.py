import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    w = np.zeros((X.shape[1], num_classes), dtype=np.float64)

    bias = np.zeros(num_classes, dtype=float)
    
    targets = np.eye(num_classes, dtype=float)[y]

    for _ in range(n_iters):
        z = X @ w + bias
        
        numerator = np.exp(z - np.max(z, axis=1, keepdims=True))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        
        proba = numerator/denominator

        error = proba - targets

        w -= lr * (X.T @ error) / len(y)
        bias -= lr * np.mean(error, axis=0)

    return w, bias

        

    
    
