import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    m, n = X.shape
    y = y.reshape(-1)
    
    w = np.zeros(n)
    b = 0.0
    
    for _ in range(steps):
        y_pred = _sigmoid(np.dot(X, w) + b)
        error = y_pred - y
        
        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)
        
        w -= lr * dw
        b -= lr * float(db)
        
    return w, float(b)