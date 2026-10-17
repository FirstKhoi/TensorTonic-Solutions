import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    # Write code here
    predicts = np.array(y_pred, dtype=float)
    targets = np.array(y_true, dtype=float)
    return float(np.mean((predicts - targets) ** 2))