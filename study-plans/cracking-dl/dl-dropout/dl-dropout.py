import numpy as np

def dropout(X: list, mask: list, drop_prob: float, mode: str) -> list:
    """
    Returns the train-mode dropout result or unchanged test input.
    """
    if mode == "train":
        
        X = np.array(X, dtype=float)
        mask = np.array(mask, dtype=np.float64)
    
        y = (X * mask) / (1 - drop_prob)
        return y.tolist()
    return X