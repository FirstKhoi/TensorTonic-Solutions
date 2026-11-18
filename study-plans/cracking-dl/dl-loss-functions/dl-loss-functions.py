import numpy as np

def loss_functions(y_true: list, y_pred: list, loss_type: str) -> float:
    """
    Returns the selected mean loss.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1.0-1e-15)
    if loss_type == "mse":
        return np.mean((y_pred - y_true)** 2)

    
    elif loss_type == "bce":
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    
    elif loss_type == "hinge":
        return np.mean(np.maximum(0, 1 - y_pred * y_true))

    
    elif loss_type == "cce":
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true.astype(int)]
            
        numerator = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        proba = numerator / denominator
        
        loss = -np.mean(np.sum(y_true * np.log(proba), axis=1))
        return round(float(loss), 4)

    
    else:
        return -1.0

    return float(loss)