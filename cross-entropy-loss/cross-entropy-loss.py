import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    y_pred_arr = np.array(y_pred, dtype=float)
    y_true_arr = np.array(y_true, dtype=int)
    
    # Clip probabilities to prevent log(0) numerical instability
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_arr, eps, 1 - eps)
    
    # Extract predicted probability for the true class of each sample
    n_samples = len(y_true_arr)
    correct_class_probs = y_pred_clipped[np.arange(n_samples), y_true_arr]
    
    # Mean negative log-likelihood
    loss = -np.mean(np.log(correct_class_probs))
    
    return float(loss)