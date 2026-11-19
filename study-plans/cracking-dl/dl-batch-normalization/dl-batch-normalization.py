import numpy as np

def batch_norm(X: list, gamma: list, beta: list, running_mean: list, running_var: list, mode: str) -> dict:
    """
    Returns normalized output and resulting running statistics.
    """
    X = np.array(X, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)
    running_mean = np.array(running_mean, dtype=float)
    running_var = np.array(running_var, dtype=float)

    if mode == "train":
        mean, variance = np.mean(X, axis=0), np.var(X, axis=0, ddof=0)
        
        x_hat = (X - mean) / (np.sqrt(variance + 1e-5))
    
        y = gamma * x_hat + beta
    
        running_mean = 0.9 * running_mean + 0.1 * mean
    
        running_var = 0.9 * running_var + 0.1 * variance
    else:
        y = gamma * (X - running_mean) / np.sqrt(running_var + 1e-5) + beta
    return {
        "output": y, "running_mean": running_mean, "running_var": running_var
    }