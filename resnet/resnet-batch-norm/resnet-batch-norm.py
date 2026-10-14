
import numpy as np

def formula(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * normalized + beta


def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode
    in a dictionary.
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    gamma1 = np.array(gamma1, dtype=float)
    beta1 = np.array(beta1, dtype=float)
    gamma2 = np.array(gamma2, dtype=float)
    beta2 = np.array(beta2, dtype=float)

    x_copy = x.copy()

    if mode == "post":
        out = x @ W1
        out = formula(out, gamma1, beta1)
        out = np.maximum(0, out)

        out = out @ W2
        out = formula(out, gamma2, beta2)

        out = out + x_copy
        out = np.maximum(0, out)

    elif mode == "pre":
        out = formula(x, gamma1, beta1)
        out = np.maximum(0, out)

        out = out @ W1
        out = formula(out, gamma2, beta2)
        out = np.maximum(0, out)

        out = out @ W2
        out = out + x_copy

    else:
        raise ValueError("mode must be 'pre' or 'post'")

    return {
        "output": np.round(out, 4),
        "mode": mode
    }
