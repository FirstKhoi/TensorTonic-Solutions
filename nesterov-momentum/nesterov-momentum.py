import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    # Write code here
    w = np.asarray(w, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)

    v_new = momentum * v + lr * grad

    w_new = w - v_new

    return {
        "new_w": w_new, "new_v": v_new
    }