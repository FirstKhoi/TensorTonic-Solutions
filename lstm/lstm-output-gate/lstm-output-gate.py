import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns output_gate and hidden_state as float64 arrays.
    """
    grouped = np.concatenate([h_prev, x_t], axis=-1)

    ot = sigmoid(grouped @ W_o.T + b_o).astype(np.float64)

    ht = ot * np.tanh(C_t).astype(np.float64)

    return {
        "output_gate": ot, "hidden_state": ht
    }

    