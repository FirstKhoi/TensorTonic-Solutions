import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> dict:
    """
    Returns input_gate and candidate_state as float64 arrays.
    """
    grouped = np.concatenate([h_prev, x_t], axis=-1)
    branch_1 = sigmoid(grouped @ W_i.T + b_i).astype(np.float64)

    branch_2 = np.tanh(grouped @ W_c.T + b_c).astype(np.float64)

    return{
        "input_gate": branch_1, "candidate_state": branch_2
    }