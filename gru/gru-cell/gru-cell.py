import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Returns the float64 next hidden state.
    """

    concatenated = np.concatenate([h_prev, x_t], axis=-1)

    reset_gate = sigmoid(concatenated @ W_r.T + b_r)

    update_gate = sigmoid(concatenated @ W_z.T + b_z)

    concatenated_ht = np.concatenate([reset_gate * h_prev, x_t], axis=-1)
    
    canidate_hidden = np.tanh(concatenated_ht @ W_h.T + b_h)
    
    hidden_state = update_gate * h_prev + (1 - update_gate) * canidate_hidden

    return hidden_state.astype(np.float64)