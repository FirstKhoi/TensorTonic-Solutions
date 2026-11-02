import numpy as np

def candidate_hidden(r_t: np.ndarray, h_prev: np.ndarray, x_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Returns the float64 candidate hidden state.
    """
    concatenated = np.concatenate([r_t * h_prev, x_t], axis=-1)

    return np.tanh(concatenated @ W_h.T + b_h).astype(np.float64)