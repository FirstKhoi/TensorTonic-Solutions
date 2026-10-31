import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns hidden_state and cell_state as float64 arrays.
    """
    grouped = np.concatenate([h_prev, x_t], axis=-1)

    forget_gate = sigmoid(grouped @ W_f.T + b_f).astype(np.float64)

    input_gate = sigmoid(grouped @ W_i.T + b_i).astype(np.float64)

    cell_candidate = np.tanh(grouped @ W_c.T + b_c).astype(np.float64)

    cell_update = forget_gate * C_prev + input_gate * cell_candidate

    output_gate = sigmoid(grouped @ W_o.T + b_o).astype(np.float64)

    hidden_state = output_gate * np.tanh(cell_update)

    return {
        "hidden_state": hidden_state, "cell_state": cell_update
    }