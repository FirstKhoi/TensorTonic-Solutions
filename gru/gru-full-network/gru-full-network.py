import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)
        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> dict:
        """
        Returns outputs and final_hidden_state as float64 arrays.
        """
        batch, steps, _ = X.shape
        hidden = np.zeros((batch, self.hidden_dim), dtype=np.float64)
        outputs = []
        for step in range(steps):
            x_t = X[:, step, :]
            joined = np.concatenate([hidden, x_t], axis=-1)
            reset = sigmoid(joined @ self.W_r.T + self.b_r)
            update = sigmoid(joined @ self.W_z.T + self.b_z)
            candidate_input = np.concatenate([reset * hidden, x_t], axis=-1)
            candidate = np.tanh(candidate_input @ self.W_h.T + self.b_h)
            hidden = update * hidden + (1.0 - update) * candidate
            outputs.append(hidden @ self.W_y.T + self.b_y)
        return {"outputs": np.stack(outputs, axis=1), "final_hidden_state": hidden}