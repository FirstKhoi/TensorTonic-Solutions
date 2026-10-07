import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))

    pos = np.arange(seq_len).reshape(-1, 1)

    two_k = np.arange(0, d_model, 2)
    
    w = 1.0 / (base ** (two_k / d_model))

    angels = pos * w

    pe[:, 0::2] = np.sin(angels)
    pe[:, 1::2] = np.cos(angels[:, : d_model // 2])

    return pe