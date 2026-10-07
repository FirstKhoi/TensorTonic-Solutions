import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    T = 10000.0
    
    pe = np.zeros((seq_length, d_model))

    pos = np.arange(seq_length).reshape(-1, 1)

    two_k = np.arange(0, d_model, 2)

    w = 1.0 / (T ** (two_k / d_model))

    angles = pos * w

    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)

    return pe