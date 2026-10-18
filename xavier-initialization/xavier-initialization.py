import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    # Write code here
    W = np.asarray(W, dtype=float)
    L = np.sqrt(6 / (fan_in + fan_out))

    return [[float(w * 2 * L - L) for w in row] for row in W]