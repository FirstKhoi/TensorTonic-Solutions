import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    # Write code here
    W = np.array(W, dtype=float)
    L = np.sqrt(6 / fan_in)

    for i in range(len(W)):
        for j in range(len(W[0])):
            W_ij = W * 2 * L - L

    return W_ij