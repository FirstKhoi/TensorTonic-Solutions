import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return float(x @ y)