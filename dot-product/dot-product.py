import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    res = np.dot(x, y)
    return float(res)
    