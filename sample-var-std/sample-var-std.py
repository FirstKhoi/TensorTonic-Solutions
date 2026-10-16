import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    if not x or len(x) < 2:
        return None
    # Write code here

    return {
        "variance": float(np.var(x, ddof=1)),
        "standard_deviation": float(np.std(x, ddof=1))
    }
