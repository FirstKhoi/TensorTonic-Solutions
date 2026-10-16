import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    # Write code here
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)

    pmf = (p ** x) * ((1.0 - p) ** (1.0 - x))
    mean = float(p)
    var_vals = float(p * (1 - p))

    return {
        "pmf": pmf,
        "mean": mean,
        "variance": var_vals
    }

    
    