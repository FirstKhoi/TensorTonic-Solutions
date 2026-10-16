from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    if not x:
        return {"mean": None, "median": None, "mode": None}
    x = np.array(x, dtype=float)
    freq = Counter(x.tolist())
    most_freq = max(freq.values())
    mode = min(value for value, count in freq.items() if count == most_freq)

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(mode),
    }
    