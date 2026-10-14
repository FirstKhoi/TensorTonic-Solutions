import numpy as np

def dropout(x: np.ndarray, p: float, training: bool,
            mask: np.ndarray) -> np.ndarray:
    """
    Returns the float64 inverted-dropout output.
    """
    if training == True:
        x = np.asarray(x, dtype=np.float64)
        mask = np.asarray(mask, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        return (x * mask) / (1 - p)
    if training == False:
        x = x.copy().astype(np.float64)
        return x