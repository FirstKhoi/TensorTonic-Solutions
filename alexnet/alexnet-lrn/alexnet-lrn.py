import numpy as np

def local_response_normalization(x: np.ndarray, k: float, n: int,
                                 alpha: float, beta: float) -> np.ndarray:
    """
    Returns the float64 channel-normalized tensor.
    """
    output = np.empty_like(x, dtype=np.float64)
    radius = n // 2
    channels = x.shape[-1]
    for channel in range(channels):
        start = max(0, channel - radius)
        end = min(channels, channel + radius + 1)
        square_sum = np.sum(x[..., start:end] ** 2, axis=-1)
        output[...,channel] = x[...,channel] / (k + alpha * square_sum  / n) ** beta
    return output