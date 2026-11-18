import numpy as np

def pooling(input: list, pool_size: int, stride: int, pool_type: str) -> list:
    """
    Returns the pooled channel-first tensor.
    """
    input = np.array(input, dtype=float)
    C, H, W = input.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    output = np.empty((C, out_h, out_w), dtype=float)
    for c in range(C):
        for rows in range(out_h):
            for cols in range(out_w):
                window = input[c, rows * stride : rows * stride + pool_size, cols * stride : cols * stride + pool_size]
                output[c, rows, cols] = np.max(window) if pool_type == "max" else np.mean(window)
    return np.round(output, 4).tolist()
            