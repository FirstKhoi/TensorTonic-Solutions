import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
    """
    Returns the float64 NHWC max-pooled tensor.
    """
    N, H, W, C = x.shape

    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    output = np.empty((N, H_out, W_out, C), dtype=np.float64)

    for i in range(H_out):
        h_start = i * stride
        h_end = h_start + kernel_size

        for j in range(W_out):
            w_start = j * stride
            w_end = w_start + kernel_size

            output[:, i, j, :] = np.max(x[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
    return output
        