import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    N, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = W.shape

    H_out = H_in - K_h + 1
    W_out = W_in - K_w + 1

    output = np.empty((N, C_out, H_out, W_out), dtype=np.float64)

    for c_out in range(C_out):
        kernel = W[c_out]
        bias = b[c_out]

        for i in range(H_out):
            for j in range(W_out):
                window = x[:, :, i : i + K_h, j : j + K_w]
                
                output[:, c_out, i, j] = np.sum(window * kernel, axis=(1, 2, 3)) + bias

    return output