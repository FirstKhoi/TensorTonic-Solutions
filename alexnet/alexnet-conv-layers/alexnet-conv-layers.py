import numpy as np

def alexnet_conv1(image: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                   stride: int, padding: int) -> np.ndarray:
    """
    Returns the float64 NHWC convolution output.
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    #define shape
    N, H, W, C = image.shape
    K_h, K_w, C_in, C_out = kernel.shape

    #padding processing
    padded = np.pad(image, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1
    out = np.zeros((N, H_out, W_out, C_out), dtype=np.float64)

    for i in range(H_out):
        for j in range(W_out):
            patch = padded[:, i * stride:i * stride + K_h, j * stride:j * stride + K_w, :]
            out[:, i, j, :] = np.tensordot(patch, kernel, axes=((1, 2, 3), (0, 1, 2))) + bias
    return out