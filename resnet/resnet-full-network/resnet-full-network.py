import numpy as np

def conv2d(x, kernel):
    x = np.array(x, dtype=float)
    kernel = np.array(kernel, dtype=float)

    squeeze = x.ndim == 2 and kernel.ndim == 2
    
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    if kernel.ndim == 2:
        kernel = kernel[:, :, np.newaxis, np.newaxis]

    H, W, C_in = x.shape
    K_h, K_w, _, C_out = kernel.shape

    H_out = H - K_h + 1
    W_out = W - K_w + 1

    out = np.zeros((H_out, W_out, C_out))

    for i in range(H_out):
        for j in range(W_out):
            patch = x[i:i + K_h, j:j + K_w, :]
            for c in range(C_out):
                out[i, j, c] = np.sum(patch * kernel[:, :, :, c])

    return out[:, :, 0] if squeeze else out
    
def relu(x):
    return np.maximum(0, x)

def residual_block(x, W1, W2):
    shortcut = x.copy()

    out1 = relu(x@W1)
    
    return relu(out1@W2 + shortcut)
    
def projection_residual_block(x, W1, W2, Ws):
    shortcut = x@Ws

    out1 = relu(x@W1)
    return relu(out1@W2 + shortcut)

def fully_connected(x, fc):
    return x @ fc


def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    x = np.array(x, dtype=float)
    conv1 = np.array(conv1, dtype=float)

    out = relu(x @ conv1)
    out = residual_block(out, W1_b1, W2_b1)
    out = projection_residual_block(out, W1_b2, W2_b2, Ws_b2)

    if out.ndim == 3:
        out = out.mean(axis=(0, 1))

    logits = fully_connected(out, fc)
    return logits.tolist()