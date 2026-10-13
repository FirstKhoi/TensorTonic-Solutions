import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.array(x, dtype=np.float32)
    W1 = np.array(W1, dtype=np.float32)
    W2 = np.array(W2, dtype=np.float32)
    W3 = np.array(W3, dtype=np.float32)
    Ws = np.array(Ws, dtype=np.float32)

    relu1 = np.maximum(0, x@W1)
    relu2 = np.maximum(0, relu1@W2)
    relu3 = np.maximum(0, relu2@W3 + x@Ws)

    return relu3