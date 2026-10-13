import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    x = np.array(x, dtype=np.float32)
    W1 = np.array(W1, dtype=np.float32)
    W2 = np.array(W2, dtype=np.float32)
    Ws = np.array(Ws, dtype=np.float32)

    relu1 = np.maximum(0, x@W1)
    relu2 = np.maximum(0, relu1@W2 + x@Ws)

    return relu2