import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    relu = np.maximum(0 ,x)
    return relu
    pass