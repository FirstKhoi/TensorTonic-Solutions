import numpy as np

def sigmoid(z):
    val = 1.0 / (1.0 + np.exp(-z))
    grad = val * (1.0 - val)
    return float(val), float(grad)

def tanh_act(z):
    t = np.tanh(z)
    grad = 1.0 - t ** 2
    return float(t), float(grad)

def relu(z):
    val = np.maximum(0.0, z)
    grad = 1.0 if z > 0 else 0.0
    return float(val), float(grad)

def gelu(z):
    k = np.sqrt(2.0 / np.pi)
    u = k * (z + 0.044715 * (z ** 3))
    tanh_u = np.tanh(u)
    
    val = 0.5 * z * (1.0 + tanh_u)
    du_dz = k * (1.0 + 3.0 * 0.044715 * (z ** 2))
    grad = 0.5 * (1.0 + tanh_u) + 0.5 * z * (1.0 - tanh_u ** 2) * du_dz
    
    return float(val), float(grad)

def leaky_relu(z, alpha: float = 0.01):
    val = z if z > 0 else alpha * z
    grad = 1.0 if z > 0 else alpha
    return float(val), float(grad)

def swish(z):
    sig, _ = sigmoid(z)
    val = z * sig
    grad = val + sig * (1.0 - val)
    return float(val), float(grad)

def activation_functions(x: float, activation: str) -> list:
    """
    Returns the activation value and analytical derivative.
    """
    x = float(x)

    if activation == "relu":
        val, grad = relu(x)
    elif activation == "sigmoid":
        val, grad = sigmoid(x)
    elif activation == "tanh":
        val, grad = tanh_act(x)
    elif activation == "gelu":
        val, grad = gelu(x)
    elif activation == "leaky_relu":
        val, grad = leaky_relu(x)
    elif activation == "swish":
        val, grad = swish(x)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    return [round(val, 4), round(grad, 4)]