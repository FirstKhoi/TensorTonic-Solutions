import numpy as np

def forward_pass(x: list, weights: list, biases: list) -> dict:
    """
    Returns all activations and pre-activations.
    """
    x = np.array(x, dtype=float)
    
    pre_activations = []
    activations = [np.round(x, 4).tolist()]
    
    num_layers = len(weights)
    
    for i, (w, b) in enumerate(zip(weights, biases)):
        w = np.array(w, dtype=float)
        b = np.array(b, dtype=float)
        
        # pre-activation: z = W @ x + b
        z = w @ x + b
        pre_activations.append(np.round(z, 4).tolist())
        
        # activation: ReLU cho các lớp ẩn, tuyến tính cho lớp cuối
        if i < num_layers - 1:
            x = np.maximum(0.0, z)
        else:
            x = z
            
        activations.append(np.round(x, 4).tolist())
        
    return {
        "pre_activations": pre_activations,
        "activations": activations
    }