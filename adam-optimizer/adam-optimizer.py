import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    m_t = np.asarray(m)
    v_t = np.asarray(v) 
    grad = np.asarray(grad) 
    theta = np.asarray(param) 

    # first momentum and second momentum  
    m_t = (beta1 * m_t) + (grad - beta1 * grad) 
    v_t = (beta2 * v_t) + ((1 - beta2) * (grad**2))

    # bias correction 
    hat_m_t =  m_t / (1 - (beta1 ** t))
    hat_v_t =  v_t / (1 - (beta2 ** t))

    # update params 
    theta = theta - (lr * (hat_m_t / (np.sqrt(hat_v_t) + eps)))


    return (theta, m_t, v_t)