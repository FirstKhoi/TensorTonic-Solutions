import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    
    Q = Q@W_q
    K = K@W_k
    V = V@W_v

    head_dim = Q.shape[-1] // num_heads

    N, Lq, E = Q.shape
    Lk = K.shape[1]
    
    Q = Q.reshape(N, Lq, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(N, Lk, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(N, Lk, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    out = weights @ V

    out = out.transpose(0, 2, 1, 3).reshape(N, Lq, E)

    return out @ W_o
