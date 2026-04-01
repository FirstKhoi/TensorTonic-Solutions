import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    rows = A.shape[0]
    cols = A.shape[1]
    T = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            T[j, i] = A[i, j]
    return T
    pass
