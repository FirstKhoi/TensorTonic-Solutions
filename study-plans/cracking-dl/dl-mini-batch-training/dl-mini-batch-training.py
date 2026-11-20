import numpy as np

def mini_batch_training(X: list, y: list, weights: list, biases: list, lr: float, epochs: int, batch_size: int) -> list:
    """
    Returns the full-dataset loss after each epoch.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    w = [np.array(W, dtype=float) for W in weights]
    b = [np.array(B, dtype=float) for B in biases]

    def relu(z):
        return np.maximum(0, z)

    def relu_deriv(z):
        return (z > 0).astype(float)

    def forward_pass(batch):
        activations, pre_activations = [batch], []
        a = batch
        for i in range(len(w)):
            z = a @ w[i].T + b[i]
            pre_activations.append(z)
            a = relu(z) if i < len(w) - 1 else z
            activations.append(a)
        return activations, pre_activations

    def backward_pass(activations, pre_activations, y_batch):
        m = y_batch.shape[0]
        # Đạo hàm của L = (1 / 2m) * sum((y_pred - y)^2)
        dz = (activations[-1] - y_batch) / m

        dw_list = [None] * len(w)
        db_list = [None] * len(b)

        for i in reversed(range(len(w))):
            dw_list[i] = dz.T @ activations[i]
            db_list[i] = np.sum(dz, axis=0).reshape(b[i].shape)
            if i > 0:
                da = dz @ w[i]
                dz = da * relu_deriv(pre_activations[i - 1])

        return dw_list, db_list

    n_samples = X.shape[0]
    epoch_losses = []

    for _ in range(epochs):
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            activations, pre_activations = forward_pass(X_batch)
            dw_list, db_list = backward_pass(activations, pre_activations, y_batch)

            for i in range(len(w)):
                w[i] -= lr * dw_list[i]
                b[i] -= lr * db_list[i]

        full_activations, _ = forward_pass(X)
        full_loss = np.sum((full_activations[-1] - y) ** 2) / (2 * n_samples)
        epoch_losses.append(round(float(full_loss), 4))

    return epoch_losses