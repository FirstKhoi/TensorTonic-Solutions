import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(seed)

    train_indices = []
    test_indices = []

    classes = np.unique(y)

    for cls in classes:

        indices = np.where(y == cls)[0]

        rng.shuffle(indices)


        n_test = int(round(len(indices) * test_size))

        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])


    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    return {
        "X_train": X[train_indices],
        "X_test": X[test_indices],
        "y_train": y[train_indices],
        "y_test": y[test_indices]
    }