import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)
    classes = np.unique(y_train)
    priors, means, variances = [], [], []
    for label in classes:
        rows = X_train[y_train == label]
        priors.append(len(rows) / len(X_train))
        means.append(rows.mean(axis=0))
        variances.append(rows.var(axis=0, ddof=0) + 1e-9)
    scores = []
    for point in X_test:
        scores.append([np.log(priors[i]) - 0.5 * np.sum(np.log(2.0 * np.pi * variances[i]) + (point - means[i]) ** 2 / variances[i]) for i in range(len(classes))])
    return [int(value) for value in classes[np.argmax(np.asarray(scores), axis=1)]]