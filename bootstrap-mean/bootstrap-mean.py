import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    means = x[indices].mean(axis=1)
    alpha = (1 - ci) / 2
    return {
        "bootstrap_mean": float(means.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1 - alpha)),
    }