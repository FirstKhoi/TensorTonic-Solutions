import numpy as np

def time_based_split(timestamps: list, train_end: int, gap: int, val_end: int) -> dict:
    """
    Returns a dictionary with train_idx and val_idx as sorted integer NumPy arrays.
    """
    times = np.asarray(timestamps)
    indices = np.arange(times.size)
    train_mask = times < train_end
    val_mask = (times >= train_end + gap) & (times < val_end)
    return {
        "train_idx": indices[train_mask].astype(int),
        "val_idx": indices[val_mask].astype(int),
    }
