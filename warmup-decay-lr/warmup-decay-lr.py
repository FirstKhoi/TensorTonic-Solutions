import numpy as np
def warmup_decay_schedule(base_lr: float, warmup_steps: int, total_steps: int, current_step: int) -> float:
    """
    Returns the learning rate for the requested training step.
    """
    # Write code here
    base_lr = np.array(base_lr, dtype=float)
    warmup_steps = np.array(warmup_steps, dtype=float)
    total_steps = np.array(total_steps, dtype=float)
    current_step = np.array(current_step, dtype=float)

    if current_step < warmup_steps:
        lr = base_lr * current_step / warmup_steps
    else:
        lr = base_lr * (total_steps - current_step) / (total_steps - warmup_steps)

    return lr
    