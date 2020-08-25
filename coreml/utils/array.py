import torch


def is_scalar(value):
    if isinstance(value, (int, float)) or (
            isinstance(value, torch.Tensor) and not value.ndim):
        return True

    return False
