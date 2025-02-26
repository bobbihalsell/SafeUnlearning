import torch
from functools import wraps
from torch.utils.data import DataLoader


class UnsupportedModelError(Exception):
    def __init__(self, message='Model type is not supported for this operation.'):
        super().__init__(message)


def setup_device():
    """ Setup a torch device.

    Returns:
        str
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def l2_penalty(model, model_init, weight_decay):
    l2_loss = 0
    for (k, p), (k_init, p_init) in zip(
        model.named_parameters(), model_init.named_parameters()
    ):
        if p.requires_grad:
            l2_loss += (p - p_init).pow(2).sum()
    l2_loss *= weight_decay / 2.0
    return l2_loss


def available_if(condition):
    """ Makes a method available based on the output of a callable condition."""
    def decorator(method):
        @wraps(method)
        def inner(self, *args, **kwargs):
            if not condition(self):
                missing_cond = condition.__name__
                raise AttributeError(
                    f"Failed condition check: {missing_cond}. "
                    f"{self.__class__.__name__} requires {missing_cond} to be "
                    f"true to use {method.__name__}. Ensure that the "
                    "corresponding attribute has been initialized.")
            return method(self, *args, **kwargs)
        return inner
    return decorator


def _has_forget_dataloader(unlearner):
    return isinstance(unlearner.forget_dataloader, DataLoader)


def _has_retain_and_forget_dataloader(unlearner):
    return (isinstance(unlearner.forget_dataloader, DataLoader) and
            isinstance(unlearner.retain_dataloader, DataLoader))


def _has_retain_dataloader(unlearner):
    return isinstance(unlearner.retain_dataloader, DataLoader)
