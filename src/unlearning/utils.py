import torch
from functools import wraps


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


def has_forget_dataloader(unlearner):
    return unlearner.forget_dataloader is not None
