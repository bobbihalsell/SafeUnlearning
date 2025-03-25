import torch
from functools import wraps
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any
import pathlib


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    payload: Dict[str, Any],
    filename: str = "checkpoint.pth",
    save_dir: str = "checkpoints",
) -> None:
    """ Save a checkpoint during training."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "payload": payload,
    }
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    torch.save(state, path)
    print(f"Checkpoint saved at {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    filename: str = "checkpoint.pth",
) -> Tuple[torch.nn.Module,
           torch.optim.Optimizer,
           torch.optim.lr_scheduler.LRScheduler,
           int, 
           Dict[str, Any]]:
    """ Load a saved checkpoint."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    payload = checkpoint["payload"]
    return model, optimizer, scheduler, epoch, payload


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
