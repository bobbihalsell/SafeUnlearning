import torch
import torch.nn as nn
from functools import wraps
from torch.utils.data import DataLoader
import random
import numpy as np
import os


def save_model(model: nn.Module,
               unlearning_algorithm: str,
               model_name: str,
               seed: str,
               model_type: str,
               ):
    """ Saves an entire model at a filepath specified by a convention.

    artifacts/unlearn/{unlearning_algorithm}/{model_name}_{seed}_{model_type}.pt

    Args:
        model (nn.Module): The model to be saved.
        unlearning_algorithm (str): The name of the unlearning algorithm
        model_name (str): The model name (e.g. resnet18)
        seed (int): The seed used during the experiment
        model_type (str): The model is either 'original' or 'unlearned'.

    Returns:
        None
    """
    directory = f'src/artifacts/unlearn/{unlearning_algorithm}'
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    
    filepath = os.path.join(directory, f'{model_name}_{seed}_{model_type}.pt')
    torch.save(model, filepath)


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


def set_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior
