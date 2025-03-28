import torch
import torch.nn as nn
import random
import numpy as np
import os


def save_model(model: nn.Module,
               output_dir: str,
               unlearning_algorithm: str,
               model_name: str,
               seed: str,
               model_type: str,
               payload: dict = None):
    """ Saves a model's state_dict and training info at a filepath specified by a convention.

    Saves to:
        artifacts/unlearn/{unlearning_algorithm}/{model_name}_{seed}_{model_type}.pt

    Args:
        model (nn.Module): The model to be saved.
        unlearning_algorithm (str): The name of the unlearning algorithm.
        model_name (str): The model name (e.g., resnet18).
        seed (int): The seed used during the experiment.
        model_type (str): The model is either 'original' or 'unlearned'.
        payload (dict, optional): Additional information (e.g., losses, metrics).

    Returns:
        str: The filepath where the model was saved.
    """
    directory = f'{output_dir}/unlearn/{unlearning_algorithm}'
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    filepath = os.path.join(directory, f'{model_name}_{seed}_{model_type}.pt')

    # Save only the state_dict
    save_data = {
        'state_dict': model.state_dict(),
        'model_name': model_name,
        'unlearning_algorithm': unlearning_algorithm
    }
    
    if payload:
        save_data.update(payload)  # Merge additional metadata

    torch.save(save_data, filepath)
    print(f"Model state_dict saved to {filepath}")

    return filepath


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
