import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader
from src.unlearning.utils import setup_device, UnsupportedModelError
import copy


class FinetuneUnlearner:
    """
    A class for fine-tuning a given PyTorch model using a subset of data.
    Supports both single-batch and mini-batch fine-tuning.
    """
    def __init__(self, original_model: nn.Module):
        """
        Initialize the Finetune class.

        Args:
            original_model (nn.Module): The base model to perform fine-tune unlearning on.
        """
        if not isinstance(original_model, nn.Module):
            raise UnsupportedModelError('original_model must be a Pytorch model.')
        self.device = setup_device()
        self.original_model = original_model.to(self.device)

    def unlearn(self, retain_dataloader: DataLoader,
                loss_fn: nn.Module, optimizer: Optional[optim.Optimizer],
                num_epochs: int = 200):
        """
        Fine-tune the model using mini-batch data loaders.

        Args:
            retain_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader, optional): DataLoader for test data. Defaults to None.
            test_name (str, optional): Name of the test dataset. Defaults to 'Test'.
            criterion (nn.Module, optional): Loss function. Defaults to None.
            optimizer (optim.Optimizer, optional): Optimizer. Defaults to None.
            epochs (int, optional): Number of fine-tuning epochs. Defaults to 200.
        """
        unlearned_model = copy.deepcopy(self.original_model)
        unlearned_model.to(self.device)
        unlearned_model.train()

        if optimizer is None or loss_fn is None:
            raise ValueError("Both optimizer and criterion must be provided.")

        for _ in range(num_epochs):
            unlearned_model.train()
            # running_retain_loss = 0.0
            for retain_X, retain_y in retain_dataloader:
                retain_X, retain_y = retain_X.to(self.device), retain_y.to(self.device)
                optimizer.zero_grad()
                retain_outputs = unlearned_model(retain_X)
                retain_loss = loss_fn(retain_outputs, retain_y)
                retain_loss.backward()
                optimizer.step()

        return unlearned_model
