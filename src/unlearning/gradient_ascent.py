import torch
import torch.nn as nn
import copy
from typing import Optional
from collections.abc import Callable
from src.unlearning.utils import (setup_device,
                                  UnsupportedModelError,
                                  available_if,
                                  has_forget_dataloader)


class GradientAscentUnlearner:
    """ Perform gradient ascent unlearning on a model."""
    def __init__(
        self,
        original_model: nn.Module,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        retain_dataloader: Optional[torch.utils.data.DataLoader] = None,
        forget_dataloader: Optional[torch.utils.data.DataLoader] = None,
        score_fn: Optional[Callable[[nn.Module], float]] = None
    ):
        """
        Args:
            original_model: The original model to be unlearned.
            val_dataloader: The validation set dataloader (for evaluation)
        """
        if not isinstance(original_model, nn.Module):
            raise UnsupportedModelError('original_model must be a Pytorch model.')
        self.device = setup_device()
        self.original_model = original_model
        self.val_dataloader = val_dataloader
        self.retain_dataloader = retain_dataloader
        self.forget_dataloader = forget_dataloader
        self.score_fn = score_fn

    @available_if(has_forget_dataloader)
    def unlearn(self, loss_fn, num_epochs, lr=1e-4):
        """
        Perform gradient ascent on model parameters using forget set.

        Args:
            loss_fn (torch.nn loss function): Loss function that takes in logits.
            num_epochs (int): Number of passes over the forget set.
            lr: Learning rate (Default: 1e-4)

        Returns:
            loss (float): The computed loss.
        """
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("lr must be a positive number.")

        unlearned_model = copy.deepcopy(self.original_model)
        unlearned_model.to(self.device)
        unlearned_model.train()
        for _ in range(num_epochs):
            for inputs, labels in self.forget_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Zero the gradients in the model's parameters
                unlearned_model.zero_grad()
                # Forward pass through the model
                output = unlearned_model(inputs)
                loss = loss_fn(output, labels)
                loss.backward()
                # Perform gradient ascent
                with torch.no_grad():
                    for param in unlearned_model.parameters():
                        if param.grad is not None:
                            param += lr * param.grad

        # Return the model
        return unlearned_model
