import torch
import torch.nn as nn
import copy
from src.unlearning.utils import setup_device, UnsupportedModelError


class GradientAscentUnlearner:
    """ Perform gradient ascent unlearning on a model."""
    def __init__(self,
                 original_model: nn.Module,
                 unlearned_model=None):
        if not isinstance(original_model, nn.Module):
            raise UnsupportedModelError('original_model must be a Pytorch model.')
        self.original_model = original_model
        self.unlearned_model = unlearned_model
        self.device = setup_device()

        if self.unlearned_model is None:
            self.original_model.train()

    def loss_steps(self, forget_dataloader, loss_fn, num_epochs, lr=1e-4):
        """
        Perform gradient ascent on model parameters using forget set.

        Will not work if self.unlearned_model is already defined.

        Args:
            forget_dataloader (torch.utils.data.DataLoader): The dataloader for the forget set.
            loss_fn (torch.nn loss function): Loss function that uses logits (e.g. nn.CrossEntropyLoss)
            num_epochs (int): Number of passes over the forget set for unlearning.

        Returns:
            loss (float): The computed loss.
        """
        if self.unlearned_model is None:
            self.unlearned_model = copy.deepcopy(self.original_model)
            self.unlearned_model.to(self.device)
            for _ in range(num_epochs):
                for inputs, labels in forget_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # Zero the gradients in the model's parameters
                    self.unlearned_model.zero_grad()
                    # Forward pass through the model
                    output = self.unlearned_model(inputs)
                    loss = loss_fn(output, labels)
                    loss.backward()
                    # Perform gradient ascent
                    with torch.no_grad():
                        for param in self.unlearned_model.parameters():
                            if param.grad is not None:
                                param += lr * param.grad

            # Return the model
            return self.unlearned_model

        else:
            raise Exception("""Trying to perform unlearning when an unlearning model has already been provided!
                            Instantiate this object without passing in unlearned_model if you wish to perform unlearning.""")
