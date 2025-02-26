import torch
import torch.nn as nn
import copy
from typing import Optional
from collections.abc import Callable
from src.unlearning.utils import (setup_device,
                                  UnsupportedModelError,
                                  available_if,
                                  _has_forget_dataloader,
                                  _has_retain_and_forget_dataloader,
                                  _has_retain_dataloader)


class NegGrad:
    """ Perform NegGrad unlearning on a model."""
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
            retain_dataloader: The retain set dataloader
            forget_dataloader: The forget set dataloader
            score_fn: A callable to score the model at each epoch based on
                      performance on the retain/forget set.
        """
        if not isinstance(original_model, nn.Module):
            raise UnsupportedModelError('original_model must be a Pytorch model.')
        self.device = setup_device()
        self.original_model = original_model
        self.val_dataloader = val_dataloader
        self.retain_dataloader = retain_dataloader
        self.forget_dataloader = forget_dataloader
        self.score_fn = score_fn

    @available_if(_has_forget_dataloader)
    def unlearn(self, loss_fn, num_epochs, lr=1e-4, weight_decay=0, beta=0):
        """
        Perform gradient ascent on model parameters using forget set.

        Args:
            loss_fn (torch.nn loss function): Loss function that takes in logits.
            num_epochs (int): Number of passes over the forget set.
            lr: Learning rate (Default: 1e-4)
            weight_decay: Weight decay (Default: 0)
            beta: Tradeoff parameter between retain and forget loss for NegGrad+. (Default: 0)

        Returns:
            unlearned_model (nn.Module): The unlearned model.
        """
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("lr must be a positive number.")
        if not isinstance(beta, (int, float)) or beta < 0 or beta > 1:
            raise ValueError("Beta must be in [0,1].")
        if not _has_retain_dataloader(self) and beta > 0:
            raise AttributeError("Trying to perform NegGrad+ without retain "
                                 "dataloader, but this is required by the "
                                 "NegGrad+ objective.")

        unlearned_model = copy.deepcopy(self.original_model)
        unlearned_model.to(self.device)
        unlearned_model.train()

        optimizer = torch.optim.SGD(params=unlearned_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        for _ in range(num_epochs):
            # Zero the gradient at each epoch (1 step per epoch)
            optimizer.zero_grad()
            forget_loss = 0
            num_forget_batches = 0
            for forget_inputs, forget_labels in self.forget_dataloader:
                forget_inputs = forget_inputs.to(self.device)
                forget_labels = forget_labels.to(self.device)

                forget_output = unlearned_model(forget_inputs)
                forget_loss += loss_fn(forget_output, forget_labels)
                num_forget_batches += 1

            retain_loss = 0
            num_retain_batches = 0
            if self.retain_dataloader and beta > 0:
                for retain_inputs, retain_labels in self.retain_dataloader:
                    retain_inputs = retain_inputs.to(self.device)
                    retain_labels = retain_labels.to(self.device)

                    retain_output = unlearned_model(retain_inputs)
                    retain_loss += loss_fn(retain_output, retain_labels)
                    num_retain_batches += 1

                loss = beta * retain_loss/num_retain_batches - (1-beta) * forget_loss/num_forget_batches
            else:
                loss = -forget_loss/num_forget_batches

            # Perform whole-batch gradient descent on the loss.
            loss.backward()
            optimizer.step()

        return unlearned_model
