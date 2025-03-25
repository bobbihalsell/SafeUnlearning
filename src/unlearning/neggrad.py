import torch
import torch.nn as nn
import copy
from typing import Optional
from src.unlearning.utils import (setup_device,
                                  UnsupportedModelError,
                                  available_if,
                                  _has_forget_dataloader,
                                  _has_retain_and_forget_dataloader,
                                  l2_penalty)
from itertools import cycle


class NegGrad:
    """ Implements NegGrad unlearning.

    As introduced in https://openreview.net/pdf?id=OveBaTtUAT
    """
    def __init__(
        self,
        original_model: nn.Module,
        forget_dataloader: torch.utils.data.DataLoader,
        retain_dataloader: Optional[torch.utils.data.DataLoader] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ):
        """
        Args:
            original_model: The original model to be unlearned.
            forget_dataloader: The forget set dataloader
            retain_dataloader: The retain set dataloader
            val_dataloader: The validation set dataloader (for evaluation)
        """
        if not isinstance(original_model, nn.Module):
            raise UnsupportedModelError('original_model must be a nn.Module.')
        if not isinstance(forget_dataloader, torch.utils.data.DataLoader):
            raise TypeError("forget_dataloader must be a "
                            "torch.utils.data.DataLoader.")
        self.device = setup_device()
        self.original_model = original_model.to(self.device)
        self.val_dataloader = val_dataloader
        self.retain_dataloader = retain_dataloader
        self.forget_dataloader = forget_dataloader

    @available_if(_has_forget_dataloader)
    def unlearn(self,
                loss_fn: nn,
                num_epochs: int,
                lr=1e-4,
                weight_decay=0,
                use_l2_penalty=False):
        """
        Perform NegGrad unlearning.

        NegGrad performs gradient ascent on the forget set.

        For more details, see https://openreview.net/pdf?id=OveBaTtUAT

        Args:
            loss_fn (torch.nn loss function): Loss function that takes logits.
            num_epochs (int): Number of passes over the forget set.
            lr: Learning rate
            weight_decay: Weight decay
            use_l2_penalty: Whether to add L2 penalty to the loss function.

        Returns:
            unlearned_model (nn.Module): The unlearned model.
        """
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("lr must be a positive number.")

        unlearned_model = copy.deepcopy(self.original_model)
        unlearned_model.to(self.device)

        optimizer = torch.optim.SGD(params=unlearned_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        for _ in range(num_epochs):
            for forget_inputs, forget_labels in self.forget_dataloader:
                unlearned_model.train()
                optimizer.zero_grad()

                forget_inputs = forget_inputs.to(self.device)
                forget_labels = forget_labels.to(self.device)

                forget_output = unlearned_model(forget_inputs)
                forget_loss = loss_fn(forget_output, forget_labels)
                # Negative loss to perform gradient ascent
                loss = -forget_loss

                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=self.original_model,
                                         weight_decay=weight_decay)
                    loss += l2_loss

                loss.backward()
                optimizer.step()

        return unlearned_model

    def _evaluate(self,
                  model: nn.Module,
                  val_dataloader: torch.utils.data.DataLoader,
                  loss_fn: nn.Module) -> torch.Tensor:
        """ Compute the validation loss of a model."""
        model = model.eval()
        batch_losses = torch.zeros(len(self.val_dataloader))
        for batch_ndx, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_losses[batch_ndx] = loss.detach().item()

        return batch_losses


class NegGradPlus:
    """ Implements NegGrad+ unlearning.

    Loss function is constructed based on the tradeoff between
    retain/forget performance.

    As introduced in https://openreview.net/pdf?id=OveBaTtUAT
    """
    def __init__(
        self,
        original_model: nn.Module,
        retain_dataloader: torch.utils.data.DataLoader,
        forget_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ):
        """
        Args:
            original_model: The original model to be unlearned.
            forget_dataloader: The forget set dataloaderz
            retain_dataloader: The retain set dataloader
            val_dataloader: The validation set dataloader
        """
        if not isinstance(original_model, nn.Module):
            raise UnsupportedModelError('original_model must be a nn.Module.')
        if not isinstance(forget_dataloader, torch.utils.data.DataLoader):
            raise TypeError("forget_dataloader must be a "
                            "torch.utils.data.DataLoader.")
        if not isinstance(retain_dataloader, torch.utils.data.DataLoader):
            raise TypeError("retain_dataloader must be a "
                            "torch.utils.data.DataLoader.")
        self.device = setup_device()
        self.original_model = original_model.to(self.device)
        self.val_dataloader = val_dataloader
        self.retain_dataloader = retain_dataloader
        self.forget_dataloader = forget_dataloader

    @available_if(_has_retain_and_forget_dataloader)
    def unlearn(self,
                loss_fn: nn,
                num_epochs: int,
                lr=1e-4,
                weight_decay=0,
                beta=0.995,
                use_l2_penalty=True):
        """
        Perform NegGrad+ unlearning.

        NegGrad+ performs gradient descent on retain/forget loss tradeoff.

        For more details, see https://openreview.net/pdf?id=OveBaTtUAT

        Args:
            loss_fn (torch.nn loss function): Loss function that takes logits.
            num_epochs (int): Number of passes over the forget set.
            lr: Learning rate (Default: 1e-4)
            weight_decay: Weight decay (Default: 0)
            beta: Tradeoff between retain and forget loss for NegGrad+.
            use_l2_penalty: Whether to add L2 penalty to the loss function.

        Returns:
            unlearned_model (nn.Module): The unlearned model.
        """
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("lr must be a positive number.")
        if not isinstance(beta, (int, float)) or beta < 0 or beta > 1:
            raise ValueError("Beta must be in (0,1).")
        if beta == 0:
            raise ValueError("Please use NegGrad if you wish to perform "
                             "gradient ascent on only the forget set.")
        if beta == 1:
            raise ValueError("Please use FinetuneUnlearner if you wish to "
                             "perform gradient descent on only the retain set")

        unlearned_model = copy.deepcopy(self.original_model)
        unlearned_model.to(self.device)
        optimizer = torch.optim.SGD(params=unlearned_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        for _ in range(num_epochs):
            for retain_batch, forget_batch in zip(self.retain_dataloader,
                                                  cycle(self.forget_dataloader)
                                                  ):
                unlearned_model.train()
                optimizer.zero_grad()
                forget_batch = [
                    tensor.to(self.device) for tensor in forget_batch
                ]
                # Compute the forget set and retain set loss. Cycle forget set.
                forget_inputs, forget_labels = forget_batch
                forget_output = unlearned_model(forget_inputs)
                forget_loss = loss_fn(forget_output, forget_labels)

                retain_batch = [
                    tensor.to(self.device) for tensor in retain_batch
                ]
                retain_inputs, retain_labels = retain_batch
                retain_output = unlearned_model(retain_inputs)
                retain_loss = loss_fn(retain_output, retain_labels)

                # Compute loss based on tradeoff
                loss = beta * retain_loss - (1-beta) * forget_loss
                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=self.original_model,
                                         weight_decay=weight_decay)
                    loss += l2_loss
                loss.backward()
                optimizer.step()

        return unlearned_model
