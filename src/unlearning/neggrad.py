import torch
import torch.nn as nn
import copy
from typing import Optional
from unlearning.utils import (setup_device,
                              UnsupportedModelError,
                              l2_penalty)
from itertools import cycle


class NegGrad:
    """ Implements NegGrad unlearning.

    As introduced in https://openreview.net/pdf?id=OveBaTtUAT
    """
    def __init__(
        self,
        loss_fn: torch.nn.functional,
    ):
        """
        Args:
            loss_fn: The loss function.
        """
        self.device = setup_device()
        self.loss_fn = loss_fn

    def unlearn(self,
                model: nn.Module,
                retain_dataloader: torch.utils.data.DataLoader,
                forget_dataloader: torch.utils.data.DataLoader,
                retain_val_dataloader: Optional[torch.utils.data.DataLoader] = None,
                forget_val_dataloader: Optional[torch.utils.data.DataLoader] = None,
                **kwargs):
        """
        Perform NegGrad unlearning.

        NegGrad performs gradient ascent on the forget set.

        For more details, see https://openreview.net/pdf?id=OveBaTtUAT

        Args:
            model (nn.Module): The model to perform unlearning on.
            retain_dataloader (DataLoader): The retain set DataLoader
            forget_dataloader (DataLoader): The forget set DataLoader
            retain_val_dataloader (DataLoader): The validation retain set DataLoader
            forget_val_dataloader (DataLoader): The validation forget set DataLoader

        Keyword Args:
            loss_fn (callable): Loss function that takes logits.
            num_epochs (int): Number of passes over the forget set.
            lr (float, optional): Learning rate. Default is 1e-4.
            weight_decay (float, optional): Weight decay. Default is 0.
            use_l2_penalty (bool, optional): Whether to add L2 penalty to the loss function. Default is False.
            evaluate (bool, optional): Whether to check performance on validation set. Default is False. 

        Returns:
            unlearned_model (nn.Module): The unlearned model.
        """
        num_epochs = kwargs.get('num_epochs')
        lr = kwargs.get('lr', 1e-4)
        weight_decay = kwargs.get('weight_decay', 0)
        use_l2_penalty = kwargs.get('use_l2_penalty', False)
        evaluate = kwargs.get('evaluate', False)

        # Checks that the arguments have been passed in correct format
        if not isinstance(model, nn.Module):
            raise UnsupportedModelError('model must be a nn.Module.')
        if not isinstance(forget_dataloader, torch.utils.data.DataLoader):
            raise TypeError("forget_dataloader must be a "
                            "torch.utils.data.DataLoader.")
        if not callable(self.loss_fn):
            raise ValueError("loss_fn must be a callable loss function.")
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("lr must be a positive number.")
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError("weight_decay must be a non-negative number.")
        if not isinstance(use_l2_penalty, bool):
            raise ValueError("use_l2_penalty must be a boolean.")

        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        optimizer = torch.optim.SGD(params=unlearned_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        for i in range(num_epochs):
            for forget_inputs, forget_labels in forget_dataloader:
                unlearned_model.train()
                optimizer.zero_grad()

                forget_inputs = forget_inputs.to(self.device)
                forget_labels = forget_labels.to(self.device)

                forget_output = unlearned_model(forget_inputs)
                forget_loss = self.loss_fn(forget_output, forget_labels)
                # Negative loss to perform gradient ascent
                loss = -forget_loss

                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=model,
                                         weight_decay=weight_decay)
                    loss += l2_loss

                loss.backward()
                optimizer.step()

            if evaluate:
                # Evaluate the model's performance on retain and forget sets
                forget_loss = self._evaluate(
                    model=unlearned_model,
                    val_dataloader=forget_dataloader,
                )
                retain_loss = self._evaluate(
                    model=unlearned_model,
                    val_dataloader=retain_dataloader
                )
                forget_val_loss = None
                retain_val_loss = None
                print(f'Epoch {i+1} forget loss: {forget_loss}, '
                      f'retain loss: {retain_loss}, '
                      f'val forget loss: {forget_val_loss}, '
                      f'val retain loss: {retain_val_loss}')

        return model

    def _evaluate(self,
                  model: nn.Module,
                  val_dataloader: torch.utils.data.DataLoader) -> float:
        """ Measure the average validation loss of the model on a val dataloader."""
        model = model.eval()
        total_loss = 0.0
        num_batches = len(val_dataloader)

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        return total_loss / num_batches if num_batches > 0 else float("nan")


class NegGradPlus:
    """ Implements NegGrad+ unlearning.

    Loss function is constructed based on the tradeoff between
    retain/forget performance.

    As introduced in https://openreview.net/pdf?id=OveBaTtUAT
    """
    def __init__(
        self,
        loss_fn: torch.nn.functional,
    ):
        """
        Args:
            loss_fn: The loss function.
        """
        self.device = setup_device()
        self.loss_fn = loss_fn

    def unlearn(self,
                model: nn.Module,
                retain_dataloader: torch.utils.data.DataLoader,
                forget_dataloader: torch.utils.data.DataLoader,
                retain_val_dataloader: Optional[torch.utils.data.DataLoader] = None,
                forget_val_dataloader: Optional[torch.utils.data.DataLoader] = None,
                **kwargs):
        """
        Perform NegGrad+ unlearning.

        NegGrad+ performs gradient descent on retain/forget loss tradeoff.

        For more details, see https://openreview.net/pdf?id=OveBaTtUAT

        Args:
            model (nn.Module): The model to perform unlearning on.
            retain_dataloader (DataLoader): The retain set DataLoader
            forget_dataloader (DataLoader): The forget set DataLoader
            retain_val_dataloader (DataLoader): The validation retain set DataLoader
            forget_val_dataloader (DataLoader): The validation forget set DataLoader

        Keyword Args:
            loss_fn (callable): Loss function that takes logits.
            num_epochs (int): Number of passes over the forget set.
            lr (float, optional): Learning rate. Default is 1e-4.
            weight_decay (float, optional): Weight decay. Default is 0.
            use_l2_penalty (bool, optional): Whether to add L2 penalty to the loss function. Default is False.
            evaluate (bool, optional): Whether to check performance on validation set. Default is False. 

        Returns:
            unlearned_model (nn.Module): The unlearned model.
        """
        num_epochs = kwargs.get('num_epochs')
        lr = kwargs.get('lr', 1e-4)
        weight_decay = kwargs.get('weight_decay', 0)
        use_l2_penalty = kwargs.get('use_l2_penalty', False)
        beta = kwargs.get('beta', 0.95)
        evaluate = kwargs.get('evaluate', False)

        if not isinstance(model, nn.Module):
            raise UnsupportedModelError('model must be a nn.Module.')
        if not isinstance(forget_dataloader, torch.utils.data.DataLoader):
            raise TypeError("forget_dataloader must be a "
                            "torch.utils.data.DataLoader.")
        if not isinstance(retain_dataloader, torch.utils.data.DataLoader):
            raise TypeError("retain_dataloader must be a "
                            "torch.utils.data.DataLoader.")
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

        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        optimizer = torch.optim.SGD(params=unlearned_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        for i in range(num_epochs):
            for retain_batch, forget_batch in zip(retain_dataloader,
                                                  cycle(forget_dataloader)
                                                  ):
                unlearned_model.train()
                optimizer.zero_grad()
                forget_batch = [
                    tensor.to(self.device) for tensor in forget_batch
                ]
                # Compute the forget set and retain set loss. Cycle forget set.
                forget_inputs, forget_labels = forget_batch
                forget_output = unlearned_model(forget_inputs)
                forget_loss = self.loss_fn(forget_output, forget_labels)

                retain_batch = [
                    tensor.to(self.device) for tensor in retain_batch
                ]
                retain_inputs, retain_labels = retain_batch
                retain_output = unlearned_model(retain_inputs)
                retain_loss = self.loss_fn(retain_output, retain_labels)

                # Compute loss based on tradeoff
                loss = beta * retain_loss - (1-beta) * forget_loss
                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=model,
                                         weight_decay=weight_decay)
                    loss += l2_loss
                loss.backward()
                optimizer.step()

            if evaluate:
                # Evaluate the model's performance on retain and forget sets
                forget_loss = self._evaluate(
                    model=unlearned_model,
                    val_dataloader=forget_dataloader,
                )
                retain_loss = self._evaluate(
                    model=unlearned_model,
                    val_dataloader=retain_dataloader
                )
                forget_val_loss = None
                retain_val_loss = None
                print(f'Epoch {i+1} forget loss: {forget_loss}, '
                        f'retain loss: {retain_loss}, '
                        f'val forget loss: {forget_val_loss}, '
                        f'val retain loss: {retain_val_loss}')

        return unlearned_model

    def _evaluate(self,
                  model: nn.Module,
                  val_dataloader: torch.utils.data.DataLoader) -> float:
        """ Measure the average validation loss of the model on a val dataloader."""
        model = model.eval()
        total_loss = 0.0
        num_batches = len(val_dataloader)

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        return total_loss / num_batches if num_batches > 0 else float("nan")
