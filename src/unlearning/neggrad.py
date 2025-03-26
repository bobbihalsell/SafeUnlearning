import os
import timm
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
from base import BaseUnlearner


from typing import Optional, Union, Tuple, List, Dict, Any
from torch.utils.data import DataLoader, TensorDataset



class NegGrad(BaseUnlearner):
    """
    Implements NegGrad unlearning as introduced in https://openreview.net/pdf?id=OveBaTtUAT
    
    This implementation performs gradient ascent on forget data to make the model
    "forget" specific samples while preserving performance on retained data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        forget_data: Union[Tuple[torch.Tensor, torch.Tensor], DataLoader],
        retain_data: Optional[Union[Tuple[torch.Tensor, torch.Tensor], DataLoader]] = None,
        test_data: Optional[Union[Tuple[torch.Tensor, torch.Tensor], DataLoader]] = None,
        device: Optional[torch.device] = None,
        save_steps: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Initialize the NegGrad unlearning object.
        
        Args:
            model: Original model to be unlearned
            forget_data: Data to be "forgotten" (tuple of tensors or DataLoader)
            retain_data: Data to be retained (tuple of tensors or DataLoader)
            test_data: Data for testing/evaluation (tuple of tensors or DataLoader)
            device: Computing device (CPU/GPU)
        """
        super.__init__(model, device, save_path, save_steps)
        # Store datasets
        self._process_data(forget_data, retain_data, test_data)
        if save_steps:
        # Initialize loss tracking
            for data_type in ["forget", "retain", "test"]:
                if hasattr(self, f"{data_type}_data"):
                    setattr(self, f"{data_type}_losses", [])

    @available_if(_has_forget_dataloader)
    def unlearn(self,
                **kwargs):
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
        num_epochs = kwargs.get('num_epochs')
        lr = kwargs.get('lr', 1e-4)
        weight_decay = kwargs.get('weight_decay', 0)
        loss_fn = kwargs.get('loss_fn')
        use_l2_penalty = kwargs.get('use_l2_penalty', False)
        evaluate = kwargs.get('evaluate', False)
        data_types = kwargs.get('evaluate', ['forget'])

        # Checks that the arguments have been passed in correct format
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
        if not isinstance(data_types, List):
            raise ValueError("data_types must be a list of strings.")
        if 'forget' not in data_types:
            raise ValueError("forget must be in data_types.")
        
        optimizer = torch.optim.SGD(params=self.unlearned_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        for _ in range(num_epochs):
            for forget_inputs, forget_labels in self.forget_loader:
                self.unlearned_model.train()
                optimizer.zero_grad()

                forget_inputs = forget_inputs.to(self.device)
                forget_labels = forget_labels.to(self.device)

                forget_output = self.unlearned_model(forget_inputs)
                forget_loss = loss_fn(forget_output, forget_labels)
                # Negative loss to perform gradient ascent
                loss = -forget_loss

                if use_l2_penalty:
                    l2_loss = l2_penalty(model=self.unlearned_model,
                                         model_init=self.original_model,
                                         weight_decay=weight_decay)
                    loss += l2_loss

                if self.save_steps:
                    self.forget_losses.append(forget_loss.item())
                loss.backward()
                optimizer.step()

            if evaluate:
                current_losses = {}
                if self.save_steps:
                    self.unlearned_model.eval()
                    for data_type in data_types:
                        loader = getattr(self, f"{data_type}_loader")
                        loader_loss = self._evaluate(self.unlearned_model, loader, self.loss_fn).mean()
                        getattr(self, f"{loader}_losses").append(loader_loss.item())

        return self.unlearned_model

class NegGradPlus:
    """ Implements NegGrad+ unlearning.

    Loss function is constructed based on the tradeoff between
    retain/forget performance.

    As introduced in https://openreview.net/pdf?id=OveBaTtUAT
    """
    def __init__(
        self,
        model: nn.Module,
        forget_data: Union[Tuple[torch.Tensor, torch.Tensor], DataLoader],
        retain_data: Optional[Union[Tuple[torch.Tensor, torch.Tensor], DataLoader]] = None,
        test_data: Optional[Union[Tuple[torch.Tensor, torch.Tensor], DataLoader]] = None,
        batch_sizes=[32],
        device: Optional[torch.device] = None,
        save_steps: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Args:
            original_model: The original model to be unlearned.
            forget_dataloader: The forget set dataloaderz
            retain_dataloader: The retain set dataloader
            val_dataloader: The validation set dataloader
        """
        super.__init__(model, device, save_path, save_steps)
        # Store datasets
        self._process_data(forget_data=forget_data, 
                           retain_data=retain_data, 
                           test_data=test_data,
                           batch_sizes=batch_sizes
        )
        if save_steps:
        # Initialize loss tracking
            for data_type in ["forget", "retain", "test"]:
                if hasattr(self, f"{data_type}_data"):
                    setattr(self, f"{data_type}_losses", [])

    def _calculate_loss(
        self, 
        beta: float, 
        criterion: nn.Module, 
        retain_outputs: torch.Tensor, 
        retain_targets: torch.Tensor,
        forget_outputs: torch.Tensor, 
        forget_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the composite loss based on retain and forget data.
        
        Args:
            beta: Weight balancing factor between retain and forget loss
            criterion: Loss function
            retain_outputs: Model outputs for retain data
            retain_targets: Ground truth for retain data
            forget_outputs: Model outputs for forget data
            forget_targets: Ground truth for forget data
            
        Returns:
            Tuple containing (total_loss, retain_loss, forget_loss)
        """
        # Calculate individual losses
        forget_loss = criterion(forget_outputs, forget_targets)
        
        # For standard NegGrad (no retain data)
        if retain_outputs is None or retain_targets is None:
            return -forget_loss, None, forget_loss
        
        # For NegGrad+ (with retain data)
        retain_loss = criterion(retain_outputs, retain_targets)
        
        # Normalize by number of samples if applicable
        Nr = len(retain_outputs)
        Nf = len(forget_outputs)
        
        # Calculate composite loss with beta weighting
        total_loss = beta * retain_loss / Nr - (1 - beta) * forget_loss / Nf
        
        return total_loss, retain_loss, forget_loss
    

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
