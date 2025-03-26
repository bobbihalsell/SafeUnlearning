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
                                  _has_retain_dataloader,
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
        device: Optional[torch.device] = None,
        evaluate: bool = False,
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
        super.__init__(device, evaluate)

    @available_if(_has_forget_dataloader)
    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
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

        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        # Check if generic unlearning args are valid
        loss_fn, num_epochs, lr, weight_decay, use_l2_penalty = self.valid_args(kwargs)

        if 'forget' not in data_dict.keys():
            raise ValueError("'forget' data must be in data_dict.")
        
        if self.evaluate:
        # Initialize loss tracking
            losses = {f"{data_type}_losses": [] for data_type in data_dict.keys()}

        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        
        eval_only_data = [data for data in data_dict.keys() if data != 'forget']

        for _ in range(num_epochs):
            total_forget_loss = 0
            for forget_inputs, forget_labels in data_dict['forget']:
                model.train()
                optimizer.zero_grad()

                forget_inputs = forget_inputs.to(self.device)
                forget_labels = forget_labels.to(self.device)

                forget_output = unlearned_model(forget_inputs)
                forget_loss = loss_fn(forget_output, forget_labels)
                # Negative loss to perform gradient ascent
                loss = -forget_loss

                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=model,
                                         weight_decay=weight_decay)
                    loss += l2_loss

                loss.backward()
                optimizer.step()
                total_forget_loss += forget_loss.item()

            if self.evaluate:
                losses['forget_losses'].append(total_forget_loss/len(data_dict['forget']))
                unlearned_model.eval()
                for data_type in eval_only_data:
                    loader_loss = self._evaluate(unlearned_model, data_dict[data_type], loss_fn).mean()
                    losses[f"{data_type}_losses"].append(loader_loss.item())
        if self.evaluate:
            return unlearned_model, losses
        return unlearned_model

class NegGradPlus:
    """ Implements NegGrad+ unlearning.

    Loss function is constructed based on the tradeoff between
    retain/forget performance.

    As introduced in https://openreview.net/pdf?id=OveBaTtUAT
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        evaluate: bool = False,
    ):
        """
        Args:
            original_model: The original model to be unlearned.
            forget_dataloader: The forget set dataloaderz
            retain_dataloader: The retain set dataloader
            val_dataloader: The validation set dataloader
        """
        super.__init__(device, evaluate)

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
    

    @available_if(_has_retain_dataloader and _has_forget_dataloader)
    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                beta: float,
                **kwargs):
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
        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        # Check if generic unlearning args are valid
        loss_fn, num_epochs, lr, weight_decay, use_l2_penalty = self.valid_args(kwargs)

        if 'forget' not in data_dict.keys() or 'retain' not in data_dict.keys():
            raise ValueError("'forget' and 'retain' data must be in data_dict.")
        
        if beta == 0:
            raise ValueError("Please use NegGrad if you wish to perform "
                             "gradient ascent on only the forget set.")
        if beta == 1:
            raise ValueError("Please use FinetuneUnlearner if you wish to "
                             "perform gradient descent on only the retain set")

        if self.evaluate:
        # Initialize loss tracking
            losses = {f"{data_type}_losses": [] for data_type in data_dict.keys()}
        
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        
        eval_only_data = [data for data in data_dict.keys() if data not in ['forget', 'retain']]

        for _ in range(num_epochs):
            total_forget_loss, total_retain_loss = 0, 0
            for retain_batch, forget_batch in zip( data_dict['retain'],
                                                  cycle( data_dict['forget'])
                                                  ):
                unlearned_model.train()
                optimizer.zero_grad()
                forget_batch = [
                    tensor.to(self.device) for tensor in forget_batch
                ]
                # Compute the forget set and retain set loss. Cycle forget set.
                forget_inputs, forget_labels = forget_batch
                forget_output = unlearned_model(forget_inputs)

                retain_batch = [
                    tensor.to(self.device) for tensor in retain_batch
                ]
                retain_inputs, retain_labels = retain_batch
                retain_output = unlearned_model(retain_inputs)

                # Compute loss based on tradeoff
                loss, retain_loss, forget_loss = self._calculate_loss(
                    beta=beta,
                    criterion=loss_fn,
                    retain_outputs=retain_output,
                    retain_targets=retain_labels,
                    forget_outputs=forget_output,
                    forget_targets=forget_labels
                )
                
                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=model,
                                         weight_decay=weight_decay)
                    loss += l2_loss
                loss.backward()
                optimizer.step()
                total_forget_loss += forget_loss.item()
                total_retain_loss += retain_loss.item()

            if self.evaluate:
                losses['forget_losses'].append(total_forget_loss/len(data_dict['forget']))
                losses['retain_losses'].append(total_retain_loss/len(data_dict['retain']))
                unlearned_model.eval()
                for data_type in eval_only_data:
                    loader_loss = self._evaluate(unlearned_model, data_dict[data_type], loss_fn).mean()
                    losses[f"{data_type}_losses"].append(loader_loss.item())

        if self.evaluate:
            return unlearned_model, losses
        return unlearned_model
