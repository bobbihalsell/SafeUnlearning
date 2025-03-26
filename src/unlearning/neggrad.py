import torch
import torch.nn as nn
import copy
from typing import Optional
from unlearning.utils import l2_penalty
from itertools import cycle
from base import BaseUnlearner
from typing import Optional, Tuple, Dict
from torch.utils.data import DataLoader


class NegGrad(BaseUnlearner):
    """
    Implements NegGrad unlearning as introduced in https://openreview.net/pdf?id=OveBaTtUAT
    
    NegGrad performs gradient ascent on the forget set to make the model "forget" 
    specific samples. This approach maximizes the loss on data that should be 
    forgotten, effectively reducing the model's ability to make accurate predictions
    on that data.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        evaluate: bool = False,
    ):
        """
        Initialize the NegGrad unlearning object.
        
        Args:
            device: Computing device (CPU/GPU) to use for computations.
                   If None, will be automatically determined.
            evaluate: Whether to track and return evaluation metrics during unlearning.
        """
        super().__init__(device, evaluate)

    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                **kwargs):
        """
        Perform NegGrad unlearning.
        NegGrad performs gradient ascent on the forget set.
        For more details, see https://openreview.net/pdf?id=OveBaTtUAT

        Args:
            model: The original model to perform unlearning on.
            data_dict: Dictionary containing dataloaders for different datasets.
                       Must include a 'forget' key with corresponding DataLoader.
            **kwargs: Additional arguments including:
                - loss_fn: Loss function for training (will be negated) and evaluation.
                - num_epochs: Number of training epochs (default: 1).
                - lr: Learning rate (default: 1e-2).
                - weight_decay: Weight decay parameter (default: 0).
                - use_l2_penalty: Whether to add L2 regularization penalty (default: False).

        Returns:
            If self.evaluate is True:
                Tuple of (unlearned_model, losses_dict) where losses_dict contains
                tracked losses for each dataset type.
            Otherwise:
                The unlearned model.
                
        Raises:
            ValueError: If 'forget' data is not in data_dict.
        """

        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        # Validate and extract common hyperparameters
        loss_fn, num_epochs, lr, weight_decay, use_l2_penalty = self.valid_args(**kwargs)

        if 'forget' not in data_dict.keys():
            raise ValueError("'forget' data must be in data_dict.")
        
        # Initialize loss tracking if evaluation is enabled
        if self.evaluate:
        # Initialize loss tracking
            losses = {f"{data_type}_losses": [] for data_type in data_dict.keys()}

        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        
        eval_only_data = [data for data in data_dict.keys() if data != 'forget']

        # Main training loop
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

                # Add L2 penalty if requested
                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=model,
                                         weight_decay=weight_decay)
                    loss += l2_loss

                loss.backward()
                optimizer.step()
                total_forget_loss += forget_loss.item()

            if self.evaluate:
                # Calculate average forget loss for this epoch
                losses['forget_losses'].append(total_forget_loss/len(data_dict['forget']))
                # Evaluate model on other datasets
                unlearned_model.eval()
                for data_type in eval_only_data:
                    loader_loss = self._evaluate(unlearned_model, data_dict[data_type], loss_fn).mean()
                    losses[f"{data_type}_losses"].append(loader_loss.item())
        if self.evaluate:
            return unlearned_model, losses
        return unlearned_model

class NegGradPlus(BaseUnlearner):
    """
    Implements NegGrad+ unlearning as introduced in https://openreview.net/pdf?id=OveBaTtUAT
    
    NegGrad+ extends NegGrad by incorporating a trade-off between retaining performance
    on keep data while forgetting the forget data. It balances gradient descent on retain
    data with gradient ascent on forget data, controlled by a beta parameter.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        evaluate: bool = False,
    ):
        """
        Initialize the NegGrad+ unlearning object.
        
        Args:
            device: Computing device (CPU/GPU) to use for computations.
                   If None, will be automatically determined.
            evaluate: Whether to track and return evaluation metrics during unlearning.
        """
        super().__init__(device, evaluate)

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
    
    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                **kwargs):
        """
        Perform NegGrad+ unlearning with balanced retain/forget optimization.

        NegGrad+ balances minimizing loss on retain data while maximizing loss
        on forget data, controlled by the beta parameter.

        Args:
            model: The original model to perform unlearning on.
            data_dict: Dictionary containing dataloaders for different datasets.
                       Must include both 'forget' and 'retain' keys.
            **kwargs: Additional arguments including:
                - loss_fn: Loss function to use for training.
                - num_epochs: Number of training epochs (default: 1).
                - lr: Learning rate (default: 1e-2).
                - weight_decay: Weight decay parameter (default: 0).
                - use_l2_penalty: Whether to add L2 regularization penalty (default: False).
                - beta: Tradeoff parameter between retain and forget objectives (0-1).
                  beta=0 is pure forgetting, beta=1 is pure retention.
                  PLEASE USE NegGrad FOR BETA = 0, FinetuneUnlearner FOR BETA = 1.

        Returns:
            If self.evaluate is True:
                Tuple of (unlearned_model, losses_dict) where losses_dict contains
                tracked losses for each dataset type.
            Otherwise:
                The unlearned model.
                
        Raises:
            ValueError: If either 'forget' or 'retain' data is missing from data_dict,
                       or if beta is 0 or 1 (which would make this equivalent to simpler methods).
        """
        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        # Validate and extract common hyperparameters
        loss_fn, num_epochs, lr, weight_decay, use_l2_penalty = self.valid_args(**kwargs)
        beta = kwargs.get("beta")
        # Ensure required data is available
        if 'forget' not in data_dict.keys() or 'retain' not in data_dict.keys():
            raise ValueError("'forget' and 'retain' data must be in data_dict.")

        if beta == 0:
            raise ValueError("Please use NegGrad if you wish to perform "
                             "gradient ascent on only the forget set.")
        if beta == 1:
            raise ValueError("Please use FinetuneUnlearner if you wish to "
                             "perform gradient descent on only the retain set")

        # Initialize loss tracking if evaluation is enabled
        if self.evaluate:
            losses = {f"{data_type}_losses": [] for data_type in data_dict.keys()}

        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        eval_only_data = [data for data in data_dict.keys() if data not in ['forget', 'retain']]

        # Main training loop
        for _ in range(num_epochs):
            total_forget_loss, total_retain_loss = 0, 0
            for retain_batch, forget_batch in zip(data_dict['retain'],
                                                  cycle(data_dict['forget'])
                                                  ):
                unlearned_model.train()
                optimizer.zero_grad()
                # Process forget batch
                forget_batch = [
                    tensor.to(self.device) for tensor in forget_batch
                ]
                # Compute the forget set and retain set loss. Cycle forget set.
                forget_inputs, forget_labels = forget_batch
                forget_output = unlearned_model(forget_inputs)

                # Process retain batch
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
                # Add L2 penalty if requested
                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=model,
                                         weight_decay=weight_decay)
                    loss += l2_loss
                loss.backward()
                optimizer.step()
                # Track losses
                total_forget_loss += forget_loss.item()
                total_retain_loss += retain_loss.item()

            # Track metrics if evaluation is enabled
            if self.evaluate:
                # Calculate average losses for this epoch
                losses['forget_losses'].append(total_forget_loss/len(data_dict['forget']))
                losses['retain_losses'].append(total_retain_loss/len(data_dict['retain']))
                # Evaluate model on other datasets
                unlearned_model.eval()
                for data_type in eval_only_data:
                    loader_loss = self._evaluate(unlearned_model, data_dict[data_type], loss_fn).mean()
                    losses[f"{data_type}_losses"].append(loader_loss.item())

        if self.evaluate:
            return unlearned_model, losses
        return unlearned_model
