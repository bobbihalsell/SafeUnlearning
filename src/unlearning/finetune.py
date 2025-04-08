import os
import timm
import torch
import torch.nn as nn
import copy
from unlearning.utils import (l2_penalty)
from itertools import cycle
from unlearning.base import BaseUnlearner
from typing import Dict
from torch.utils.data import DataLoader

class FinetuneUnlearner(BaseUnlearner):
    """
    A class for machine unlearning through fine-tuning on retain data only.
    
    This method creates a copy of the original model and fine-tunes it
    using only the retain dataset, effectively causing the model to "forget"
    the forget dataset by not reinforcing those patterns during retraining.
    
    This method is computationally efficient but may not provide strong forgetting
    guarantees for models that have already memorized the forget data.
    """
    def __init__(self, device):
        """
        Initialize the FinetuneUnlearner class.

        Args:
            device: Computing device (CPU/GPU) to use for computations.
                   If None, will be automatically determined.
        """
        super().__init__(device)

    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                verbose: bool = False,
                **kwargs):
        """
        Unlearn by fine-tuning the model on retain data only.

        Args:
            model: The original model to perform unlearning on.
            data_dict: Dictionary of dataloaders, must include a 'retain' key with
                      the data to retain. Other keys (e.g., 'forget', 'test') will
                      be used for evaluation.
            **kwargs: Additional arguments including:
                - loss_fn: Loss function to use for training.
                - num_epochs: Number of training epochs (default: 1).
                - lr: Learning rate (default: 1e-2).
                - weight_decay: Weight decay parameter (default: 0).
                - use_l2_penalty: Whether to add L2 regularization (default: False).

        Returns:
            Tuple of (unlearned_model, losses_dict) where losses_dict contains
            tracked losses for each dataset type.

        Raises:
            ValueError: If 'retain' data is not in data_dict.
        """
        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        # Validate and extract common hyperparameters
        loss_fn, num_epochs, lr, weight_decay, use_l2_penalty = self.valid_args(**kwargs)

        # Ensure required data is available
        if 'retain' not in data_dict.keys():
            raise ValueError("'retain' data must be in data_dict.")

        # Initialize loss tracking
        losses = {f"{data_type}_losses": [] for data_type in data_dict.keys()}

        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

        eval_only_data = [data for data in data_dict.keys() if 
                          data != 'retain' and
                          data_dict[data] is not None]

        # Main training loop
        for epoch in range(num_epochs):
            total_retain_loss = 0
            num_batches = 0
            for retain_inputs, retain_labels in data_dict['retain']:
                num_batches += 1
                model.train()
                optimizer.zero_grad()

                retain_inputs = retain_inputs.to(self.device)
                retain_labels = retain_labels.to(self.device)

                retain_output = unlearned_model(retain_inputs)
                retain_loss = loss_fn(retain_output, retain_labels)
                total_retain_loss += retain_loss.item()

                # Add L2 penalty if requested to maintain similarity to original model
                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=model,
                                         weight_decay=weight_decay)
                    retain_loss += l2_loss

                retain_loss.backward()
                optimizer.step()

            avg_retain_loss = total_retain_loss/num_batches
            if verbose:
                print(f'Epoch {epoch+1}: Retain Loss: {avg_retain_loss}')

            # Calculate average retain loss for this epoch
            losses['retain_losses'].append(total_retain_loss)
            unlearned_model.eval()
            # Evaluate model on other datasets
            for data_type in eval_only_data:
                loader_loss = self._evaluate(unlearned_model, data_dict[data_type], loss_fn).mean()
                losses[f"{data_type}_losses"].append(loader_loss.item())
                if verbose:
                    print(f'{data_type.capitalize()} Loss: {loader_loss}',
                          end=' ')

        return unlearned_model, losses
