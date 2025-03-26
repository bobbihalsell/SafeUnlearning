import deepcopy
import importlib
import pathlib
import typing as typ
from typing import Optional, Union, Tuple, List, Dict, Any
from torch.utils.data import DataLoader, TensorDataset
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from src.unlearning.utils import setup_device, UnsupportedModelError


class BaseUnlearner:
    """ Base class for all """
    def __init__(self, 
                device,
                evaluate: bool = False
                ):
        self.device = device if device is not None else setup_device()        
        self.evaluate = evaluate

    @abstractmethod
    def unlearn(
        self,
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        """
        Unlearns the model on the forget_loader and then retrains it on the retain_loader.
        :param model: The model to unlearn.
        :param retain_loader: The data to retain.
        :param forget_loader: The data to forget.
        :param val_loader: The data to validate on.
        :return: The unlearned model.
        """
        # Unlearn 
        # Save Model

    def _process_data(
        self, 
        *data: Union[Tuple[torch.Tensor, torch.Tensor], DataLoader],
        batch_sizes: List[int],
    ) -> None:
        """
        Process and validate input data formats.
        
        Args:
            forget_data: Data to be "forgotten" (tuple of tensors or DataLoader)
            retain_data: Data to be retained (tuple of tensors or DataLoader)
            test_data: Data for testing/evaluation (tuple of tensors or DataLoader)
        """
        for i, data_type in enumerate(data):
            if isinstance(data_type, tuple) and len(data) == 2:
                X, y = data
                data = DataLoader(
                    TensorDataset(X.to(self.device), y.to(self.device)),
                    batch_size=batch_sizes[i],
                    shuffle=True
                )
                setattr(self, f"has_{data_type}_dataloader", True)
            elif isinstance(data, DataLoader):
                data = data
                setattr(self, f"has_{data_type}_dataloader", False)
            else:
                raise TypeError(f"data must be a tuple of (X, y) tensors or a DataLoader")
            setattr(self, f"{data_type}_loader", data)

    def _evaluate(self,
                  model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: nn.Module,
                  ) -> torch.Tensor:
        """ Compute the validation loss of a model."""
        model = model.eval()
        batch_losses = torch.zeros(len(self.val_dataloader))
        for batch_ndx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_losses[batch_ndx] = loss.detach().item()
        return batch_losses
            
    def valid_args(self, **kwargs):
        """ Validate input arguments."""
        num_epochs = kwargs.get('num_epochs', 1)
        lr = kwargs.get('lr', 1e-4)
        weight_decay = kwargs.get('weight_decay', 0)
        loss_fn = kwargs.get('loss_fn')
        use_l2_penalty = kwargs.get('use_l2_penalty', False)

        if not callable(loss_fn):
            raise ValueError("loss_fn must be a callable loss function.")
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("lr must be a positive number.")
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError("weight_decay must be a non-negative number.")
        if not isinstance(use_l2_penalty, bool):
            raise ValueError("use_l2_penalty must be a boolean.")
        
        return loss_fn, num_epochs, lr, weight_decay, use_l2_penalty