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
                model: nn.Module,
                device,
                save_path: str = None,
                save_steps: bool = False,
                **kwargs):
        if not isinstance(model, nn.Module):
            raise UnsupportedModelError('original_model must be a Pytorch model.')
        self.original_model = model.deepcopy()
        self.unlearned_model = model.deepcopy()
        self.device = device if device is not None else setup_device()        
        self.save_path = save_path
        self.save_steps = save_steps
        # for key, value in kwargs.items():
        #     setattr(self, key, value)


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
                setattr(self, f"has_{data_type}_tensor", True)
            elif isinstance(data, DataLoader):
                data = data
                setattr(self, f"has_{data_type}_tensor", False)
            else:
                raise TypeError(f"data must be a tuple of (X, y) tensors or a DataLoader")
            setattr(self, f"{data_type}_loader", data)

    def _evaluate(self,
                  model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: nn.Module) -> torch.Tensor:
        """ Compute the validation loss of a model."""
        model = model.eval()
        batch_losses = torch.zeros(len(self.val_dataloader))
        for batch_ndx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            batch_losses[batch_ndx] = loss.detach().item()
        return batch_losses
            