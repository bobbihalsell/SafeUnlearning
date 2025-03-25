import deepcopy
import importlib
import pathlib
import typing as typ
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

class BaseUnlearner:
    """ Base class for all """
    def __init__(self, 
                model: nn.Module,
                device,
                save_path: str = None,
                save_steps: bool = False,
                **kwargs):
        self.original_model = model.deepcopy()
        for key, value in kwargs.items():
            setattr(self, key, value)


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
