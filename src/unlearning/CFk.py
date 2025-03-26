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


class KUnlearn(FinetuneUnlearner):
    """
    A class for fine-tuning a given PyTorch model using a subset of data,
    while freezing the first k layers.
    """
    def __init__(self, 
                k: int,
                device,
                evaluate: bool = False,
                method: str = 'CFk',
                ):
        """
        Initialize the KUnlearn class.

        Args:
            k (int): Number of layers to freeze from the beginning of the model.
            device: The device to use for computations.
            evaluate (bool): Whether to track evaluation metrics.
            method (str): 'CFk' catastrophic forgeting or 'EUk' exact unlearning (reset weights).
        """
        super().__init__(device, evaluate)
        self.k = k
        self.method = method

    def _freeze_first_k_layers(self, model):
        """Freezes the first k layers of the model to prevent updates."""
        layers = list(model.children())
        if len(layers) < self.k:
            raise ValueError("k is larger than the total number of layers in the model.")
        
        for i in range(self.k):
            for param in layers[i].parameters():
                param.requires_grad = False
        return model

    def _reinitialize_weights(self, model, method="zero"):
        """
        Reinitializes the weights of all layers after the k-th layer.

        Args:
            method (str): The initialization method. Options:
                - "zero": Sets weights to zero.
                - "xavier": Applies Xavier uniform initialization.
                - "randn": Initializes weights with a std normal distribution.
        """
        layers = list(model.children())
        for i in range(self.k, len(layers)):
            for param in layers[i].parameters():
                if method == "zero":
                    param.data.zero_()
                elif method == "xavier":
                    if param.dim() > 1:  # Only apply Xavier to weight tensors, not biases
                        nn.init.xavier_uniform_(param)
                elif method == "randn":
                    param.data.normal_()
                else:
                    raise ValueError(f"Unknown method: {method}")
        return model

    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                reinit_method = 'random',
                **kwargs):
        """
        Fine-tune the model using provided retain data while freezing the first k layers.

        Args:
            model (nn.Module): The model to unlearn.
            data_dict (Dict[str, DataLoader]): Dictionary of dataloaders for different datasets.
            **kwargs: Additional arguments passed to the parent unlearn method.

        Returns:
            Tuple containing the unlearned model and loss dictionary.
        """
        # Create a copy of the model to avoid modifying the original
        modified_model = copy.deepcopy(model)
        
        # Freeze the first k layers of the model
        modified_model = self._freeze_first_k_layers(modified_model)
        if self.method == 'EUk':
            modified_model = self._reinitialize_weights(modified_model, method=reinit_method)

                
        # Call the parent class's unlearn method with the modified model
        return super().unlearn(modified_model, data_dict, **kwargs)