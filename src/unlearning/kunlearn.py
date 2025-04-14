import torch.nn as nn
import copy
from unlearning.finetune import FinetuneUnlearner
from typing import Optional, Union, Tuple, List, Dict, Any
from torch.utils.data import DataLoader, TensorDataset


class KUnlearn(FinetuneUnlearner):
    """
    Implementation of layer-selective unlearning that freezes the first k layers.
    
    This approach is based on the concept that different layers in neural networks
    capture different levels of abstraction. By freezing the first k layers and
    either:
    1. Fine-tuning only the remaining layers (cfk - Catastrophic Forgetting)
    2. Reinitializing and then fine-tuning the remaining layers (euk - Exact Unlearning)
    
    The model selectively retains general knowledge in early layers while modifying
    later layers to "forget" specific data points or classes.
    """
    def __init__(self, 
                k: int,
                device,
                evaluate: bool = False,
                method: str = 'cfk',
                init=None
                ):
        """
        Initialize the KUnlearn class.

        Args:
            k (int): Number of layers to freeze from the beginning of the model.
                    These layers will maintain their original weights.
            device: Computing device (CPU/GPU) to use for computations.
                    If None, will be automatically determined.
            evaluate (bool): Whether to track and return evaluation metrics during unlearning.
            method (str): Unlearning approach to use. Options:
                         - 'cfk': Catastrophic Forgetting - freeze first k layers and fine-tune the rest.
                         - 'euk': Exact Unlearning - freeze first k layers, reinitialize the rest,
                                  and then fine-tune.
        """
        super().__init__(device, evaluate)
        self.k = k
        assert method in ['cfk', 'euk'], "Method must be either 'cfk' or 'euk'."
        self.method = method
        self.init = init


    def _freeze_first_k_layers(self, model):
        """
        Freezes the first k layers of the model to prevent updates during training.

        Args:
            model (nn.Module): PyTorch model to modify
            
        Returns:
            nn.Module: Model with first k layers frozen
        """
        # Get top-level layers (direct children) of the model
        layers = list(model.children())

        # Validate k value against model structure
        if len(layers) < self.k:
            raise ValueError("k is larger than the total number of layers in the model.")
        
        # Freeze parameters in the first k layers
        for i in range(self.k):
            for param in layers[i].parameters():
                param.requires_grad = False
        return model

    def _reinitialize_weights(self, model):
        """
        Reinitializes the weights of all layers after the k-th layer.
        
        This is used in the euk approach to completely reset specific layers,
        forcing the model to relearn patterns from scratch on those layers.

        Args:
            model (nn.Module): PyTorch model to modify
            method (str): The initialization method. Options:
                - "zero": Sets weights to zero (most aggressive reset)
                - "randn": Initializes weights with a standard normal distribution

        Returns:
            nn.Module: Model with reinitialized weights in layers after k
        """
        # Get top-level layers of the model
        layers = list(model.children())

        # Reinitialize parameters in layers after the k-th layer
        for i in range(self.k, len(layers)):
            for param in layers[i].parameters():
                if self.init == "zero":
                    # Set all weights to zero
                    param.data.zero_()
                elif self.init == "randn":
                    # Initialize with standard normal distribution
                    param.data.normal_()
                else:
                    raise ValueError(f"Invalid initialiastion: {self.init}")
        return model

    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                **kwargs):
            """
            Perform K-unlearning by freezing the first k layers and fine-tuning the rest.
            
            This method implements two approaches to k-unlearning:
            1. CFk (Catastrophic Forgetting): Freeze first k layers and fine-tune the rest
            2. EUk (Exact Unlearning): Freeze first k layers, reinitialize remaining layers,
            and then fine-tune

            Args:
                model (nn.Module): The original model to unlearn from
                data_dict (Dict[str, DataLoader]): Dictionary containing dataloaders for different datasets
                                                Must include 'retain' data
                reinit_method (str): Method to use for reinitializing weights in EUk approach
                                    Options: 'zero', 'randn'
                **kwargs: Additional arguments passed to the parent unlearn method, including:
                        - loss_fn: Loss function to use for training
                        - num_epochs: Number of training epochs
                        - lr: Learning rate
                        - weight_decay: Weight decay parameter
                        - use_l2_penalty: Whether to add L2 regularization

            Returns:
                Tuple of (unlearned_model, losses_dict) where losses_dict contains
                tracked losses for each dataset type
                    
            Raises:
                ValueError: If k is larger than the number of layers in the model
            """
            # Create a copy of the model to avoid modifying the original
            modified_model = copy.deepcopy(model)
            
            # Freeze the first k layers of the model
            modified_model = self._freeze_first_k_layers(modified_model)
            if self.method == 'euk':
                modified_model = self._reinitialize_weights(modified_model)
            return super().unlearn(modified_model, data_dict, **kwargs)