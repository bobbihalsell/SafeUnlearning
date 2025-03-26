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

class FinetuneUnlearner(BaseUnlearner):
    """
    A class for fine-tuning a given PyTorch model using a subset of data.
    Supports both single-batch and mini-batch fine-tuning.
    """
    def __init__(self, 
                device,
                evaluate: bool = False,
                ):
        """
        Initialize the Finetune class.

        Args:
            original_model (nn.Module): The base model to perform fine-tune unlearning on.
        """
        super().__init__(device, save_steps)

    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                **kwargs):
        """
        Fine-tune the model using provided retain data and optionally evaluate on test datasets.

        Args:
            retain_data (DataLoader): 
                Dataloader for the retain dataset.
            criterion (nn.Module): 
                Loss function used for training and evaluation.
            optimizer (torch.optim.Optimizer): 
                Optimizer for updating model parameters.
            epochs (int): 
                Number of fine-tuning epochs.
            test_data_dict (Dict[str, DataLoader], optional): 
                Dictionary mapping dataset names (e.g., 'test', 'forget') to dataloaders.
                Defaults to None.
            track_loss (bool, optional): 
                If True, stores training and test losses for each epoch. Defaults to True.

        Returns:
            None
        """
        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        # Check if generic unlearning args are valid
        loss_fn, num_epochs, lr, weight_decay, use_l2_penalty = self.valid_args(kwargs)

        if 'retain' not in data_dict.keys():
            raise ValueError("'retain' data must be in data_dict.")
        
        if self.evaluate:
        # Initialize loss tracking
            losses = {f"{data_type}_losses": [] for data_type in data_dict.keys()}

        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        
        eval_only_data = [data for data in data_dict.keys() if data != 'retain']

        for _ in range(num_epochs):
            total_retain_loss = 0
            for retain_inputs, retain_labels in data_dict['retain']:
                batch_size = retain_inputs.size(0)

                model.train()
                optimizer.zero_grad()

                retain_inputs = retain_inputs.to(self.device)
                retain_labels = retain_labels.to(self.device)

                retain_output = unlearned_model(retain_inputs)
                retain_loss = loss_fn(retain_output, retain_labels)
                total_retain_loss += retain_loss.item()/batch_size

                if use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=model,
                                         weight_decay=weight_decay)
                    retain_loss += l2_loss

                retain_loss.backward()
                optimizer.step()
            

            if self.evaluate:
                losses['retain_losses'].append(total_retain_loss/len(data_dict['retain']))
                unlearned_model.eval()
                for data_type in eval_only_data:
                    loader_loss = self._evaluate(unlearned_model, data_dict[data_type], loss_fn).mean()
                    losses[f"{data_type}_losses"].append(loader_loss.item())

        return unlearned_model, losses