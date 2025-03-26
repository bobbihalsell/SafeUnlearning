import os
import timm
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
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


class SCRUB(BaseUnlearner):
    """
    A class for performing SCRUB unlearning a given PyTorch model using a subset of retain and forget data.
    The algorithm is based on the paper "Towards Unbounded Machine Unlearning".
    which can be found at https://arxiv.org/abs/2302.09880
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
        super().__init__(device, evaluate)

    def _kl_divergence(self, 
                       model1_logits: torch.Tensor, 
                       model2_logits: torch.Tensor
                       ) -> torch.Tensor:
        
        if model1_logits.shape != model2_logits.shape:
            raise ValueError("Model logits must have the same shape.")
        
        model1_logits = model1_logits.to(dtype=torch.float32)
        model2_logits = model2_logits.to(dtype=torch.float32)

        log_model1_probs = F.log_softmax(model1_logits, dim=1)
        model2_probs = F.softmax(model2_logits, dim=1)
        model2_probs = torch.clamp(model2_probs, min=1e-10)  # Avoid log(0)

        return F.kl_div(log_model1_probs, model2_probs, reduction='sum')
    
    def forget_loss(self, original_out, unl_out):
        Nf = len(original_out)
        forget_kl = self._kl_divergence(original_out, unl_out)
        return forget_kl/Nf
    
    def retain_loss(self, original_out, unl_out, true_y, alpha, gamma, criterion):
        Nr = len(original_out)
        retain_kl = self._kl_divergence(original_out, unl_out)
        retain_ce = criterion(unl_out, true_y)
        return (alpha * retain_kl + gamma * retain_ce)/Nr, retain_kl/Nr, retain_ce/Nr
    
    def max_epoch(self, forget_data, optimizer, step=True):
        if isinstance(forget_data, tuple):
            forget_dataset = TensorDataset(*forget_data)
            forget_loader = DataLoader(forget_dataset, batch_size=len(forget_dataset), shuffle=False)
        else:
            forget_loader = forget_data

        avg_loss = 0.0
        for forget_batch in forget_loader:
            forget_x = forget_batch[0]
            loss = self.forget_loss(forget_x)
            if step:
                optimizer.zero_grad()
                (-loss).backward()
                optimizer.step()
            avg_loss += loss
        avg_loss = avg_loss/len(forget_loader)
        return avg_loss
    
    def min_epoch(self, retain_data, optimizer, alpha, gamma, criterion):
        if isinstance(retain_data, tuple):
            retain_dataset = TensorDataset(*retain_data)
            retain_loader = DataLoader(retain_dataset, batch_size=len(retain_dataset), shuffle=False)
        else:
            retain_loader = retain_data
        avg_loss = 0.0
        avg_kl_loss = 0.0
        avg_ce_loss = 0.0
        for retain_batch in retain_loader:
            retain_x, retain_y = retain_batch
            loss, kl_loss, ce_loss = self.retain_loss(retain_x, retain_y, alpha, gamma, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss
            avg_kl_loss += kl_loss
            avg_ce_loss += ce_loss
        avg_loss = avg_loss/len(retain_loader)
        avg_kl_loss = avg_kl_loss/len(retain_loader)
        avg_ce_loss = avg_ce_loss/len(retain_loader)

        return avg_loss, avg_kl_loss, avg_ce_loss

    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                min_epochs: int,
                max_epochs: int,
                **kwargs):
        """
        """
        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        # Check if generic unlearning args are valid
        loss_fn, _, lr, weight_decay, use_l2_penalty = self.valid_args(kwargs)
        if min_epochs < 1 or max_epochs < 1:
            raise ValueError("Number of min and max epochs must be greater than 0.")
        if 'retain' not in data_dict.keys() and 'forget' not in data_dict.keys():
            raise ValueError("'forget' and 'retain' data must be in data_dict.")
        
        if self.evaluate:
        # Initialize loss tracking
            losses = {f"{data_type}_losses": [] for data_type in data_dict.keys()}

        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        
        eval_only_data = [data for data in data_dict.keys() if data not in ['retain']]

        num_epochs = max(min_epochs, max_epochs)
        min_i=0
        max_i=0

        for _ in range(num_epochs):
            unlearned_model.train()
            total_retain_loss = 0
            if max_i < max_epochs:
                self.max_epoch(data_dict['forget'], optimizer)
                max_i+=1
            if min_i < min_epochs:
                _, _, retain_loss = self.min_epoch(data_dict['retain'], optimizer, **kwargs)
                min_i+=1
                total_retain_loss += retain_loss

            if self.evaluate:
                losses['retain_losses'].append(total_retain_loss/len(data_dict['retain']))
                unlearned_model.eval()
                for data_type in eval_only_data:
                    loader_loss = self._evaluate(unlearned_model, data_dict[data_type], loss_fn).mean()
                    losses[f"{data_type}_losses"].append(loader_loss.item())

        if self.evaluate:
            return unlearned_model, losses
        return unlearned_model