import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from typing import Optional, Tuple, List, Dict
from torch.utils.data import DataLoader
from src.unlearning.utils import setup_device, UnsupportedModelError
import copy
import deepcopy
from base import BaseUnlearner

class FinetuneUnlearner(BaseUnlearner):
    """
    A class for fine-tuning a given PyTorch model using a subset of data.
    Supports both single-batch and mini-batch fine-tuning.
    """
    def __init__(self, 
                model: nn.Module,
                device,
                save_path: str = None,
                save_steps: bool = False,
                **kwargs):
        """
        Initialize the Finetune class.

        Args:
            original_model (nn.Module): The base model to perform fine-tune unlearning on.
        """
        super().__init__(model, device, save_path, save_steps, **kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fine-tuned model.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Model predictions.
        """
        return self.finetuned_model(x)


    def unlearn(self, 
                retain_data: DataLoader, 
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                epochs: int, 
                test_data_dict: Dict[str, DataLoader] = None
                ):
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

        # Initialize loss storage
        if self.save_steps:
            self.retain_losses = []
            self.save_steps = []
            if test_data_dict is not None:
                for name in test_data_dict:
                    setattr(self, f"{name}_losses", [])

        for epoch in range(epochs):
            self.unlearned_model.train()
            running_retain_loss = 0.0

            # Training loop with mini-batches
            for retain_X, retain_y in retain_data:
                retain_X, retain_y = retain_X.to(self.device), retain_y.to(self.device)
                optimizer.zero_grad()
                retain_outputs = self.unlearned_model(retain_X)
                retain_loss = criterion(retain_outputs, retain_y)
                retain_loss.backward()
                optimizer.step()
                running_retain_loss += retain_loss.item()
            
            avg_retain_loss = running_retain_loss / len(retain_data)
            self.retain_losses.append(avg_retain_loss)

            # Evaluate on test datasets if provided
            if test_data_dict is not None:
                self.unlearned_model.eval()
                test_losses = {}  # Dictionary to store test losses for printing
                with torch.no_grad():
                    for name, loader in test_data_dict.items():
                        running_test_loss = 0.0
                        for X, y in loader:
                            X, y = X.to(self.device), y.to(self.device)
                            out = self.unlearned_model(X)
                            loss = criterion(out, y)
                            running_test_loss += loss.item()
                        
                        avg_loss = running_test_loss / len(loader)
                        getattr(self, f"{name}_losses").append(avg_loss)
                        test_losses[name] = avg_loss
                # Print epoch loss
                loss_str = f"Epoch [{epoch+1}/{epochs}], Retain Loss: {avg_retain_loss:.4f}"
                if test_losses:
                    test_loss_str = ", ".join([f"{name} Loss: {loss:.4f}" for name, loss in test_losses.items()])
                    loss_str += f", {test_loss_str}"
                    print(loss_str)
                else:
                    print(f"Epoch [{epoch+1}/{epochs}], Retain Loss: {avg_retain_loss:.4f}")

    def reset(self):
        """Reset the fine-tuned model to its initial state."""
        self.unlearned_model = copy.deepcopy(self.model).to(self.device)
    
    def get_model(self) -> nn.Module:
        """Return the fine-tuned model."""
        return self.unlearned_model
    