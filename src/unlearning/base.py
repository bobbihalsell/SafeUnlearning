from torch.utils.data import DataLoader
from abc import abstractmethod
import torch
import torch.nn as nn
from unlearning.utils import setup_device


class BaseUnlearner:
    """ Base class for all machine unlearning implementations."""
    def __init__(self,
                 device,
                 ):
        """
        Initialize the BaseUnlearner.

        Args:
            device: Computing device (GPU/CPU) to use for computations.
                   If None, will be automatically determined.
        """
        self.device = device if device is not None else setup_device()        

    @abstractmethod
    def unlearn(
        self,
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        """
        Unlearns specific data from a model while retaining performance on other data.

        This is an abstract method that must be implemented by all subclasses
        with their specific unlearning strategy.

        Args:
            model: The model to perform unlearning on.
            retain_loader: DataLoader containing data the model should continue to perform well on.
            forget_loader: DataLoader containing data the model should "forget".
            val_loader: DataLoader containing validation data to evaluate performance.

        Returns:
            The unlearned model.
        """
        # This method must be implemented by subclasses
        pass

    def _evaluate(self,
                  model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: nn.Module,
                  ) -> torch.Tensor:
        """
        Compute the evaluation loss of a model on a given dataset.

        Args:
            model: The model to evaluate.
            dataloader: DataLoader containing evaluation data.
            loss_fn: Loss function to compute model performance.

        Returns:
            torch.Tensor: Tensor containing loss values for each batch in the dataloader.
        """
        model = model.eval()
        # Initialize tensor to store batch losses
        batch_losses = torch.zeros(len(dataloader), device=self.device)
        # Evaluate model on all batches
        for batch_ndx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            # Store loss value
            batch_losses[batch_ndx] = loss.detach().item()
        return batch_losses

    def valid_args(self, **kwargs):
        """
        Validate and extract common unlearning hyperparameters from kwargs.

        Extracts and validates learning parameters that are common across
        different unlearning methods.

        Args:
            **kwargs: Arbitrary keyword arguments containing hyperparameters.

        Returns:
            tuple: A tuple containing (loss_fn, num_epochs, lr, weight_decay, use_l2_penalty).

        Raises:
            ValueError: If any of the parameters fails validation checks.
        """
        num_epochs = kwargs['epochs']
        lr = kwargs['lr']
        # Extract parameters with default values
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