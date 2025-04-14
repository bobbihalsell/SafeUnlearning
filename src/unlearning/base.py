from torch.utils.data import DataLoader
from abc import abstractmethod
import torch
import torch.nn as nn
from unlearning.utils import setup_device


class BaseUnlearner:
    """ Base class for all machine unlearning implementations."""
    def __init__(self,
                 device,
                 evaluate: bool = False
                 ):
        """
        Initialize the BaseUnlearner.

        Args:
            device: Computing device (GPU/CPU) to use for computations.
                   If None, will be automatically determined.
        """
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
            None

        Raises:
            ValueError: If any of the parameters fails validation checks.
        """
        lr = kwargs['lr']
        weight_decay = kwargs['weight_decay']
        use_l2_penalty = kwargs['use_l2_penalty']

        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("lr must be a positive number.")
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError("weight_decay must be a non-negative number.")
        if not isinstance(use_l2_penalty, bool):
            raise ValueError("use_l2_penalty must be a boolean.")

    def extract_hyperparameters(self, **kwargs):
        """ Extract unlearning hyperparameters from kwargs"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize_optimizer(self, model: nn.Module, optimizer_name: str):
        """ Initialize an optimizer."""
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(
                'Only adam and sgd optimizers supported. '
                f'Received {optimizer_name}.')

        return optimizer

    def initialize_scheduler(self, optimizer):
        """ Initialize a learning rate scheduler from hyperparameters."""
        epochs_per_lr_decay = getattr(self, "epochs_per_lr_decay")
        lr_decay_factor = getattr(self, "lr_decay_factor")

        if epochs_per_lr_decay is not None and lr_decay_factor is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=epochs_per_lr_decay,
                gamma=lr_decay_factor)

            return scheduler

        else:
            raise Exception(
                'Attempted to initialize scheduler, but '
                'schedule information not available.')
