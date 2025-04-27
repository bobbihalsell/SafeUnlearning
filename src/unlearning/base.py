from torch.utils.data import DataLoader
from abc import abstractmethod
import torch
import torch.nn as nn
from utils import setup_device
from typing import Dict
import copy
import time
import wandb


class BaseUnlearner:
    """ Base class for all machine unlearning implementations."""
    def __init__(self,
                 device,
                 evaluate: bool = False,
                 wandb_enabled: bool = False,
                 verbose: bool = True
                 ):
        """
        Initialize the BaseUnlearner.

        Args:
            device: Computing device (GPU/CPU) to use for computations.
                   If None, will be automatically determined.
        """
        self.device = device if device is not None else setup_device()
        self.evaluate = evaluate
        self.criterion = nn.CrossEntropyLoss()
        self.wandb_enabled = wandb_enabled
        self.verbose = verbose

    @abstractmethod
    def unlearn(
        self,
        model: nn.Module,
        data_dict: Dict[str, DataLoader],
        **kwargs
    ):
        """
        Unlearns specific data from a model while retaining performance on
        other data.

        This is an abstract method that must be implemented by all subclasses
        with their specific unlearning strategy.

        Args:
            model: The model to perform unlearning on.
            data_dict: A dictionary of dataloaders with the data type as keys
            **kwargs: Algorithm-specific hyperparameters
        Returns:
            Tuple of the unlearned model, and a logs dictionary.
        """
        # This method must be implemented by subclasses
        pass

    def _evaluate(self,
                  model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  ) -> torch.Tensor:
        """
        Compute the evaluation loss of a model on a given dataset.

        Args:
            model: The model to evaluate.
            dataloader: DataLoader containing evaluation data.

        Returns:
            Tuple (val_loss, val_acc)
        """
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        if len(dataloader.dataset) > 0:
            val_loss /= len(dataloader.dataset)
        else:
            val_loss = 0

        val_acc = 100.0 * correct / total if total > 0 else 0

        return val_loss, val_acc

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
            momentum = getattr(self, "momentum", None)
            if momentum is None:
                momentum = 0
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=self.lr,
                                        momentum=momentum,
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

    def _evaluate_all_splits(self,
                             model: nn.Module,
                             data_dict: Dict[str, DataLoader],
                             epoch: int,
                             ):
        """ Evaluate the model on the data splits in data dict.

        Args:
            model: nn.Module
            data_dict: The dictionary of dataloaders relevant to unlearning.

        Returns:
            self.logs (dict): A logs dictionary
        """
        model.eval()

        for data_type, loader in data_dict.items():
            start_eval_time = time.time()
            loader_loss, loader_acc = self._evaluate(model,
                                                     loader)

            self.logs[f"{data_type}"].append(loader_loss)
            self.logs[f"{data_type}_acc"].append(loader_acc)
            elapsed = time.time() - start_eval_time
            self.logs[f"{data_type}_time"].append(elapsed)
            if self.verbose:
                print(f'{data_type.capitalize()} Loss: {loader_loss:.4f} '
                      f'Acc: {loader_acc:.2f}%. '
                      f'Time: {elapsed:.1f} s', end=' || ')
        if self.verbose:
            print('')

        if self.wandb_enabled:
            self._log_metrics_in_wandb(
                epoch=epoch,
                retain_loss=self.logs['retain'][-1],
                retain_acc=self.logs['retain_acc'][-1],
                retain_time=self.logs['retain_time'][-1],
                forget_loss=self.logs['forget'][-1],
                forget_acc=self.logs['forget_acc'][-1],
                forget_time=self.logs['forget_time'][-1],
                val_loss=self.logs['val'][-1],
                val_acc=self.logs['val_acc'][-1],
                val_time=self.logs['val_time'][-1],
            )

        return self.logs

    def _setup_unlearning(self, model, data_dict, **kwargs):
        """ Setup unlearned model, losses dictionary, optimizer, scheduler."""
        unlearned_model = copy.deepcopy(model)

        # Validate and extract common hyperparameters
        self.valid_args(**kwargs)
        self.extract_hyperparameters(**kwargs)

        # Initialize loss and accuracy tracking
        self.logs = {f"{data_type}": [] for data_type in data_dict.keys()}
        self.logs.update({
            f"{data_type}_acc": [] for data_type in data_dict.keys()
        })
        self.logs.update({
            f"{data_type}_time": [] for data_type in data_dict.keys()
        })
        self._eval_initial_model(model,
                                 data_dict)

        self.optimizer = self.initialize_optimizer(
            unlearned_model,
            optimizer_name=self.optimizer)

        if (self.epochs_per_lr_decay is not None and
                self.lr_decay_factor is not None):
            scheduler = self.initialize_scheduler(optimizer=self.optimizer)
        else:
            scheduler = None

        return unlearned_model, scheduler

    def _eval_initial_model(self, model, data_dict):
        """ Evaluate the initial model's performance on all dataset splits."""
        for data_type in data_dict.keys():
            split_loss, split_acc = self._evaluate(
                model,
                dataloader=data_dict[data_type],
            )
            if self.wandb_enabled:
                wandb.log({
                    f'Initial {data_type.capitalize()} Loss': split_loss,
                    f'Initial {data_type.capitalize()} Acc': split_acc
                })

            self.logs[data_type].append(split_loss)
            self.logs[f"{data_type}_acc"].append(split_acc)

            if self.verbose:
                print(f'Initial {data_type.capitalize()} Loss: '
                      f'{split_loss:.4f}. '
                      f'Acc: {split_acc:.2f}%.', end=' || ')
        if self.verbose:
            print('')

    def _print_forward_pass_metrics(self, epoch_num, time_taken):
        """ Print metrics for the user when forward pass is complete."""
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch_num+1} Forward Pass Complete. '
              f'LR: {current_lr:.5f}. '
              f'Time taken: {time_taken:.1f} s')

    def _log_forward_pass_time_in_wandb(self, epoch, time):
        wandb.log({
            'Epoch': epoch,
            'forward_pass_time': time
        })

    def _log_metrics_in_wandb(self,
                              epoch: int,
                              retain_loss: float,
                              retain_acc: float,
                              retain_time: float,
                              forget_loss: float,
                              forget_acc: float,
                              forget_time: float,
                              val_loss: float,
                              val_acc: float,
                              val_time: float):
        """ Log evaluation metrics in WandB."""
        wandb.log({
            "Retain/Loss": retain_loss,
            "Retain/Accuracy": retain_acc,
            "Retain/Time": retain_time,
            "Forget/Loss": forget_loss,
            "Forget/Accuracy": forget_acc,
            "Forget/Time": forget_time,
            "Val/Loss": val_loss,
            "Val/Accuracy": val_acc,
            "Val/Time": val_time,
            "Epoch": epoch
            })
