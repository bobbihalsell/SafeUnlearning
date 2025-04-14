import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from unlearning.base import BaseUnlearner
from typing import Dict
from torch.utils.data import DataLoader


class SCRUB(BaseUnlearner):
    """
    Implementation of the SCRUB unlearning algorithm as described in 
    "Towards Unbounded Machine Unlearning" (https://arxiv.org/abs/2302.09880).

    SCRUB employs a bi-level optimization strategy to:
    1. Maximize divergence on forget data (maximize KL divergence)
    2. Minimize divergence on retain data (minimize KL divergence + classification error)

    This approach ensures the model "forgets" specific data while maintaining
    performance on data that should be retained.
    """
    def __init__(self,
                 device,
                 evaluate: bool = False,
                 ):
        """
        Initialize the SCRUB unlearning class.

        Args:
            device: Computing device (CPU/GPU) to use for computations.
                   If None, will be automatically determined.
            evaluate: Whether to track and return evaluation metrics during unlearning.
        """
        super().__init__(device, evaluate)

    def _kl_divergence(self,
                       model1_logits: torch.Tensor,
                       model2_logits: torch.Tensor
                       ) -> torch.Tensor:
        """
        Calculate the Kullback-Leibler divergence between outputs of two models.

        Args:
            model1_logits: Logits (pre-softmax outputs) from the first model
            model2_logits: Logits (pre-softmax outputs) from the second model

        Returns:
            KL divergence between the two distributions

        Raises:
            ValueError: If the shapes of the logits don't match
        """

        if model1_logits.shape != model2_logits.shape:
            raise ValueError("Model logits must have the same shape.")

        model1_logits = model1_logits.to(dtype=torch.float32)
        model2_logits = model2_logits.to(dtype=torch.float32)

        log_model1_probs = F.log_softmax(model1_logits, dim=1)
        model2_probs = F.softmax(model2_logits, dim=1)
        model2_probs = torch.clamp(model2_probs, min=1e-10)  # Avoid log(0)

        return F.kl_div(log_model1_probs, model2_probs, reduction='sum')

    def forget_loss(self, original_out, unl_out):
        """
        Compute the training loss over a set of forget data.
        For forget data, we want to maximize the KL divergence between original
        and unlearned model outputs.

        Args:
            original_out: Output logits from the original model
            unl_out: Output logits from the unlearned model

        Returns:
            Tuple of normalized KL divergence, unnormalized KL divergence
        """
        # Get batch size for normalization
        Nf = len(original_out)
        # Compute KL divergence between original and unlearned model outputs
        forget_kl = self._kl_divergence(original_out, unl_out)

        return forget_kl/Nf, forget_kl

    def retain_loss(self, original_out, unl_out, true_y, alpha, gamma, criterion):
        """
        Compute the composite loss over a set of retain data.

        For retain data, we want to:
            - Minimize KL divergence between original and unlearned models
            - Maintain classification accuracy through cross-entropy loss

        Args:
            original_out: Output logits from the original model
            unl_out: Output logits from the unlearned model
            true_y: Ground truth labels
            alpha: Weight for the KL divergence component
            gamma: Weight for the cross-entropy component
            criterion: Loss function for classification error

        Returns:
            Tuple containing:
            - Combined weighted loss (KL + CE)
            - KL divergence component (raw)
            - Cross-entropy component (raw)
        """
        Nr = len(original_out)
        retain_kl = self._kl_divergence(original_out, unl_out)
        retain_ce = criterion(unl_out, true_y)

        return (alpha * retain_kl + gamma * retain_ce)/Nr, retain_kl, retain_ce

    def max_epoch(self, model, unlearned_model,
                  forget_loader, optimizer, step=True):
        """
        Perform one epoch of maximizing divergence on forget data.
        This is the "forgetting" step where we make the model outputs diverge
        from the original model on data that should be forgotten.

        Args:
            forget_loader: The forget DataLoader
            optimizer: Optimizer for updating model parameters
            step: Whether to perform optimization step (True) or just compute loss (False)

        Returns:
            Average loss across all batches
        """
        avg_loss = 0.0
        for forget_batch in forget_loader:
            forget_x = forget_batch[0]
            forget_x = forget_x.to(self.device)
            # Compute divergence loss
            original_out = model(forget_x)
            unl_out = unlearned_model(forget_x)
            loss, _ = self.forget_loss(original_out, unl_out)
            if step:
                optimizer.zero_grad()
                # We negate the loss because we want to maximize divergence
                (-loss).backward()
                optimizer.step()
            avg_loss += loss
        # Normalize by number of batches
        avg_loss = avg_loss/len(forget_loader)

        return avg_loss

    def min_epoch(self, model, unlearned_model, retain_loader, optimizer, alpha, gamma, criterion):
        """
        Perform one epoch of minimizing divergence on retain data.

        This is the "retaining" step where we ensure the model maintains
        performance on data that should be retained.

        Args:
            retain_loader: The retain DataLoader
            optimizer: Optimizer for updating model parameters
            alpha: Weight for the KL divergence component
            gamma: Weight for the cross-entropy component
            criterion: Loss function for classification error

        Returns:
            Tuple containing average values for:
            - Combined loss
            - KL divergence component
            - Cross-entropy component
        """
        # Average CE loss is only for reporting
        avg_ce_loss = 0.0

        for retain_batch in retain_loader:
            retain_x, retain_y = retain_batch
            retain_x = retain_x.to(self.device)
            retain_y = retain_y.to(self.device)
            # Compute retain losses
            original_out = model(retain_x)
            unl_out = unlearned_model(retain_x)
            loss, kl_loss, ce_loss = self.retain_loss(
                original_out, unl_out, retain_y, alpha, gamma, criterion
                )
            optimizer.zero_grad()
            # Perform optimization using combined loss
            loss.backward()
            optimizer.step()

            avg_ce_loss += ce_loss
        avg_ce_loss = avg_ce_loss/len(retain_loader)

        return avg_ce_loss

    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                min_epochs: int,
                max_epochs: int,
                verbose: bool = False,
                **kwargs):
        """
        Perform SCRUB unlearning
        
        SCRUB alternates between two optimization objectives:
        1. Maximizing the divergence on forget data
        2. Minimizing the divergence and classification error on retain data
        
        Args:
            model: The original model to unlearn from
            data_dict: Dictionary containing dataloaders for different datasets
                       Must include both 'forget' and 'retain' keys
            min_epochs: Number of epochs for the minimization phase (retain)
            max_epochs: Number of epochs for the maximization phase (forget)
            alpha: Weight for the KL divergence term in retain loss
            gamma: Weight for the cross-entropy term in retain loss
            **kwargs: Additional arguments including:
                - loss_fn: Loss function to use for training
                - lr: Learning rate (default: 1e-2)
                - weight_decay: Weight decay parameter (default: 0)
                - use_l2_penalty: Whether to add L2 regularization (default: False)

        Returns:
            Tuple of (unlearned_model, losses) where losses contains
            tracked losses for each dataset type
        """
        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        # Validate and extract common hyperparameters
        loss_fn, _, lr, weight_decay, _ = self.valid_args(**kwargs)

        if ('retain' not in data_dict.keys() or
                'forget' not in data_dict.keys()):
            raise KeyError("forget and retain data must be in data_dict.")

        # Extract additional hyperparameters
        alpha = kwargs['alpha']
        gamma = kwargs['gamma']
        if alpha < 0 or gamma < 0:
            raise ValueError("Alpha and gamma must be non-negative.")

        # Initialize loss tracking
        losses = {f"{data_type}_losses": [] for data_type in data_dict.keys()}

        optimizer = torch.optim.SGD(params=unlearned_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        if min_epochs > 0:
            eval_dataloaders = (['val', 'forget'] if
                                data_dict['val'] is not None else ['forget'])
        else:
            eval_dataloaders = ['retain', 'forget', 'val']

        # Calculate total number of epochs and initialize counters
        num_epochs = max(min_epochs, max_epochs)
        min_i = 0
        max_i = 0

        for e in range(num_epochs):
            model.eval()
            unlearned_model.eval()

            # Maximize divergence on forget data
            if max_i < max_epochs:
                self.max_epoch(model,
                               unlearned_model,
                               data_dict['forget'],
                               optimizer)
                max_i += 1

            # Minimize divergence on retain data
            if min_i < min_epochs:
                retain_loss = self.min_epoch(
                                    model,
                                    unlearned_model,
                                    data_dict['retain'],
                                    optimizer,
                                    alpha=alpha,
                                    gamma=gamma,
                                    criterion=loss_fn
                                )
                min_i += 1

            if verbose and min_epochs > 0:
                print(f'Epoch {e}: Retain Loss: {retain_loss}')

            if self.evaluate:
                if min_epochs > 0:
                    losses['retain_losses'].append(retain_loss)

                unlearned_model.eval()
                # Evaluate model on other datasets
                for data_type in eval_dataloaders:
                    loader_loss = self._evaluate(unlearned_model,
                                                 data_dict[data_type],
                                                 loss_fn).mean()
                    losses[f"{data_type}_losses"].append(loader_loss.item())
                    if verbose:
                        print(f'{data_type.capitalize()} Loss: {loader_loss}',
                              end='  ')
                if verbose:
                    print()

        return unlearned_model, losses
