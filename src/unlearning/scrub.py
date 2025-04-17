import torch
import torch.nn as nn
import torch.nn.functional as F
from unlearning.base import BaseUnlearner
from typing import Dict
from torch.utils.data import DataLoader
import time


class SCRUB(BaseUnlearner):
    """
    Implementation of the SCRUB unlearning algorithm as described in 
    "Towards Unbounded Machine Unlearning" (https://arxiv.org/abs/2302.09880).

    SCRUB employs a bi-level optimization strategy to:
    1. Maximize divergence on forget data (maximize KL divergence)
    2. Minimize divergence on retain data (minimize KL divergence + CE loss)

    This approach ensures the model "forgets" specific data while maintaining
    performance on data that should be retained.
    """
    def __init__(self,
                 device,
                 evaluate: bool = False,
                 wandb_enabled: bool = False
                 ):
        """
        Initialize the SCRUB unlearning class.

        Args:
            device: Computing device (CPU/GPU) to use for computations.
                   If None, will be automatically determined.
            evaluate: Whether to track and return evaluation metrics during unlearning.
        """
        super().__init__(device, evaluate, wandb_enabled)
        # Following authors' specification
        self.criterion = nn.CrossEntropyLoss()

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

    def retain_loss(self, original_out, unl_out, true_y):
        """
        Compute the composite loss over a set of retain data.

        For retain data, we want to:
            - Minimize KL divergence between original and unlearned models
            - Maintain classification accuracy through cross-entropy loss

        Args:
            original_out: Output logits from the original model
            unl_out: Output logits from the unlearned model
            true_y: Ground truth labels

        Returns:
            Tuple containing:
            - Combined weighted loss (KL + CE)
            - KL divergence component (raw)
            - Cross-entropy component (raw)
        """
        Nr = len(original_out)
        retain_kl = self._kl_divergence(original_out, unl_out)
        retain_ce = self.criterion(unl_out, true_y)

        return (self.alpha * retain_kl)/Nr + self.gamma * retain_ce, retain_kl, retain_ce

    def max_epoch(self, model, unlearned_model,
                  forget_loader, step=True):
        """
        Perform one epoch of maximizing divergence on forget data.
        This is the "forgetting" step where we make the model outputs diverge
        from the original model on data that should be forgotten.

        Args:
            forget_loader: The forget DataLoader
            optimizer: Optimizer for updating model parameters
            step: Whether to perform optimization step (True) or just compute loss (False)

        Returns:
            Average KL loss across all batches
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
                self.optimizer.zero_grad()
                # We negate the loss because we want to maximize divergence
                (-loss).backward()
                self.optimizer.step()
            avg_loss += loss
        # Normalize by number of batches
        avg_loss = avg_loss/len(forget_loader)

        return avg_loss

    def min_epoch(self, model, unlearned_model, retain_loader):
        """
        Perform one epoch of minimizing divergence on retain data.

        This is the "retaining" step where we ensure the model maintains
        performance on data that should be retained.

        Args:
            retain_loader: The retain DataLoader

        Returns:
            Average cross-entropy loss, for performance reporting
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
                original_out, unl_out, retain_y
                )
            self.optimizer.zero_grad()
            # Perform optimization using combined loss
            loss.backward()
            self.optimizer.step()

            avg_ce_loss += ce_loss
        avg_ce_loss = (avg_ce_loss/len(retain_loader) if
                       len(retain_loader) > 0 else 0)

        return avg_ce_loss

    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
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

            **kwargs: Additional hyperparameters for SCRUB

        Returns:
            Tuple of (unlearned_model, logs)
        """
        if ('retain' not in data_dict.keys() or
                'forget' not in data_dict.keys()):
            raise KeyError("forget and retain data must be in data_dict.")
        model.to(self.device)
        unlearned_model, scheduler = self._setup_unlearning(
            model,
            data_dict,
            **kwargs)

        if self.alpha < 0 or self.gamma < 0:
            raise ValueError("Alpha and gamma must be non-negative.")

        # Calculate total number of epochs and initialize counters
        total_epochs = self.max_epochs + self.min_epochs
        for e in range(total_epochs):
            epoch_start_time = time.time()
            model.eval()
            unlearned_model.eval()

            # Maximize divergence on forget data
            if e < self.max_epochs:
                self.max_epoch(
                    model,
                    unlearned_model,
                    data_dict['forget']
                )
            # Minimize divergence on retain data
            self.min_epoch(
                    model,
                    unlearned_model,
                    data_dict['retain'],
                )
            forward_pass_elapsed = time.time() - epoch_start_time

            if verbose:
                self._print_forward_pass_metrics(e, forward_pass_elapsed)

            if self.evaluate:
                self._evaluate_all_splits(
                    model=unlearned_model,
                    data_dict=data_dict,
                    epoch=e+1,
                    verbose=verbose
                )

            if scheduler is not None:
                scheduler.step()

        return unlearned_model, self.logs
