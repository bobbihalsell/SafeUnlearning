import time
from typing import Dict

import torch.nn as nn
from torch.utils.data import DataLoader

from unlearning.base import BaseUnlearner
from unlearning.unlearn_utils import l2_penalty


class FinetuneUnlearner(BaseUnlearner):
    """
    A class for machine unlearning through fine-tuning on retain data only.

    This method creates a copy of the original model and fine-tunes it
    using only the retain dataset, effectively causing the model to "forget"
    the forget dataset by not reinforcing those patterns during retraining.

    This method is computationally efficient but may not provide strong
    forgetting guarantees for models that may have already memorized the
    forget data.
    """

    def __init__(
        self,
        device,
        evaluate: bool = False,
        wandb_enabled: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize the FinetuneUnlearner class.

        Args:
            device: Computing device (CPU/GPU) to use for computations.
                   If None, will be automatically determined.
        """
        super().__init__(device, evaluate, wandb_enabled, verbose)

    def unlearn(self, model: nn.Module, data_dict: Dict[str, DataLoader], **kwargs):
        """
        Unlearn by fine-tuning the model on retain data only.

        Args:
            model: The original model to perform unlearning on.
            data_dict: Dictionary of dataloaders, must include a 'retain' key
                      with the data to retain. Other keys (e.g., 'forget',
                      'test') will be used for evaluation if self.evaluate is
                      True.
            **kwargs: Additional arguments including:
                - loss_fn: Loss function to use for training.
                - num_epochs: Number of training epochs (default: 1).
                - lr: Learning rate (default: 1e-2).
                - weight_decay: Weight decay parameter (default: 0).
                - use_l2_penalty: Whether to add L2 regularization
                    (default: False).

        Returns:
            Tuple of (unlearned_model, self.logs)
        """
        if "retain" not in data_dict.keys():
            raise ValueError("'retain' data must be in data_dict.")

        model.to(self.device)
        unlearned_model, scheduler = self._setup_unlearning(model, data_dict, **kwargs)

        # Main training loop
        for e in range(self.epochs):
            epoch_start_time = time.time()
            for retain_inputs, retain_labels in data_dict["retain"]:
                unlearned_model.eval()
                self.optimizer.zero_grad()

                retain_inputs = retain_inputs.to(self.device)
                retain_labels = retain_labels.to(self.device)

                retain_output = unlearned_model(retain_inputs)
                retain_loss = self.criterion(retain_output, retain_labels)

                # Add L2 penalty for bias towards weight similarity to original
                if self.use_l2_penalty:
                    l2_loss = l2_penalty(
                        model=unlearned_model,
                        model_init=model,
                        weight_decay=self.weight_decay,
                    )
                    retain_loss += l2_loss

                retain_loss.backward()
                self.optimizer.step()
            forward_pass_elapsed = time.time() - epoch_start_time
            if self.wandb_enabled:
                self._log_forward_pass_time_in_wandb(
                    epoch=e + 1, time=forward_pass_elapsed
                )
            if self.verbose:
                self._print_forward_pass_metrics(e, forward_pass_elapsed)
            if self.evaluate:
                self._evaluate_all_splits(
                    model=unlearned_model,
                    data_dict=data_dict,
                    epoch=e + 1,
                )

            if scheduler is not None:
                scheduler.step()

        return unlearned_model, self.logs
