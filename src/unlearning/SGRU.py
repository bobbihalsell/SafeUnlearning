import torch
import torch.nn as nn
from unlearning.unlearn_utils import l2_penalty
from unlearning.base import BaseUnlearner
from typing import Dict
from torch.utils.data import DataLoader
import time


class SGRU(BaseUnlearner):
    """
    Subspace Gradient Redirection Unlearning (SGRU) for machine unlearning.

    SGRU identifies principal gradient directions associated with the forget
    dataset and redirects gradients from the retain dataset away from these
    directions during fine-tuning. By projecting gradients orthogonally to the
    forget subspace, the model learns to preserve knowledge from retain data
    while systematically unlearning  patterns specific to the forget data.

    This method:
    1. Identifies the principal components of gradients on forget data using
        SVD
    2. Projects retain data gradients away from these directions during
        training
    3. Periodically recalculates the forget subspace for optimal unlearning
    """
    def __init__(self,
                 device,
                 evaluate: bool = False,
                 wandb_enabled: bool = False,
                 verbose: bool = True
                 ):
        """
        Initialize the SGRU class.

        Args:
            device: Computing device (CPU/GPU) to use for computations.
                   If None, will be automatically determined.
            evaluate: Whether to track and return evaluation metrics during
                unlearning.
        """
        super().__init__(device, evaluate, wandb_enabled, verbose)
        self.criterion = nn.CrossEntropyLoss()

    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                **kwargs):
        """
        Unlearn through Subspace Gradient Redirection on retain data.

        Args:
            model: The original model to perform unlearning on.
            data_dict: Dictionary of dataloaders, must include both 'retain'
                    and 'forget' keys with the respective datasets. Other keys
                    (e.g., 'test') will
                    be used for evaluation if self.evaluate is True.
            **kwargs: Additional arguments including:
                - loss_fn: Loss function to use for training.
                - num_epochs: Number of training epochs (default: 1).
                - lr: Learning rate (default: 1e-2).
                - weight_decay: Weight decay parameter (default: 0).
                - use_l2_penalty: Whether to add L2 regularization
                    (default: False).
                - recalc_freq: Frequency to recalculate forget subspace
                - num_components: Number of principal components to use
                - redirection_strength: Lambda value for gradient deflection
                - retain_strength: Value for gradient retention
                - max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Tuple of (unlearned_model, self.logs)
        """
        model.to(self.device)
        unlearned_model, scheduler = self._setup_unlearning(
            model,
            data_dict,
            **kwargs)

        if self.recalc_freq < 0:
            raise ValueError("recalc_freq must be non-negative.")
        if not isinstance(self.recalc_freq, int):
            raise TypeError("recalc_freq must an integer.")
        if self.num_components < 0:
            raise ValueError("num_components must be non-negative.")
        if not isinstance(self.num_components, int):
            raise TypeError("num_components must an integer.")
        if self.redirection_strength < 0:
            raise ValueError("redirection_strength must be non-negative.")
        if self.retain_strencth < 0:
            raise ValueError("retain_strength must be non-negative.")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative.")

        # Ensure required data is available
        if ('retain' not in data_dict.keys()
           or 'forget' not in data_dict.keys()):
            raise ValueError(
                "'retain' and 'forget' data must be in data_dict."
                )

        # Group parameters by layers/modules for efficiency
        param_groups = {}
        for name, param in unlearned_model.named_parameters():
            if param.requires_grad:
                # Extract module name (e.g., 'layer1.0.conv1' -> 'layer1')
                module_name = name.split('.')[0]
                if module_name not in param_groups:
                    param_groups[module_name] = []
                param_groups[module_name].append((name, param))

        forget_directions = {}

        # Main training loop
        for e in range(self.epochs):
            epoch_start_time = time.time()
            # Recalculate forget subspace periodically
            if e % self.recalc_freq == 0:
                unlearned_model.eval()
                for group_name, params in param_groups.items():
                    # Collect gradients for this parameter group across batches
                    group_grads = []

                    for forget_inputs, forget_labels in data_dict['forget']:
                        forget_inputs = forget_inputs.to(self.device)
                        forget_labels = forget_labels.to(self.device)

                        self.optimizer.zero_grad()
                        forget_output = unlearned_model(forget_inputs)
                        forget_loss = self.criterion(forget_output,
                                                     forget_labels)

                        # Check for NaN loss
                        if torch.isnan(forget_loss).any():
                            if self.verbose:
                                print(
                                    f"Warning: NaN loss detected in forget "
                                    f"data during SVD calculation at epoch {e}"
                                    )
                            continue

                        forget_loss.backward()

                        # Collect gradients for this parameter group
                        batch_grads = []
                        for _, param in params:
                            if param.grad is not None:
                                # Skip if gradients are NaN
                                if torch.isnan(param.grad).any():
                                    continue
                                batch_grads.append(param.grad.clone().view(-1))

                        if batch_grads:
                            group_grads.append(torch.cat(batch_grads))

                        grad_matrix = torch.stack(group_grads)

                        # Check for NaN in gradient matrix
                        if torch.isnan(grad_matrix).any():
                            print(f"Warning: NaN detected in gradient matrix "
                                  f"for {group_name}")
                            continue

                        k = min(self.num_components,
                                grad_matrix.size(0),
                                grad_matrix.size(1)
                                )

                        if k > 0:
                            try:
                                if grad_matrix.size(0) == grad_matrix.size(1):
                                    epsilon = 1e-8
                                    reg_matrix = grad_matrix + torch.eye(
                                                    grad_matrix.size(0),
                                                    device=grad_matrix.device
                                                    ) * epsilon
                                else:
                                    reg_matrix = grad_matrix

                                U, S, V = torch.svd(reg_matrix)
                                # Normalize the directions for stability
                                directions = V[:, :k]
                                norms = torch.norm(directions,
                                                   dim=0,
                                                   keepdim=True
                                                   )
                                # Avoid division by zero
                                norms = torch.clamp(norms, min=1e-8)
                                norm_directions = directions / norms

                                forget_directions[group_name] = norm_directions

                            except Exception as e:
                                print(f"SVD failed for {group_name}: {e}")
                                # Fallback - use random orthogonal directions
                                # for randomisation adding noise for fogetting
                                # Q has orthonormal columns
                                # R is an upper triangular matrix
                                random_dirs = torch.randn(
                                    grad_matrix.size(1),
                                    k,
                                    device=self.device
                                )
                                qr_result = torch.linalg.qr(random_dirs)
                                forget_directions[group_name], _ = qr_result

            # Train on retain data
            unlearned_model.train()
            total_retain_loss = 0

            for retain_inputs, retain_labels in data_dict['retain']:
                self.optimizer.zero_grad()

                retain_inputs = retain_inputs.to(self.device)
                retain_labels = retain_labels.to(self.device)

                retain_output = unlearned_model(retain_inputs)
                retain_loss = self.criterion(retain_output, retain_labels)

                total_retain_loss += retain_loss.item()

                if self.use_l2_penalty:
                    l2_loss = l2_penalty(model=unlearned_model,
                                         model_init=model,
                                         weight_decay=self.weight_decay)
                    retain_loss += l2_loss

                retain_loss.backward()

                # Apply gradient deflection
                with torch.no_grad():
                    for group_name, params in param_groups.items():
                        if group_name in forget_directions:
                            directions = forget_directions[group_name]

                            # Collect current gradients for this group
                            group_grads = []
                            param_shapes = []  # Ror reshaping later
                            for _, param in params:
                                if param.grad is not None:
                                    group_grads.append(param.grad.view(-1))
                                    param_shapes.append(param.grad.shape)

                            flat_grad = torch.cat(group_grads)
                            for direction in directions.t():
                                proj = torch.dot(flat_grad, direction)
                                # Arbitrary threshold to prevent extreme values
                                max_proj = 10.0
                                proj = torch.clamp(proj, -max_proj, max_proj)
                                # Apply the deflection with strength
                                flat_grad = (flat_grad * self.retain_strength
                                             - (proj * direction 
                                                * self.redirection_strength))

                            # DEBUGGING
                            if torch.isnan(flat_grad).any():
                                print(f"Warning: NaN detected in deflected "
                                      f"gradient for {group_name}")
                                continue

                            # Map deflected gradient back to the parameter
                            start_idx = 0
                            for i, (_, param) in enumerate(params):
                                if (param.grad is not None
                                   and i < len(param_shapes)):
                                    num_params = param.numel()
                                    end_idx = start_idx + num_params
                                    param_grad = flat_grad[start_idx:end_idx]
                                    # Reshape and assign
                                    param.grad = param_grad.view_as(param.grad)
                                    start_idx += num_params

                    # Apply gradient clipping to all parameters
                    torch.nn.utils.clip_grad_norm_(
                        unlearned_model.parameters(),
                        self.max_grad_norm
                        )

                self.optimizer.step()

            forward_pass_elapsed = time.time() - epoch_start_time
            if self.wandb_enabled:
                self._log_forward_pass_time_in_wandb(
                    epoch=e+1,
                    time=forward_pass_elapsed
                    )
            if self.verbose:
                self._print_forward_pass_metrics(e, forward_pass_elapsed)
            if self.evaluate:
                self._evaluate_all_splits(
                    model=unlearned_model,
                    data_dict=data_dict,
                    epoch=e+1,
                )

            if scheduler is not None:
                scheduler.step()

        return unlearned_model, self.logs
