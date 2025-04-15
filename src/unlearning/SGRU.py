import os
import torch
import torch.nn as nn
import copy
from unlearning.utils import (l2_penalty)
from itertools import cycle
from unlearning.base import BaseUnlearner
from typing import Dict
from torch.utils.data import DataLoader

class SGRU(BaseUnlearner):
    """
    Subspace Gradient Redirection Unlearning (SGRU) for machine unlearning.
    
    SGRU identifies principal gradient directions associated with the forget dataset
    and redirects gradients from the retain dataset away from these directions during
    fine-tuning. By projecting gradients orthogonally to the forget subspace, the model
    learns to preserve knowledge from retain data while systematically unlearning 
    patterns specific to the forget data.
    
    This method:
    1. Identifies the principal components of gradients on forget data using SVD
    2. Projects retain data gradients away from these directions during training
    3. Periodically recalculates the forget subspace for optimal unlearning
    """
    def __init__(self, 
                device,
                evaluate: bool = False,
                ):
        """
        Initialize the SGRU class.

        Args:
            device: Computing device (CPU/GPU) to use for computations.
                   If None, will be automatically determined.
            evaluate: Whether to track and return evaluation metrics during unlearning.
        """
        super().__init__(device, evaluate)


    def unlearn(self,
                model: nn.Module,
                data_dict: Dict[str, DataLoader],
                verbose: bool = True,
                **kwargs):
        """
        Unlearn through Subspace Gradient Redirection on retain data.

        Args:
            model: The original model to perform unlearning on.
            data_dict: Dictionary of dataloaders, must include both 'retain' and 'forget' keys
                    with the respective datasets. Other keys (e.g., 'test') will
                    be used for evaluation if self.evaluate is True.
            verbose: Whether to print progress during unlearning.
            save_checkpoint: Whether to save model checkpoints during unlearning.
            save_freq: Frequency (in epochs) for saving checkpoints.
            save_path: Directory to save checkpoints.
            **kwargs: Additional arguments including:
                - loss_fn: Loss function to use for training.
                - num_epochs: Number of training epochs (default: 1).
                - lr: Learning rate (default: 1e-2).
                - weight_decay: Weight decay parameter (default: 0).
                - use_l2_penalty: Whether to add L2 regularization (default: False).
                - recalc_freq: Frequency to recalculate forget subspace (default: 1).
                - num_components: Number of principal components to use (default: 10).
                - redirection_strength: Lambda value for gradient deflection (default: 1.0).
                - max_grad_norm: Maximum gradient norm for clipping (default: 1.0).

        Returns:
            If self.evaluate is True:
                Tuple of (unlearned_model, losses_dict) where losses_dict contains
                tracked losses for each dataset type.
            Otherwise:
                The unlearned model.

        Raises:
            ValueError: If 'retain' or 'forget' data is not in data_dict.
        """
        model.to(self.device)
        unlearned_model = copy.deepcopy(model)

        # Validate and extract common hyperparameters
        self.valid_args(**kwargs)
        self.extract_hyperparameters(**kwargs)

        # Extract gradient deflection specific parameters
        recalc_freq = kwargs.get('recalc_freq', 1)  # Recalculate forget subspace every N epochs
        num_components = kwargs.get('num_components', 10)  # Number of principal components to use
        redirection_strength = kwargs.get('redirection_strength', 1.0)  # Lambda value for active deflection
        max_grad_norm = kwargs.get('max_grad_norm', 1.0)  # For gradient clipping

        # Ensure required data is available
        if 'retain' not in data_dict.keys() or 'forget' not in data_dict.keys():
            raise ValueError("'retain' and 'forget' data must be in data_dict.")

        # Initialize loss tracking
        losses = {f"{data_type}_losses": [] for data_type in data_dict.keys()}

        optimizer = torch.optim.SGD(params=unlearned_model.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)

        eval_only_data = [key for key in data_dict.keys() if key != 'retain']

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
            # Recalculate forget subspace periodically
            if e % recalc_freq == 0:
                unlearned_model.eval() 
                for group_name, params in param_groups.items():
                    # Collect gradients for this parameter group across batches
                    group_grads = []
                    
                    for forget_inputs, forget_labels in data_dict['forget']:
                        forget_inputs = forget_inputs.to(self.device)
                        forget_labels = forget_labels.to(self.device)
                        
                        optimizer.zero_grad()
                        forget_output = unlearned_model(forget_inputs)
                        forget_loss = self.criterion(forget_output,
                                                     forget_labels)
                        
                        # Check for NaN loss
                        if torch.isnan(forget_loss).any():
                            if verbose:
                                print(f"Warning: NaN loss detected in forget data during SVD calculation at epoch {e}")
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
                    
                    # Compute principal components if we have enough gradient data
                    if len(group_grads) > 0:
                        grads_matrix = torch.stack(group_grads)
                        
                        # Check for NaN in gradient matrix
                        if torch.isnan(grads_matrix).any():
                            print(f"Warning: NaN detected in gradient matrix for {group_name}")
                            continue
                        
                        k = min(num_components, grads_matrix.size(0), grads_matrix.size(1))
                        
                        if k > 0:
                            try:
                                # Add a small value to diagonal for numerical stability
                                epsilon = 1e-8
                                reg_matrix = grads_matrix + torch.eye(
                                    grads_matrix.size(0), 
                                    device=grads_matrix.device
                                ) * epsilon if grads_matrix.size(0) == grads_matrix.size(1) else grads_matrix
                                
                                U, S, V = torch.svd(reg_matrix)
                                
                                # Normalize the directions for stability
                                directions = V[:, :k]
                                norms = torch.norm(directions, dim=0, keepdim=True)
                                # Avoid division by zero
                                norms = torch.clamp(norms, min=1e-8)
                                normalized_directions = directions / norms
                                
                                forget_directions[group_name] = normalized_directions
                                                                    
                            except Exception as e:
                                print(f"SVD failed for {group_name}: {e}")
                                # Fallback - use random orthogonal directions for randomisation
                                random_dirs = torch.randn(grads_matrix.size(1), k, device=self.device)
                                forget_directions[group_name], _ = torch.linalg.qr(random_dirs)
        
            # Train on retain data
            unlearned_model.train()
            total_retain_loss = 0
            
            for retain_inputs, retain_labels in data_dict['retain']:
                optimizer.zero_grad()

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
                            param_shapes = [] # Ror reshaping later
                            for _, param in params:
                                if param.grad is not None:
                                    group_grads.append(param.grad.view(-1))
                                    param_shapes.append(param.grad.shape)
                                
                            flat_grad = torch.cat(group_grads)
                            for direction in directions.t():
                                proj = torch.dot(flat_grad, direction)
                                max_proj = 10.0  # Arbitrary threshold to prevent extreme values
                                proj = torch.clamp(proj, -max_proj, max_proj)
                                # Apply the deflection with strength
                                flat_grad = flat_grad - proj * direction * redirection_strength
                            
                            # DEBUGGING
                            if torch.isnan(flat_grad).any():
                                print(f"Warning: NaN detected in deflected gradient for {group_name}")
                                continue
                            
                            # Map deflected gradient back to the parameter
                            start_idx = 0
                            for i, (_, param) in enumerate(params):
                                if param.grad is not None and i < len(param_shapes):
                                    num_params = param.numel()
                                    param_grad = flat_grad[start_idx:start_idx + num_params]
                                    # Reshape and assign
                                    param.grad = param_grad.view_as(param.grad)
                                    start_idx += num_params

                    # Apply gradient clipping to all parameters
                    torch.nn.utils.clip_grad_norm_(unlearned_model.parameters(), max_grad_norm)

                optimizer.step()

            if verbose:
                print(f'Epoch {e}: Retain Loss: {total_retain_loss/len(data_dict["retain"])}')

            if self.evaluate:
                # Calculate average retain loss for this epoch
                losses['retain_losses'].append(total_retain_loss/len(data_dict["retain"]))
                unlearned_model.eval()
                # Evaluate model on other datasets
                for data_type in eval_only_data:
                    loader_loss = self._evaluate(unlearned_model,
                                                 data_dict[data_type],
                                                 self.criterion).mean()
                    losses[f"{data_type}_losses"].append(loader_loss.item())

                    if verbose:
                        print(f'{data_type.capitalize()} Loss: {loader_loss}',
                              end='  ')
                if verbose:
                    print()

        return unlearned_model, losses
