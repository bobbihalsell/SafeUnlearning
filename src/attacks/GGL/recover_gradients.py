import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.fx as fx
from torch.fx import GraphModule
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
                                                    
import os
import copy
import re
import math
from tqdm import tqdm
from collections import OrderedDict
from typing import Optional, Union, List, Dict, Any

def find_subspace(original_model, unlearned_model, background_data, device, 
                  loss_fn= nn.CrossEntropyLoss(), variance_threshold=0.9, cache_dir=None):
    # Get gradients from both models on background data
    orig_grads = get_model_gradients(original_model, background_data, device, loss_fn)
    unl_grads = get_model_gradients(unlearned_model, background_data, device, loss_fn)
    
    if not orig_grads or not unl_grads:
        print("Failed to compute gradients")
        return None
    
    # Calculate gradient differences
    grad_differences = {}
    total_diff_norm = 0
    
    for param_name in orig_grads:
        if param_name in unl_grads and '.weight' in param_name:
            diff = orig_grads[param_name] - unl_grads[param_name]
            layer_name = param_name.replace('.weight', '')
            grad_differences[layer_name] = diff
            total_diff_norm += torch.norm(diff).item()
    
    print(f"Total gradient difference magnitude: {total_diff_norm:.6f}")
    print(f"Gradient differences from {len(grad_differences)} layers")
    
    # Convert gradients to covariance matrices and extract subspaces
    subspace_signature = {}
    
    for layer_name, grad_diff in grad_differences.items():
        try:
            # Move to CPU and flatten
            grad_diff_cpu = grad_diff.detach().cpu().float()
            grad_flat = grad_diff_cpu.flatten()
            num_params = grad_flat.shape[0]
            
            # Memory-efficient covariance computation
            if num_params > 1024:  # For large layers, use approximation
                # Random projection to reduce dimensionality
                target_dim = min(256, num_params // 8)
                torch.manual_seed(42)  # For reproducibility
                proj_matrix = torch.randn(target_dim, num_params) / np.sqrt(target_dim)
                # Project gradient to lower dimension
                grad_projected = torch.mv(proj_matrix, grad_flat)
                # Create covariance in projected space
                grad_cov = torch.outer(grad_projected, grad_projected)
            else:
                # Create full covariance matrix
                grad_cov = torch.outer(grad_flat, grad_flat)
            
            # Add small epsilon for numerical stability
            grad_cov = grad_cov + 1e-6 * torch.eye(grad_cov.shape[0], dtype=grad_cov.dtype)
            
            # SVD decomposition
            U, S, V = torch.svd(grad_cov)
            
            # Apply variance threshold
            if len(S) > 1:
                cumulative_var = torch.cumsum(S, dim=0) / torch.sum(S)
                components_to_keep = torch.sum(cumulative_var <= variance_threshold).item()
                components_to_keep = max(1, min(components_to_keep + 1, len(S)))
            else:
                components_to_keep = min(5, len(S))
            
            # Truncate to keep most important components
            U_truncated = U[:, :components_to_keep]
            S_truncated = S[:components_to_keep]
            
            subspace_signature[layer_name] = {
                'basis': U_truncated,
                'singular_values': S_truncated,
                'explained_variance_ratio': (S_truncated.sum() / S.sum()).item(),
                'num_components': components_to_keep,
                'total_components': len(S),
                'layer_name': layer_name
            }
            
        except Exception as e:
            print(f"  Subspace extraction failed for {layer_name}: {e}")
            continue

        if cache_dir:
            # Save the subspace signature to cache
            cache_path = os.path.join(cache_dir, f"subspace.pt")
            torch.save(subspace_signature, cache_path)
    return subspace_signature
    

def get_model_gradients(model, data_loader, device, loss_fn=nn.CrossEntropyLoss(), cache_dir=None, label=''):
    """Get gradients for a model on given data."""

    original_mode = model.training
    model.train()
    
    accumulated_gradients = {}
    num_batches = 0
    
    try:
        for (images, labels) in data_loader:
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # Accumulate gradients (only weights)
            for name, param in model.named_parameters():
                if param.grad is not None and '.weight' in name:
                    if name not in accumulated_gradients:
                        accumulated_gradients[name] = param.grad.detach().clone()
                    else:
                        accumulated_gradients[name] += param.grad.detach().clone()
            
            num_batches += 1
            del images, labels, outputs, loss
            torch.cuda.empty_cache()
    finally:
        model.train(original_mode)
    # Average by number of batches
    for name in accumulated_gradients:
        accumulated_gradients[name] = accumulated_gradients[name] / num_batches

    if cache_dir:
            # Save the subspace signature to cache
            cache_path = os.path.join(cache_dir, f"{label}_gradient.pt")
            torch.save(accumulated_gradients, cache_path)
    
    return accumulated_gradients

def project_onto_subspace(target, subspace):
    projections = {}

    for layer_name in subspace:
        if layer_name not in target:
            print(f"Layer {layer_name} not found in target, skipping projection.")
            continue
            
        try:
            subspace_basis = subspace[layer_name]['basis']
            
            # Flatten and move to CPU
            target_flat = target[layer_name].detach().cpu().flatten().float()
            
            # Handle dimension mismatch (from random projection) IS THERE A BETTER WAY??
            if target_flat.shape[0] != subspace_basis.shape[0]:
                print(f"Dimension mismatch for {layer_name}: target {target_flat.shape[0]}, subspace {subspace_basis.shape[0]}.")
                # Apply same random projection as used in estimation
                torch.manual_seed(42)  # Same seed as estimation
                if target_flat.shape[0] > subspace_basis.shape[0]:
                    proj_matrix = torch.randn(subspace_basis.shape[0], target_flat.shape[0]) / np.sqrt(subspace_basis.shape[0])
                    target_flat = torch.mv(proj_matrix, target_flat)
                else:
                    # Skip if dimensions don't match and we can't project
                    continue
            
            # Project onto subspace (sum of absolute projections onto all basis vectors)
            projection = torch.sum(torch.abs(subspace_basis.T @ target_flat)).item()
            
            projections[layer_name] = projection
            
        except Exception as e:
            print(f"Subspace projection failed for {layer_name}: {e}")
            continue
    
    return projections

def compute_parameter_difference(original_model, unlearned_model, learning_rate, cache_dir=None):
    """
    Compute the parameter difference (theta_u - theta_o) / learning_rate between unlearned and original models.
    
    Args:
        original_model: The original trained model
        unlearned_model: The unlearned model 
        learning_rate: Learning rate used during unlearning
        device: Device to perform computations on
        cache_dir: Optional directory to cache results
    
    Returns:
        Dictionary containing parameter differences for each layer
    """
    
    # Get model parameters in order
    original_params = OrderedDict()
    unlearned_params = OrderedDict()
    
    # Collect parameters from both models
    for (orig_name, orig_param), (unl_name, unl_param) in zip(
        original_model.named_parameters(), unlearned_model.named_parameters()
    ):
        if orig_name != unl_name:
            raise ValueError(f"Parameter name mismatch: {orig_name} vs {unl_name}")
        
        # Only include weight parameters (skip biases if desired)
        if '.weight' in orig_name:
            layer_name = orig_name.replace('.weight', '')
            original_params[layer_name] = orig_param.detach().cpu()
            unlearned_params[layer_name] = unl_param.detach().cpu()
    
    # Compute parameter differences
    param_differences = {}
    for layer_name in original_params:
        if layer_name in unlearned_params:
            # Compute (theta_u - theta_o) / lr
            diff = (unlearned_params[layer_name] - original_params[layer_name]) / learning_rate
            param_differences[layer_name] = diff
            
            # Print some statistics
            diff_norm = torch.norm(diff).item()
            print(f"Layer {layer_name}: parameter difference norm = {diff_norm:.6f}")
    
    print(f"Computed parameter differences for {len(param_differences)} layers")
    
    # Cache results if requested
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "parameter_differences.pt")
        torch.save(param_differences, cache_path)
        print(f"Cached parameter differences to {cache_path}")
    
    return param_differences

def get_feature_dict(model):
    """
    Automatically generate conv_fea_dict and linear_fea_dict for a given model.
    """
    _, eval_nodes = get_graph_node_names(model)
    conv_fea_dict = OrderedDict()
    linear_fea_dict = OrderedDict()
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name in eval_nodes:
            conv_fea_dict[name] = name
        elif isinstance(module, torch.nn.Linear) and name in eval_nodes:
            linear_fea_dict[name] = name
    return conv_fea_dict, linear_fea_dict

def get_module_by_name(model, layer_name):
    """
    Get a module from a model by its name (supports nested access).
    """
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

