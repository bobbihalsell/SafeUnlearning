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
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


# def calculate_gradient_covariance(model, data_loader, device, loss_fn=None, epochs=1):
#     """
#     Calculate covariance matrices per parameter (matching unlearning approach).
#     Each parameter gets its own covariance matrix with proper reshaping.
#     """
#     if loss_fn is None:
#         loss_fn = torch.nn.CrossEntropyLoss()
    
#     # Collect per-sample gradients for each parameter
#     all_gradients = {}  # param_name -> list of gradient tensors (properly reshaped)
    
#     model.train()
    
#     for epoch in range(epochs):
#         for batch_idx, (imgs, labels) in enumerate(tqdm(data_loader, desc=f"Computing gradient covariance (epoch {epoch+1}/{epochs})")):
#             imgs, labels = imgs.to(device), labels.to(device)
            
#             # Process each sample individually
#             for i in range(imgs.size(0)):
#                 model.zero_grad()
                
#                 # Forward pass for single sample
#                 output = model(imgs[i:i+1])
#                 loss = loss_fn(output, labels[i:i+1])
#                 loss.backward()
                
#                 # Collect gradients for each parameter (with proper reshaping)
#                 for name, param in model.named_parameters():
#                     if param.grad is None:
#                         continue
                    
#                     if name not in all_gradients:
#                         all_gradients[name] = []
                    
#                     grad_tensor = param.grad.clone()
                    
#                     # Reshape according to unlearning approach
#                     if len(grad_tensor.shape) == 4:  # Conv weights [out_ch, in_ch, h, w]
#                         # Reshape to [out_ch, in_ch*h*w] 
#                         sz = grad_tensor.shape[0]  # output channels
#                         grad_reshaped = grad_tensor.view(sz, -1)  # [out_ch, in_ch*h*w]
#                         all_gradients[name].append(grad_reshaped)
                        
#                     elif len(grad_tensor.shape) == 2:  # Linear weights [out_features, in_features]
#                         # Keep as is: [out_features, in_features]
#                         all_gradients[name].append(grad_tensor)
                        
#                     elif len(grad_tensor.shape) == 1:  # Bias [features]
#                         # Reshape to [features, 1] to match matrix operations
#                         grad_reshaped = grad_tensor.unsqueeze(-1)  # [features, 1]
#                         all_gradients[name].append(grad_reshaped)
    
#     # Calculate covariance matrices per parameter
#     gradient_covar = {}
#     sample_counts = {}
    
#     for param_name, grad_list in all_gradients.items():
#         if grad_list:
#             # Stack into [num_samples, ...] then compute covariance on the last dimension
#             grad_stack = torch.stack(grad_list)  # [num_samples, sz, features]
#             num_samples = grad_stack.shape[0]
            
#             # For each output channel/feature, compute covariance across input dimensions
#             if len(grad_stack.shape) == 3:  # [num_samples, sz, features]
#                 sz, features = grad_stack.shape[1], grad_stack.shape[2]
#                 # Reshape to [num_samples * sz, features]
#                 grad_matrix = grad_stack.view(-1, features)
#                 # Compute covariance: features x features
#                 gradient_covar[param_name] = grad_matrix.T @ grad_matrix
#                 sample_counts[param_name] = grad_matrix.shape[0]
#             else:
#                 # Handle other cases
#                 grad_matrix = grad_stack.view(num_samples, -1)
#                 gradient_covar[param_name] = grad_matrix.T @ grad_matrix  
#                 sample_counts[param_name] = num_samples
#         else:
#             gradient_covar[param_name] = None
#             sample_counts[param_name] = 0
    
#     return gradient_covar, sample_counts

def calculate_gradient_covariance(model, data_loader, device, loss_fn=None, epochs=1):
    """
    Calculate covariance matrices per parameter (matching unlearning approach).
    Each parameter gets its own covariance matrix with proper reshaping.
    """
    print(f"\n=== DEBUG calculate_gradient_covariance ===")
    
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    # Collect per-sample gradients for each parameter
    all_gradients = {}  # param_name -> list of gradient tensors (properly reshaped)
    
    model.train()
    
    # Check what parameters we're working with
    param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"Model parameters: {len(param_names)} total")
    print(f"First few parameters: {param_names[:5]}")
    
    for epoch in range(epochs):
        for batch_idx, (imgs, labels) in enumerate(tqdm(data_loader, desc=f"Computing gradient covariance (epoch {epoch+1}/{epochs})")):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Process each sample individually
            for i in range(imgs.size(0)):
                model.zero_grad()
                
                # Forward pass for single sample
                output = model(imgs[i:i+1])
                loss = loss_fn(output, labels[i:i+1])
                loss.backward()
                
                # Collect gradients for each parameter (with proper reshaping)
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    
                    if name not in all_gradients:
                        all_gradients[name] = []
                    
                    grad_tensor = param.grad.clone()
                    
                    # Reshape according to unlearning approach
                    if len(grad_tensor.shape) == 4:  # Conv weights [out_ch, in_ch, h, w]
                        # Reshape to [out_ch, in_ch*h*w] 
                        sz = grad_tensor.shape[0]  # output channels
                        grad_reshaped = grad_tensor.view(sz, -1)  # [out_ch, in_ch*h*w]
                        all_gradients[name].append(grad_reshaped)
                        
                    elif len(grad_tensor.shape) == 2:  # Linear weights [out_features, in_features]
                        # Keep as is: [out_features, in_features]
                        all_gradients[name].append(grad_tensor)
                        
                    elif len(grad_tensor.shape) == 1:  # Bias [features]
                        # Reshape to [features, 1] to match matrix operations
                        grad_reshaped = grad_tensor.unsqueeze(-1)  # [features, 1]
                        all_gradients[name].append(grad_reshaped)
            
            # Only process first batch for faster debugging
            if batch_idx == 0:
                break
    
    print(f"Collected gradients for {len(all_gradients)} parameters")
    print(f"First few gradient keys: {list(all_gradients.keys())[:5]}")
    
    # Calculate covariance matrices per parameter
    gradient_covar = {}
    sample_counts = {}
    
    for param_name, grad_list in all_gradients.items():
        if grad_list:
            # Stack into [num_samples, ...] then compute covariance on the last dimension
            grad_stack = torch.stack(grad_list)  # [num_samples, sz, features]
            num_samples = grad_stack.shape[0]
            
            # For each output channel/feature, compute covariance across input dimensions
            if len(grad_stack.shape) == 3:  # [num_samples, sz, features]
                sz, features = grad_stack.shape[1], grad_stack.shape[2]
                # Reshape to [num_samples * sz, features]
                grad_matrix = grad_stack.view(-1, features)
                # Compute covariance: features x features
                gradient_covar[param_name] = grad_matrix.T @ grad_matrix
                sample_counts[param_name] = grad_matrix.shape[0]
            else:
                # Handle other cases
                grad_matrix = grad_stack.view(num_samples, -1)
                gradient_covar[param_name] = grad_matrix.T @ grad_matrix  
                sample_counts[param_name] = num_samples
        else:
            gradient_covar[param_name] = None
            sample_counts[param_name] = 0
    
    print(f"Covariance matrices computed for {len(gradient_covar)} parameters")
    print(f"First few covariance keys: {list(gradient_covar.keys())[:5]}")
    
    return gradient_covar, sample_counts


def get_gradients(model, data_loader, device, loss_fn=None, epochs=1):
    """
    Get accumulated gradients per parameter (matching unlearning approach).
    Returns gradients with same reshaping as covariance computation.
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    gradients = {}
    
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (imgs, labels) in enumerate(tqdm(data_loader, desc=f"Computing gradients (epoch {epoch+1}/{epochs})")):
            imgs, labels = imgs.to(device), labels.to(device)
            
            model.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # Accumulate gradients with proper reshaping
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                
                if name not in gradients:
                    # Initialize with proper shape
                    grad_tensor = param.grad.clone()
                    if len(grad_tensor.shape) == 4:  # Conv weights
                        sz = grad_tensor.shape[0]
                        gradients[name] = torch.zeros(sz, grad_tensor.numel() // sz, device=device)
                    elif len(grad_tensor.shape) == 2:  # Linear weights  
                        gradients[name] = torch.zeros_like(grad_tensor)
                    elif len(grad_tensor.shape) == 1:  # Bias
                        gradients[name] = torch.zeros(grad_tensor.shape[0], 1, device=device)
                
                # Add gradients with proper reshaping
                grad_tensor = param.grad.clone()
                if len(grad_tensor.shape) == 4:  # Conv weights
                    sz = grad_tensor.shape[0]
                    grad_reshaped = grad_tensor.view(sz, -1)
                    gradients[name] += grad_reshaped
                elif len(grad_tensor.shape) == 2:  # Linear weights
                    gradients[name] += grad_tensor
                elif len(grad_tensor.shape) == 1:  # Bias
                    gradients[name] += grad_tensor.unsqueeze(-1)
    
    return gradients


# def project_gradients(target_gradients, svd_basis, k=None):
#     """
#     Project gradients parameter-wise (exactly like unlearning method).
#     """
#     projected_gradients = {}
    
#     for param_name in target_gradients:
#         if param_name not in svd_basis or svd_basis[param_name] is None:
#             continue
            
#         grad_tensor = target_gradients[param_name]  # Already properly shaped
#         U = svd_basis[param_name]['U']
        
#         if k is not None:
#             U = U[:, :k]
        
#         # Create projection matrix P = U @ U.T (same as unlearning)
#         P = U @ U.T
        
#         # Check dimension compatibility
#         if grad_tensor.shape[-1] != P.shape[0]:
#             print(f"Dimension mismatch for {param_name}: grad {grad_tensor.shape} vs proj {P.shape}")
#             continue
        
#         # Apply projection: grad - grad @ P (exactly like unlearning)
#         if len(grad_tensor.shape) == 2:  # [sz, features] or [out_features, in_features]
#             proj_grad = grad_tensor @ P
#             projected_gradients[param_name] = grad_tensor - proj_grad
#         else:
#             projected_gradients[param_name] = grad_tensor  # Skip if unexpected shape
    
#     return projected_gradients


# def calculate_svd(covar_dict, k=None, eps=1e-6):
#     """
#     Calculate SVD per parameter.
#     """
#     svd_results = {}
#     for param_name, C in covar_dict.items():
#         if C is None:
#             svd_results[param_name] = None
#             continue
            
#         # Add regularization for numerical stability
#         U, S, _ = torch.svd(C + eps * torch.eye(C.size(0), device=C.device))
#         if k is not None:
#             U = U[:, :k]
#             S = S[:k]
#         svd_results[param_name] = {'U': U, 'S': S}
    
#     return svd_results

def calculate_svd(covar_dict, k=None, eps=1e-6):
    """
    Calculate SVD per parameter.
    """
    print(f"\n=== DEBUG calculate_svd ===")
    print(f"Input covar_dict has {len(covar_dict)} parameters")
    print(f"First few input keys: {list(covar_dict.keys())[:5]}")
    
    svd_results = {}
    none_count = 0
    
    for param_name, C in covar_dict.items():
        if C is None:
            svd_results[param_name] = None
            none_count += 1
            continue
            
        # Add regularization for numerical stability
        U, S, _ = torch.svd(C + eps * torch.eye(C.size(0), device=C.device))
        if k is not None:
            U = U[:, :k]
            S = S[:k]
        svd_results[param_name] = {'U': U, 'S': S}
    
    print(f"SVD results: {len(svd_results)} parameters, {none_count} None values")
    print(f"First few SVD result keys: {list(svd_results.keys())[:5]}")
    
    return svd_results


# def calculate_svd_difference(covar_original, covar_unlearned, device, k=None):
#     """
#     Calculate the difference in representation spaces between original and unlearned models (parameter-wise).
#     """
#     diff_svd = {}
    
#     for param_name in covar_original:
#         if param_name not in covar_unlearned:
#             continue
            
#         C_orig = covar_original[param_name]
#         C_unlearn = covar_unlearned[param_name]
        
#         if C_orig is None or C_unlearn is None:
#             diff_svd[param_name] = None
#             continue
            
#         C_orig = C_orig.to(device)
#         C_unlearn = C_unlearn.to(device)

#         # Calculate matrix difference
#         matrix_diff = torch.norm(C_orig - C_unlearn).item()
        
#         U_orig, S_orig, _ = torch.svd(C_orig)
#         U_unlearn, S_unlearn, _ = torch.svd(C_unlearn)
        
#         # Reconstruct and subtract
#         M_orig = U_orig @ torch.diag(S_orig) @ U_orig.T
#         M_unlearn = U_unlearn @ torch.diag(S_unlearn) @ U_unlearn.T
#         diff = M_orig - M_unlearn
        
#         U, S, _ = torch.svd(diff)
#         pos_mask = S > 1e-8
#         U = U[:, pos_mask]
#         S = S[pos_mask]
        
#         if k is not None and k < U.shape[1]:
#             U = U[:, :k]
#             S = S[:k]
            
#         diff_svd[param_name] = {'U': U, 'S': S}
    
#     return diff_svd
def calculate_svd_difference(covar_original, covar_unlearned, device, k=None):
    """
    Calculate the difference in representation spaces between original and unlearned models (parameter-wise).
    """
    print(f"\n=== DEBUG calculate_svd_difference ===")
    print(f"covar_original: {len(covar_original)} parameters")
    print(f"covar_unlearned: {len(covar_unlearned)} parameters")
    print(f"Original keys: {list(covar_original.keys())[:5]}")
    print(f"Unlearned keys: {list(covar_unlearned.keys())[:5]}")
    
    diff_svd = {}
    
    for param_name in covar_original:
        if param_name not in covar_unlearned:
            print(f"Skipping {param_name}: not in unlearned covariance")
            continue
            
        C_orig = covar_original[param_name]
        C_unlearn = covar_unlearned[param_name]
        
        if C_orig is None or C_unlearn is None:
            diff_svd[param_name] = None
            continue
            
        C_orig = C_orig.to(device)
        C_unlearn = C_unlearn.to(device)

        # Calculate matrix difference
        matrix_diff = torch.norm(C_orig - C_unlearn).item()
        
        U_orig, S_orig, _ = torch.svd(C_orig)
        U_unlearn, S_unlearn, _ = torch.svd(C_unlearn)
        
        # Reconstruct and subtract
        M_orig = U_orig @ torch.diag(S_orig) @ U_orig.T
        M_unlearn = U_unlearn @ torch.diag(S_unlearn) @ U_unlearn.T
        diff = M_orig - M_unlearn
        
        U, S, _ = torch.svd(diff)
        pos_mask = S > 1e-8
        U = U[:, pos_mask]
        S = S[pos_mask]
        
        if k is not None and k < U.shape[1]:
            U = U[:, :k]
            S = S[:k]
            
        diff_svd[param_name] = {'U': U, 'S': S}
    
    print(f"Difference SVD results: {len(diff_svd)} parameters")
    print(f"First few diff SVD keys: {list(diff_svd.keys())[:5]}")
    
    return diff_svd

# Quick fix: Map between parameter names and layer names

def map_svd_to_gradient_keys(svd_dict, gradient_dict):
    """
    Map SVD results from layer names to parameter names to match gradients.
    
    SVD keys: ['conv1', 'layer1.0.conv1', ...]
    Gradient keys: ['conv1.weight', 'conv1.bias', 'layer1.0.conv1.weight', ...]
    """
    print(f"\n=== DEBUG map_svd_to_gradient_keys ===")
    print(f"SVD keys: {list(svd_dict.keys())[:5]}")
    print(f"Gradient keys: {list(gradient_dict.keys())[:5]}")
    
    mapped_svd = {}
    
    # For each gradient parameter, find matching SVD layer
    for grad_param_name in gradient_dict.keys():
        # Extract layer name from parameter name
        # 'conv1.weight' -> 'conv1'
        # 'layer1.0.conv1.weight' -> 'layer1.0.conv1'
        if '.weight' in grad_param_name:
            layer_name = grad_param_name.replace('.weight', '')
        elif '.bias' in grad_param_name:
            layer_name = grad_param_name.replace('.bias', '')
        else:
            layer_name = grad_param_name
        
        # If this layer has SVD results, map them to the parameter name
        if layer_name in svd_dict and svd_dict[layer_name] is not None:
            mapped_svd[grad_param_name] = svd_dict[layer_name]
            
    print(f"Mapped {len(mapped_svd)} SVD results to gradient parameter names")
    print(f"Mapped keys: {list(mapped_svd.keys())[:5]}")
    
    return mapped_svd

# Replace your functions with these debug versions:

def compare_projections(proj1, proj2, k=None, plot=True, id='comparison'):
    """
    Compare two sets of parameter-wise projections.
    """
    print(f"\n=== DEBUG compare_projections: {id} ===")
    print(f"proj1: {type(proj1)}, {len(proj1) if proj1 else 0} keys")
    print(f"proj2: {type(proj2)}, {len(proj2) if proj2 else 0} keys")
    
    if not proj1:
        print("ERROR: proj1 is empty!")
        return {}
    if not proj2:
        print("ERROR: proj2 is empty!")
        return {}
        
    print(f"proj1 keys: {list(proj1.keys())[:5]}...")
    print(f"proj2 keys: {list(proj2.keys())[:5]}...")
    
    results = {}
    common_params = set(proj1.keys()) & set(proj2.keys())
    print(f"Common parameters: {len(common_params)}")
    
    if len(common_params) == 0:
        print("ERROR: No common parameters between projections!")
        return {}
    
    processed = 0
    for param_name in common_params:
        P1 = proj1[param_name]
        P2 = proj2[param_name]
        
        if P1 is None:
            print(f"Skipping {param_name}: P1 is None")
            continue
        if P2 is None:
            print(f"Skipping {param_name}: P2 is None")
            continue
            
        if processed < 3:  # Debug first few
            print(f"Processing {param_name}: P1 {P1.shape}, P2 {P2.shape}")
        
        try:
            # Calculate various metrics
            l2_diff = torch.norm(P1 - P2, 'fro').item()
            l2_p1 = torch.norm(P1, 'fro').item()
            l2_p2 = torch.norm(P2, 'fro').item()
            
            # Normalized difference
            rel_diff = l2_diff / (l2_p1 + 1e-8)
            
            # SVD for subspace comparison
            U1, S1, _ = torch.svd(P1)
            U2, S2, _ = torch.svd(P2)
            
            # Compare singular values
            min_rank = min(len(S1), len(S2))
            if k is not None:
                min_rank = min(min_rank, k)
                
            s1_trunc = S1[:min_rank]
            s2_trunc = S2[:min_rank]
            
            l2_singular_diff = torch.norm(s1_trunc - s2_trunc).item()
            
            # Subspace overlap
            overlap_metrics = subspace_overlap(U1, U2, k=min_rank)
            
            # Recall
            recall = subspace_recall(U1, U2, k=min_rank)
            
            results[param_name] = {
                'l2_diff': l2_diff,
                'l2_p1': l2_p1,
                'l2_p2': l2_p2,
                'rel_diff': rel_diff,
                'l2_singular_value_diff': l2_singular_diff,
                'cos_angles': overlap_metrics['cos_angles'],
                'mean_subspace_overlap': overlap_metrics['mean_overlap'],
                'recall@k': recall,
                'min_angle': overlap_metrics['min_angle'],
                'max_angle': overlap_metrics['max_angle']
            }
            processed += 1
            
            if processed <= 3:
                print(f"  Result: rel_diff={rel_diff:.4f}, overlap={overlap_metrics['mean_overlap']:.4f}")
            
        except Exception as e:
            print(f"ERROR processing {param_name}: {e}")
    
    print(f"Final results: {len(results)} parameters processed")
    
    if plot and len(results) > 0:
        plot_comparison_results(results, id=id)
    
    return results


def plot_comparison_results(results, id='comparison'):
    """
    Plot comparison results across parameters.
    """
    print(f"\n=== DEBUG plot_comparison_results: {id} ===")
    params = list(results.keys())
    n_params = len(params)
    print(f"Plotting {n_params} parameters")
    
    if n_params == 0:
        print("No parameters to plot!")
        return
    
    # Show only top parameters to avoid cluttered plots
    max_params_to_show = 10
    if n_params > max_params_to_show:
        # Sort by relative difference to show most interesting parameters
        params_by_reldiff = sorted(params, key=lambda x: results[x]['rel_diff'], reverse=True)
        params = params_by_reldiff[:max_params_to_show]
        n_params = len(params)
        print(f"Reduced to top {n_params} parameters")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Parameter-wise Subspace Comparison Results - {id}', fontsize=16)
        
        # Metrics to plot
        metrics = ['l2_singular_value_diff', 'mean_subspace_overlap', 'recall@k', 'rel_diff']
        titles = ['L2 Singular Value Diff', 'Mean Subspace Overlap', 'Recall@k', 'Relative Difference']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            values = [results[param][metric] for param in params]
            
            ax.bar(range(len(params)), values, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel('Parameter')
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(params)))
            
            # Truncate parameter names for readability
            param_labels = [param[:20] + '...' if len(param) > 20 else param for param in params]
            ax.set_xticklabels(param_labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(j, v + max(values) * 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=6)
        
        plt.tight_layout()
        plt.savefig(f'subspace_comparison_{id}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved as subspace_comparison_{id}.png")
        
        # Print summary
        print(f"\nParameter-wise comparison summary ({len(params)} parameters shown):")
        print(f"Mean relative difference: {np.mean([results[p]['rel_diff'] for p in params]):.4f}")
        print(f"Mean subspace overlap: {np.mean([results[p]['mean_subspace_overlap'] for p in params]):.4f}")
        print(f"Mean recall@k: {np.mean([results[p]['recall@k'] for p in params]):.4f}")
        
        # Show top 5 most different parameters
        top_different = sorted(params, key=lambda x: results[x]['rel_diff'], reverse=True)[:5]
        print(f"\nTop 5 most different parameters:")
        for param in top_different:
            print(f"  {param}: rel_diff={results[param]['rel_diff']:.4f}")
            
    except Exception as e:
        print(f"ERROR in plotting: {e}")


def print_all_metrics(results, title):
    """Print all metrics for all layers"""
    print(f"\n=== DEBUG print_all_metrics: {title} ===")
    print(f"Results dict has {len(results)} entries")
    
    if len(results) == 0:
        print("No results to print!")
        return
        
    print(f"\n{title}:")
    print("-" * 80)
    
    for layer, vals in results.items():
        print(f"[{layer}]:")
        print(f"  L2 Diff: {vals['l2_diff']:.6f}")
        print(f"  L2 P1 (norm): {vals['l2_p1']:.6f}")
        print(f"  L2 P2 (norm): {vals['l2_p2']:.6f}")
        print(f"  Relative Diff: {vals['rel_diff']:.6f}")
        print(f"  L2 Singular Value Diff: {vals['l2_singular_value_diff']:.6f}")
        print(f"  Mean Subspace Overlap: {vals['mean_subspace_overlap']:.6f}")
        print(f"  Recall@k: {vals['recall@k']:.6f}")
        print(f"  Min Angle: {vals['min_angle']:.2f}°")
        print(f"  Max Angle: {vals['max_angle']:.2f}°")
        print()


# # Also add debug to project_gradients:
# def project_gradients(target_gradients, svd_basis, k=None):
#     """
#     Project gradients parameter-wise (exactly like unlearning method).
#     """
#     print(f"\n=== DEBUG project_gradients ===")
#     print(f"target_gradients: {len(target_gradients) if target_gradients else 0} keys")
#     print(f"svd_basis: {len(svd_basis) if svd_basis else 0} keys")
    
#     if not target_gradients:
#         print("ERROR: target_gradients is empty!")
#         return {}
#     if not svd_basis:
#         print("ERROR: svd_basis is empty!")
#         return {}
    
#     # Check overlap
#     overlap = set(target_gradients.keys()) & set(svd_basis.keys())
#     print(f"Parameter overlap: {len(overlap)}")
    
#     if len(overlap) == 0:
#         print("ERROR: No overlapping parameters!")
#         print(f"target_gradients keys: {list(target_gradients.keys())[:5]}")
#         print(f"svd_basis keys: {list(svd_basis.keys())[:5]}")
#         return {}
    
#     projected_gradients = {}
#     processed = 0
#     skipped = 0
    
#     for param_name in target_gradients:
#         if param_name not in svd_basis or svd_basis[param_name] is None:
#             skipped += 1
#             if skipped <= 3:
#                 print(f"Skipping {param_name}: not in SVD basis or None")
#             continue
            
#         grad_tensor = target_gradients[param_name]  # Already properly shaped
#         svd_result = svd_basis[param_name]
        
#         if grad_tensor is None:
#             print(f"Skipping {param_name}: gradient is None")
#             skipped += 1
#             continue
            
#         U = svd_result['U']
        
#         if k is not None:
#             U = U[:, :k]
        
#         # Create projection matrix P = U @ U.T (same as unlearning)
#         P = U @ U.T
        
#         # Check dimension compatibility
#         if grad_tensor.shape[-1] != P.shape[0]:
#             print(f"DIMENSION MISMATCH {param_name}: grad {grad_tensor.shape} vs proj {P.shape}")
#             continue
        
#         # Apply projection: grad - grad @ P (exactly like unlearning)
#         if len(grad_tensor.shape) == 2:  # [sz, features] or [out_features, in_features]
#             proj_grad = grad_tensor @ P
#             projected_gradients[param_name] = grad_tensor - proj_grad
#             processed += 1
            
#             if processed <= 3:
#                 print(f"Success {param_name}: {grad_tensor.shape} -> {projected_gradients[param_name].shape}")
#         else:
#             print(f"Unexpected shape {param_name}: {grad_tensor.shape}")
#             projected_gradients[param_name] = grad_tensor  # Skip if unexpected shape
    
#     print(f"project_gradients result: {processed} processed, {skipped} skipped")
#     return projected_gradients

def project_gradients(target_gradients, svd_basis, k=None):
    """
    Project gradients parameter-wise (exactly like unlearning method).
    """
    print(f"\n=== DEBUG project_gradients ===")
    print(f"target_gradients: {len(target_gradients) if target_gradients else 0} keys")
    print(f"svd_basis: {len(svd_basis) if svd_basis else 0} keys")
    
    if not target_gradients:
        print("ERROR: target_gradients is empty!")
        return {}
    if not svd_basis:
        print("ERROR: svd_basis is empty!")
        return {}
    
    # Check overlap
    overlap = set(target_gradients.keys()) & set(svd_basis.keys())
    print(f"Parameter overlap: {len(overlap)}")
    
    if len(overlap) == 0:
        print("ERROR: No overlapping parameters!")
        print(f"target_gradients keys: {list(target_gradients.keys())[:5]}")
        print(f"svd_basis keys: {list(svd_basis.keys())[:5]}")
        return {}
    
    projected_gradients = {}
    processed = 0
    skipped = 0
    
    for param_name in target_gradients:
        if param_name not in svd_basis or svd_basis[param_name] is None:
            skipped += 1
            if skipped <= 3:
                print(f"Skipping {param_name}: not in SVD basis or None")
            continue
            
        grad_tensor = target_gradients[param_name]  # Already properly shaped
        svd_result = svd_basis[param_name]
        
        if grad_tensor is None:
            print(f"Skipping {param_name}: gradient is None")
            skipped += 1
            continue
            
        U = svd_result['U']
        
        if k is not None:
            U = U[:, :k]
        
        # DTYPE FIX: Ensure both tensors have same dtype
        if grad_tensor.dtype != U.dtype:
            if processed < 3:
                print(f"Dtype conversion {param_name}: grad {grad_tensor.dtype} -> U {U.dtype}")
            # Convert to match gradient tensor dtype (usually float32)
            U = U.to(grad_tensor.dtype)
        
        # Create projection matrix P = U @ U.T (same as unlearning)
        P = U @ U.T
        
        # Check dimension compatibility
        if grad_tensor.shape[-1] != P.shape[0]:
            print(f"DIMENSION MISMATCH {param_name}: grad {grad_tensor.shape} vs proj {P.shape}")
            continue
        
        # Apply projection: grad - grad @ P (exactly like unlearning)
        if len(grad_tensor.shape) == 2:  # [sz, features] or [out_features, in_features]
            try:
                proj_grad = grad_tensor @ P
                projected_gradients[param_name] = grad_tensor - proj_grad
                processed += 1
                
                if processed <= 3:
                    print(f"Success {param_name}: {grad_tensor.shape} -> {projected_gradients[param_name].shape}")
                    print(f"  Dtypes: grad={grad_tensor.dtype}, U={U.dtype}, P={P.dtype}")
                    
            except Exception as e:
                print(f"ERROR projecting {param_name}: {e}")
                continue
        else:
            print(f"Unexpected shape {param_name}: {grad_tensor.shape}")
            projected_gradients[param_name] = grad_tensor  # Skip if unexpected shape
    
    print(f"project_gradients result: {processed} processed, {skipped} skipped")
    return projected_gradients


# def compare_projections(proj1, proj2, k=None, plot=True, id='comparison'):
#     """
#     Compare two sets of parameter-wise projections.
#     """
#     results = {}
    
#     common_params = set(proj1.keys()) & set(proj2.keys())
    
#     for param_name in common_params:
#         P1 = proj1[param_name]
#         P2 = proj2[param_name]
        
#         if P1 is None or P2 is None:
#             continue
            
#         # Calculate various metrics
#         l2_diff = torch.norm(P1 - P2, 'fro').item()
#         l2_p1 = torch.norm(P1, 'fro').item()
#         l2_p2 = torch.norm(P2, 'fro').item()
        
#         # Normalized difference
#         rel_diff = l2_diff / (l2_p1 + 1e-8)
        
#         # SVD for subspace comparison
#         U1, S1, _ = torch.svd(P1)
#         U2, S2, _ = torch.svd(P2)
        
#         # Compare singular values
#         min_rank = min(len(S1), len(S2))
#         if k is not None:
#             min_rank = min(min_rank, k)
            
#         s1_trunc = S1[:min_rank]
#         s2_trunc = S2[:min_rank]
        
#         l2_singular_diff = torch.norm(s1_trunc - s2_trunc).item()
        
#         # Subspace overlap
#         overlap_metrics = subspace_overlap(U1, U2, k=min_rank)
        
#         # Recall
#         recall = subspace_recall(U1, U2, k=min_rank)
        
#         results[param_name] = {
#             'l2_diff': l2_diff,
#             'l2_p1': l2_p1,
#             'l2_p2': l2_p2,
#             'rel_diff': rel_diff,
#             'l2_singular_value_diff': l2_singular_diff,
#             'cos_angles': overlap_metrics['cos_angles'],
#             'mean_subspace_overlap': overlap_metrics['mean_overlap'],
#             'recall@k': recall,
#             'min_angle': overlap_metrics['min_angle'],
#             'max_angle': overlap_metrics['max_angle']
#         }
    
#     if plot:
#         plot_comparison_results(results, id=id)
    
#     return results


# def plot_comparison_results(results, id='comparison'):
#     """
#     Plot comparison results across parameters.
#     """
#     params = list(results.keys())
#     n_params = len(params)
    
#     if n_params == 0:
#         return
    
#     # Show only top parameters to avoid cluttered plots
#     max_params_to_show = 10
#     if n_params > max_params_to_show:
#         # Sort by relative difference to show most interesting parameters
#         params_by_reldiff = sorted(params, key=lambda x: results[x]['rel_diff'], reverse=True)
#         params = params_by_reldiff[:max_params_to_show]
#         n_params = len(params)
    
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle(f'Parameter-wise Subspace Comparison Results - {id}', fontsize=16)
    
#     # Metrics to plot
#     metrics = ['l2_singular_value_diff', 'mean_subspace_overlap', 'recall@k', 'rel_diff']
#     titles = ['L2 Singular Value Diff', 'Mean Subspace Overlap', 'Recall@k', 'Relative Difference']
    
#     for i, (metric, title) in enumerate(zip(metrics, titles)):
#         ax = axes[i//2, i%2]
#         values = [results[param][metric] for param in params]
        
#         ax.bar(range(len(params)), values, alpha=0.7)
#         ax.set_title(title)
#         ax.set_xlabel('Parameter')
#         ax.set_ylabel(metric)
#         ax.set_xticks(range(len(params)))
        
#         # Truncate parameter names for readability
#         param_labels = [param[:20] + '...' if len(param) > 20 else param for param in params]
#         ax.set_xticklabels(param_labels, rotation=45, ha='right')
        
#         # Add value labels on bars
#         for j, v in enumerate(values):
#             ax.text(j, v + max(values) * 0.01, f'{v:.3f}', 
#                    ha='center', va='bottom', fontsize=6)
    
#     plt.tight_layout()
#     plt.savefig(f'subspace_comparison_{id}.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # Print summary
#     print(f"\nParameter-wise comparison summary ({len(params)} parameters shown):")
#     print(f"Mean relative difference: {np.mean([results[p]['rel_diff'] for p in params]):.4f}")
#     print(f"Mean subspace overlap: {np.mean([results[p]['mean_subspace_overlap'] for p in params]):.4f}")
#     print(f"Mean recall@k: {np.mean([results[p]['recall@k'] for p in params]):.4f}")
    
#     # Show top 5 most different parameters
#     top_different = sorted(params, key=lambda x: results[x]['rel_diff'], reverse=True)[:5]
#     print(f"\nTop 5 most different parameters:")
#     for param in top_different:
#         print(f"  {param}: rel_diff={results[param]['rel_diff']:.4f}")


# Additional helper function to map layer names to parameter names
def get_parameter_to_layer_mapping(model):
    """
    Create mapping from parameter names to layer names (like unlearning code does).
    """
    mapping = {}
    for name, param in model.named_parameters():
        # Extract layer name from parameter name
        if '.weight' in name:
            layer_name = name.replace('.weight', '')
        elif '.bias' in name:
            layer_name = name.replace('.bias', '')
        else:
            layer_name = name
        mapping[name] = layer_name
    return mapping




def subspace_recall(U_true, U_est, k=None):
    """
    Calculate recall between two subspaces.
    """
    if k is None:
        k = min(U_true.shape[1], U_est.shape[1])
    
    U_true_k = U_true[:, :k]
    U_est_k = U_est[:, :k]
    proj = U_est_k @ U_est_k.T
    recall = torch.trace(U_true_k.T @ proj @ U_true_k) / k
    return recall.item()


def subspace_overlap(U1, U2, k=None):
    """
    Calculate overlap between two subspaces using principal angles.
    """
    if k is None:
        k = min(U1.shape[1], U2.shape[1])
    
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]
    
    # Calculate cosines of principal angles
    _, s, _ = torch.svd(U1_k.T @ U2_k)
    cos_angles = s
    
    # Mean overlap
    mean_overlap = cos_angles.mean().item()
    
    return {
        'cos_angles': cos_angles,
        'mean_overlap': mean_overlap,
        'min_angle': torch.acos(cos_angles.max()).item() * 180 / np.pi,
        'max_angle': torch.acos(cos_angles.min()).item() * 180 / np.pi
    }






# def print_all_metrics(results, title):
#     """Print all metrics for all layers"""
#     print(f"\n{title}:")
#     print("-" * 80)
    
#     for layer, vals in results.items():
#         print(f"[{layer}]:")
#         print(f"  L2 Diff: {vals['l2_diff']:.6f}")
#         print(f"  L2 P1 (norm): {vals['l2_p1']:.6f}")
#         print(f"  L2 P2 (norm): {vals['l2_p2']:.6f}")
#         print(f"  Relative Diff: {vals['rel_diff']:.6f}")
#         print(f"  L2 Singular Value Diff: {vals['l2_singular_value_diff']:.6f}")
#         print(f"  Mean Subspace Overlap: {vals['mean_subspace_overlap']:.6f}")
#         print(f"  Recall@k: {vals['recall@k']:.6f}")
#         print(f"  Min Angle: {vals['min_angle']:.2f}°")
#         print(f"  Max Angle: {vals['max_angle']:.2f}°")
#         print()


def analyze_forget_subspace(original_model, unlearned_model, background_data, 
                          forget_data, retain_data, device, k=None, 
                          difference_method='direct', epochs=1):
    """
    Main function to analyze the forget subspace using model differences.
    """
    print("=== Analyzing Forget Subspace Recovery ===")
    
    # Calculate covariances
    print('1. Calculating covariances...')
    print('   - Original model on background data...')
    original_covar, _ = calculate_gradient_covariance(original_model, background_data, device, epochs)
    
    print('   - Unlearned model on background data...')
    unlearned_covar, _ = calculate_gradient_covariance(unlearned_model, background_data, device, epochs)
    
    print('   - Original model on forget data...')
    forget_covar, _ = calculate_gradient_covariance(original_model, forget_data, device, epochs)
    
    print('   - Original model on retain data...')
    retain_covar, _ = calculate_gradient_covariance(original_model, retain_data, device, epochs)
    
    # Calculate difference (estimated forget subspace)
    print('2. Calculating estimated forget subspace...')
    est_forget_svd = calculate_svd_difference(original_covar, unlearned_covar, device, k, difference_method)
    
    # Calculate true forget subspace
    print('3. Calculating true forget subspace...')
    forget_svd = calculate_svd(forget_covar, k=k)
    
    # Calculate retain subspace for comparison
    print('4. Calculating retain subspace...')
    retain_svd = calculate_svd(retain_covar, k=k)
    
    # Project original covariances onto different subspaces
    print('5. Projecting onto subspaces...')
    est_projected_forget = project_gradients(original_covar, est_forget_svd, k=k)
    true_projected_forget = project_gradients(original_covar, forget_svd, k=k)
    projected_retain = project_gradients(original_covar, retain_svd, k=k)
    
    # Compare results
    print('6. Comparing results...')
    forget_results = compare_projections(est_projected_forget, true_projected_forget, 
                                       k=k, plot=True, id='forget_comparison')
    
    retain_results = compare_projections(est_projected_forget, projected_retain, 
                                       k=k, plot=True, id='retain_comparison')
    
    # Print comprehensive summary with ALL metrics for ALL layers
    print("\n" + "="*80)
    print("COMPREHENSIVE FORGET SUBSPACE RECOVERY RESULTS")
    print("="*80)
    
    print_all_metrics(forget_results, "ESTIMATED vs TRUE FORGET SUBSPACE (should be SIMILAR)")
    print_all_metrics(retain_results, "ESTIMATED FORGET vs RETAIN SUBSPACE (should be DIFFERENT)")
    
    return {
        'forget_results': forget_results,
        'retain_results': retain_results,
        'est_forget_svd': est_forget_svd,
        'true_forget_svd': forget_svd,
        'retain_svd': retain_svd
    }


