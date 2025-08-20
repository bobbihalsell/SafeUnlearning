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

def debug_layer_names(subspace, target_dict, gradient_dict, dict_name="data", show_details=False):
    """
    Debug function to check layer name mismatches between dictionaries.
    """
    subspace_layers = set(subspace.keys()) if subspace else set()
    target_layers = set(target_dict.keys()) if target_dict else set()
    gradient_layers = set(gradient_dict.keys()) if gradient_dict else set()
    
    print(f"\nLAYER NAME DEBUG - {dict_name.upper()}")
    print("-" * 50)
    print(f"Subspace layers ({len(subspace_layers)}): {sorted(list(subspace_layers)[:5])}{'...' if len(subspace_layers) > 5 else ''}")
    print(f"Target layers ({len(target_layers)}): {sorted(list(target_layers)[:5])}{'...' if len(target_layers) > 5 else ''}")
    print(f"{dict_name} layers ({len(gradient_layers)}): {sorted(list(gradient_layers)[:5])}{'...' if len(gradient_layers) > 5 else ''}")
    
    # Check for mismatches
    missing_in_target = subspace_layers - target_layers
    missing_in_data = subspace_layers - gradient_layers
    common_layers = subspace_layers & target_layers & gradient_layers
    
    if missing_in_target:
        print(f"⚠️  Layers in subspace but missing in target: {sorted(list(missing_in_target)[:3])}{'...' if len(missing_in_target) > 3 else ''}")
    if missing_in_data:
        print(f"⚠️  Layers in subspace but missing in {dict_name}: {sorted(list(missing_in_data)[:3])}{'...' if len(missing_in_data) > 3 else ''}")
        
    print(f"✅ Common layers across all ({len(common_layers)}): {sorted(list(common_layers)[:3])}{'...' if len(common_layers) > 3 else ''}")
    
    if show_details and len(common_layers) < len(subspace_layers):
        print(f"\n DETAILED LAYER COMPARISON:")
        print(f"All subspace layers: {sorted(list(subspace_layers))}")
        print(f"All {dict_name} layers: {sorted(list(gradient_layers))}")
    
    return common_layers

def _get_feature_dict(model):
    """
    Enhanced feature dictionary generation with better layer coverage.
    Same as PGU implementation.
    """
    try:
        _, eval_nodes = get_graph_node_names(model)
        conv_fea_dict = OrderedDict()
        linear_fea_dict = OrderedDict()
        
        # Traverse all modules, not just those in eval_nodes
        for name, module in model.named_modules():
            # Include Conv2d layers
            if isinstance(module, torch.nn.Conv2d):
                conv_fea_dict[name] = name
            # Include Linear layers  
            elif isinstance(module, torch.nn.Linear):
                linear_fea_dict[name] = name
        
        print(f"Generated {len(conv_fea_dict)} conv layers and {len(linear_fea_dict)} linear layers")
        print(f"Conv layers: {list(conv_fea_dict.keys())}")
        print(f"Linear layers: {list(linear_fea_dict.keys())}")
        return conv_fea_dict, linear_fea_dict
    except Exception as e:
        print(f"Error in _get_feature_dict: {e}")
        raise

def _get_module_by_name(model, layer_name):
    """Get a module by its name from the model. Same as PGU implementation."""
    try:
        parts = layer_name.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    except Exception as e:
        print(f"Error getting module {layer_name}: {e}")
        raise

def get_model_gradients(model, data_loader, device, loss_fn=nn.CrossEntropyLoss(), 
                                  epochs=1, max_batches=50, cache_dir=None, label=''):
    """
    Get gradients for a model on given data using PGU style gradient collection.
    Returns gradients organized by layer name (not parameter name).
    """
    print(f"Computing gradients...")
    
    # Get feature dictionaries
    conv_fea_dict, linear_fea_dict = _get_feature_dict(model)
    all_layers = {**conv_fea_dict, **linear_fea_dict}
    
    original_mode = model.training
    model.train()
    
    # Collect gradients for each layer
    layer_gradients = {}
    for layer_name in all_layers:
        layer_gradients[layer_name] = []
    
    try:
        # Collect gradients over multiple batches
        total_batches = 0
        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc=f"Collecting gradients epoch {epoch+1}")):
                if total_batches >= max_batches:  # Limit to prevent memory issues
                    break
                    
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                model.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                
                # Collect gradients for each target layer
                for layer_name in all_layers:
                    try:
                        module = _get_module_by_name(model, layer_name)
                        if hasattr(module, 'weight') and module.weight.grad is not None:
                            # Flatten the gradient - same as PGU
                            grad_flat = module.weight.grad.data.view(module.weight.shape[0], -1)
                            layer_gradients[layer_name].append(grad_flat.cpu().clone())
                    except Exception as e:
                        print(f"Error collecting gradient for {layer_name}: {e}")
                        continue
                
                total_batches += 1
                
                del images, labels, outputs, loss
                torch.cuda.empty_cache()
        
        # Average gradients across batches
        averaged_gradients = {}
        for layer_name, grads in layer_gradients.items():
            if len(grads) > 0:
                # Concatenate and average
                all_grads = torch.cat(grads, dim=0)  # [total_samples, input_dim]
                avg_grad = torch.mean(all_grads, dim=0, keepdim=True)  # [1, input_dim]
                averaged_gradients[layer_name] = avg_grad.squeeze(0)  # [input_dim]
                print(f'Layer {layer_name}: averaged {len(grads)} gradient samples, final shape: {averaged_gradients[layer_name].shape}')
            else:
                print(f"No gradients collected for {layer_name}")
        
    finally:
        model.train(original_mode)
    
    # Cache results if requested
    if cache_dir:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_path = os.path.join(cache_dir, f"{label}_gradients.pt")
        torch.save(averaged_gradients, cache_path)
        print(f"Cached gradients to {cache_path}")
    
    return averaged_gradients

def compute_parameter_difference(original_model, unlearned_model, learning_rate, cache_dir=None):
    """
    Compute parameter differences organized by layer name (same as gradients).
    Returns (theta_u - theta_o) / learning_rate for each layer.
    """
    print(f"Computing parameter differences...")
    
    # Get feature dictionaries to know which layers to process
    conv_fea_dict, linear_fea_dict = _get_feature_dict(original_model)
    all_layers = {**conv_fea_dict, **linear_fea_dict}
    
    param_differences = {}
    
    # Process each layer
    for layer_name in all_layers:
        try:
            # Get modules from both models
            orig_module = _get_module_by_name(original_model, layer_name)
            unl_module = _get_module_by_name(unlearned_model, layer_name)
            
            if hasattr(orig_module, 'weight') and hasattr(unl_module, 'weight'):
                # Compute parameter difference and flatten like gradients
                param_diff = (unl_module.weight.data - orig_module.weight.data) / learning_rate
                param_diff_flat = param_diff.view(param_diff.shape[0], -1)  # Same as gradient flattening
                param_diff_flat = torch.mean(param_diff_flat, dim=0)  # Average to match gradient format
                
                param_differences[layer_name] = param_diff_flat.cpu()
                
                diff_norm = torch.norm(param_diff_flat).item()
                print(f"Layer {layer_name}: parameter difference norm = {diff_norm:.6f}, shape = {param_diff_flat.shape}")
            
        except Exception as e:
            print(f"Error computing parameter difference for {layer_name}: {e}")
            continue
    
    print(f"Computed parameter differences for {len(param_differences)} layers")
    
    # Cache results if requested
    if cache_dir:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_path = os.path.join(cache_dir, "parameter_differences.pt")
        torch.save(param_differences, cache_path)
        print(f"Cached parameter differences to {cache_path}")
    
    return param_differences

def find_subspace(original_model, unlearned_model, background_data, device, 
                            loss_fn=nn.CrossEntropyLoss(), variance_threshold=0.9, 
                            epochs=1, cache_dir=None):
    """
    Find subspace using PGU style gradient computation.
    No dimensionality reduction for now as requested.
    """
    print("Finding subspace...")
    
    # Get gradients from both models using PGU style
    orig_grads = get_model_gradients(
        original_model, background_data, device, loss_fn, 
        epochs=epochs, cache_dir=cache_dir, label='original'
    )
    unl_grads = get_model_gradients(
        unlearned_model, background_data, device, loss_fn, 
        epochs=epochs, cache_dir=cache_dir, label='unlearned'
    )
    
    if not orig_grads or not unl_grads:
        print("Failed to compute gradients")
        return None
    
    # Calculate gradient differences
    grad_differences = {}
    total_diff_norm = 0
    
    for layer_name in orig_grads:
        if layer_name in unl_grads:
            diff = orig_grads[layer_name] - unl_grads[layer_name]
            grad_differences[layer_name] = diff
            total_diff_norm += torch.norm(diff).item()
    
    print(f"Total gradient difference magnitude: {total_diff_norm:.6f}")
    print(f"Gradient differences from {len(grad_differences)} layers")
    
    # Extract subspaces from gradient differences
    subspace_signature = {}
    
    for layer_name, grad_diff in grad_differences.items():
        try:
            # Keep as is - no dimensionality reduction as requested
            grad_diff_cpu = grad_diff.detach().cpu().float()
            
            # Create covariance matrix from the gradient difference
            # For simplicity, treat gradient as single sample covariance
            grad_cov = torch.outer(grad_diff_cpu, grad_diff_cpu)
            
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
            
            print(f"Layer {layer_name}: kept {components_to_keep}/{len(S)} components, "
                  f"variance ratio: {subspace_signature[layer_name]['explained_variance_ratio']:.4f}")
            
        except Exception as e:
            print(f"Subspace extraction failed for {layer_name}: {e}")
            continue

    if cache_dir:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_path = os.path.join(cache_dir, f"subspace.pt")
        torch.save(subspace_signature, cache_path)
        print(f"Cached subspace to {cache_path}")
    
    return subspace_signature

def project_onto_subspace(target, subspace):
    """
    Project target onto subspace with consistent dimensionality.
    No random projection - dimensions should match now.
    """
    projections = {}

    for layer_name in subspace:
        if layer_name not in target:
            print(f"Layer {layer_name} not found in target, skipping projection.")
            continue
            
        try:
            subspace_basis = subspace[layer_name]['basis']
            target_vec = target[layer_name].detach().cpu().flatten().float()
            
            # Check dimension compatibility
            if target_vec.shape[0] != subspace_basis.shape[0]:
                print(f"Dimension mismatch for {layer_name}: target {target_vec.shape[0]}, "
                      f"subspace {subspace_basis.shape[0]}. Skipping.")
                continue
            
            # Project onto subspace (sum of absolute projections onto all basis vectors)
            projection = torch.sum(torch.abs(subspace_basis.T @ target_vec)).item()
            projections[layer_name] = projection
            
            print(f"Layer {layer_name}: projection = {projection:.6f}")
            
        except Exception as e:
            print(f"Subspace projection failed for {layer_name}: {e}")
            continue
    
    return projections

# Updated test functions
def test_layer_names(dataloader1, dataloader2, original_model, unlearned_model, 
                               device, loss_fn=nn.CrossEntropyLoss(), variance_threshold=0.9):
    """Test layer names using PGU style functions."""
    print("Testing layer names...")
    
    subspace = find_subspace(
        original_model, unlearned_model, dataloader1, device, 
        variance_threshold=variance_threshold, cache_dir='test_layer_names_cache'
    )
    
    param_difference = compute_parameter_difference(
        original_model, unlearned_model, learning_rate=0.001, 
        cache_dir='test_layer_names_cache'
    )
    
    gradients = get_model_gradients(
        unlearned_model, dataloader2, device, loss_fn, 
        cache_dir='test_layer_names_cache', label='test'
    )
    
    debug_layer_names(subspace, param_difference, gradients, dict_name="gradients", show_details=True)
    
    return subspace, param_difference, gradients

def test_projection(forgetdata, retaindata, original_model, unlearned_model, 
                              bdata, learning_rate, device, loss_fn=nn.CrossEntropyLoss(), 
                              variance_threshold=0.9):
    """Test projection using PGU style functions."""
    print("Testing projection...")
    
    print("Finding approximate subspace on available data...")
    subspace = find_subspace(
        original_model, unlearned_model, bdata, device, 
        variance_threshold=variance_threshold, cache_dir='test_projection_cache'
    )

    print("Calculating parameter differences...")
    normalised_param_difference = compute_parameter_difference(
        original_model, unlearned_model, learning_rate, 
        cache_dir='test_projection_cache'
    )

    print("Calculating forget gradients...")
    forget_grads = get_model_gradients(
        unlearned_model, forgetdata, device, loss_fn, 
        cache_dir='test_projection_cache', label='forget'
    )

    print("Calculating retain gradients...")
    retain_grads = get_model_gradients(
        unlearned_model, retaindata, device, loss_fn, 
        cache_dir='test_projection_cache', label='retain'
    )

    print("Calculating target projection...")
    target_projection = project_onto_subspace(normalised_param_difference, subspace)

    print("Calculating forget projection...")
    forget_projection = project_onto_subspace(forget_grads, subspace)

    print("Calculating retain projection...")
    retain_projection = project_onto_subspace(retain_grads, subspace)
    
    # Analysis code continues as before...
    forget_projection_diff = {}
    retain_projection_diff = {}
    
    for layer in target_projection:
        forget_val = forget_projection.get(layer, 0)
        retain_val = retain_projection.get(layer, 0)
        target_val = target_projection[layer]
        
        forget_projection_diff[layer] = target_val - forget_val
        retain_projection_diff[layer] = target_val - retain_val
    
    # Comprehensive Analysis (same as before)
    print("\n" + "="*60)
    print("PROJECTION ANALYSIS RESULTS")
    print("="*60)
    
    # Calculate alignment scores
    target_total = sum(target_projection.values())
    forget_total = sum(forget_projection.values()) 
    retain_total = sum(retain_projection.values())
    
    print(f"\nTotal Projection Magnitudes:")
    print(f"  Target Direction (θ_u - θ_o)/lr: {target_total:.6f}")
    print(f"  Forget Data Gradients:          {forget_total:.6f}")
    print(f"  Retain Data Gradients:          {retain_total:.6f}")
    
    # Relative alignments
    forget_alignment = forget_total / target_total if target_total > 0 else 0
    retain_alignment = retain_total / target_total if target_total > 0 else 0
    
    print(f"\nRelative Alignment with Target Direction:")
    print(f"  Forget Data: {forget_alignment:.4f} ({'HIGH' if forget_alignment > 0.5 else 'LOW'})")
    print(f"  Retain Data: {retain_alignment:.4f} ({'HIGH' if retain_alignment > 0.5 else 'LOW'})")
    
    # Difference analysis
    forget_diff_total = sum(abs(v) for v in forget_projection_diff.values())
    retain_diff_total = sum(abs(v) for v in retain_projection_diff.values())
    
    print(f"\nProjection Differences from Target:")
    print(f"  |Target - Forget|: {forget_diff_total:.6f} ({'LOW' if forget_diff_total < retain_diff_total else 'HIGH'})")
    print(f"  |Target - Retain|: {retain_diff_total:.6f} ({'HIGH' if retain_diff_total > forget_diff_total else 'LOW'})")
    
    # Layer-wise analysis
    print(f"\nLayer-wise Projection Analysis:")
    print(f"{'Layer':<25} {'Target':<12} {'Forget':<12} {'Retain':<12} {'F-Diff':<12} {'R-Diff':<12}")
    print("-" * 95)
    
    for layer in target_projection.keys():
        target_val = target_projection.get(layer, 0)
        forget_val = forget_projection.get(layer, 0) 
        retain_val = retain_projection.get(layer, 0)
        forget_diff = forget_projection_diff.get(layer, 0)
        retain_diff = retain_projection_diff.get(layer, 0)
        
        print(f"{layer:<25} {target_val:<12.6f} {forget_val:<12.6f} {retain_val:<12.6f} "
              f"{forget_diff:<12.6f} {retain_diff:<12.6f}")
    
    # Theoretical validation
    print(f"\n" + "="*60)
    print("THEORETICAL PROPOSITION VALIDATION:")
    print("="*60)
    
    proposition1_satisfied = forget_alignment > retain_alignment
    proposition2_satisfied = forget_diff_total < retain_diff_total
    
    print(f"Proposition 1 - Forget > Retain alignment: {'✓ PASS' if proposition1_satisfied else '✗ FAIL'}")
    print(f"Proposition 2 - Forget closer to target:   {'✓ PASS' if proposition2_satisfied else '✗ FAIL'}")
    
    # Compile results
    results = {
        'subspace_signature': subspace,
        'target_projection': target_projection,
        'forget_projection': forget_projection, 
        'retain_projection': retain_projection,
        'forget_projection_diff': forget_projection_diff,
        'retain_projection_diff': retain_projection_diff,
        'target_total': target_total,
        'forget_total': forget_total,
        'retain_total': retain_total,
        'forget_alignment': forget_alignment,
        'retain_alignment': retain_alignment,
        'forget_diff_total': forget_diff_total,
        'retain_diff_total': retain_diff_total,
        'proposition1_satisfied': proposition1_satisfied,
        'proposition2_satisfied': proposition2_satisfied
    }
    
    # Cache results
    cache_path = os.path.join('test_projection_cache', 'projection_test_results.pt')
    torch.save(results, cache_path)
    print(f"\nResults cached to: {cache_path}")
    
    return results
