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


def calculate_covar(model, data_loader, device, epochs=1):
    """
    Calculate covariance matrices of input representations for each layer.
    This is similar to R^l(R^l)^T from the paper.
    """
    conv_fea_dict, linear_fea_dict = get_feature_dict(model)
    
    # Setup feature extraction
    fea_dict = {val: val for val in {**conv_fea_dict, **linear_fea_dict}.values()}
    tmp_fea_dict = fea_dict.copy()
    features = {'input': []} if 'input' in fea_dict else {}
    
    if 'input' in tmp_fea_dict:
        tmp_fea_dict.pop('input')
    
    fea_ext = create_feature_extractor(model, tmp_fea_dict)
    
    # Initialize covariance dictionary
    covar = {layer: 0 for layer in {**conv_fea_dict, **linear_fea_dict}}
    sample_counts = {layer: 0 for layer in {**conv_fea_dict, **linear_fea_dict}}
    
    model.eval()
    with torch.no_grad():
        for epoch in range(epochs):
            for imgs, _ in tqdm(data_loader, desc=f"Computing covariance (epoch {epoch+1}/{epochs})"):
                imgs = imgs.to(device)
                
                if 'input' in fea_dict:
                    features['input'] = imgs
                
                feats = fea_ext(imgs)
                for fea_name, val in feats.items():
                    features[fea_name] = val.detach()
                
                # Process conv layers
                for layer in conv_fea_dict:
                    layer_module = get_module_by_name(model, layer)
                    ks = layer_module.kernel_size
                    pad = layer_module.padding
                    
                    f = features[conv_fea_dict[layer]]
                    patch = F.unfold(f, ks, dilation=1, padding=pad, stride=1)
                    patch = patch.permute(0, 2, 1).reshape(-1, patch.shape[1]).double()
                    covar[layer] += patch.T @ patch
                    sample_counts[layer] += patch.shape[0]
                
                # Process linear layers
                for layer in linear_fea_dict:
                    f = features[linear_fea_dict[layer]].double().squeeze()
                    if f.ndim == 1:
                        f = f.unsqueeze(0)
                    covar[layer] += f.T @ f
                    sample_counts[layer] += f.shape[0]
    
    return covar, sample_counts



def calculate_svd(covar, k=None, eps=1e-6):
    """
    Calculate SVD of covariance matrices.
    """
    svd = {}
    for layer, C in covar.items():
        # Add regularization for numerical stability
        U, S, _ = torch.svd(C + eps * torch.eye(C.size(0), device=C.device))
        if k is not None:
            U = U[:, :k]
            S = S[:k]
        svd[layer] = {'U': U, 'S': S}
    return svd


def calculate_svd_difference(covar_original, covar_unlearned, device, k=None):
    """
    Calculate the difference in representation spaces between original and unlearned models.
    
    Args:
        covar_original: covariance from original model
        covar_unlearned: covariance from unlearned model  
        device: computation device
        k: top-k components
    """
    diff_svd = {}
    
    for layer in covar_original:
        if layer not in covar_unlearned:
            continue
            
        C_orig = covar_original[layer].to(device)
        C_unlearn = covar_unlearned[layer].to(device)

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
            
        diff_svd[layer] = {'U': U, 'S': S}
    
    return diff_svd


def project_onto_subspace(gradients, svd_basis, k=None, debug=True):
    """
    Project target gradients onto subspaces defined by SVDs.
    """
    projected_grads = {}
    
    for layer in gradients:
        if layer not in svd_basis:
            if debug:
                print(f"Layer {layer}: not in SVD basis")
            continue
            
        C_target = gradients[layer]
        U = svd_basis[layer]['U']
        
        if debug:
            print(f"Layer {layer}:")
            print(f"  Target shape: {C_target.shape}")
            print(f"  U shape: {U.shape}")
        
        if k is not None:
            U = U[:, :k]
            if debug:
                print(f"  U truncated shape: {U.shape}")
        
        # Create projection matrix
        P = U @ U.T
        if debug:
            print(f"  P shape: {P.shape}")
        
        # Check dimension compatibility
        if C_target.shape[0] != P.shape[0]:
            if debug:
                print(f"  Dimension mismatch: {C_target.shape[0]} vs {P.shape[0]}")
            continue
        
        C_proj = C_target @ P
        if debug:
            print(f"  Projected shape: {C_proj.shape}")

        projected_grads[layer] = C_proj
    
    return projected_grads

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


def compare_projections(proj1, proj2, k=None, plot=True, id='comparison'):
    """
    Compare two sets of projections across layers.
    """
    results = {}
    
    for layer in proj1:
        if layer not in proj2:
            continue
            
        P1 = proj1[layer]
        P2 = proj2[layer]
        
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
        
        results[layer] = {
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
    
    if plot:
        plot_comparison_results(results, id=id)
    
    return results


def plot_comparison_results(results, id='comparison'):
    """
    Plot comparison results across layers.
    """
    layers = list(results.keys())
    n_layers = len(layers)
    
    if n_layers == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Subspace Comparison Results - {id}', fontsize=16)
    
    # Metrics to plot
    metrics = ['l2_singular_value_diff', 'mean_subspace_overlap', 'recall@k', 'rel_diff']
    titles = ['L2 Singular Value Diff', 'Mean Subspace Overlap', 'Recall@k', 'Relative Difference']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        values = [results[layer][metric] for layer in layers]
        
        ax.bar(range(len(layers)), values, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Layer')
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        
        # Add value labels on bars
        for j, v in enumerate(values):
            ax.text(j, v + max(values) * 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'subspace_comparison_{id}.png', dpi=300, bbox_inches='tight')
    plt.show()


# def analyze_forget_subspace(original_model, unlearned_model, background_data, 
#                           forget_data, retain_data, device, k=None, 
#                           difference_method='direct', epochs=1):
#     """
#     Main function to analyze the forget subspace using model differences.
#     """
#     print("=== Analyzing Forget Subspace Recovery ===")
    
#     # Calculate covariances
#     print('1. Calculating covariances...')
#     print('   - Original model on background data...')
#     original_covar, _ = calculate_covar(original_model, background_data, device, epochs)
    
#     print('   - Unlearned model on background data...')
#     unlearned_covar, _ = calculate_covar(unlearned_model, background_data, device, epochs)
    
#     print('   - Original model on forget data...')
#     forget_covar, _ = calculate_covar(original_model, forget_data, device, epochs)
    
#     print('   - Original model on retain data...')
#     retain_covar, _ = calculate_covar(original_model, retain_data, device, epochs)
    
#     # Calculate difference (estimated forget subspace)
#     print('2. Calculating estimated forget subspace...')
#     est_forget_svd = calculate_svd_difference(original_covar, unlearned_covar, device, k, difference_method)
    
#     # Calculate true forget subspace
#     print('3. Calculating true forget subspace...')
#     forget_svd = calculate_svd(forget_covar, k=k)
    
#     # Calculate retain subspace for comparison
#     print('4. Calculating retain subspace...')
#     retain_svd = calculate_svd(retain_covar, k=k)
    
#     # Project original covariances onto different subspaces
#     print('5. Projecting onto subspaces...')
#     est_projected_forget = project_onto_subspace(original_covar, est_forget_svd, k=k)
#     true_projected_forget = project_onto_subspace(original_covar, forget_svd, k=k)
#     projected_retain = project_onto_subspace(original_covar, retain_svd, k=k)
    
#     # Compare results
#     print('6. Comparing results...')
#     forget_results = compare_projections(est_projected_forget, true_projected_forget, 
#                                        k=k, plot=True, id='forget_comparison')
    
#     retain_results = compare_projections(est_projected_forget, projected_retain, 
#                                        k=k, plot=True, id='retain_comparison')
    
#     # Print summary
#     print("\n=== FORGET SUBSPACE RECOVERY RESULTS ===")
#     print("Estimated vs True Forget Subspace:")
#     for layer, vals in forget_results.items():
#         print(f"  [{layer}] L2 diff: {vals['l2_singular_value_diff']:.4f}, "
#               f"Overlap: {vals['mean_subspace_overlap']:.4f}, "
#               f"Recall: {vals['recall@k']:.4f}")
    
#     print("\nEstimated Forget vs Retain Subspace (should be different):")
#     for layer, vals in retain_results.items():
#         print(f"  [{layer}] L2 diff: {vals['l2_singular_value_diff']:.4f}, "
#               f"Overlap: {vals['mean_subspace_overlap']:.4f}, "
#               f"Recall: {vals['recall@k']:.4f}")
    
#     return {
#         'forget_results': forget_results,
#         'retain_results': retain_results,
#         'est_forget_svd': est_forget_svd,
#         'true_forget_svd': forget_svd,
#         'retain_svd': retain_svd
#     }


def print_all_metrics(results, title):
    """Print all metrics for all layers"""
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
    original_covar, _ = calculate_covar(original_model, background_data, device, epochs)
    
    print('   - Unlearned model on background data...')
    unlearned_covar, _ = calculate_covar(unlearned_model, background_data, device, epochs)
    
    print('   - Original model on forget data...')
    forget_covar, _ = calculate_covar(original_model, forget_data, device, epochs)
    
    print('   - Original model on retain data...')
    retain_covar, _ = calculate_covar(original_model, retain_data, device, epochs)
    
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
    est_projected_forget = project_onto_subspace(original_covar, est_forget_svd, k=k)
    true_projected_forget = project_onto_subspace(original_covar, forget_svd, k=k)
    projected_retain = project_onto_subspace(original_covar, retain_svd, k=k)
    
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





# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.fx as fx
# from torch.fx import GraphModule
# from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
                                                    
# import os
# import copy
# import re
# import math
# from tqdm import tqdm
# from collections import OrderedDict
# from typing import Optional, Union, List, Dict, Any


# def get_feature_dict(model):
#     """
#     Automatically generate conv_fea_dict and linear_fea_dict for a given model.

#     Returns:
#         conv_fea_dict (dict): maps layer names to feature extractor keys for Conv layers
#         linear_fea_dict (dict): maps layer names to feature extractor keys for Linear layers
#     """
#     _, eval_nodes = get_graph_node_names(model)
#     conv_fea_dict = OrderedDict()
#     linear_fea_dict = OrderedDict()
#     # Traverse the model's modules and keep track of the module names
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv2d) and name in eval_nodes:
#             conv_fea_dict[name] = name
#         elif isinstance(module, torch.nn.Linear) and name in eval_nodes:
#             linear_fea_dict[name] = name
#     return conv_fea_dict, linear_fea_dict


# # calculate subspace on data wrt the models

# def get_module_by_name(model, layer_name):
#     parts = layer_name.split('.')
#     module = model
#     for part in parts:
#         if part.isdigit():
#             module = module[int(part)]
#         else:
#             module = getattr(module, part)
#     return module

# def calculate_covar(model, data_loader, device, epochs=1):
#     conv_fea_dict, linear_fea_dict = get_feature_dict(model)

#     # Allow dropout to be active if needed
#     for md in model.modules():
#         if md.__class__.__name__ == 'Dropout':
#             md.training = True

#     # Setup feature extraction dicts
#     fea_dict = {val: val for val in {**conv_fea_dict, **linear_fea_dict}.values()}
#     tmp_fea_dict = fea_dict.copy()
#     features = {'input': []} if 'input' in fea_dict else {}

#     if 'input' in tmp_fea_dict:
#         tmp_fea_dict.pop('input')

#     fea_ext = create_feature_extractor(model, tmp_fea_dict)

#     # Initialize covariance dictionary
#     covar = {layer: 0 for layer in {**conv_fea_dict, **linear_fea_dict}}

#     with torch.no_grad():
#         for _ in range(epochs):
#             for imgs, _ in tqdm(data_loader, desc="Computing covariance"):
#                 imgs = imgs.to(device)

#                 if 'input' in fea_dict:
#                     features['input'] = imgs

#                 feats = fea_ext(imgs)
#                 for fea_name, val in feats.items():
#                     features[fea_name] = val.detach()

#                 # Process conv layers
#                 for layer in conv_fea_dict:
#                     layer_module = get_module_by_name(model, layer)
#                     ks = layer_module.kernel_size
#                     pad = layer_module.padding

#                     f = features[conv_fea_dict[layer]]
#                     patch = F.unfold(f, ks, dilation=1, padding=pad, stride=1)
#                     patch = patch.permute(0, 2, 1).reshape(-1, patch.shape[1]).double()
#                     covar[layer] += patch.T @ patch

#                 # Process linear layers
#                 for layer in linear_fea_dict:
#                     f = features[linear_fea_dict[layer]].double().squeeze()
#                     if f.ndim == 1:
#                         f = f.unsqueeze(0)
#                     covar[layer] += f.T @ f

#     return covar


# def calculate_svd(covar, k=None, eps=1e-6):
#     svd = {}
#     for layer, C in covar.items():
#         U, S, _ = torch.svd(C + eps * torch.eye(C.size(0), device=C.device))
#         if k is not None:
#             U = U[:, :k]
#             S = S[:k]
#         svd[layer] = {'U': U, 'S': S.sqrt()}
#     return svd


# # def project_onto_subspace(covar_target, svd_basis, k=None):
# #     """
# #     Project target covariances onto subspaces defined by SVDs.

# #     Args:
# #         covar_target (dict): layer -> covariance matrix (original background covars)
# #         svd_basis (dict): layer -> {'U': ..., 'S': ...} (basis from forget/retain SVD)
# #         k (int): top-k components to use

# #     Returns:
# #         projected_covar (dict): layer -> projected covariance matrix
# #     """
# #     projected_covar = {}

# #     for layer in covar_target:
# #         if layer not in svd_basis:
# #             continue

# #         C_target = covar_target[layer]
# #         U = svd_basis[layer]['U']
# #         if k is not None:
# #             U = U[:, :k]

# #         P = U @ U.T
# #         # C_proj = P @ C_target @ P
# #         C_proj = C_target @ P
# #         projected_covar[layer] = C_proj

# #     return projected_covar

# def project_onto_subspace(covar_target, svd_basis, k=None):
#     """
#     Project target covariances onto subspaces defined by SVDs.

#     Args:
#         covar_target (dict): layer -> covariance matrix (original background covars)
#         svd_basis (dict): layer -> {'U': ..., 'S': ...} (basis from forget/retain SVD)
#         k (int): top-k components to use

#     Returns:
#         projected_covar (dict): layer -> projected covariance matrix
#     """
#     projected_covar = {}

#     for layer in covar_target:
#         if layer not in svd_basis:
#             continue

#         C_target = covar_target[layer]
#         U = svd_basis[layer]['U']
#         if k is not None:
#             U = U[:, :k]

#         P = U @ U.T
#         C_proj = C_target @ P  # Right-side projection only: C @ R @ R^T
#         projected_covar[layer] = C_proj

#     return projected_covar


# def calculate_svd_difference(covar_ref, covar_other, device,  k=None):
#     svd_ref = calculate_svd(covar_ref, k)
#     diff_svd = {}

#     for layer in covar_other:
#         if layer not in svd_ref:
#             raise ValueError(f"Layer {layer} not found in reference SVD.")

#         U = svd_ref[layer]['U'].to(device)
#         S = svd_ref[layer]['S'].to(device)
#         M = U @ torch.diag(S ** 2) @ U.T

#         diff = M - covar_other[layer].to(device)
#         U1, S1, _ = torch.svd(diff)
#         diff_svd[layer] = {'U': U1, 'S': S1.sqrt()}
#     return diff_svd


# def subspace_recall(U_true, U_est, k):
#     k = k if k is not None else min(U_true.shape[1], U_est.shape[1])
#     U_true_k = U_true[:, :k]
#     U_est_k = U_est[:, :k]
#     proj = U_est_k @ U_est_k.T  # Projection matrix
#     recall = torch.trace(U_true_k.T @ proj @ U_true_k) / k
#     return recall.item()


# # def compare_svds(svd_estimated: Dict, svd_true: Dict, k: int = 20, plot=True, save_dir="./gradients", id=''):
# #     """
# #     Compare estimated vs. true SVDs per layer using subspace recall@k,
# #     singular values, and cosine angles. Plots all 3 per layer.

# #     Args:
# #         svd_estimated: dict of {'layer': {'U': ..., 'S': ...}}
# #         svd_true: dict of {'layer': {'U': ..., 'S': ...}}
# #         k: top-k subspace dimension
# #         plot: whether to generate plots
# #         save_dir: where to save figures
# #     Returns:
# #         results: dict of per-layer metrics including recall@k
# #     """

# #     os.makedirs(save_dir, exist_ok=True)
# #     results = {}

# #     layers = [layer for layer in svd_estimated if layer in svd_true]
# #     num_layers = len(layers)
# #     cols = 3  # SVD, cos(θ), recall@k
# #     rows = num_layers
# #     fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

# #     for row_idx, layer in enumerate(layers):
# #         S1 = svd_estimated[layer]['S'][:k]
# #         S2 = svd_true[layer]['S'][:k]
# #         U1 = svd_estimated[layer]['U'][:, :k]
# #         U2 = svd_true[layer]['U'][:, :k]

# #         k = k if k is not None else min(U1.shape[1], U2.shape[1])

# #         # Metrics
# #         proj = U1 @ U1.T
# #         recall = torch.trace(U2.T @ proj @ U2) / k
# #         cos_angles = torch.linalg.svdvals(U1.T @ U2)
# #         l2_diff = torch.norm(S1 - S2).item()
# #         mean_overlap = cos_angles.mean().item()

# #         results[layer] = {
# #             'l2_singular_value_diff': l2_diff,
# #             'mean_subspace_overlap': mean_overlap,
# #             'recall@k': recall.item(),
# #             'cos_angles': cos_angles.cpu().numpy()
# #         }

# #         # Left: Singular values
# #         ax_s = axs[row_idx, 0]
# #         ax_s.plot(S1.cpu(), label='Estimated S')
# #         ax_s.plot(S2.cpu(), label='True S')
# #         ax_s.set_title(f"{layer} - SVD (L2: {l2_diff:.2f})")
# #         ax_s.legend()

# #         # Middle: Cosine angles
# #         ax_a = axs[row_idx, 1]
# #         ax_a.plot(cos_angles.cpu())
# #         ax_a.set_title(f"{layer} - cos(θ) (Mean: {mean_overlap:.2f})")
# #         ax_a.set_xlabel("Component")
# #         ax_a.set_ylabel("cos(θ)")

# #         # Right: Recall@k
# #         ax_r = axs[row_idx, 2]
# #         ax_r.bar([0], [recall.item()])
# #         ax_r.set_ylim(0, 1)
# #         ax_r.set_xticks([0])
# #         ax_r.set_xticklabels([f"Recall@{k}"])
# #         ax_r.set_title(f"{layer} - Recall@{k}: {recall.item():.2f}")

# #     if plot:
# #         plt.tight_layout()
# #         save_path = os.path.join(save_dir, f"svd_comparison_{id}.png")
# #         plt.savefig(save_path)
# #         plt.close(fig)

# #     return results

# # def compare_projections(U_dict_1, U_dict_2, k=20, plot=True, save_dir="./gradients", id=''):
# #     """
# #     Compare two subspace projections via cosine angles and recall@k.

# #     Args:
# #         U_dict_1 (dict): layer -> U matrix (e.g., estimated forget)
# #         U_dict_2 (dict): layer -> U matrix (e.g., true forget or retain)
# #         k (int): number of top components to compare
# #         plot (bool): whether to generate plots
# #         save_dir (str): directory to save figures
# #         id (str): identifier for figure name

# #     Returns:
# #         results (dict): per-layer metrics
# #     """
# #     os.makedirs(save_dir, exist_ok=True)
# #     results = {}

# #     layers = [layer for layer in U_dict_1 if layer in U_dict_2]
# #     num_layers = len(layers)
# #     cols, rows = 2, num_layers
# #     fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

# #     for row_idx, layer in enumerate(layers):
# #         U1 = U_dict_1[layer][:, :k]
# #         U2 = U_dict_2[layer][:, :k]
# #         k = k if k is not None else min(U1.shape[1], U2.shape[1])

# #         P1 = U1 @ U1.T
# #         recall = torch.trace(U2.T @ P1 @ U2) / k
# #         cos_angles = torch.linalg.svdvals(U1.T @ U2)
# #         mean_overlap = cos_angles.mean().item()
# #         l2_diff = torch.norm(P1 - (U2 @ U2.T)).item()

# #         results[layer] = {
# #             'recall@k': recall.item(),
# #             'mean_subspace_overlap': mean_overlap,
# #             'l2_singular_value_diff': l2_diff,
# #             'cos_angles': cos_angles.cpu().numpy()
# #         }

# #         axs[row_idx, 0].plot(cos_angles.cpu())
# #         axs[row_idx, 0].set_title(f"{layer} - cos(θ) (Mean: {mean_overlap:.2f})")
# #         axs[row_idx, 0].set_xlabel("Component")
# #         axs[row_idx, 0].set_ylabel("cos(θ)")

# #         axs[row_idx, 1].bar([0], [recall.item()])
# #         axs[row_idx, 1].set_ylim(0, 1)
# #         axs[row_idx, 1].set_xticks([0])
# #         axs[row_idx, 1].set_xticklabels([f"Recall@{k}"])
# #         axs[row_idx, 1].set_title(f"{layer} - Recall@{k}: {recall.item():.2f}")

# #     if plot:
# #         plt.tight_layout()
# #         save_path = os.path.join(save_dir, f"projection_comparison_{id}.png")
# #         plt.savefig(save_path)
# #         plt.close(fig)

# #     return results

# def compare_projected_covariances(proj1, proj2, k=20, plot=True, save_dir="./gradients", id=''):
#     """
#     Compare two sets of projected covariances using Frobenius norm difference and optionally plot results.

#     Args:
#         proj1 (dict): projected covariances from first subspace
#         proj2 (dict): projected covariances from second subspace
#         k (int): number of top entries to summarize (if needed)
#         plot (bool): whether to generate a bar plot
#         save_dir (str): directory to save the plot
#         id (str): identifier string for the plot filename

#     Returns:
#         results (dict): layer -> {'frobenius_diff': ..., 'relative_diff': ...}
#     """
#     import os
#     import matplotlib.pyplot as plt
#     os.makedirs(save_dir, exist_ok=True)

#     results = {}
#     layers = []
#     frob_diffs = []
#     rel_diffs = []

#     for layer in proj1:
#         if layer not in proj2:
#             continue
#         C1 = proj1[layer]
#         C2 = proj2[layer]
#         diff = torch.norm(C1 - C2, p='fro')
#         denom = torch.norm(C1, p='fro') + 1e-8
#         rel_diff = diff / denom
#         results[layer] = {
#             'frobenius_diff': diff.item(),
#             'relative_diff': rel_diff.item()
#         }
#         layers.append(layer)
#         frob_diffs.append(diff.item())
#         rel_diffs.append(rel_diff.item())

#     if plot and layers:
#         fig, axs = plt.subplots(1, 2, figsize=(12, 4))

#         axs[0].bar(layers, frob_diffs)
#         axs[0].set_title("Frobenius Norm Differences")
#         axs[0].set_ylabel("||C1 - C2||_F")
#         axs[0].tick_params(axis='x', rotation=45)

#         axs[1].bar(layers, rel_diffs)
#         axs[1].set_title("Relative Differences")
#         axs[1].set_ylabel("||C1 - C2||_F / ||C1||_F")
#         axs[1].tick_params(axis='x', rotation=45)

#         plt.tight_layout()
#         save_path = os.path.join(save_dir, f"projection_comparison_{id}.png")
#         plt.savefig(save_path)
#         plt.close(fig)

#     return results



# def residual_onto_complement(covar_target, svd_basis, k=None):
#     """
#     Project target covariances onto the orthogonal complement of a subspace.

#     Args:
#         covar_target (dict): layer -> original covariance
#         svd_basis (dict): layer -> {'U': ..., 'S': ...}
#         k (int): use top-k components from basis

#     Returns:
#         residual_covar (dict): layer -> residual covariance (orthogonal component)
#     """
#     residual_covar = {}

#     for layer in covar_target:
#         if layer not in svd_basis:
#             continue

#         C = covar_target[layer]
#         U = svd_basis[layer]['U']
#         if k is not None:
#             U = U[:, :k]

#         P = U @ U.T
#         I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
#         P_comp = I - P
#         C_residual = P_comp @ C @ P_comp

#         residual_covar[layer] = C_residual

#     return residual_covar

# def plot_residual_energy(residual_covar, original_covar, save_dir="./gradients", id="residual"):
#     """
#     Plot Frobenius norm of residual covariances as a fraction of the original.

#     Args:
#         residual_covar (dict): layer -> residual cov matrix
#         original_covar (dict): layer -> original cov matrix
#         save_dir (str): directory to save the plot
#         id (str): filename identifier
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     energy_ratios = {}

#     for layer in residual_covar:
#         if layer in original_covar:
#             res_norm = torch.norm(residual_covar[layer], p='fro')
#             orig_norm = torch.norm(original_covar[layer], p='fro')
#             energy_ratios[layer] = (res_norm / orig_norm).item()

#     # Plot
#     layers = list(energy_ratios.keys())
#     values = [energy_ratios[layer] for layer in layers]

#     plt.figure(figsize=(8, 4))
#     plt.bar(layers, values)
#     plt.xticks(rotation=45, ha='right')
#     plt.ylabel("Residual Energy / Original Energy")
#     plt.title("Fraction of Energy in Orthogonal Complement")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"residual_energy_{id}.png"))
#     plt.close()

#     return energy_ratios
