#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import time
import pickle
import os
from pathlib import Path

from importmodel import ImportModel
from utils import setup_device, initialize_dataloaders

# ==================================================================
# CONFIGURATION
# ==================================================================

# DATA ARGS
DATASET_NAME = 'cifar10'
DATASET_DIR = './data1'

# MODEL ARGS
LOAD_METHOD = 'torchhub'
MODEL_NAME = 'cifar10_resnet20'
NUM_CLASSES = 10
INIT_PATH = 'chenyaofo/pytorch-cifar-models'
ORIGINAL_CKP = './artifacts/unlearn/torchhub_test/unlearn/pgu0/cifar10_resnet20_42_original_001.pt'
UNLEARNED_CKP = './artifacts/unlearn/torchhub_test/unlearn/pgu0/cifar10_resnet20_42_unlearned_001.pt'

# ATTACK SETTINGS
VARIANCE_THRESHOLD = 0.85
CACHE_DIR = './svd_cache'
FORCE_RECOMPUTE = True

# THEORETICAL ANALYSIS SETTINGS
SUCCESS_THRESHOLD = 0.7  # Minimum recovery bound for predicted success
MIN_SEPARATION_ANGLE = 0.5  # Minimum subspace separation (radians)
MIN_SNR = 2.0  # Minimum signal-to-noise ratio

# ==================================================================
# CORE PIPELINE (SIMPLIFIED FROM RCACHE)
# ==================================================================

class ProjectionRecoveryPipeline:
    """
    Simplified pipeline focused on projection-based gradient recovery
    with theoretical analysis and multi-order statistics.
    """
    
    def __init__(self, device='cuda', verbose=True):
        self.device = device
        self.verbose = verbose
        
    def get_target_layers(self, model):
        """Get analyzable conv and linear layers."""
        from torchvision.models.feature_extraction import get_graph_node_names
        
        _, eval_nodes = get_graph_node_names(model)
        conv_layers = OrderedDict()
        linear_layers = OrderedDict()
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and name in eval_nodes:
                conv_layers[name] = name
            elif isinstance(module, torch.nn.Linear) and name in eval_nodes:
                linear_layers[name] = name
        
        if self.verbose:
            print(f"Found {len(conv_layers)} conv layers: {list(conv_layers.keys())}")
            print(f"Found {len(linear_layers)} linear layers: {list(linear_layers.keys())}")
        
        return conv_layers, linear_layers
    
    def get_module_by_name(self, model, layer_name):
        """Helper to get module by dotted name path."""
        parts = layer_name.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def calculate_input_activations(self, model, data_loader, conv_layers, linear_layers, max_batches=None):
        """
        Calculate input activations for each layer.
        Returns raw activations for multi-order analysis.
        """
        if self.verbose:
            print(f"Computing input activations...")
        
        model.eval()
        
        # Storage for activations
        layer_activations = {}
        for layer_name in {**conv_layers, **linear_layers}.keys():
            layer_activations[layer_name] = []
        
        # Hook function to capture inputs
        captured_inputs = {}
        
        def create_input_hook(layer_name):
            def hook_fn(module, input_tensors, output):
                if len(input_tensors) > 0:
                    captured_inputs[layer_name] = input_tensors[0].detach()
            return hook_fn
        
        # Register hooks
        hooks = []
        for layer_name in {**conv_layers, **linear_layers}.keys():
            try:
                module = self.get_module_by_name(model, layer_name)
                hook = module.register_forward_hook(create_input_hook(layer_name))
                hooks.append(hook)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to register hook for {layer_name}: {e}")
                continue
        
        try:
            with torch.no_grad():
                progress_bar = tqdm(data_loader, desc="Computing activations") if self.verbose else data_loader
                
                for batch_idx, (images, _) in enumerate(progress_bar):
                    if max_batches and batch_idx >= max_batches:
                        break
                        
                    images = images.to(self.device)
                    
                    # Forward pass triggers hooks
                    _ = model(images)
                    
                    # Store captured inputs
                    for layer_name in captured_inputs:
                        input_tensor = captured_inputs[layer_name]
                        
                        if layer_name in conv_layers:
                            # For conv layers: unfold input according to kernel
                            module = self.get_module_by_name(model, layer_name)
                            kernel_size = module.kernel_size
                            padding = module.padding
                            
                            patches = F.unfold(input_tensor, kernel_size, 
                                             dilation=1, padding=padding, stride=1)
                            # patches shape: [batch_size, in_channels*kh*kw, num_patches]
                            
                            feature_dim = patches.shape[1]
                            patches_flat = patches.permute(0, 2, 1).reshape(-1, feature_dim)
                            layer_activations[layer_name].append(patches_flat.cpu())
                            
                        elif layer_name in linear_layers:
                            # For linear layers: flatten input if needed
                            if input_tensor.dim() > 2:
                                input_flat = input_tensor.view(input_tensor.shape[0], -1)
                            else:
                                input_flat = input_tensor
                            
                            layer_activations[layer_name].append(input_flat.cpu())
                    
                    # Clear captured inputs for next batch
                    captured_inputs.clear()
        
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()
        
        # Concatenate all batches
        for layer_name in layer_activations:
            if layer_activations[layer_name]:
                layer_activations[layer_name] = torch.cat(layer_activations[layer_name], dim=0)
                if self.verbose:
                    print(f"Layer {layer_name}: activation shape {layer_activations[layer_name].shape}")
        
        return layer_activations
    
    def calculate_multi_order_statistics(self, activations, max_order=4):
        """
        Calculate multi-order statistics (moments/cumulants) beyond covariance.
        """
        if self.verbose:
            print(f"Computing multi-order statistics up to order {max_order}...")
        
        statistics = {}
        
        for layer_name, acts in activations.items():
            if acts is None or len(acts) == 0:
                continue
                
            acts = acts.double()  # Use double precision for numerical stability
            n_samples, n_features = acts.shape
            
            if self.verbose:
                print(f"Layer {layer_name}: {n_samples} samples, {n_features} features")
            
            layer_stats = {}
            
            # Center the data
            mean = acts.mean(dim=0, keepdim=True)
            centered = acts - mean
            
            # Order 1: Mean (should be zero after centering)
            layer_stats[1] = mean.squeeze()
            
            # Order 2: Covariance matrix
            layer_stats[2] = torch.mm(centered.T, centered) / (n_samples - 1)
            
            if max_order >= 3:
                # Order 3: Third-order cumulant (skewness tensor)
                # Simplified: diagonal third moments
                third_moments = torch.mean(centered**3, dim=0)
                layer_stats[3] = third_moments
            
            if max_order >= 4:
                # Order 4: Fourth-order cumulant (kurtosis tensor)
                # Simplified: diagonal fourth moments minus Gaussian part
                fourth_moments = torch.mean(centered**4, dim=0)
                gaussian_part = 3 * torch.diag(layer_stats[2])**2  # For Gaussian: E[X^4] = 3*E[X^2]^2
                layer_stats[4] = fourth_moments - gaussian_part
            
            statistics[layer_name] = layer_stats
            
            if self.verbose:
                print(f"  Order 2 (covariance) shape: {layer_stats[2].shape}")
                if max_order >= 3:
                    print(f"  Order 3 (skewness) shape: {layer_stats[3].shape}")
                if max_order >= 4:
                    print(f"  Order 4 (kurtosis) shape: {layer_stats[4].shape}")
        
        return statistics

# ==================================================================
# THEORETICAL VIABILITY ANALYSIS
# ==================================================================

def estimate_true_subspace(model, data_loader, pipeline, max_batches=10):
    """Estimate the true subspace for a dataset."""
    conv_layers, linear_layers = pipeline.get_target_layers(model)
    activations = pipeline.calculate_input_activations(
        model, data_loader, conv_layers, linear_layers, max_batches
    )
    
    # Calculate covariances and SVD
    subspaces = {}
    for layer_name, acts in activations.items():
        if acts is not None and len(acts) > 0:
            acts = acts.double()
            # Center and compute covariance
            centered = acts - acts.mean(dim=0, keepdim=True)
            cov = torch.mm(centered.T, centered) / (acts.shape[0] - 1)
            
            # SVD to get subspace
            U, S, _ = torch.svd(cov)
            # Keep components with significant variance
            significant = S > 1e-6
            subspaces[layer_name] = U[:, significant]
    
    return subspaces

def subspace_angle(subspace1, subspace2):
    """Compute principal angle between two subspaces."""
    # Compute SVD of U1^T @ U2
    _, S, _ = torch.svd(subspace1.T @ subspace2)
    # Principal angle is arccos of largest singular value
    cos_angle = torch.clamp(S.max(), 0, 1)
    return torch.acos(cos_angle).item()

def estimate_signal_strength(subspace):
    """Estimate signal strength of a subspace."""
    # Signal strength as the sum of squared norms
    return torch.norm(subspace, 'fro').item()**2

def estimate_noise_level(original_model, unlearned_model):
    """Estimate noise level from model differences."""
    total_diff = 0
    total_norm = 0
    
    for (name1, p1), (name2, p2) in zip(original_model.named_parameters(), 
                                        unlearned_model.named_parameters()):
        diff = torch.norm(p1 - p2).item()
        norm = torch.norm(p1).item()
        total_diff += diff**2
        total_norm += norm**2
    
    return total_diff / (total_norm + 1e-8)

def estimate_projection_rank(original_model, unlearned_model, pipeline, background_data):
    """Estimate the rank of the projection used during unlearning."""
    conv_layers, linear_layers = pipeline.get_target_layers(original_model)
    
    # Get activations from both models
    orig_acts = pipeline.calculate_input_activations(
        original_model, background_data, conv_layers, linear_layers, max_batches=5
    )
    unl_acts = pipeline.calculate_input_activations(
        unlearned_model, background_data, conv_layers, linear_layers, max_batches=5
    )
    
    total_rank = 0
    total_dim = 0
    
    for layer_name in orig_acts:
        if layer_name in unl_acts and orig_acts[layer_name] is not None:
            orig = orig_acts[layer_name].double()
            unl = unl_acts[layer_name].double()
            
            # Compute covariance difference
            orig_cov = torch.mm(orig.T, orig) / orig.shape[0]
            unl_cov = torch.mm(unl.T, unl) / unl.shape[0]
            diff_cov = orig_cov - unl_cov
            
            # Estimate rank of difference
            _, S, _ = torch.svd(diff_cov)
            rank = torch.sum(S > 1e-6).item()
            
            total_rank += rank
            total_dim += S.shape[0]
    
    return total_rank / max(total_dim, 1)

def get_ambient_dimension(model):
    """Get total ambient dimension of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def compute_recovery_bound(separation, snr, compression_ratio):
    """
    Compute theoretical recovery bound based on:
    - Subspace separation
    - Signal-to-noise ratio  
    - Compression ratio
    """
    # Heuristic bound: higher separation and SNR = better recovery
    # Lower compression ratio = easier recovery
    separation_factor = np.sin(separation)  # Better when subspaces are orthogonal
    snr_factor = min(snr / 10.0, 1.0)  # Saturate at SNR=10
    compression_factor = 1.0 - compression_ratio  # Better when less compressed
    
    recovery_bound = (separation_factor * snr_factor * compression_factor)
    return recovery_bound

def theoretical_recoverability_analysis(original_model, unlearned_model, 
                                       forget_data, retain_data, background_data,
                                       pipeline):
    """Analyze theoretical conditions for successful recovery."""
    
    print("\n" + "="*60)
    print("THEORETICAL RECOVERABILITY ANALYSIS")
    print("="*60)
    
    # Condition 1: Subspace separation
    print("\n1. 📊 SUBSPACE SEPARATION ANALYSIS")
    print("-" * 40)
    
    forget_subspace = estimate_true_subspace(original_model, forget_data, pipeline)
    retain_subspace = estimate_true_subspace(original_model, retain_data, pipeline)
    
    separations = []
    for layer_name in forget_subspace:
        if layer_name in retain_subspace:
            sep = subspace_angle(forget_subspace[layer_name], retain_subspace[layer_name])
            separations.append(sep)
            print(f"  {layer_name:25s}: separation = {sep:.3f} radians ({sep*180/np.pi:.1f}°)")
    
    mean_separation = np.mean(separations) if separations else 0
    print(f"\nMean subspace separation: {mean_separation:.3f} radians ({mean_separation*180/np.pi:.1f}°)")
    
    # Condition 2: Signal-to-noise ratio
    print("\n2. 📊 SIGNAL-TO-NOISE ANALYSIS")
    print("-" * 40)
    
    forget_signal_strength = np.mean([
        estimate_signal_strength(subspace) for subspace in forget_subspace.values()
    ])
    noise_level = estimate_noise_level(original_model, unlearned_model)
    
    snr = forget_signal_strength / (noise_level + 1e-8)
    print(f"Forget signal strength: {forget_signal_strength:.6f}")
    print(f"Noise level: {noise_level:.6f}")
    print(f"Signal-to-noise ratio: {snr:.3f}")
    
    # Condition 3: Projection rank vs ambient dimension
    print("\n3. 📊 COMPRESSION ANALYSIS")
    print("-" * 40)
    
    projection_rank = estimate_projection_rank(original_model, unlearned_model, pipeline, background_data)
    ambient_dim = get_ambient_dimension(original_model)
    
    compression_ratio = projection_rank
    print(f"Estimated projection rank ratio: {projection_rank:.6f}")
    print(f"Total model parameters: {ambient_dim}")
    print(f"Compression ratio: {compression_ratio:.6f}")
    
    # Theoretical bound
    print("\n4. 📊 RECOVERY BOUND ANALYSIS")
    print("-" * 40)
    
    recovery_bound = compute_recovery_bound(mean_separation, snr, compression_ratio)
    print(f"Theoretical recovery bound: {recovery_bound:.3f}")
    print(f"Success threshold: {SUCCESS_THRESHOLD}")
    
    predicted_success = recovery_bound > SUCCESS_THRESHOLD
    
    # Analysis summary
    print("\n" + "="*60)
    print("VIABILITY ANALYSIS SUMMARY")
    print("="*60)
    
    separation_ok = mean_separation > MIN_SEPARATION_ANGLE
    snr_ok = snr > MIN_SNR
    bound_ok = recovery_bound > SUCCESS_THRESHOLD
    
    print(f"✅ Subspace separation adequate: {separation_ok} ({mean_separation:.3f} > {MIN_SEPARATION_ANGLE})")
    print(f"✅ Signal-to-noise ratio adequate: {snr_ok} ({snr:.3f} > {MIN_SNR})")
    print(f"✅ Recovery bound adequate: {bound_ok} ({recovery_bound:.3f} > {SUCCESS_THRESHOLD})")
    
    overall_viability = separation_ok and snr_ok and bound_ok
    
    if overall_viability:
        print("\n🎯 PREDICTED OUTCOME: ATTACK LIKELY TO SUCCEED")
        print("   All theoretical conditions are met!")
    elif bound_ok:
        print("\n⚠️  PREDICTED OUTCOME: ATTACK MAY PARTIALLY SUCCEED")
        print("   Recovery bound is adequate but some conditions are marginal.")
    else:
        print("\n❌ PREDICTED OUTCOME: ATTACK LIKELY TO FAIL")
        print("   Theoretical conditions are not met.")
        
        # Provide specific recommendations
        print("\n💡 RECOMMENDATIONS:")
        if not separation_ok:
            print("   - Forget and retain subspaces are too similar")
            print("   - Try different data splits with clearer separation")
        if not snr_ok:
            print("   - Signal-to-noise ratio is too low")
            print("   - Check if unlearning actually modified the model significantly")
        if not bound_ok:
            print("   - Overall recovery conditions are not favorable")
            print("   - Consider alternative attack methods")
    
    return {
        'separation': mean_separation,
        'snr': snr, 
        'compression_ratio': compression_ratio,
        'recovery_bound': recovery_bound,
        'predicted_success': predicted_success,
        'separation_adequate': separation_ok,
        'snr_adequate': snr_ok,
        'bound_adequate': bound_ok,
        'overall_viable': overall_viability
    }

# ==================================================================
# MULTI-ORDER STATISTICS RECOVERY
# ==================================================================

def multi_order_subspace_recovery(original_model, unlearned_model, background_data, 
                                 pipeline, max_order=4, variance_threshold=0.85):
    """
    Novel: Use multi-order statistics for more robust subspace recovery.
    """
    
    print("\n" + "="*60)
    print("MULTI-ORDER STATISTICS RECOVERY")
    print("="*60)
    
    conv_layers, linear_layers = pipeline.get_target_layers(original_model)
    
    # Get activations from both models
    print("\n1. Computing activations from both models...")
    orig_acts = pipeline.calculate_input_activations(
        original_model, background_data, conv_layers, linear_layers, max_batches=10
    )
    unl_acts = pipeline.calculate_input_activations(
        unlearned_model, background_data, conv_layers, linear_layers, max_batches=10
    )
    
    # Calculate multi-order statistics
    print("\n2. Computing multi-order statistics...")
    orig_stats = pipeline.calculate_multi_order_statistics(orig_acts, max_order)
    unl_stats = pipeline.calculate_multi_order_statistics(unl_acts, max_order)
    
    # Compute differences across orders
    print("\n3. Analyzing statistical differences...")
    estimated_subspaces = {}
    
    for layer_name in orig_stats:
        if layer_name not in unl_stats:
            continue
            
        layer_results = {}
        
        for order in range(2, max_order + 1):
            if order in orig_stats[layer_name] and order in unl_stats[layer_name]:
                orig_stat = orig_stats[layer_name][order]
                unl_stat = unl_stats[layer_name][order]
                
                if order == 2:  # Covariance matrices
                    stat_diff = orig_stat - unl_stat
                    U, S, _ = torch.svd(stat_diff)
                    
                    # Apply variance threshold
                    cumulative_var = torch.cumsum(S, dim=0) / torch.sum(S)
                    k = torch.sum(cumulative_var <= variance_threshold).item()
                    k = max(1, k)
                    
                    layer_results[order] = {
                        'U': U[:, :k],
                        'S': S[:k],
                        'explained_variance': cumulative_var[k-1].item()
                    }
                    
                else:  # Higher-order moments (simplified analysis)
                    moment_diff = orig_stat - unl_stat
                    # For higher orders, we analyze the magnitude of changes
                    significant_changes = torch.abs(moment_diff) > torch.std(moment_diff)
                    
                    layer_results[order] = {
                        'difference': moment_diff,
                        'significant_indices': significant_changes,
                        'num_significant': significant_changes.sum().item()
                    }
                
                if pipeline.verbose:
                    if order == 2:
                        print(f"  {layer_name} order-{order}: kept {k} components, "
                              f"variance {layer_results[order]['explained_variance']:.3f}")
                    else:
                        print(f"  {layer_name} order-{order}: {layer_results[order]['num_significant']} "
                              f"significant changes")
        
        estimated_subspaces[layer_name] = layer_results
    
    # Combine evidence from multiple orders
    print("\n4. Combining multi-order evidence...")
    combined_subspaces = {}
    
    for layer_name, layer_results in estimated_subspaces.items():
        if 2 in layer_results:  # We need at least covariance
            # Primary subspace from covariance (order 2)
            primary_subspace = layer_results[2]['U']
            
            # Weight by higher-order evidence
            weights = torch.ones(primary_subspace.shape[1])
            
            for order in range(3, max_order + 1):
                if order in layer_results:
                    # Higher-order moments provide additional confidence
                    significance = layer_results[order]['num_significant']
                    total_features = len(layer_results[order]['significant_indices'])
                    confidence_boost = significance / max(total_features, 1)
                    
                    # Boost weights for components with higher-order support
                    weights *= (1 + confidence_boost)
            
            # Normalize weights
            weights = weights / weights.sum()
            
            combined_subspaces[layer_name] = {
                'U': primary_subspace,
                'S': layer_results[2]['S'],
                'weights': weights,
                'multi_order_confidence': weights.mean().item()
            }
            
            if pipeline.verbose:
                print(f"  {layer_name}: multi-order confidence = {weights.mean().item():.3f}")
    
    return combined_subspaces

def verify_subspace_separation(forget_subspaces, retain_subspaces, pipeline):
    """
    Verify that forget and retain subspaces are actually different.
    """
    
    print("\n" + "="*60)
    print("SUBSPACE SEPARATION VERIFICATION")
    print("="*60)
    
    separation_results = {}
    
    for layer_name in forget_subspaces:
        if layer_name in retain_subspaces:
            # Handle both tensor and dictionary formats
            if isinstance(forget_subspaces[layer_name], dict):
                forget_U = forget_subspaces[layer_name]['U']
            else:
                forget_U = forget_subspaces[layer_name]
                
            if isinstance(retain_subspaces[layer_name], dict):
                retain_U = retain_subspaces[layer_name]['U']
            else:
                retain_U = retain_subspaces[layer_name]
            
            # Compute principal angles
            angle = subspace_angle(forget_U, retain_U)
            
            # Compute subspace overlap
            min_dim = min(forget_U.shape[1], retain_U.shape[1])
            forget_trunc = forget_U[:, :min_dim]
            retain_trunc = retain_U[:, :min_dim]
            
            # Handle dimension mismatch
            if forget_U.shape[0] != retain_U.shape[0]:
                min_feat_dim = min(forget_U.shape[0], retain_U.shape[0])
                forget_trunc = forget_trunc[:min_feat_dim, :]
                retain_trunc = retain_trunc[:min_feat_dim, :]
            
            # Compute overlap
            _, S, _ = torch.svd(forget_trunc.T @ retain_trunc)
            overlap = S.mean().item()
            
            separation_results[layer_name] = {
                'angle_radians': angle,
                'angle_degrees': angle * 180 / np.pi,
                'overlap': overlap,
                'separable': angle > MIN_SEPARATION_ANGLE
            }
            
            status = "✅ SEPARABLE" if angle > MIN_SEPARATION_ANGLE else "❌ TOO SIMILAR"
            print(f"  {layer_name:25s}: angle={angle:.3f}rad ({angle*180/np.pi:.1f}°), "
                  f"overlap={overlap:.3f} {status}")
    
    # Summary
    separable_layers = [name for name, result in separation_results.items() if result['separable']]
    total_layers = len(separation_results)
    
    print(f"\nSeparation Summary:")
    print(f"  Separable layers: {len(separable_layers)}/{total_layers}")
    print(f"  Mean angle: {np.mean([r['angle_radians'] for r in separation_results.values()]):.3f} radians")
    print(f"  Mean overlap: {np.mean([r['overlap'] for r in separation_results.values()]):.3f}")
    
    overall_separable = len(separable_layers) >= total_layers * 0.5
    
    if overall_separable:
        print("✅ VERIFICATION PASSED: Subspaces are sufficiently separated")
    else:
        print("❌ VERIFICATION FAILED: Subspaces are too similar")
        print("   Recommendation: Try different forget/retain splits")
    
    return separation_results, overall_separable

# ==================================================================
# MAIN EXECUTION
# ==================================================================

def main():
    """
    Main execution: Test theoretical viability first, then run advanced recovery.
    """
    
    print("🚀 PROJECTION-BASED GRADIENT RECOVERY ATTACK")
    print("="*60)
    
    # Setup
    device = setup_device()
    print(f"Using device: {device}")
    
    # Load models
    print("\n📋 Loading models...")
    original_importer = ImportModel(
        load_method=LOAD_METHOD,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        init_path=INIT_PATH,
        model_ckpt_path=ORIGINAL_CKP
    )
    original_model = original_importer.model
    
    unlearned_importer = ImportModel(
        load_method=LOAD_METHOD,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        init_path=INIT_PATH,
        model_ckpt_path=UNLEARNED_CKP
    )
    unlearned_model = unlearned_importer.model
    
    # Load data
    print("\n📋 Loading datasets...")
    dataloaders = initialize_dataloaders(
        splits=["bset", "forget", "retain"],
        batch_sizes={"bset": 64, "forget": 64, "retain": 64},
        num_workers=1,
        dataset_name=DATASET_NAME,
        dataset_save_dir=DATASET_DIR,
    )
    
    bdata = dataloaders["bset"]
    forgetdata = dataloaders["forget"]
    retaindata = dataloaders["retain"]
    
    # Initialize pipeline
    pipeline = ProjectionRecoveryPipeline(device=device, verbose=True)
    
    # ==================================================================
    # PHASE 1: THEORETICAL VIABILITY ANALYSIS
    # ==================================================================
    
    print("\n🔬 PHASE 1: THEORETICAL VIABILITY ANALYSIS")
    print("="*50)
    
    viability_results = theoretical_recoverability_analysis(
        original_model, unlearned_model, forgetdata, retaindata, bdata, pipeline
    )
    
    if not viability_results['overall_viable']:
        print("\n⚠️  THEORETICAL ANALYSIS SUGGESTS ATTACK MAY FAIL")
        print("   Proceeding anyway for empirical validation...")
    else:
        print("\n✅ THEORETICAL ANALYSIS SUGGESTS ATTACK SHOULD SUCCEED")
    
    # ==================================================================
    # PHASE 2: MULTI-ORDER STATISTICS RECOVERY
    # ==================================================================
    
    print("\n🔬 PHASE 2: MULTI-ORDER STATISTICS RECOVERY")
    print("="*50)
    
    estimated_subspaces = multi_order_subspace_recovery(
        original_model, unlearned_model, bdata, pipeline, 
        max_order=4, variance_threshold=VARIANCE_THRESHOLD
    )
    
    # ==================================================================
    # PHASE 3: GROUND TRUTH VERIFICATION
    # ==================================================================
    
    print("\n🔬 PHASE 3: GROUND TRUTH VERIFICATION")
    print("="*50)
    
    # Get ground truth subspaces
    forget_subspaces = estimate_true_subspace(original_model, forgetdata, pipeline)
    retain_subspaces = estimate_true_subspace(original_model, retaindata, pipeline)
    
    # Verify separation
    separation_results, separation_ok = verify_subspace_separation(
        forget_subspaces, retain_subspaces, pipeline
    )
    
    # Compare estimated vs ground truth
    print("\n📊 COMPARING ESTIMATED VS GROUND TRUTH")
    print("-" * 40)
    
    recovery_scores = {}
    for layer_name in estimated_subspaces:
        if layer_name in forget_subspaces:
            estimated_U = estimated_subspaces[layer_name]['U']
            true_U = forget_subspaces[layer_name]
            
            # Compute recovery accuracy
            angle = subspace_angle(estimated_U, true_U)
            recovery_score = np.cos(angle)  # 1.0 = perfect, 0.0 = orthogonal
            
            recovery_scores[layer_name] = recovery_score
            
            status = "🎯 EXCELLENT" if recovery_score > 0.9 else \
                    "✅ GOOD" if recovery_score > 0.7 else \
                    "⚠️  MODERATE" if recovery_score > 0.5 else "❌ POOR"
            
            print(f"  {layer_name:25s}: recovery_score={recovery_score:.3f} {status}")
    
    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    
    print("\n" + "="*60)
    print("FINAL ATTACK ASSESSMENT")
    print("="*60)
    
    mean_recovery_score = np.mean(list(recovery_scores.values())) if recovery_scores else 0
    
    print(f"📊 Theoretical viability: {'✅ PASS' if viability_results['overall_viable'] else '❌ FAIL'}")
    print(f"📊 Subspace separation: {'✅ PASS' if separation_ok else '❌ FAIL'}")
    print(f"📊 Mean recovery score: {mean_recovery_score:.3f}")
    print(f"📊 Recovery bound: {viability_results['recovery_bound']:.3f}")
    
    # Overall assessment
    empirical_success = mean_recovery_score > 0.7
    theoretical_success = viability_results['overall_viable']
    
    if empirical_success and theoretical_success:
        print("\n🎯 ATTACK SUCCESSFUL!")
        print("   Both theoretical and empirical results are positive.")
    elif empirical_success:
        print("\n✅ ATTACK SUCCEEDED EMPIRICALLY!")
        print("   Despite theoretical concerns, empirical results are good.")
    elif theoretical_success:
        print("\n⚠️  MIXED RESULTS!")
        print("   Theory suggests success but empirical results are poor.")
    else:
        print("\n❌ ATTACK FAILED!")
        print("   Both theoretical and empirical results suggest failure.")
    
    # Research implications
    print("\n💡 RESEARCH IMPLICATIONS:")
    if not viability_results['separation_adequate']:
        print("   - Subspace separation is a critical factor")
        print("   - Consider data splits with clearer semantic differences")
    if not viability_results['snr_adequate']:
        print("   - Signal-to-noise ratio limits recovery")
        print("   - Stronger unlearning may be needed for detectability")
    if empirical_success != theoretical_success:
        print("   - Theory-practice gap suggests model refinement needed")
    
    return {
        'viability_results': viability_results,
        'estimated_subspaces': estimated_subspaces,
        'recovery_scores': recovery_scores,
        'separation_results': separation_results,
        'empirical_success': empirical_success,
        'theoretical_success': theoretical_success
    }

if __name__ == "__main__":
    results = main()
    print("\n✅ Analysis completed!")