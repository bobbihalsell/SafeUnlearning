import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from collections import OrderedDict
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

class GradientRecoveryPipeline:
    """
    Clean implementation of gradient recovery pipeline for machine unlearning attacks.
    
    The key insight: We need to capture INPUT activations to layers, not outputs,
    because gradients are computed with respect to inputs.
    """
    
    def __init__(self, device='cuda', cache_dir='./cache', verbose=True):
        self.device = device
        self.cache_dir = cache_dir
        self.verbose = verbose
        
    def get_target_layers(self, model):
        """
        Get conv and linear layers that we can meaningfully analyze.
        Excludes BatchNorm, Dropout, etc.
        """
        try:
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
        except Exception as e:
            print(f"Error in get_target_layers: {e}")
            raise
    
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
    
    def calculate_input_covariances(self, model, data_loader, conv_layers, linear_layers, epochs=1):
        """
        Calculate covariance matrices using INPUT activations to each layer.
        This is the key fix - we capture inputs, not outputs.
        """
        if self.verbose:
            print(f"Calculating input covariances with {len(data_loader)} batches, {epochs} epochs")
        
        model.eval()
        
        # Initialize covariance accumulators
        covariances = {}
        for layer_name in {**conv_layers, **linear_layers}.keys():
            covariances[layer_name] = None
        
        # Hook function to capture inputs
        captured_inputs = {}
        
        def create_input_hook(layer_name):
            def hook_fn(module, input_tensors, output):
                # input_tensors is a tuple, we want the first element
                if len(input_tensors) > 0:
                    captured_inputs[layer_name] = input_tensors[0].detach()
            return hook_fn
        
        # Register hooks for all target layers
        hooks = []
        for layer_name in {**conv_layers, **linear_layers}.keys():
            try:
                module = self.get_module_by_name(model, layer_name)
                hook = module.register_forward_hook(create_input_hook(layer_name))
                hooks.append(hook)
            except Exception as e:
                print(f"Failed to register hook for {layer_name}: {e}")
                continue
        
        if self.verbose:
            print(f"Registered {len(hooks)} input capture hooks")
        
        try:
            with torch.no_grad():
                for epoch in range(epochs):
                    if self.verbose and epochs > 1:
                        print(f"Epoch {epoch + 1}/{epochs}")
                    
                    for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc="Computing covariances")):
                        images = images.to(self.device)
                        
                        # Forward pass triggers hooks
                        _ = model(images)
                        
                        # Process captured inputs
                        for layer_name in captured_inputs:
                            input_tensor = captured_inputs[layer_name]
                            
                            if layer_name in conv_layers:
                                # For conv layers: unfold input according to kernel
                                module = self.get_module_by_name(model, layer_name)
                                kernel_size = module.kernel_size
                                padding = module.padding
                                
                                # Unfold creates patches that match what conv sees
                                patches = F.unfold(input_tensor, kernel_size, 
                                                 dilation=1, padding=padding, stride=1)
                                # patches shape: [batch_size, in_channels*kh*kw, num_patches]
                                
                                feature_dim = patches.shape[1]
                                patches_flat = patches.permute(0, 2, 1).reshape(-1, feature_dim).double()
                                
                                # Accumulate covariance: X^T X
                                batch_cov = torch.mm(patches_flat.T, patches_flat)
                                
                            elif layer_name in linear_layers:
                                # For linear layers: flatten input if needed
                                if input_tensor.dim() > 2:
                                    input_flat = input_tensor.view(input_tensor.shape[0], -1)
                                else:
                                    input_flat = input_tensor
                                
                                input_flat = input_flat.double()
                                batch_cov = torch.mm(input_flat.T, input_flat)
                            
                            # Accumulate covariance
                            if covariances[layer_name] is None:
                                covariances[layer_name] = batch_cov
                            else:
                                covariances[layer_name] += batch_cov
                        
                        # Clear captured inputs for next batch
                        captured_inputs.clear()
        
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()
        
        # Normalize covariances
        total_samples = len(data_loader.dataset) * epochs
        for layer_name in covariances:
            if covariances[layer_name] is not None:
                covariances[layer_name] = covariances[layer_name] / epochs
                if self.verbose:
                    print(f"Layer {layer_name}: covariance shape {covariances[layer_name].shape}")
        
        return covariances
    
    def calculate_svd(self, covariance_dict, k=None, variance_threshold=None, eps=1e-6):
        """
        Calculate SVD decomposition of covariance matrices.
        
        Args:
            covariance_dict: Dictionary of covariance matrices
            k: Number of components to keep (if None, use variance_threshold)
            variance_threshold: Keep components explaining this much variance
            eps: Numerical stability epsilon
        """
        svd_results = {}
        
        if self.verbose:
            print("Computing SVD decompositions...")
        
        for layer_name, cov_matrix in covariance_dict.items():
            if cov_matrix is None:
                continue
                
            try:
                # Add small epsilon for numerical stability
                stabilized = cov_matrix + eps * torch.eye(cov_matrix.shape[0], 
                                                        device=cov_matrix.device, 
                                                        dtype=cov_matrix.dtype)
                
                # SVD decomposition
                U, S, _ = torch.svd(stabilized)
                
                # Determine how many components to keep
                if k is not None:
                    components_to_keep = min(k, len(S))
                elif variance_threshold is not None:
                    cumulative_var = torch.cumsum(S, dim=0) / torch.sum(S)
                    components_to_keep = torch.sum(cumulative_var <= variance_threshold).item()
                    components_to_keep = max(1, components_to_keep)  # Keep at least 1
                else:
                    components_to_keep = len(S)
                
                # Truncate
                U_truncated = U[:, :components_to_keep]
                S_truncated = S[:components_to_keep]
                
                svd_results[layer_name] = {
                    'U': U_truncated,
                    'S': S_truncated,
                    'explained_variance_ratio': cumulative_var[components_to_keep-1].item() if variance_threshold else None
                }
                
                if self.verbose:
                    var_explained = (S_truncated.sum() / S.sum()).item()
                    print(f"Layer {layer_name}: kept {components_to_keep}/{len(S)} components, "
                          f"explaining {var_explained:.3f} variance")
                    
            except Exception as e:
                print(f"SVD failed for layer {layer_name}: {e}")
                continue
        
        return svd_results
    
    def estimate_forget_subspace(self, original_cov, unlearned_cov, variance_threshold=0.95):
        """
        Estimate the forget subspace by analyzing the difference between 
        original and unlearned covariances.
        
        This implements the core insight: the difference should reveal 
        what was "subtracted" during unlearning.
        """
        if self.verbose:
            print("Estimating forget subspace from covariance differences...")
        
        # First, get full SVD of original data
        original_svd = self.calculate_svd(original_cov, variance_threshold=None)
        
        estimated_forget_svd = {}
        
        for layer_name in original_cov:
            if layer_name not in unlearned_cov or layer_name not in original_svd:
                continue
                
            try:
                # Get original subspace
                U_orig = original_svd[layer_name]['U'].to(self.device)
                S_orig = original_svd[layer_name]['S'].to(self.device)
                
                # Reconstruct original covariance matrix
                original_reconstructed = torch.mm(torch.mm(U_orig, torch.diag(S_orig**2)), U_orig.T)
                
                # Subtract unlearned covariance to get "difference"
                difference_matrix = original_reconstructed - unlearned_cov[layer_name].to(self.device)
                
                # SVD of the difference should reveal forget subspace
                U_diff, S_diff, _ = torch.svd(difference_matrix)
                
                # Apply variance threshold
                cumulative_var = torch.cumsum(S_diff, dim=0) / torch.sum(S_diff)
                k = torch.sum(cumulative_var <= variance_threshold).item()
                k = max(1, k)  # Keep at least 1 component
                
                estimated_forget_svd[layer_name] = {
                    'U': U_diff[:, :k],
                    'S': S_diff[:k],
                    'explained_variance_ratio': cumulative_var[k-1].item()
                }
                
                if self.verbose:
                    print(f"Layer {layer_name}: estimated forget subspace has {k} components")
                    
            except Exception as e:
                print(f"Error estimating forget subspace for {layer_name}: {e}")
                continue
        
        return estimated_forget_svd
    
    def create_projection_matrices(self, svd_results):
        """
        Create projection matrices P = U U^T from SVD results.
        """
        projection_matrices = {}
        
        for layer_name, svd_data in svd_results.items():
            U = svd_data['U']
            P = torch.mm(U, U.T).to(self.device).float()
            projection_matrices[layer_name] = P
            
            if self.verbose:
                print(f"Layer {layer_name}: projection matrix shape {P.shape}")
        
        return projection_matrices
    
    def get_gradients(self, model, data_loader, loss_fn=None, max_batches=None):
        """
        Calculate gradients for the model on given data.
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        model.train()
        gradients = {}
        total_samples = 0
        
        if self.verbose:
            print("Computing gradients...")
        
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Computing gradients")):
            if max_batches and batch_idx >= max_batches:
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            model.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # Accumulate gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in gradients:
                        gradients[name] = param.grad.clone()
                    else:
                        gradients[name] += param.grad
            
            total_samples += images.shape[0]
        
        # Average gradients
        for name in gradients:
            gradients[name] = gradients[name] / total_samples
        
        return gradients
    
    def project_gradients(self, gradients, projection_matrices, conv_layers, linear_layers):
        """
        Apply projection matrices to gradients.
        This simulates the projection that was applied during unlearning.
        """
        projected_gradients = {}
        
        for param_name, grad in gradients.items():
            if grad is None:
                projected_gradients[param_name] = None
                continue
            
            # Extract layer name from parameter name
            if '.weight' in param_name:
                layer_name = param_name.replace('.weight', '')
            elif '.bias' in param_name:
                layer_name = param_name.replace('.bias', '')
                # Skip bias terms for now
                projected_gradients[param_name] = grad.clone()
                continue
            else:
                layer_name = param_name
            
            # Skip if no projection matrix available
            if layer_name not in projection_matrices:
                projected_gradients[param_name] = grad.clone()
                continue
            
            proj_matrix = projection_matrices[layer_name]
            
            try:
                if len(grad.shape) == 4:  # Conv weights [out_ch, in_ch, h, w]
                    out_channels = grad.shape[0]
                    grad_flat = grad.view(out_channels, -1)  # [out_ch, in_ch*h*w]
                    
                    if grad_flat.shape[1] == proj_matrix.shape[0]:
                        # Apply projection: original_grad - projected_component
                        projected_component = torch.mm(grad_flat, proj_matrix)
                        projected_grad = grad - projected_component.view(grad.shape)
                    else:
                        print(f"Dimension mismatch for {layer_name}: {grad_flat.shape[1]} vs {proj_matrix.shape[0]}")
                        projected_grad = grad.clone()
                
                elif len(grad.shape) == 2:  # Linear weights [out_feat, in_feat]
                    out_features = grad.shape[0]
                    grad_flat = grad.view(out_features, -1)
                    
                    if grad_flat.shape[1] == proj_matrix.shape[0]:
                        projected_component = torch.mm(grad_flat, proj_matrix)
                        projected_grad = grad - projected_component.view(grad.shape)
                    else:
                        print(f"Dimension mismatch for {layer_name}: {grad_flat.shape[1]} vs {proj_matrix.shape[0]}")
                        projected_grad = grad.clone()
                
                else:
                    projected_grad = grad.clone()
                
                projected_gradients[param_name] = projected_grad
                
            except Exception as e:
                print(f"Projection failed for {param_name}: {e}")
                projected_gradients[param_name] = grad.clone()
        
        return projected_gradients
    
    def compare_subspaces(self, svd1, svd2, name1="SVD1", name2="SVD2"):
        """
        Compare two sets of SVD results to measure subspace similarity.
        """
        comparison_results = {}
        
        common_layers = set(svd1.keys()) & set(svd2.keys())
        
        for layer_name in common_layers:
            U1 = svd1[layer_name]['U']
            U2 = svd2[layer_name]['U']
            
            # Ensure same device
            U1 = U1.to(self.device)
            U2 = U2.to(self.device)
            
            # Principal angles between subspaces
            try:
                # Compute overlap matrix
                min_dim = min(U1.shape[1], U2.shape[1])
                U1_trunc = U1[:, :min_dim]
                U2_trunc = U2[:, :min_dim]
                
                # Handle dimension mismatch
                if U1.shape[0] != U2.shape[0]:
                    min_feat_dim = min(U1.shape[0], U2.shape[0])
                    U1_trunc = U1_trunc[:min_feat_dim, :]
                    U2_trunc = U2_trunc[:min_feat_dim, :]
                
                # Compute principal angles
                _, S, _ = torch.svd(U1_trunc.T @ U2_trunc)
                cos_angles = torch.clamp(S, 0, 1)  # Clamp for numerical stability
                
                # Metrics
                mean_cosine = cos_angles.mean().item()
                max_cosine = cos_angles.max().item()
                min_cosine = cos_angles.min().item()
                
                # Subspace overlap (trace of projection)
                P1 = U1_trunc @ U1_trunc.T
                P2 = U2_trunc @ U2_trunc.T
                overlap = torch.trace(P1 @ P2).item() / min_dim
                
                comparison_results[layer_name] = {
                    'mean_cosine': mean_cosine,
                    'max_cosine': max_cosine,
                    'min_cosine': min_cosine,
                    'subspace_overlap': overlap,
                    'mean_angle_degrees': torch.acos(cos_angles.mean()).item() * 180 / np.pi
                }
                
            except Exception as e:
                print(f"Comparison failed for {layer_name}: {e}")
                comparison_results[layer_name] = None
        
        if self.verbose:
            print(f"\nSubspace comparison: {name1} vs {name2}")
            for layer_name, results in comparison_results.items():
                if results:
                    print(f"  {layer_name}: mean_cosine={results['mean_cosine']:.3f}, "
                          f"overlap={results['subspace_overlap']:.3f}, "
                          f"angle={results['mean_angle_degrees']:.1f}°")
        
        return comparison_results
    
    def run_attack(self, original_model, unlearned_model, background_data, 
                   forget_data=None, retain_data=None, variance_threshold=0.95):
        """
        Run the complete gradient recovery attack.
        
        Args:
            original_model: Model trained on full data
            unlearned_model: Model after unlearning
            background_data: DataLoader for background/auxiliary data
            forget_data: DataLoader for forget set (for verification only)
            retain_data: DataLoader for retain set (for verification only)
            variance_threshold: Variance threshold for subspace selection
        """
        print("="*60)
        print("RUNNING GRADIENT RECOVERY ATTACK")
        print("="*60)
        
        # Step 1: Get target layers
        conv_layers, linear_layers = self.get_target_layers(original_model)
        
        # Step 2: Calculate covariances
        print("\nStep 1: Computing covariances...")
        original_cov = self.calculate_input_covariances(
            original_model, background_data, conv_layers, linear_layers
        )
        
        unlearned_cov = self.calculate_input_covariances(
            unlearned_model, background_data, conv_layers, linear_layers
        )
        
        # Step 3: Estimate forget subspace
        print("\nStep 2: Estimating forget subspace...")
        estimated_forget_svd = self.estimate_forget_subspace(
            original_cov, unlearned_cov, variance_threshold
        )
        
        # Step 4: Create projection matrices
        print("\nStep 3: Creating projection matrices...")
        estimated_proj_matrices = self.create_projection_matrices(estimated_forget_svd)
        
        # Step 5: Test projection on gradients
        print("\nStep 4: Testing gradient projection...")
        gradients = self.get_gradients(original_model, background_data, max_batches=5)
        projected_gradients = self.project_gradients(
            gradients, estimated_proj_matrices, conv_layers, linear_layers
        )
        
        results = {
            'estimated_forget_svd': estimated_forget_svd,
            'projection_matrices': estimated_proj_matrices,
            'original_gradients': gradients,
            'projected_gradients': projected_gradients
        }
        
        # Step 6: Verification (if ground truth data available)
        if forget_data is not None:
            print("\nStep 5: Verification with ground truth...")
            
            # Calculate true forget subspace
            forget_cov = self.calculate_input_covariances(
                original_model, forget_data, conv_layers, linear_layers
            )
            true_forget_svd = self.calculate_svd(forget_cov, variance_threshold=variance_threshold)
            
            # Compare estimated vs true forget subspaces
            forget_comparison = self.compare_subspaces(
                estimated_forget_svd, true_forget_svd, 
                "Estimated Forget", "True Forget"
            )
            results['forget_comparison'] = forget_comparison
            
            if retain_data is not None:
                # Calculate retain subspace for contrast
                retain_cov = self.calculate_input_covariances(
                    original_model, retain_data, conv_layers, linear_layers
                )
                retain_svd = self.calculate_svd(retain_cov, variance_threshold=variance_threshold)
                
                # Compare estimated forget vs retain (should be different)
                retain_comparison = self.compare_subspaces(
                    estimated_forget_svd, retain_svd,
                    "Estimated Forget", "True Retain"
                )
                results['retain_comparison'] = retain_comparison
        
        print("\n" + "="*60)
        print("ATTACK COMPLETED")
        print("="*60)
        
        return results

# Example usage function
def run_gradient_recovery_attack(original_model, unlearned_model, background_data, 
                                forget_data=None, retain_data=None, 
                                device='cuda', variance_threshold=0.95):
    """
    Convenient function to run the complete attack.
    """
    pipeline = GradientRecoveryPipeline(device=device, verbose=True)
    
    results = pipeline.run_attack(
        original_model=original_model,
        unlearned_model=unlearned_model,
        background_data=background_data,
        forget_data=forget_data,
        retain_data=retain_data,
        variance_threshold=variance_threshold
    )
    
    return results, pipeline