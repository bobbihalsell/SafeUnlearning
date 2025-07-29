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
import pickle
import os
import hashlib
import json
from pathlib import Path

class CachedGradientRecoveryPipeline:
    """
    Gradient recovery pipeline with comprehensive caching for expensive operations.
    
    The key insight: We need to capture INPUT activations to layers, not outputs,
    because gradients are computed with respect to inputs.
    """
    
    def __init__(self, device='cuda', cache_dir='./gradient_recovery_cache', 
                 force_recompute=False, verbose=True):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.force_recompute = force_recompute
        self.verbose = verbose
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"Initialized pipeline with cache dir: {self.cache_dir}")
            print(f"Force recompute: {self.force_recompute}")
    
    def _get_model_hash(self, model):
        """
        Generate a hash representing the model architecture and state.
        """
        # Get model architecture signature
        arch_str = str(model)
        
        # Get parameter shapes (faster than hashing all parameters)
        param_shapes = []
        for name, param in model.named_parameters():
            param_shapes.append(f"{name}:{param.shape}")
        
        # Combine architecture and parameter info
        model_signature = arch_str + "|" + "|".join(param_shapes)
        
        # Create hash
        hash_obj = hashlib.md5(model_signature.encode())
        return hash_obj.hexdigest()[:12]  # 12 chars should be enough
    
    def _get_data_hash(self, data_loader, max_batches_to_hash=3):
        """
        Generate a hash representing the dataset characteristics.
        """
        try:
            # Get basic properties
            batch_size = data_loader.batch_size
            dataset_len = len(data_loader.dataset)
            
            # Sample a few batches to get data characteristics
            data_samples = []
            with torch.no_grad():
                for i, (data, targets) in enumerate(data_loader):
                    if i >= max_batches_to_hash:
                        break
                    # Hash shape and a few sample values
                    data_info = f"shape:{data.shape}|mean:{data.mean().item():.6f}|std:{data.std().item():.6f}"
                    target_info = f"targets:{targets[:min(10, len(targets))].tolist()}"
                    data_samples.append(data_info + "|" + target_info)
            
            # Combine all info
            data_signature = f"len:{dataset_len}|batch:{batch_size}|samples:" + "|".join(data_samples)
            
            hash_obj = hashlib.md5(data_signature.encode())
            return hash_obj.hexdigest()[:12]
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not hash dataset, using timestamp: {e}")
            return str(int(time.time()))[-8:]
    
    def _get_cache_key(self, operation, model=None, data_loader=None, **kwargs):
        """
        Generate a unique cache key for an operation.
        """
        key_parts = [operation]
        
        if model is not None:
            key_parts.append(f"model_{self._get_model_hash(model)}")
        
        if data_loader is not None:
            key_parts.append(f"data_{self._get_data_hash(data_loader)}")
        
        # Add other parameters
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (int, float, str, bool)):
                key_parts.append(f"{k}_{v}")
            elif isinstance(v, (list, tuple)):
                key_parts.append(f"{k}_{'_'.join(map(str, v))}")
        
        return "_".join(key_parts)
    
    def _get_cache_path(self, cache_key):
        """Get the full path for a cache file."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _save_to_cache(self, data, cache_key):
        """Save data to cache with metadata."""
        cache_path = self._get_cache_path(cache_key)
        
        # Create metadata
        metadata = {
            'timestamp': time.time(),
            'cache_key': cache_key,
            'data_type': type(data).__name__
        }
        
        cache_data = {
            'metadata': metadata,
            'data': data
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            if self.verbose:
                print(f"💾 Saved to cache: {cache_path.name}")
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to save cache {cache_path}: {e}")
    
    def _load_from_cache(self, cache_key):
        """Load data from cache if it exists and is valid."""
        if self.force_recompute:
            return None
            
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Extract data and metadata
            data = cache_data['data']
            metadata = cache_data['metadata']
            
            if self.verbose:
                age_hours = (time.time() - metadata['timestamp']) / 3600
                print(f"📁 Loaded from cache: {cache_path.name} (age: {age_hours:.1f}h)")
            
            return data
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to load cache {cache_path}: {e}")
            return None
    
    def get_target_layers(self, model):
        """
        Get conv and linear layers that we can meaningfully analyze.
        Cached based on model architecture.
        """
        cache_key = self._get_cache_key("target_layers", model=model)
        cached_result = self._load_from_cache(cache_key)
        
        if cached_result is not None:
            return cached_result['conv_layers'], cached_result['linear_layers']
        
        try:
            _, eval_nodes = get_graph_node_names(model)
            conv_layers = OrderedDict()
            linear_layers = OrderedDict()
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) and name in eval_nodes:
                    conv_layers[name] = name
                elif isinstance(module, torch.nn.Linear) and name in eval_nodes:
                    linear_layers[name] = name
            
            result = {
                'conv_layers': conv_layers,
                'linear_layers': linear_layers
            }
            
            self._save_to_cache(result, cache_key)
            
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
        This is heavily cached since it's the most expensive operation.
        """
        cache_key = self._get_cache_key(
            "input_covariances", 
            model=model, 
            data_loader=data_loader,
            epochs=epochs,
            conv_layers=len(conv_layers),
            linear_layers=len(linear_layers)
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print(f"🔄 Computing input covariances (this will be cached)...")
            print(f"   {len(data_loader)} batches, {epochs} epochs")
        
        model.eval()
        
        # Initialize covariance accumulators
        covariances = {}
        for layer_name in {**conv_layers, **linear_layers}.keys():
            covariances[layer_name] = None
        
        # Hook function to capture inputs
        captured_inputs = {}
        
        def create_input_hook(layer_name):
            def hook_fn(module, input_tensors, output):
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
                if self.verbose:
                    print(f"Failed to register hook for {layer_name}: {e}")
                continue
        
        if self.verbose:
            print(f"Registered {len(hooks)} input capture hooks")
        
        total_samples = 0
        
        try:
            with torch.no_grad():
                for epoch in range(epochs):
                    if self.verbose and epochs > 1:
                        print(f"Epoch {epoch + 1}/{epochs}")
                    
                    progress_bar = tqdm(data_loader, desc="Computing covariances") if self.verbose else data_loader
                    
                    for batch_idx, (images, _) in enumerate(progress_bar):
                        images = images.to(self.device)
                        batch_size = images.shape[0]
                        
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
                        
                        total_samples += batch_size
                        
                        # Clear captured inputs for next batch
                        captured_inputs.clear()
        
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()
        
        # Normalize covariances
        for layer_name in covariances:
            if covariances[layer_name] is not None:
                covariances[layer_name] = covariances[layer_name] / epochs
                if self.verbose:
                    print(f"Layer {layer_name}: covariance shape {covariances[layer_name].shape}")
        
        # Cache the result
        self._save_to_cache(covariances, cache_key)
        
        return covariances
    
    def calculate_svd(self, covariance_dict, k=None, variance_threshold=None, eps=1e-6):
        """
        Calculate SVD decomposition of covariance matrices.
        Cached based on covariance content and parameters.
        """
        # Create a hash of the covariance matrices for caching
        cov_info = []
        for layer_name, cov_matrix in covariance_dict.items():
            if cov_matrix is not None:
                # Use shape and some statistics as a signature
                cov_signature = f"{layer_name}:{cov_matrix.shape}:{cov_matrix.mean().item():.6f}:{cov_matrix.std().item():.6f}"
                cov_info.append(cov_signature)
        
        cache_key = self._get_cache_key(
            "svd",
            covariances_hash=hashlib.md5("|".join(cov_info).encode()).hexdigest()[:12],
            k=k,
            variance_threshold=variance_threshold,
            eps=eps
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print("🔄 Computing SVD decompositions (this will be cached)...")
        
        svd_results = {}
        
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
                    'explained_variance_ratio': (S_truncated.sum() / S.sum()).item()
                }
                
                if self.verbose:
                    var_explained = (S_truncated.sum() / S.sum()).item()
                    print(f"Layer {layer_name}: kept {components_to_keep}/{len(S)} components, "
                          f"explaining {var_explained:.3f} variance")
                    
            except Exception as e:
                if self.verbose:
                    print(f"SVD failed for layer {layer_name}: {e}")
                continue
        
        # Cache the result
        self._save_to_cache(svd_results, cache_key)
        
        return svd_results
    
    def estimate_forget_subspace(self, original_cov, unlearned_cov, variance_threshold=0.95):
        """
        Estimate the forget subspace by analyzing the difference between 
        original and unlearned covariances. Cached based on input covariances.
        """
        # Create cache key based on input covariances
        orig_info = []
        unl_info = []
        
        for layer_name in original_cov:
            if original_cov[layer_name] is not None:
                cov = original_cov[layer_name]
                orig_info.append(f"{layer_name}:{cov.shape}:{cov.mean().item():.6f}")
        
        for layer_name in unlearned_cov:
            if unlearned_cov[layer_name] is not None:
                cov = unlearned_cov[layer_name]
                unl_info.append(f"{layer_name}:{cov.shape}:{cov.mean().item():.6f}")
        
        cache_key = self._get_cache_key(
            "estimate_forget_subspace",
            original_hash=hashlib.md5("|".join(orig_info).encode()).hexdigest()[:12],
            unlearned_hash=hashlib.md5("|".join(unl_info).encode()).hexdigest()[:12],
            variance_threshold=variance_threshold
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print("🔄 Estimating forget subspace from covariance differences (this will be cached)...")
        
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
                if self.verbose:
                    print(f"Error estimating forget subspace for {layer_name}: {e}")
                continue
        
        # Cache the result
        self._save_to_cache(estimated_forget_svd, cache_key)
        
        return estimated_forget_svd
    
    def create_projection_matrices(self, svd_results):
        """
        Create projection matrices P = U U^T from SVD results.
        Fast operation, but cached for consistency.
        """
        # Create cache key based on SVD results
        svd_info = []
        for layer_name, svd_data in svd_results.items():
            U = svd_data['U']
            svd_info.append(f"{layer_name}:{U.shape}:{U.mean().item():.6f}")
        
        cache_key = self._get_cache_key(
            "projection_matrices",
            svd_hash=hashlib.md5("|".join(svd_info).encode()).hexdigest()[:12]
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print("🔄 Creating projection matrices (this will be cached)...")
        
        projection_matrices = {}
        
        for layer_name, svd_data in svd_results.items():
            U = svd_data['U']
            P = torch.mm(U, U.T).to(self.device).float()
            projection_matrices[layer_name] = P
            
            if self.verbose:
                print(f"Layer {layer_name}: projection matrix shape {P.shape}")
        
        # Cache the result
        self._save_to_cache(projection_matrices, cache_key)
        
        return projection_matrices
    
    def get_gradients(self, model, data_loader, loss_fn=None, max_batches=None):
        """
        Calculate gradients for the model on given data.
        Cached based on model and data characteristics.
        """
        cache_key = self._get_cache_key(
            "gradients",
            model=model,
            data_loader=data_loader,
            max_batches=max_batches,
            loss_fn=type(loss_fn).__name__ if loss_fn else "CrossEntropyLoss"
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print(f"🔄 Computing gradients (this will be cached)...")
            if max_batches:
                print(f"   Limited to {max_batches} batches")
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        model.train()
        gradients = {}
        total_samples = 0
        
        progress_bar = tqdm(data_loader, desc="Computing gradients") if self.verbose else data_loader
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
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
        
        # Cache the result
        self._save_to_cache(gradients, cache_key)
        
        return gradients
    
    def project_gradients(self, gradients, projection_matrices, conv_layers, linear_layers):
        """
        Apply projection matrices to gradients.
        This simulates the projection that was applied during unlearning.
        Fast operation, cached based on inputs.
        """
        # Create cache keys for inputs
        grad_info = []
        for name, grad in gradients.items():
            if grad is not None:
                grad_info.append(f"{name}:{grad.shape}:{grad.mean().item():.6f}")
        
        proj_info = []
        for name, proj in projection_matrices.items():
            proj_info.append(f"{name}:{proj.shape}:{proj.mean().item():.6f}")
        
        cache_key = self._get_cache_key(
            "project_gradients",
            gradients_hash=hashlib.md5("|".join(grad_info).encode()).hexdigest()[:12],
            projections_hash=hashlib.md5("|".join(proj_info).encode()).hexdigest()[:12]
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print("🔄 Projecting gradients (this will be cached)...")
        
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
                        if self.verbose:
                            print(f"Dimension mismatch for {layer_name}: {grad_flat.shape[1]} vs {proj_matrix.shape[0]}")
                        projected_grad = grad.clone()
                
                elif len(grad.shape) == 2:  # Linear weights [out_feat, in_feat]
                    out_features = grad.shape[0]
                    grad_flat = grad.view(out_features, -1)
                    
                    if grad_flat.shape[1] == proj_matrix.shape[0]:
                        projected_component = torch.mm(grad_flat, proj_matrix)
                        projected_grad = grad - projected_component.view(grad.shape)
                    else:
                        if self.verbose:
                            print(f"Dimension mismatch for {layer_name}: {grad_flat.shape[1]} vs {proj_matrix.shape[0]}")
                        projected_grad = grad.clone()
                
                else:
                    projected_grad = grad.clone()
                
                projected_gradients[param_name] = projected_grad
                
            except Exception as e:
                if self.verbose:
                    print(f"Projection failed for {param_name}: {e}")
                projected_gradients[param_name] = grad.clone()
        
        # Cache the result
        self._save_to_cache(projected_gradients, cache_key)
        
        return projected_gradients
    
    def compare_subspaces(self, svd1, svd2, name1="SVD1", name2="SVD2"):
        """
        Compare two sets of SVD results to measure subspace similarity.
        Cached based on input SVD results.
        """
        # Create cache key
        svd1_info = []
        for layer_name, svd_data in svd1.items():
            U = svd_data['U']
            svd1_info.append(f"{layer_name}:{U.shape}:{U.mean().item():.6f}")
        
        svd2_info = []
        for layer_name, svd_data in svd2.items():
            U = svd_data['U']
            svd2_info.append(f"{layer_name}:{U.shape}:{U.mean().item():.6f}")
        
        cache_key = self._get_cache_key(
            "compare_subspaces",
            svd1_hash=hashlib.md5("|".join(svd1_info).encode()).hexdigest()[:12],
            svd2_hash=hashlib.md5("|".join(svd2_info).encode()).hexdigest()[:12]
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print(f"🔄 Comparing subspaces: {name1} vs {name2} (this will be cached)...")
        
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
                if self.verbose:
                    print(f"Comparison failed for {layer_name}: {e}")
                comparison_results[layer_name] = None
        
        # Cache the result
        self._save_to_cache(comparison_results, cache_key)
        
        if self.verbose:
            print(f"\nSubspace comparison: {name1} vs {name2}")
            for layer_name, results in comparison_results.items():
                if results:
                    print(f"  {layer_name}: mean_cosine={results['mean_cosine']:.3f}, "
                          f"overlap={results['subspace_overlap']:.3f}, "
                          f"angle={results['mean_angle_degrees']:.1f}°")
        
        return comparison_results
    
    def clear_cache(self, pattern="*"):
        """
        Clear cache files matching a pattern.
        """
        if pattern == "*":
            cache_files = list(self.cache_dir.glob("*.pkl"))
        else:
            cache_files = list(self.cache_dir.glob(f"*{pattern}*.pkl"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                if self.verbose:
                    print(f"🗑️  Deleted cache file: {cache_file.name}")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to delete {cache_file}: {e}")
        
        if self.verbose:
            print(f"Cleared {len(cache_files)} cache files")
    
    def get_cache_info(self):
        """
        Get information about current cache.
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)
        
        if self.verbose:
            print(f"\nCache Info:")
            print(f"  Directory: {self.cache_dir}")
            print(f"  Files: {len(cache_files)}")
            print(f"  Total size: {total_size_mb:.1f} MB")
            
            if cache_files:
                print(f"  Recent files:")
                # Sort by modification time
                recent_files = sorted(cache_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
                for f in recent_files:
                    age_hours = (time.time() - f.stat().st_mtime) / 3600
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"    {f.name[:50]:<50} ({size_mb:.1f}MB, {age_hours:.1f}h ago)")
        
        return {
            'num_files': len(cache_files),
            'total_size_mb': total_size_mb,
            'cache_dir': str(self.cache_dir)
        }
    
    def run_attack(self, original_model, unlearned_model, background_data, 
                   forget_data=None, retain_data=None, variance_threshold=0.95):
        """
        Run the complete gradient recovery attack with comprehensive caching.
        """
        print("="*60)
        print("RUNNING CACHED GRADIENT RECOVERY ATTACK")
        print("="*60)
        
        # Show cache info
        self.get_cache_info()
        
        # Step 1: Get target layers (cached)
        conv_layers, linear_layers = self.get_target_layers(original_model)
        
        # Step 2: Calculate covariances (heavily cached)
        print("\nStep 1: Computing covariances...")
        original_cov = self.calculate_input_covariances(
            original_model, background_data, conv_layers, linear_layers
        )
        
        unlearned_cov = self.calculate_input_covariances(
            unlearned_model, background_data, conv_layers, linear_layers
        )
        
        # Step 3: Estimate forget subspace (cached)
        print("\nStep 2: Estimating forget subspace...")
        estimated_forget_svd = self.estimate_forget_subspace(
            original_cov, unlearned_cov, variance_threshold
        )
        
        # Step 4: Create projection matrices (cached)
        print("\nStep 3: Creating projection matrices...")
        estimated_proj_matrices = self.create_projection_matrices(estimated_forget_svd)
        
        # Step 5: Test projection on gradients (cached)
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
        
        # Step 6: Verification (if ground truth data available) (cached)
        if forget_data is not None:
            print("\nStep 5: Verification with ground truth...")
            
            # Calculate true forget subspace (cached)
            forget_cov = self.calculate_input_covariances(
                original_model, forget_data, conv_layers, linear_layers
            )
            true_forget_svd = self.calculate_svd(forget_cov, variance_threshold=variance_threshold)
            
            # Compare estimated vs true forget subspaces (cached)
            forget_comparison = self.compare_subspaces(
                estimated_forget_svd, true_forget_svd, 
                "Estimated Forget", "True Forget"
            )
            results['forget_comparison'] = forget_comparison
            
            if retain_data is not None:
                # Calculate retain subspace for contrast (cached)
                retain_cov = self.calculate_input_covariances(
                    original_model, retain_data, conv_layers, linear_layers
                )
                retain_svd = self.calculate_svd(retain_cov, variance_threshold=variance_threshold)
                
                # Compare estimated forget vs retain (cached)
                retain_comparison = self.compare_subspaces(
                    estimated_forget_svd, retain_svd,
                    "Estimated Forget", "True Retain"
                )
                results['retain_comparison'] = retain_comparison
        
        print("\n" + "="*60)
        print("CACHED ATTACK COMPLETED")
        print("="*60)
        
        # Show final cache info
        self.get_cache_info()
        
        return results

# Convenience function with caching
def run_cached_gradient_recovery_attack(original_model, unlearned_model, background_data, 
                                       forget_data=None, retain_data=None, 
                                       device='cuda', variance_threshold=0.95,
                                       cache_dir='./gradient_recovery_cache',
                                       force_recompute=False):
    """
    Convenient function to run the complete attack with caching.
    """
    pipeline = CachedGradientRecoveryPipeline(
        device=device, 
        cache_dir=cache_dir,
        force_recompute=force_recompute,
        verbose=True
    )
    
    results = pipeline.run_attack(
        original_model=original_model,
        unlearned_model=unlearned_model,
        background_data=background_data,
        forget_data=forget_data,
        retain_data=retain_data,
        variance_threshold=variance_threshold
    )
    
    return results, pipeline