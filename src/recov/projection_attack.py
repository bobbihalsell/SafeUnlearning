import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import hashlib
import time
from recover import CachedGradientRecoveryPipeline

def run_projection_attack(original_model, unlearned_model, background_data, 
                         forget_data, retain_data, device='cuda', 
                         variance_threshold=0.85, cache_dir='./svd_cache',
                         force_recompute=False, verbose=True):
    """
    Run projection-based attack using gradient difference subspace analysis.
    Key insight: Compare subspaces of gradient differences between models on forget vs retain data.
    """
    
    print("PROJECTION ATTACK")
    print("="*50)
    
    # Initialize pipeline
    pipeline = GradientProjectionPipeline(
        device=device,
        cache_dir=cache_dir,
        force_recompute=force_recompute, 
        verbose=verbose
    )
    
    # Step 1: Calculate gradient covariances of model differences on forget data
    print("\nStep 1: Computing gradient difference covariances on forget data...")
    
    forget_grad_diff_cov = pipeline.calculate_gradient_diff_covariances(
        original_model, unlearned_model, forget_data, "forget"
    )
    
    # Step 2: Calculate gradient covariances of model differences on retain data  
    print("\nStep 2: Computing gradient difference covariances on retain data...")
    
    retain_grad_diff_cov = pipeline.calculate_gradient_diff_covariances(
        original_model, unlearned_model, retain_data, "retain"
    )
    
    # Step 3: Extract subspaces from the gradient covariances
    print("\nStep 3: Extracting gradient subspaces...")
    
    forget_grad_svd = pipeline.calculate_svd(forget_grad_diff_cov, variance_threshold=variance_threshold)
    retain_grad_svd = pipeline.calculate_svd(retain_grad_diff_cov, variance_threshold=variance_threshold)
    
    # Step 4: Compare the gradient subspaces
    print("\nStep 4: Comparing forget vs retain gradient subspaces...")
    
    subspace_comparison = pipeline.compare_subspaces(
        forget_grad_svd, retain_grad_svd,
        "Forget Grad Subspace", "Retain Grad Subspace"
    )
    
    # Step 5: Analyze results
    results = analyze_gradient_projection_results(subspace_comparison, verbose)
    
    # Step 6: Create plots
    plot_dir = Path("./plots/projection")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_gradient_projection_plots(
        forget_grad_svd, retain_grad_svd, subspace_comparison, 
        plot_dir, verbose
    )
    
    # Extract the gradient-based forget signature for potential GAN use
    forget_signature = extract_gradient_forget_signature(forget_grad_svd, retain_grad_svd, subspace_comparison)
    
    # Compile final results
    final_results = {
        'forget_grad_svd': forget_grad_svd,
        'retain_grad_svd': retain_grad_svd,
        'subspace_comparison': subspace_comparison,
        'forget_signature': forget_signature,
        'summary': results
    }
    
    return final_results


class GradientProjectionPipeline:
    """
    Pipeline for gradient-based projection attacks with caching.
    """
    
    def __init__(self, device='cuda', cache_dir='./gradient_projection_cache',
                 force_recompute=False, verbose=True):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.force_recompute = force_recompute
        self.verbose = verbose
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, operation, **kwargs):
        """Generate cache key for operation."""
        key_parts = [operation]
        for k, v in sorted(kwargs.items()):
            if hasattr(v, '__name__'):  # For models
                key_parts.append(f"{k}_{v.__name__}")
            elif isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}_{v}")
        
        return "_".join(key_parts) + f"_{int(time.time())//3600}"  # Hour-based cache
    
    def _save_to_cache(self, data, cache_key):
        """Save data to cache."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            if self.verbose:
                print(f"Saved to cache: {cache_path.name}")
        except Exception as e:
            if self.verbose:
                print(f"Cache save failed: {e}")
    
    def _load_from_cache(self, cache_key):
        """Load data from cache."""
        if self.force_recompute:
            return None
        
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            if self.verbose:
                print(f"Loaded from cache: {cache_path.name}")
            return data
        except Exception as e:
            if self.verbose:
                print(f"Cache load failed: {e}")
            return None
    
    def get_gradients(self, model, data_loader, max_batches=3):
        """
        Calculate gradients for model on data (memory-efficient version).
        """
        loss_fn = nn.CrossEntropyLoss()
        original_mode = model.training
        model.train()
        
        accumulated_gradients = {}
        num_batches = 0
        
        try:
            for batch_idx, (images, labels) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                model.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                
                # Accumulate gradients (detach and clone to avoid graph retention)
                for name, param in model.named_parameters():
                    if param.grad is not None and '.weight' in name:  # Focus on weight gradients
                        if name not in accumulated_gradients:
                            accumulated_gradients[name] = param.grad.detach().clone()
                        else:
                            accumulated_gradients[name] += param.grad.detach().clone()
                
                num_batches += 1
                
                # Clear GPU memory after each batch
                del images, labels, outputs, loss
                torch.cuda.empty_cache()
        
        finally:
            model.train(original_mode)
        
        # Average by number of batches
        for name in accumulated_gradients:
            accumulated_gradients[name] = accumulated_gradients[name] / num_batches
        
        return accumulated_gradients
    
    def calculate_gradient_diff_covariances(self, model1, model2, data_loader, data_type):
        """
        Calculate memory-efficient covariances of gradient differences between two models.
        """
        cache_key = self._get_cache_key(
            "grad_diff_cov_efficient",
            data_type=data_type,
            model1=model1.__class__,
            model2=model2.__class__
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print(f"Computing memory-efficient gradient difference covariances on {data_type} data...")
        
        # Get gradients from both models on same data
        grad1 = self.get_gradients(model1, data_loader)
        grad2 = self.get_gradients(model2, data_loader)
        
        # Calculate gradient differences
        grad_differences = {}
        for name in grad1:
            if name in grad2 and grad1[name] is not None and grad2[name] is not None:
                grad_differences[name] = grad1[name] - grad2[name]
        
        # Convert gradients to memory-efficient covariance approximations
        gradient_covariances = {}
        
        # Process layers one at a time to avoid memory buildup
        for param_name, grad_diff in grad_differences.items():
            if grad_diff is None:
                continue
            
            try:
                # Move to CPU to save GPU memory
                grad_diff_cpu = grad_diff.detach().cpu().float()
                
                # Flatten gradient
                grad_flat = grad_diff_cpu.flatten()  # [num_params]
                num_params = grad_flat.shape[0]
                
                layer_name = param_name.replace('.weight', '')
                
                # Memory-efficient covariance approximation
                if num_params > 2048:  # For large layers, use approximation
                    if self.verbose:
                        print(f"Layer {layer_name}: Using approximation for {num_params} parameters")
                    
                    # Use random projection to reduce dimensionality
                    target_dim = min(512, num_params // 4)  # Reduce to manageable size
                    
                    # Create random projection matrix
                    torch.manual_seed(42)  # For reproducibility
                    proj_matrix = torch.randn(target_dim, num_params) / np.sqrt(target_dim)
                    
                    # Project gradient to lower dimension
                    grad_projected = torch.mv(proj_matrix, grad_flat)  # [target_dim]
                    
                    # Create covariance in projected space
                    grad_cov = torch.outer(grad_projected, grad_projected)  # [target_dim, target_dim]
                    
                else:  # For small layers, use full covariance but on CPU
                    if self.verbose:
                        print(f"Layer {layer_name}: Using full covariance for {num_params} parameters")
                    
                    # Create full covariance matrix on CPU
                    grad_cov = torch.outer(grad_flat, grad_flat)  # [num_params, num_params]
                
                gradient_covariances[layer_name] = grad_cov.double()  # Keep on CPU
                
                if self.verbose:
                    print(f"Layer {layer_name}: covariance shape {grad_cov.shape}")
                
                # Clear GPU memory
                torch.cuda.empty_cache()
                
            except Exception as e:
                if self.verbose:
                    print(f"Failed to compute covariance for {param_name}: {e}")
                # Clear GPU memory on failure too
                torch.cuda.empty_cache()
                continue
        
        # Validate that we have meaningful differences
        total_diff_norm = 0
        for layer_name, cov in gradient_covariances.items():
            total_diff_norm += torch.norm(cov).item()
        
        if self.verbose:
            print(f"Total gradient covariance norm on {data_type}: {total_diff_norm:.6f}")
        
        self._save_to_cache(gradient_covariances, cache_key)
        return gradient_covariances
    
    def calculate_svd(self, covariance_dict, variance_threshold=0.95, eps=1e-6):
        """
        Calculate SVD decomposition of gradient covariance matrices (memory-efficient).
        """
        # Create cache hash
        cov_info = []
        for layer_name, cov_matrix in covariance_dict.items():
            if cov_matrix is not None:
                cov_signature = f"{layer_name}:{cov_matrix.shape}:{cov_matrix.mean().item():.6f}"
                cov_info.append(cov_signature)
        
        cache_key = self._get_cache_key(
            "gradient_svd_efficient",
            covariances_hash=hashlib.md5("|".join(cov_info).encode()).hexdigest()[:12],
            variance_threshold=variance_threshold,
            eps=eps
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print("Computing memory-efficient gradient SVD decompositions...")
        
        svd_results = {}
        
        for layer_name, cov_matrix in covariance_dict.items():
            if cov_matrix is None:
                continue
                
            try:
                # Ensure we're working on CPU for large matrices
                if cov_matrix.device.type == 'cuda':
                    cov_matrix = cov_matrix.cpu()
                
                # Add small epsilon for numerical stability
                stabilized = cov_matrix + eps * torch.eye(cov_matrix.shape[0], 
                                                        device=cov_matrix.device, 
                                                        dtype=cov_matrix.dtype)
                
                if self.verbose:
                    print(f"Computing SVD for layer {layer_name}: shape {stabilized.shape}")
                
                # SVD decomposition on CPU
                U, S, V = torch.svd(stabilized)
                
                # Apply variance threshold
                if variance_threshold is not None and len(S) > 1:
                    cumulative_var = torch.cumsum(S, dim=0) / torch.sum(S)
                    components_to_keep = torch.sum(cumulative_var <= variance_threshold).item()
                    components_to_keep = max(1, min(components_to_keep, len(S)))  # Keep at least 1, at most all
                else:
                    components_to_keep = min(10, len(S))  # Limit to top 10 components for efficiency
                
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
                
                # Clear memory
                del U, S, V, stabilized
                torch.cuda.empty_cache()
                    
            except Exception as e:
                if self.verbose:
                    print(f"SVD failed for layer {layer_name}: {e}")
                # Clear memory on failure
                torch.cuda.empty_cache()
                continue
        
        self._save_to_cache(svd_results, cache_key)
        return svd_results
    
    def compare_subspaces(self, svd1, svd2, name1="SVD1", name2="SVD2"):
        """
        Compare two sets of gradient SVD results to measure subspace similarity (memory-efficient).
        """
        if self.verbose:
            print(f"Comparing gradient subspaces: {name1} vs {name2}...")
        
        comparison_results = {}
        common_layers = set(svd1.keys()) & set(svd2.keys())
        
        for layer_name in common_layers:
            if layer_name not in svd1 or layer_name not in svd2:
                continue
                
            try:
                U1 = svd1[layer_name]['U']
                U2 = svd2[layer_name]['U']
                
                # Ensure both are on CPU for memory efficiency
                if U1.device.type == 'cuda':
                    U1 = U1.cpu()
                if U2.device.type == 'cuda':
                    U2 = U2.cpu()
                
                # Compute overlap matrix
                min_dim = min(U1.shape[1], U2.shape[1])
                if min_dim == 0:
                    if self.verbose:
                        print(f"Warning: Zero-dimensional subspace for {layer_name}")
                    continue
                    
                U1_trunc = U1[:, :min_dim]
                U2_trunc = U2[:, :min_dim]
                
                # Handle dimension mismatch
                if U1.shape[0] != U2.shape[0]:
                    min_feat_dim = min(U1.shape[0], U2.shape[0])
                    U1_trunc = U1_trunc[:min_feat_dim, :]
                    U2_trunc = U2_trunc[:min_feat_dim, :]
                
                if U1_trunc.shape[0] == 0 or U1_trunc.shape[1] == 0:
                    if self.verbose:
                        print(f"Warning: Empty subspace after truncation for {layer_name}")
                    continue
                
                # Compute principal angles (on CPU)
                overlap_matrix = U1_trunc.T @ U2_trunc
                _, S, _ = torch.svd(overlap_matrix)
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
                continue
        
        return comparison_results


def analyze_gradient_projection_results(subspace_comparison, verbose=True):
    """
    Analyze gradient projection attack results.
    Success = forget and retain gradient subspaces are DIFFERENT.
    """
    
    print("\nGRADIENT PROJECTION ATTACK ANALYSIS")
    print("-" * 40)
    
    # Extract similarity scores between forget and retain gradient subspaces
    similarities = [c['mean_cosine'] for c in subspace_comparison.values() if c]
    
    if not similarities:
        return {
            'success': False,
            'description': 'Could not compute gradient subspace similarities'
        }
    
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    print(f"Forget vs Retain Gradient Subspaces: {mean_similarity:.3f} (±{std_similarity:.3f})")
    
    # Success criteria: LOW similarity means distinct gradient patterns
    high_success_threshold = 0.3  # Very distinct patterns
    moderate_success_threshold = 0.6  # Moderately distinct patterns
    
    if mean_similarity < high_success_threshold:
        attack_success = True
        status = "✅ ATTACK SUCCESSFUL"
        description = f"Highly distinct gradient subspaces detected (similarity: {mean_similarity:.3f})"
        confidence = "High"
    elif mean_similarity < moderate_success_threshold:
        attack_success = True
        status = "✅ ATTACK SUCCESSFUL"
        description = f"Distinct gradient subspaces detected (similarity: {mean_similarity:.3f})"
        confidence = "Moderate"
    else:
        attack_success = False
        status = "❌ ATTACK FAILED"
        description = f"No distinct gradient subspaces (similarity: {mean_similarity:.3f})"
        confidence = "None"
    
    print(f"\n{status}")
    print(f"Description: {description}")
    print(f"Confidence: {confidence}")
    
    # Provide GAN optimization guidance
    if attack_success and mean_similarity < high_success_threshold:
        print("💡 GAN Optimization: Strong gradient signal - use gradient subspace projection")
        print("💡 Expected reconstruction quality: High")
    elif attack_success:
        print("💡 GAN Optimization: Moderate gradient signal - use weighted projection")
        print("💡 Expected reconstruction quality: Moderate")
    else:
        print("❌ GAN Optimization: Weak gradient signal - reconstruction will be difficult")
    
    return {
        'success': attack_success,
        'status': status,
        'description': description,
        'confidence': confidence,
        'forget_retain_similarity': mean_similarity,
        'similarity_std': std_similarity,
        'recoverable_signal': attack_success
    }


def extract_gradient_forget_signature(forget_grad_svd, retain_grad_svd, subspace_comparison):
    """
    Extract the gradient-based forget signature for GAN optimization.
    """
    signature = {}
    
    # Find layers with strongest gradient subspace differentiation
    layer_scores = []
    for layer_name, comp in subspace_comparison.items():
        if comp and layer_name in forget_grad_svd:
            # Lower similarity = better differentiation
            differentiation_score = 1.0 - comp['mean_cosine']
            layer_scores.append((layer_name, differentiation_score, comp))
    
    # Sort by differentiation score
    layer_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract signature from most discriminative layers
    for layer_name, score, comp in layer_scores:
        signature[layer_name] = {
            'gradient_subspace': forget_grad_svd[layer_name]['U'],
            'singular_values': forget_grad_svd[layer_name]['S'],
            'differentiation_score': score,
            'layer_name': layer_name,
            'importance_weight': score,  # For weighted GAN loss
            'subspace_dim': forget_grad_svd[layer_name]['U'].shape[1]
        }
    
    return signature


def create_gradient_projection_plots(forget_grad_svd, retain_grad_svd, 
                                    subspace_comparison, plot_dir, verbose=True):
    """
    Create visualization plots for gradient projection attack.
    """
    
    if verbose:
        print(f"Creating gradient projection plots in {plot_dir}...")
    
    # Plot 1: Gradient subspace similarity comparison
    layers = list(subspace_comparison.keys())
    similarities = [subspace_comparison[l]['mean_cosine'] if subspace_comparison[l] else 0 for l in layers]
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(layers)), similarities, color='darkgreen', alpha=0.7)
    
    # Add threshold lines
    plt.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='High success (0.3)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate success (0.6)')
    
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Gradient Projection Attack: Forget vs Retain Subspace Similarities\n(Lower = Better)')
    plt.xticks(range(len(layers)), [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Color bars based on success
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        if sim < 0.3:
            bar.set_color('green')
        elif sim < 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
        
        # Add value labels
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{sim:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'gradient_subspace_similarities.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Explained variance comparison
    plt.figure(figsize=(12, 6))
    
    forget_var = [forget_grad_svd[l]['explained_variance_ratio'] if l in forget_grad_svd else 0 for l in layers]
    retain_var = [retain_grad_svd[l]['explained_variance_ratio'] if l in retain_grad_svd else 0 for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    plt.bar(x - width/2, forget_var, width, label='Forget Gradient Subspace', color='red', alpha=0.7)
    plt.bar(x + width/2, retain_var, width, label='Retain Gradient Subspace', color='blue', alpha=0.7)
    
    plt.xlabel('Layer')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Gradient Projection Attack: Subspace Explained Variance')
    plt.xticks(x, [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_dir / 'gradient_explained_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Subspace dimensions
    plt.figure(figsize=(12, 6))
    
    forget_dims = [forget_grad_svd[l]['U'].shape[1] if l in forget_grad_svd else 0 for l in layers]
    retain_dims = [retain_grad_svd[l]['U'].shape[1] if l in retain_grad_svd else 0 for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    plt.bar(x - width/2, forget_dims, width, label='Forget Subspace Dimension', color='red', alpha=0.7)
    plt.bar(x + width/2, retain_dims, width, label='Retain Subspace Dimension', color='blue', alpha=0.7)
    
    plt.xlabel('Layer')
    plt.ylabel('Subspace Dimension')
    plt.title('Gradient Projection Attack: Subspace Dimensions')
    plt.xticks(x, [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_dir / 'gradient_subspace_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Overall attack summary
    plt.figure(figsize=(8, 6))
    
    mean_similarity = np.mean(similarities)
    success = mean_similarity < 0.6
    
    # Create summary visualization
    if mean_similarity < 0.3:
        color = 'green'
        label = 'HIGH SUCCESS'
        value = 1.0 - mean_similarity
    elif mean_similarity < 0.6:
        color = 'orange'
        label = 'SUCCESS'
        value = 0.6 - mean_similarity
    else:
        color = 'red'
        label = 'FAILED'
        value = 0.4
    
    plt.bar([label], [value], color=color, alpha=0.7)
    plt.ylabel('Attack Strength')
    plt.title(f'Gradient Projection Attack Result\nSimilarity: {mean_similarity:.3f}')
    
    # Add text annotation
    if mean_similarity < 0.3:
        result_text = "Highly distinct gradient subspaces!"
    elif mean_similarity < 0.6:
        result_text = "Distinct gradient patterns!"
    else:
        result_text = "No distinct gradient patterns"
    
    plt.text(0, value/2, result_text, ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'gradient_projection_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Gradient projection plots saved to {plot_dir}")