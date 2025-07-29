import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import hashlib
import time
from analysis import (
    analyze_gradient_results_standardized, 
    create_standardized_gradient_plots
)

def run_gradient_attack(original_model, unlearned_model, background_data,
                       forget_data, retain_data, device='cuda', 
                       cache_dir='./gradient_cache', force_recompute=False, 
                       verbose=True):
    """
    Run gradient-based attack using model gradient differences.
    Key insight: Compare gradient differences between models on forget vs retain data.
    """
    
    print("GRADIENT ATTACK")
    print("="*50)
    
    # Initialize pipeline
    pipeline = GradientAttackPipeline(
        device=device,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
        verbose=verbose
    )
    
    # Step 1: Calculate gradient differences between models on forget data
    print("\nStep 1: Computing gradient differences on forget data...")
    
    forget_grad_diff = pipeline.calculate_model_gradient_differences(
        original_model, unlearned_model, forget_data, "forget"
    )
    
    # Step 2: Calculate gradient differences between models on retain data
    print("\nStep 2: Computing gradient differences on retain data...")
    
    retain_grad_diff = pipeline.calculate_model_gradient_differences(
        original_model, unlearned_model, retain_data, "retain"
    )
    
    # Step 3: Compare the gradient difference patterns
    print("\nStep 3: Comparing forget vs retain gradient differences...")
    
    gradient_comparison = pipeline.compare_gradients(
        forget_grad_diff, retain_grad_diff, "Forget Grad Diff", "Retain Grad Diff"
    )
    # Step 4: Analyze results using STANDARDIZED function
    results = analyze_gradient_results_standardized(gradient_comparison, verbose)
    
    # Step 5: Create plots using STANDARDIZED function
    plot_dir = Path("./plots/gradient")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_standardized_gradient_plots(
        forget_grad_diff, retain_grad_diff, gradient_comparison,
        results, plot_dir, verbose
    )
    
    # Extract forget signature for GAN use (keep existing)
    forget_signature = extract_gradient_signature(forget_grad_diff, retain_grad_diff, gradient_comparison)
    
    # Compile final results with STANDARDIZED summary
    final_results = {
        'forget_grad_diff': forget_grad_diff,
        'retain_grad_diff': retain_grad_diff,
        'gradient_comparison': gradient_comparison,
        'forget_signature': forget_signature,
        'summary': results  # This is now standardized
    }
    
    return final_results
    
    # # Step 4: Analyze results
    # results = analyze_gradient_results_corrected(gradient_comparison, verbose)
    
    # # Step 5: Create plots
    # plot_dir = Path("./plots/gradient")
    # plot_dir.mkdir(parents=True, exist_ok=True)
    
    # create_gradient_plots_corrected(
    #     forget_grad_diff, retain_grad_diff, gradient_comparison,
    #     plot_dir, verbose
    # )
    
    # # Extract forget signature for GAN use
    # forget_signature = extract_gradient_signature(forget_grad_diff, retain_grad_diff, gradient_comparison)
    
    # # Compile final results
    # final_results = {
    #     'forget_grad_diff': forget_grad_diff,
    #     'retain_grad_diff': retain_grad_diff,
    #     'gradient_comparison': gradient_comparison,
    #     'forget_signature': forget_signature,
    #     'summary': results
    # }
    
    # return final_results


class GradientAttackPipeline:
    """
    Pipeline for gradient-based attacks with caching.
    """
    
    def __init__(self, device='cuda', cache_dir='./gradient_cache',
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
        
        return "_".join(key_parts)  # Hour-based cache
    
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
        Calculate gradients for model on data (fixed version).
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
                
                # Accumulate gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if name not in accumulated_gradients:
                            accumulated_gradients[name] = param.grad.clone().detach()
                        else:
                            accumulated_gradients[name] += param.grad.clone().detach()
                
                num_batches += 1
        
        finally:
            model.train(original_mode)
        
        # Average by number of batches
        for name in accumulated_gradients:
            accumulated_gradients[name] = accumulated_gradients[name] / num_batches
        
        return accumulated_gradients
    
    def calculate_model_gradient_differences(self, model1, model2, data_loader, data_type):
        """
        Calculate gradient differences between two models on same data.
        """
        cache_key = self._get_cache_key(
            "model_grad_diff",
            data_type=data_type,
            model1=model1.__class__,
            model2=model2.__class__
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print(f"Computing gradient differences between models on {data_type} data...")
        
        # Get gradients from both models on same data
        grad1 = self.get_gradients(model1, data_loader)
        grad2 = self.get_gradients(model2, data_loader)
        
        # Calculate differences
        grad_diff = {}
        for name in grad1:
            if name in grad2 and grad1[name] is not None and grad2[name] is not None:
                grad_diff[name] = grad1[name] - grad2[name]
        
        # Validate that we have meaningful differences
        total_diff_norm = 0
        for name, diff in grad_diff.items():
            if '.weight' in name:
                total_diff_norm += torch.norm(diff).item()
        
        if self.verbose:
            print(f"Total gradient difference norm on {data_type}: {total_diff_norm:.6f}")
        
        self._save_to_cache(grad_diff, cache_key)
        return grad_diff
    
    def compare_gradients(self, grad1, grad2, name1="Grad1", name2="Grad2"):
        """
        Compare two sets of gradients using various metrics.
        """
        if self.verbose:
            print(f"Comparing {name1} vs {name2}...")
        
        comparison_results = {}
        all_similarities = []
        
        for param_name in grad1:
            if param_name not in grad2 or '.weight' not in param_name:
                continue
            
            g1 = grad1[param_name]
            g2 = grad2[param_name]
            
            if g1 is None or g2 is None:
                continue
            
            try:
                # Flatten gradients
                g1_flat = g1.flatten()
                g2_flat = g2.flatten()
                
                # Check for zero gradients
                g1_norm = torch.norm(g1_flat).item()
                g2_norm = torch.norm(g2_flat).item()
                
                if g1_norm < 1e-8 or g2_norm < 1e-8:
                    if self.verbose:
                        print(f"Warning: Near-zero gradients for {param_name}")
                    continue
                
                # Cosine similarity
                cosine_sim = torch.cosine_similarity(g1_flat, g2_flat, dim=0).item()
                
                # L2 distance  
                l2_dist = torch.norm(g1_flat - g2_flat).item()
                
                # Relative L2 distance
                rel_l2_dist = l2_dist / (g1_norm + g2_norm + 1e-8)
                
                # Normalized dot product
                normalized_dot = torch.dot(g1_flat / g1_norm, g2_flat / g2_norm).item()
                
                layer_name = param_name.replace('.weight', '')
                comparison_results[layer_name] = {
                    'cosine_similarity': cosine_sim,
                    'l2_distance': l2_dist,
                    'relative_l2_distance': rel_l2_dist,
                    'normalized_dot_product': normalized_dot,
                    'grad1_norm': g1_norm,
                    'grad2_norm': g2_norm
                }
                
                all_similarities.append(cosine_sim)
                
            except Exception as e:
                if self.verbose:
                    print(f"Comparison failed for {param_name}: {e}")
                continue
        
        if self.verbose and all_similarities:
            mean_sim = np.mean(all_similarities)
            print(f"Mean similarity: {mean_sim:.4f}")
        
        return comparison_results


def analyze_gradient_results_corrected(gradient_comparison, verbose=True):
    """
    Analyze corrected gradient attack results.
    Success = forget and retain gradient differences are DIFFERENT.
    """
    
    print("\nGRADIENT ATTACK ANALYSIS")
    print("-" * 40)
    
    # Extract similarity scores
    similarities = [c['cosine_similarity'] for c in gradient_comparison.values() if c]
    
    if not similarities:
        return {
            'success': False,
            'description': 'Could not compute gradient similarities'
        }
    
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    print(f"Forget vs Retain Gradient Differences: {mean_similarity:.3f} (±{std_similarity:.3f})")
    
    # Success criteria: LOW similarity means distinct gradient patterns
    similarity_threshold = 0.3  # Below this means good separation
    
    attack_success = mean_similarity < similarity_threshold
    
    if attack_success:
        status = "✅ ATTACK SUCCESSFUL"
        description = f"Distinct gradient patterns detected (similarity: {mean_similarity:.3f})"
    elif mean_similarity < 0.6:
        status = "⚠️  PARTIALLY SUCCESSFUL"
        description = f"Moderate gradient pattern differences (similarity: {mean_similarity:.3f})"
    else:
        status = "❌ ATTACK FAILED"
        description = f"No distinct gradient patterns (similarity: {mean_similarity:.3f})"
    
    print(f"\n{status}")
    print(f"Description: {description}")
    
    # Provide GAN optimization guidance
    if attack_success:
        print("💡 GAN Optimization: Use forget gradient differences as target signature")
        print("💡 Expected reconstruction quality: High")
    elif mean_similarity < 0.6:
        print("⚠️  GAN Optimization: Moderate signal, reconstruction possible but may be noisy")
    else:
        print("❌ GAN Optimization: Weak signal, reconstruction will be difficult")
    
    return {
        'success': attack_success,
        'status': status,
        'description': description,
        'forget_retain_similarity': mean_similarity,
        'similarity_std': std_similarity,
        'recoverable_signal': attack_success or mean_similarity < 0.6
    }


def extract_gradient_signature(forget_grad_diff, retain_grad_diff, gradient_comparison):
    """
    Extract the gradient signature for GAN optimization.
    """
    signature = {}
    
    # Find layers with strongest differentiation
    layer_scores = []
    for layer_name, comp in gradient_comparison.items():
        if comp:
            # Lower similarity = better differentiation
            differentiation_score = 1.0 - comp['cosine_similarity']
            layer_scores.append((layer_name, differentiation_score, comp))
    
    # Sort by differentiation score
    layer_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract signature from most discriminative layers
    for layer_name, score, comp in layer_scores:
        if layer_name + '.weight' in forget_grad_diff:
            signature[layer_name] = {
                'gradient_diff': forget_grad_diff[layer_name + '.weight'],
                'differentiation_score': score,
                'layer_name': layer_name,
                'importance_weight': score  # Can be used to weight GAN loss
            }
    
    return signature


def create_gradient_plots_corrected(forget_grad_diff, retain_grad_diff, 
                                   gradient_comparison, plot_dir, verbose=True):
    """
    Create visualization plots for corrected gradient attack results.
    """
    
    if verbose:
        print(f"Creating gradient plots in {plot_dir}...")
    
    # Extract data for plotting
    layers = list(gradient_comparison.keys())
    similarities = [gradient_comparison[l]['cosine_similarity'] for l in layers]
    l2_distances = [gradient_comparison[l]['relative_l2_distance'] for l in layers]
    
    # Plot 1: Gradient pattern similarities
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(layers)), similarities, color='orange', alpha=0.7)
    
    # Add threshold lines
    plt.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Success threshold (0.3)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Partial success (0.6)')
    
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Gradient Attack: Forget vs Retain Pattern Similarities\n(Lower = Better)')
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
                f'{sim:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'gradient_similarities.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: L2 distances
    plt.figure(figsize=(12, 6))
    
    plt.bar(range(len(layers)), l2_distances, color='blue', alpha=0.7)
    plt.xlabel('Layer')
    plt.ylabel('Relative L2 Distance')
    plt.title('Gradient Attack: Pattern Differentiation Strength')
    plt.xticks(range(len(layers)), [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_dir / 'gradient_distances.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Gradient magnitude comparison
    forget_norms = []
    retain_norms = []
    
    for layer in layers:
        param_name = layer + '.weight'
        if param_name in forget_grad_diff and param_name in retain_grad_diff:
            forget_norm = torch.norm(forget_grad_diff[param_name]).item()
            retain_norm = torch.norm(retain_grad_diff[param_name]).item()
            forget_norms.append(forget_norm)
            retain_norms.append(retain_norm)
        else:
            forget_norms.append(0)
            retain_norms.append(0)
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    plt.bar(x - width/2, forget_norms, width, label='Forget Grad Diff Magnitude', color='red', alpha=0.7)
    plt.bar(x + width/2, retain_norms, width, label='Retain Grad Diff Magnitude', color='blue', alpha=0.7)
    
    plt.xlabel('Layer')
    plt.ylabel('Gradient Difference Magnitude')
    plt.title('Gradient Attack: Model Response Magnitudes')
    plt.xticks(x, [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_dir / 'gradient_magnitudes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Overall attack summary
    plt.figure(figsize=(8, 6))
    
    mean_similarity = np.mean(similarities)
    success = mean_similarity < 0.3
    partial = 0.3 <= mean_similarity < 0.6
    
    # Create summary visualization
    if success:
        color = 'green'
        label = 'SUCCESS'
        value = 1.0 - mean_similarity
    elif partial:
        color = 'orange'
        label = 'PARTIAL'
        value = 0.6 - mean_similarity
    else:
        color = 'red'
        label = 'FAILED'
        value = mean_similarity
    
    plt.bar([label], [value], color=color, alpha=0.7)
    plt.ylabel('Attack Strength')
    plt.title(f'Gradient Attack Result\nSimilarity: {mean_similarity:.3f}')
    
    # Add text annotation
    if success:
        result_text = "Distinct gradient patterns!"
    elif partial:
        result_text = "Moderate differentiation"
    else:
        result_text = "No distinct patterns"
    
    plt.text(0, value/2, result_text, ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'gradient_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Gradient plots saved to {plot_dir}")