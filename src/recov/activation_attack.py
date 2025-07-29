import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import hashlib
import time
from collections import OrderedDict
from analysis import (
    analyze_focused_activation_results_standardized, 
    create_standardized_activation_plots
)

def run_activation_attack(original_model, unlearned_model, background_data,
                         forget_data, retain_data, device='cuda',
                         cache_dir='./activation_cache', force_recompute=False,
                         verbose=True):
    """
    Run activation-based attack with focused layer selection.
    Key insight: First find layers that actually changed, then compare forget vs retain on those layers only.
    """
    
    print("ACTIVATION ATTACK")
    print("="*50)
    
    # Initialize pipeline
    pipeline = ActivationAttackPipeline(
        device=device,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
        verbose=verbose
    )
    
    # Step 1: Find layers that actually changed using background data
    print("\nStep 1: Identifying layers with significant model differences...")
    
    important_layers = pipeline.find_most_different_layers(
        original_model, unlearned_model, background_data, variance_explained=0.9
    )
    
    if verbose:
        print(f"Selected {len(important_layers)} most important layers:")
        for layer, diff in important_layers[:10]:  # Show top 10
            print(f"  {layer}: difference magnitude = {diff:.6f}")
    
    # Extract just the layer names
    selected_layer_names = [layer for layer, _ in important_layers]
    
    # Step 2: Calculate activation differences on forget data (focused)
    print(f"\nStep 2: Computing focused activation differences on forget data...")
    
    forget_activation_diff = pipeline.calculate_focused_activation_differences(
        original_model, unlearned_model, forget_data, selected_layer_names, "forget"
    )
    
    # Step 3: Calculate activation differences on retain data (focused)
    print(f"\nStep 3: Computing focused activation differences on retain data...")
    
    retain_activation_diff = pipeline.calculate_focused_activation_differences(
        original_model, unlearned_model, retain_data, selected_layer_names, "retain"
    )
    
    # Step 4: Compare the focused activation difference patterns
    print("\nStep 4: Comparing focused forget vs retain activation differences...")
    
    activation_comparison = pipeline.compare_activations(
        forget_activation_diff, retain_activation_diff, 
        "Focused Forget Diff", "Focused Retain Diff"
    )

    # Step 5: Analyze results using STANDARDIZED function
    results = analyze_focused_activation_results_standardized(
        activation_comparison, important_layers, selected_layer_names, verbose
    )
    
    # Step 6: Create plots using STANDARDIZED function
    plot_dir = Path("./plots/activation")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_standardized_activation_plots(
        forget_activation_diff, retain_activation_diff, activation_comparison,
        important_layers, results, plot_dir, verbose
    )
    
    # Extract forget signature for GAN use (keep existing)
    forget_signature = extract_focused_activation_signature(
        forget_activation_diff, retain_activation_diff, activation_comparison, important_layers
    )
    
    # Compile final results with STANDARDIZED summary
    final_results = {
        'important_layers': important_layers,
        'selected_layer_names': selected_layer_names,
        'forget_activation_diff': forget_activation_diff,
        'retain_activation_diff': retain_activation_diff,
        'activation_comparison': activation_comparison,
        'forget_signature': forget_signature,
        'summary': results  # This is now standardized
    }
    
    return final_results
    
    # # Step 5: Analyze results
    # results = analyze_focused_activation_results(
    #     activation_comparison, important_layers, selected_layer_names, verbose
    # )
    
    # # Step 6: Create plots
    # plot_dir = Path("./plots/activation")
    # plot_dir.mkdir(parents=True, exist_ok=True)
    
    # create_focused_activation_plots(
    #     forget_activation_diff, retain_activation_diff, activation_comparison,
    #     important_layers, plot_dir, verbose
    # )
    
    # # Extract forget signature for GAN use
    # forget_signature = extract_focused_activation_signature(
    #     forget_activation_diff, retain_activation_diff, activation_comparison, important_layers
    # )
    
    # # Compile final results
    # final_results = {
    #     'important_layers': important_layers,
    #     'selected_layer_names': selected_layer_names,
    #     'forget_activation_diff': forget_activation_diff,
    #     'retain_activation_diff': retain_activation_diff,
    #     'activation_comparison': activation_comparison,
    #     'forget_signature': forget_signature,
    #     'summary': results
    # }
    
    # return final_results


class ActivationAttackPipeline:
    """
    Pipeline for focused activation-based attacks with caching.
    """
    
    def __init__(self, device='cuda', cache_dir='./activation_cache',
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
            elif isinstance(v, list) and len(v) < 20:  # For layer lists
                key_parts.append(f"{k}_{'_'.join(map(str, v))}")
        
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
    
    def get_all_target_layers(self, model):
        """Get all target layers for activation extraction."""
        target_layers = OrderedDict()
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                target_layers[name] = name
        
        return target_layers
    
    def find_most_different_layers(self, model1, model2, background_data, variance_explained=0.9, max_batches=3):
        """
        Find layers with most significant activation differences using background data.
        """
        cache_key = self._get_cache_key(
            "find_different_layers",
            model1=model1.__class__,
            model2=model2.__class__,
            variance_explained=variance_explained,
            max_batches=max_batches
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print("Finding layers with most significant differences...")
        
        all_layers = self.get_all_target_layers(model1)
        layer_differences = {}
        
        # Get activations from both models on background data
        model1_activations = self.get_activations(model1, background_data, list(all_layers.keys()), max_batches)
        model2_activations = self.get_activations(model2, background_data, list(all_layers.keys()), max_batches)
        
        # Calculate difference magnitude for each layer
        for layer_name in all_layers.keys():
            if layer_name in model1_activations and layer_name in model2_activations:
                act1 = model1_activations[layer_name]
                act2 = model2_activations[layer_name]
                
                if act1 is not None and act2 is not None:
                    # Calculate difference magnitude
                    difference = act1 - act2
                    diff_magnitude = torch.norm(difference).item()
                    layer_differences[layer_name] = diff_magnitude
                else:
                    layer_differences[layer_name] = 0.0
            else:
                layer_differences[layer_name] = 0.0
        
        # Sort layers by difference magnitude
        sorted_layers = sorted(layer_differences.items(), key=lambda x: x[1], reverse=True)
        
        # Select layers that explain specified variance
        total_difference = sum(layer_differences.values())
        
        if total_difference < 1e-8:
            if self.verbose:
                print("⚠️  WARNING: Models show minimal differences!")
            # Return all layers if no differences found
            selected_layers = sorted_layers
        else:
            cumulative_diff = 0
            selected_layers = []
            
            for layer_name, diff in sorted_layers:
                cumulative_diff += diff
                selected_layers.append((layer_name, diff))
                
                if cumulative_diff / total_difference >= variance_explained:
                    break
            
            # Ensure we have at least a few layers
            if len(selected_layers) < 3:
                selected_layers = sorted_layers[:max(3, len(sorted_layers)//2)]
        
        if self.verbose:
            explained_variance = sum(diff for _, diff in selected_layers) / total_difference if total_difference > 0 else 0
            print(f"Selected {len(selected_layers)} layers explaining {explained_variance:.1%} of differences")
        
        # Cache and return
        self._save_to_cache(selected_layers, cache_key)
        return selected_layers
    
    def get_activations(self, model, data_loader, target_layer_names, max_batches=3):
        """
        Extract activations from specified layers only.
        """
        model.eval()
        
        # Storage for activations
        activations = {name: [] for name in target_layer_names}
        
        # Hook function to capture activations
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                # Store output activations
                if isinstance(module, nn.Conv2d):
                    # For conv layers: average over spatial dimensions
                    if len(output.shape) == 4:  # [batch, channels, h, w]
                        act = output.mean(dim=[2, 3])  # [batch, channels]
                    else:
                        act = output
                else:
                    # For linear layers: use directly
                    act = output
                
                activations[layer_name].append(act.detach().cpu())
            return hook_fn
        
        # Register hooks only for target layers
        hooks = []
        for layer_name in target_layer_names:
            try:
                layer = self._get_layer_by_name(model, layer_name)
                hook = layer.register_forward_hook(create_hook(layer_name))
                hooks.append(hook)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to register hook for {layer_name}: {e}")
                continue
        
        # Forward pass to collect activations
        try:
            with torch.no_grad():
                batch_count = 0
                for batch_idx, (images, _) in enumerate(data_loader):
                    if batch_idx >= max_batches:
                        break
                    
                    images = images.to(self.device)
                    _ = model(images)  # Forward pass triggers hooks
                    batch_count += 1
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Concatenate and average activations across batches
        final_activations = {}
        for layer_name, layer_activations in activations.items():
            if layer_activations:
                # Concatenate all batches: [total_samples, features]
                concat_acts = torch.cat(layer_activations, dim=0)
                # Average across samples: [features]
                final_activations[layer_name] = concat_acts.mean(dim=0)
            else:
                final_activations[layer_name] = None
        
        return final_activations
    
    def _get_layer_by_name(self, model, layer_name):
        """Get layer by dotted name path."""
        parts = layer_name.split('.')
        layer = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer
    
    def calculate_focused_activation_differences(self, model1, model2, data_loader, 
                                               selected_layers, data_type):
        """
        Calculate activation differences between models on selected layers only.
        """
        cache_key = self._get_cache_key(
            "focused_act_diff",
            data_type=data_type,
            model1=model1.__class__,
            model2=model2.__class__,
            selected_layers=selected_layers[:5]  # Use first 5 for cache key
        )
        
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        if self.verbose:
            print(f"Computing focused activation differences on {data_type} data...")
            print(f"Analyzing {len(selected_layers)} selected layers")
        
        # Get activations from both models on selected layers only
        act1 = self.get_activations(model1, data_loader, selected_layers)
        act2 = self.get_activations(model2, data_loader, selected_layers)
        
        # Calculate differences
        activation_diff = {}
        total_diff_norm = 0
        
        for layer_name in selected_layers:
            if layer_name in act1 and layer_name in act2:
                if act1[layer_name] is not None and act2[layer_name] is not None:
                    diff = act1[layer_name] - act2[layer_name]
                    activation_diff[layer_name] = diff
                    total_diff_norm += torch.norm(diff).item()
        
        if self.verbose:
            print(f"Total focused activation difference norm on {data_type}: {total_diff_norm:.6f}")
            print(f"Successfully computed differences for {len(activation_diff)} layers")
        
        self._save_to_cache(activation_diff, cache_key)
        return activation_diff
    
    def compare_activations(self, act1, act2, name1="Act1", name2="Act2"):
        """
        Compare two sets of activations using various metrics.
        """
        if self.verbose:
            print(f"Comparing {name1} vs {name2}...")
        
        comparison_results = {}
        all_similarities = []
        
        for layer_name in act1:
            if layer_name not in act2:
                continue
            
            a1 = act1[layer_name]
            a2 = act2[layer_name]
            
            if a1 is None or a2 is None:
                continue
            
            try:
                # Flatten activations for comparison
                a1_flat = a1.flatten()
                a2_flat = a2.flatten()
                
                # Check for zero activations
                a1_norm = torch.norm(a1_flat).item()
                a2_norm = torch.norm(a2_flat).item()
                
                if a1_norm < 1e-8 or a2_norm < 1e-8:
                    if self.verbose:
                        print(f"Warning: Near-zero activations for {layer_name}")
                    continue
                
                # Cosine similarity
                cosine_sim = torch.cosine_similarity(a1_flat, a2_flat, dim=0).item()
                
                # L2 distance
                l2_dist = torch.norm(a1_flat - a2_flat).item()
                
                # Relative L2 distance
                rel_l2_dist = l2_dist / (a1_norm + a2_norm + 1e-8)
                
                # Mean squared error
                mse = torch.mean((a1_flat - a2_flat) ** 2).item()
                
                # Correlation coefficient
                if len(a1_flat) > 1:
                    try:
                        correlation = torch.corrcoef(torch.stack([a1_flat, a2_flat]))[0, 1].item()
                        if torch.isnan(torch.tensor(correlation)):
                            correlation = 0.0
                    except:
                        correlation = 0.0
                else:
                    correlation = 0.0
                
                comparison_results[layer_name] = {
                    'cosine_similarity': cosine_sim,
                    'l2_distance': l2_dist,
                    'relative_l2_distance': rel_l2_dist,
                    'correlation': correlation,
                    'mse': mse,
                    'act1_norm': a1_norm,
                    'act2_norm': a2_norm
                }
                
                all_similarities.append(cosine_sim)
                
            except Exception as e:
                if self.verbose:
                    print(f"Comparison failed for {layer_name}: {e}")
                continue
        
        if self.verbose and all_similarities:
            mean_sim = np.mean(all_similarities)
            print(f"Mean focused similarity: {mean_sim:.4f}")
        
        return comparison_results


def analyze_focused_activation_results(activation_comparison, important_layers, 
                                     selected_layer_names, verbose=True):
    """
    Analyze focused activation attack results.
    Success = forget and retain activation differences are DIFFERENT on important layers.
    """
    
    print("\nFOCUSED ACTIVATION ATTACK ANALYSIS")
    print("-" * 40)
    
    # Extract similarity scores
    similarities = [c['cosine_similarity'] for c in activation_comparison.values() if c]
    
    if not similarities:
        return {
            'success': False,
            'description': 'Could not compute focused activation similarities'
        }
    
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    median_similarity = np.median(similarities)
    
    print(f"Focused Analysis Results:")
    print(f"  Selected {len(selected_layer_names)} most important layers")
    print(f"  Mean similarity: {mean_similarity:.3f} (±{std_similarity:.3f})")
    print(f"  Median similarity: {median_similarity:.3f}")
    
    # Show layer importance distribution
    if len(important_layers) > 0:
        total_importance = sum(diff for _, diff in important_layers)
        print(f"  Top 5 most important layers:")
        for i, (layer, diff) in enumerate(important_layers[:5]):
            importance_pct = (diff / total_importance) * 100 if total_importance > 0 else 0
            print(f"    {i+1}. {layer}: {importance_pct:.1f}% of total difference")
    
    # Adjusted success criteria for focused analysis
    # Since we filtered out noise, we can be more stringent
    high_success_threshold = 0.2  # Very distinct patterns
    moderate_success_threshold = 0.5  # Moderately distinct patterns
    
    if mean_similarity < high_success_threshold:
        attack_success = True
        status = "✅ ATTACK SUCCESSFUL"
        description = f"Highly distinct activation patterns detected (similarity: {mean_similarity:.3f})"
        confidence = "High"
    elif mean_similarity < moderate_success_threshold:
        attack_success = True
        status = "✅ ATTACK SUCCESSFUL"
        description = f"Distinct activation patterns detected (similarity: {mean_similarity:.3f})"
        confidence = "Moderate"
    elif mean_similarity < 0.7:
        attack_success = False
        status = "⚠️  PARTIALLY SUCCESSFUL"
        description = f"Some activation pattern differences (similarity: {mean_similarity:.3f})"
        confidence = "Low"
    else:
        attack_success = False
        status = "❌ ATTACK FAILED"
        description = f"No distinct activation patterns (similarity: {mean_similarity:.3f})"
        confidence = "None"
    
    print(f"\n{status}")
    print(f"Description: {description}")
    print(f"Confidence: {confidence}")
    
    # Provide GAN optimization guidance
    if attack_success and mean_similarity < high_success_threshold:
        print("💡 GAN Optimization: Strong signal - use focused activation differences as target")
        print("💡 Expected reconstruction quality: High")
    elif attack_success:
        print("💡 GAN Optimization: Moderate signal - use weighted activation differences")
        print("💡 Expected reconstruction quality: Moderate")
    else:
        print("❌ GAN Optimization: Weak signal - reconstruction will be difficult")
    
    return {
        'success': attack_success,
        'status': status,
        'description': description,
        'confidence': confidence,
        'forget_retain_similarity': mean_similarity,
        'similarity_std': std_similarity,
        'median_similarity': median_similarity,
        'num_focused_layers': len(selected_layer_names),
        'recoverable_signal': attack_success
    }


def extract_focused_activation_signature(forget_activation_diff, retain_activation_diff, 
                                       activation_comparison, important_layers):
    """
    Extract the focused activation signature for GAN optimization.
    """
    signature = {}
    
    # Create importance weighting from layer analysis
    layer_importance = {layer: diff for layer, diff in important_layers}
    total_importance = sum(layer_importance.values()) if layer_importance else 1.0
    
    # Find layers with strongest differentiation
    layer_scores = []
    for layer_name, comp in activation_comparison.items():
        if comp and layer_name in forget_activation_diff:
            # Lower similarity = better differentiation
            differentiation_score = 1.0 - comp['cosine_similarity']
            
            # Weight by layer importance
            importance_weight = layer_importance.get(layer_name, 0.0) / total_importance
            
            # Combined score
            combined_score = differentiation_score * (1.0 + importance_weight)
            
            layer_scores.append((layer_name, combined_score, differentiation_score, importance_weight, comp))
    
    # Sort by combined score
    layer_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract signature from most discriminative layers
    for layer_name, combined_score, diff_score, importance, comp in layer_scores:
        signature[layer_name] = {
            'activation_diff': forget_activation_diff[layer_name],
            'differentiation_score': diff_score,
            'importance_weight': importance,
            'combined_score': combined_score,
            'layer_name': layer_name,
            'magnitude': torch.norm(forget_activation_diff[layer_name]).item(),
            'gan_weight': combined_score  # For weighted GAN loss
        }
    
    return signature


def create_focused_activation_plots(forget_activation_diff, retain_activation_diff,
                                   activation_comparison, important_layers, 
                                   plot_dir, verbose=True):
    """
    Create visualization plots for focused activation attack results.
    """
    
    if verbose:
        print(f"Creating focused activation plots in {plot_dir}...")
    
    # Extract data for plotting
    layers = list(activation_comparison.keys())
    similarities = [activation_comparison[l]['cosine_similarity'] for l in layers]
    mse_values = [activation_comparison[l]['mse'] for l in layers]
    
    # Get layer importance scores
    layer_importance = {layer: diff for layer, diff in important_layers}
    importance_scores = [layer_importance.get(l, 0.0) for l in layers]
    
    # Plot 1: Focused activation pattern similarities
    plt.figure(figsize=(15, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top plot: Similarities
    bars1 = ax1.bar(range(len(layers)), similarities, color='purple', alpha=0.7)
    
    # Add threshold lines
    ax1.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='High success (0.2)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate success (0.5)')
    
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Focused Activation Attack: Forget vs Retain Pattern Similarities\n(Lower = Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Color bars based on success
    for i, (bar, sim) in enumerate(zip(bars1, similarities)):
        if sim < 0.2:
            bar.set_color('green')
        elif sim < 0.5:
            bar.set_color('orange')  
        else:
            bar.set_color('red')
        
        # Add value labels
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{sim:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Bottom plot: Layer importance
    bars2 = ax2.bar(range(len(layers)), importance_scores, color='cyan', alpha=0.7)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Importance Score')
    ax2.set_title('Layer Importance from Background Analysis')
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis labels for both plots
    layer_labels = [l.replace('layer', 'L') for l in layers]
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layer_labels, rotation=45, ha='right')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layer_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'focused_activation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Layer importance distribution
    plt.figure(figsize=(12, 6))
    
    # Show top 15 most important layers
    top_layers = important_layers[:15]
    if top_layers:
        layer_names = [layer for layer, _ in top_layers]
        importance_values = [diff for _, diff in top_layers]
        
        bars = plt.bar(range(len(layer_names)), importance_values, color='steelblue', alpha=0.7)
        
        plt.xlabel('Layer Rank')
        plt.ylabel('Difference Magnitude')
        plt.title('Top 15 Most Important Layers (Background Analysis)')
        plt.xticks(range(len(layer_names)), 
                  [l.replace('layer', 'L') for l in layer_names], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add cumulative percentage
        total_importance = sum(diff for _, diff in important_layers)
        cumulative_pct = 0
        for i, (bar, (_, diff)) in enumerate(zip(bars, top_layers)):
            cumulative_pct += (diff / total_importance) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance_values)*0.01,
                    f'{cumulative_pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'layer_importance_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Activation magnitude comparison
    forget_norms = []
    retain_norms = []
    
    for layer in layers:
        if layer in forget_activation_diff and layer in retain_activation_diff:
            forget_norm = torch.norm(forget_activation_diff[layer]).item()
            retain_norm = torch.norm(retain_activation_diff[layer]).item()
            forget_norms.append(forget_norm)
            retain_norms.append(retain_norm)
        else:
            forget_norms.append(0)
            retain_norms.append(0)
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    plt.bar(x - width/2, forget_norms, width, label='Forget Diff Magnitude', color='red', alpha=0.7)
    plt.bar(x + width/2, retain_norms, width, label='Retain Diff Magnitude', color='blue', alpha=0.7)
    
    plt.xlabel('Layer')
    plt.ylabel('Activation Difference Magnitude')
    plt.title('Focused Activation Attack: Model Response Magnitudes')
    plt.xticks(x, [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_dir / 'focused_activation_magnitudes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Overall attack summary with layer count info
    plt.figure(figsize=(10, 6))
    
    mean_similarity = np.mean(similarities)
    num_layers = len(layers)
    success = mean_similarity < 0.5
    
    # Create summary visualization
    if mean_similarity < 0.2:
        color = 'green'
        label = 'HIGH SUCCESS'
        value = 1.0 - mean_similarity
    elif mean_similarity < 0.5:
        color = 'orange'
        label = 'SUCCESS'
        value = 0.8 - mean_similarity
    elif mean_similarity < 0.7:
        color = 'orange'
        label = 'PARTIAL'
        value = 0.7 - mean_similarity
    else:
        color = 'red'
        label = 'FAILED'
        value = 0.3
    
    plt.bar([label], [value], color=color, alpha=0.7)
    plt.ylabel('Attack Strength')
    plt.title(f'Focused Activation Attack Result\nSimilarity: {mean_similarity:.3f} | Analyzed {num_layers} focused layers')
    
    # Add text annotation
    if mean_similarity < 0.2:
        result_text = "Highly distinct patterns!"
    elif mean_similarity < 0.5:
        result_text = "Distinct patterns detected!"
    elif mean_similarity < 0.7:
        result_text = "Some differentiation"
    else:
        result_text = "No distinct patterns"
    
    plt.text(0, value/2, result_text, ha='center', va='center', fontweight='bold')
    
    # Add layer info
    plt.text(0, value + 0.05, f"Focused on {num_layers} most important layers", 
             ha='center', va='bottom', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'focused_activation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Focused activation plots saved to {plot_dir}")