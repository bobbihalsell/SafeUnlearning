import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime
from collections import OrderedDict

# Set consistent plot style
plt.style.use('default')
sns.set_palette("husl")

def run_activation_comparison(original_model, unlearned_model, forget_data, retain_data, 
                             device='cuda', verbose=True):
    """
    Compare activation differences between original and unlearned models on forget vs retain data.
    
    This directly compares:
    - Forget Pattern: Activation differences (Original - Unlearned) when processing FORGET data
    - Retain Pattern: Activation differences (Original - Unlearned) when processing RETAIN data
    
    Returns similarity metrics and creates visualization plots.
    """
    
    print("ACTIVATION COMPARISON ANALYSIS")
    print("="*60)
    print("Direct comparison of activation differences on forget vs retain data")
    print("="*60)
    
    # Step 1: Find layers that changed significantly
    print("\nStep 1: Identifying layers with significant changes...")
    important_layers = find_changed_layers(original_model, unlearned_model, forget_data, device)
    
    # Step 2: Calculate activation differences on forget data
    print("\nStep 2: Computing activation differences on forget data...")
    forget_act_diff = calculate_activation_differences(
        original_model, unlearned_model, forget_data, device, "forget"
    )
    
    # Step 3: Calculate activation differences on retain data
    print("\nStep 3: Computing activation differences on retain data...")
    retain_act_diff = calculate_activation_differences(
        original_model, unlearned_model, retain_data, device, "retain"
    )
    
    # Step 4: Compare the activation difference patterns
    print("\nStep 4: Comparing forget vs retain activation difference patterns...")
    comparison_results = compare_activation_patterns(forget_act_diff, retain_act_diff, verbose)
    
    # Step 5: Create plots
    plot_dir = Path("./attack_plots/comparison")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_activation_comparison_plots(
        forget_act_diff, retain_act_diff, comparison_results, important_layers, plot_dir, verbose
    )
    
    # Step 6: Analyze results
    analysis_results = analyze_activation_comparison_results(comparison_results, important_layers, verbose)
    
    return {
        'forget_act_diff': forget_act_diff,
        'retain_act_diff': retain_act_diff,
        'comparison_results': comparison_results,
        'important_layers': important_layers,
        'analysis': analysis_results,
        'plot_directory': str(plot_dir)
    }


def find_changed_layers(original_model, unlearned_model, data_loader, device, max_batches=2):
    """Find layers that changed most during unlearning."""
    
    # Get all target layers
    target_layers = get_all_target_layers(original_model)
    
    # Get activations from both models
    orig_activations = get_activations(original_model, data_loader, list(target_layers.keys()), device, max_batches)
    unl_activations = get_activations(unlearned_model, data_loader, list(target_layers.keys()), device, max_batches)
    
    # Calculate difference magnitude for each layer
    layer_differences = {}
    for layer_name in target_layers.keys():
        if layer_name in orig_activations and layer_name in unl_activations:
            if orig_activations[layer_name] is not None and unl_activations[layer_name] is not None:
                diff = orig_activations[layer_name] - unl_activations[layer_name]
                diff_magnitude = torch.norm(diff).item()
                layer_differences[layer_name] = diff_magnitude
    
    # Sort by importance
    sorted_layers = sorted(layer_differences.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Found {len(sorted_layers)} layers with measurable changes")
    if sorted_layers:
        print(f"  Top 5 changed layers:")
        for i, (layer, change) in enumerate(sorted_layers[:5]):
            print(f"    {i+1}. {layer}: {change:.6f}")
    
    return sorted_layers


def get_all_target_layers(model):
    """Get all target layers for activation extraction."""
    target_layers = OrderedDict()
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            target_layers[name] = name
    
    return target_layers


def get_activations(model, data_loader, target_layer_names, device, max_batches=3):
    """Extract activations from specified layers."""
    
    model.eval()
    
    # Storage for activations
    activations = {name: [] for name in target_layer_names}
    
    # Hook function to capture activations
    def create_hook(layer_name):
        def hook_fn(module, input, output):
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
    
    # Register hooks
    hooks = []
    for layer_name in target_layer_names:
        try:
            layer = get_layer_by_name(model, layer_name)
            hook = layer.register_forward_hook(create_hook(layer_name))
            hooks.append(hook)
        except Exception as e:
            print(f"Failed to register hook for {layer_name}: {e}")
            continue
    
    # Forward pass to collect activations
    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                images = images.to(device)
                _ = model(images)
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Concatenate and average activations
    final_activations = {}
    for layer_name, layer_activations in activations.items():
        if layer_activations:
            concat_acts = torch.cat(layer_activations, dim=0)
            final_activations[layer_name] = concat_acts.mean(dim=0)
        else:
            final_activations[layer_name] = None
    
    return final_activations


def get_layer_by_name(model, layer_name):
    """Get layer by dotted name path."""
    parts = layer_name.split('.')
    layer = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    return layer


def calculate_activation_differences(model1, model2, data_loader, device, data_type, max_batches=3):
    """Calculate activation differences between two models on given data."""
    
    # Get all target layers
    target_layers = get_all_target_layers(model1)
    layer_names = list(target_layers.keys())
    
    print(f"  Computing activations for {data_type} data on {len(layer_names)} layers...")
    
    # Get activations from both models
    act1 = get_activations(model1, data_loader, layer_names, device, max_batches)
    act2 = get_activations(model2, data_loader, layer_names, device, max_batches)
    
    # Calculate differences
    activation_differences = {}
    total_diff_norm = 0
    
    for layer_name in layer_names:
        if layer_name in act1 and layer_name in act2:
            if act1[layer_name] is not None and act2[layer_name] is not None:
                diff = act1[layer_name] - act2[layer_name]
                activation_differences[layer_name] = diff
                total_diff_norm += torch.norm(diff).item()
    
    print(f"  Total activation difference norm on {data_type}: {total_diff_norm:.6f}")
    return activation_differences


def compare_activation_patterns(forget_act_diff, retain_act_diff, verbose=True):
    """Compare activation difference patterns between forget and retain data."""
    
    comparison_results = {}
    all_similarities = []
    
    common_layers = set(forget_act_diff.keys()) & set(retain_act_diff.keys())
    
    for layer_name in common_layers:
        forget_act = forget_act_diff[layer_name]
        retain_act = retain_act_diff[layer_name]
        
        if forget_act is None or retain_act is None:
            continue
        
        try:
            # Flatten activations
            forget_flat = forget_act.flatten()
            retain_flat = retain_act.flatten()
            
            # Check for zero activations
            forget_norm = torch.norm(forget_flat).item()
            retain_norm = torch.norm(retain_flat).item()
            
            if forget_norm < 1e-8 or retain_norm < 1e-8:
                continue
            
            # Cosine similarity
            cosine_sim = torch.cosine_similarity(forget_flat, retain_flat, dim=0).item()
            
            # L2 distance
            l2_dist = torch.norm(forget_flat - retain_flat).item()
            
            # Relative L2 distance
            rel_l2_dist = l2_dist / (forget_norm + retain_norm + 1e-8)
            
            # Mean squared error
            mse = torch.mean((forget_flat - retain_flat) ** 2).item()
            
            # Correlation coefficient
            if len(forget_flat) > 1:
                try:
                    correlation = torch.corrcoef(torch.stack([forget_flat, retain_flat]))[0, 1].item()
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
                'forget_norm': forget_norm,
                'retain_norm': retain_norm
            }
            
            all_similarities.append(cosine_sim)
            
        except Exception as e:
            if verbose:
                print(f"Comparison failed for {layer_name}: {e}")
            continue
    
    if verbose and all_similarities:
        mean_sim = np.mean(all_similarities)
        print(f"  Mean activation pattern similarity: {mean_sim:.4f}")
    
    return comparison_results


def analyze_activation_comparison_results(comparison_results, important_layers, verbose=True):
    """Analyze the activation comparison results."""
    
    similarities = [r['cosine_similarity'] for r in comparison_results.values()]
    
    if not similarities:
        return {
            'success': False,
            'description': 'No valid comparisons computed',
            'mean_similarity': 0.0
        }
    
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    pattern_difference = 1.0 - mean_similarity
    
    print(f"\n🔍 ACTIVATION COMPARISON ANALYSIS")
    print(f"   • Mean Similarity: {mean_similarity:.3f} (±{std_similarity:.3f})")
    print(f"   • Pattern Difference: {pattern_difference:.3f}")
    print(f"   • Layers Analyzed: {len(similarities)}")
    print(f"   • Important Layers Found: {len(important_layers)}")
    
    # Determine success level (stricter thresholds for activations)
    if pattern_difference >= 0.8:
        success_level = "HIGH"
        success = True
        description = f"Forget and retain activation patterns are highly distinct (similarity: {mean_similarity:.3f})"
    elif pattern_difference >= 0.6:
        success_level = "MODERATE"
        success = True
        description = f"Forget and retain activation patterns show clear differences (similarity: {mean_similarity:.3f})"
    elif pattern_difference >= 0.4:
        success_level = "LOW"
        success = False
        description = f"Forget and retain activation patterns show weak differences (similarity: {mean_similarity:.3f})"
    else:
        success_level = "FAILED"
        success = False
        description = f"Forget and retain activation patterns are very similar (similarity: {mean_similarity:.3f})"
    
    print(f"   • Assessment: {success_level}")
    print(f"   • Description: {description}")
    
    return {
        'success': success,
        'success_level': success_level,
        'description': description,
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'pattern_difference': pattern_difference,
        'num_layers': len(similarities),
        'num_important_layers': len(important_layers)
    }


def create_activation_comparison_plots(forget_act_diff, retain_act_diff, comparison_results, 
                                     important_layers, plot_dir, verbose=True):
    """Create visualization plots for activation comparison analysis."""
    
    if verbose:
        print(f"\n🎨 Creating activation comparison plots in {plot_dir}...")
    
    # Plot 1: Layer-wise Similarity Comparison
    create_activation_similarity_plot(comparison_results, important_layers, plot_dir)
    
    # Plot 2: Activation Magnitude Comparison
    create_activation_magnitude_plot(forget_act_diff, retain_act_diff, plot_dir)
    
    # Plot 3: Layer Importance Analysis
    create_layer_importance_plot(important_layers, comparison_results, plot_dir)
    
    # Plot 4: Distribution Analysis
    create_activation_distribution_plot(comparison_results, plot_dir)
    
    if verbose:
        print("✅ All activation comparison plots created!")


def create_activation_similarity_plot(comparison_results, important_layers, plot_dir):
    """Create layer-wise activation similarity comparison plot."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    layers = list(comparison_results.keys())
    similarities = [comparison_results[l]['cosine_similarity'] for l in layers]
    
    # Create importance mapping for color coding
    important_layer_names = [layer for layer, _ in important_layers[:10]] if important_layers else []
    
    # Create bar plot with color coding
    bars = ax.bar(range(len(layers)), similarities, alpha=0.8, edgecolor='black')
    
    # Color code bars based on similarity levels and importance
    for i, (bar, sim, layer) in enumerate(zip(bars, similarities, layers)):
        if layer in important_layer_names:
            # Important layers get special colors
            if sim <= 0.2:
                bar.set_color('darkgreen')  # High difference + important
            elif sim <= 0.4:
                bar.set_color('green')  # Moderate difference + important
            elif sim <= 0.6:
                bar.set_color('orange')  # Low difference + important
            else:
                bar.set_color('darkred')  # High similarity + important (concerning)
        else:
            # Regular layers
            if sim <= 0.2:
                bar.set_color('lightgreen')
            elif sim <= 0.4:
                bar.set_color('yellow')
            elif sim <= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    # Add threshold lines
    ax.axhline(y=0.2, color='green', linestyle='--', linewidth=2, label='High Separation (<0.2)')
    ax.axhline(y=0.4, color='orange', linestyle='--', linewidth=2, label='Moderate Separation (<0.4)')
    ax.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='Low Separation (<0.6)')
    
    ax.set_xlabel('Neural Network Layers')
    ax.set_ylabel('Cosine Similarity (Forget vs Retain Patterns)')
    ax.set_title('Activation Pattern Similarity: Forget vs Retain Data\n(Lower = Better Separation, Dark Colors = Important Layers)')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels for important layers
    for i, (bar, sim, layer) in enumerate(zip(bars, similarities, layers)):
        if layer in important_layer_names:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{sim:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# ACTIVATION SIMILARITY ANALYSIS:
# This plot shows how similar the activation difference patterns are between forget and retain data.
# Dark colored bars indicate layers that changed significantly during unlearning (important layers).

# INTERPRETATION:
# • Lower similarity (↓) = Better separation between forget and retain patterns
# • Higher similarity (↑) = Patterns are more similar, harder to distinguish
# • Dark colors = Layers that changed most during unlearning (focus attention here)

# COLOR CODING:
# • Dark Green: Excellent separation + Important layer - ideal scenario
# • Green/Light Green: Good separation (similarity < 0.4)
# • Orange/Yellow: Moderate separation (similarity < 0.6)
# • Red/Dark Red: Poor separation (similarity > 0.6) - concerning if dark red

# PRIVACY IMPLICATIONS:
# • Good separation on important layers indicates targeted unlearning effects
# • Poor separation on important layers suggests incomplete unlearning
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.savefig(plot_dir / 'activation_similarity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_activation_magnitude_plot(forget_act_diff, retain_act_diff, plot_dir):
    """Create activation magnitude comparison plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate magnitudes
    layers = list(set(forget_act_diff.keys()) & set(retain_act_diff.keys()))
    forget_mags = []
    retain_mags = []
    
    for layer in layers:
        if forget_act_diff[layer] is not None and retain_act_diff[layer] is not None:
            forget_mag = torch.norm(forget_act_diff[layer]).item()
            retain_mag = torch.norm(retain_act_diff[layer]).item()
            forget_mags.append(forget_mag)
            retain_mags.append(retain_mag)
    
    # Plot 1: Side-by-side magnitude comparison
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, forget_mags, width, label='Forget Data', 
                   color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, retain_mags, width, label='Retain Data', 
                   color='skyblue', alpha=0.8)
    
    ax1.set_xlabel('Neural Network Layers')
    ax1.set_ylabel('Activation Difference Magnitude')
    ax1.set_title('Activation Magnitude Comparison: Forget vs Retain')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-scale ratio analysis (activations can vary widely)
    ratios = [f / (r + 1e-8) for f, r in zip(forget_mags, retain_mags)]
    bars = ax2.bar(range(len(layers)), ratios, color='mediumpurple', alpha=0.8)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Equal Magnitude (1.0)')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=1, label='2x Difference')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='0.5x Difference')
    
    ax2.set_xlabel('Neural Network Layers')
    ax2.set_ylabel('Magnitude Ratio (Forget/Retain)')
    ax2.set_title('Activation Magnitude Ratio Analysis')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization of ratios
    
    # Color code ratio bars
    for bar, ratio in zip(bars, ratios):
        if ratio > 3.0 or ratio < 0.33:
            bar.set_color('red')  # High difference
        elif ratio > 1.5 or ratio < 0.67:
            bar.set_color('orange')  # Moderate difference
        else:
            bar.set_color('green')  # Similar magnitudes
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# ACTIVATION MAGNITUDE ANALYSIS:
# Left plot shows the absolute magnitude of activation differences for each layer.
# Right plot shows the ratio of forget/retain magnitudes on log scale (1.0 = equal magnitudes).

# INTERPRETATION:
# • Left: Higher bars indicate larger activation changes for that data type on that layer
# • Right: Ratios far from 1.0 indicate differential effects between forget and retain data
# • Log scale helps visualize both large and small ratio differences

# INSIGHTS:
# • Large magnitude differences suggest the unlearning process affects representations differently
# • Extreme ratios (>3x or <0.33x) indicate strong differential effects
# • Consistent patterns across layers suggest systematic unlearning behavior
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.savefig(plot_dir / 'activation_magnitude_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_layer_importance_plot(important_layers, comparison_results, plot_dir):
    """Create layer importance analysis plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Layer importance ranking
    if important_layers:
        top_layers = important_layers[:15]  # Show top 15
        layer_names = [layer for layer, _ in top_layers]
        importance_values = [importance for _, importance in top_layers]
        
        bars = ax1.bar(range(len(layer_names)), importance_values, 
                      color='steelblue', alpha=0.8, edgecolor='navy')
        ax1.set_xlabel('Layer Rank (Most Changed → Least Changed)')
        ax1.set_ylabel('Change Magnitude During Unlearning')
        ax1.set_title('Layer Importance: Which Layers Changed Most?')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels([l.replace('layer', 'L') for l in layer_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add cumulative percentage
        total_importance = sum(importance for _, importance in important_layers)
        cumulative_pct = 0
        for i, (bar, (_, importance)) in enumerate(zip(bars, top_layers)):
            cumulative_pct += (importance / total_importance) * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance_values)*0.01,
                    f'{cumulative_pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Importance vs Similarity for top layers
    if important_layers and comparison_results:
        top_layer_names = [layer for layer, _ in important_layers[:10]]
        similarities = []
        importances = []
        
        for layer, importance in important_layers[:10]:
            if layer in comparison_results:
                similarities.append(comparison_results[layer]['cosine_similarity'])
                importances.append(importance)
        
        if similarities and importances:
            scatter = ax2.scatter(importances, similarities, c=range(len(similarities)), 
                                cmap='viridis', s=100, alpha=0.8, edgecolors='black')
            
            # Add layer labels
            for i, (imp, sim, layer) in enumerate(zip(importances, similarities, top_layer_names)):
                ax2.annotate(layer.replace('layer', 'L'), (imp, sim), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax2.set_xlabel('Layer Importance (Change Magnitude)')
            ax2.set_ylabel('Pattern Similarity (Forget vs Retain)')
            ax2.set_title('Importance vs Similarity for Top Changed Layers')
            ax2.grid(True, alpha=0.3)
            
            # Add ideal region
            ax2.axhspan(0, 0.4, alpha=0.2, color='green', label='Good Separation (<0.4)')
            ax2.axhspan(0.4, 1.0, alpha=0.2, color='red', label='Poor Separation (>0.4)')
            ax2.legend()
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Layer Rank')
    
    plt.tight_layout()
    
    # Add description
#     description = """
# LAYER IMPORTANCE ANALYSIS:
# Left: Ranking of layers by how much they changed during unlearning
# Right: Relationship between layer importance and pattern similarity

# INTERPRETATION:
# • Left plot: Higher bars = layers that changed more during unlearning
# • Right plot: Important layers should ideally have low similarity (good separation)
# • Percentages show cumulative variance explained by top layers

# KEY INSIGHTS:
# • Most important layers (highest change) should show good forget/retain separation
# • Points in green region (right plot) indicate well-functioning unlearning
# • Points in red region may indicate incomplete or problematic unlearning

# PRIVACY IMPLICATIONS:
# • Important layers with poor separation may leak information about forgotten data
# • Good separation on important layers suggests effective targeted unlearning
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.savefig(plot_dir / 'activation_layer_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_activation_distribution_plot(comparison_results, plot_dir):
    """Create distribution analysis plot for activations."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract metrics
    similarities = [r['cosine_similarity'] for r in comparison_results.values()]
    l2_distances = [r['l2_distance'] for r in comparison_results.values()]
    correlations = [r['correlation'] for r in comparison_results.values()]
    mse_values = [r['mse'] for r in comparison_results.values()]
    
    # Plot 1: Similarity distribution
    ax1.hist(similarities, bins=15, color='skyblue', alpha=0.8, edgecolor='black')
    ax1.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(similarities):.3f}')
    ax1.axvline(0.4, color='orange', linestyle='--', linewidth=2, label='Separation Threshold')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Number of Layers')
    ax1.set_title('Distribution of Activation Pattern Similarities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: L2 distance distribution (log scale for activations)
    ax2.hist(l2_distances, bins=15, color='lightcoral', alpha=0.8, edgecolor='black')
    ax2.axvline(np.mean(l2_distances), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(l2_distances):.3e}')
    ax2.set_xlabel('L2 Distance')
    ax2.set_ylabel('Number of Layers')
    ax2.set_title('Distribution of L2 Distances')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correlation distribution
    ax3.hist(correlations, bins=15, color='lightgreen', alpha=0.8, edgecolor='black')
    ax3.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(correlations):.3f}')
    ax3.axvline(0.0, color='black', linestyle='-', linewidth=1, label='No Correlation')
    ax3.set_xlabel('Correlation Coefficient')
    ax3.set_ylabel('Number of Layers')
    ax3.set_title('Distribution of Correlations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: MSE distribution
    ax4.hist(mse_values, bins=15, color='mediumpurple', alpha=0.8, edgecolor='black')
    ax4.axvline(np.mean(mse_values), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(mse_values):.3e}')
    ax4.set_xlabel('Mean Squared Error')
    ax4.set_ylabel('Number of Layers')
    ax4.set_title('Distribution of MSE Values')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# ACTIVATION DISTRIBUTION ANALYSIS:
# These histograms show the distribution of similarity metrics across all network layers.

# INTERPRETATION:
# • Cosine Similarity: How similar activation patterns are (lower = better separation)
# • L2 Distance: Absolute difference between patterns (higher = better separation)
# • Correlation: Statistical relationship (closer to 0 = better separation)
# • MSE: Mean squared error between patterns (higher = better separation)

# LOG SCALES:
# • L2 Distance and MSE use log scales due to wide ranges in activation magnitudes
# • This helps visualize both small and large differences effectively

# INSIGHTS:
# • Distributions skewed toward separation indicate good unlearning
# • Bimodal distributions may indicate different layer types respond differently
# • Wide distributions suggest heterogeneous effects across the network
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.savefig(plot_dir / 'activation_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()