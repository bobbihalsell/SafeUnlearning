import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Set consistent plot style
plt.style.use('default')
sns.set_palette("husl")

def run_gradient_comparison(original_model, unlearned_model, forget_data, retain_data, 
                           device='cuda', verbose=True):
    """
    Compare gradient differences between original and unlearned models on forget vs retain data.
    
    This directly compares:
    - Forget Pattern: Gradient differences (Original - Unlearned) when processing FORGET data
    - Retain Pattern: Gradient differences (Original - Unlearned) when processing RETAIN data
    
    Returns similarity metrics and creates visualization plots.
    """
    
    print("GRADIENT COMPARISON ANALYSIS")
    print("="*60)
    print("Direct comparison of gradient differences on forget vs retain data")
    print("="*60)
    
    # Step 1: Calculate gradient differences on forget data
    print("\nStep 1: Computing gradient differences on forget data...")
    forget_grad_diff = calculate_gradient_differences(
        original_model, unlearned_model, forget_data, device, "forget"
    )
    
    # Step 2: Calculate gradient differences on retain data
    print("\nStep 2: Computing gradient differences on retain data...")
    retain_grad_diff = calculate_gradient_differences(
        original_model, unlearned_model, retain_data, device, "retain"
    )
    
    # Step 3: Compare the gradient difference patterns
    print("\nStep 3: Comparing forget vs retain gradient difference patterns...")
    comparison_results = compare_gradient_patterns(forget_grad_diff, retain_grad_diff, verbose)
    
    # Step 4: Create plots
    plot_dir = Path("./attack_plots/comparison")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_gradient_comparison_plots(
        forget_grad_diff, retain_grad_diff, comparison_results, plot_dir, verbose
    )
    
    # Step 5: Analyze results
    analysis_results = analyze_gradient_comparison_results(comparison_results, verbose)
    
    return {
        'forget_grad_diff': forget_grad_diff,
        'retain_grad_diff': retain_grad_diff,
        'comparison_results': comparison_results,
        'analysis': analysis_results,
        'plot_directory': str(plot_dir)
    }


def calculate_gradient_differences(model1, model2, data_loader, device, data_type, max_batches=3):
    """Calculate gradient differences between two models on given data."""
    
    print(f"  Computing gradients for {data_type} data...")
    
    # Get gradients from both models
    grad1 = get_model_gradients(model1, data_loader, device, max_batches)
    grad2 = get_model_gradients(model2, data_loader, device, max_batches)
    
    # Calculate differences
    grad_differences = {}
    total_diff_norm = 0
    
    for param_name in grad1:
        if param_name in grad2 and '.weight' in param_name:
            diff = grad1[param_name] - grad2[param_name]
            layer_name = param_name.replace('.weight', '')
            grad_differences[layer_name] = diff
            total_diff_norm += torch.norm(diff).item()
    
    print(f"  Total gradient difference norm on {data_type}: {total_diff_norm:.6f}")
    return grad_differences


def get_model_gradients(model, data_loader, device, max_batches=3):
    """Get gradients for a model on given data."""
    
    loss_fn = nn.CrossEntropyLoss()
    original_mode = model.training
    model.train()
    
    accumulated_gradients = {}
    num_batches = 0
    
    try:
        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            
            images, labels = images.to(device), labels.to(device)
            
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
            
            # Clean up
            del images, labels, outputs, loss
            torch.cuda.empty_cache()
    
    finally:
        model.train(original_mode)
    
    # Average by number of batches
    for name in accumulated_gradients:
        accumulated_gradients[name] = accumulated_gradients[name] / num_batches
    
    return accumulated_gradients


def compare_gradient_patterns(forget_grad_diff, retain_grad_diff, verbose=True):
    """Compare gradient difference patterns between forget and retain data."""
    
    comparison_results = {}
    all_similarities = []
    
    common_layers = set(forget_grad_diff.keys()) & set(retain_grad_diff.keys())
    
    for layer_name in common_layers:
        forget_grad = forget_grad_diff[layer_name]
        retain_grad = retain_grad_diff[layer_name]
        
        if forget_grad is None or retain_grad is None:
            continue
        
        try:
            # Flatten gradients
            forget_flat = forget_grad.flatten()
            retain_flat = retain_grad.flatten()
            
            # Check for zero gradients
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
        print(f"  Mean gradient pattern similarity: {mean_sim:.4f}")
    
    return comparison_results


def analyze_gradient_comparison_results(comparison_results, verbose=True):
    """Analyze the gradient comparison results."""
    
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
    
    print(f"\n🔍 GRADIENT COMPARISON ANALYSIS")
    print(f"   • Mean Similarity: {mean_similarity:.3f} (±{std_similarity:.3f})")
    print(f"   • Pattern Difference: {pattern_difference:.3f}")
    print(f"   • Layers Analyzed: {len(similarities)}")
    
    # Determine success level
    if pattern_difference >= 0.7:
        success_level = "HIGH"
        success = True
        description = f"Forget and retain gradient patterns are highly distinct (similarity: {mean_similarity:.3f})"
    elif pattern_difference >= 0.5:
        success_level = "MODERATE"
        success = True
        description = f"Forget and retain gradient patterns show clear differences (similarity: {mean_similarity:.3f})"
    elif pattern_difference >= 0.3:
        success_level = "LOW"
        success = False
        description = f"Forget and retain gradient patterns show weak differences (similarity: {mean_similarity:.3f})"
    else:
        success_level = "FAILED"
        success = False
        description = f"Forget and retain gradient patterns are very similar (similarity: {mean_similarity:.3f})"
    
    print(f"   • Assessment: {success_level}")
    print(f"   • Description: {description}")
    
    return {
        'success': success,
        'success_level': success_level,
        'description': description,
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'pattern_difference': pattern_difference,
        'num_layers': len(similarities)
    }


def create_gradient_comparison_plots(forget_grad_diff, retain_grad_diff, comparison_results, 
                                   plot_dir, verbose=True):
    """Create visualization plots for gradient comparison analysis."""
    
    if verbose:
        print(f"\n🎨 Creating gradient comparison plots in {plot_dir}...")
    
    # Plot 1: Layer-wise Similarity Comparison
    create_similarity_plot(comparison_results, plot_dir)
    
    # Plot 2: Gradient Magnitude Comparison
    create_magnitude_comparison_plot(forget_grad_diff, retain_grad_diff, plot_dir)
    
    # Plot 3: Distribution Analysis
    create_distribution_plot(comparison_results, plot_dir)
    
    # Plot 4: Correlation Analysis
    create_correlation_plot(comparison_results, plot_dir)
    
    if verbose:
        print("✅ All gradient comparison plots created!")


def create_similarity_plot(comparison_results, plot_dir):
    """Create layer-wise similarity comparison plot."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    layers = list(comparison_results.keys())
    similarities = [comparison_results[l]['cosine_similarity'] for l in layers]
    
    # Create bar plot with color coding
    bars = ax.bar(range(len(layers)), similarities, alpha=0.8, edgecolor='black')
    
    # Color code bars based on similarity levels
    for bar, sim in zip(bars, similarities):
        if sim <= 0.3:
            bar.set_color('green')  # High difference = good separation
        elif sim <= 0.5:
            bar.set_color('orange')  # Moderate difference
        elif sim <= 0.7:
            bar.set_color('yellow')  # Low difference
        else:
            bar.set_color('red')  # High similarity = poor separation
    
    # Add threshold lines
    ax.axhline(y=0.3, color='green', linestyle='--', linewidth=2, label='High Separation (<0.3)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Moderate Separation (<0.5)')
    ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Low Separation (<0.7)')
    
    ax.set_xlabel('Neural Network Layers')
    ax.set_ylabel('Cosine Similarity (Forget vs Retain Patterns)')
    ax.set_title('Gradient Pattern Similarity: Forget vs Retain Data\n(Lower = Better Separation)')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, sim in zip(bars, similarities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{sim:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# GRADIENT SIMILARITY ANALYSIS:
# This plot shows how similar the gradient difference patterns are between forget and retain data.

# INTERPRETATION:
# • Lower similarity (↓) = Better separation between forget and retain patterns
# • Higher similarity (↑) = Patterns are more similar, harder to distinguish

# COLOR CODING:
# • Green bars: Excellent separation (similarity < 0.3) - forget and retain have very different gradient patterns
# • Orange bars: Good separation (similarity < 0.5) - clear differences between patterns  
# • Yellow bars: Weak separation (similarity < 0.7) - some differences but not strong
# • Red bars: Poor separation (similarity > 0.7) - patterns are very similar

# PRIVACY IMPLICATIONS:
# • Good separation indicates the unlearning process affects forget and retain data differently
# • Poor separation suggests similar processing, which may indicate incomplete unlearning
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.savefig(plot_dir / 'gradient_similarity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_magnitude_comparison_plot(forget_grad_diff, retain_grad_diff, plot_dir):
    """Create gradient magnitude comparison plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate magnitudes
    layers = list(set(forget_grad_diff.keys()) & set(retain_grad_diff.keys()))
    forget_mags = []
    retain_mags = []
    
    for layer in layers:
        forget_mag = torch.norm(forget_grad_diff[layer]).item()
        retain_mag = torch.norm(retain_grad_diff[layer]).item()
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
    ax1.set_ylabel('Gradient Difference Magnitude')
    ax1.set_title('Gradient Magnitude Comparison: Forget vs Retain')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio analysis
    ratios = [f / (r + 1e-8) for f, r in zip(forget_mags, retain_mags)]
    bars = ax2.bar(range(len(layers)), ratios, color='mediumpurple', alpha=0.8)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Equal Magnitude (1.0)')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=1, label='2x Difference')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='0.5x Difference')
    
    ax2.set_xlabel('Neural Network Layers')
    ax2.set_ylabel('Magnitude Ratio (Forget/Retain)')
    ax2.set_title('Gradient Magnitude Ratio Analysis')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Color code ratio bars
    for bar, ratio in zip(bars, ratios):
        if ratio > 2.0 or ratio < 0.5:
            bar.set_color('red')  # High difference
        elif ratio > 1.5 or ratio < 0.67:
            bar.set_color('orange')  # Moderate difference
        else:
            bar.set_color('green')  # Similar magnitudes
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# GRADIENT MAGNITUDE ANALYSIS:
# Left plot shows the absolute magnitude of gradient differences for each layer.
# Right plot shows the ratio of forget/retain magnitudes (1.0 = equal magnitudes).

# INTERPRETATION:
# • Left: Higher bars indicate larger gradient changes for that data type on that layer
# • Right: Ratios far from 1.0 indicate differential effects between forget and retain data

# PRIVACY INSIGHTS:
# • Large magnitude differences suggest the unlearning process affects layers differently
# • Ratios significantly different from 1.0 indicate selective unlearning effects
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.savefig(plot_dir / 'gradient_magnitude_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_distribution_plot(comparison_results, plot_dir):
    """Create distribution analysis plot."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract metrics
    similarities = [r['cosine_similarity'] for r in comparison_results.values()]
    l2_distances = [r['l2_distance'] for r in comparison_results.values()]
    correlations = [r['correlation'] for r in comparison_results.values()]
    rel_distances = [r['relative_l2_distance'] for r in comparison_results.values()]
    
    # Plot 1: Similarity distribution
    ax1.hist(similarities, bins=15, color='skyblue', alpha=0.8, edgecolor='black')
    ax1.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(similarities):.3f}')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Number of Layers')
    ax1.set_title('Distribution of Gradient Pattern Similarities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: L2 distance distribution
    ax2.hist(l2_distances, bins=15, color='lightcoral', alpha=0.8, edgecolor='black')
    ax2.axvline(np.mean(l2_distances), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(l2_distances):.3f}')
    ax2.set_xlabel('L2 Distance')
    ax2.set_ylabel('Number of Layers')
    ax2.set_title('Distribution of L2 Distances')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correlation distribution
    ax3.hist(correlations, bins=15, color='lightgreen', alpha=0.8, edgecolor='black')
    ax3.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(correlations):.3f}')
    ax3.set_xlabel('Correlation Coefficient')
    ax3.set_ylabel('Number of Layers')
    ax3.set_title('Distribution of Correlations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Relative distance distribution
    ax4.hist(rel_distances, bins=15, color='mediumpurple', alpha=0.8, edgecolor='black')
    ax4.axvline(np.mean(rel_distances), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rel_distances):.3f}')
    ax4.set_xlabel('Relative L2 Distance')
    ax4.set_ylabel('Number of Layers')
    ax4.set_title('Distribution of Relative Distances')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# DISTRIBUTION ANALYSIS:
# These histograms show the distribution of similarity metrics across all network layers.

# INTERPRETATION:
# • Cosine Similarity: How similar the gradient patterns are (lower = better separation)
# • L2 Distance: Absolute difference between patterns (higher = better separation)  
# • Correlation: Statistical relationship between patterns (lower absolute value = better separation)
# • Relative L2 Distance: Normalized difference (accounts for magnitude differences)

# INSIGHTS:
# • Wide distributions suggest heterogeneous effects across layers
# • Narrow distributions indicate consistent behavior across the network
# • Mean values provide overall assessment of forget vs retain separability
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.savefig(plot_dir / 'gradient_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_correlation_plot(comparison_results, plot_dir):
    """Create correlation analysis plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    layers = list(comparison_results.keys())
    forget_norms = [comparison_results[l]['forget_norm'] for l in layers]
    retain_norms = [comparison_results[l]['retain_norm'] for l in layers]
    similarities = [comparison_results[l]['cosine_similarity'] for l in layers]
    l2_distances = [comparison_results[l]['l2_distance'] for l in layers]
    
    # Plot 1: Norm comparison
    ax1.scatter(forget_norms, retain_norms, c=similarities, cmap='RdYlBu_r', 
               s=60, alpha=0.8, edgecolors='black')
    
    # Add diagonal line
    max_norm = max(max(forget_norms), max(retain_norms))
    ax1.plot([0, max_norm], [0, max_norm], 'k--', alpha=0.5, label='Equal Magnitude')
    
    ax1.set_xlabel('Forget Gradient Norm')
    ax1.set_ylabel('Retain Gradient Norm')
    ax1.set_title('Gradient Magnitude Correlation\n(Color = Similarity)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Cosine Similarity')
    
    # Plot 2: Similarity vs Distance relationship
    ax2.scatter(similarities, l2_distances, c=range(len(similarities)), 
               cmap='viridis', s=60, alpha=0.8, edgecolors='black')
    
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('L2 Distance')
    ax2.set_title('Similarity vs Distance Relationship\n(Color = Layer Index)')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label('Layer Index')
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# CORRELATION ANALYSIS:
# Left: Relationship between forget and retain gradient magnitudes (points colored by similarity)
# Right: Relationship between similarity and distance metrics (points colored by layer depth)

# INTERPRETATION:
# • Left plot: Points near diagonal have similar magnitudes; color shows pattern similarity
# • Right plot: Generally, lower similarity should correspond to higher L2 distance

# INSIGHTS:
# • Points far from diagonal in left plot indicate differential magnitude effects
# • Strong negative correlation in right plot indicates consistent metrics
# • Color patterns may reveal layer-depth dependencies in unlearning effects
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.savefig(plot_dir / 'gradient_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()