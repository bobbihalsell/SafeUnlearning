import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime
import hashlib

# Set consistent plot style
plt.style.use('default')
sns.set_palette("husl")

def run_subspace_comparison(original_model, unlearned_model, forget_data, retain_data, 
                           device='cuda', variance_threshold=0.9, verbose=True):
    """
    Compare gradient subspaces between original and unlearned models on forget vs retain data.
    
    This directly compares:
    - Forget Pattern: Gradient subspaces from (Original - Unlearned) when processing FORGET data
    - Retain Pattern: Gradient subspaces from (Original - Unlearned) when processing RETAIN data
    
    Returns subspace similarity metrics and creates visualization plots.
    """
    
    print("SUBSPACE COMPARISON ANALYSIS")
    print("="*60)
    print("Direct comparison of gradient subspaces on forget vs retain data")
    print("="*60)
    
    # Step 1: Calculate gradient covariances on forget data
    print("\nStep 1: Computing gradient covariances on forget data...")
    forget_grad_cov = calculate_gradient_covariances(
        original_model, unlearned_model, forget_data, device, "forget"
    )
    
    # Step 2: Calculate gradient covariances on retain data
    print("\nStep 2: Computing gradient covariances on retain data...")
    retain_grad_cov = calculate_gradient_covariances(
        original_model, unlearned_model, retain_data, device, "retain"
    )
    
    # Step 3: Extract subspaces using SVD
    print("\nStep 3: Extracting gradient subspaces...")
    forget_subspaces = extract_subspaces(forget_grad_cov, variance_threshold, "forget")
    retain_subspaces = extract_subspaces(retain_grad_cov, variance_threshold, "retain")
    
    # Step 4: Compare the subspaces
    print("\nStep 4: Comparing forget vs retain gradient subspaces...")
    comparison_results = compare_subspaces(forget_subspaces, retain_subspaces, verbose)
    
    # Step 5: Create plots
    plot_dir = Path("./attack_plots/comparison")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_subspace_comparison_plots(
        forget_subspaces, retain_subspaces, comparison_results, plot_dir, verbose
    )
    
    # Step 6: Analyze results
    analysis_results = analyze_subspace_comparison_results(comparison_results, verbose)
    
    return {
        'forget_subspaces': forget_subspaces,
        'retain_subspaces': retain_subspaces,
        'comparison_results': comparison_results,
        'analysis': analysis_results,
        'plot_directory': str(plot_dir)
    }


def calculate_gradient_covariances(model1, model2, data_loader, device, data_type, max_batches=3):
    """Calculate covariances of gradient differences between two models."""
    
    print(f"  Computing gradient covariances for {data_type} data...")
    
    # Get gradients from both models
    grad1 = get_model_gradients(model1, data_loader, device, max_batches)
    grad2 = get_model_gradients(model2, data_loader, device, max_batches)
    
    # Calculate gradient differences
    grad_differences = {}
    for param_name in grad1:
        if param_name in grad2 and '.weight' in param_name:
            diff = grad1[param_name] - grad2[param_name]
            layer_name = param_name.replace('.weight', '')
            grad_differences[layer_name] = diff
    
    # Convert to covariance matrices
    gradient_covariances = {}
    
    for layer_name, grad_diff in grad_differences.items():
        try:
            # Move to CPU and flatten
            grad_diff_cpu = grad_diff.detach().cpu().float()
            grad_flat = grad_diff_cpu.flatten()
            num_params = grad_flat.shape[0]
            
            # Memory-efficient covariance computation
            if num_params > 2048:  # For large layers, use approximation
                print(f"    Layer {layer_name}: Using approximation for {num_params} parameters")
                
                # Random projection to reduce dimensionality
                target_dim = min(512, num_params // 4)
                torch.manual_seed(42)  # For reproducibility
                proj_matrix = torch.randn(target_dim, num_params) / np.sqrt(target_dim)
                
                # Project gradient to lower dimension
                grad_projected = torch.mv(proj_matrix, grad_flat)
                
                # Create covariance in projected space
                grad_cov = torch.outer(grad_projected, grad_projected)
                
            else:
                print(f"    Layer {layer_name}: Using full covariance for {num_params} parameters")
                
                # Create full covariance matrix
                grad_cov = torch.outer(grad_flat, grad_flat)
            
            gradient_covariances[layer_name] = grad_cov.double()
            
        except Exception as e:
            print(f"    Failed to compute covariance for {layer_name}: {e}")
            continue
    
    total_cov_norm = sum(torch.norm(cov).item() for cov in gradient_covariances.values())
    print(f"  Total gradient covariance norm on {data_type}: {total_cov_norm:.6f}")
    
    return gradient_covariances


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
            
            # Accumulate gradients (only weights)
            for name, param in model.named_parameters():
                if param.grad is not None and '.weight' in name:
                    if name not in accumulated_gradients:
                        accumulated_gradients[name] = param.grad.detach().clone()
                    else:
                        accumulated_gradients[name] += param.grad.detach().clone()
            
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


def extract_subspaces(covariance_dict, variance_threshold=0.9, data_type="data"):
    """Extract subspaces from gradient covariance matrices using SVD."""
    
    print(f"  Extracting subspaces from {data_type} covariances...")
    
    subspace_results = {}
    
    for layer_name, cov_matrix in covariance_dict.items():
        try:
            # Ensure we're working on CPU
            if cov_matrix.device.type == 'cuda':
                cov_matrix = cov_matrix.cpu()
            
            # Add epsilon for numerical stability
            stabilized = cov_matrix + 1e-6 * torch.eye(cov_matrix.shape[0], 
                                                      dtype=cov_matrix.dtype)
            
            print(f"    Computing SVD for layer {layer_name}: shape {stabilized.shape}")
            
            # SVD decomposition
            U, S, V = torch.svd(stabilized)
            
            # Apply variance threshold
            if len(S) > 1:
                cumulative_var = torch.cumsum(S, dim=0) / torch.sum(S)
                components_to_keep = torch.sum(cumulative_var <= variance_threshold).item()
                components_to_keep = max(1, min(components_to_keep + 1, len(S)))
            else:
                components_to_keep = min(10, len(S))
            
            # Truncate
            U_truncated = U[:, :components_to_keep]
            S_truncated = S[:components_to_keep]
            
            subspace_results[layer_name] = {
                'U': U_truncated,
                'S': S_truncated,
                'explained_variance_ratio': (S_truncated.sum() / S.sum()).item(),
                'num_components': components_to_keep,
                'total_components': len(S)
            }
            
            var_explained = (S_truncated.sum() / S.sum()).item()
            print(f"    Layer {layer_name}: kept {components_to_keep}/{len(S)} components, "
                  f"explaining {var_explained:.3f} variance")
            
        except Exception as e:
            print(f"    SVD failed for layer {layer_name}: {e}")
            continue
    
    return subspace_results


def compare_subspaces(forget_subspaces, retain_subspaces, verbose=True):
    """Compare gradient subspaces between forget and retain data."""
    
    comparison_results = {}
    common_layers = set(forget_subspaces.keys()) & set(retain_subspaces.keys())
    
    print(f"  Comparing subspaces for {len(common_layers)} common layers...")
    
    for layer_name in common_layers:
        try:
            forget_U = forget_subspaces[layer_name]['U']
            retain_U = retain_subspaces[layer_name]['U']
            
            # Ensure both are on CPU
            if forget_U.device.type == 'cuda':
                forget_U = forget_U.cpu()
            if retain_U.device.type == 'cuda':
                retain_U = retain_U.cpu()
            
            # Handle dimension mismatch
            min_components = min(forget_U.shape[1], retain_U.shape[1])
            if min_components == 0:
                continue
            
            forget_U_trunc = forget_U[:, :min_components]
            retain_U_trunc = retain_U[:, :min_components]
            
            # Handle feature dimension mismatch
            if forget_U.shape[0] != retain_U.shape[0]:
                min_feat_dim = min(forget_U.shape[0], retain_U.shape[0])
                forget_U_trunc = forget_U_trunc[:min_feat_dim, :]
                retain_U_trunc = retain_U_trunc[:min_feat_dim, :]
            
            if forget_U_trunc.shape[0] == 0 or forget_U_trunc.shape[1] == 0:
                continue
            
            # Compute subspace similarities
            
            # 1. Principal angles (canonical angles between subspaces)
            overlap_matrix = forget_U_trunc.T @ retain_U_trunc
            _, S_overlap, _ = torch.svd(overlap_matrix)
            cos_angles = torch.clamp(S_overlap, 0, 1)
            
            # 2. Subspace overlap (trace of projection)
            P_forget = forget_U_trunc @ forget_U_trunc.T
            P_retain = retain_U_trunc @ retain_U_trunc.T
            subspace_overlap = torch.trace(P_forget @ P_retain).item() / min_components
            
            # 3. Frobenius norm distance between projections
            proj_distance = torch.norm(P_forget - P_retain, p='fro').item()
            
            # 4. Grassmann distance (geodesic distance on Grassmann manifold)
            # Approximation using principal angles
            angles_rad = torch.acos(cos_angles)
            grassmann_distance = torch.norm(angles_rad).item()
            
            comparison_results[layer_name] = {
                'mean_cosine_angle': cos_angles.mean().item(),
                'max_cosine_angle': cos_angles.max().item(),
                'min_cosine_angle': cos_angles.min().item(),
                'subspace_overlap': subspace_overlap,
                'projection_distance': proj_distance,
                'grassmann_distance': grassmann_distance,
                'mean_angle_degrees': torch.acos(cos_angles.mean()).item() * 180 / np.pi,
                'num_components_forget': forget_subspaces[layer_name]['num_components'],
                'num_components_retain': retain_subspaces[layer_name]['num_components'],
                'variance_explained_forget': forget_subspaces[layer_name]['explained_variance_ratio'],
                'variance_explained_retain': retain_subspaces[layer_name]['explained_variance_ratio']
            }
            
        except Exception as e:
            if verbose:
                print(f"    Comparison failed for {layer_name}: {e}")
            continue
    
    if comparison_results and verbose:
        mean_overlap = np.mean([r['subspace_overlap'] for r in comparison_results.values()])
        print(f"  Mean subspace overlap: {mean_overlap:.4f}")
    
    return comparison_results


def analyze_subspace_comparison_results(comparison_results, verbose=True):
    """Analyze the subspace comparison results."""
    
    if not comparison_results:
        return {
            'success': False,
            'description': 'No valid subspace comparisons computed',
            'mean_overlap': 0.0
        }
    
    overlaps = [r['subspace_overlap'] for r in comparison_results.values()]
    grassmann_distances = [r['grassmann_distance'] for r in comparison_results.values()]
    mean_angles = [r['mean_angle_degrees'] for r in comparison_results.values()]
    
    mean_overlap = np.mean(overlaps)
    mean_grassmann = np.mean(grassmann_distances) 
    mean_angle = np.mean(mean_angles)
    
    # Convert to interpretable metrics
    subspace_difference = 1.0 - mean_overlap  # Higher = more different = better separation
    
    print(f"\n🔍 SUBSPACE COMPARISON ANALYSIS")
    print(f"   • Mean Subspace Overlap: {mean_overlap:.3f}")
    print(f"   • Mean Subspace Difference: {subspace_difference:.3f}")
    print(f"   • Mean Grassmann Distance: {mean_grassmann:.3f}")
    print(f"   • Mean Principal Angle: {mean_angle:.1f}°")
    print(f"   • Layers Analyzed: {len(comparison_results)}")
    
    # Determine success level (stricter thresholds for subspaces)
    if subspace_difference >= 0.8:  # overlap <= 0.2
        success_level = "HIGH"
        success = True
        description = f"Forget and retain subspaces are highly distinct (overlap: {mean_overlap:.3f}, angle: {mean_angle:.1f}°)"
    elif subspace_difference >= 0.6:  # overlap <= 0.4
        success_level = "MODERATE"
        success = True
        description = f"Forget and retain subspaces show clear differences (overlap: {mean_overlap:.3f}, angle: {mean_angle:.1f}°)"
    elif subspace_difference >= 0.4:  # overlap <= 0.6
        success_level = "LOW"
        success = False
        description = f"Forget and retain subspaces show weak differences (overlap: {mean_overlap:.3f}, angle: {mean_angle:.1f}°)"
    else:
        success_level = "FAILED"
        success = False
        description = f"Forget and retain subspaces are very similar (overlap: {mean_overlap:.3f}, angle: {mean_angle:.1f}°)"
    
    print(f"   • Assessment: {success_level}")
    print(f"   • Description: {description}")
    
    return {
        'success': success,
        'success_level': success_level,
        'description': description,
        'mean_overlap': mean_overlap,
        'mean_grassmann_distance': mean_grassmann,
        'mean_angle_degrees': mean_angle,
        'subspace_difference': subspace_difference,
        'num_layers': len(comparison_results)
    }


def create_subspace_comparison_plots(forget_subspaces, retain_subspaces, comparison_results, 
                                   plot_dir, verbose=True):
    """Create visualization plots for subspace comparison analysis."""
    
    if verbose:
        print(f"\n🎨 Creating subspace comparison plots in {plot_dir}...")
    
    # Plot 1: Subspace Overlap Analysis
    create_subspace_overlap_plot(comparison_results, plot_dir)
    
    # Plot 2: Principal Angles Analysis
    create_principal_angles_plot(comparison_results, plot_dir)
    
    # Plot 3: Subspace Dimensions Comparison
    create_dimensions_comparison_plot(forget_subspaces, retain_subspaces, plot_dir)
    
    # Plot 4: Grassmann Distance Analysis
    create_grassmann_distance_plot(comparison_results, plot_dir)
    
    if verbose:
        print("✅ All subspace comparison plots created!")


def create_subspace_overlap_plot(comparison_results, plot_dir):
    """Create subspace overlap comparison plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    layers = list(comparison_results.keys())
    overlaps = [comparison_results[l]['subspace_overlap'] for l in layers]
    angles = [comparison_results[l]['mean_angle_degrees'] for l in layers]
    
    # Plot 1: Subspace overlaps
    bars = ax1.bar(range(len(layers)), overlaps, alpha=0.8, edgecolor='black')
    
    # Color code bars based on overlap levels
    for bar, overlap in zip(bars, overlaps):
        if overlap <= 0.2:
            bar.set_color('green')  # Low overlap = good separation
        elif overlap <= 0.4:
            bar.set_color('orange')  # Moderate overlap
        elif overlap <= 0.6:
            bar.set_color('yellow')  # High overlap
        else:
            bar.set_color('red')  # Very high overlap = poor separation
    
    # Add threshold lines
    ax1.axhline(y=0.2, color='green', linestyle='--', linewidth=2, label='Excellent Separation (<0.2)')
    ax1.axhline(y=0.4, color='orange', linestyle='--', linewidth=2, label='Good Separation (<0.4)')
    ax1.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='Poor Separation (>0.6)')
    
    ax1.set_xlabel('Neural Network Layers')
    ax1.set_ylabel('Subspace Overlap (Forget vs Retain)')
    ax1.set_title('Gradient Subspace Overlap Analysis\n(Lower = Better Separation)')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, overlap in zip(bars, overlaps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{overlap:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Principal angles
    bars2 = ax2.bar(range(len(layers)), angles, alpha=0.8, edgecolor='black')
    
    # Color code bars based on angles
    for bar, angle in zip(bars2, angles):
        if angle >= 60:
            bar.set_color('green')  # Large angles = good separation
        elif angle >= 30:
            bar.set_color('orange')  # Moderate angles
        elif angle >= 15:
            bar.set_color('yellow')  # Small angles
        else:
            bar.set_color('red')  # Very small angles = poor separation
    
    # Add threshold lines
    ax2.axhline(y=60, color='green', linestyle='--', linewidth=2, label='Excellent (>60°)')
    ax2.axhline(y=30, color='orange', linestyle='--', linewidth=2, label='Good (>30°)')
    ax2.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Poor (<15°)')
    
    ax2.set_xlabel('Neural Network Layers')
    ax2.set_ylabel('Mean Principal Angle (Degrees)')
    ax2.set_title('Principal Angles Between Subspaces\n(Higher = Better Separation)')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, angle in zip(bars2, angles):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{angle:.1f}°', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# SUBSPACE OVERLAP ANALYSIS:
# Left: Subspace overlap measures how much the forget and retain gradient subspaces overlap
# Right: Principal angles measure the geometric separation between subspaces

# INTERPRETATION:
# • Lower overlap (↓) = Better separation between forget and retain subspaces
# • Higher angles (↑) = Better geometric separation between subspaces
# • Good separation indicates the unlearning affects data types in different gradient directions

# GEOMETRIC MEANING:
# • Overlap = 0: Completely orthogonal subspaces (perfect separation)
# • Overlap = 1: Identical subspaces (no separation)
# • Angle = 90°: Orthogonal subspaces (perfect separation)
# • Angle = 0°: Identical directions (no separation)

# PRIVACY IMPLICATIONS:
# • Good subspace separation suggests unlearning creates distinct gradient patterns
# • Poor separation may indicate similar processing of forget and retain data
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.savefig(plot_dir / 'subspace_overlap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_principal_angles_plot(comparison_results, plot_dir):
    """Create detailed principal angles analysis plot."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    layers = list(comparison_results.keys())
    min_angles = [comparison_results[l]['min_cosine_angle'] for l in layers]
    max_angles = [comparison_results[l]['max_cosine_angle'] for l in layers]
    mean_angles = [comparison_results[l]['mean_cosine_angle'] for l in layers]
    grassmann_dists = [comparison_results[l]['grassmann_distance'] for l in layers]
    
    # Plot 1: Min/Max/Mean cosine angles
    x = np.arange(len(layers))
    width = 0.25
    
    bars1 = ax1.bar(x - width, min_angles, width, label='Min Cosine', alpha=0.8, color='lightcoral')
    bars2 = ax1.bar(x, mean_angles, width, label='Mean Cosine', alpha=0.8, color='skyblue')
    bars3 = ax1.bar(x + width, max_angles, width, label='Max Cosine', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Neural Network Layers')
    ax1.set_ylabel('Cosine of Principal Angles')
    ax1.set_title('Distribution of Principal Angles (Cosine Values)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Angle distribution (degrees)
    angle_degrees_min = [np.arccos(c) * 180 / np.pi for c in min_angles]
    angle_degrees_max = [np.arccos(c) * 180 / np.pi for c in max_angles]
    angle_degrees_mean = [np.arccos(c) * 180 / np.pi for c in mean_angles]
    
    bars1 = ax2.bar(x - width, angle_degrees_min, width, label='Min Angle', alpha=0.8, color='lightcoral')
    bars2 = ax2.bar(x, angle_degrees_mean, width, label='Mean Angle', alpha=0.8, color='skyblue')
    bars3 = ax2.bar(x + width, angle_degrees_max, width, label='Max Angle', alpha=0.8, color='lightgreen')
    
    ax2.set_xlabel('Neural Network Layers')
    ax2.set_ylabel('Principal Angles (Degrees)')
    ax2.set_title('Distribution of Principal Angles (Degrees)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 90)
    
    # Plot 3: Grassmann distance
    bars = ax3.bar(range(len(layers)), grassmann_dists, alpha=0.8, edgecolor='black')
    
    # Color code based on distance
    for bar, dist in zip(bars, grassmann_dists):
        if dist >= 1.0:
            bar.set_color('green')  # High distance = good separation
        elif dist >= 0.5:
            bar.set_color('orange')  # Moderate distance
        else:
            bar.set_color('red')  # Low distance = poor separation
    
    ax3.set_xlabel('Neural Network Layers')
    ax3.set_ylabel('Grassmann Distance')
    ax3.set_title('Grassmann Distance Between Subspaces\n(Higher = Better Separation)')
    ax3.set_xticks(range(len(layers)))
    ax3.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation between metrics
    overlaps = [comparison_results[l]['subspace_overlap'] for l in layers]
    
    ax4.scatter(overlaps, grassmann_dists, alpha=0.8, s=60, edgecolors='black')
    ax4.set_xlabel('Subspace Overlap')
    ax4.set_ylabel('Grassmann Distance')
    ax4.set_title('Overlap vs Distance Relationship')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    if len(overlaps) > 1:
        z = np.polyfit(overlaps, grassmann_dists, 1)
        p = np.poly1d(z)
        ax4.plot(sorted(overlaps), p(sorted(overlaps)), "r--", alpha=0.8, linewidth=2)
    
    # Add layer labels
    for i, layer in enumerate(layers):
        if i < 10:  # Avoid overcrowding
            ax4.annotate(layer.replace('layer', 'L'), (overlaps[i], grassmann_dists[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# DETAILED PRINCIPAL ANGLES ANALYSIS:
# Top plots show the distribution of principal angles between forget and retain subspaces.
# Bottom plots show Grassmann distances and their relationship to overlap metrics.

# INTERPRETATION:
# • Cosine values closer to 0 = angles closer to 90° = better separation
# • Higher Grassmann distances indicate better geometric separation
# • Negative correlation between overlap and distance indicates consistent metrics

# SUBSPACE GEOMETRY:
# • Principal angles measure how "aligned" the subspaces are
# • Multiple angles provide a complete geometric picture
# • Grassmann distance provides a single summary metric for subspace separation
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.savefig(plot_dir / 'principal_angles_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_dimensions_comparison_plot(forget_subspaces, retain_subspaces, plot_dir):
    """Create subspace dimensions comparison plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    common_layers = set(forget_subspaces.keys()) & set(retain_subspaces.keys())
    layers = sorted(list(common_layers))
    
    forget_dims = [forget_subspaces[l]['num_components'] for l in layers]
    retain_dims = [retain_subspaces[l]['num_components'] for l in layers]
    forget_variance = [forget_subspaces[l]['explained_variance_ratio'] for l in layers]
    retain_variance = [retain_subspaces[l]['explained_variance_ratio'] for l in layers]
    
    # Plot 1: Subspace dimensions comparison
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, forget_dims, width, label='Forget Subspace', 
                   color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, retain_dims, width, label='Retain Subspace', 
                   color='skyblue', alpha=0.8)
    
    ax1.set_xlabel('Neural Network Layers')
    ax1.set_ylabel('Number of Principal Components')
    ax1.set_title('Subspace Dimensionality Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, dim in zip(bars1, forget_dims):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{dim}', ha='center', va='bottom', fontsize=8)
    for bar, dim in zip(bars2, retain_dims):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{dim}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Variance explained comparison
    bars1 = ax2.bar(x - width/2, forget_variance, width, label='Forget Subspace', 
                   color='coral', alpha=0.8)
    bars2 = ax2.bar(x + width/2, retain_variance, width, label='Retain Subspace', 
                   color='skyblue', alpha=0.8)
    
    ax2.set_xlabel('Neural Network Layers')
    ax2.set_ylabel('Variance Explained Ratio')
    ax2.set_title('Subspace Quality Comparison (Variance Explained)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # Add value labels
    for bar, var in zip(bars1, forget_variance):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{var:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, var in zip(bars2, retain_variance):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{var:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# SUBSPACE DIMENSIONS ANALYSIS:
# Left: Number of principal components kept in each subspace (after variance thresholding)
# Right: Proportion of total variance explained by the kept components

# INTERPRETATION:
# • Left plot: Higher dimensions may indicate more complex gradient patterns
# • Right plot: Values closer to 1.0 indicate better subspace quality
# • Large differences between forget/retain may indicate different complexity patterns

# INSIGHTS:
# • Similar dimensions suggest comparable complexity of unlearning effects
# • Different dimensions may indicate selective effects (some data types affected more)
# • High variance explained (>0.9) indicates good subspace capture of the gradient patterns

# IMPLICATIONS:
# • Dimension differences may reveal differential unlearning complexity
# • Quality differences may indicate better subspace extraction for one data type
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.savefig(plot_dir / 'subspace_dimensions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_grassmann_distance_plot(comparison_results, plot_dir):
    """Create Grassmann distance analysis plot."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract all metrics
    layers = list(comparison_results.keys())
    grassmann_dists = [comparison_results[l]['grassmann_distance'] for l in layers]
    overlaps = [comparison_results[l]['subspace_overlap'] for l in layers]
    proj_distances = [comparison_results[l]['projection_distance'] for l in layers]
    mean_angles = [comparison_results[l]['mean_angle_degrees'] for l in layers]
    
    # Plot 1: Grassmann distance distribution
    ax1.hist(grassmann_dists, bins=15, color='mediumpurple', alpha=0.8, edgecolor='black')
    ax1.axvline(np.mean(grassmann_dists), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(grassmann_dists):.3f}')
    ax1.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Good Separation (>1.0)')
    ax1.set_xlabel('Grassmann Distance')
    ax1.set_ylabel('Number of Layers')
    ax1.set_title('Distribution of Grassmann Distances')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance vs Overlap correlation
    ax2.scatter(overlaps, grassmann_dists, alpha=0.8, s=60, edgecolors='black', c='orange')
    ax2.set_xlabel('Subspace Overlap')
    ax2.set_ylabel('Grassmann Distance')
    ax2.set_title('Grassmann Distance vs Subspace Overlap')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if len(overlaps) > 1:
        corr_coeff = np.corrcoef(overlaps, grassmann_dists)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr_coeff:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: Distance vs Projection distance
    ax3.scatter(proj_distances, grassmann_dists, alpha=0.8, s=60, edgecolors='black', c='lightgreen')
    ax3.set_xlabel('Projection Distance (Frobenius)')
    ax3.set_ylabel('Grassmann Distance')
    ax3.set_title('Grassmann vs Projection Distance')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary heatmap of all metrics
    metrics_matrix = np.array([overlaps, grassmann_dists, proj_distances, mean_angles])
    
    im = ax4.imshow(metrics_matrix, cmap='viridis', aspect='auto')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(['Overlap', 'Grassmann Dist', 'Proj Dist', 'Mean Angle'])
    ax4.set_xlabel('Layers (Index)')
    ax4.set_title('All Metrics Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Normalized Values')
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# GRASSMANN DISTANCE COMPREHENSIVE ANALYSIS:
# This plot provides multiple perspectives on subspace separation quality.

# INTERPRETATION:
# • Top left: Distribution shows how consistently subspaces are separated
# • Top right: Negative correlation expected (high overlap = low distance)
# • Bottom left: Relationship between different distance metrics
# • Bottom right: Heatmap shows patterns across all layers and metrics

# GRASSMANN GEOMETRY:
# • Grassmann distance measures geodesic distance on the Grassmann manifold
# • Provides theoretically principled measure of subspace separation
# • Values > 1.0 generally indicate good separation

# PRACTICAL INSIGHTS:
# • Consistent high distances across layers suggest systematic unlearning effects
# • Strong correlations between metrics indicate robust measurements
# • Heatmap patterns may reveal layer-depth dependencies in unlearning
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.savefig(plot_dir / 'grassmann_distance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()