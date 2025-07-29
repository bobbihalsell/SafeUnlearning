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

def run_subspace_attack(original_model, unlearned_model, background_data, forget_data, 
                       retain_data, device='cuda', variance_threshold=0.9, verbose=True):
    """
    Realistic subspace attack: Estimate forget subspace from background data only,
    then test how well it predicts forget vs retain data.
    
    This simulates a real attack where you don't have access to the actual forget data.
    
    Phase 1: Estimate forget subspace from background data (gradient covariance analysis)
    Phase 2: Test prediction accuracy on real forget/retain data
    """
    
    print("SUBSPACE ATTACK")
    print("="*50)
    print("Phase 1: Estimate forget subspace from background data only")
    print("Phase 2: Test prediction on real forget/retain data")
    print("="*50)
    
    # Phase 1: Estimate forget subspace from background data
    print("\n🔍 PHASE 1: ESTIMATING FORGET SUBSPACE FROM BACKGROUND DATA")
    print("-" * 50)
    
    estimated_subspace = estimate_subspace_signature(
        original_model, unlearned_model, background_data, device, variance_threshold, verbose
    )
    
    if not estimated_subspace:
        return {
            'success': False,
            'description': 'Failed to estimate subspace signature from background data',
            'phase_1_success': False,
            'phase_2_success': False
        }
    
    # Phase 2: Test the estimated subspace on real data
    print("\n🎯 PHASE 2: TESTING ESTIMATED SUBSPACE ON REAL DATA")
    print("-" * 50)
    
    attack_results = test_subspace_signature(
        estimated_subspace, original_model, unlearned_model, forget_data, retain_data, device, verbose
    )
    
    # Phase 3: Create visualizations
    plot_dir = Path("./attack_plots/subspace")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_subspace_attack_plots(
        estimated_subspace, attack_results, plot_dir, verbose
    )
    
    # Phase 4: Analyze attack success
    analysis_results = analyze_subspace_attack_results(attack_results, verbose)
    
    return {
        'estimated_subspace': estimated_subspace,
        'attack_results': attack_results,
        'analysis': analysis_results,
        'plot_directory': str(plot_dir),
        'phase_1_success': bool(estimated_subspace),
        'phase_2_success': attack_results.get('success', False)
    }


def estimate_subspace_signature(original_model, unlearned_model, background_data, 
                               device, variance_threshold=0.9, verbose=True):
    """
    Estimate the forget subspace using only background data gradient covariances.
    Key insight: Gradient covariance differences reveal unlearning subspace structure.
    """
    
    if verbose:
        print("Computing gradient covariances on background data...")
    
    # Get gradients from both models on background data
    orig_grads = get_model_gradients(original_model, background_data, device)
    unl_grads = get_model_gradients(unlearned_model, background_data, device)
    
    if not orig_grads or not unl_grads:
        if verbose:
            print("❌ Failed to compute gradients")
        return None
    
    # Calculate gradient differences
    grad_differences = {}
    total_diff_norm = 0
    
    for param_name in orig_grads:
        if param_name in unl_grads and '.weight' in param_name:
            diff = orig_grads[param_name] - unl_grads[param_name]
            layer_name = param_name.replace('.weight', '')
            grad_differences[layer_name] = diff
            total_diff_norm += torch.norm(diff).item()
    
    if verbose:
        print(f"Total gradient difference magnitude: {total_diff_norm:.6f}")
        print(f"Gradient differences from {len(grad_differences)} layers")
        
    if total_diff_norm < 1e-6:
        if verbose:
            print("⚠️  WARNING: Models show minimal gradient differences on background data")
        return None
    
    # Convert gradients to covariance matrices and extract subspaces
    subspace_signature = {}
    
    for layer_name, grad_diff in grad_differences.items():
        try:
            # Move to CPU and flatten
            grad_diff_cpu = grad_diff.detach().cpu().float()
            grad_flat = grad_diff_cpu.flatten()
            num_params = grad_flat.shape[0]
            
            # Memory-efficient covariance computation
            if num_params > 1024:  # For large layers, use approximation
                if verbose and len(subspace_signature) < 3:  # Show first 3
                    print(f"  Layer {layer_name}: Using approximation for {num_params} parameters")
                
                # Random projection to reduce dimensionality
                target_dim = min(256, num_params // 8)
                torch.manual_seed(42)  # For reproducibility
                proj_matrix = torch.randn(target_dim, num_params) / np.sqrt(target_dim)
                
                # Project gradient to lower dimension
                grad_projected = torch.mv(proj_matrix, grad_flat)
                
                # Create covariance in projected space
                grad_cov = torch.outer(grad_projected, grad_projected)
                
            else:
                if verbose and len(subspace_signature) < 3:  # Show first 3
                    print(f"  Layer {layer_name}: Using full covariance for {num_params} parameters")
                
                # Create full covariance matrix
                grad_cov = torch.outer(grad_flat, grad_flat)
            
            # Add small epsilon for numerical stability
            stabilized = grad_cov + 1e-6 * torch.eye(grad_cov.shape[0], dtype=grad_cov.dtype)
            
            # SVD decomposition
            U, S, V = torch.svd(stabilized)
            
            # Apply variance threshold
            if len(S) > 1:
                cumulative_var = torch.cumsum(S, dim=0) / torch.sum(S)
                components_to_keep = torch.sum(cumulative_var <= variance_threshold).item()
                components_to_keep = max(1, min(components_to_keep + 1, len(S)))
            else:
                components_to_keep = min(5, len(S))
            
            # Truncate to keep most important components
            U_truncated = U[:, :components_to_keep]
            S_truncated = S[:components_to_keep]
            
            subspace_signature[layer_name] = {
                'basis': U_truncated,
                'singular_values': S_truncated,
                'explained_variance_ratio': (S_truncated.sum() / S.sum()).item(),
                'num_components': components_to_keep,
                'total_components': len(S),
                'layer_name': layer_name
            }
            
            var_explained = (S_truncated.sum() / S.sum()).item()
            if verbose and len(subspace_signature) <= 5:  # Show first 5
                print(f"  Layer {layer_name}: {components_to_keep}/{len(S)} components, "
                      f"variance explained: {var_explained:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"  Subspace extraction failed for {layer_name}: {e}")
            continue
    
    if verbose:
        print(f"Successfully extracted subspaces from {len(subspace_signature)} layers")
    
    return subspace_signature


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


def test_subspace_signature(estimated_subspace, original_model, unlearned_model,
                           forget_data, retain_data, device, verbose=True):
    """
    Test how well the estimated subspace predicts forget vs retain data.
    """
    
    if verbose:
        print("Computing gradient differences on forget data...")
    
    # Get gradient differences for forget data
    forget_grads_orig = get_model_gradients(original_model, forget_data, device)
    forget_grads_unl = get_model_gradients(unlearned_model, forget_data, device)
    
    if verbose:
        print("Computing gradient differences on retain data...")
    
    # Get gradient differences for retain data
    retain_grads_orig = get_model_gradients(original_model, retain_data, device)
    retain_grads_unl = get_model_gradients(unlearned_model, retain_data, device)
    
    if not all([forget_grads_orig, forget_grads_unl, retain_grads_orig, retain_grads_unl]):
        return {
            'success': False, 
            'description': 'Failed to compute test gradients',
            'forget_projections': {},
            'retain_projections': {}
        }
    
    # Calculate gradient differences for test data
    forget_grad_diff = {}
    retain_grad_diff = {}
    
    for layer_name in estimated_subspace.keys():
        param_name = layer_name + '.weight'
        if (param_name in forget_grads_orig and param_name in forget_grads_unl and
            param_name in retain_grads_orig and param_name in retain_grads_unl):
            
            forget_grad_diff[layer_name] = forget_grads_orig[param_name] - forget_grads_unl[param_name]
            retain_grad_diff[layer_name] = retain_grads_orig[param_name] - retain_grads_unl[param_name]
    
    # Project gradient differences onto estimated subspaces
    forget_projections = {}
    retain_projections = {}
    
    for layer_name in estimated_subspace:
        if layer_name not in forget_grad_diff or layer_name not in retain_grad_diff:
            continue
            
        try:
            subspace_basis = estimated_subspace[layer_name]['basis']
            
            # Get test gradient differences for this layer
            forget_grad = forget_grad_diff[layer_name]
            retain_grad = retain_grad_diff[layer_name]
            
            if forget_grad is None or retain_grad is None:
                continue
            
            # Flatten and move to CPU
            forget_flat = forget_grad.detach().cpu().flatten().float()
            retain_flat = retain_grad.detach().cpu().flatten().float()
            
            # Handle dimension mismatch (from random projection)
            if forget_flat.shape[0] != subspace_basis.shape[0]:
                # Apply same random projection as used in estimation
                torch.manual_seed(42)  # Same seed as estimation
                if forget_flat.shape[0] > subspace_basis.shape[0]:
                    proj_matrix = torch.randn(subspace_basis.shape[0], forget_flat.shape[0]) / np.sqrt(subspace_basis.shape[0])
                    forget_flat = torch.mv(proj_matrix, forget_flat)
                    retain_flat = torch.mv(proj_matrix, retain_flat)
                else:
                    # Skip if dimensions don't match and we can't project
                    continue
            
            # Project onto subspace (sum of absolute projections onto all basis vectors)
            forget_projection = torch.sum(torch.abs(subspace_basis.T @ forget_flat)).item()
            retain_projection = torch.sum(torch.abs(subspace_basis.T @ retain_flat)).item()
            
            forget_projections[layer_name] = forget_projection
            retain_projections[layer_name] = retain_projection
            
            if verbose and len(forget_projections) <= 5:  # Show first 5
                ratio = forget_projection / (retain_projection + 1e-8)
                print(f"  Layer {layer_name}: forget_proj={forget_projection:.6f}, retain_proj={retain_projection:.6f}, ratio={ratio:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"Subspace projection failed for {layer_name}: {e}")
            continue
    
    return {
        'forget_projections': forget_projections,
        'retain_projections': retain_projections,
        'success': bool(forget_projections and retain_projections)
    }


def analyze_subspace_attack_results(attack_results, verbose=True):
    """Analyze the subspace attack results."""
    
    if not attack_results.get('success', False):
        return {
            'success': False,
            'description': attack_results.get('description', 'Attack failed'),
            'prediction_ratio': 0.0
        }
    
    forget_projections = list(attack_results['forget_projections'].values())
    retain_projections = list(attack_results['retain_projections'].values())
    
    if not forget_projections or not retain_projections:
        return {
            'success': False,
            'description': 'No valid projection scores computed',
            'prediction_ratio': 0.0
        }
    
    mean_forget_proj = np.mean(forget_projections)
    mean_retain_proj = np.mean(retain_projections)
    prediction_ratio = mean_forget_proj / (mean_retain_proj + 1e-8)
    
    print(f"\n🔍 SUBSPACE ATTACK ANALYSIS")
    print(f"   • Forget Projection Strength: {mean_forget_proj:.6f}")
    print(f"   • Retain Projection Strength: {mean_retain_proj:.6f}")
    print(f"   • Prediction Ratio: {prediction_ratio:.3f}")
    print(f"   • Layers Analyzed: {len(forget_projections)}")
    
    # Determine success level and privacy implications (subspace-specific thresholds)
    if prediction_ratio >= 2.0:
        success_level = "HIGH"
        success = True
        privacy_risk = "CRITICAL"
        description = f"Background subspaces strongly predict forget patterns (ratio: {prediction_ratio:.3f}). CRITICAL privacy risk - forgotten data subspace signatures can be identified from background information alone."
    elif prediction_ratio >= 1.6:
        success_level = "MODERATE"
        success = True
        privacy_risk = "HIGH"
        description = f"Background subspaces moderately predict forget patterns (ratio: {prediction_ratio:.3f}). HIGH privacy risk - some forgotten information may be recoverable through subspace analysis."
    elif prediction_ratio >= 1.2:
        success_level = "LOW"
        success = False
        privacy_risk = "MODERATE"
        description = f"Background subspaces weakly predict forget patterns (ratio: {prediction_ratio:.3f}). MODERATE privacy risk - limited subspace-based information leakage."
    else:
        success_level = "FAILED"
        success = False
        privacy_risk = "LOW"
        description = f"Background subspaces cannot predict forget patterns (ratio: {prediction_ratio:.3f}). LOW privacy risk - good unlearning privacy protection against subspace attacks."
    
    print(f"   • Attack Assessment: {success_level}")
    print(f"   • Privacy Risk Level: {privacy_risk}")
    print(f"   • Description: {description}")
    
    return {
        'success': success,
        'success_level': success_level,
        'description': description,
        'prediction_ratio': prediction_ratio,
        'mean_forget_projection': mean_forget_proj,
        'mean_retain_projection': mean_retain_proj,
        'privacy_risk_level': privacy_risk,
        'num_layers': len(forget_projections)
    }


def create_subspace_attack_plots(estimated_subspace, attack_results, plot_dir, verbose=True):
    """Create visualization plots for subspace attack analysis."""
    
    if verbose:
        print(f"\n🎨 Creating subspace attack plots in {plot_dir}...")
    
    # Plot 1: Subspace Quality Analysis
    create_subspace_quality_plot(estimated_subspace, plot_dir)
    
    # Plot 2: Projection Results Analysis
    create_subspace_projection_plot(attack_results, plot_dir)
    
    # Plot 3: Subspace Dimensionality Analysis
    create_subspace_dimensionality_plot(estimated_subspace, attack_results, plot_dir)
    
    # Plot 4: Attack Performance Summary
    create_subspace_summary_plot(attack_results, plot_dir)
    
    if verbose:
        print("✅ All subspace attack plots created!")


def create_subspace_quality_plot(estimated_subspace, plot_dir):
    """Create subspace quality analysis plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    layers = list(estimated_subspace.keys())
    explained_variance = [estimated_subspace[l]['explained_variance_ratio'] for l in layers]
    num_components = [estimated_subspace[l]['num_components'] for l in layers]
    singular_values = [estimated_subspace[l]['singular_values'] for l in layers]
    
    # Plot 1: Explained variance per layer
    bars = ax1.bar(range(len(layers)), explained_variance, alpha=0.8, edgecolor='black')
    
    # Color code bars based on quality
    for bar, var_exp in zip(bars, explained_variance):
        if var_exp >= 0.95:
            bar.set_color('green')  # Excellent subspace quality
        elif var_exp >= 0.9:
            bar.set_color('yellow')  # Good quality
        elif var_exp >= 0.8:
            bar.set_color('orange')  # Moderate quality
        else:
            bar.set_color('red')  # Poor quality
    
    ax1.axhline(y=0.9, color='green', linestyle='--', linewidth=2, 
                label='Target Threshold (0.9)')
    ax1.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, 
                label='Minimum Threshold (0.8)')
    
    ax1.set_xlabel('Neural Network Layers')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Subspace Quality: Variance Explained by Estimated Subspace\n(Higher = Better Subspace Capture)')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Add value labels for interesting layers
    for bar, var_exp in zip(bars, explained_variance):
        if var_exp < 0.9 or var_exp > 0.98:  # Show low quality or very high quality
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{var_exp:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Singular value distributions (subspace strength)
    # Show distribution of singular values across layers
    all_singular_values = []
    layer_labels = []
    
    for i, (layer, svs) in enumerate(zip(layers, singular_values)):
        sv_values = svs.cpu().numpy() if hasattr(svs, 'cpu') else svs
        all_singular_values.extend(sv_values)
        layer_labels.extend([i] * len(sv_values))
    
    # Create violin plot or box plot
    if len(layers) <= 8:  # Use violin plot for reasonable number of layers
        violin_data = [singular_values[i].cpu().numpy() if hasattr(singular_values[i], 'cpu') 
                      else singular_values[i] for i in range(len(layers))]
        parts = ax2.violinplot(violin_data, positions=range(len(layers)), 
                              showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, var_exp in zip(parts['bodies'], explained_variance):
            if var_exp >= 0.95:
                pc.set_facecolor('green')
            elif var_exp >= 0.9:
                pc.set_facecolor('yellow')
            elif var_exp >= 0.8:
                pc.set_facecolor('orange')
            else:
                pc.set_facecolor('red')
            pc.set_alpha(0.7)
        
        ax2.set_xlabel('Neural Network Layers')
        ax2.set_ylabel('Singular Values (Subspace Strength)')
        ax2.set_title('Subspace Strength Distribution\n(Wider = More Diverse Singular Values)')
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
    else:
        # Use histogram for many layers
        ax2.hist(all_singular_values, bins=30, alpha=0.8, edgecolor='black', color='skyblue')
        ax2.axvline(np.mean(all_singular_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(all_singular_values):.3e}')
        ax2.set_xlabel('Singular Values')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Overall Singular Value Distribution')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# SUBSPACE QUALITY ANALYSIS:
# Left: Quality of subspace estimation (how much variance is captured)
# Right: Distribution of singular values (strength of subspace components)

# INTERPRETATION:
# • Left plot: Higher values indicate better subspace estimation quality
# • Right plot: Shows the strength distribution of subspace components
# • Good subspaces need both high variance explained AND strong singular values

# QUALITY INDICATORS:
# • Green: Excellent subspace quality (>95% variance explained)
# • Yellow: Good quality (90-95% variance explained)
# • Orange: Moderate quality (80-90% variance explained)
# • Red: Poor quality (<80% variance explained)

# SUBSPACE STRENGTH:
# • Higher singular values indicate stronger subspace components
# • Wide distributions indicate diverse component strengths
# • Log scale used due to wide range of singular value magnitudes

# PRIVACY IMPLICATIONS:
# • High-quality subspaces are more dangerous for privacy
# • Strong, well-defined subspaces can better capture forget patterns
# • Poor quality subspaces provide better privacy protection
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.savefig(plot_dir / 'subspace_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_subspace_projection_plot(attack_results, plot_dir):
    """Create subspace projection results analysis plot."""
    
    if not attack_results.get('success', False):
        # Create a failure plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Attack Failed\nNo projection results to display', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(plot_dir / 'subspace_projection_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    layers = list(attack_results['forget_projections'].keys())
    forget_projections = [attack_results['forget_projections'][l] for l in layers]
    retain_projections = [attack_results['retain_projections'][l] for l in layers]
    ratios = [f / (r + 1e-8) for f, r in zip(forget_projections, retain_projections)]
    
    # Plot 1: Projection strength comparison (log scale for better visualization)
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, forget_projections, width, label='Forget Data Projection', 
                   color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, retain_projections, width, label='Retain Data Projection', 
                   color='skyblue', alpha=0.8)
    
    ax1.set_xlabel('Neural Network Layers')
    ax1.set_ylabel('Subspace Projection Strength (Log Scale)')
    ax1.set_title('Subspace Attack Projection Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Highlight significant differences
    for i, (f_proj, r_proj) in enumerate(zip(forget_projections, retain_projections)):
        if f_proj > r_proj * 1.2:  # Significant difference for subspaces
            ax1.text(i, max(f_proj, r_proj) * 1.2, '⚠️', ha='center', va='bottom', fontsize=12)
    
    # Plot 2: Projection ratios with subspace-specific thresholds
    bars = ax2.bar(range(len(layers)), ratios, alpha=0.8, edgecolor='black')
    
    # Color code based on attack success (subspace-specific thresholds)
    for bar, ratio in zip(bars, ratios):
        if ratio >= 2.0:
            bar.set_color('darkred')  # Critical privacy risk
        elif ratio >= 1.6:
            bar.set_color('red')  # High privacy risk
        elif ratio >= 1.2:
            bar.set_color('orange')  # Moderate privacy risk
        else:
            bar.set_color('green')  # Low privacy risk
    
    # Add threshold lines (subspace-specific)
    ax2.axhline(y=2.0, color='darkred', linestyle='--', linewidth=2, label='Critical Risk (>2.0)')
    ax2.axhline(y=1.6, color='red', linestyle='--', linewidth=2, label='High Risk (>1.6)')
    ax2.axhline(y=1.2, color='orange', linestyle='--', linewidth=2, label='Moderate Risk (>1.2)')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Break-even (1.0)')
    
    ax2.set_xlabel('Neural Network Layers')
    ax2.set_ylabel('Projection Ratio (Forget/Retain)')
    ax2.set_title('Subspace Attack Success Ratio by Layer\n(Higher = Better Attack Success)')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels for high-risk layers
    for bar, ratio in zip(bars, ratios):
        if ratio >= 1.2:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ratios)*0.02,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# SUBSPACE PROJECTION ANALYSIS:
# Left: Comparison of how strongly forget vs retain data project onto estimated subspaces
# Right: Ratio of projection strengths (key metric for subspace attack success)

# INTERPRETATION:
# • Left plot: Higher bars indicate stronger projection onto the estimated subspace
# • Log scale used to handle wide range of projection magnitudes
# • ⚠️ symbols indicate significant projection differences

# SUBSPACE ATTACK CRITERIA:
# • Ratio > 2.0: Critical privacy risk - very successful subspace attack
# • Ratio > 1.6: High privacy risk - successful subspace attack  
# • Ratio > 1.2: Moderate privacy risk - partially successful attack
# • Ratio < 1.2: Low privacy risk - attack largely failed

# SUBSPACE-SPECIFIC FEATURES:
# • Projections measure how well data aligns with estimated gradient subspaces
# • Strong projections indicate the subspace captures relevant forget patterns
# • Subspace attacks reveal structural vulnerabilities in gradient space

# PRIVACY IMPLICATIONS:
# • High ratios mean an attacker can identify forget patterns through subspace analysis
# • Subspace attacks can reveal lower-dimensional structures in gradient changes
# • Results indicate effectiveness of subspace-level unlearning protection
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.savefig(plot_dir / 'subspace_projection_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_subspace_dimensionality_plot(estimated_subspace, attack_results, plot_dir):
    """Create subspace dimensionality analysis plot."""
    
    if not attack_results.get('success', False):
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    layers = list(estimated_subspace.keys())
    num_components = [estimated_subspace[l]['num_components'] for l in layers]
    total_components = [estimated_subspace[l]['total_components'] for l in layers]
    explained_variance = [estimated_subspace[l]['explained_variance_ratio'] for l in layers]
    
    if layers[0] in attack_results['forget_projections']:
        ratios = [attack_results['forget_projections'][l] / (attack_results['retain_projections'][l] + 1e-8) 
                 for l in layers]
    else:
        ratios = [0] * len(layers)
    
    # Plot 1: Dimensionality vs attack success
    scatter = ax1.scatter(num_components, ratios, c=explained_variance, cmap='RdYlBu_r', 
                         s=100, alpha=0.8, edgecolors='black')
    
    # Add layer labels for interesting points
    for i, layer in enumerate(layers):
        if ratios[i] > np.mean(ratios) or num_components[i] > np.mean(num_components):
            ax1.annotate(layer.replace('layer', 'L'), (num_components[i], ratios[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Number of Subspace Components')
    ax1.set_ylabel('Attack Success Ratio (Forget/Retain)')
    ax1.set_title('Subspace Dimensionality vs Attack Success\n(Color = Explained Variance Ratio)')
    ax1.grid(True, alpha=0.3)
    
    # Add success regions
    ax1.axhspan(1.6, max(ratios)*1.1 if ratios else 3.0, alpha=0.2, color='red', label='High Success')
    ax1.axhspan(1.2, 1.6, alpha=0.2, color='orange', label='Moderate Success')
    ax1.axhspan(0, 1.2, alpha=0.2, color='green', label='Low Success')
    ax1.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Explained Variance Ratio')
    
    # Plot 2: Dimensionality efficiency analysis
    efficiency_scores = [r / (c + 1e-8) for r, c in zip(ratios, num_components)]
    compression_ratios = [c / t for c, t in zip(num_components, total_components)]
    
    scatter2 = ax2.scatter(compression_ratios, efficiency_scores, c=range(len(layers)), 
                          cmap='viridis', s=100, alpha=0.8, edgecolors='black')
    
    # Add layer labels for high-efficiency layers
    for i, layer in enumerate(layers):
        if efficiency_scores[i] > np.mean(efficiency_scores):
            ax2.annotate(layer.replace('layer', 'L'), (compression_ratios[i], efficiency_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Compression Ratio (Kept Components / Total Components)')
    ax2.set_ylabel('Attack Efficiency (Success / Dimensionality)')
    ax2.set_title('Subspace Compression vs Attack Efficiency\n(Color = Layer Index)')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Layer Index (Network Depth)')
    
    # Add trend line
    if len(compression_ratios) > 2:
        z = np.polyfit(compression_ratios, efficiency_scores, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(compression_ratios), p(sorted(compression_ratios)), 
                "r--", alpha=0.8, linewidth=2, label='Trend')
        ax2.legend()
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# SUBSPACE DIMENSIONALITY ANALYSIS:
# Top: Relationship between subspace dimensionality and attack success
# Bottom: Analysis of compression efficiency vs attack effectiveness

# INTERPRETATION:
# • Top plot: Shows if higher-dimensional subspaces lead to more successful attacks
# • Color indicates subspace quality (variance explained)
# • Bottom plot: Analyzes efficiency of dimension reduction

# KEY INSIGHTS:
# • Higher dimensions may capture more complex patterns but reduce efficiency
# • Good compression (low ratios) with high efficiency indicates structural vulnerabilities
# • Network depth (color) may influence subspace characteristics

# SUBSPACE COMPRESSION:
# • Compression ratio = kept components / total possible components
# • Lower ratios indicate more aggressive dimensionality reduction
# • Efficiency = attack success per unit of dimensionality

# PRIVACY IMPLICATIONS:
# • High efficiency at low compression suggests exploitable low-dimensional structures
# • Understanding dimensionality trade-offs guides defensive strategies
# • Results inform optimal subspace protection mechanisms
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.savefig(plot_dir / 'subspace_dimensionality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_subspace_summary_plot(attack_results, plot_dir):
    """Create subspace attack performance summary plot."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    if not attack_results.get('success', False):
        # Create failure summary
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'Attack Failed\nInsufficient Data', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        plt.savefig(plot_dir / 'subspace_attack_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    forget_projections = list(attack_results['forget_projections'].values())
    retain_projections = list(attack_results['retain_projections'].values())
    ratios = [f / (r + 1e-8) for f, r in zip(forget_projections, retain_projections)]
    
    # Plot 1: Projection strength distributions (log scale)
    forget_log = np.log10(np.array(forget_projections) + 1e-10)
    retain_log = np.log10(np.array(retain_projections) + 1e-10)
    
    ax1.hist(forget_log, bins=15, alpha=0.7, label='Forget Projections', 
             color='coral', edgecolor='black', density=True)
    ax1.hist(retain_log, bins=15, alpha=0.7, label='Retain Projections', 
             color='skyblue', edgecolor='black', density=True)
    ax1.axvline(np.mean(forget_log), color='red', linestyle='--', linewidth=2, 
                label=f'Forget Mean: {np.mean(forget_projections):.3e}')
    ax1.axvline(np.mean(retain_log), color='blue', linestyle='--', linewidth=2,
                label=f'Retain Mean: {np.mean(retain_projections):.3e}')
    ax1.set_xlabel('Log10(Projection Strength)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Subspace Projection Strengths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio distribution with statistical measures
    ax2.hist(ratios, bins=15, color='mediumpurple', alpha=0.8, edgecolor='black')
    ax2.axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(ratios):.3f}')
    ax2.axvline(np.median(ratios), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(ratios):.3f}')
    ax2.axvline(2.0, color='darkred', linestyle='--', linewidth=2, label='Critical Threshold')
    ax2.axvline(1.6, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    ax2.axvline(1.0, color='gray', linestyle='--', linewidth=1, label='Break-even')
    ax2.set_xlabel('Projection Ratio (Forget/Retain)')
    ax2.set_ylabel('Number of Layers')
    ax2.set_title('Distribution of Subspace Attack Success Ratios')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Risk categorization (subspace-specific thresholds)
    critical_risk = sum(1 for r in ratios if r >= 2.0)
    high_risk = sum(1 for r in ratios if 1.6 <= r < 2.0)
    moderate_risk = sum(1 for r in ratios if 1.2 <= r < 1.6)
    low_risk = sum(1 for r in ratios if r < 1.2)
    
    categories = ['Low Risk\n(<1.2)', 'Moderate Risk\n(1.2-1.6)', 'High Risk\n(1.6-2.0)', 'Critical Risk\n(≥2.0)']
    counts = [low_risk, moderate_risk, high_risk, critical_risk]
    colors = ['green', 'orange', 'red', 'darkred']
    
    bars = ax3.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Number of Layers')
    ax3.set_title('Subspace Attack Risk Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels
    total_layers = len(ratios)
    for bar, count in zip(bars, counts):
        if count > 0:
            percentage = (count / total_layers) * 100
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Overall attack assessment with geometric mean
    overall_ratio = np.mean(ratios)
    geometric_mean = np.exp(np.mean(np.log(np.array(ratios) + 1e-8)))
    ratio_std = np.std(ratios)
    
    if overall_ratio >= 2.0:
        risk_level = "CRITICAL"
        risk_color = "darkred"
        risk_message = "Very High\nPrivacy Risk"
    elif overall_ratio >= 1.6:
        risk_level = "HIGH"
        risk_color = "red"
        risk_message = "High\nPrivacy Risk"
    elif overall_ratio >= 1.2:
        risk_level = "MODERATE"
        risk_color = "orange"
        risk_message = "Moderate\nPrivacy Risk"
    else:
        risk_level = "LOW"
        risk_color = "green"
        risk_message = "Good Privacy\nProtection"
    
    bar = ax4.bar([risk_level], [overall_ratio], color=risk_color, alpha=0.8)
    
    # Add error bar and geometric mean
    ax4.errorbar([risk_level], [overall_ratio], yerr=[ratio_std], 
                fmt='none', color='black', capsize=10, capthick=2)
    ax4.plot([0], [geometric_mean], 'D', color='yellow', markersize=10, 
             label=f'Geometric Mean: {geometric_mean:.3f}')
    
    ax4.set_ylabel('Overall Attack Ratio')
    ax4.set_title('Overall Subspace Attack Assessment')
    ax4.text(0, overall_ratio/2, risk_message, ha='center', va='center', 
             fontweight='bold', fontsize=11, color='white')
    ax4.set_ylim(0, max(2.5, overall_ratio + ratio_std + 0.3))
    ax4.legend()
    
    # Add threshold lines
    for threshold, color, label in [(2.0, 'darkred', 'Critical'), (1.6, 'red', 'High'), 
                                   (1.2, 'orange', 'Moderate'), (1.0, 'gray', 'Break-even')]:
        if threshold <= max(2.5, overall_ratio + ratio_std + 0.3):
            ax4.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# SUBSPACE ATTACK COMPREHENSIVE SUMMARY:
# Analysis of subspace attack success across all layers with subspace-specific metrics.

# INTERPRETATION:
# • Top left: Log-scale distribution shows wide range of projection strengths
# • Top right: Success ratio distribution with statistical measures
# • Bottom left: Risk categorization using subspace-specific thresholds
# • Bottom right: Overall assessment with both arithmetic and geometric means

# SUBSPACE-SPECIFIC FEATURES:
# • Log scale used for projection strengths due to wide dynamic range
# • Geometric mean provides alternative view less affected by outliers
# • Thresholds adjusted for subspace attack characteristics

# RISK ASSESSMENT:
# • Good separation between forget/retain distributions indicates successful attack
# • Mean ratio > 1.6 suggests significant privacy vulnerabilities in subspace
# • High percentage of high-risk layers indicates systematic subspace problems

# PRIVACY IMPLICATIONS:
# • Successful subspace attacks reveal lower-dimensional vulnerabilities
# • Results show effectiveness of subspace-level privacy protection
# • Critical for understanding structural privacy risks in gradient space
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.savefig(plot_dir / 'subspace_attack_summary.png', dpi=300, bbox_inches='tight')
    plt.close()