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

def run_activation_attack(original_model, unlearned_model, background_data, forget_data, 
                         retain_data, device='cuda', verbose=True):
    """
    Realistic activation attack: Estimate forget signature from background data only,
    then test how well it predicts forget vs retain data.
    
    This simulates a real attack where you don't have access to the actual forget data.
    
    Phase 1: Estimate forget signature from background data (activation differences)
    Phase 2: Test prediction accuracy on real forget/retain data
    """
    
    print("ACTIVATION ATTACK")
    print("="*50)
    print("Phase 1: Estimate forget signature from background data only")
    print("Phase 2: Test prediction on real forget/retain data")
    print("="*50)
    
    # Phase 1: Estimate forget signature from background data
    print("\n🔍 PHASE 1: ESTIMATING FORGET SIGNATURE FROM BACKGROUND DATA")
    print("-" * 50)
    
    estimated_signature = estimate_activation_signature(
        original_model, unlearned_model, background_data, device, verbose
    )
    
    if not estimated_signature:
        return {
            'success': False,
            'description': 'Failed to estimate activation signature from background data',
            'phase_1_success': False,
            'phase_2_success': False
        }
    
    # Phase 2: Test the estimated signature on real data
    print("\n🎯 PHASE 2: TESTING ESTIMATED SIGNATURE ON REAL DATA")
    print("-" * 50)
    
    attack_results = test_activation_signature(
        estimated_signature, original_model, unlearned_model, forget_data, retain_data, device, verbose
    )
    
    # Phase 3: Create visualizations
    plot_dir = Path("./attack_plots/activation")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_activation_attack_plots(
        estimated_signature, attack_results, plot_dir, verbose
    )
    
    # Phase 4: Analyze attack success
    analysis_results = analyze_activation_attack_results(attack_results, verbose)
    
    return {
        'estimated_signature': estimated_signature,
        'attack_results': attack_results,
        'analysis': analysis_results,
        'plot_directory': str(plot_dir),
        'phase_1_success': bool(estimated_signature),
        'phase_2_success': attack_results.get('success', False)
    }


def estimate_activation_signature(original_model, unlearned_model, background_data, 
                                 device, verbose=True):
    """
    Estimate the forget signature using only background data activation differences.
    Key insight: Activation differences on background data reveal unlearning patterns.
    """
    
    if verbose:
        print("Computing activation differences on background data...")
    
    # Get all target layers
    target_layers = get_all_target_layers(original_model)
    layer_names = list(target_layers.keys())
    
    # Get activations from both models on background data
    orig_activations = get_activations(original_model, background_data, layer_names, device)
    unl_activations = get_activations(unlearned_model, background_data, layer_names, device)
    
    if not orig_activations or not unl_activations:
        if verbose:
            print("❌ Failed to compute activations")
        return None
    
    # Calculate activation differences (this is our estimated signature)
    activation_signature = {}
    total_diff_norm = 0
    
    for layer_name in layer_names:
        if layer_name in orig_activations and layer_name in unl_activations:
            if orig_activations[layer_name] is not None and unl_activations[layer_name] is not None:
                diff = orig_activations[layer_name] - unl_activations[layer_name]
                activation_signature[layer_name] = diff
                total_diff_norm += torch.norm(diff).item()
    
    if verbose:
        print(f"Total activation signature magnitude: {total_diff_norm:.6f}")
        print(f"Signature extracted from {len(activation_signature)} layers")
        
    if total_diff_norm < 1e-6:
        if verbose:
            print("⚠️  WARNING: Models show minimal activation differences on background data")
        return None
    
    # Normalize signatures for comparison and focus on most important layers
    layer_importances = [(name, torch.norm(diff).item()) 
                        for name, diff in activation_signature.items()]
    layer_importances.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top layers that explain 90% of the variance
    total_importance = sum(imp for _, imp in layer_importances)
    cumulative = 0
    important_layers = []
    
    for name, importance in layer_importances:
        cumulative += importance
        important_layers.append((name, importance))
        if cumulative / total_importance >= 0.9:
            break
    
    # Create normalized signature for important layers only
    normalized_signature = {}
    for layer_name, importance in important_layers:
        diff = activation_signature[layer_name]
        diff_norm = torch.norm(diff).item()
        
        if diff_norm > 1e-8:
            normalized_signature[layer_name] = {
                'signature': diff / diff_norm,  # Unit vector
                'magnitude': diff_norm,
                'importance_rank': len(normalized_signature) + 1,
                'layer_name': layer_name
            }
            
            if verbose and len(normalized_signature) <= 5:  # Show first 5
                print(f"  Layer {layer_name}: signature magnitude {diff_norm:.6f} (rank {len(normalized_signature)})")
    
    if verbose:
        print(f"Selected {len(normalized_signature)} most important layers for signature")
    
    return normalized_signature


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


def test_activation_signature(estimated_signature, original_model, unlearned_model,
                             forget_data, retain_data, device, verbose=True):
    """
    Test how well the estimated signature predicts forget vs retain data.
    """
    
    if verbose:
        print("Computing activation differences on forget data...")
    
    # Get activation differences for forget data
    forget_activations_orig = get_activations(original_model, forget_data, 
                                            list(estimated_signature.keys()), device)
    forget_activations_unl = get_activations(unlearned_model, forget_data, 
                                           list(estimated_signature.keys()), device)
    
    if verbose:
        print("Computing activation differences on retain data...")
    
    # Get activation differences for retain data
    retain_activations_orig = get_activations(original_model, retain_data, 
                                            list(estimated_signature.keys()), device)
    retain_activations_unl = get_activations(unlearned_model, retain_data, 
                                           list(estimated_signature.keys()), device)
    
    if not all([forget_activations_orig, forget_activations_unl, 
                retain_activations_orig, retain_activations_unl]):
        return {
            'success': False, 
            'description': 'Failed to compute test activations',
            'forget_scores': {},
            'retain_scores': {}
        }
    
    # Calculate activation differences for test data
    forget_diff = {}
    retain_diff = {}
    
    for layer_name in estimated_signature.keys():
        if (layer_name in forget_activations_orig and layer_name in forget_activations_unl and
            layer_name in retain_activations_orig and layer_name in retain_activations_unl):
            
            forget_diff[layer_name] = forget_activations_orig[layer_name] - forget_activations_unl[layer_name]
            retain_diff[layer_name] = retain_activations_orig[layer_name] - retain_activations_unl[layer_name]
    
    # Calculate similarity scores with estimated signature
    forget_scores = {}
    retain_scores = {}
    
    for layer_name in estimated_signature:
        if layer_name not in forget_diff or layer_name not in retain_diff:
            continue
            
        try:
            signature = estimated_signature[layer_name]['signature']
            sig_magnitude = estimated_signature[layer_name]['magnitude']
            
            # Get test activation differences for this layer
            forget_act_diff = forget_diff[layer_name]
            retain_act_diff = retain_diff[layer_name]
            
            if forget_act_diff is None or retain_act_diff is None:
                continue
            
            # Normalize test activation differences
            forget_norm = torch.norm(forget_act_diff).item()
            retain_norm = torch.norm(retain_act_diff).item()
            
            if forget_norm > 1e-8 and retain_norm > 1e-8:
                forget_normalized = forget_act_diff / forget_norm
                retain_normalized = retain_act_diff / retain_norm
                
                # Calculate similarity scores (cosine similarity with signature)
                forget_score = torch.abs(torch.sum(forget_normalized * signature)).item()
                retain_score = torch.abs(torch.sum(retain_normalized * signature)).item()
                
                # Weight by signature magnitude (more important layers get higher weight)
                forget_scores[layer_name] = forget_score * sig_magnitude
                retain_scores[layer_name] = retain_score * sig_magnitude
                
                if verbose and len(forget_scores) <= 5:  # Show first 5
                    ratio = forget_score / (retain_score + 1e-8)
                    print(f"  Layer {layer_name}: forget={forget_score:.4f}, retain={retain_score:.4f}, ratio={ratio:.3f}")
                    
        except Exception as e:
            if verbose:
                print(f"Signature test failed for {layer_name}: {e}")
            continue
    
    return {
        'forget_scores': forget_scores,
        'retain_scores': retain_scores,
        'success': bool(forget_scores and retain_scores)
    }


def analyze_activation_attack_results(attack_results, verbose=True):
    """Analyze the activation attack results."""
    
    if not attack_results.get('success', False):
        return {
            'success': False,
            'description': attack_results.get('description', 'Attack failed'),
            'prediction_ratio': 0.0
        }
    
    forget_scores = list(attack_results['forget_scores'].values())
    retain_scores = list(attack_results['retain_scores'].values())
    
    if not forget_scores or not retain_scores:
        return {
            'success': False,
            'description': 'No valid prediction scores computed',
            'prediction_ratio': 0.0
        }
    
    mean_forget_score = np.mean(forget_scores)
    mean_retain_score = np.mean(retain_scores)
    prediction_ratio = mean_forget_score / (mean_retain_score + 1e-8)
    
    print(f"\n🔍 ACTIVATION ATTACK ANALYSIS")
    print(f"   • Forget Prediction Score: {mean_forget_score:.6f}")
    print(f"   • Retain Prediction Score: {mean_retain_score:.6f}")
    print(f"   • Prediction Ratio: {prediction_ratio:.3f}")
    print(f"   • Layers Analyzed: {len(forget_scores)}")
    
    # Determine success level and privacy implications (stricter thresholds for activations)
    if prediction_ratio >= 2.5:
        success_level = "HIGH"
        success = True
        privacy_risk = "CRITICAL"
        description = f"Background activations strongly predict forget patterns (ratio: {prediction_ratio:.3f}). CRITICAL privacy risk - forgotten data activation signatures can be identified from background information alone."
    elif prediction_ratio >= 1.8:
        success_level = "MODERATE"
        success = True
        privacy_risk = "HIGH"
        description = f"Background activations moderately predict forget patterns (ratio: {prediction_ratio:.3f}). HIGH privacy risk - some forgotten information may be recoverable through activation analysis."
    elif prediction_ratio >= 1.3:
        success_level = "LOW"
        success = False
        privacy_risk = "MODERATE"
        description = f"Background activations weakly predict forget patterns (ratio: {prediction_ratio:.3f}). MODERATE privacy risk - limited activation-based information leakage."
    else:
        success_level = "FAILED"
        success = False
        privacy_risk = "LOW"
        description = f"Background activations cannot predict forget patterns (ratio: {prediction_ratio:.3f}). LOW privacy risk - good unlearning privacy protection against activation attacks."
    
    print(f"   • Attack Assessment: {success_level}")
    print(f"   • Privacy Risk Level: {privacy_risk}")
    print(f"   • Description: {description}")
    
    return {
        'success': success,
        'success_level': success_level,
        'description': description,
        'prediction_ratio': prediction_ratio,
        'mean_forget_score': mean_forget_score,
        'mean_retain_score': mean_retain_score,
        'privacy_risk_level': privacy_risk,
        'num_layers': len(forget_scores)
    }


def create_activation_attack_plots(estimated_signature, attack_results, plot_dir, verbose=True):
    """Create visualization plots for activation attack analysis."""
    
    if verbose:
        print(f"\n🎨 Creating activation attack plots in {plot_dir}...")
    
    # Plot 1: Signature Strength Analysis
    create_activation_signature_plot(estimated_signature, plot_dir)
    
    # Plot 2: Prediction Results Analysis
    create_activation_prediction_plot(attack_results, plot_dir)
    
    # Plot 3: Layer Importance vs Attack Success
    create_activation_layer_analysis_plot(estimated_signature, attack_results, plot_dir)
    
    # Plot 4: Attack Performance Summary
    create_activation_summary_plot(attack_results, plot_dir)
    
    if verbose:
        print("✅ All activation attack plots created!")


def create_activation_signature_plot(estimated_signature, plot_dir):
    """Create activation signature strength analysis plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    layers = list(estimated_signature.keys())
    magnitudes = [estimated_signature[l]['magnitude'] for l in layers]
    importance_ranks = [estimated_signature[l]['importance_rank'] for l in layers]
    
    # Plot 1: Signature magnitudes with importance ranking
    bars = ax1.bar(range(len(layers)), magnitudes, alpha=0.8, edgecolor='black')
    
    # Color code bars based on importance rank (lower rank = more important)
    max_rank = max(importance_ranks) if importance_ranks else 1
    for bar, rank in zip(bars, importance_ranks):
        if rank <= 3:
            bar.set_color('red')  # Top 3 most important = high privacy risk
        elif rank <= 6:
            bar.set_color('orange')  # Moderately important
        elif rank <= 10:
            bar.set_color('yellow')  # Less important
        else:
            bar.set_color('green')  # Least important = lower privacy risk
    
    ax1.set_xlabel('Neural Network Layers (Sorted by Importance)')
    ax1.set_ylabel('Activation Signature Magnitude')
    ax1.set_title('Estimated Forget Signature Strength\n(Red = Most Important, Green = Least Important)')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add rank labels for top layers
    for i, (bar, rank, mag) in enumerate(zip(bars, importance_ranks, magnitudes)):
        if rank <= 5:  # Label top 5 layers
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(magnitudes)*0.01,
                    f'#{rank}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Importance vs magnitude relationship
    scatter = ax2.scatter(importance_ranks, magnitudes, c=range(len(layers)), 
                         cmap='viridis', s=80, alpha=0.8, edgecolors='black')
    
    # Add layer labels for top important layers
    for i, layer in enumerate(layers):
        if importance_ranks[i] <= 3:
            ax2.annotate(layer.replace('layer', 'L'), (importance_ranks[i], magnitudes[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Importance Rank (Lower = More Important)')
    ax2.set_ylabel('Signature Magnitude')
    ax2.set_title('Layer Importance vs Signature Strength')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Layer Selection Order')
    
    # Add trend line
    if len(importance_ranks) > 1:
        z = np.polyfit(importance_ranks, magnitudes, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(importance_ranks), p(sorted(importance_ranks)), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# ACTIVATION SIGNATURE ANALYSIS:
# Left: Magnitude of estimated forget signature for selected important layers
# Right: Relationship between layer importance rank and signature strength

# INTERPRETATION:
# • Higher magnitudes indicate stronger signature components (higher privacy risk)
# • Lower importance ranks (#1, #2, etc.) indicate layers that changed most during unlearning
# • Strong signatures in important layers are most concerning for privacy

# LAYER SELECTION:
# • Only layers explaining 90% of activation changes are included (focused analysis)
# • Importance ranking based on magnitude of activation differences on background data
# • This represents what an attacker could realistically estimate

# COLOR CODING (Left plot):
# • Red bars: Top 3 most important layers - highest privacy risk
# • Orange bars: Moderately important layers (ranks 4-6)
# • Yellow bars: Less important layers (ranks 7-10)
# • Green bars: Least important layers - lowest privacy risk

# PRIVACY IMPLICATIONS:
# • Strong signatures in top-ranked layers indicate targeted vulnerabilities
# • The signature was estimated from background data only (realistic attack scenario)
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.savefig(plot_dir / 'activation_signature_strength.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_activation_prediction_plot(attack_results, plot_dir):
    """Create activation prediction results analysis plot."""
    
    if not attack_results.get('success', False):
        # Create a failure plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Attack Failed\nNo prediction results to display', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(plot_dir / 'activation_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    layers = list(attack_results['forget_scores'].keys())
    forget_scores = [attack_results['forget_scores'][l] for l in layers]
    retain_scores = [attack_results['retain_scores'][l] for l in layers]
    ratios = [f / (r + 1e-8) for f, r in zip(forget_scores, retain_scores)]
    
    # Plot 1: Score comparison with error bars (showing variance in predictions)
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, forget_scores, width, label='Forget Data Prediction', 
                   color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, retain_scores, width, label='Retain Data Prediction', 
                   color='skyblue', alpha=0.8)
    
    ax1.set_xlabel('Neural Network Layers (Important Layers Only)')
    ax1.set_ylabel('Prediction Score (Activation Signature Similarity)')
    ax1.set_title('Activation Attack Prediction Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight significant differences
    for i, (f_score, r_score) in enumerate(zip(forget_scores, retain_scores)):
        if f_score > r_score * 1.5:  # Significant difference
            ax1.text(i, max(f_score, r_score) + max(forget_scores + retain_scores) * 0.02,
                    '⚠️', ha='center', va='bottom', fontsize=12)
    
    # Plot 2: Prediction ratios with threshold zones
    bars = ax2.bar(range(len(layers)), ratios, alpha=0.8, edgecolor='black')
    
    # Color code based on attack success (stricter thresholds for activations)
    for bar, ratio in zip(bars, ratios):
        if ratio >= 2.5:
            bar.set_color('darkred')  # Critical privacy risk
        elif ratio >= 1.8:
            bar.set_color('red')  # High privacy risk
        elif ratio >= 1.3:
            bar.set_color('orange')  # Moderate privacy risk
        else:
            bar.set_color('green')  # Low privacy risk
    
    # Add threshold lines (stricter for activations)
    ax2.axhline(y=2.5, color='darkred', linestyle='--', linewidth=2, label='Critical Risk (>2.5)')
    ax2.axhline(y=1.8, color='red', linestyle='--', linewidth=2, label='High Risk (>1.8)')
    ax2.axhline(y=1.3, color='orange', linestyle='--', linewidth=2, label='Moderate Risk (>1.3)')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Break-even (1.0)')
    
    ax2.set_xlabel('Neural Network Layers (Important Layers Only)')
    ax2.set_ylabel('Prediction Ratio (Forget/Retain)')
    ax2.set_title('Activation Attack Success Ratio by Layer\n(Higher = Better Attack Success)')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels for high-risk layers
    for bar, ratio in zip(bars, ratios):
        if ratio >= 1.3:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ratios)*0.02,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# ACTIVATION PREDICTION ANALYSIS:
# Left: Comparison of how well the estimated signature predicts forget vs retain data
# Right: Ratio of prediction scores (key metric for activation attack success)

# INTERPRETATION:
# • Left plot: Higher bars indicate stronger prediction (activation signature similarity)
# • Right plot: Ratios > 1.0 mean forget data is predicted better than retain data
# • ⚠️ symbols indicate significant prediction differences

# ACTIVATION ATTACK CRITERIA (Stricter than gradient attacks):
# • Ratio > 2.5: Critical privacy risk - very successful activation attack
# • Ratio > 1.8: High privacy risk - successful activation attack  
# • Ratio > 1.3: Moderate privacy risk - partially successful attack
# • Ratio < 1.3: Low privacy risk - attack largely failed

# FOCUS ON IMPORTANT LAYERS:
# • Only the most important layers (90% of changes) are analyzed
# • These layers showed the most activation differences during unlearning
# • Successful attacks on these layers indicate targeted vulnerabilities

# PRIVACY IMPLICATIONS:
# • High ratios mean an attacker can identify forget patterns from activation analysis
# • Activation attacks can reveal internal representation changes
# • Results indicate effectiveness of representation-level unlearning
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.savefig(plot_dir / 'activation_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_activation_layer_analysis_plot(estimated_signature, attack_results, plot_dir):
    """Create layer importance vs attack success analysis plot."""
    
    if not attack_results.get('success', False):
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    layers = list(estimated_signature.keys())
    magnitudes = [estimated_signature[l]['magnitude'] for l in layers]
    importance_ranks = [estimated_signature[l]['importance_rank'] for l in layers]
    
    if layers[0] in attack_results['forget_scores']:
        ratios = [attack_results['forget_scores'][l] / (attack_results['retain_scores'][l] + 1e-8) 
                 for l in layers]
    else:
        ratios = [0] * len(layers)
    
    # Plot 1: Importance rank vs attack success
    scatter = ax1.scatter(importance_ranks, ratios, c=magnitudes, cmap='Reds', 
                         s=100, alpha=0.8, edgecolors='black')
    
    # Add layer labels for top important layers
    for i, layer in enumerate(layers):
        if importance_ranks[i] <= 3 or ratios[i] >= np.mean(ratios):
            ax1.annotate(layer.replace('layer', 'L'), (importance_ranks[i], ratios[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Layer Importance Rank (Lower = More Important)')
    ax1.set_ylabel('Attack Success Ratio (Forget/Retain)')
    ax1.set_title('Layer Importance vs Activation Attack Success\n(Color = Signature Magnitude)')
    ax1.grid(True, alpha=0.3)
    
    # Add success regions
    ax1.axhspan(1.8, max(ratios)*1.1 if ratios else 3.0, alpha=0.2, color='red', label='High Success')
    ax1.axhspan(1.3, 1.8, alpha=0.2, color='orange', label='Moderate Success')
    ax1.axhspan(0, 1.3, alpha=0.2, color='green', label='Low Success')
    ax1.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Signature Magnitude')
    
    # Plot 2: Layer depth vs attack vulnerability
    # Extract layer depths (assuming layer names contain depth information)
    layer_depths = []
    for layer in layers:
        # Try to extract numeric layer information
        depth = 0
        for part in layer.split('.'):
            if part.isdigit():
                depth = max(depth, int(part))
        layer_depths.append(depth)
    
    if len(set(layer_depths)) > 1:  # Only plot if we have varying depths
        scatter2 = ax2.scatter(layer_depths, ratios, c=importance_ranks, cmap='viridis_r', 
                              s=100, alpha=0.8, edgecolors='black')
        
        ax2.set_xlabel('Layer Depth in Network')
        ax2.set_ylabel('Attack Success Ratio (Forget/Retain)')
        ax2.set_title('Network Depth vs Activation Attack Vulnerability\n(Color = Importance Rank)')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(layer_depths) > 2:
            z = np.polyfit(layer_depths, ratios, 1)
            p = np.poly1d(z)
            ax2.plot(sorted(set(layer_depths)), p(sorted(set(layer_depths))), 
                    "r--", alpha=0.8, linewidth=2, label='Trend')
            ax2.legend()
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Importance Rank (Lower = More Important)')
    else:
        # If no depth variation, show importance distribution
        bars = ax2.bar(range(len(layers)), importance_ranks, alpha=0.8, 
                      color=[plt.cm.viridis_r(r/max(ratios)) for r in ratios], edgecolor='black')
        ax2.set_xlabel('Layers (Sorted by Selection Order)')
        ax2.set_ylabel('Importance Rank (Lower = More Important)')
        ax2.set_title('Layer Importance Distribution\n(Color = Attack Success)')
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Lower ranks at top
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# LAYER IMPORTANCE vs ATTACK SUCCESS:
# Top: Relationship between layer importance and activation attack success
# Bottom: Analysis by network depth or importance distribution

# INTERPRETATION:
# • Top plot: Shows if more important layers are more vulnerable to attacks
# • Color coding reveals relationship between signature strength and success
# • Lower importance ranks (#1, #2, etc.) indicate layers that changed most

# KEY INSIGHTS:
# • Important layers (low rank numbers) should ideally have low attack success
# • High attack success on important layers indicates serious privacy vulnerabilities
# • Network depth patterns may reveal architectural vulnerabilities

# ACTIVATION-SPECIFIC ANALYSIS:
# • Focuses only on layers showing significant activation changes
# • Importance ranking based on magnitude of activation differences
# • Attack success measured by prediction ratio (forget vs retain)

# PRIVACY IMPLICATIONS:
# • Vulnerable important layers need additional protection
# • Patterns may guide targeted defense strategies
# • Understanding layer-specific risks helps improve unlearning methods
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.savefig(plot_dir / 'activation_layer_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_activation_summary_plot(attack_results, plot_dir):
    """Create activation attack performance summary plot."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    if not attack_results.get('success', False):
        # Create failure summary
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'Attack Failed\nInsufficient Data', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        plt.savefig(plot_dir / 'activation_attack_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    forget_scores = list(attack_results['forget_scores'].values())
    retain_scores = list(attack_results['retain_scores'].values())
    ratios = [f / (r + 1e-8) for f, r in zip(forget_scores, retain_scores)]
    
    # Plot 1: Score distributions with KDE
    ax1.hist(forget_scores, bins=15, alpha=0.7, label='Forget Scores', 
             color='coral', edgecolor='black', density=True)
    ax1.hist(retain_scores, bins=15, alpha=0.7, label='Retain Scores', 
             color='skyblue', edgecolor='black', density=True)
    ax1.axvline(np.mean(forget_scores), color='red', linestyle='--', linewidth=2, 
                label=f'Forget Mean: {np.mean(forget_scores):.4f}')
    ax1.axvline(np.mean(retain_scores), color='blue', linestyle='--', linewidth=2,
                label=f'Retain Mean: {np.mean(retain_scores):.4f}')
    ax1.set_xlabel('Prediction Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Activation Prediction Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio distribution with statistical analysis
    ax2.hist(ratios, bins=15, color='mediumpurple', alpha=0.8, edgecolor='black')
    ax2.axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(ratios):.3f}')
    ax2.axvline(np.median(ratios), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(ratios):.3f}')
    ax2.axvline(2.5, color='darkred', linestyle='--', linewidth=2, label='Critical Threshold')
    ax2.axvline(1.8, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    ax2.axvline(1.0, color='gray', linestyle='--', linewidth=1, label='Break-even')
    ax2.set_xlabel('Prediction Ratio (Forget/Retain)')
    ax2.set_ylabel('Number of Layers')
    ax2.set_title('Distribution of Activation Attack Success Ratios')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Risk categorization (stricter thresholds for activations)
    critical_risk = sum(1 for r in ratios if r >= 2.5)
    high_risk = sum(1 for r in ratios if 1.8 <= r < 2.5)
    moderate_risk = sum(1 for r in ratios if 1.3 <= r < 1.8)
    low_risk = sum(1 for r in ratios if r < 1.3)
    
    categories = ['Low Risk\n(<1.3)', 'Moderate Risk\n(1.3-1.8)', 'High Risk\n(1.8-2.5)', 'Critical Risk\n(≥2.5)']
    counts = [low_risk, moderate_risk, high_risk, critical_risk]
    colors = ['green', 'orange', 'red', 'darkred']
    
    bars = ax3.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Number of Layers')
    ax3.set_title('Activation Attack Risk Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels
    total_layers = len(ratios)
    for bar, count in zip(bars, counts):
        if count > 0:
            percentage = (count / total_layers) * 100
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Overall attack assessment with confidence interval
    overall_ratio = np.mean(ratios)
    ratio_std = np.std(ratios)
    
    if overall_ratio >= 2.5:
        risk_level = "CRITICAL"
        risk_color = "darkred"
        risk_message = "Very High\nPrivacy Risk"
    elif overall_ratio >= 1.8:
        risk_level = "HIGH"
        risk_color = "red"
        risk_message = "High\nPrivacy Risk"
    elif overall_ratio >= 1.3:
        risk_level = "MODERATE"
        risk_color = "orange"
        risk_message = "Moderate\nPrivacy Risk"
    else:
        risk_level = "LOW"
        risk_color = "green"
        risk_message = "Good Privacy\nProtection"
    
    bar = ax4.bar([risk_level], [overall_ratio], color=risk_color, alpha=0.8)
    
    # Add error bar for confidence
    ax4.errorbar([risk_level], [overall_ratio], yerr=[ratio_std], 
                fmt='none', color='black', capsize=10, capthick=2)
    
    ax4.set_ylabel('Overall Attack Ratio')
    ax4.set_title('Overall Activation Attack Assessment')
    ax4.text(0, overall_ratio/2, risk_message, ha='center', va='center', 
             fontweight='bold', fontsize=11, color='white')
    ax4.set_ylim(0, max(3.0, overall_ratio + ratio_std + 0.5))
    
    # Add threshold lines
    for threshold, color, label in [(2.5, 'darkred', 'Critical'), (1.8, 'red', 'High'), 
                                   (1.3, 'orange', 'Moderate'), (1.0, 'gray', 'Break-even')]:
        if threshold <= max(3.0, overall_ratio + ratio_std + 0.5):
            ax4.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# ACTIVATION ATTACK COMPREHENSIVE SUMMARY:
# Analysis of activation attack success across all important layers.

# INTERPRETATION:
# • Top left: Distribution comparison shows separation between forget/retain predictions
# • Top right: Success ratio distribution with statistical measures
# • Bottom left: Risk categorization using stricter activation-specific thresholds
# • Bottom right: Overall assessment with confidence interval (error bars)

# ACTIVATION-SPECIFIC FEATURES:
# • Focuses only on layers showing significant activation changes (90% of variance)
# • Uses stricter success thresholds than gradient attacks
# • Analyzes internal representation vulnerabilities rather than gradient patterns

# RISK ASSESSMENT:
# • Good separation between forget/retain distributions indicates successful attack
# • Mean ratio > 1.8 suggests significant privacy vulnerabilities in activation space
# • High percentage of high-risk layers indicates systematic representation problems

# PRIVACY IMPLICATIONS:
# • Successful activation attacks reveal internal representation leakage
# • Results show effectiveness of representation-level unlearning
# • Critical for understanding deep neural network privacy vulnerabilities
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.savefig(plot_dir / 'activation_attack_summary.png', dpi=300, bbox_inches='tight')
    plt.close()