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

def run_gradient_attack(original_model, unlearned_model, background_data, forget_data, 
                       retain_data, device='cuda', verbose=True):
    """
    Realistic gradient attack: Estimate forget signature from background data only,
    then test how well it predicts forget vs retain data.
    
    This simulates a real attack where you don't have access to the actual forget data.
    
    Phase 1: Estimate forget signature from background data (model differences)
    Phase 2: Test prediction accuracy on real forget/retain data
    """
    
    print("GRADIENT ATTACK")
    print("="*50)
    print("Phase 1: Estimate forget signature from background data only")
    print("Phase 2: Test prediction on real forget/retain data")
    print("="*50)
    
    # Phase 1: Estimate forget signature from background data
    print("\n🔍 PHASE 1: ESTIMATING FORGET SIGNATURE FROM BACKGROUND DATA")
    print("-" * 50)
    
    estimated_signature = estimate_gradient_signature(
        original_model, unlearned_model, background_data, device, verbose
    )
    
    if not estimated_signature:
        return {
            'success': False,
            'description': 'Failed to estimate gradient signature from background data',
            'phase_1_success': False,
            'phase_2_success': False
        }
    
    # Phase 2: Test the estimated signature on real data
    print("\n🎯 PHASE 2: TESTING ESTIMATED SIGNATURE ON REAL DATA")
    print("-" * 50)
    
    attack_results = test_gradient_signature(
        estimated_signature, original_model, forget_data, retain_data, device, verbose
    )
    
    # Phase 3: Create visualizations
    plot_dir = Path("./attack_plots/gradient")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    create_gradient_attack_plots(
        estimated_signature, attack_results, plot_dir, verbose
    )
    
    # Phase 4: Analyze attack success
    analysis_results = analyze_gradient_attack_results(attack_results, verbose)
    
    return {
        'estimated_signature': estimated_signature,
        'attack_results': attack_results,
        'analysis': analysis_results,
        'plot_directory': str(plot_dir),
        'phase_1_success': bool(estimated_signature),
        'phase_2_success': attack_results.get('success', False)
    }


def estimate_gradient_signature(original_model, unlearned_model, background_data, 
                               device, verbose=True):
    """
    Estimate the forget signature using only background data.
    Key insight: Model differences on background data reveal unlearning patterns.
    """
    
    if verbose:
        print("Computing gradient differences on background data...")
    
    # Get gradients from both models on background data
    orig_grads = get_model_gradients(original_model, background_data, device)
    unl_grads = get_model_gradients(unlearned_model, background_data, device)
    
    if not orig_grads or not unl_grads:
        if verbose:
            print("❌ Failed to compute gradients")
        return None
    
    # Calculate gradient differences (this is our estimated signature)
    grad_signature = {}
    total_diff_norm = 0
    
    for param_name in orig_grads:
        if param_name in unl_grads and '.weight' in param_name:
            diff = orig_grads[param_name] - unl_grads[param_name]
            layer_name = param_name.replace('.weight', '')
            grad_signature[layer_name] = diff
            total_diff_norm += torch.norm(diff).item()
    
    if verbose:
        print(f"Total gradient signature magnitude: {total_diff_norm:.6f}")
        print(f"Signature extracted from {len(grad_signature)} layers")
        
    if total_diff_norm < 1e-6:
        if verbose:
            print("⚠️  WARNING: Models show minimal differences on background data")
        return None
    
    # Normalize signatures for comparison
    normalized_signature = {}
    for layer_name, grad_diff in grad_signature.items():
        grad_norm = torch.norm(grad_diff).item()
        if grad_norm > 1e-8:
            normalized_signature[layer_name] = {
                'signature': grad_diff / grad_norm,  # Unit vector
                'magnitude': grad_norm,
                'layer_name': layer_name
            }
            
            if verbose and len(normalized_signature) <= 5:  # Show first 5
                print(f"  Layer {layer_name}: signature magnitude {grad_norm:.6f}")
    
    return normalized_signature


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


def test_gradient_signature(estimated_signature, original_model, forget_data, 
                           retain_data, device, verbose=True):
    """
    Test how well the estimated signature predicts forget vs retain data.
    """
    
    if verbose:
        print("Testing estimated signature on forget data...")
    
    # Get gradients of original model on forget data
    forget_grads = get_model_gradients(original_model, forget_data, device)
    
    if verbose:
        print("Testing estimated signature on retain data...")
    
    # Get gradients of original model on retain data  
    retain_grads = get_model_gradients(original_model, retain_data, device)
    
    if not forget_grads or not retain_grads:
        return {
            'success': False, 
            'description': 'Failed to compute test gradients',
            'forget_scores': {},
            'retain_scores': {}
        }
    
    # Calculate similarity scores with estimated signature
    forget_scores = {}
    retain_scores = {}
    
    for layer_name in estimated_signature:
        if layer_name + '.weight' not in forget_grads or layer_name + '.weight' not in retain_grads:
            continue
            
        try:
            signature = estimated_signature[layer_name]['signature']
            sig_magnitude = estimated_signature[layer_name]['magnitude']
            
            # Get test gradients for this layer
            forget_grad = forget_grads[layer_name + '.weight']
            retain_grad = retain_grads[layer_name + '.weight']
            
            # Normalize test gradients
            forget_norm = torch.norm(forget_grad).item()
            retain_norm = torch.norm(retain_grad).item()
            
            if forget_norm > 1e-8 and retain_norm > 1e-8:
                forget_normalized = forget_grad / forget_norm
                retain_normalized = retain_grad / retain_norm
                
                # Calculate similarity scores (dot product with signature)
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


def analyze_gradient_attack_results(attack_results, verbose=True):
    """Analyze the gradient attack results."""
    
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
    
    print(f"\n🔍 GRADIENT ATTACK ANALYSIS")
    print(f"   • Forget Prediction Score: {mean_forget_score:.6f}")
    print(f"   • Retain Prediction Score: {mean_retain_score:.6f}")
    print(f"   • Prediction Ratio: {prediction_ratio:.3f}")
    print(f"   • Layers Analyzed: {len(forget_scores)}")
    
    # Determine success level and privacy implications
    if prediction_ratio >= 3.0:
        success_level = "HIGH"
        success = True
        privacy_risk = "CRITICAL"
        description = f"Background gradients strongly predict forget patterns (ratio: {prediction_ratio:.3f}). CRITICAL privacy risk - forgotten data signatures can be identified from background information alone."
    elif prediction_ratio >= 2.0:
        success_level = "MODERATE"
        success = True
        privacy_risk = "HIGH"
        description = f"Background gradients moderately predict forget patterns (ratio: {prediction_ratio:.3f}). HIGH privacy risk - some forgotten information may be recoverable through gradient analysis."
    elif prediction_ratio >= 1.5:
        success_level = "LOW"
        success = False
        privacy_risk = "MODERATE"
        description = f"Background gradients weakly predict forget patterns (ratio: {prediction_ratio:.3f}). MODERATE privacy risk - limited gradient-based information leakage."
    else:
        success_level = "FAILED"
        success = False
        privacy_risk = "LOW"
        description = f"Background gradients cannot predict forget patterns (ratio: {prediction_ratio:.3f}). LOW privacy risk - good unlearning privacy protection against gradient attacks."
    
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


def create_gradient_attack_plots(estimated_signature, attack_results, plot_dir, verbose=True):
    """Create visualization plots for gradient attack analysis."""
    
    if verbose:
        print(f"\n🎨 Creating gradient attack plots in {plot_dir}...")
    
    # Plot 1: Signature Strength Analysis
    create_signature_strength_plot(estimated_signature, plot_dir)
    
    # Plot 2: Prediction Results Analysis
    create_prediction_results_plot(attack_results, plot_dir)
    
    # Plot 3: Layer-wise Attack Success
    create_layer_attack_success_plot(estimated_signature, attack_results, plot_dir)
    
    # Plot 4: Attack Performance Summary
    create_attack_summary_plot(attack_results, plot_dir)
    
    if verbose:
        print("✅ All gradient attack plots created!")


def create_signature_strength_plot(estimated_signature, plot_dir):
    """Create signature strength analysis plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    layers = list(estimated_signature.keys())
    magnitudes = [estimated_signature[l]['magnitude'] for l in layers]
    
    # Plot 1: Signature magnitudes
    bars = ax1.bar(range(len(layers)), magnitudes, alpha=0.8, edgecolor='black')
    
    # Color code bars based on strength
    max_mag = max(magnitudes) if magnitudes else 1
    for bar, mag in zip(bars, magnitudes):
        relative_strength = mag / max_mag
        if relative_strength >= 0.7:
            bar.set_color('red')  # Strong signature = high privacy risk
        elif relative_strength >= 0.4:
            bar.set_color('orange')  # Moderate signature
        elif relative_strength >= 0.2:
            bar.set_color('yellow')  # Weak signature
        else:
            bar.set_color('green')  # Very weak signature = good privacy
    
    ax1.set_xlabel('Neural Network Layers')
    ax1.set_ylabel('Signature Magnitude')
    ax1.set_title('Estimated Forget Signature Strength\n(Higher = Stronger Privacy Risk)')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels for top layers
    for bar, mag in zip(bars, magnitudes):
        if mag >= max_mag * 0.5:  # Label top 50% of layers
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_mag*0.01,
                    f'{mag:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Cumulative signature strength
    sorted_mags = sorted(magnitudes, reverse=True)
    cumulative = np.cumsum(sorted_mags) / sum(sorted_mags)
    
    ax2.plot(range(1, len(cumulative)+1), cumulative, 'o-', linewidth=2, markersize=6)
    ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% Threshold')
    ax2.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90% Threshold')
    
    ax2.set_xlabel('Number of Top Layers')
    ax2.set_ylabel('Cumulative Signature Strength')
    ax2.set_title('Cumulative Signature Distribution\n(How concentrated is the signature?)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Find concentration point
    concentration_80 = np.argmax(cumulative >= 0.8) + 1 if any(cumulative >= 0.8) else len(cumulative)
    ax2.annotate(f'80% in {concentration_80} layers', 
                xy=(concentration_80, 0.8), xytext=(concentration_80+2, 0.7),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# SIGNATURE STRENGTH ANALYSIS:
# Left: Magnitude of estimated forget signature for each layer (from background data analysis)
# Right: Cumulative distribution showing how concentrated the signature is across layers

# INTERPRETATION:
# • Higher magnitudes indicate stronger signature components (higher privacy risk)
# • Concentrated signatures (steep cumulative curve) suggest targeted vulnerabilities
# • Distributed signatures (gradual curve) indicate widespread but weaker effects

# COLOR CODING (Left plot):
# • Red bars: Strong signature components (>70% of max) - high privacy risk
# • Orange bars: Moderate signature components (40-70% of max)
# • Yellow bars: Weak signature components (20-40% of max)  
# • Green bars: Very weak components (<20% of max) - low privacy risk

# PRIVACY IMPLICATIONS:
# • Strong, concentrated signatures are easier to exploit in attacks
# • Weak, distributed signatures indicate better privacy protection
# • The signature was estimated from background data only (realistic attack scenario)
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.savefig(plot_dir / 'gradient_signature_strength.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_prediction_results_plot(attack_results, plot_dir):
    """Create prediction results analysis plot."""
    
    if not attack_results.get('success', False):
        # Create a failure plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Attack Failed\nNo prediction results to display', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(plot_dir / 'gradient_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    layers = list(attack_results['forget_scores'].keys())
    forget_scores = [attack_results['forget_scores'][l] for l in layers]
    retain_scores = [attack_results['retain_scores'][l] for l in layers]
    ratios = [f / (r + 1e-8) for f, r in zip(forget_scores, retain_scores)]
    
    # Plot 1: Side-by-side score comparison
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, forget_scores, width, label='Forget Data Prediction', 
                   color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, retain_scores, width, label='Retain Data Prediction', 
                   color='skyblue', alpha=0.8)
    
    ax1.set_xlabel('Neural Network Layers')
    ax1.set_ylabel('Prediction Score (Signature Similarity)')
    ax1.set_title('Gradient Attack Prediction Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction ratios (key attack metric)
    bars = ax2.bar(range(len(layers)), ratios, alpha=0.8, edgecolor='black')
    
    # Color code based on attack success
    for bar, ratio in zip(bars, ratios):
        if ratio >= 3.0:
            bar.set_color('darkred')  # Critical privacy risk
        elif ratio >= 2.0:
            bar.set_color('red')  # High privacy risk
        elif ratio >= 1.5:
            bar.set_color('orange')  # Moderate privacy risk
        else:
            bar.set_color('green')  # Low privacy risk
    
    # Add threshold lines
    ax2.axhline(y=3.0, color='darkred', linestyle='--', linewidth=2, label='Critical Risk (>3.0)')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='High Risk (>2.0)')
    ax2.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, label='Moderate Risk (>1.5)')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Break-even (1.0)')
    
    ax2.set_xlabel('Neural Network Layers')
    ax2.set_ylabel('Prediction Ratio (Forget/Retain)')
    ax2.set_title('Attack Success Ratio by Layer\n(Higher = Better Attack Success)')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels for high-risk layers
    for bar, ratio in zip(bars, ratios):
        if ratio >= 1.5:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ratios)*0.02,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# PREDICTION RESULTS ANALYSIS:
# Left: Comparison of how well the estimated signature predicts forget vs retain data
# Right: Ratio of prediction scores (key metric for attack success)

# INTERPRETATION:
# • Left plot: Higher bars indicate stronger prediction (signature similarity)
# • Right plot: Ratios > 1.0 mean forget data is predicted better than retain data
# • Higher ratios indicate more successful attacks (worse privacy)

# ATTACK SUCCESS CRITERIA:
# • Ratio > 3.0: Critical privacy risk - very successful attack
# • Ratio > 2.0: High privacy risk - successful attack  
# • Ratio > 1.5: Moderate privacy risk - partially successful attack
# • Ratio < 1.5: Low privacy risk - attack largely failed

# PRIVACY IMPLICATIONS:
# • High ratios mean an attacker using only background data can identify forget patterns
# • Low ratios indicate the unlearning process provides good privacy protection
# • Consistent high ratios across layers suggest systematic vulnerabilities
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.savefig(plot_dir / 'gradient_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_layer_attack_success_plot(estimated_signature, attack_results, plot_dir):
    """Create layer-wise attack success analysis plot."""
    
    if not attack_results.get('success', False):
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    layers = list(estimated_signature.keys())
    magnitudes = [estimated_signature[l]['magnitude'] for l in layers]
    
    if layers[0] in attack_results['forget_scores']:
        ratios = [attack_results['forget_scores'][l] / (attack_results['retain_scores'][l] + 1e-8) 
                 for l in layers]
    else:
        ratios = [0] * len(layers)
    
    # Plot 1: Signature magnitude vs attack success
    scatter = ax1.scatter(magnitudes, ratios, c=range(len(layers)), cmap='viridis', 
                         s=80, alpha=0.8, edgecolors='black')
    
    # Add layer labels for interesting points
    for i, layer in enumerate(layers):
        if magnitudes[i] > np.mean(magnitudes) or ratios[i] > np.mean(ratios):
            ax1.annotate(layer.replace('layer', 'L'), (magnitudes[i], ratios[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Signature Magnitude (Background Data)')
    ax1.set_ylabel('Attack Success Ratio (Forget/Retain)')
    ax1.set_title('Signature Strength vs Attack Success Correlation')
    ax1.grid(True, alpha=0.3)
    
    # Add success regions
    ax1.axhspan(2.0, max(ratios)*1.1 if ratios else 4.0, alpha=0.2, color='red', label='High Success')
    ax1.axhspan(1.5, 2.0, alpha=0.2, color='orange', label='Moderate Success')
    ax1.axhspan(0, 1.5, alpha=0.2, color='green', label='Low Success')
    ax1.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Layer Index (Network Depth)')
    
    # Plot 2: Layer efficiency analysis
    efficiency_scores = [r / (m + 1e-8) for r, m in zip(ratios, magnitudes)]
    
    bars = ax2.bar(range(len(layers)), efficiency_scores, alpha=0.8, edgecolor='black')
    
    # Color code by efficiency
    max_eff = max(efficiency_scores) if efficiency_scores else 1
    for bar, eff in zip(bars, efficiency_scores):
        relative_eff = eff / max_eff if max_eff > 0 else 0
        if relative_eff >= 0.7:
            bar.set_color('red')  # High efficiency = concerning
        elif relative_eff >= 0.4:
            bar.set_color('orange')
        elif relative_eff >= 0.2:
            bar.set_color('yellow')
        else:
            bar.set_color('green')  # Low efficiency = good privacy
    
    ax2.set_xlabel('Neural Network Layers')
    ax2.set_ylabel('Attack Efficiency (Success/Signature Strength)')
    ax2.set_title('Layer Attack Efficiency\n(Higher = More Vulnerable Per Unit Signature)')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([l.replace('layer', 'L') for l in layers], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels for high-efficiency layers
    for bar, eff in zip(bars, efficiency_scores):
        if eff >= max_eff * 0.6:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_eff*0.02,
                    f'{eff:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# LAYER-WISE ATTACK SUCCESS ANALYSIS:
# Top: Correlation between signature strength and attack success
# Bottom: Attack efficiency (success normalized by signature strength)

# INTERPRETATION:
# • Top plot: Shows if stronger signatures lead to more successful attacks
# • Bottom plot: Identifies layers that are disproportionately vulnerable
# • High efficiency layers are particularly concerning for privacy

# KEY INSIGHTS:
# • Strong positive correlation suggests signature estimation is effective
# • High-efficiency layers may have structural vulnerabilities
# • Network depth (color) may influence attack susceptibility

# PRIVACY IMPLICATIONS:
# • Layers with high efficiency scores need additional protection
# • Strong correlations indicate systematic vulnerabilities
# • Understanding layer-specific risks can guide defensive strategies
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.savefig(plot_dir / 'gradient_layer_attack_success.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_attack_summary_plot(attack_results, plot_dir):
    """Create attack performance summary plot."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    if not attack_results.get('success', False):
        # Create failure summary
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'Attack Failed\nInsufficient Data', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        plt.savefig(plot_dir / 'gradient_attack_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    forget_scores = list(attack_results['forget_scores'].values())
    retain_scores = list(attack_results['retain_scores'].values())
    ratios = [f / (r + 1e-8) for f, r in zip(forget_scores, retain_scores)]
    
    # Plot 1: Score distributions
    ax1.hist(forget_scores, bins=15, alpha=0.7, label='Forget Scores', color='coral', edgecolor='black')
    ax1.hist(retain_scores, bins=15, alpha=0.7, label='Retain Scores', color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(forget_scores), color='red', linestyle='--', linewidth=2, 
                label=f'Forget Mean: {np.mean(forget_scores):.3f}')
    ax1.axvline(np.mean(retain_scores), color='blue', linestyle='--', linewidth=2,
                label=f'Retain Mean: {np.mean(retain_scores):.3f}')
    ax1.set_xlabel('Prediction Score')
    ax1.set_ylabel('Number of Layers')
    ax1.set_title('Distribution of Prediction Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio distribution
    ax2.hist(ratios, bins=15, color='mediumpurple', alpha=0.8, edgecolor='black')
    ax2.axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2,
                label=f'Mean Ratio: {np.mean(ratios):.3f}')
    ax2.axvline(2.0, color='orange', linestyle='--', linewidth=2, label='High Risk Threshold')
    ax2.axvline(1.0, color='gray', linestyle='--', linewidth=1, label='Break-even')
    ax2.set_xlabel('Prediction Ratio (Forget/Retain)')
    ax2.set_ylabel('Number of Layers')
    ax2.set_title('Distribution of Attack Success Ratios')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Attack success categories
    high_risk = sum(1 for r in ratios if r >= 2.0)
    moderate_risk = sum(1 for r in ratios if 1.5 <= r < 2.0)
    low_risk = sum(1 for r in ratios if 1.0 <= r < 1.5)
    no_risk = sum(1 for r in ratios if r < 1.0)
    
    categories = ['No Risk\n(<1.0)', 'Low Risk\n(1.0-1.5)', 'Moderate Risk\n(1.5-2.0)', 'High Risk\n(≥2.0)']
    counts = [no_risk, low_risk, moderate_risk, high_risk]
    colors = ['green', 'yellow', 'orange', 'red']
    
    bars = ax3.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Number of Layers')
    ax3.set_title('Attack Risk Distribution Across Layers')
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels
    total_layers = len(ratios)
    for bar, count in zip(bars, counts):
        if count > 0:
            percentage = (count / total_layers) * 100
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Overall attack assessment
    overall_ratio = np.mean(ratios)
    
    if overall_ratio >= 3.0:
        risk_level = "CRITICAL"
        risk_color = "darkred"
        risk_message = "Very High\nPrivacy Risk"
    elif overall_ratio >= 2.0:
        risk_level = "HIGH"
        risk_color = "red"
        risk_message = "High\nPrivacy Risk"
    elif overall_ratio >= 1.5:
        risk_level = "MODERATE"
        risk_color = "orange"
        risk_message = "Moderate\nPrivacy Risk"
    else:
        risk_level = "LOW"
        risk_color = "green"
        risk_message = "Good Privacy\nProtection"
    
    ax4.bar([risk_level], [overall_ratio], color=risk_color, alpha=0.8)
    ax4.set_ylabel('Overall Attack Ratio')
    ax4.set_title('Overall Attack Assessment')
    ax4.text(0, overall_ratio/2, risk_message, ha='center', va='center', 
             fontweight='bold', fontsize=12, color='white')
    ax4.set_ylim(0, max(4.0, overall_ratio * 1.2))
    
    # Add threshold lines
    for threshold, color, label in [(3.0, 'darkred', 'Critical'), (2.0, 'red', 'High'), 
                                   (1.5, 'orange', 'Moderate'), (1.0, 'gray', 'Break-even')]:
        if threshold <= max(4.0, overall_ratio * 1.2):
            ax4.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    
#     # Add description
#     description = """
# ATTACK PERFORMANCE SUMMARY:
# Comprehensive analysis of gradient attack success across all layers.

# INTERPRETATION:
# • Top left: Distribution of prediction scores for forget vs retain data
# • Top right: Distribution of success ratios (key attack metric)
# • Bottom left: Categorization of layers by privacy risk level
# • Bottom right: Overall attack assessment

# RISK ASSESSMENT:
# • High separation between forget/retain score distributions indicates successful attack
# • Mean ratio > 2.0 suggests significant privacy vulnerabilities
# • High percentage of high-risk layers indicates systematic problems

# PRIVACY IMPLICATIONS:
# • This attack uses only background data (realistic threat model)
# • Successful attacks mean forgotten data patterns can be identified
# • Results inform the effectiveness of the unlearning process
# """
    
#     fig.text(0.02, 0.02, description, fontsize=9, verticalalignment='bottom',
#              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.savefig(plot_dir / 'gradient_attack_summary.png', dpi=300, bbox_inches='tight')
    plt.close()