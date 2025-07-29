import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_gradient_results_standardized(gradient_comparison, verbose=True):
    """
    Standardized analysis for gradient attack results.
    Success = forget and retain gradient differences are DIFFERENT.
    """
    
    print("\nGRADIENT ATTACK ANALYSIS")
    print("-" * 40)
    
    # Extract similarity scores
    similarities = [c['cosine_similarity'] for c in gradient_comparison.values() if c]
    
    if not similarities:
        return create_standard_result(
            success=False,
            attack_type="Gradient",
            description='Could not compute gradient similarities',
            confidence_level='None',
            attack_strength=0.0,
            attack_specific={'error': 'computation_failed'}
        )
    
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    median_similarity = np.median(similarities)
    
    # Convert similarity to attack strength (lower similarity = higher attack strength)
    attack_strength = max(0.0, 1.0 - mean_similarity)
    
    print(f"Forget vs Retain Gradient Differences: {mean_similarity:.3f} (±{std_similarity:.3f})")
    print(f"Median similarity: {median_similarity:.3f}")
    print(f"Attack strength: {attack_strength:.3f}")
    
    # Standardized success criteria
    if attack_strength >= 0.7:  # mean_similarity <= 0.3
        success = True
        status = "ATTACK_SUCCESSFUL"
        confidence_level = "High"
        description = f"Highly distinct gradient patterns detected"
    elif attack_strength >= 0.4:  # mean_similarity <= 0.6
        success = True
        status = "ATTACK_SUCCESSFUL"
        confidence_level = "Moderate"
        description = f"Distinct gradient patterns detected"
    elif attack_strength >= 0.2:  # mean_similarity <= 0.8
        success = False
        status = "PARTIALLY_SUCCESSFUL"
        confidence_level = "Low"
        description = f"Some gradient pattern differences detected"
    else:
        success = False
        status = "ATTACK_FAILED"
        confidence_level = "None"
        description = f"No distinct gradient patterns detected"
    
    print(f"\n✅ {status} ({confidence_level} confidence)")
    print(f"Description: {description}")
    
    # Attack-specific metrics
    attack_specific = {
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'median_similarity': median_similarity,
        'similarity_threshold_high': 0.3,
        'similarity_threshold_moderate': 0.6,
        'num_layers_analyzed': len(similarities)
    }
    
    # GAN optimization guidance
    if success and confidence_level == "High":
        print("💡 GAN Optimization: Strong signal - use gradient differences as target signature")
        gan_guidance = "high_quality_reconstruction_expected"
    elif success:
        print("💡 GAN Optimization: Moderate signal - use weighted gradient differences")
        gan_guidance = "moderate_quality_reconstruction_expected"
    else:
        print("❌ GAN Optimization: Weak signal - reconstruction will be difficult")
        gan_guidance = "poor_reconstruction_expected"
    
    attack_specific['gan_guidance'] = gan_guidance
    
    return create_standard_result(
        success=success,
        attack_type="Gradient",
        status=status,
        description=description,
        confidence_level=confidence_level,
        attack_strength=attack_strength,
        attack_specific=attack_specific
    )


def analyze_realistic_results_standardized(results, verbose=True):
    """
    Standardized analysis for realistic projection attack results.
    Success = estimated subspace predicts forget data better than retain data.
    """
    
    print(f"\nREALISTIC PROJECTION ATTACK ANALYSIS")
    print("-" * 40)
    
    if not results.get('success', False):
        return create_standard_result(
            success=False,
            attack_type="Realistic_Projection",
            description=results.get('description', 'Unknown error'),
            confidence_level='None',
            attack_strength=0.0,
            attack_specific={'error': 'computation_failed'}
        )
    
    forget_projs = list(results['forget_projections'].values())
    retain_projs = list(results['retain_projections'].values())
    
    if not forget_projs or not retain_projs:
        return create_standard_result(
            success=False,
            attack_type="Realistic_Projection",
            description='No projection results to analyze',
            confidence_level='None',
            attack_strength=0.0,
            attack_specific={'error': 'no_projections'}
        )
    
    mean_forget_proj = np.mean(forget_projs)
    mean_retain_proj = np.mean(retain_projs)
    
    # Calculate ratios for each layer
    layer_ratios = []
    for layer in results['forget_projections']:
        if layer in results['retain_projections']:
            forget_p = results['forget_projections'][layer]
            retain_p = results['retain_projections'][layer]
            ratio = forget_p / (retain_p + 1e-8)
            layer_ratios.append(ratio)
    
    mean_ratio = np.mean(layer_ratios) if layer_ratios else 0
    median_ratio = np.median(layer_ratios) if layer_ratios else 0
    
    # Convert ratio to attack strength (higher ratio = higher attack strength, capped at 1.0)
    attack_strength = min(1.0, max(0.0, (mean_ratio - 1.0) / 3.0))  # Scale so ratio=4.0 gives strength=1.0
    
    print(f"Mean forget projection strength: {mean_forget_proj:.6f}")
    print(f"Mean retain projection strength: {mean_retain_proj:.6f}")
    print(f"Mean forget/retain ratio: {mean_ratio:.3f}")
    print(f"Median ratio: {median_ratio:.3f}")
    print(f"Attack strength: {attack_strength:.3f}")
    
    # Standardized success criteria
    if mean_ratio >= 3.0:
        success = True
        status = "ATTACK_SUCCESSFUL"
        confidence_level = "High"
        description = f"Strong predictive power detected - high privacy risk"
    elif mean_ratio >= 2.0:
        success = True
        status = "ATTACK_SUCCESSFUL"
        confidence_level = "Moderate"
        description = f"Moderate predictive power detected - privacy risk"
    elif mean_ratio >= 1.5:
        success = False
        status = "PARTIALLY_SUCCESSFUL"
        confidence_level = "Low"
        description = f"Weak predictive power detected - limited privacy risk"
    else:
        success = False
        status = "ATTACK_FAILED"
        confidence_level = "None"
        description = f"No predictive power - good privacy protection"
    
    print(f"\n✅ {status} ({confidence_level} confidence)")
    print(f"Description: {description}")
    
    # Attack-specific metrics
    attack_specific = {
        'mean_forget_projection': mean_forget_proj,
        'mean_retain_projection': mean_retain_proj,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'ratio_threshold_high': 3.0,
        'ratio_threshold_moderate': 2.0,
        'ratio_threshold_low': 1.5,
        'num_layers_analyzed': len(layer_ratios),
        'privacy_risk_level': confidence_level.lower() if success else 'low'
    }
    
    # Privacy implications
    if success and confidence_level == "High":
        print("⚠️  HIGH PRIVACY RISK: Background data alone reveals forget patterns!")
        privacy_guidance = "high_privacy_risk"
    elif success:
        print("⚠️  MODERATE PRIVACY RISK: Some pattern leakage detected")
        privacy_guidance = "moderate_privacy_risk"
    else:
        print("✅ LOW PRIVACY RISK: Good privacy protection")
        privacy_guidance = "low_privacy_risk"
    
    attack_specific['privacy_guidance'] = privacy_guidance
    
    return create_standard_result(
        success=success,
        attack_type="Realistic_Projection",
        status=status,
        description=description,
        confidence_level=confidence_level,
        attack_strength=attack_strength,
        attack_specific=attack_specific
    )


def analyze_focused_activation_results_standardized(activation_comparison, important_layers, 
                                                  selected_layer_names, verbose=True):
    """
    Standardized analysis for focused activation attack results.
    Success = forget and retain activation differences are DIFFERENT on important layers.
    """
    
    print("\nFOCUSED ACTIVATION ATTACK ANALYSIS")
    print("-" * 40)
    
    # Extract similarity scores
    similarities = [c['cosine_similarity'] for c in activation_comparison.values() if c]
    
    if not similarities:
        return create_standard_result(
            success=False,
            attack_type="Focused_Activation",
            description='Could not compute focused activation similarities',
            confidence_level='None',
            attack_strength=0.0,
            attack_specific={'error': 'computation_failed'}
        )
    
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    median_similarity = np.median(similarities)
    
    # Convert similarity to attack strength (lower similarity = higher attack strength)
    attack_strength = max(0.0, 1.0 - mean_similarity)
    
    print(f"Focused Analysis Results:")
    print(f"  Selected {len(selected_layer_names)} most important layers")
    print(f"  Mean similarity: {mean_similarity:.3f} (±{std_similarity:.3f})")
    print(f"  Median similarity: {median_similarity:.3f}")
    print(f"  Attack strength: {attack_strength:.3f}")
    
    # Show layer importance distribution
    if len(important_layers) > 0:
        total_importance = sum(diff for _, diff in important_layers)
        print(f"  Top 5 most important layers:")
        for i, (layer, diff) in enumerate(important_layers[:5]):
            importance_pct = (diff / total_importance) * 100 if total_importance > 0 else 0
            print(f"    {i+1}. {layer}: {importance_pct:.1f}% of total difference")
    
    # Standardized success criteria (more stringent for focused analysis)
    if attack_strength >= 0.8:  # mean_similarity <= 0.2
        success = True
        status = "ATTACK_SUCCESSFUL"
        confidence_level = "High"
        description = f"Highly distinct activation patterns detected"
    elif attack_strength >= 0.5:  # mean_similarity <= 0.5
        success = True
        status = "ATTACK_SUCCESSFUL"
        confidence_level = "Moderate"
        description = f"Distinct activation patterns detected"
    elif attack_strength >= 0.3:  # mean_similarity <= 0.7
        success = False
        status = "PARTIALLY_SUCCESSFUL"
        confidence_level = "Low"
        description = f"Some activation pattern differences detected"
    else:
        success = False
        status = "ATTACK_FAILED"
        confidence_level = "None"
        description = f"No distinct activation patterns detected"
    
    print(f"\n✅ {status} ({confidence_level} confidence)")
    print(f"Description: {description}")
    
    # Attack-specific metrics
    attack_specific = {
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'median_similarity': median_similarity,
        'similarity_threshold_high': 0.2,
        'similarity_threshold_moderate': 0.5,
        'similarity_threshold_low': 0.7,
        'num_focused_layers': len(selected_layer_names),
        'num_important_layers': len(important_layers),
        'layer_selection_variance': 0.9,  # From original analysis
    }
    
    # Add layer importance statistics
    if important_layers:
        importance_values = [diff for _, diff in important_layers]
        attack_specific.update({
            'total_layer_importance': sum(importance_values),
            'max_layer_importance': max(importance_values),
            'top_5_importance_pct': sum(importance_values[:5]) / sum(importance_values) * 100 if sum(importance_values) > 0 else 0
        })
    
    # GAN optimization guidance
    if success and confidence_level == "High":
        print("💡 GAN Optimization: Strong focused signal - use weighted activation differences")
        gan_guidance = "high_quality_reconstruction_expected"
    elif success:
        print("💡 GAN Optimization: Moderate signal - use layer-weighted approach")
        gan_guidance = "moderate_quality_reconstruction_expected"
    else:
        print("❌ GAN Optimization: Weak signal - reconstruction will be difficult")
        gan_guidance = "poor_reconstruction_expected"
    
    attack_specific['gan_guidance'] = gan_guidance
    
    return create_standard_result(
        success=success,
        attack_type="Focused_Activation",
        status=status,
        description=description,
        confidence_level=confidence_level,
        attack_strength=attack_strength,
        attack_specific=attack_specific
    )


def create_standard_result(success, attack_type, status=None, description="", 
                          confidence_level="None", attack_strength=0.0, 
                          attack_specific=None):
    """
    Create standardized result structure for all attacks.
    
    Args:
        success: bool - Whether the attack was successful
        attack_type: str - Type of attack ("Gradient", "Realistic_Projection", "Focused_Activation")
        status: str - Attack status ("ATTACK_SUCCESSFUL", "PARTIALLY_SUCCESSFUL", "ATTACK_FAILED")
        description: str - Human-readable description of results
        confidence_level: str - Confidence level ("High", "Moderate", "Low", "None")
        attack_strength: float - Normalized attack strength (0.0 to 1.0, higher is better)
        attack_specific: dict - Attack-specific metrics and details
    
    Returns:
        dict: Standardized result structure
    """
    
    if status is None:
        if success and confidence_level in ["High", "Moderate"]:
            status = "ATTACK_SUCCESSFUL"
        elif success and confidence_level == "Low":
            status = "PARTIALLY_SUCCESSFUL"
        else:
            status = "ATTACK_FAILED"
    
    # Determine recoverable signal based on success and strength
    recoverable_signal = success or attack_strength >= 0.2
    
    # Create GAN optimization recommendation
    if attack_strength >= 0.7:
        gan_optimization = "high_quality_expected"
    elif attack_strength >= 0.4:
        gan_optimization = "moderate_quality_expected"
    elif attack_strength >= 0.2:
        gan_optimization = "low_quality_possible"
    else:
        gan_optimization = "reconstruction_unlikely"
    
    result = {
        # Core standardized fields
        'success': success,
        'attack_type': attack_type,
        'status': status,
        'description': description,
        'confidence_level': confidence_level,
        'attack_strength': attack_strength,  # 0.0 to 1.0 scale
        'recoverable_signal': recoverable_signal,
        'gan_optimization': gan_optimization,
        
        # Metadata
        'analysis_timestamp': np.datetime64('now'),
        'standardized_version': '1.0',
        
        # Attack-specific details
        'attack_specific': attack_specific if attack_specific is not None else {}
    }
    
    return result


def compare_attack_results(gradient_result, projection_result, activation_result, verbose=True):
    """
    Compare results from all three standardized attacks.
    
    Args:
        gradient_result: dict - Result from analyze_gradient_results_standardized
        projection_result: dict - Result from analyze_realistic_results_standardized  
        activation_result: dict - Result from analyze_focused_activation_results_standardized
        verbose: bool - Whether to print detailed comparison
    
    Returns:
        dict: Comprehensive comparison and recommendations
    """
    
    if verbose:
        print("\n" + "="*60)
        print("COMPREHENSIVE ATTACK COMPARISON")
        print("="*60)
    
    results = {
        'Gradient': gradient_result,
        'Realistic_Projection': projection_result,
        'Focused_Activation': activation_result
    }
    
    # Extract key metrics
    attack_strengths = {name: result['attack_strength'] for name, result in results.items()}
    success_count = sum(1 for result in results.values() if result['success'])
    
    # Find strongest attack
    strongest_attack = max(attack_strengths.items(), key=lambda x: x[1])
    
    if verbose:
        print(f"\nATTACK STRENGTH COMPARISON:")
        for name, strength in sorted(attack_strengths.items(), key=lambda x: x[1], reverse=True):
            result = results[name]
            print(f"  {name:20s}: {strength:.3f} ({result['confidence_level']:8s}) - {result['status']}")
        
        print(f"\nSUCCESSFUL ATTACKS: {success_count}/3")
        print(f"STRONGEST ATTACK: {strongest_attack[0]} (strength: {strongest_attack[1]:.3f})")
    
    # Overall assessment
    max_strength = strongest_attack[1]
    if success_count >= 2 and max_strength >= 0.7:
        overall_risk = "HIGH"
        overall_description = "Multiple strong attacks succeeded - significant privacy vulnerability"
    elif success_count >= 1 and max_strength >= 0.5:
        overall_risk = "MODERATE" 
        overall_description = "At least one attack succeeded - privacy concerns exist"
    elif success_count >= 1 or max_strength >= 0.3:
        overall_risk = "LOW"
        overall_description = "Limited attack success - some privacy leakage possible"
    else:
        overall_risk = "MINIMAL"
        overall_description = "No significant attacks succeeded - good privacy protection"
    
    # GAN reconstruction prospects
    successful_attacks = [name for name, result in results.items() if result['success']]
    if len(successful_attacks) >= 2:
        gan_recommendation = "Multi-attack ensemble approach recommended"
    elif len(successful_attacks) == 1:
        gan_recommendation = f"Focus on {successful_attacks[0]} attack signature"
    else:
        gan_recommendation = "GAN reconstruction unlikely to succeed"
    
    if verbose:
        print(f"\nOVERALL PRIVACY RISK: {overall_risk}")
        print(f"Assessment: {overall_description}")
        print(f"GAN Recommendation: {gan_recommendation}")
    
    comparison_result = {
        'attack_results': results,
        'attack_strengths': attack_strengths,
        'successful_attacks': successful_attacks,
        'success_count': success_count,
        'strongest_attack': strongest_attack[0],
        'max_attack_strength': max_strength,
        'overall_risk_level': overall_risk,
        'overall_description': overall_description,
        'gan_recommendation': gan_recommendation,
        'comparison_timestamp': np.datetime64('now')
    }
    
    return comparison_result


def create_standardized_gradient_plots(forget_grad_diff, retain_grad_diff, 
                                      gradient_comparison, results, plot_dir, verbose=True):
    """
    Create standardized visualization plots for gradient attack results.
    """
    
    if verbose:
        print(f"Creating standardized gradient plots in {plot_dir}...")
    
    # Extract data for plotting
    layers = list(gradient_comparison.keys())
    similarities = [gradient_comparison[l]['cosine_similarity'] for l in layers]
    l2_distances = [gradient_comparison[l]['relative_l2_distance'] for l in layers]
    
    # Get attack strength from standardized results
    attack_strength = results['attack_strength']
    confidence_level = results['confidence_level']
    
    # Plot 1: Gradient pattern similarities with standardized thresholds
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(layers)), similarities, color='orange', alpha=0.7)
    
    # Add standardized threshold lines
    plt.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='High success threshold')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate success threshold')
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Low success threshold')
    
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity (Lower = Better)')
    plt.title(f'Gradient Attack Results\nStrength: {attack_strength:.3f} | Confidence: {confidence_level}')
    plt.xticks(range(len(layers)), [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Color bars based on standardized success criteria
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        if sim <= 0.3:
            bar.set_color('green')
        elif sim <= 0.6:
            bar.set_color('orange')
        elif sim <= 0.8:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
        
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{sim:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'gradient_similarities_standardized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Attack strength summary
    create_attack_strength_summary_plot(results, plot_dir, 'gradient')
    
    # Plot 3: Layer-wise analysis
    create_layer_analysis_plot(similarities, layers, "Similarity", "Gradient Attack Layer Analysis", 
                              plot_dir / 'gradient_layer_analysis.png')


def create_standardized_projection_plots(projection_results, results, plot_dir, verbose=True):
    """
    Create standardized visualization plots for realistic projection attack results.
    """
    
    if verbose:
        print(f"Creating standardized projection plots in {plot_dir}...")
    
    if not projection_results.get('success', False):
        return
    
    # Extract data
    layers = list(projection_results['forget_projections'].keys())
    forget_projs = [projection_results['forget_projections'][l] for l in layers]
    retain_projs = [projection_results['retain_projections'][l] for l in layers]
    ratios = [forget_projs[i] / (retain_projs[i] + 1e-8) for i in range(len(layers))]
    
    attack_strength = results['attack_strength']
    confidence_level = results['confidence_level']
    
    # Plot 1: Projection strengths comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    plt.bar(x - width/2, forget_projs, width, label='Forget Data Projection', color='red', alpha=0.7)
    plt.bar(x + width/2, retain_projs, width, label='Retain Data Projection', color='blue', alpha=0.7)
    
    plt.xlabel('Layer')
    plt.ylabel('Projection Strength')
    plt.title(f'Projection Attack Results\nStrength: {attack_strength:.3f} | Confidence: {confidence_level}')
    plt.xticks(x, [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_dir / 'projection_strengths_standardized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Prediction ratios with standardized thresholds
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(layers)), ratios, color='green', alpha=0.7)
    
    # Add standardized threshold lines
    plt.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='High success threshold')
    plt.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Moderate success threshold')
    plt.axhline(y=1.5, color='yellow', linestyle='--', alpha=0.7, label='Low success threshold')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Break-even')
    
    plt.xlabel('Layer')
    plt.ylabel('Forget/Retain Projection Ratio (Higher = Better)')
    plt.title(f'Projection Attack Prediction Performance\nStrength: {attack_strength:.3f} | Confidence: {confidence_level}')
    plt.xticks(range(len(layers)), [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Color bars based on standardized success criteria
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        if ratio >= 3.0:
            bar.set_color('red')
        elif ratio >= 2.0:
            bar.set_color('orange')
        elif ratio >= 1.5:
            bar.set_color('yellow')
        else:
            bar.set_color('green')
        
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'projection_ratios_standardized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Attack strength summary
    create_attack_strength_summary_plot(results, plot_dir, 'projection')
    
    # Plot 4: Privacy risk assessment
    create_privacy_risk_plot(results, plot_dir)


def create_standardized_activation_plots(forget_activation_diff, retain_activation_diff,
                                        activation_comparison, important_layers, results,
                                        plot_dir, verbose=True):
    """
    Create standardized visualization plots for focused activation attack results.
    """
    
    if verbose:
        print(f"Creating standardized activation plots in {plot_dir}...")
    
    # Extract data
    layers = list(activation_comparison.keys())
    similarities = [activation_comparison[l]['cosine_similarity'] for l in layers]
    
    attack_strength = results['attack_strength']
    confidence_level = results['confidence_level']
    
    # Get layer importance scores
    layer_importance = {layer: diff for layer, diff in important_layers}
    importance_scores = [layer_importance.get(l, 0.0) for l in layers]
    
    # Plot 1: Combined similarity and importance analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top plot: Similarities
    bars1 = ax1.bar(range(len(layers)), similarities, color='purple', alpha=0.7)
    
    # Add standardized threshold lines
    ax1.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='High success threshold')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate success threshold')
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Low success threshold')
    
    ax1.set_ylabel('Cosine Similarity (Lower = Better)')
    ax1.set_title(f'Activation Attack Results\nStrength: {attack_strength:.3f} | Confidence: {confidence_level}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Color bars based on standardized success criteria
    for i, (bar, sim) in enumerate(zip(bars1, similarities)):
        if sim <= 0.2:
            bar.set_color('green')
        elif sim <= 0.5:
            bar.set_color('orange')
        elif sim <= 0.7:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
        
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
    plt.savefig(plot_dir / 'activation_analysis_standardized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Attack strength summary
    create_attack_strength_summary_plot(results, plot_dir, 'activation')
    
    # Plot 3: Layer importance ranking
    create_layer_importance_ranking_plot(important_layers, plot_dir)


def create_attack_strength_summary_plot(results, plot_dir, attack_name):
    """
    Create standardized attack strength summary plot.
    """
    plt.figure(figsize=(8, 6))
    
    attack_strength = results['attack_strength']
    confidence_level = results['confidence_level']
    status = results['status']
    
    # Determine color based on standardized criteria
    if attack_strength >= 0.7:
        color = 'green'
        label = 'HIGH'
        message = 'Strong attack signal'
    elif attack_strength >= 0.4:
        color = 'orange'
        label = 'MODERATE'
        message = 'Moderate attack signal'
    elif attack_strength >= 0.2:
        color = 'yellow'
        label = 'LOW'
        message = 'Weak attack signal'
    else:
        color = 'red'
        label = 'MINIMAL'
        message = 'No significant signal'
    
    plt.bar([label], [attack_strength], color=color, alpha=0.7)
    plt.ylabel('Attack Strength (0.0 - 1.0)')
    plt.title(f'{attack_name.title()} Attack Strength\nStrength: {attack_strength:.3f} | Status: {status}')
    plt.ylim(0, 1.0)
    
    # Add horizontal reference lines
    plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High threshold')
    plt.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
    plt.axhline(y=0.2, color='yellow', linestyle='--', alpha=0.5, label='Low threshold')
    
    plt.text(0, attack_strength/2, message, ha='center', va='center', fontweight='bold')
    plt.text(0, attack_strength + 0.05, f'Confidence: {confidence_level}', 
             ha='center', va='bottom', fontsize=10, style='italic')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{attack_name}_strength_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_layer_analysis_plot(values, layers, value_name, title, save_path):
    """
    Create standardized layer analysis plot.
    """
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(layers)), values, alpha=0.7)
    
    # Color gradient based on values
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Layer')
    plt.ylabel(value_name)
    plt.title(title)
    plt.xticks(range(len(layers)), [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_privacy_risk_plot(results, plot_dir):
    """
    Create privacy risk assessment plot for projection attacks.
    """
    plt.figure(figsize=(8, 6))
    
    risk_level = results['attack_specific'].get('privacy_risk_level', 'low')
    attack_strength = results['attack_strength']
    
    # Map risk levels to colors and positions
    risk_mapping = {
        'high': {'color': 'red', 'pos': 0.8, 'label': 'HIGH RISK'},
        'moderate': {'color': 'orange', 'pos': 0.6, 'label': 'MODERATE RISK'},
        'low': {'color': 'yellow', 'pos': 0.4, 'label': 'LOW RISK'},
        'none': {'color': 'green', 'pos': 0.2, 'label': 'MINIMAL RISK'}
    }
    
    risk_info = risk_mapping.get(risk_level, risk_mapping['none'])
    
    plt.bar([risk_info['label']], [attack_strength], color=risk_info['color'], alpha=0.7)
    plt.ylabel('Attack Strength')
    plt.title(f'Privacy Risk Assessment\nStrength: {attack_strength:.3f}')
    plt.ylim(0, 1.0)
    
    # Add message
    if risk_level == 'high':
        message = 'Background data\nreveals patterns!'
    elif risk_level == 'moderate':
        message = 'Some pattern\nleakage detected'
    else:
        message = 'Good privacy\nprotection'
    
    plt.text(0, attack_strength/2, message, ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'privacy_risk_assessment.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_layer_importance_ranking_plot(important_layers, plot_dir):
    """
    Create layer importance ranking plot for activation attacks.
    """
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


def create_comprehensive_comparison_plot(comparison_result, plot_dir, verbose=True):
    """
    Create comprehensive comparison plot of all three attacks.
    """
    if verbose:
        print(f"Creating comprehensive comparison plot in {plot_dir}...")
    
    plt.figure(figsize=(15, 10))
    
    # Extract data
    attack_names = list(comparison_result['attack_strengths'].keys())
    attack_strengths = list(comparison_result['attack_strengths'].values())
    success_indicators = [comparison_result['attack_results'][name]['success'] for name in attack_names]
    confidence_levels = [comparison_result['attack_results'][name]['confidence_level'] for name in attack_names]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Attack strength comparison
    colors = ['green' if success else 'red' for success in success_indicators]
    bars1 = ax1.bar(attack_names, attack_strengths, color=colors, alpha=0.7)
    
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High threshold')
    ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
    ax1.axhline(y=0.2, color='yellow', linestyle='--', alpha=0.5, label='Low threshold')
    
    ax1.set_ylabel('Attack Strength')
    ax1.set_title('Attack Strength Comparison')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, strength, conf in zip(bars1, attack_strengths, confidence_levels):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{strength:.3f}\n({conf})', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Success count
    success_count = comparison_result['success_count']
    ax2.pie([success_count, 3-success_count], labels=[f'Successful ({success_count})', f'Failed ({3-success_count})'],
           colors=['green', 'red'], autopct='%1.0f%%', startangle=90)
    ax2.set_title('Attack Success Rate')
    
    # Plot 3: Risk level assessment
    risk_level = comparison_result['overall_risk_level']
    risk_colors = {'HIGH': 'red', 'MODERATE': 'orange', 'LOW': 'yellow', 'MINIMAL': 'green'}
    
    ax3.bar([risk_level], [comparison_result['max_attack_strength']], 
           color=risk_colors.get(risk_level, 'gray'), alpha=0.7)
    ax3.set_ylabel('Max Attack Strength')
    ax3.set_title('Overall Privacy Risk Level')
    ax3.set_ylim(0, 1.0)
    
    # Plot 4: GAN reconstruction prospects
    gan_scores = []
    gan_labels = []
    for name in attack_names:
        gan_guidance = comparison_result['attack_results'][name]['gan_optimization']
        if 'high' in gan_guidance:
            gan_scores.append(0.8)
        elif 'moderate' in gan_guidance:
            gan_scores.append(0.6)
        elif 'low' in gan_guidance:
            gan_scores.append(0.4)
        else:
            gan_scores.append(0.2)
        gan_labels.append(name)
    
    bars4 = ax4.bar(gan_labels, gan_scores, color='purple', alpha=0.7)
    ax4.set_ylabel('GAN Reconstruction Prospect')
    ax4.set_title('GAN Optimization Prospects')
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Comprehensive Attack Analysis Summary\n'
                f'Overall Risk: {risk_level} | Successful Attacks: {success_count}/3 | '
                f'Strongest: {comparison_result["strongest_attack"]}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'comprehensive_attack_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


# Complete standardized usage example
def run_complete_standardized_attack_pipeline(original_model, unlearned_model, 
                                            background_data, forget_data, retain_data,
                                            device='cuda', base_plot_dir='./plots', verbose=True):
    """
    Complete pipeline using all standardized attack analysis and plotting functions.
    
    This replaces the individual attack functions and provides unified analysis.
    """
    
    print("RUNNING COMPLETE STANDARDIZED ATTACK PIPELINE")
    print("="*60)
    
    # Run all three attacks (using your existing attack code)
    # Note: You would replace these with your actual attack function calls
    
    # 1. Run gradient attack
    print("\n1. RUNNING GRADIENT ATTACK...")
    # gradient_attack_results = run_gradient_attack(...)  # Your existing function
    # For demo, using mock data:
    mock_gradient_comparison = {
        'layer1': {'cosine_similarity': 0.2, 'relative_l2_distance': 0.8},
        'layer2': {'cosine_similarity': 0.4, 'relative_l2_distance': 0.6},
        'layer3': {'cosine_similarity': 0.1, 'relative_l2_distance': 0.9},
        'layer4': {'cosine_similarity': 0.3, 'relative_l2_distance': 0.7}
    }
    
    # 2. Run projection attack  
    print("\n2. RUNNING PROJECTION ATTACK...")
    # projection_attack_results = run_realistic_projection_attack(...)  # Your existing function
    # For demo, using mock data:
    mock_projection_results = {
        'success': True,
        'forget_projections': {'layer1': 0.8, 'layer2': 0.6, 'layer3': 0.9},
        'retain_projections': {'layer1': 0.3, 'layer2': 0.4, 'layer3': 0.2}
    }
    
    # 3. Run activation attack
    print("\n3. RUNNING ACTIVATION ATTACK...")
    # activation_attack_results = run_activation_attack(...)  # Your existing function
    # For demo, using mock data:
    mock_activation_comparison = {
        'layer1': {'cosine_similarity': 0.3},
        'layer2': {'cosine_similarity': 0.1}, 
        'layer3': {'cosine_similarity': 0.5}
    }
    mock_important_layers = [('layer2', 0.9), ('layer1', 0.7), ('layer3', 0.4)]
    mock_selected_layers = ['layer1', 'layer2', 'layer3']
    
    # STANDARDIZED ANALYSIS
    print("\n" + "="*60)
    print("STANDARDIZED ANALYSIS PHASE")
    print("="*60)
    
    # Analyze each attack with standardized functions
    gradient_results = analyze_gradient_results_standardized(mock_gradient_comparison, verbose=verbose)
    projection_results = analyze_realistic_results_standardized(mock_projection_results, verbose=verbose)
    activation_results = analyze_focused_activation_results_standardized(
        mock_activation_comparison, mock_important_layers, mock_selected_layers, verbose=verbose
    )
    
    # Comprehensive comparison
    overall_comparison = compare_attack_results(
        gradient_results, projection_results, activation_results, verbose=verbose
    )
    
    # STANDARDIZED PLOTTING
    print("\n" + "="*60)
    print("STANDARDIZED PLOTTING PHASE")
    print("="*60)
    
    # Create plot directories
    from pathlib import Path
    base_plot_path = Path(base_plot_dir)
    gradient_plot_dir = base_plot_path / "gradient_standardized"
    projection_plot_dir = base_plot_path / "projection_standardized"  
    activation_plot_dir = base_plot_path / "activation_standardized"
    comparison_plot_dir = base_plot_path / "comparison"
    
    for plot_dir in [gradient_plot_dir, projection_plot_dir, activation_plot_dir, comparison_plot_dir]:
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create standardized plots for each attack
    # Note: You would pass your actual attack data here
    mock_forget_grad_diff = {}  # Your actual gradient differences
    mock_retain_grad_diff = {}  # Your actual gradient differences
    
    create_standardized_gradient_plots(
        mock_forget_grad_diff, mock_retain_grad_diff, mock_gradient_comparison, 
        gradient_results, gradient_plot_dir, verbose=verbose
    )
    
    create_standardized_projection_plots(
        mock_projection_results, projection_results, projection_plot_dir, verbose=verbose
    )
    
    mock_forget_activation_diff = {}  # Your actual activation differences
    mock_retain_activation_diff = {}  # Your actual activation differences
    
    create_standardized_activation_plots(
        mock_forget_activation_diff, mock_retain_activation_diff, mock_activation_comparison,
        mock_important_layers, activation_results, activation_plot_dir, verbose=verbose
    )
    
    # Create comprehensive comparison plot
    create_comprehensive_comparison_plot(overall_comparison, comparison_plot_dir, verbose=verbose)
    
    # FINAL RESULTS
    print("\n" + "="*60)
    print("FINAL STANDARDIZED RESULTS")
    print("="*60)
    
    print(f"\nATTACK STRENGTH SUMMARY:")
    print(f"  Gradient Attack:    {gradient_results['attack_strength']:.3f} ({gradient_results['confidence_level']})")
    print(f"  Projection Attack:  {projection_results['attack_strength']:.3f} ({projection_results['confidence_level']})")
    print(f"  Activation Attack:  {activation_results['attack_strength']:.3f} ({activation_results['confidence_level']})")
    
    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Privacy Risk Level: {overall_comparison['overall_risk_level']}")
    print(f"  Successful Attacks: {overall_comparison['success_count']}/3")
    print(f"  Strongest Attack:   {overall_comparison['strongest_attack']}")
    print(f"  Max Attack Strength: {overall_comparison['max_attack_strength']:.3f}")
    
    print(f"\nGAN RECONSTRUCTION GUIDANCE:")
    print(f"  Recommendation: {overall_comparison['gan_recommendation']}")
    
    return {
        'gradient_results': gradient_results,
        'projection_results': projection_results, 
        'activation_results': activation_results,
        'overall_comparison': overall_comparison,
        'plot_directories': {
            'gradient': gradient_plot_dir,
            'projection': projection_plot_dir,
            'activation': activation_plot_dir,
            'comparison': comparison_plot_dir
        }
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing standardized analysis and plotting functions...")
    
    # Test individual analysis functions
    mock_gradient_comparison = {
        'layer1': {'cosine_similarity': 0.2, 'relative_l2_distance': 0.8},
        'layer2': {'cosine_similarity': 0.4, 'relative_l2_distance': 0.6},
        'layer3': {'cosine_similarity': 0.1, 'relative_l2_distance': 0.9},
        'layer4': {'cosine_similarity': 0.3, 'relative_l2_distance': 0.7}
    }
    
    mock_projection_results = {
        'success': True,
        'forget_projections': {'layer1': 0.8, 'layer2': 0.6, 'layer3': 0.9},
        'retain_projections': {'layer1': 0.3, 'layer2': 0.4, 'layer3': 0.2}
    }
    
    mock_activation_comparison = {
        'layer1': {'cosine_similarity': 0.3},
        'layer2': {'cosine_similarity': 0.1}, 
        'layer3': {'cosine_similarity': 0.5}
    }
    mock_important_layers = [('layer2', 0.9), ('layer1', 0.7), ('layer3', 0.4)]
    mock_selected_layers = ['layer1', 'layer2', 'layer3']
    
    # Test each analysis function
    grad_result = analyze_gradient_results_standardized(mock_gradient_comparison)
    proj_result = analyze_realistic_results_standardized(mock_projection_results)
    act_result = analyze_focused_activation_results_standardized(
        mock_activation_comparison, mock_important_layers, mock_selected_layers
    )
    
    # Compare all results
    comparison = compare_attack_results(grad_result, proj_result, act_result)
    
    print("\n" + "="*50)
    print("STANDARDIZATION COMPLETE!")
    print("="*50)
    print("✅ Consistent analysis functions")
    print("✅ Standardized plotting functions") 
    print("✅ Unified comparison framework")
    print("✅ Complete pipeline integration")
    print("\nAll functions now return consistent structures and create unified visualizations!")