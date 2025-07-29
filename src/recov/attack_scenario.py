import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from analysis import (
    analyze_realistic_results_standardized, 
    create_standardized_projection_plots
)

def run_realistic_projection_attack(original_model, unlearned_model, background_data,
                                   forget_data, retain_data, device='cuda', 
                                   variance_threshold=0.85, verbose=True):
    """
    Realistic projection attack: Estimate forget subspace from background data only,
    then test how well it predicts forget vs retain data.
    
    This simulates a real attack where you don't have access to the actual forget data.
    """
    
    print("REALISTIC PROJECTION ATTACK")
    print("="*60)
    print("Phase 1: Estimate forget subspace from background data only")
    print("Phase 2: Test prediction on real forget/retain data")
    print("="*60)
    
    # Phase 1: Estimate forget subspace from background data
    print("\n🔍 PHASE 1: ESTIMATING FORGET SUBSPACE FROM BACKGROUND DATA")
    print("-" * 50)
    
    estimated_forget_subspace = estimate_forget_subspace_from_background(
        original_model, unlearned_model, background_data, device, verbose
    )
    
    if not estimated_forget_subspace:
        return {
            'success': False,
            'description': 'Failed to estimate forget subspace from background data'
        }
    
    # Phase 2: Test the estimated subspace on real data
    print("\n🎯 PHASE 2: TESTING ESTIMATED SUBSPACE ON REAL DATA")
    print("-" * 50)
    
    results = test_estimated_subspace(
        estimated_forget_subspace, original_model, 
        forget_data, retain_data, device, verbose
    )

    # Phase 3: Create visualizations using STANDARDIZED function
    plot_dir = Path("./plots/realistic_projection")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Use STANDARDIZED analysis function
    standardized_summary = analyze_realistic_results_standardized(results, verbose)
    
    create_standardized_projection_plots(
        results, standardized_summary, plot_dir, verbose
    )
    
    return {
        'estimated_forget_subspace': estimated_forget_subspace,
        'test_results': results,
        'summary': standardized_summary  
    }
    
    # # Phase 3: Create visualizations  
    # plot_dir = Path("./plots/realistic_projection")
    # plot_dir.mkdir(parents=True, exist_ok=True)
    
    # create_realistic_projection_plots(
    #     estimated_forget_subspace, results, plot_dir, verbose
    # )
    
    # return {
    #     'estimated_forget_subspace': estimated_forget_subspace,
    #     'test_results': results,
    #     'summary': analyze_realistic_results(results, verbose)
    # }


def estimate_forget_subspace_from_background(original_model, unlearned_model, 
                                           background_data, device, verbose):
    """
    Estimate the forget subspace using only background data.
    Key insight: Model differences on background data reveal unlearning patterns.
    """
    
    if verbose:
        print("Computing gradient differences on background data...")
    
    # Get gradients from both models on background data
    orig_grads = get_gradients_efficient(original_model, background_data, device)
    unl_grads = get_gradients_efficient(unlearned_model, background_data, device)
    
    if not orig_grads or not unl_grads:
        if verbose:
            print("❌ Failed to compute gradients")
        return None
    
    # Calculate gradient differences
    grad_differences = {}
    total_diff_norm = 0
    
    for layer_name in orig_grads:
        if layer_name in unl_grads and orig_grads[layer_name] is not None:
            diff = orig_grads[layer_name] - unl_grads[layer_name]
            grad_differences[layer_name] = diff
            total_diff_norm += torch.norm(diff).item()
    
    if verbose:
        print(f"Total gradient difference norm: {total_diff_norm:.6f}")
        
    if total_diff_norm < 1e-6:
        if verbose:
            print("⚠️  WARNING: Models show minimal differences on background data")
        return None
    
    # Extract subspace from gradient differences using SVD
    estimated_subspace = {}
    
    for layer_name, grad_diff in grad_differences.items():
        try:
            # Move to CPU and flatten
            grad_flat = grad_diff.detach().cpu().flatten().float()
            
            if torch.norm(grad_flat) < 1e-8:
                continue
                
            # Normalize gradient
            grad_normalized = grad_flat / torch.norm(grad_flat)
            
            # For subspace, we want the direction of change
            # Create a 1D subspace (the principal direction of unlearning)
            subspace_basis = grad_normalized.unsqueeze(1)  # [num_params, 1]
            
            estimated_subspace[layer_name] = {
                'basis': subspace_basis,
                'magnitude': torch.norm(grad_flat).item(),
                'layer_name': layer_name
            }
            
            if verbose:
                print(f"Layer {layer_name}: estimated subspace dimension 1, magnitude {torch.norm(grad_flat).item():.6f}")
                
        except Exception as e:
            if verbose:
                print(f"Failed to extract subspace for {layer_name}: {e}")
            continue
    
    return estimated_subspace


def test_estimated_subspace(estimated_subspace, original_model, forget_data, 
                           retain_data, device, verbose):
    """
    Test how well the estimated subspace predicts forget vs retain data.
    """
    
    if verbose:
        print("Testing estimated subspace on forget data...")
    
    # Get gradients of original model on forget data
    forget_grads = get_gradients_efficient(original_model, forget_data, device)
    
    if verbose:
        print("Testing estimated subspace on retain data...")
    
    # Get gradients of original model on retain data  
    retain_grads = get_gradients_efficient(original_model, retain_data, device)
    
    if not forget_grads or not retain_grads:
        return {'success': False, 'description': 'Failed to compute test gradients'}
    
    # Project forget and retain gradients onto estimated subspace
    forget_projections = {}
    retain_projections = {}
    
    for layer_name in estimated_subspace:
        if layer_name not in forget_grads or layer_name not in retain_grads:
            continue
            
        try:
            subspace_basis = estimated_subspace[layer_name]['basis']  # [num_params, 1]
            
            # Project forget gradients
            forget_grad_flat = forget_grads[layer_name].detach().cpu().flatten().float()
            forget_projection = torch.abs(torch.dot(forget_grad_flat, subspace_basis.squeeze())).item()
            
            # Project retain gradients
            retain_grad_flat = retain_grads[layer_name].detach().cpu().flatten().float()
            retain_projection = torch.abs(torch.dot(retain_grad_flat, subspace_basis.squeeze())).item()
            
            forget_projections[layer_name] = forget_projection
            retain_projections[layer_name] = retain_projection
            
            if verbose:
                ratio = forget_projection / (retain_projection + 1e-8)
                print(f"Layer {layer_name}: forget_proj={forget_projection:.6f}, retain_proj={retain_projection:.6f}, ratio={ratio:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"Projection failed for {layer_name}: {e}")
            continue
    
    return {
        'forget_projections': forget_projections,
        'retain_projections': retain_projections,
        'success': True
    }


def get_gradients_efficient(model, data_loader, device, max_batches=3):
    """
    Get gradients efficiently with memory management.
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
                
            images, labels = images.to(device), labels.to(device)
            
            model.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # Accumulate gradients for weight parameters only
            for name, param in model.named_parameters():
                if param.grad is not None and '.weight' in name:
                    layer_name = name.replace('.weight', '')
                    if layer_name not in accumulated_gradients:
                        accumulated_gradients[layer_name] = param.grad.detach().clone()
                    else:
                        accumulated_gradients[layer_name] += param.grad.detach().clone()
            
            num_batches += 1
            
            # Clean up
            del images, labels, outputs, loss
            torch.cuda.empty_cache()
    
    finally:
        model.train(original_mode)
    
    # Average gradients
    for layer_name in accumulated_gradients:
        accumulated_gradients[layer_name] = accumulated_gradients[layer_name] / num_batches
    
    return accumulated_gradients


def analyze_realistic_results(results, verbose=True):
    """
    Analyze results of realistic projection attack.
    """
    
    if not results.get('success', False):
        return {'success': False, 'description': results.get('description', 'Unknown error')}
    
    forget_projs = list(results['forget_projections'].values())
    retain_projs = list(results['retain_projections'].values())
    
    if not forget_projs or not retain_projs:
        return {'success': False, 'description': 'No projection results to analyze'}
    
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
    
    print(f"\nREALISTIC PROJECTION ATTACK ANALYSIS")
    print("-" * 40)
    print(f"Mean forget projection strength: {mean_forget_proj:.6f}")
    print(f"Mean retain projection strength: {mean_retain_proj:.6f}")
    print(f"Mean forget/retain ratio: {mean_ratio:.3f}")
    
    # Success criteria: forget should project more strongly than retain
    success_threshold = 2.0  # Forget should be at least 2x stronger
    
    attack_success = mean_ratio > success_threshold
    
    if attack_success:
        status = "✅ REALISTIC ATTACK SUCCESSFUL"
        description = f"Estimated subspace predicts forget data well (ratio: {mean_ratio:.3f})"
    elif mean_ratio > 1.5:
        status = "⚠️  PARTIALLY SUCCESSFUL"
        description = f"Some predictive power but weak (ratio: {mean_ratio:.3f})"
    else:
        status = "❌ REALISTIC ATTACK FAILED"
        description = f"No predictive power (ratio: {mean_ratio:.3f})"
    
    print(f"\n{status}")
    print(f"Description: {description}")
    
    if attack_success:
        print("💡 This means: Background data alone can reveal forget patterns!")
        print("💡 Implication: Unlearning may not be as private as expected")
    else:
        print("✅ This means: Background data cannot reveal forget patterns")
        print("✅ Implication: Unlearning provides good privacy protection")
    
    return {
        'success': attack_success,
        'status': status,
        'description': description,
        'forget_projection': mean_forget_proj,
        'retain_projection': mean_retain_proj,
        'prediction_ratio': mean_ratio,
        'privacy_risk': attack_success
    }


def create_realistic_projection_plots(estimated_subspace, results, plot_dir, verbose=True):
    """
    Create plots for realistic projection attack results.
    """
    
    if verbose:
        print(f"Creating realistic projection plots in {plot_dir}...")
    
    if not results.get('success', False):
        return
    
    # Plot 1: Projection strengths by layer
    layers = list(results['forget_projections'].keys())
    forget_projs = [results['forget_projections'][l] for l in layers]
    retain_projs = [results['retain_projections'][l] for l in layers]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    plt.bar(x - width/2, forget_projs, width, label='Forget Data Projection', color='red', alpha=0.7)
    plt.bar(x + width/2, retain_projs, width, label='Retain Data Projection', color='blue', alpha=0.7)
    
    plt.xlabel('Layer')
    plt.ylabel('Projection Strength')
    plt.title('Realistic Attack: Estimated Subspace Projection Strengths')
    plt.xticks(x, [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_dir / 'projection_strengths.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Prediction ratios
    ratios = [forget_projs[i] / (retain_projs[i] + 1e-8) for i in range(len(layers))]
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(layers)), ratios, color='green', alpha=0.7)
    
    # Add success threshold line
    plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Success threshold (2.0)')
    plt.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Break-even (1.0)')
    
    plt.xlabel('Layer')
    plt.ylabel('Forget/Retain Projection Ratio')
    plt.title('Realistic Attack: Prediction Performance by Layer\n(Higher = Better Prediction)')
    plt.xticks(range(len(layers)), [l.replace('layer', 'L') for l in layers], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Color bars based on success
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        if ratio > 2.0:
            bar.set_color('green')
        elif ratio > 1.0:
            bar.set_color('orange')
        else:
            bar.set_color('red')
        
        # Add value labels
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{ratio:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'prediction_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Overall attack summary
    mean_ratio = np.mean(ratios)
    
    plt.figure(figsize=(8, 6))
    
    if mean_ratio > 2.0:
        color = 'red'
        label = 'HIGH RISK'
        message = 'Background data reveals\nforgotten patterns!'
    elif mean_ratio > 1.0:
        color = 'orange'
        label = 'MEDIUM RISK'
        message = 'Some pattern leakage\ndetected'
    else:
        color = 'green'
        label = 'LOW RISK'
        message = 'Good privacy\nprotection'
    
    plt.bar([label], [mean_ratio], color=color, alpha=0.7)
    plt.ylabel('Mean Prediction Ratio')
    plt.title(f'Realistic Projection Attack: Privacy Risk Assessment\nRatio: {mean_ratio:.3f}')
    
    plt.text(0, mean_ratio/2, message, ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'privacy_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Realistic projection plots saved to {plot_dir}")