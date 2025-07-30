from attacks.GGL.recover_gradients import *
import torch

def privacy_attack_without_forget_data(original_model, unlearned_model, bdata, device, K=50):
    """

    Args:
        original_model: Original trained model
        unlearned_model: Model after unlearning
        bdata: Available auxillary data 
        device: Device to run the calculations on
        K: number of top components to use
    """
    
    # Step 1: Calculate covariances on available data
    print("Calculating covariances on available data...")
    original_covar, original_counts = calculate_covar(original_model, bdata, device)
    unlearned_covar, unlearned_counts = calculate_covar(unlearned_model, bdata, device)
    
    # Step 2: Calculate covariance differences 
    print("Calculating covariance differences...")
    covar_diff = {}
    for layer in original_covar:
        if layer in unlearned_covar:
            covar_diff[layer] = original_covar[layer] - unlearned_covar[layer]
    
    # Step 3: Get SVD of covariance differences
    print("Computing SVD of covariance differences...")
    diff_svd = calculate_svd(covar_diff, k=K)
    
    # Step 4: Project covariance differences onto their own subspace
    print("Projecting covariance differences...")
    projected_covar = project_onto_subspace(
        covar_diff, diff_svd, k=K, projection_type='full'
    )
    
    # Step 5: Calculate parameter differences and modify unlearned model
    print("Modifying unlearned model parameters...")
    
    # Get parameter differences
    param_diff = {}
    for name, param in original_model.named_parameters():
        if name in dict(unlearned_model.named_parameters()):
            param_diff[name] = param - unlearned_model.state_dict()[name]
    
    # Calculate reconstruction weights 
    reconstruction_weights = {}
    for name, diff in param_diff.items():
        # Extract layer name from parameter name
        layer_name = name.replace('.weight', '').replace('.bias', '')
        
        # Default 
        weight = 0.5
        
        # If we have projection information for this layer
        if layer_name in projected_covar and layer_name in diff_svd:
            # Weight by projection strength
            proj_strength = torch.norm(projected_covar[layer_name], 'fro')
            svd_strength = torch.sum(diff_svd[layer_name]['S'])
            
            if svd_strength > 1e-8:
                weight = min(1.0, (proj_strength / svd_strength).item())
        
        reconstruction_weights[name] = weight
    
    modified_params = 0
    for name, param in unlearned_model.named_parameters():
        if name in param_diff:
            weight = reconstruction_weights[name]
            if weight > 0.01:  
                param.data += weight * param_diff[name]
                modified_params += 1
                print(f"Modified {name}: weight={weight:.4f}")
    
    print(f"Modified {modified_params} parameter tensors")
    
    return projected_covar, param_diff, reconstruction_weights


def projection_attack_set_up(original_model, unlearned_model, bdata, learning_rate, device, loss_fn=nn.CrossEntropyLoss(), variance_threshold=0.9):
    """

    Args:
        original_model: Original trained model
        unlearned_model: Model after unlearning
        bdata: Available auxillary data 
        device: Device to run the calculations on
        learning_rate: Learning rate used in the unlearning process
        variance_threshold: Threshold for variance explained by the subspace
    """
    
    # Step 1: Calculate covariances on available data
    print("Finding approximate subspace on available data...")
    subspace = find_subspace(original_model, unlearned_model, bdata, device, loss_fn=loss_fn, variance_threshold=variance_threshold, cache_dir='projection_cache')
    print("Calculating parameter differences...")
    normalised_param_difference = compute_parameter_difference(original_model, unlearned_model, learning_rate, cache_dir='projection_cache')
    print("Calculating target projection...")
    target_projection = project_onto_subspace(normalised_param_difference, subspace)
    return subspace, target_projection

def projection_attack(x, unlearned_model, subspace, target_projection, device, loss_fn=nn.CrossEntropyLoss()):
    """

    Args:
        x: Input data to project
        subspace: Subspace obtained from the projection attack setup
        target_projection: Target projection of the parameter differences
    """
    print('Calculating gradients of x in unlearned model...')
    x_grads = get_model_gradients(unlearned_model, x, device, loss_fn)
    print("Projecting x on subspace...")
    x_projection = project_onto_subspace(x_grads, subspace)
    return target_projection, x_projection



def test_projection(forgetdata, retaindata, original_model, unlearned_model, bdata, learning_rate, device, loss_fn=nn.CrossEntropyLoss(), variance_threshold=0.9):
    print("Finding approximate subspace on available data...")
    subspace = find_subspace(original_model, unlearned_model, bdata, device, variance_threshold = variance_threshold, cache_dir='test_projection_cache')

    print("Calculating parameter differences...")
    normalised_param_difference = compute_parameter_difference(original_model, unlearned_model, learning_rate, cache_dir='test_projection_cache')

    print("Calculating forget gradients...")
    forget_grads = get_model_gradients(unlearned_model, forgetdata, device, loss_fn, cache_dir='test_projection_cache', label='forget')

    print("Calculating retain gradients...")
    retain_grads = get_model_gradients(unlearned_model, retaindata, device, loss_fn, cache_dir='test_projection_cache', label='retain')

    print("Calculating target projection...")
    target_projection = project_onto_subspace(normalised_param_difference, subspace)

    print("proposition 1: Calculating forget projection... (should be high)")
    forget_projection = project_onto_subspace(forget_grads, subspace)

    print("proposition 2: Calculating forget projection... (should be low)")
    forget_projection_diff = target_projection - forget_projection

    print("proposition 1: Calculating forget projection... (should be low)")
    retain_projection = project_onto_subspace(retain_grads, subspace)

    print("proposition 2: Calculating forget projection... (should be high)")
    retain_projection_diff = target_projection - retain_projection
    


    
    forget_projection_diff = {}
    retain_projection_diff = {}
    
    for layer in target_projection:
        forget_val = forget_projection.get(layer, 0)
        retain_val = retain_projection.get(layer, 0)
        target_val = target_projection[layer]
        
        forget_projection_diff[layer] = target_val - forget_val
        retain_projection_diff[layer] = target_val - retain_val
    
    # Step 7: Comprehensive Analysis
    print("\n" + "="*60)
    print("PROJECTION ANALYSIS RESULTS")
    print("="*60)
    
    # Calculate alignment scores
    target_total = sum(target_projection.values())
    forget_total = sum(forget_projection.values()) 
    retain_total = sum(retain_projection.values())
    
    print(f"\nTotal Projection Magnitudes:")
    print(f"  Target Direction (θ_u - θ_o)/lr: {target_total:.6f}")
    print(f"  Forget Data Gradients:          {forget_total:.6f}")
    print(f"  Retain Data Gradients:          {retain_total:.6f}")
    
    # Relative alignments
    forget_alignment = forget_total / target_total if target_total > 0 else 0
    retain_alignment = retain_total / target_total if target_total > 0 else 0
    
    print(f"\nRelative Alignment with Target Direction:")
    print(f"  Forget Data: {forget_alignment:.4f} ({'HIGH' if forget_alignment > 0.5 else 'LOW'})")
    print(f"  Retain Data: {retain_alignment:.4f} ({'HIGH' if retain_alignment > 0.5 else 'LOW'})")
    
    # Difference analysis (your original approach)
    forget_diff_total = sum(abs(v) for v in forget_projection_diff.values())
    retain_diff_total = sum(abs(v) for v in retain_projection_diff.values())
    
    print(f"\nProjection Differences from Target:")
    print(f"  |Target - Forget|: {forget_diff_total:.6f} ({'LOW' if forget_diff_total < retain_diff_total else 'HIGH'})")
    print(f"  |Target - Retain|: {retain_diff_total:.6f} ({'HIGH' if retain_diff_total > forget_diff_total else 'LOW'})")
    
    # Layer-wise analysis
    print(f"\nLayer-wise Projection Analysis:")
    print(f"{'Layer':<25} {'Target':<12} {'Forget':<12} {'Retain':<12} {'F-Diff':<12} {'R-Diff':<12}")
    print("-" * 95)
    
    for layer in target_projection.keys():
        target_val = target_projection.get(layer, 0)
        forget_val = forget_projection.get(layer, 0) 
        retain_val = retain_projection.get(layer, 0)
        forget_diff = forget_projection_diff.get(layer, 0)
        retain_diff = retain_projection_diff.get(layer, 0)
        
        print(f"{layer:<25} {target_val:<12.6f} {forget_val:<12.6f} {retain_val:<12.6f} "
              f"{forget_diff:<12.6f} {retain_diff:<12.6f}")
    
    # Theoretical validation
    print(f"\n" + "="*60)
    print("THEORETICAL PROPOSITION VALIDATION:")
    print("="*60)
    
    # Proposition 1: Forget data should align better with unlearning direction
    proposition1_satisfied = forget_alignment > retain_alignment
    
    # Proposition 2: Forget data should have smaller difference from target
    proposition2_satisfied = forget_diff_total < retain_diff_total
    
    print(f"Proposition 1 - Forget > Retain alignment: {'✓ PASS' if proposition1_satisfied else '✗ FAIL'}")
    print(f"Proposition 2 - Forget closer to target:   {'✓ PASS' if proposition2_satisfied else '✗ FAIL'}")
    
    # Summary recommendation
    print(f"\nSUMMARY:")
    if proposition1_satisfied and proposition2_satisfied:
        print("   Forget data gradients align well with unlearning direction.")
        print("   Retain data gradients show lower alignment and larger differences.")
    elif proposition1_satisfied or proposition2_satisfied:
        print("   Consider investigating the unlearning process further.")
    else:
        print("   This may indicate issues with the unlearning process.")
    
    # Compile comprehensive results
    results = {
        'subspace_signature': subspace,
        'target_projection': target_projection,
        'forget_projection': forget_projection, 
        'retain_projection': retain_projection,
        'forget_projection_diff': forget_projection_diff,
        'retain_projection_diff': retain_projection_diff,
        'target_total': target_total,
        'forget_total': forget_total,
        'retain_total': retain_total,
        'forget_alignment': forget_alignment,
        'retain_alignment': retain_alignment,
        'forget_diff_total': forget_diff_total,
        'retain_diff_total': retain_diff_total,
        'proposition1_satisfied': proposition1_satisfied,
        'proposition2_satisfied': proposition2_satisfied,
        'layer_analysis': {
            layer: {
                'target': target_projection.get(layer, 0),
                'forget': forget_projection.get(layer, 0),
                'retain': retain_projection.get(layer, 0),
                'forget_diff': forget_projection_diff.get(layer, 0),
                'retain_diff': retain_projection_diff.get(layer, 0)
            } for layer in target_projection.keys()
        }
    }
    
    # Cache results
    cache_path = os.path.join('test_projection_cache', 'projection_test_results.pt')
    torch.save(results, cache_path)
    print(f"\n📁 Results cached to: {cache_path}")
    
    return results