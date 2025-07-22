from attacks.GGL.recover_gradients import (
    calculate_covar,
    calculate_svd,
    calculate_svd_difference,
    project_onto_subspace)
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

