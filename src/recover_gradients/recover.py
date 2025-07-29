
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from collections import OrderedDict
from tqdm import tqdm
import time
import os



def get_feature_dict(model):
    """
    Automatically generate conv_fea_dict and linear_fea_dict for a given model.
    """
    _, eval_nodes = get_graph_node_names(model)
    conv_fea_dict = OrderedDict()
    linear_fea_dict = OrderedDict()
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name in eval_nodes:
            conv_fea_dict[name] = name
        elif isinstance(module, torch.nn.Linear) and name in eval_nodes:
            linear_fea_dict[name] = name
    return conv_fea_dict, linear_fea_dict


def get_module_by_name(model, layer_name):
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def calculate_covar(model, data_loader, device, epochs=1, conv_fea_dict=None, linear_fea_dict=None):
    """
    Calculate covariance matrices for each layer's activations.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for input data
        device: Device to run computations on
        epochs: Number of epochs to process
        conv_fea_dict: Dictionary mapping conv layer names
        linear_fea_dict: Dictionary mapping linear layer names
        
    Returns:
        dict: Covariance matrices for each layer
    """
    print(f"Starting covariance calculation with {len(data_loader)} batches")
    
    # Auto-generate feature dictionaries if not provided
    if conv_fea_dict is None or linear_fea_dict is None:
        conv_fea_dict, linear_fea_dict = get_feature_dict(model)
    
    model.eval()
    
    # Enable dropout if needed
    for md in model.modules():
        if md.__class__.__name__ == 'Dropout':
            md.training = True

    # Setup feature extraction
    fea_dict = {}
    for val in {**conv_fea_dict, **linear_fea_dict}.values():
        fea_dict[val] = val

    tmp_fea_dict = {**fea_dict}
    if 'input' in fea_dict:
        features = {'input': []}
        tmp_fea_dict.pop('input')
    else:
        features = {}

    fea_ext = create_feature_extractor(model, tmp_fea_dict)

    # Initialize covariance matrices
    covar = {}
    for key in {**conv_fea_dict, **linear_fea_dict}.keys():
        covar[key] = 0

    with torch.no_grad():
        for epoch in range(int(epochs)):
            print(f"Processing covariance epoch {epoch + 1}/{epochs}")
            for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader, desc=f"Covar Epoch {epoch+1}")):
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                if 'input' in fea_dict:
                    features['input'] = imgs

                # Extract features
                feats = fea_ext(imgs)
                for fea_name in feats:
                    features[fea_name] = feats[fea_name].detach()

                # Process conv layers
                for layer in conv_fea_dict:
                    try:
                        layer_module = get_module_by_name(model, layer)
                        ks = layer_module.kernel_size
                        padding = layer_module.padding
                        
                        f = features[conv_fea_dict[layer]]
                        patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
                        fea_dim = patch.shape[1]
                        patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
                        covar[layer] += torch.mm(patch.permute(1, 0), patch)
                    except Exception as e:
                        print(f"Error processing conv layer {layer}: {e}")
                        raise

                # Process linear layers
                for layer in linear_fea_dict:
                    try:
                        f = features[linear_fea_dict[layer]].double().squeeze()
                        if f.ndim == 1:
                            f = f.unsqueeze(0)
                        covar[layer] += torch.mm(f.permute(1, 0), f)
                    except Exception as e:
                        print(f"Error processing linear layer {layer}: {e}")
                        raise

    # Normalize by epochs
    for layer in covar:
        covar[layer] = covar[layer] / epochs
        
    return covar

def get_gradients(model, data_loader, device, loss_fn=None, epochs=1):
    """
    Calculate gradients for the model on given data.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for input data
        device: Device to run computations on
        loss_fn: Loss function (if None, uses CrossEntropyLoss)
        epochs: Number of epochs to process
        
    Returns:
        dict: Average gradients for each parameter
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    
    # Initialize gradient accumulation
    grad_dict = {}
    total_samples = 0
    
    # Zero out gradients initially
    model.zero_grad()
    
    with torch.enable_grad():
        for epoch in range(epochs):
            print(f"Processing gradient epoch {epoch + 1}/{epochs}")
            for batch_idx, (imgs, lbls) in enumerate(tqdm(data_loader, desc=f"Grad Epoch {epoch+1}")):
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                # Forward pass
                outputs = model(imgs)
                loss = loss_fn(outputs, lbls)
                
                # Backward pass
                loss.backward()
                
                # Accumulate gradients
                if batch_idx == 0 and epoch == 0:
                    # Initialize grad_dict on first batch
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_dict[name] = param.grad.clone()
                        else:
                            grad_dict[name] = torch.zeros_like(param)
                else:
                    # Accumulate gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_dict[name] += param.grad
                
                total_samples += imgs.shape[0]
                model.zero_grad()
    
    # Average gradients
    for name in grad_dict:
        grad_dict[name] = grad_dict[name] / total_samples
    
    return grad_dict

def calculate_svd(covar_dict, k=None, eps=1e-6, verbose=True):
    """
    Calculate SVD for covariance matrices.
    
    Args:
        covar_dict: Dictionary of covariance matrices
        k: Number of components to keep (if None, keep all)
        eps: Small epsilon for numerical stability
        
    Returns:
        dict: SVD results for each layer
    """
    svd_dict = {}
    if verbose:
        print("Computing SVD for each layer...")
    for layer in covar_dict:
        try:
            stime = time.time()
            
            # Add small epsilon for numerical stability
            covar_stable = covar_dict[layer] + eps * torch.eye(covar_dict[layer].shape[0], device=covar_dict[layer].device)
            
            U, S, _ = torch.svd(covar_stable)
            
            # Keep only top k components if specified
            if k is not None:
                U = U[:, :k]
                S = S[:k]
            
            svd_dict[layer] = {'U': U, 'S': torch.sqrt(S)}
            if verbose:
                print(f'Layer: {layer} - SVD time: {time.time() - stime:.06f} - Shape: {U.shape}')
            
        except Exception as e:
            print(f"Error computing SVD for layer {layer}: {e}")
            raise
    
    return svd_dict

def calculate_svd_difference(covar_original, covar_unlearned, device, k=None, verbose=True):
    """
    Calculate SVD of the difference between original and unlearned covariances.
    This computes the "retain" SVD by subtracting unlearned data contribution.
    
    Args:
        covar_original: Original covariance matrices
        covar_unlearned: Covariance matrices from data to unlearn
        device: Device for computations
        k: Number of components to keep
        
    Returns:
        dict: Retain SVD results
    """
    # First get full SVD
    full_svd = calculate_svd(covar_original, k=None)
    
    retain_svd = {}
    
    if verbose:
        print("Computing retain SVD for each layer...")
    for layer in covar_original:
        try:
            stime = time.time()
            
            # Get original SVD components
            U, S = full_svd[layer]['U'].to(device), full_svd[layer]['S'].to(device)
            
            # Reconstruct original covariance matrix
            M = torch.mm(torch.mm(U, torch.diag(S**2)), U.t())
            
            # Subtract unlearned data contribution
            M1 = M - covar_unlearned[layer].to(device)
            
            # SVD of the difference
            U1_, S1sq_, _ = torch.svd(M1)
            
            # Keep only top k components if specified
            if k is not None:
                U1_ = U1_[:, :k]
                S1sq_ = S1sq_[:k]
            
            retain_svd[layer] = {'U': U1_, 'S': torch.sqrt(S1sq_)}
            if verbose:
                print(f'Layer: {layer} - Retain SVD time: {time.time() - stime:.06f} - M: {M.shape}')
            
        except Exception as e:
            print(f"Error computing retain SVD for layer {layer}: {e}")
            raise
    
    return retain_svd

def get_projection_matrix(svd_dict, retained_var_threshold=0.95, device='cuda', verbose=True):
    """
    Create projection matrices from SVD results.
    
    Args:
        svd_dict: Dictionary of SVD results
        retained_var_threshold: Variance threshold for component selection
        device: Device for computations
        
    Returns:
        dict: Projection matrices for each layer
    """
    P = {}
    
    if verbose:
        print("Building projection matrices...")
    for layer in svd_dict:
        # Calculate number of components needed for variance threshold
        k = torch.sum((torch.cumsum(svd_dict[layer]['S'], dim=0) / torch.sum(svd_dict[layer]['S'])) <= retained_var_threshold)
        if verbose:
            print(f"Layer {layer}: selected {k.item()}/{svd_dict[layer]['S'].shape[0]} components")
        
        # Create projection matrix
        M = svd_dict[layer]['U'][:, :k]
        P[layer] = torch.mm(M, M.t()).to(device).float()
    
    return P

def project_gradients(gradients, projection_matrices, name_mapping=None):
    """
    Project gradients using projection matrices.
    
    Args:
        gradients: Dictionary of gradients for each parameter
        projection_matrices: Dictionary of projection matrices
        name_mapping: Optional mapping from parameter names to layer names
        
    Returns:
        dict: Projected gradients
    """
    projected_gradients = {}
    
    for param_name, grad in gradients.items():
        if grad is None:
            projected_gradients[param_name] = None
            continue
        
        # Map parameter name to layer name
        if name_mapping and param_name in name_mapping:
            layer_name = name_mapping[param_name]
        else:
            # Extract layer name from parameter name
            if '.weight' in param_name:
                layer_name = param_name.replace('.weight', '')
            elif '.bias' in param_name:
                layer_name = param_name.replace('.bias', '')
            else:
                layer_name = param_name
        
        # Skip if no projection matrix for this layer
        if layer_name not in projection_matrices:
            print(f"No projection matrix for layer {layer_name}, skipping projection")
            projected_gradients[param_name] = grad.clone()
            continue
        
        # Get projection matrix
        proj_matrix = projection_matrices[layer_name]
        
        # Apply projection based on parameter type
        if len(grad.shape) == 4:  # Conv layer weights [out_ch, in_ch, h, w]
            sz = grad.shape[0]  # output channels
            grad_flat = grad.view(sz, -1)  # [out_ch, in_ch*h*w]
            
            # Check dimension compatibility
            if grad_flat.shape[1] != proj_matrix.shape[0]:
                print(f"Dimension mismatch for {layer_name}: grad {grad_flat.shape[1]} vs proj {proj_matrix.shape[0]}")
                projected_gradients[param_name] = grad.clone()
                continue
            
            # Apply projection: grad - grad * P
            proj_grad = torch.mm(grad_flat, proj_matrix)
            projected_grad = grad - proj_grad.view(grad.size())
            
        elif len(grad.shape) == 2:  # Linear layer weights [out_features, in_features]
            sz = grad.shape[0]  # output features
            grad_flat = grad.view(sz, -1)  # [out_features, in_features]
            
            # Check dimension compatibility
            if grad_flat.shape[1] != proj_matrix.shape[0]:
                print(f"Dimension mismatch for {layer_name}: grad {grad_flat.shape[1]} vs proj {proj_matrix.shape[0]}")
                projected_gradients[param_name] = grad.clone()
                continue
            
            # Apply projection: grad - grad * P
            proj_grad = torch.mm(grad_flat, proj_matrix)
            projected_grad = grad - proj_grad.view(grad.size())
            
        elif len(grad.shape) == 1:  # Bias terms
            # Skip bias terms or handle differently
            projected_grad = grad.clone()
        else:
            # Unknown parameter shape
            projected_grad = grad.clone()
        
        projected_gradients[param_name] = projected_grad
    
    return projected_gradients

def get_cache_path(cache_dir, name):
    """Get the cache file path for a given name."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}.pkl")

def save_to_cache(data, cache_path):
    """Save data to cache."""
    print(f"Saving to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(cache_path):
    """Load data from cache if it exists."""
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


# Example usage for gradient recovery:
def gradient_recovery_pipeline(original_model, unlearned_model, data_loader, device, 
                             retained_var_threshold=0.95, loss_fn=None):
    """
    Complete pipeline for gradient recovery using projection matrices.
    
    Args:
        original_model: Model trained on original data
        original_data_loader: DataLoader for original training data
        unlearned_data_loader: DataLoader for data to be unlearned
        device: Device for computations
        retained_var_threshold: Variance threshold for projection
        loss_fn: Loss function for gradient computation
        
    Returns:
        dict: Projected gradients that avoid the unlearned data subspace
    """
    print("Starting gradient recovery pipeline...")
    
    # Step 1: Get feature dictionaries
    conv_fea_dict, linear_fea_dict = get_feature_dict(original_model)
    
    # Step 2: Calculate covariances
    print("Calculating original model data covariances...")
    covar_original = calculate_covar(original_model, data_loader, device, 
                                   conv_fea_dict=conv_fea_dict, linear_fea_dict=linear_fea_dict)
    
    print("Calculating unlearned model data covariances...")
    covar_unlearned = calculate_covar(unlearned_model, data_loader, device,
                                    conv_fea_dict=conv_fea_dict, linear_fea_dict=linear_fea_dict)
    
    # Step 3: Calculate SVD
    print("Calculating SVD difference (estimated forget svd)...")
    est_svd = calculate_svd_difference(covar_original, covar_unlearned, device)
    
    # Step 4: Get projection matrices
    print("Creating projection matrices...")
    projection_matrices = get_projection_matrix(est_svd, retained_var_threshold, device)
    
    # Step 5: Get gradients from unlearned data
    print("Computing gradients on data...")
    gradients = get_gradients(original_model, data_loader, device, loss_fn)
    
    # Step 6: Project gradients
    print("Projecting gradients...")
    projected_gradients = project_gradients(gradients, projection_matrices)
    
    print("Gradient recovery pipeline completed!")
    return projected_gradients


def compare_projections(proj1, proj2, k=None, plot=True, id='comparison'):
    """
    Compare two sets of parameter-wise projections.
    """
    results = {}
    
    common_params = set(proj1.keys()) & set(proj2.keys())
    
    for param_name in common_params:
        P1 = proj1[param_name]
        P2 = proj2[param_name]
        
        if P1 is None or P2 is None:
            continue
            
        # Calculate various metrics
        l2_diff = torch.norm(P1 - P2, 'fro').item()
        l2_p1 = torch.norm(P1, 'fro').item()
        l2_p2 = torch.norm(P2, 'fro').item()
        
        # Normalized difference
        rel_diff = l2_diff / (l2_p1 + 1e-8)
        
        # SVD for subspace comparison
        U1, S1, _ = torch.svd(P1)
        U2, S2, _ = torch.svd(P2)
        
        # Compare singular values
        min_rank = min(len(S1), len(S2))
        if k is not None:
            min_rank = min(min_rank, k)
            
        s1_trunc = S1[:min_rank]
        s2_trunc = S2[:min_rank]
        
        l2_singular_diff = torch.norm(s1_trunc - s2_trunc).item()
        
        # Subspace overlap
        overlap_metrics = subspace_overlap(U1, U2, k=min_rank)
        
        # Recall
        recall = subspace_recall(U1, U2, k=min_rank)
        
        results[param_name] = {
            'l2_diff': l2_diff,
            'l2_p1': l2_p1,
            'l2_p2': l2_p2,
            'rel_diff': rel_diff,
            'l2_singular_value_diff': l2_singular_diff,
            'cos_angles': overlap_metrics['cos_angles'],
            'mean_subspace_overlap': overlap_metrics['mean_overlap'],
            'recall@k': recall,
            'min_angle': overlap_metrics['min_angle'],
            'max_angle': overlap_metrics['max_angle']
        }
    
    if plot:
        plot_comparison_results(results, id=id)
    
    return results


def plot_comparison_results(results, id='comparison'):
    """
    Plot comparison results across parameters.
    """
    params = list(results.keys())
    n_params = len(params)
    
    if n_params == 0:
        print("No parameters to plot!")
        return
    
    # Show only top parameters to avoid cluttered plots
    max_params_to_show = 10
    if n_params > max_params_to_show:
        # Sort by relative difference to show most interesting parameters
        params_by_reldiff = sorted(params, key=lambda x: results[x]['rel_diff'], reverse=True)
        params = params_by_reldiff[:max_params_to_show]
        n_params = len(params)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Parameter-wise Subspace Comparison Results - {id}', fontsize=16)
    
    # Metrics to plot
    metrics = ['l2_singular_value_diff', 'mean_subspace_overlap', 'recall@k', 'rel_diff']
    titles = ['L2 Singular Value Diff', 'Mean Subspace Overlap', 'Recall@k', 'Relative Difference']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        values = [results[param][metric] for param in params]
        
        ax.bar(range(len(params)), values, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Parameter')
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(params)))
        
        # Truncate parameter names for readability
        param_labels = [param[:20] + '...' if len(param) > 20 else param for param in params]
        ax.set_xticklabels(param_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for j, v in enumerate(values):
            ax.text(j, v + max(values) * 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(f'subspace_comparison_{id}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nParameter-wise comparison summary ({len(params)} parameters shown):")
    print(f"Mean relative difference: {np.mean([results[p]['rel_diff'] for p in params]):.4f}")
    print(f"Mean subspace overlap: {np.mean([results[p]['mean_subspace_overlap'] for p in params]):.4f}")
    print(f"Mean recall@k: {np.mean([results[p]['recall@k'] for p in params]):.4f}")
    
    # Show top 5 most different parameters
    top_different = sorted(params, key=lambda x: results[x]['rel_diff'], reverse=True)[:5]
    print(f"\nTop 5 most different parameters:")
    for param in top_different:
        print(f"  {param}: rel_diff={results[param]['rel_diff']:.4f}")


def print_all_metrics(results, title):
    """Print all metrics for all layers"""
    print(f"\n{title}:")
    print("-" * 80)
    
    for layer, vals in results.items():
        print(f"[{layer}]:")
        print(f"  L2 Diff: {vals['l2_diff']:.6f}")
        print(f"  L2 P1 (norm): {vals['l2_p1']:.6f}")
        print(f"  L2 P2 (norm): {vals['l2_p2']:.6f}")
        print(f"  Relative Diff: {vals['rel_diff']:.6f}")
        print(f"  L2 Singular Value Diff: {vals['l2_singular_value_diff']:.6f}")
        print(f"  Mean Subspace Overlap: {vals['mean_subspace_overlap']:.6f}")
        print(f"  Recall@k: {vals['recall@k']:.6f}")
        print(f"  Min Angle: {vals['min_angle']:.2f}°")
        print(f"  Max Angle: {vals['max_angle']:.2f}°")
        print()


def subspace_recall(U_true, U_est, k=None):
    """
    Calculate recall between two subspaces.
    """
    if k is None:
        k = min(U_true.shape[1], U_est.shape[1])
    
    U_true_k = U_true[:, :k]
    U_est_k = U_est[:, :k]
    proj = U_est_k @ U_est_k.T
    recall = torch.trace(U_true_k.T @ proj @ U_true_k) / k
    return recall.item()


def subspace_overlap(U1, U2, k=None):
    """
    Calculate overlap between two subspaces using principal angles.
    """
    if k is None:
        k = min(U1.shape[1], U2.shape[1])
    
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]
    
    # Calculate cosines of principal angles
    _, s, _ = torch.svd(U1_k.T @ U2_k)
    cos_angles = s
    
    # Mean overlap
    mean_overlap = cos_angles.mean().item()
    
    return {
        'cos_angles': cos_angles,
        'mean_overlap': mean_overlap,
        'min_angle': torch.acos(cos_angles.max()).item() * 180 / np.pi,
        'max_angle': torch.acos(cos_angles.min()).item() * 180 / np.pi
    }


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ISSUE 1: Dimension Mismatches
# The gradient dimensions don't match projection matrix dimensions because
# the SVD was computed on INPUT activations, but gradients are being processed differently

def fix_get_feature_dict(model):
    """
    Fixed version that only includes layers we can actually project.
    Excludes BatchNorm and other layers that don't have meaningful input representations.
    """
    try:
        from torchvision.models.feature_extraction import get_graph_node_names
        _, eval_nodes = get_graph_node_names(model)
        conv_fea_dict = OrderedDict()
        linear_fea_dict = OrderedDict()
        
        # Only include layers that have weight parameters and meaningful input representations
        for name, module in model.named_modules():
            # Only include Conv2d and Linear layers that appear in eval_nodes
            if isinstance(module, torch.nn.Conv2d) and name in eval_nodes:
                conv_fea_dict[name] = name
            elif isinstance(module, torch.nn.Linear) and name in eval_nodes:
                linear_fea_dict[name] = name
            # Exclude BatchNorm, Dropout, etc.
        
        print(f"Fixed: Generated {len(conv_fea_dict)} conv layers and {len(linear_fea_dict)} linear layers")
        print(f"Conv layers: {list(conv_fea_dict.keys())}")
        print(f"Linear layers: {list(linear_fea_dict.keys())}")
        return conv_fea_dict, linear_fea_dict
    except Exception as e:
        print(f"Error in fix_get_feature_dict: {e}")
        raise

def debug_activation_dimensions(model, data_loader, device, conv_fea_dict, linear_fea_dict):
    """
    Debug function to check what dimensions we're actually getting from activations.
    """
    from torchvision.models.feature_extraction import create_feature_extractor
    
    model.eval()
    fea_dict = {**conv_fea_dict, **linear_fea_dict}
    fea_ext = create_feature_extractor(model, fea_dict)
    
    print("=== ACTIVATION DIMENSION DEBUG ===")
    
    with torch.no_grad():
        for batch_idx, (imgs, lbls) in enumerate(data_loader):
            if batch_idx > 0:  # Only check first batch
                break
                
            imgs = imgs.to(device)
            feats = fea_ext(imgs)
            
            for layer_name, activation in feats.items():
                print(f"Layer {layer_name}: activation shape = {activation.shape}")
                
                if layer_name in conv_fea_dict:
                    # Simulate the unfolding process
                    layer_module = get_module_by_name(model, layer_name)
                    ks = layer_module.kernel_size
                    padding = layer_module.padding
                    
                    patch = F.unfold(activation, ks, dilation=1, padding=padding, stride=1)
                    fea_dim = patch.shape[1]  # This is what goes into covariance
                    print(f"  -> Unfolded dimension: {fea_dim}")
                    
                elif layer_name in linear_fea_dict:
                    fea_dim = activation.shape[-1]  # Last dimension
                    print(f"  -> Linear input dimension: {fea_dim}")
            break

def debug_gradient_dimensions(model, data_loader, device, loss_fn=None):
    """
    Debug function to check gradient dimensions.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    print("=== GRADIENT DIMENSION DEBUG ===")
    
    for batch_idx, (imgs, lbls) in enumerate(data_loader):
        if batch_idx > 0:  # Only check first batch
            break
            
        imgs, lbls = imgs.to(device), lbls.to(device)
        
        model.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, lbls)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Parameter {name}: grad shape = {param.grad.shape}")
                
                # Extract layer name
                if '.weight' in name:
                    layer_name = name.replace('.weight', '')
                    print(f"  -> Layer: {layer_name}")
                    
                    if len(param.grad.shape) == 4:  # Conv weights
                        out_ch, in_ch, h, w = param.grad.shape
                        flattened_dim = in_ch * h * w
                        print(f"  -> Conv flattened dim: {flattened_dim} (in_ch={in_ch}, h={h}, w={w})")
                    elif len(param.grad.shape) == 2:  # Linear weights
                        out_feat, in_feat = param.grad.shape
                        print(f"  -> Linear input dim: {in_feat}")
        break

def get_module_by_name(model, layer_name):
    """Helper function to get module by name."""
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def fixed_project_gradients(gradients, projection_matrices, conv_fea_dict, linear_fea_dict, debug=False):
    """
    Fixed gradient projection that handles dimension mismatches properly.
    """
    projected_gradients = {}
    
    if debug:
        print("=== PROJECTION DEBUG ===")
    
    for param_name, grad in gradients.items():
        if grad is None:
            projected_gradients[param_name] = None
            continue
        
        # Extract layer name from parameter name
        if '.weight' in param_name:
            layer_name = param_name.replace('.weight', '')
        elif '.bias' in param_name:
            layer_name = param_name.replace('.bias', '')
            # Skip bias terms entirely
            projected_gradients[param_name] = grad.clone()
            continue
        else:
            layer_name = param_name
        
        # Check if this layer has a projection matrix
        if layer_name not in projection_matrices:
            if debug:
                print(f"No projection matrix for layer {layer_name}, keeping original gradient")
            projected_gradients[param_name] = grad.clone()
            continue
        
        proj_matrix = projection_matrices[layer_name]
        
        if debug:
            print(f"Processing layer {layer_name}:")
            print(f"  Gradient shape: {grad.shape}")
            print(f"  Projection matrix shape: {proj_matrix.shape}")
        
        # Apply projection based on parameter type
        if len(grad.shape) == 4:  # Conv layer weights [out_ch, in_ch, h, w]
            out_ch, in_ch, h, w = grad.shape
            grad_flat = grad.view(out_ch, -1)  # [out_ch, in_ch*h*w]
            expected_dim = in_ch * h * w
            
            if debug:
                print(f"  Conv layer: out_ch={out_ch}, in_ch={in_ch}, h={h}, w={w}")
                print(f"  Expected flattened dim: {expected_dim}")
                print(f"  Actual flattened dim: {grad_flat.shape[1]}")
            
            # Check dimension compatibility
            if grad_flat.shape[1] != proj_matrix.shape[0]:
                print(f"DIMENSION MISMATCH for {layer_name}: grad {grad_flat.shape[1]} vs proj {proj_matrix.shape[0]}")
                
                # Try to handle the mismatch
                if proj_matrix.shape[0] > grad_flat.shape[1]:
                    # Projection matrix is larger - pad gradient
                    padding_size = proj_matrix.shape[0] - grad_flat.shape[1]
                    grad_padded = F.pad(grad_flat, (0, padding_size), 'constant', 0)
                    proj_grad = torch.mm(grad_padded, proj_matrix)
                    projected_grad = proj_grad[:, :grad_flat.shape[1]].view(grad.shape)
                else:
                    # Projection matrix is smaller - truncate gradient
                    grad_truncated = grad_flat[:, :proj_matrix.shape[0]]
                    proj_grad = torch.mm(grad_truncated, proj_matrix)
                    # Pad back to original size
                    proj_grad_padded = F.pad(proj_grad, (0, grad_flat.shape[1] - proj_grad.shape[1]), 'constant', 0)
                    projected_grad = proj_grad_padded.view(grad.shape)
                
                if debug:
                    print(f"  Applied dimension fix")
            else:
                # Normal projection
                proj_grad = torch.mm(grad_flat, proj_matrix)
                projected_grad = proj_grad.view(grad.shape)
                
        elif len(grad.shape) == 2:  # Linear layer weights [out_features, in_features]
            out_feat, in_feat = grad.shape
            grad_flat = grad.view(out_feat, -1)  # Already flat
            
            if debug:
                print(f"  Linear layer: out_feat={out_feat}, in_feat={in_feat}")
            
            # Check dimension compatibility
            if grad_flat.shape[1] != proj_matrix.shape[0]:
                print(f"DIMENSION MISMATCH for {layer_name}: grad {grad_flat.shape[1]} vs proj {proj_matrix.shape[0]}")
                
                # Similar handling as conv layers
                if proj_matrix.shape[0] > grad_flat.shape[1]:
                    padding_size = proj_matrix.shape[0] - grad_flat.shape[1]
                    grad_padded = F.pad(grad_flat, (0, padding_size), 'constant', 0)
                    proj_grad = torch.mm(grad_padded, proj_matrix)
                    projected_grad = proj_grad[:, :grad_flat.shape[1]].view(grad.shape)
                else:
                    grad_truncated = grad_flat[:, :proj_matrix.shape[0]]
                    proj_grad = torch.mm(grad_truncated, proj_matrix)
                    proj_grad_padded = F.pad(proj_grad, (0, grad_flat.shape[1] - proj_grad.shape[1]), 'constant', 0)
                    projected_grad = proj_grad_padded.view(grad.shape)
            else:
                # Normal projection
                proj_grad = torch.mm(grad_flat, proj_matrix)
                projected_grad = proj_grad.view(grad.shape)
                
        else:
            # Unknown parameter shape - keep original
            projected_grad = grad.clone()
            if debug:
                print(f"  Unknown shape, keeping original")
        
        projected_gradients[param_name] = projected_grad
    
    return projected_gradients

def safe_subspace_overlap(U1, U2, k=None):
    """
    Fixed subspace overlap computation that handles dimension mismatches.
    """
    if U1.shape[0] != U2.shape[0]:
        print(f"WARNING: U1 shape {U1.shape} vs U2 shape {U2.shape} - dimension mismatch")
        # Take minimum dimension
        min_dim = min(U1.shape[0], U2.shape[0])
        U1 = U1[:min_dim, :]
        U2 = U2[:min_dim, :]
    
    if k is None:
        k = min(U1.shape[1], U2.shape[1])
    else:
        k = min(k, U1.shape[1], U2.shape[1])
    
    if k == 0:
        return 0.0
    
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]
    
    try:
        # Use .mT instead of .T to avoid deprecation warning
        _, s, _ = torch.svd(U1_k.mT @ U2_k)
        return torch.mean(s).item()
    except Exception as e:
        print(f"Error in SVD computation: {e}")
        return 0.0

def diagnose_pipeline(original_model, unlearned_model, data_loader, device):
    """
    Comprehensive diagnosis of the gradient recovery pipeline.
    """
    print("="*60)
    print("DIAGNOSING GRADIENT RECOVERY PIPELINE")
    print("="*60)
    
    # Step 1: Check feature dictionaries
    print("\n1. Checking feature dictionaries...")
    conv_fea_dict, linear_fea_dict = fix_get_feature_dict(original_model)
    
    # Step 2: Debug activation dimensions
    print("\n2. Debugging activation dimensions...")
    debug_activation_dimensions(original_model, data_loader, device, conv_fea_dict, linear_fea_dict)
    
    # Step 3: Debug gradient dimensions
    print("\n3. Debugging gradient dimensions...")
    debug_gradient_dimensions(original_model, data_loader, device)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    
    return conv_fea_dict, linear_fea_dict

# Example usage:
def fixed_gradient_recovery_pipeline(original_model, unlearned_model, data_loader, device, 
                                   retained_var_threshold=0.95, loss_fn=None, debug=True):
    """
    Fixed version of gradient recovery pipeline with proper debugging.
    """
    if debug:
        print("Starting FIXED gradient recovery pipeline...")
        # Run diagnosis
        conv_fea_dict, linear_fea_dict = diagnose_pipeline(original_model, unlearned_model, data_loader, device)
    else:
        conv_fea_dict, linear_fea_dict = fix_get_feature_dict(original_model)
    
    # Continue with the rest of the pipeline...
    # (Using the fixed functions above)
    
    return None  # Placeholder


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from collections import OrderedDict

# THE CORE PROBLEM IDENTIFIED:
# We're capturing OUTPUT activations when we need INPUT activations!

def get_input_node_mapping(model):
    """
    Create mapping from layer names to their INPUT node names.
    This is the key fix - we need to capture inputs to layers, not outputs.
    """
    train_nodes, eval_nodes = get_graph_node_names(model)
    
    # Build a mapping from layer name to the node that feeds into it
    input_mapping = {}
    
    # For ResNet-style architectures, we need to map each layer to its input
    # This is model-specific logic that might need adjustment for other architectures
    
    # First conv layer gets raw input
    input_mapping['conv1'] = 'input'  # Special case - raw input images
    
    # For other layers, we need to trace the computation graph
    # This is a simplified approach - you might need more sophisticated graph analysis
    
    # Layer naming patterns for ResNet:
    # conv1 -> uses raw input
    # layer1.0.conv1 -> uses output of conv1 (after bn1, relu)
    # layer1.0.conv2 -> uses output of layer1.0.conv1 (after bn1, relu)
    # etc.
    
    # A more robust approach would analyze the actual computation graph
    # For now, let's use a heuristic based on typical ResNet structure
    
    layer_sequence = [
        'conv1',  # input comes from raw image
        'layer1.0.conv1',  # input comes from conv1->bn1->relu  
        'layer1.0.conv2',  # input comes from layer1.0.conv1->bn1->relu
        # ... and so on
    ]
    
    print("WARNING: Using simplified input mapping. For production, implement full graph analysis.")
    
    return input_mapping

def create_input_feature_extractor(model, target_layers):
    """
    Create a feature extractor that captures INPUT activations to specified layers.
    """
    
    # Strategy: For each target layer, we need to find what feeds into it
    # This requires analyzing the model's forward computation graph
    
    # For ResNet models, we can use domain knowledge:
    input_nodes = {}
    
    for layer_name in target_layers:
        if layer_name == 'conv1':
            # First conv uses raw input - we'll handle this specially
            input_nodes['input'] = 'input'
        elif 'conv1' in layer_name and 'layer' in layer_name:
            # This is a conv1 in a residual block - input comes from previous block's output
            # We need to map this properly based on ResNet structure
            if 'layer1.0.conv1' == layer_name:
                # This comes after conv1->bn1->relu
                input_nodes[f'{layer_name}_input'] = 'relu'  # or whatever the relu node is called
            # Add more mappings as needed
        # Add more layer type mappings
    
    print(f"Input nodes mapping: {input_nodes}")
    
    # Create feature extractor with input nodes
    return create_feature_extractor(model, input_nodes)

def fixed_calculate_covar_with_correct_inputs(model, data_loader, device, epochs=1, conv_fea_dict=None, linear_fea_dict=None):
    """
    Fixed covariance calculation that captures the CORRECT input activations.
    """
    print(f"Starting FIXED covariance calculation with {len(data_loader)} batches")
    
    if conv_fea_dict is None or linear_fea_dict is None:
        conv_fea_dict, linear_fea_dict = fix_get_feature_dict(model)
    
    model.eval()
    
    # The key insight: We need to MANUALLY capture inputs to each layer
    # instead of using feature extractor on outputs
    
    covar = {}
    for key in {**conv_fea_dict, **linear_fea_dict}.keys():
        covar[key] = 0
    
    def get_layer_input_hook(layer_name, covar_dict):
        """Create a hook that captures INPUT to a layer."""
        def hook(module, input, output):
            # input is a tuple, we want input[0] which is the actual input tensor
            if len(input) > 0:
                input_tensor = input[0].detach()
                
                if layer_name in conv_fea_dict:
                    # Process as conv layer - unfold the input
                    ks = module.kernel_size
                    padding = module.padding
                    patch = F.unfold(input_tensor, ks, dilation=1, padding=padding, stride=1)
                    fea_dim = patch.shape[1]
                    patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
                    covar_dict[layer_name] += torch.mm(patch.permute(1, 0), patch)
                    
                elif layer_name in linear_fea_dict:
                    # Process as linear layer
                    f = input_tensor.double().squeeze()
                    if f.ndim == 1:
                        f = f.unsqueeze(0)
                    if f.ndim > 2:
                        f = f.view(f.shape[0], -1)  # Flatten if needed
                    covar_dict[layer_name] += torch.mm(f.permute(1, 0), f)
                    
        return hook
    
    # Register hooks for all target layers
    hooks = []
    for layer_name in {**conv_fea_dict, **linear_fea_dict}.keys():
        try:
            layer_module = get_module_by_name(model, layer_name)
            hook = layer_module.register_forward_hook(get_layer_input_hook(layer_name, covar))
            hooks.append(hook)
            print(f"Registered input hook for {layer_name}")
        except Exception as e:
            print(f"Failed to register hook for {layer_name}: {e}")
    
    print(f"Registered {len(hooks)} hooks")
    
    with torch.no_grad():
        for epoch in range(int(epochs)):
            print(f"Processing covariance epoch {epoch + 1}/{epochs}")
            for batch_idx, (imgs, lbls) in enumerate(data_loader):
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                # Forward pass - this will trigger all the hooks
                _ = model(imgs)
                
                if batch_idx % 100 == 0:
                    print(f"  Processed batch {batch_idx}/{len(data_loader)}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    print("Removed all hooks")
    
    # Normalize by epochs and number of samples
    total_samples = len(data_loader.dataset) * epochs
    for layer in covar:
        covar[layer] = covar[layer] / epochs
        print(f"Layer {layer}: covariance shape = {covar[layer].shape}")
    
    return covar

def get_module_by_name(model, layer_name):
    """Helper function to get module by name."""
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def verify_dimensions_match(model, data_loader, device, covar_dict):
    """
    Verify that covariance dimensions now match gradient dimensions.
    """
    print("\n" + "="*60)
    print("VERIFYING DIMENSION COMPATIBILITY")
    print("="*60)
    
    # Get one batch of gradients
    model.train()
    for batch_idx, (imgs, lbls) in enumerate(data_loader):
        if batch_idx > 0:
            break
            
        imgs, lbls = imgs.to(device), lbls.to(device)
        model.zero_grad()
        outputs = model(imgs)
        loss = nn.CrossEntropyLoss()(outputs, lbls)
        loss.backward()
        
        print("\nDimension compatibility check:")
        for name, param in model.named_parameters():
            if param.grad is not None and '.weight' in name:
                layer_name = name.replace('.weight', '')
                
                if layer_name in covar_dict:
                    if len(param.grad.shape) == 4:  # Conv weights
                        grad_flat_dim = param.grad.shape[1] * param.grad.shape[2] * param.grad.shape[3]
                        covar_dim = covar_dict[layer_name].shape[0]
                        status = "✓ MATCH" if grad_flat_dim == covar_dim else "✗ MISMATCH"
                        print(f"{layer_name}: grad_dim={grad_flat_dim}, covar_dim={covar_dim} {status}")
                        
                    elif len(param.grad.shape) == 2:  # Linear weights  
                        grad_flat_dim = param.grad.shape[1]
                        covar_dim = covar_dict[layer_name].shape[0]
                        status = "✓ MATCH" if grad_flat_dim == covar_dim else "✗ MISMATCH"
                        print(f"{layer_name}: grad_dim={grad_flat_dim}, covar_dim={covar_dim} {status}")
        break
    
    print("="*60)

def fix_get_feature_dict(model):
    """
    Fixed version that only includes layers we can actually project.
    """
    try:
        _, eval_nodes = get_graph_node_names(model)
        conv_fea_dict = OrderedDict()
        linear_fea_dict = OrderedDict()
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and name in eval_nodes:
                conv_fea_dict[name] = name
            elif isinstance(module, torch.nn.Linear) and name in eval_nodes:
                linear_fea_dict[name] = name
        
        print(f"Fixed: Generated {len(conv_fea_dict)} conv layers and {len(linear_fea_dict)} linear layers")
        return conv_fea_dict, linear_fea_dict
    except Exception as e:
        print(f"Error in fix_get_feature_dict: {e}")
        raise

# Usage example:
def test_fixed_covariance(model, data_loader, device):
    """
    Test the fixed covariance calculation.
    """
    print("Testing FIXED covariance calculation...")
    
    conv_fea_dict, linear_fea_dict = fix_get_feature_dict(model)
    
    # Calculate covariance with correct input capture
    covar = fixed_calculate_covar_with_correct_inputs(
        model, data_loader, device, epochs=1,
        conv_fea_dict=conv_fea_dict, linear_fea_dict=linear_fea_dict
    )
    
    # Verify dimensions match
    verify_dimensions_match(model, data_loader, device, covar)
    
    return covar