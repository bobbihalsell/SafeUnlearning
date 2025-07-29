# Integration example - replace the problematic sections in your main script

from gradient_recovery_fix import run_gradient_recovery_attack, GradientRecoveryPipeline

from importmodel import ImportModel
from utils import setup_device, initialize_dataloaders

# DATA ARGS
DATASET_NAME='cifar10'
DATASET_DIR = './data1'

# MODEL ARGS
LOAD_METHOD = 'torchhub'
MODEL_NAME = 'cifar10_resnet20'
NUM_CLASSES = 10
INIT_PATH = 'chenyaofo/pytorch-cifar-models'
ORIGINAL_CKP = './artifacts/unlearn/torchhub_test/unlearn/neggradplus/cifar10_resnet20_42_original_001.pt'
UNLEARNED_CKP = './artifacts/unlearn/torchhub_test/unlearn/neggradplus/cifar10_resnet20_42_unlearned_001.pt'

K = None
VARIANCE_THRESHOLD = 1.0

# CACHE SETTINGS
CACHE_DIR = './svd_cache'
FORCE_RECOMPUTE = True  # Set to True to force recomputation

def main():
    # Your existing setup code...
    device = setup_device()
    
    # Load models (your existing code)
    original_importer = ImportModel(
        load_method=LOAD_METHOD,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        init_path=INIT_PATH,
        model_ckpt_path=ORIGINAL_CKP
    )
    original_model = original_importer.model
    
    unlearned_importer = ImportModel(
        load_method=LOAD_METHOD,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        init_path=INIT_PATH,
        model_ckpt_path=UNLEARNED_CKP
    )
    unlearned_model = unlearned_importer.model
    
    # Initialize data (your existing code)
    dataloaders = initialize_dataloaders(
        splits=["bset", "forget", "retain"],
        batch_sizes={"bset": 64, "forget": 64, "retain": 64},
        num_workers=1,
        dataset_name=DATASET_NAME,
        dataset_save_dir=DATASET_DIR,
    )
    
    bdata = dataloaders["bset"]
    forgetdata = dataloaders["forget"]
    retaindata = dataloaders["retain"]
    
    # ==================================================================
    # REPLACE YOUR ENTIRE PIPELINE WITH THIS CLEAN VERSION
    # ==================================================================
    
    print('='*60)
    print('RUNNING FIXED GRADIENT RECOVERY ATTACK')
    print('='*60)
    
    # Run the attack
    results, pipeline = run_gradient_recovery_attack(
        original_model=original_model,
        unlearned_model=unlearned_model,
        background_data=bdata,  # Background data for attack
        forget_data=forgetdata,  # For verification only
        retain_data=retaindata,  # For verification only
        device=device,
        variance_threshold=VARIANCE_THRESHOLD
    )
    
    # ==================================================================
    # ANALYSIS OF RESULTS
    # ==================================================================
    
    print('\n' + '='*60)
    print('ATTACK RESULTS ANALYSIS')
    print('='*60)
    
    # Check if attack was successful
    if 'forget_comparison' in results:
        forget_comp = results['forget_comparison']
        
        print("\nEstimated vs True Forget Subspace Similarity:")
        print("-" * 50)
        
        similarities = []
        for layer_name, comp in forget_comp.items():
            if comp:
                sim = comp['mean_cosine']
                similarities.append(sim)
                print(f"{layer_name:25s}: cosine={sim:.3f}, angle={comp['mean_angle_degrees']:5.1f}°")
        
        if similarities:
            mean_similarity = np.mean(similarities)
            print(f"\nOverall Mean Cosine Similarity: {mean_similarity:.3f}")
            
            if mean_similarity > 0.8:
                print("🎯 ATTACK SUCCESSFUL: High similarity to true forget subspace!")
            elif mean_similarity > 0.6:
                print("⚠️  ATTACK PARTIALLY SUCCESSFUL: Moderate similarity")
            else:
                print("❌ ATTACK FAILED: Low similarity to true forget subspace")
    
    if 'retain_comparison' in results:
        retain_comp = results['retain_comparison']
        
        print("\nEstimated Forget vs True Retain Subspace (should be different):")
        print("-" * 50)
        
        retain_similarities = []
        for layer_name, comp in retain_comp.items():
            if comp:
                sim = comp['mean_cosine']
                retain_similarities.append(sim)
                print(f"{layer_name:25s}: cosine={sim:.3f}, angle={comp['mean_angle_degrees']:5.1f}°")
        
        if retain_similarities:
            mean_retain_sim = np.mean(retain_similarities)
            print(f"\nMean Cosine with Retain: {mean_retain_sim:.3f}")
            
            if mean_retain_sim < 0.3:
                print("✅ GOOD: Low similarity to retain subspace (as expected)")
            else:
                print("⚠️  WARNING: High similarity to retain subspace")
    
    # Test gradient projection effectiveness
    print("\n" + "="*60)
    print("GRADIENT PROJECTION TEST")
    print("="*60)
    
    original_grads = results['original_gradients']
    projected_grads = results['projected_gradients']
    
    # Calculate how much the gradients changed
    total_original_norm = 0
    total_projection_norm = 0
    
    for param_name in original_grads:
        if param_name in projected_grads:
            orig_grad = original_grads[param_name]
            proj_grad = projected_grads[param_name]
            
            if orig_grad is not None and proj_grad is not None:
                diff = orig_grad - proj_grad
                
                orig_norm = torch.norm(orig_grad).item()
                diff_norm = torch.norm(diff).item()
                
                total_original_norm += orig_norm
                total_projection_norm += diff_norm
                
                if '.weight' in param_name:
                    layer_name = param_name.replace('.weight', '')
                    if layer_name in results['projection_matrices']:
                        relative_change = diff_norm / (orig_norm + 1e-8)
                        print(f"{layer_name:25s}: projection strength = {relative_change:.3f}")
    
    overall_projection_strength = total_projection_norm / (total_original_norm + 1e-8)
    print(f"\nOverall Projection Strength: {overall_projection_strength:.3f}")
    
    # ==================================================================
    # DETAILED LAYER-BY-LAYER ANALYSIS
    # ==================================================================
    
    print("\n" + "="*60)
    print("DETAILED LAYER ANALYSIS")
    print("="*60)
    
    est_forget_svd = results['estimated_forget_svd']
    
    for layer_name in list(est_forget_svd.keys())[:5]:  # Show first 5 layers
        svd_data = est_forget_svd[layer_name]
        
        print(f"\nLayer: {layer_name}")
        print(f"  Subspace dimension: {svd_data['U'].shape[1]}")
        print(f"  Feature space dimension: {svd_data['U'].shape[0]}")
        print(f"  Explained variance: {svd_data['explained_variance_ratio']:.3f}")
        
        # Show singular values
        S = svd_data['S']
        if len(S) > 0:
            print(f"  Top 3 singular values: {S[:3].tolist()}")
            print(f"  Singular value ratio (1st/last): {S[0]/S[-1]:.1f}")

if __name__ == "__main__":
    main()


# ==================================================================
# ALTERNATIVE: If you want to use the pipeline step-by-step
# ==================================================================

def step_by_step_example():
    """
    Example showing how to use the pipeline components individually
    for more control over the process.
    """
    
    # Initialize pipeline
    pipeline = GradientRecoveryPipeline(device='cuda', verbose=True)
    
    # Your model and data setup...
    # original_model, unlearned_model, bdata = ...
    
    # Step 1: Get target layers
    conv_layers, linear_layers = pipeline.get_target_layers(original_model)
    
    # Step 2: Calculate covariances
    print("Computing covariances...")
    original_cov = pipeline.calculate_input_covariances(
        original_model, bdata, conv_layers, linear_layers
    )
    
    unlearned_cov = pipeline.calculate_input_covariances(
        unlearned_model, bdata, conv_layers, linear_layers
    )
    
    # Step 3: Estimate forget subspace
    print("Estimating forget subspace...")
    estimated_forget_svd = pipeline.estimate_forget_subspace(
        original_cov, unlearned_cov, variance_threshold=0.95
    )
    
    # Step 4: Create projection matrices
    projection_matrices = pipeline.create_projection_matrices(estimated_forget_svd)
    
    # Step 5: Test on gradients
    gradients = pipeline.get_gradients(original_model, bdata, max_batches=10)
    projected_gradients = pipeline.project_gradients(
        gradients, projection_matrices, conv_layers, linear_layers
    )
    
    # Step 6: Analysis
    # Custom analysis code here...
    
    return {
        'estimated_forget_svd': estimated_forget_svd,
        'projection_matrices': projection_matrices,
        'gradients': gradients,
        'projected_gradients': projected_gradients
    }

# ==================================================================
# DEBUGGING HELPER FUNCTIONS
# ==================================================================

def debug_dimension_compatibility(pipeline, model, data_loader):
    """
    Debug function to verify that dimensions are now compatible.
    """
    print("="*50)
    print("DEBUGGING DIMENSION COMPATIBILITY")
    print("="*50)
    
    conv_layers, linear_layers = pipeline.get_target_layers(model)
    
    # Calculate covariances
    covariances = pipeline.calculate_input_covariances(
        model, data_loader, conv_layers, linear_layers
    )
    
    # Get gradients for one batch
    gradients = pipeline.get_gradients(model, data_loader, max_batches=1)
    
    # Check compatibility
    print("\nDimension Compatibility Check:")
    print("-" * 30)
    
    for param_name, grad in gradients.items():
        if '.weight' in param_name:
            layer_name = param_name.replace('.weight', '')
            
            if layer_name in covariances:
                cov_dim = covariances[layer_name].shape[0]
                
                if len(grad.shape) == 4:  # Conv
                    grad_dim = grad.shape[1] * grad.shape[2] * grad.shape[3]
                elif len(grad.shape) == 2:  # Linear
                    grad_dim = grad.shape[1]
                else:
                    continue
                
                status = "✅ MATCH" if grad_dim == cov_dim else "❌ MISMATCH"
                print(f"{layer_name:20s}: grad={grad_dim:6d}, cov={cov_dim:6d} {status}")

def verify_attack_success(results):
    """
    Automated verification of attack success based on multiple criteria.
    """
    success_score = 0
    max_score = 0
    
    print("\n" + "="*50)
    print("ATTACK SUCCESS VERIFICATION")
    print("="*50)
    
    # Criterion 1: Similarity to forget subspace
    if 'forget_comparison' in results:
        forget_similarities = [
            comp['mean_cosine'] for comp in results['forget_comparison'].values() 
            if comp is not None
        ]
        if forget_similarities:
            mean_forget_sim = np.mean(forget_similarities)
            if mean_forget_sim > 0.8:
                success_score += 3
                print(f"✅ High similarity to forget subspace: {mean_forget_sim:.3f}")
            elif mean_forget_sim > 0.6:
                success_score += 2
                print(f"⚠️  Moderate similarity to forget subspace: {mean_forget_sim:.3f}")
            else:
                success_score += 0
                print(f"❌ Low similarity to forget subspace: {mean_forget_sim:.3f}")
        max_score += 3
    
    # Criterion 2: Dissimilarity to retain subspace
    if 'retain_comparison' in results:
        retain_similarities = [
            comp['mean_cosine'] for comp in results['retain_comparison'].values() 
            if comp is not None
        ]
        if retain_similarities:
            mean_retain_sim = np.mean(retain_similarities)
            if mean_retain_sim < 0.3:
                success_score += 2
                print(f"✅ Low similarity to retain subspace: {mean_retain_sim:.3f}")
            elif mean_retain_sim < 0.5:
                success_score += 1
                print(f"⚠️  Moderate similarity to retain subspace: {mean_retain_sim:.3f}")
            else:
                success_score += 0
                print(f"❌ High similarity to retain subspace: {mean_retain_sim:.3f}")
        max_score += 2
    
    # Criterion 3: Projection strength
    if 'original_gradients' in results and 'projected_gradients' in results:
        original_grads = results['original_gradients']
        projected_grads = results['projected_gradients']
        
        total_orig_norm = sum(torch.norm(g).item() for g in original_grads.values() if g is not None)
        total_diff_norm = sum(
            torch.norm(original_grads[k] - projected_grads[k]).item() 
            for k in original_grads if k in projected_grads and original_grads[k] is not None
        )
        
        projection_strength = total_diff_norm / (total_orig_norm + 1e-8)
        
        if projection_strength > 0.1:
            success_score += 2
            print(f"✅ Strong gradient projection: {projection_strength:.3f}")
        elif projection_strength > 0.05:
            success_score += 1
            print(f"⚠️  Moderate gradient projection: {projection_strength:.3f}")
        else:
            success_score += 0
            print(f"❌ Weak gradient projection: {projection_strength:.3f}")
        max_score += 2
    
    # Overall assessment
    success_percentage = (success_score / max_score) * 100 if max_score > 0 else 0
    
    print(f"\nOverall Success Score: {success_score}/{max_score} ({success_percentage:.1f}%)")
    
    if success_percentage >= 80:
        print("🎯 ATTACK HIGHLY SUCCESSFUL!")
    elif success_percentage >= 60:
        print("✅ ATTACK SUCCESSFUL!")
    elif success_percentage >= 40:
        print("⚠️  ATTACK PARTIALLY SUCCESSFUL")
    else:
        print("❌ ATTACK FAILED")
    
    return success_score, max_score