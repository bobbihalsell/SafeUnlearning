# Integration example with caching - replace your existing pipeline code

# from rcache import run_cached_gradient_recovery_attack, CachedGradientRecoveryPipeline
# from grad_analysis import analyze_attack_results
# import numpy as np
# import torch


# from importmodel import ImportModel
# from utils import setup_device, initialize_dataloaders

# # DATA ARGS
# DATASET_NAME='cifar10'
# DATASET_DIR = './data1'

# # MODEL ARGS
# LOAD_METHOD = 'torchhub'
# MODEL_NAME = 'cifar10_resnet20'
# NUM_CLASSES = 10
# INIT_PATH = 'chenyaofo/pytorch-cifar-models'
# ORIGINAL_CKP = './artifacts/unlearn/torchhub_test/unlearn/pgu/cifar10_resnet20_42_original_001.pt'
# UNLEARNED_CKP = './artifacts/unlearn/torchhub_test/unlearn/pgu/cifar10_resnet20_42_unlearned_001.pt'

# K = None
# VARIANCE_THRESHOLD = 0.85

# # CACHE SETTINGS
# CACHE_DIR = './svd_cache'
# FORCE_RECOMPUTE = True  # Set to True to force recomputation


# def main_cached():
#     """
#     Main function using the cached gradient recovery pipeline.
#     Subsequent runs will be much faster due to caching!
#     """
#     # Your existing setup code...
#     device = setup_device()
    
#     # Load models (your existing code)
#     original_importer = ImportModel(
#         load_method=LOAD_METHOD,
#         model_name=MODEL_NAME,
#         num_classes=NUM_CLASSES,
#         init_path=INIT_PATH,
#         model_ckpt_path=ORIGINAL_CKP
#     )
#     original_model = original_importer.model
    
#     unlearned_importer = ImportModel(
#         load_method=LOAD_METHOD,
#         model_name=MODEL_NAME,
#         num_classes=NUM_CLASSES,
#         init_path=INIT_PATH,
#         model_ckpt_path=UNLEARNED_CKP
#     )
#     unlearned_model = unlearned_importer.model
    
#     # Initialize data (your existing code)
#     dataloaders = initialize_dataloaders(
#         splits=["bset", "forget", "retain"],
#         batch_sizes={"bset": 64, "forget": 64, "retain": 64},
#         num_workers=1,
#         dataset_name=DATASET_NAME,
#         dataset_save_dir=DATASET_DIR,
#     )
    
#     bdata = dataloaders["bset"]
#     forgetdata = dataloaders["forget"]
#     retaindata = dataloaders["retain"]
    
#     # ==================================================================
#     # CACHED GRADIENT RECOVERY ATTACK
#     # ==================================================================
    
#     print('='*60)
#     print('RUNNING CACHED GRADIENT RECOVERY ATTACK')
#     print('='*60)
    
#     # Run the attack with caching
#     # First run will compute everything and cache it
#     # Subsequent runs will load from cache (much faster!)
#     results, pipeline = run_cached_gradient_recovery_attack(
#         original_model=original_model,
#         unlearned_model=unlearned_model,
#         background_data=bdata,  # Background data for attack
#         forget_data=forgetdata,  # For verification only  
#         retain_data=retaindata,  # For verification only
#         device=device,
#         variance_threshold=VARIANCE_THRESHOLD,
#         cache_dir=CACHE_DIR,  # Your cache directory
#         force_recompute=FORCE_RECOMPUTE  # Set to True to ignore cache and recompute
#     )
    
#     # ==================================================================
#     # CACHE MANAGEMENT EXAMPLES
#     # ==================================================================
    
#     print('\n' + '='*60)
#     print('CACHE MANAGEMENT')
#     print('='*60)
    
#     # Get cache information
#     cache_info = pipeline.get_cache_info()
#     print(f"Cache contains {cache_info['num_files']} files using {cache_info['total_size_mb']:.1f} MB")
    
#     # Clear specific cache entries (optional)
#     # pipeline.clear_cache("covariances")  # Clear only covariance caches
#     # pipeline.clear_cache("*")            # Clear all cache
    
#     # ==================================================================
#     # ANALYSIS OF RESULTS (same as before)
#     # ==================================================================
    
#     analyze_attack_results(results)
    
#     return results, pipeline

# def analyze_attack_results(results):
#     """
#     Analyze and display attack results.
#     """
#     print('\n' + '='*60)
#     print('ATTACK RESULTS ANALYSIS')
#     print('='*60)
    
#     # Check if attack was successful
#     if 'forget_comparison' in results:
#         forget_comp = results['forget_comparison']
        
#         print("\nEstimated vs True Forget Subspace Similarity:")
#         print("-" * 50)
        
#         similarities = []
#         for layer_name, comp in forget_comp.items():
#             if comp:
#                 sim = comp['mean_cosine']
#                 similarities.append(sim)
#                 print(f"{layer_name:25s}: cosine={sim:.3f}, angle={comp['mean_angle_degrees']:5.1f}°")
        
#         if similarities:
#             mean_similarity = np.mean(similarities)
#             print(f"\nOverall Mean Cosine Similarity: {mean_similarity:.3f}")
            
#             if mean_similarity > 0.8:
#                 print("🎯 ATTACK SUCCESSFUL: High similarity to true forget subspace!")
#             elif mean_similarity > 0.6:
#                 print("⚠️  ATTACK PARTIALLY SUCCESSFUL: Moderate similarity")
#             else:
#                 print("❌ ATTACK FAILED: Low similarity to true forget subspace")
    
#     if 'retain_comparison' in results:
#         retain_comp = results['retain_comparison']
        
#         print("\nEstimated Forget vs True Retain Subspace (should be different):")
#         print("-" * 50)
        
#         retain_similarities = []
#         for layer_name, comp in retain_comp.items():
#             if comp:
#                 sim = comp['mean_cosine']
#                 retain_similarities.append(sim)
#                 print(f"{layer_name:25s}: cosine={sim:.3f}, angle={comp['mean_angle_degrees']:5.1f}°")
        
#         if retain_similarities:
#             mean_retain_sim = np.mean(retain_similarities)
#             print(f"\nMean Cosine with Retain: {mean_retain_sim:.3f}")
            
#             if mean_retain_sim < 0.3:
#                 print("✅ GOOD: Low similarity to retain subspace (as expected)")
#             else:
#                 print("⚠️  WARNING: High similarity to retain subspace")
    
#     # Test gradient projection effectiveness
#     print("\n" + "="*60)
#     print("GRADIENT PROJECTION TEST")
#     print("="*60)
    
#     original_grads = results['original_gradients']
#     projected_grads = results['projected_gradients']
    
#     # Calculate how much the gradients changed
#     total_original_norm = 0
#     total_projection_norm = 0
    
#     for param_name in original_grads:
#         if param_name in projected_grads:
#             orig_grad = original_grads[param_name]
#             proj_grad = projected_grads[param_name]
            
#             if orig_grad is not None and proj_grad is not None:
#                 diff = orig_grad - proj_grad
                
#                 orig_norm = torch.norm(orig_grad).item()
#                 diff_norm = torch.norm(diff).item()
                
#                 total_original_norm += orig_norm
#                 total_projection_norm += diff_norm
                
#                 if '.weight' in param_name:
#                     layer_name = param_name.replace('.weight', '')
#                     if layer_name in results['projection_matrices']:
#                         relative_change = diff_norm / (orig_norm + 1e-8)
#                         print(f"{layer_name:25s}: projection strength = {relative_change:.3f}")
    
#     overall_projection_strength = total_projection_norm / (total_original_norm + 1e-8)
#     print(f"\nOverall Projection Strength: {overall_projection_strength:.3f}")

# # ==================================================================
# # ADVANCED USAGE: Manual cache control
# # ==================================================================

# def advanced_cached_usage():
#     """
#     Example showing advanced usage with manual cache control.
#     """
    
#     # Initialize pipeline with custom settings
#     pipeline = CachedGradientRecoveryPipeline(
#         device='cuda',
#         cache_dir='./custom_cache',
#         force_recompute=False,  # Respect cache
#         verbose=True
#     )
    
#     # Your model and data setup...
#     # original_model, unlearned_model, bdata, forgetdata, retaindata = ...
    
#     # Check what's in cache before starting
#     print("Initial cache state:")
#     pipeline.get_cache_info()
    
#     # Clear old cache if needed
#     # pipeline.clear_cache("*")  # Clear everything
#     # pipeline.clear_cache("covariances")  # Clear only covariances
    
#     # Step-by-step execution with caching
#     print("\n" + "="*60)
#     print("STEP-BY-STEP CACHED EXECUTION")
#     print("="*60)
    
#     # Step 1: Get target layers (cached)
#     conv_layers, linear_layers = pipeline.get_target_layers(original_model)
    
#     # Step 2: Calculate covariances (heavily cached - most expensive operation)
#     print("\n1. Computing original model covariances...")
#     original_cov = pipeline.calculate_input_covariances(
#         original_model, bdata, conv_layers, linear_layers
#     )
    
#     print("\n2. Computing unlearned model covariances...")
#     unlearned_cov = pipeline.calculate_input_covariances(
#         unlearned_model, bdata, conv_layers, linear_layers
#     )
    
#     # Step 3: Estimate forget subspace (cached)
#     print("\n3. Estimating forget subspace...")
#     estimated_forget_svd = pipeline.estimate_forget_subspace(
#         original_cov, unlearned_cov, variance_threshold=0.95
#     )
    
#     # Step 4: Create projection matrices (cached)
#     print("\n4. Creating projection matrices...")
#     projection_matrices = pipeline.create_projection_matrices(estimated_forget_svd)
    
#     # Step 5: Test on gradients (cached)
#     print("\n5. Computing and projecting gradients...")
#     gradients = pipeline.get_gradients(original_model, bdata, max_batches=10)
#     projected_gradients = pipeline.project_gradients(
#         gradients, projection_matrices, conv_layers, linear_layers
#     )
    
#     # Step 6: Verification with ground truth (cached)
#     if forgetdata is not None:
#         print("\n6. Verification with ground truth...")
        
#         forget_cov = pipeline.calculate_input_covariances(
#             original_model, forgetdata, conv_layers, linear_layers
#         )
#         true_forget_svd = pipeline.calculate_svd(forget_cov, variance_threshold=0.95)
        
#         # Compare subspaces (cached)
#         forget_comparison = pipeline.compare_subspaces(
#             estimated_forget_svd, true_forget_svd,
#             "Estimated Forget", "True Forget"
#         )
        
#         if retaindata is not None:
#             retain_cov = pipeline.calculate_input_covariances(
#                 original_model, retaindata, conv_layers, linear_layers
#             )
#             retain_svd = pipeline.calculate_svd(retain_cov, variance_threshold=0.95)
            
#             retain_comparison = pipeline.compare_subspaces(
#                 estimated_forget_svd, retain_svd,
#                 "Estimated Forget", "True Retain"
#             )
    
#     # Show final cache state
#     print("\nFinal cache state:")
#     pipeline.get_cache_info()
    
#     return {
#         'estimated_forget_svd': estimated_forget_svd,
#         'projection_matrices': projection_matrices,
#         'gradients': gradients,
#         'projected_gradients': projected_gradients
#     }

# # ==================================================================
# # CACHE MANAGEMENT UTILITIES
# # ==================================================================

# def cache_management_examples():
#     """
#     Examples of cache management operations.
#     """
    
#     pipeline = CachedGradientRecoveryPipeline(
#         cache_dir='./svd_cache',
#         verbose=True
#     )
    
#     print("="*50)
#     print("CACHE MANAGEMENT EXAMPLES")
#     print("="*50)
    
#     # 1. Check cache status
#     print("\n1. Current cache status:")
#     cache_info = pipeline.get_cache_info()
    
#     # 2. Clear specific cache types
#     print("\n2. Clearing specific cache types:")
#     # pipeline.clear_cache("covariances")     # Clear covariance caches
#     # pipeline.clear_cache("svd")             # Clear SVD caches  
#     # pipeline.clear_cache("gradients")       # Clear gradient caches
#     # pipeline.clear_cache("projection")      # Clear projection matrix caches
    
#     # 3. Clear all cache
#     print("\n3. To clear all cache (uncomment to use):")
#     # pipeline.clear_cache("*")
    
#     # 4. Force recomputation for testing
#     print("\n4. Force recomputation example:")
#     pipeline_force = CachedGradientRecoveryPipeline(
#         cache_dir='./svd_cache',
#         force_recompute=True,  # This ignores all cache
#         verbose=True
#     )
    
#     print("\nCache management complete!")

# # ==================================================================
# # PERFORMANCE COMPARISON
# # ==================================================================

# def performance_comparison():
#     """
#     Compare performance with and without caching.
#     """
#     import time
    
#     # Your model and data setup...
#     # original_model, unlearned_model, bdata = ...
    
#     print("="*60)
#     print("PERFORMANCE COMPARISON: CACHED vs UNCACHED")
#     print("="*60)
    
#     # Test 1: First run (will populate cache)
#     print("\n🔄 First run (populating cache)...")
#     start_time = time.time()
    
#     results1, pipeline1 = run_cached_gradient_recovery_attack(
#         original_model, unlearned_model, bdata,
#         cache_dir='./perf_test_cache',
#         force_recompute=True  # Force computation
#     )
    
#     first_run_time = time.time() - start_time
#     print(f"⏱️  First run time: {first_run_time:.1f} seconds")
    
#     # Test 2: Second run (will use cache)
#     print("\n📁 Second run (using cache)...")
#     start_time = time.time()
    
#     results2, pipeline2 = run_cached_gradient_recovery_attack(
#         original_model, unlearned_model, bdata,
#         cache_dir='./perf_test_cache',
#         force_recompute=False  # Use cache
#     )
    
#     second_run_time = time.time() - start_time
#     print(f"⏱️  Second run time: {second_run_time:.1f} seconds")
    
#     # Calculate speedup
#     speedup = first_run_time / second_run_time
#     print(f"\n🚀 Speedup: {speedup:.1f}x faster with cache!")
#     print(f"   Time saved: {first_run_time - second_run_time:.1f} seconds")
    
#     # Verify results are identical
#     print("\n🔍 Verifying cached results are identical...")
    
#     # Compare a few key metrics
#     identical = True
    
#     if 'estimated_forget_svd' in results1 and 'estimated_forget_svd' in results2:
#         for layer in results1['estimated_forget_svd']:
#             if layer in results2['estimated_forget_svd']:
#                 U1 = results1['estimated_forget_svd'][layer]['U']
#                 U2 = results2['estimated_forget_svd'][layer]['U']
                
#                 if not torch.allclose(U1, U2, atol=1e-6):
#                     print(f"❌ Difference found in layer {layer}")
#                     identical = False
#                     break
    
#     if identical:
#         print("✅ Cached results are identical to computed results!")
#     else:
#         print("❌ Warning: Cached results differ from computed results")
    
#     # Clean up test cache
#     pipeline2.clear_cache("*")
#     print("\n🗑️  Cleaned up test cache")
    
#     return {
#         'first_run_time': first_run_time,
#         'second_run_time': second_run_time,
#         'speedup': speedup,
#         'identical': identical
#     }

# # ==================================================================
# # REPLACE YOUR ORIGINAL MAIN FUNCTION WITH THIS
# # ==================================================================

# if __name__ == "__main__":
#     # Replace your entire original pipeline with this:
    
#     print("🚀 Starting cached gradient recovery attack...")
    
#     # Option 1: Simple usage (recommended)
#     results, pipeline = main_cached()
    
#     # Option 2: Advanced usage with manual control (optional)
#     # results = advanced_cached_usage()
    
#     # Option 3: Performance testing (optional)  
#     # perf_results = performance_comparison()
    
#     # Option 4: Cache management (optional)
#     # cache_management_examples()
    
#     print("\n✅ Gradient recovery attack completed!")

# # ==================================================================
# # KEY BENEFITS OF CACHING:
# # ==================================================================

"""
🚀 CACHING BENEFITS:

1. **Massive Speed Improvements**: 
   - First run: ~10-30 minutes (depending on dataset size)
   - Subsequent runs: ~30 seconds to 2 minutes
   - 10-50x speedup for repeated experiments!

2. **Automatic Cache Management**:
   - Intelligent cache keys based on model + data characteristics
   - Automatic cache invalidation when inputs change
   - Cache hit/miss reporting

3. **Granular Caching**:
   - Covariance calculations (most expensive) cached separately
   - SVD computations cached
   - Gradient calculations cached
   - Comparisons cached

4. **Development Friendly**:
   - Experiment with different parameters quickly
   - Debug specific steps without recomputing everything
   - Resume interrupted experiments

5. **Storage Efficient**:
   - Only caches results, not intermediate data
   - Automatic compression via pickle
   - Easy cache cleanup utilities

6. **Production Ready**:
   - Robust error handling
   - Cache validation
   - Performance monitoring
   - Easy deployment

USAGE TIPS:
- Set force_recompute=True when you change models or key parameters
- Use clear_cache() to manage disk space
- Cache directory can be shared across experiments
- Monitor cache size with get_cache_info()
"""

#!/usr/bin/env python3

from rcache import run_cached_gradient_recovery_attack, CachedGradientRecoveryPipeline
from grad_analysis import analyze_attack_results, alternative_attack_metrics, test_attack_on_ideal_scenarios
from importmodel import ImportModel
from utils import setup_device, initialize_dataloaders
import numpy as np
import torch

# ==================================================================
# CONFIGURATION
# ==================================================================

# DATA ARGS
DATASET_NAME = 'cifar10'
DATASET_DIR = './data1'

# MODEL ARGS
LOAD_METHOD = 'torchhub'
MODEL_NAME = 'cifar10_resnet20'
NUM_CLASSES = 10
INIT_PATH = 'chenyaofo/pytorch-cifar-models'
ORIGINAL_CKP = './artifacts/unlearn/torchhub_test/unlearn/pgu/cifar10_resnet20_42_original_001.pt'
UNLEARNED_CKP = './artifacts/unlearn/torchhub_test/unlearn/pgu/cifar10_resnet20_42_unlearned_001.pt'

# ATTACK SETTINGS
VARIANCE_THRESHOLD = 0.85
CACHE_DIR = './svd_cache'
FORCE_RECOMPUTE = True

# ==================================================================
# EXECUTION CONTROL FLAGS
# ==================================================================

# Set these to True/False to control what runs
RUN_BASIC_ATTACK = True
RUN_ALTERNATIVE_ANALYSIS = True
RUN_IDEAL_SCENARIOS = True
RUN_DEBUG_ANALYSIS = True
RUN_PARAMETER_ANALYSIS = True

# Exit points for testing specific parts
EXIT_AFTER_SETUP = False
EXIT_AFTER_BASIC_ATTACK = False
EXIT_AFTER_ALTERNATIVE_ANALYSIS = True

def main():
    """
    Main function with selective execution control.
    """
    
    print("🚀 STARTING GRADIENT RECOVERY ANALYSIS")
    print("="*60)
    
    # ==================================================================
    # SETUP PHASE
    # ==================================================================
    
    print("\n📋 PHASE 1: SETUP")
    print("-" * 30)
    
    device = setup_device()
    print(f"Using device: {device}")
    
    # Load models
    print("Loading original model...")
    original_importer = ImportModel(
        load_method=LOAD_METHOD,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        init_path=INIT_PATH,
        model_ckpt_path=ORIGINAL_CKP
    )
    original_model = original_importer.model
    
    print("Loading unlearned model...")
    unlearned_importer = ImportModel(
        load_method=LOAD_METHOD,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        init_path=INIT_PATH,
        model_ckpt_path=UNLEARNED_CKP
    )
    unlearned_model = unlearned_importer.model
    
    # Initialize data
    print("Loading datasets...")
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
    
    print("✅ Setup completed successfully!")
    
    if EXIT_AFTER_SETUP:
        print("🛑 Exiting after setup (EXIT_AFTER_SETUP=True)")
        return
    
    # ==================================================================
    # BASIC ATTACK PHASE
    # ==================================================================
    
    if RUN_BASIC_ATTACK:
        print(f"\n🎯 PHASE 2: BASIC GRADIENT RECOVERY ATTACK")
        print("-" * 50)
        
        results, pipeline = run_cached_gradient_recovery_attack(
            original_model=original_model,
            unlearned_model=unlearned_model,
            background_data=bdata,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            variance_threshold=VARIANCE_THRESHOLD,
            cache_dir=CACHE_DIR,
            force_recompute=FORCE_RECOMPUTE
        )
        
        print("\n📊 ANALYZING BASIC ATTACK RESULTS:")
        analyze_attack_results(results)
        
        # Store results for later phases
        basic_results = results
        basic_pipeline = pipeline
        
        print("✅ Basic attack completed!")
        
        if EXIT_AFTER_BASIC_ATTACK:
            print("🛑 Exiting after basic attack (EXIT_AFTER_BASIC_ATTACK=True)")
            return basic_results
    else:
        print("⏭️  Skipping basic attack (RUN_BASIC_ATTACK=False)")
        basic_results = None
        basic_pipeline = CachedGradientRecoveryPipeline(device=device, cache_dir=CACHE_DIR)
    
    # ==================================================================
    # ALTERNATIVE ANALYSIS PHASE
    # ==================================================================
    
    if RUN_ALTERNATIVE_ANALYSIS:
        print(f"\n🔬 PHASE 3: ALTERNATIVE ANALYSIS METHODS")
        print("-" * 50)
        
        alt_results = alternative_attack_metrics(
            original_model, unlearned_model, bdata, 
            forgetdata, retaindata, device
        )
        
        print("✅ Alternative analysis completed!")
        
        if EXIT_AFTER_ALTERNATIVE_ANALYSIS:
            print("🛑 Exiting after alternative analysis (EXIT_AFTER_ALTERNATIVE_ANALYSIS=True)")
            return basic_results, alt_results
    else:
        print("⏭️  Skipping alternative analysis (RUN_ALTERNATIVE_ANALYSIS=False)")
    
    # ==================================================================
    # IDEAL SCENARIOS PHASE
    # ==================================================================
    
    if RUN_IDEAL_SCENARIOS:
        print(f"\n🧪 PHASE 4: IDEAL SCENARIO TESTING")
        print("-" * 50)
        
        scenario_results = test_attack_on_ideal_scenarios(original_model, device)
        
        print("✅ Ideal scenario testing completed!")
    else:
        print("⏭️  Skipping ideal scenarios (RUN_IDEAL_SCENARIOS=False)")
    
    # ==================================================================
    # DEBUG ANALYSIS PHASE
    # ==================================================================
    
    if RUN_DEBUG_ANALYSIS:
        print(f"\n🔍 PHASE 5: DEEP DEBUG ANALYSIS")
        print("-" * 50)
        
        debug_results = run_debug_analysis(
            original_model, unlearned_model, bdata,
            forgetdata, retaindata, device
        )
        
        print("✅ Debug analysis completed!")
    else:
        print("⏭️  Skipping debug analysis (RUN_DEBUG_ANALYSIS=False)")
    
    # ==================================================================
    # PARAMETER ANALYSIS PHASE
    # ==================================================================
    
    if RUN_PARAMETER_ANALYSIS:
        print(f"\n📊 PHASE 6: PARAMETER CHANGE ANALYSIS")
        print("-" * 50)
        
        param_results = run_parameter_analysis(original_model, unlearned_model)
        
        print("✅ Parameter analysis completed!")
    else:
        print("⏭️  Skipping parameter analysis (RUN_PARAMETER_ANALYSIS=False)")
    
    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    
    print(f"\n🎉 ALL ANALYSES COMPLETED!")
    print("="*60)
    
    if RUN_BASIC_ATTACK and basic_results:
        print_final_summary(basic_results)
    
    return basic_results if RUN_BASIC_ATTACK else None

def run_debug_analysis(original_model, unlearned_model, bdata,
                      forgetdata, retaindata, device):
    """
    Run deep debug analysis to identify root causes.
    """
    
    try:
        from deep_attack_analysis import deep_attack_analysis
        return deep_attack_analysis(
            original_model, unlearned_model, bdata,
            forgetdata, retaindata, device
        )
    except ImportError:
        print("⚠️  Deep debug analysis module not available")
        print("   Put deep_attack_analysis.py in the same directory")
        
        # Fallback: run basic parameter analysis
        return run_parameter_analysis(original_model, unlearned_model)

def run_parameter_analysis(original_model, unlearned_model):
    """
    Analyze which parameters changed most during unlearning.
    """
    
    print("Analyzing parameter changes...")
    
    param_changes = []
    total_change = 0
    total_norm = 0
    
    for (name1, p1), (name2, p2) in zip(original_model.named_parameters(), 
                                        unlearned_model.named_parameters()):
        assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
        
        change = torch.norm(p1 - p2).item()
        norm = torch.norm(p1).item()
        relative_change = change / (norm + 1e-8)
        
        total_change += change
        total_norm += norm
        
        if '.weight' in name1:
            layer_name = name1.replace('.weight', '')
            param_changes.append((layer_name, relative_change, change))
    
    # Sort by relative change
    param_changes.sort(key=lambda x: x[1], reverse=True)
    
    print("\nLayers with most parameter changes:")
    for layer_name, rel_change, abs_change in param_changes[:10]:
        print(f"  {layer_name:25s}: rel_change={rel_change:.6f}, abs_change={abs_change:.6f}")
    
    overall_change = total_change / (total_norm + 1e-8)
    print(f"\nOverall model change: {overall_change:.6f}")
    
    if overall_change < 1e-4:
        print("❌ CRITICAL: Models are nearly identical!")
        print("   Unlearning may have failed or been too weak.")
    elif overall_change < 1e-2:
        print("⚠️  WARNING: Models are very similar")
        print("   Unlearning effect may be minimal.")
    else:
        print("✅ Models show significant differences")
        print("   Unlearning appears to have worked.")
    
    return param_changes

def print_final_summary(results):
    """
    Print a concise final summary of the attack results.
    """
    
    print("FINAL SUMMARY:")
    print("-" * 20)
    
    if 'forget_comparison' in results and 'retain_comparison' in results:
        forget_sims = [c['mean_cosine'] for c in results['forget_comparison'].values() if c]
        retain_sims = [c['mean_cosine'] for c in results['retain_comparison'].values() if c]
        
        if forget_sims and retain_sims:
            mean_forget = np.mean(forget_sims)
            mean_retain = np.mean(retain_sims)
            
            print(f"📊 Forget similarity: {mean_forget:.3f}")
            print(f"📊 Retain similarity: {mean_retain:.3f}")
            print(f"📊 Difference: {mean_forget - mean_retain:.3f}")
            
            if mean_forget > 0.8 and mean_retain < 0.5:
                print("🎯 ATTACK SUCCESSFUL!")
                print("   High similarity to forget, low similarity to retain")
            elif mean_forget > 0.7:
                print("⚠️  ATTACK PARTIALLY SUCCESSFUL")
                print("   Good forget similarity but retain similarity too high")
            else:
                print("❌ ATTACK FAILED")
                print("   Low similarity to forget subspace")
        else:
            print("❌ Could not compute similarities")
    else:
        print("❌ Missing comparison data")

# ==================================================================
# QUICK TEST FUNCTIONS
# ==================================================================

def quick_test_basic_attack():
    """
    Quick function to test just the basic attack.
    """
    global RUN_BASIC_ATTACK, EXIT_AFTER_BASIC_ATTACK
    global RUN_ALTERNATIVE_ANALYSIS, RUN_IDEAL_SCENARIOS, RUN_DEBUG_ANALYSIS, RUN_PARAMETER_ANALYSIS
    
    # Set flags for basic attack only
    RUN_BASIC_ATTACK = True
    EXIT_AFTER_BASIC_ATTACK = True
    RUN_ALTERNATIVE_ANALYSIS = False
    RUN_IDEAL_SCENARIOS = False
    RUN_DEBUG_ANALYSIS = False
    RUN_PARAMETER_ANALYSIS = False
    
    print("🚀 QUICK TEST: Basic Attack Only")
    return main()

def quick_test_parameter_analysis():
    """
    Quick function to test just parameter analysis.
    """
    global RUN_BASIC_ATTACK, RUN_PARAMETER_ANALYSIS
    global RUN_ALTERNATIVE_ANALYSIS, RUN_IDEAL_SCENARIOS, RUN_DEBUG_ANALYSIS
    
    # Set flags for parameter analysis only
    RUN_BASIC_ATTACK = False
    RUN_PARAMETER_ANALYSIS = True
    RUN_ALTERNATIVE_ANALYSIS = False
    RUN_IDEAL_SCENARIOS = False
    RUN_DEBUG_ANALYSIS = False
    
    print("🚀 QUICK TEST: Parameter Analysis Only")
    return main()

def quick_test_alternative_methods():
    """
    Quick function to test alternative analysis methods.
    """
    global RUN_BASIC_ATTACK, RUN_ALTERNATIVE_ANALYSIS, EXIT_AFTER_ALTERNATIVE_ANALYSIS
    global RUN_IDEAL_SCENARIOS, RUN_DEBUG_ANALYSIS, RUN_PARAMETER_ANALYSIS
    
    # Set flags for alternative analysis
    RUN_BASIC_ATTACK = True
    RUN_ALTERNATIVE_ANALYSIS = True
    EXIT_AFTER_ALTERNATIVE_ANALYSIS = True
    RUN_IDEAL_SCENARIOS = False
    RUN_DEBUG_ANALYSIS = False
    RUN_PARAMETER_ANALYSIS = False
    
    print("🚀 QUICK TEST: Basic + Alternative Analysis")
    return main()

def quick_test_ideal_scenarios():
    """
    Quick function to test ideal scenarios only.
    """
    global RUN_BASIC_ATTACK, RUN_IDEAL_SCENARIOS
    global RUN_ALTERNATIVE_ANALYSIS, RUN_DEBUG_ANALYSIS, RUN_PARAMETER_ANALYSIS
    
    # Set flags for ideal scenarios only
    RUN_BASIC_ATTACK = False
    RUN_IDEAL_SCENARIOS = True
    RUN_ALTERNATIVE_ANALYSIS = False
    RUN_DEBUG_ANALYSIS = False
    RUN_PARAMETER_ANALYSIS = False
    
    print("🚀 QUICK TEST: Ideal Scenarios Only")
    return main()

# ==================================================================
# MAIN EXECUTION
# ==================================================================

if __name__ == "__main__":
    
    # ==================================================================
    # CHOOSE YOUR EXECUTION MODE
    # ==================================================================
    
    # Option 1: Run everything (default)
    results = main()
    
    # Option 2: Quick tests (uncomment one of these)
    results = quick_test_basic_attack()
    results = quick_test_parameter_analysis()  
    results = quick_test_alternative_methods()
    results = quick_test_ideal_scenarios()
    
    # Option 3: Custom configuration (modify flags at top of file)
    # Modify the RUN_* flags at the top and run main()
    
    print("\n✅ All requested analyses completed!")

# ==================================================================
# USAGE EXAMPLES:
# ==================================================================

"""
HOW TO USE THIS FIXED VERSION:

1. RUN EVERYTHING:
   python main.py

2. RUN ONLY BASIC ATTACK:
   # Uncomment: results = quick_test_basic_attack()
   
3. RUN ONLY PARAMETER ANALYSIS:
   # Uncomment: results = quick_test_parameter_analysis()

4. RUN BASIC + ALTERNATIVE ANALYSIS:
   # Uncomment: results = quick_test_alternative_methods()

5. RUN IDEAL SCENARIOS:
   # Uncomment: results = quick_test_ideal_scenarios()

6. CUSTOM CONFIGURATION:
   # Edit the RUN_* flags at the top:
   RUN_BASIC_ATTACK = True
   RUN_ALTERNATIVE_ANALYSIS = True  
   RUN_IDEAL_SCENARIOS = False
   RUN_DEBUG_ANALYSIS = False
   RUN_PARAMETER_ANALYSIS = True
   
7. EXIT EARLY FOR TESTING:
   # Set any EXIT_AFTER_* flag to True:
   EXIT_AFTER_SETUP = True  # Exit after loading models/data
   EXIT_AFTER_BASIC_ATTACK = True  # Exit after basic attack

8. FORCE RECOMPUTATION:
   FORCE_RECOMPUTE = True  # Ignore cache, recompute everything
   FORCE_RECOMPUTE = False # Use cache when available

RECOMMENDED WORKFLOW:
1. First run: Set EXIT_AFTER_BASIC_ATTACK = True to test basic attack
2. If basic attack has issues: Set RUN_PARAMETER_ANALYSIS = True  
3. If basic attack works: Enable RUN_ALTERNATIVE_ANALYSIS = True
4. For validation: Enable RUN_IDEAL_SCENARIOS = True

TROUBLESHOOTING:
- If imports fail, make sure all files are in the same directory
- If cache issues, set FORCE_RECOMPUTE = True
- If dimension mismatches, check VARIANCE_THRESHOLD (try 0.85, 0.9, 0.95)
"""