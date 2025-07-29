#!/usr/bin/env python3

from recover import run_cached_gradient_recovery_attack, CachedGradientRecoveryPipeline
from analysis import analyze_attack_results, alternative_attack_metrics, test_attack_on_ideal_scenarios
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
VERBOSE = True  

# ATTACK SETTINGS
VARIANCE_THRESHOLD = 0.85
CACHE_DIR = './svd_cache'
FORCE_RECOMPUTE = False

# ==================================================================
# EXECUTION CONTROL FLAGS
# ==================================================================

# Set these to True/False to control what runs
RUN_PROJECTION_ATTACK = False
RUN_ALTERNATIVE_ANALYSIS = True
RUN_IDEAL_SCENARIOS = True
RUN_DEBUG_ANALYSIS = True
RUN_PARAMETER_ANALYSIS = True

# Exit points for testing specific parts
EXIT_AFTER_BASIC_ATTACK = False
EXIT_AFTER_ALTERNATIVE_ANALYSIS = False 

def main():
    """
    Main function with selective execution control.
    """
    
    print("STARTING GRADIENT RECOVERY ANALYSIS")
    print("="*60)
    
    # ==================================================================
    # SETUP PHASE
    # ==================================================================
    
    print("\n SETUP")
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
    
    print("Setup completed successfully!")
    
    # ==================================================================
    # BASIC ATTACK PHASE
    # ==================================================================
    
    if RUN_PROJECTION_ATTACK:
        print(f"\n GRADIENT RECOVERY ATTACK")
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
            force_recompute=FORCE_RECOMPUTE,
            verbose=VERBOSE
        )
        
        print("\n ANALYZING BASIC ATTACK RESULTS:")
        analyze_attack_results(results)
        
        # Store results for later phases
        basic_results = results
        basic_pipeline = pipeline
        
        print("Attack completed!")
        
        if EXIT_AFTER_BASIC_ATTACK:
            print("Exiting after basic attack (EXIT_AFTER_BASIC_ATTACK=True)")
            return basic_results
    else:
        print("Skipping basic attack (RUN_PROJECTION_ATTACK=False)")
        basic_results = None
        basic_pipeline = CachedGradientRecoveryPipeline(device=device, cache_dir=CACHE_DIR, verbose=VERBOSE)
    
    # ==================================================================
    # ALTERNATIVE ANALYSIS PHASE
    # ==================================================================
    
    if RUN_ALTERNATIVE_ANALYSIS:
        print(f"\n ALTERNATIVE ANALYSIS METHODS")
        print("-" * 50)
        
        alt_results = alternative_attack_metrics(
            original_model, unlearned_model, bdata, 
            forgetdata, retaindata, device
        )
        
        print("Alternative analysis completed!")
        
        if EXIT_AFTER_ALTERNATIVE_ANALYSIS:
            print("Exiting after alternative analysis (EXIT_AFTER_ALTERNATIVE_ANALYSIS=True)")
            return basic_results, alt_results
    else:
        print("Skipping alternative analysis (RUN_ALTERNATIVE_ANALYSIS=False)")
    
    # ==================================================================
    # IDEAL SCENARIOS PHASE
    # ==================================================================
    
    if RUN_IDEAL_SCENARIOS:
        print(f"\n IDEAL SCENARIO TESTING")
        print("-" * 50)
        
        scenario_results = test_attack_on_ideal_scenarios(original_model, device)
        
        print("Ideal scenario testing completed!")
    else:
        print("Skipping ideal scenarios (RUN_IDEAL_SCENARIOS=False)")
    
    # ==================================================================
    # PARAMETER ANALYSIS PHASE
    # ==================================================================
    
    if RUN_PARAMETER_ANALYSIS:
        print(f"\n PHASE 6: PARAMETER CHANGE ANALYSIS")
        print("-" * 50)
        
        param_results = run_parameter_analysis(original_model, unlearned_model)
        
        print("Parameter analysis completed!")
    else:
        print("Skipping parameter analysis (RUN_PARAMETER_ANALYSIS=False)")
    
    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    
    print(f"\nALL ANALYSES COMPLETED!")
    print("="*60)
    
    if RUN_PROJECTION_ATTACK and basic_results:
        print_final_summary(basic_results)
    
    return basic_results if RUN_PROJECTION_ATTACK else None


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
# MAIN EXECUTION
# ==================================================================

if __name__ == "__main__":
    results = main()
