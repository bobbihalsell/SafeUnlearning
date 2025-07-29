#!/usr/bin/env python3

from projection_attack import run_projection_attack
from gradient_attack import run_gradient_attack  
from activation_attack import run_activation_attack
from attack_scenario import run_realistic_projection_attack
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

RUN_PROJECTION_ATTACK = True
RUN_GRADIENT_ATTACK = True
RUN_ACTIVATION_ATTACK = True
RUN_REALISTIC_PROJECTION = True

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
    
    # Store results
    results = {}
    
    # ==================================================================
    # PROJECTION ATTACK PHASE
    # ==================================================================
    
    if RUN_PROJECTION_ATTACK:
        print(f"\n PROJECTION ATTACK")
        print("-" * 50)
        
        projection_results = run_projection_attack(
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
        
        results['projection'] = projection_results
        print("Projection attack completed!")
    
    # ==================================================================
    # GRADIENT ATTACK PHASE
    # ==================================================================
    
    if RUN_GRADIENT_ATTACK:
        print(f"\n GRADIENT ATTACK")
        print("-" * 50)
        
        gradient_results = run_gradient_attack(
            original_model=original_model,
            unlearned_model=unlearned_model,
            background_data=bdata,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            cache_dir=CACHE_DIR,
            force_recompute=FORCE_RECOMPUTE,
            verbose=VERBOSE
        )
        
        results['gradient'] = gradient_results
        print("Gradient attack completed!")
    
    # ==================================================================
    # ACTIVATION ATTACK PHASE
    # ==================================================================
    
    if RUN_ACTIVATION_ATTACK:
        print(f"\n ACTIVATION ATTACK")
        print("-" * 50)
        
        activation_results = run_activation_attack(
            original_model=original_model,
            unlearned_model=unlearned_model,
            background_data=bdata,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            cache_dir=CACHE_DIR,
            force_recompute=FORCE_RECOMPUTE,
            verbose=VERBOSE
        )
        
        results['activation'] = activation_results
        print("Activation attack completed!")
    
    # ==================================================================
    # REALISTIC PROJECTION ATTACK PHASE
    # ==================================================================
    
    if RUN_REALISTIC_PROJECTION:
        print(f"\n REALISTIC PROJECTION ATTACK")
        print("-" * 50)
        
        realistic_results = run_realistic_projection_attack(
            original_model=original_model,
            unlearned_model=unlearned_model,
            background_data=bdata,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            variance_threshold=VARIANCE_THRESHOLD,
            verbose=VERBOSE
        )
        
        results['realistic_projection'] = realistic_results
        print("Realistic projection attack completed!")
    
    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    
    print(f"\nALL ANALYSES COMPLETED!")
    print("="*60)
    
    # Print summary of all attacks
    for attack_name, attack_results in results.items():
        if 'summary' in attack_results:
            print(f"\n{attack_name.upper().replace('_', ' ')} ATTACK SUMMARY:")
            summary = attack_results['summary']
            if summary.get('success', False):
                print(f"  ✅ Attack successful: {summary.get('description', 'N/A')}")
            else:
                print(f"  ❌ Attack failed: {summary.get('description', 'N/A')}")
        elif 'test_results' in attack_results:  # For realistic projection
            print(f"\n{attack_name.upper().replace('_', ' ')} ATTACK SUMMARY:")
            summary = attack_results['summary']
            if summary.get('success', False):
                print(f"  ✅ Attack successful: {summary.get('description', 'N/A')}")
                if summary.get('privacy_risk', False):
                    print(f"  ⚠️  Privacy Risk: Background data can predict forgotten patterns!")
            else:
                print(f"  ❌ Attack failed: {summary.get('description', 'N/A')}")
                print(f"  ✅ Good Privacy: Background data cannot reveal forgotten patterns")
    
    return results


# ==================================================================
# MAIN EXECUTION
# ==================================================================

if __name__ == "__main__":
    results = main()