from gradient_comparison import run_gradient_comparison  
from activation_comparison import run_activation_comparison  
from subspace_comparison import run_subspace_comparison  

from gradient_attack import run_gradient_attack  
from activation_attack import run_activation_attack
from subspace_attack import run_subspace_attack

from importmodel import ImportModel
from utils import setup_device, initialize_dataloaders

from pathlib import Path
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
ORIGINAL_CKP = './artifacts/unlearn/torchhub_test/unlearn/neggradplus/cifar10_resnet20_42_original_001.pt'
UNLEARNED_CKP = './artifacts/unlearn/torchhub_test/unlearn/neggradplus/cifar10_resnet20_42_unlearned_001.pt'
VERBOSE = True  

# ATTACK SETTINGS
VARIANCE_THRESHOLD = 0.85
CACHE_DIR = './svd_cache'
FORCE_RECOMPUTE = False

CREATE_COMPREHENSIVE_ANALYSIS = True

# ==================================================================
# EXECUTION CONTROL FLAGS
# ==================================================================

RUN_GRADIENT_COMPARISON = True
RUN_ACTIVATION_COMPARISON = True
RUN_SUBSPACE_COMPARISON = True

RUN_GRADIENT_ATTACK = True
RUN_ACTIVATION_ATTACK = True
RUN_SUBSPACE_ATTACK = True

# NEW: Enable comprehensive standardized analysis
CREATE_COMPREHENSIVE_ANALYSIS = True

def main():
    """
    Main function with comprehensive standardized analysis and beautiful plots.
    """
    
    print("STARTING COMPREHENSIVE UNLEARNING PRIVACY ANALYSIS")
    
    # ==================================================================
    # SETUP PHASE
    # ==================================================================
    
    print("\n📚 SETUP PHASE")
    print("-" * 40)
    
    device = setup_device()
    print(f"   ✅ Device: {device}")
    
    # Load models
    print("   📥 Loading original model...")
    original_importer = ImportModel(
        load_method=LOAD_METHOD,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        init_path=INIT_PATH,
        model_ckpt_path=ORIGINAL_CKP
    )
    original_model = original_importer.model
    
    print("   📥 Loading unlearned model...")
    unlearned_importer = ImportModel(
        load_method=LOAD_METHOD,
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        init_path=INIT_PATH,
        model_ckpt_path=UNLEARNED_CKP
    )
    unlearned_model = unlearned_importer.model
    
    # Initialize data
    print("   📊 Loading datasets...")
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
    
    print("   ✅ Setup completed successfully!")
    
    # Store results for comprehensive analysis
    attack_results = {}

    # ==================================================================
    # GRADIENT COMPARISON PHASE
    # ==================================================================

    if RUN_GRADIENT_COMPARISON:
        print(f"\nGRADIENT COMPARISON PHASE")
        print("-" * 50)
        print("Testing how gradient patterns differ between forget and retain data")
        
        gradient_comparison_data = run_gradient_comparison(
            original_model=original_model,
            unlearned_model=unlearned_model,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            verbose=VERBOSE
        )

        attack_results['gradient_comparison'] = gradient_comparison_data

        print("Gradient comparison analysis completed!")

    # ==================================================================
    # ACTIVATION COMPARISON PHASE
    # ==================================================================

    if RUN_ACTIVATION_COMPARISON:
        print(f"\nACTIVATION COMPARISON PHASE")
        print("-" * 50)
        print("Testing how activation patterns differ between forget and retain data")
        
        activation_comparison_data = run_activation_comparison(
            original_model=original_model,
            unlearned_model=unlearned_model,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            verbose=VERBOSE
        )

        attack_results['activation_comparison'] = activation_comparison_data


        print("Activation comparison analysis completed!")

    # ==================================================================
    # SUBSPACE COMPARISON PHASE
    # ==================================================================

    if RUN_SUBSPACE_COMPARISON:
        print(f"\nSUBSPACE COMPARISON PHASE")
        print("-" * 50)
        print("Testing how subspace patterns differ between forget and retain data")
        
        subspace_comparison_data = run_subspace_comparison(
            original_model=original_model,
            unlearned_model=unlearned_model,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            verbose=VERBOSE
        )

        attack_results['subspace_comparison'] = subspace_comparison_data

        print("Subspace comparison analysis completed!")
    
    # ==================================================================
    # GRADIENT ATTACK PHASE
    # ==================================================================
    
    if RUN_GRADIENT_ATTACK:
        print(f"\nGRADIENT ATTACK PHASE")
        print("-" * 50)
        print("Testing how gradient patterns differ between forget and retain data")
        
        gradient_attack_data = run_gradient_attack(
            original_model=original_model,
            unlearned_model=unlearned_model,
            background_data=bdata,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            verbose=VERBOSE
        )

        attack_results['gradient'] = gradient_attack_data
        
        print("Gradient attack analysis completed!")
    
    # ==================================================================
    # ACTIVATION ATTACK PHASE
    # ==================================================================
    
    if RUN_ACTIVATION_ATTACK:
        print(f"\nACTIVATION ATTACK PHASE")
        print("-" * 50)
        print("Testing how activation patterns differ on most important layers")
        
        activation_attack_data = run_activation_attack(
            original_model=original_model,
            unlearned_model=unlearned_model,
            background_data=bdata,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            verbose=VERBOSE
        )

        attack_results['activation'] = activation_attack_data

        print("Activation attack analysis completed!")
    
    # ==================================================================
    # PROJECTION ATTACK PHASE
    # ==================================================================
    
    if RUN_SUBSPACE_ATTACK:
        print(f"\n🔍 PROJECTION ATTACK PHASE")
        print("-" * 50)
        print("📋 Testing if background data can predict forgotten data (privacy risk)")
        
        subspace_attack_data = run_subspace_attack(
            original_model=original_model,
            unlearned_model=unlearned_model,
            background_data=bdata,
            forget_data=forgetdata,
            retain_data=retaindata,
            device=device,
            variance_threshold=VARIANCE_THRESHOLD,
            verbose=VERBOSE
        )

        attack_results['projection'] = subspace_attack_data

        print("Subspace attack analysis completed!")
    
    # ==================================================================
    # COMPREHENSIVE STANDARDIZED ANALYSIS
    # ==================================================================
    
    if CREATE_COMPREHENSIVE_ANALYSIS and len(attack_results) > 0:
        print(f"\nCOMPREHENSIVE ANALYSIS PHASE")
        print("="*60)
        
    # ==================================================================
    # ENHANCED FINAL SUMMARY
    # ==================================================================

    print(f"\nFINAL UNLEARNING PRIVACY ANALYSIS RESULTS")
    print("=" * 90)

    # List of all analyses you ran
    attack_names = {
        'gradient': 'GRADIENT ATTACK',
        'activation': 'ACTIVATION ATTACK',
        'projection': 'PROJECTION ATTACK',
        'gradient_comparison': 'GRADIENT COMPARISON',
        'activation_comparison': 'ACTIVATION COMPARISON',
        'subspace_comparison': 'SUBSPACE COMPARISON'
    }

    for attack_key, attack_label in attack_names.items():
        if attack_key in attack_results:
            result = attack_results[attack_key]
            print(f"\n{attack_label}")
            if attack_key.endswith('comparison'):
                print(f"   analysis: {result['analysis']}")
                print(f"   plot_directory: {result['plot_directory']}")

            else:
                print(f"   attack_results: {result['attack_results']}")
                print(f"   analysis: {result['analysis']}")
                print(f"   plot_directory: {result['plot_directory']}")

        else:
            print(f"\n{attack_label}: NOT EXECUTED")

    print("\nSummary table generated successfully!")


if __name__ == "__main__":
    main()
