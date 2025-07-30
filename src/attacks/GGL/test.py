from attacks.GGL.recover_gradients import *
from importmodel import ImportModel
from utils import setup_device, initialize_dataloaders

from pathlib import Path
import numpy as np
import torch
from torch import nn

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
LEARNING_RATE = 0.001
CACHE_DIR = './svd_cache'
FORCE_RECOMPUTE = False

TEST_LAYERS = False
TEST_PROJECTION = True


def main():
    """
    Main function with comprehensive standardized analysis using PGU-style functions.
    """
    
    print("STARTING COMPREHENSIVE UNLEARNING PRIVACY ANALYSIS (PGU STYLE)")
    
    # ==================================================================
    # SETUP PHASE
    # ==================================================================
    
    print("\n📚 SETUP PHASE")
    print("-" * 40)
    
    device = setup_device()
    print(f"Device: {device}")
    
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
    # TESTING PHASE
    # ==================================================================
    
    if TEST_LAYERS:
        print("\n🔍 TESTING LAYER NAMES WITH PGU STYLE")
        print("-" * 50)
        
        try:
            # Test layer name consistency using PGU-style functions
            subspace, param_diff, gradients = test_layer_names(
                forgetdata, forgetdata, original_model, unlearned_model, device, 
                loss_fn=nn.CrossEntropyLoss(), variance_threshold=VARIANCE_THRESHOLD
            )
            
            print("\nLayer name testing completed successfully!")
            print(f"   Subspace layers: {len(subspace)}")
            print(f"   Parameter diff layers: {len(param_diff)}")
            print(f"   Gradient layers: {len(gradients)}")
            
        except Exception as e:
            print(f"Layer name testing failed: {e}")
            import traceback
            traceback.print_exc()

    if TEST_PROJECTION:
        print("\nTESTING PROJECTION")
        print("-" * 50)
        
        try:
            result = test_projection(
                forgetdata, retaindata, original_model, unlearned_model, 
                forgetdata, LEARNING_RATE, device, 
                loss_fn=nn.CrossEntropyLoss(), variance_threshold=VARIANCE_THRESHOLD
            )
            
            print("\nProjection testing completed successfully!")
            
        except Exception as e:
            print(f"Projection testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()