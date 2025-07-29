from recover_gradients import *
from importmodel import ImportModel
from utils import setup_device, initialize_dataloaders
import pickle
import os

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

# CACHE SETTINGS
CACHE_DIR = './svd_cache'
FORCE_RECOMPUTE = True  # Set to True to force recomputation

# Helper functions for caching
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

# set device
device = setup_device()

# load the models 
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

# initialise data
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

# Calculate or load covariance matrices
print('='*60)
print('LOADING/CALCULATING COVARIANCE MATRICES')
print('='*60)

original_covar_path = get_cache_path(CACHE_DIR, "original_covar")
unlearned_covar_path = get_cache_path(CACHE_DIR, "unlearned_covar")
forget_covar_path = get_cache_path(CACHE_DIR, "forget_covar")
retain_covar_path = get_cache_path(CACHE_DIR, "retain_covar")

# Original model background data covar
if not FORCE_RECOMPUTE and (original_covar := load_from_cache(original_covar_path)) is not None:
    print('✓ Original model background data covar loaded from cache')
else:
    print('calculating original model background data covar...')
    original_covar, _ = calculate_covar(original_model, bdata, device)
    save_to_cache(original_covar, original_covar_path)

# Unlearned model background data covar
if not FORCE_RECOMPUTE and (unlearned_covar := load_from_cache(unlearned_covar_path)) is not None:
    print('✓ Unlearned model background data covar loaded from cache')
else:
    print('calculating unlearned model background data covar...')
    unlearned_covar, _ = calculate_covar(unlearned_model, bdata, device)
    save_to_cache(unlearned_covar, unlearned_covar_path)

# Forget set covar
if not FORCE_RECOMPUTE and (forget_covar := load_from_cache(forget_covar_path)) is not None:
    print('✓ Forget covar loaded from cache')
else:
    print('calculating forget covar...')
    forget_covar, _ = calculate_covar(original_model, forgetdata, device)
    save_to_cache(forget_covar, forget_covar_path)

# Retain set covar
if not FORCE_RECOMPUTE and (retain_covar := load_from_cache(retain_covar_path)) is not None:
    print('✓ Retain covar loaded from cache')
else:
    print('calculating retain covar...')
    retain_covar, _ = calculate_covar(original_model, retaindata, device)
    save_to_cache(retain_covar, retain_covar_path)

# Calculate or load SVD results
print('\n' + '='*60)
print('LOADING/CALCULATING SVD RESULTS')
print('='*60)

est_forget_svd_path = get_cache_path(CACHE_DIR, "est_forget_svd")
forget_svd_path = get_cache_path(CACHE_DIR, "forget_svd")
retain_svd_path = get_cache_path(CACHE_DIR, "retain_svd")

# Estimated forget SVD
if not FORCE_RECOMPUTE and (est_forget_svd := load_from_cache(est_forget_svd_path)) is not None:
    print('✓ Estimated forget SVD loaded from cache')
else:
    print('calculating estimated original model forget svd...')
    est_forget_svd = calculate_svd_difference(original_covar, unlearned_covar, device, K)
    save_to_cache(est_forget_svd, est_forget_svd_path)

# True forget SVD
if not FORCE_RECOMPUTE and (forget_svd := load_from_cache(forget_svd_path)) is not None:
    print('✓ True forget SVD loaded from cache')
else:
    print('calculating true forget svd...')
    forget_svd = calculate_svd(forget_covar, k=K)
    save_to_cache(forget_svd, forget_svd_path)

# True retain SVD
if not FORCE_RECOMPUTE and (retain_svd := load_from_cache(retain_svd_path)) is not None:
    print('✓ True retain SVD loaded from cache')
else:
    print('calculating true retain svd...')
    retain_svd = calculate_svd(retain_covar, k=K)
    save_to_cache(retain_svd, retain_svd_path)


# Calculate forget data gradients on the models

original_forget_gradients_path = get_cache_path(CACHE_DIR, "original_forget_gradients")
unlearned_forget_gradients_path = get_cache_path(CACHE_DIR, "unlearned_forget_gradients")

if not FORCE_RECOMPUTE and (original_forget_gradients := load_from_cache(original_forget_gradients_path)) is not None:
    print('✓ Original model forget data gradienst loaded from cache')
else:
    print('calculating original model forget data gradients...')
    original_forget_gradients = get_gradients(original_model, forgetdata, device)
    save_to_cache(original_forget_gradients, original_forget_gradients_path)

if not FORCE_RECOMPUTE and (unlearned_forget_gradients := load_from_cache(unlearned_forget_gradients_path)) is not None:
    print('✓ Original model forget data gradienst loaded from cache')
else:
    print('calculating original model forget data gradients...')
    unlearned_forget_gradients = get_gradients(original_model, forgetdata, device)
    save_to_cache(unlearned_forget_gradients, unlearned_forget_gradients_path)

print()
print('TESTING ORIGINAL MODEL')
print()
print('estimated forget svd')
est_forget_projection = project_onto_subspace(original_forget_gradients, est_forget_svd, k=None)

print()
print('true forget svd')
forget_projection = project_onto_subspace(original_forget_gradients, forget_svd, k=None)

print()
print('true retain svd')
retain_projection = project_onto_subspace(original_forget_gradients, retain_svd, k=None)

print()
print('comparing est_forget_projection and forget_projection')
compare_projections(est_forget_projection, forget_projection)

print()
print('comparing est_forget_projection and retain_projection')
compare_projections(est_forget_projection, forget_projection)
