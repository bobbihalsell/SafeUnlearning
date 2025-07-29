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
FORCE_RECOMPUTE = False  # Set to True to force recomputation

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

# TEST DIFFERENT PROJECTION TYPES
print("\n" + "="*60)
print("TESTING DIFFERENT PROJECTION TYPES")
print("="*60)

projection_types = ['full', 'orthogonal']  # Test the main ones

for proj_type in projection_types:
    print(f"\n--- Testing {proj_type} projection ---")
    
    # Project original model's covar onto different subspaces
    est_projected_forget_covar = project_onto_subspace(original_covar, est_forget_svd, k=K, projection_type=proj_type)
    projected_forget_covar = project_onto_subspace(original_covar, forget_svd, k=K, projection_type=proj_type)
    projected_retain_covar = project_onto_subspace(original_covar, retain_svd, k=K, projection_type=proj_type)
    
    # Compare the projections
    print(f'comparing true forget to estimated forget svds with {proj_type} projection...')
    forget_results = compare_projections(est_projected_forget_covar, projected_forget_covar, k=K, plot=False, id=f'forget_{proj_type}')
    
    print(f'comparing true retain to estimated forget svds with {proj_type} projection...')
    retain_results = compare_projections(est_projected_forget_covar, projected_retain_covar, k=K, plot=False, id=f'retain_{proj_type}')
    
    # Print summary for first few layers
    print(f"\n{proj_type.upper()} PROJECTION RESULTS:")
    print("Estimated vs True Forget (should be SIMILAR):")
    for layer in list(forget_results.keys())[:3]:  # Show first 3 layers
        vals = forget_results[layer]
        print(f"  [{layer}] L2 diff: {vals['l2_singular_value_diff']:.4f}, "
              f"Overlap: {vals['mean_subspace_overlap']:.4f}, "
              f"Recall: {vals['recall@k']:.4f}")
    
    print("Estimated vs True Retain (should be DIFFERENT):")
    for layer in list(retain_results.keys())[:3]:  # Show first 3 layers
        vals = retain_results[layer]
        print(f"  [{layer}] L2 diff: {vals['l2_singular_value_diff']:.4f}, "
              f"Overlap: {vals['mean_subspace_overlap']:.4f}, "
              f"Recall: {vals['recall@k']:.4f}")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print("Look for:")
print("- HIGH overlap/recall between estimated and true forget")
print("- LOW overlap/recall between estimated and true retain")
print("- 'orthogonal' projection might work best (closest to paper)")
print(f"\nCache directory: {CACHE_DIR}")
print("Set FORCE_RECOMPUTE=True to recalculate everything")


for layer in original_covar.keys():
    diff = torch.norm(original_covar[layer] - unlearned_covar[layer])
    orig_norm = torch.norm(original_covar[layer])
    rel_diff = diff / (orig_norm + 1e-8)
    print(f"{layer}: Relative difference = {rel_diff:.6f}")


print()
print()
for layer in original_covar.keys():
    diff = torch.norm(forget_covar[layer] - retain_covar[layer])
    orig_norm = torch.norm(forget_covar[layer])
    rel_diff = diff / (orig_norm + 1e-8)
    print(f"{layer}: Relative difference = {rel_diff:.6f}")


print()
print()
for layer in est_projected_forget_covar:
    norm_est = torch.norm(est_projected_forget_covar[layer]).item()
    norm_forget = torch.norm(projected_forget_covar[layer]).item()
    norm_retain = torch.norm(projected_retain_covar[layer]).item()
    print(f"[{layer}] Norms -- Estimated Forget: {norm_est:.4f}, True Forget: {norm_forget:.4f}, True Retain: {norm_retain:.4f}")

print()
print()
def inspect_projection_setup(covar_target, svd_basis, k=None):
    """
    Diagnostic function to check covariance and SVD basis sanity.
    """
    import torch

    for layer in covar_target:
        if layer not in svd_basis:
            print(f"[{layer}] SVD basis missing. Skipping.")
            continue

        C_target = covar_target[layer]
        U = svd_basis[layer]['U']

        trace_C = torch.trace(C_target).item()
        norm_C = torch.norm(C_target).item()
        rank_C = torch.linalg.matrix_rank(C_target).item()

        print(f"\nLayer: {layer}")
        print(f"  C_target trace: {trace_C:.6e}")
        print(f"  C_target norm: {norm_C:.6e}")
        print(f"  C_target rank: {rank_C}")

        if k is not None:
            U = U[:, :k]

        shape_U = U.shape
        norm_U = torch.norm(U).item()

        print(f"  U shape: {shape_U}")
        print(f"  U norm: {norm_U:.6e}")

        # Compute projection matrix P
        if U.numel() == 0:
            print("  WARNING: U is empty. Projection will be zero.")
            continue

        P = U @ U.T
        norm_P = torch.norm(P).item()

        print(f"  Projection matrix P norm: {norm_P:.6e}")
print()
# print('original_covar, est_forget_svd')
# inspect_projection_setup(original_covar, est_forget_svd, k=K)
# print()
# print('original_covar, est_forget_svd')
# inspect_projection_setup(original_covar, forget_svd, k=K)
# print()
# print('original_covar, est_forget_svd')
# inspect_projection_setup(original_covar, retain_svd, k=K)


print('checking svd')
for layer in forget_svd:
    U_f = forget_svd[layer]['U']
    U_r = retain_svd[layer]['U']
    cos_sim = torch.norm(U_f.T @ U_r) / (torch.norm(U_f) * torch.norm(U_r))
    print(f"[{layer}] U cosine similarity: {cos_sim.item():.4f}")


