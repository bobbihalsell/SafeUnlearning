from recover import *
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
VARIANCE_THRESHOLD = 1.0

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


test_fixed_covariance(original_model, bdata, device)

fixed_gradient_recovery_pipeline(original_model, unlearned_model, bdata, device, 
                                   retained_var_threshold=1.0, loss_fn=None, debug=True)
exit()


# def debug_full_pipeline(original_model, forgetdata, device, param_name='conv1.weight'):
#     """
#     Debug the full pipeline: covariance -> SVD -> projection
#     """
#     print(f"\n{'='*80}")
#     print(f"DEBUGGING FULL PIPELINE FOR {param_name}")
#     print(f"{'='*80}")
    
#     # Step 1: Full covariance calculation (not just 3 samples)
#     print("Step 1: Full covariance calculation...")
#     forget_covar, _ = calculate_gradient_covariance(original_model, forgetdata, device)
    
#     if param_name in forget_covar:
#         covar_matrix = forget_covar[param_name]
#         print(f"Full covariance for {param_name}: {covar_matrix.shape}")
#     else:
#         print(f"ERROR: {param_name} not found in covariance results!")
#         print(f"Available keys: {list(forget_covar.keys())[:10]}")
#         return
    
#     # Step 2: SVD calculation
#     print(f"\nStep 2: SVD calculation...")
#     forget_svd = calculate_svd(forget_covar, k=None)
    
#     if param_name in forget_svd:
#         svd_result = forget_svd[param_name]
#         if svd_result is not None:
#             U = svd_result['U']
#             S = svd_result['S']
#             print(f"SVD for {param_name}:")
#             print(f"  U shape: {U.shape}")
#             print(f"  S shape: {S.shape}")
#         else:
#             print(f"SVD result for {param_name} is None!")
#     else:
#         print(f"ERROR: {param_name} not found in SVD results!")
#         return
    
#     # Step 3: Layer name mapping
#     print(f"\nStep 3: Layer name mapping...")
#     forget_gradients = get_gradients(original_model, forgetdata, device)
#     mapped_svd = map_svd_to_gradient_keys(forget_svd, forget_gradients)
    
#     if param_name in mapped_svd:
#         mapped_result = mapped_svd[param_name]
#         if mapped_result is not None:
#             U_mapped = mapped_result['U'] 
#             print(f"Mapped SVD for {param_name}:")
#             print(f"  U_mapped shape: {U_mapped.shape}")
#             print(f"  Same as original? {torch.equal(U, U_mapped)}")
#         else:
#             print(f"Mapped result for {param_name} is None!")
#     else:
#         print(f"ERROR: {param_name} not found in mapped results!")
#         print(f"Available mapped keys: {list(mapped_svd.keys())[:10]}")
#         return
    
#     # Step 4: Projection matrix creation
#     print(f"\nStep 4: Projection matrix creation...")
#     P = U_mapped @ U_mapped.T
#     print(f"Projection matrix P shape: {P.shape}")
    
#     # Step 5: Test projection compatibility
#     print(f"\nStep 5: Gradient compatibility test...")
#     if param_name in forget_gradients:
#         grad_tensor = forget_gradients[param_name]
#         print(f"Target gradient shape: {grad_tensor.shape}")
#         print(f"Gradient last dim: {grad_tensor.shape[-1]}")
#         print(f"Projection first dim: {P.shape[0]}")
#         print(f"COMPATIBLE: {grad_tensor.shape[-1] == P.shape[0]}")
        
#         # Show the dimension mismatch
#         if grad_tensor.shape[-1] != P.shape[0]:
#             print(f"DIMENSION MISMATCH CONFIRMED:")
#             print(f"  Expected P shape: [{grad_tensor.shape[-1]}, {grad_tensor.shape[-1]}]")
#             print(f"  Actual P shape: {P.shape}")
#             print(f"  Difference: {P.shape[0]} - {grad_tensor.shape[-1]} = {P.shape[0] - grad_tensor.shape[-1]}")
    
#     return covar_matrix, svd_result, mapped_result, P

# # Run the debug
# debug_full_pipeline(original_model, forgetdata, device, 'conv1.weight')

# print(f"\n{'='*80}")
# print("STOPPING HERE FOR EXTENDED DEBUG")
# print(f"{'='*80}")
# exit()

# Calculate gradients (for later projection)
print('='*60)
print('LOADING/CALCULATING MODEL COVARIANCES')
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
    org_conv_fea_dict, org_linear_fea_dict = get_feature_dict(original_model)
    original_covar = calculate_covar(original_model, bdata, device,
                                    conv_fea_dict=org_conv_fea_dict, linear_fea_dict=org_linear_fea_dict)
    save_to_cache(original_covar, original_covar_path)

# Unlearned model background data covar
if not FORCE_RECOMPUTE and (unlearned_covar := load_from_cache(unlearned_covar_path)) is not None:
    print('✓ Unlearned model background data covar loaded from cache')
else:
    print('calculating unlearned model background data covar...')
    unl_conv_fea_dict, unl_linear_fea_dict = get_feature_dict(original_model)
    unlearned_covar = calculate_covar(unlearned_model, bdata, device,
                                    conv_fea_dict=unl_conv_fea_dict, linear_fea_dict=unl_linear_fea_dict)
    save_to_cache(unlearned_covar, unlearned_covar_path)

# Forget set covar
if not FORCE_RECOMPUTE and (forget_covar := load_from_cache(forget_covar_path)) is not None:
    print('✓ Forget covar loaded from cache')
else:
    print('calculating forget covar...')
    forget_covar = calculate_covar(original_model, forgetdata, device,
                                    conv_fea_dict=org_conv_fea_dict, linear_fea_dict=org_linear_fea_dict)
    save_to_cache(forget_covar, forget_covar_path)

# Retain set covar
if not FORCE_RECOMPUTE and (retain_covar := load_from_cache(retain_covar_path)) is not None:
    print('✓ Retain covar loaded from cache')
else:
    print('calculating retain covar...')
    retain_covar = calculate_covar(original_model, retaindata, device,
                                    conv_fea_dict=org_conv_fea_dict, linear_fea_dict=org_linear_fea_dict)
    save_to_cache(retain_covar, retain_covar_path)


print('='*60)
print('LOADING/CALCULATING DATA SVDS')
print('='*60)

est_forget_svd_path = get_cache_path(CACHE_DIR, "est_forget_gradients")
forget_svd_path = get_cache_path(CACHE_DIR, "forget_gradients")
retain_svd_path = get_cache_path(CACHE_DIR, "retain_gradients")

# Estimated forget svd
if not FORCE_RECOMPUTE and (est_forget_svd := load_from_cache(est_forget_svd_path)) is not None:
    print('✓ Estimated forget svd loaded from cache')
else:
    print('calculating estimated forget svd...')
    est_forget_svd = calculate_svd_difference(original_covar, unlearned_covar, device, k=K)
    save_to_cache(est_forget_svd, est_forget_svd_path)

# True forget svd
if not FORCE_RECOMPUTE and (forget_svd := load_from_cache(forget_svd_path)) is not None:
    print('✓ Estimated forget svd loaded from cache')
else:
    print('calculating true forget svd...')
    forget_svd = calculate_svd(forget_covar, k=K)
    save_to_cache(forget_svd, forget_svd_path)

# True retain svd
if not FORCE_RECOMPUTE and (retain_svd := load_from_cache(retain_svd_path)) is not None:
    print('✓ Estimated forget svd loaded from cache')
else:
    print('calculating true retain svd...')
    retain_svd = calculate_svd(retain_covar, k=K)
    save_to_cache(retain_svd, retain_svd_path)

print('='*60)
print('LOADING/CALCULATING PROJECTION MATRICES')
print('='*60)

est_forget_P_path = get_cache_path(CACHE_DIR, "est_forget_projection_matrix")
forget_P_path = get_cache_path(CACHE_DIR, "forget_projection_matrix")
retain_P_path = get_cache_path(CACHE_DIR, "retain_projection_matrix")

# Estimated forget projection matrix
if not FORCE_RECOMPUTE and (est_forget_P := load_from_cache(est_forget_P_path)) is not None:
    print('✓ Estimated forget projection matrix loaded from cache')
else:
    print('calculating estimated forget projection matrix...')
    est_forget_P = get_projection_matrix(est_forget_svd, retained_var_threshold=VARIANCE_THRESHOLD, device=device)
    save_to_cache(est_forget_P, est_forget_P_path)

# Estimated forget svd
if not FORCE_RECOMPUTE and (forget_P := load_from_cache(forget_P_path)) is not None:
    print('✓ Forget projection matrix loaded from cache')
else:
    print('calculating forget projection matrix...')
    forget_P = get_projection_matrix(forget_svd, retained_var_threshold=VARIANCE_THRESHOLD, device=device)
    save_to_cache(forget_P, forget_P_path)

# Estimated retain svd
if not FORCE_RECOMPUTE and (retain_P := load_from_cache(retain_P_path)) is not None:
    print('✓ Retain projection matrix loaded from cache')
else:
    print('calculating retain projection matrix...')
    retain_P = get_projection_matrix(retain_svd, retained_var_threshold=VARIANCE_THRESHOLD, device=device)
    save_to_cache(retain_P, retain_P_path)


print('='*60)
print('LOADING/CALCULATING GRADIENTS')
print('='*60)
    

bdata_gradients_path = get_cache_path(CACHE_DIR, "bdata_gradients")
# forget_gradients_path = get_cache_path(CACHE_DIR, "forget_gradients")
# retain_gradients_path = get_cache_path(CACHE_DIR, "retain_gradients")

# Original model on background data gradients
if not FORCE_RECOMPUTE and (gradients := load_from_cache(bdata_gradients_path)) is not None:
    print('✓ Original model background data gradients loaded from cache')
else:
    print('calculating original model background data gradients...')
    gradients = get_gradients(original_model, bdata, device)
    save_to_cache(gradients, bdata_gradients_path)

# # Original model on forget data gradients
# if not FORCE_RECOMPUTE and (forget_gradients := load_from_cache(forget_gradients_path)) is not None:
#     print('✓ Original model forget data gradients loaded from cache')
# else:
#     print('calculating original model forget data gradients...')
#     forget_gradients = get_gradients(original_model, forgetdata, device)
#     save_to_cache(forget_gradients, forget_gradients_path)

# # Original model on retain data gradients
# if not FORCE_RECOMPUTE and (retain_gradients := load_from_cache(retain_gradients_path)) is not None:
#     print('✓ Original model retain data gradients loaded from cache')
# else:
#     print('calculating original model retain data gradients...')
#     retain_gradients = get_gradients(original_model, retaindata, device)
#     save_to_cache(retain_gradients, retain_gradients_path)


print('='*60)
print('PROJECTING GRADIENTS')
print('='*60)


print("Projecting gradients on estimated forget space...")
est_forget_projected_gradients = project_gradients(gradients, est_forget_P)

print("Projecting gradients on forget space...")
forget_projected_gradients = project_gradients(gradients, forget_P)

print("Projecting gradients on retain space...")
retain_projected_gradients = project_gradients(gradients, retain_P)


# Compare projections
print('\n' + '='*60)
print('COMPARING PROJECTIONS')
print('='*60)

print('Comparing estimated forget vs true forget projections (should be SIMILAR):')
forget_comparison = compare_projections(est_forget_projected_gradients, forget_projected_gradients, 
                                       plot=True, id='est_vs_true_forget')

print('\nComparing estimated forget vs retain projections (should be DIFFERENT):')
retain_comparison = compare_projections(est_forget_projected_gradients, retain_projected_gradients, 
                                       plot=True, id='est_forget_vs_retain')

# Print summary
print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print_all_metrics(forget_comparison, "ESTIMATED vs TRUE FORGET (should be similar)")
print_all_metrics(retain_comparison, "ESTIMATED FORGET vs RETAIN (should be different)")
