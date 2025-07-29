
from recover_gradients import *
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

# calculate covars
print('calculating original model background data covar...')
original_covar, _ = calculate_covar(original_model, bdata, device)
print('calculating unlearned model background data covar...')
unlearned_covar, _ = calculate_covar(unlearned_model, bdata, device)


# calculate difference
print('calculating estimated original model forget svd...')
est_forget_svd = calculate_svd_difference(original_covar, unlearned_covar, device, K)

# compare to forget set
print('calculating true forget svd...')
forget_covar, _ = calculate_covar(original_model, forgetdata, device)
forget_svd = calculate_svd(forget_covar, k=K)

# compare to retain set
print('calculating true retain svd...')
retain_covar, _ = calculate_covar(original_model, retaindata, device)
retain_svd = calculate_svd(retain_covar, k=K)

# TEST DIFFERENT PROJECTION TYPES
print("\n" + "="*60)
print("TESTING DIFFERENT PROJECTION TYPES")
print("="*60)

projection_types = ['full', 'right', 'orthogonal']  # Test the main ones

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


# from recover_gradients import *
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
# ORIGINAL_CKP = './artifacts/unlearn/torchhub_test/unlearn/neggradplus/cifar10_resnet20_42_original_001.pt'
# UNLEARNED_CKP = './artifacts/unlearn/torchhub_test/unlearn/neggradplus/cifar10_resnet20_42_unlearned_001.pt'

# K = None

# # set device
# device = setup_device()

# # load the models 
# original_importer = ImportModel(
#     load_method=LOAD_METHOD,
#     model_name=MODEL_NAME,
#     num_classes=NUM_CLASSES,
#     init_path=INIT_PATH,
#     model_ckpt_path=ORIGINAL_CKP
# )
# original_model = original_importer.model

# unlearned_importer = ImportModel(
#     load_method=LOAD_METHOD,
#     model_name=MODEL_NAME,
#     num_classes=NUM_CLASSES,
#     init_path=INIT_PATH,
#     model_ckpt_path=UNLEARNED_CKP
# )
# unlearned_model = unlearned_importer.model

# # initialise data

# dataloaders = initialize_dataloaders(
#     splits=["bset", "forget", "retain"],
#     batch_sizes={"bset": 64, "forget": 64, "retain": 64},
#     num_workers=1,
#     dataset_name=DATASET_NAME,
#     dataset_save_dir=DATASET_DIR,
# )

# bdata = dataloaders["bset"]
# forgetdata = dataloaders["forget"]
# retaindata = dataloaders["retain"]


# # calculate covars
# print('calculating original model background data covar...')
# original_covar = calculate_covar(original_model, bdata, device)
# print('calculating unlearned model background data covar...')
# unlearned_covar = calculate_covar(unlearned_model, bdata, device)

# # calculate difference
# print('calculating estimated original model forget svd...')
# est_forget_svd = calculate_svd_difference(original_covar, unlearned_covar, device, K)
# # Project original model's covar onto forget subspace
# est_projected_forget_covar = project_onto_subspace(original_covar, est_forget_svd, k=K)

# # comapare to forget set
# print('calculating true forget svd...')
# forget_covar = calculate_covar(original_model, forgetdata, device)
# forget_svd = calculate_svd(forget_covar, k=K)
# # Project original model's covar onto forget subspace
# projected_forget_covar = project_onto_subspace(original_covar, forget_svd, k=K)

# # compare the projections
# print('comparing true forget to estimated forget svds...')
# forget_results = compare_projections(est_projected_forget_covar, projected_forget_covar, k=K, plot=True, id='forget')

# # print summary
# for layer, vals in forget_results.items():
#     print(f"[{layer}] L2 diff: {vals['l2_singular_value_diff']:.4f}, "
#           f"Subspace overlap: {vals['mean_subspace_overlap']:.4f}, "
#           f"Recall: {vals['recall@k']:.4f}, "
#           f"Cos angles (mean): {vals['cos_angles'].mean():.4f}")

# # comapare to retain set
# print('calculating true retain svd...')
# retain_covar = calculate_covar(original_model, retaindata, device)
# retain_svd = calculate_svd(retain_covar, k=K)
# # Project original model's covar onto forget subspace
# projected_retain_covar = project_onto_subspace(original_covar, retain_svd, k=K)

# # compare the projections
# print('comparing true retain to estimated forget svds...')
# retain_results = compare_projections(est_projected_forget_covar, projected_retain_covar, k=K, plot=True, id='retain')

# # print summary
# for layer, vals in retain_results.items():
#     print(f"[{layer}] L2 diff: {vals['l2_singular_value_diff']:.4f}, "
#           f"Subspace overlap: {vals['mean_subspace_overlap']:.4f}, "
#           f"Recall: {vals['recall@k']:.4f}, "
#           f"Cos angles (mean): {vals['cos_angles'].mean():.4f}")
