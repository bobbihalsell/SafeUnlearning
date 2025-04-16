"""Parts of this code are based on https://sudomake.ai/inception-score-explained/."""
import torch
import torch.nn.functional as F
import lpips

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import torch.nn.functional as F  
from torchvision.transforms.functional import resize



def apply_resizing(img_batch, ref_batch, dataset_size):
    """
    Resize both img_batch and ref_batch to a common size based on the dataset.
    """
    # Define default size based on dataset
    target_size = (dataset_size[1], dataset_size[2])

    # Resize both image batches
    resized_img_batch = [resize(img, target_size) for img in img_batch]
    resized_ref_batch = [resize(ref_img, target_size) for ref_img in ref_batch]

    # Return the resized batches
    return torch.stack(resized_img_batch), torch.stack(resized_ref_batch)


import torch
from torchvision import transforms

def apply_normalization(img_batch, ref_batch, dataset_mean, dataset_std):
    """
    Normalize img_batch and ref_batch according to dataset_name.
    """
    mean = dataset_mean
    std = dataset_std

    # Define normalization transform
    normalize = transforms.Normalize(mean=mean, std=std)

    # Normalize both image batches
    normalized_img_batch = [normalize(img) for img in img_batch]
    normalized_ref_batch = [normalize(ref_img) for ref_img in ref_batch]

    # Return the normalized batches
    return torch.stack(normalized_img_batch), torch.stack(normalized_ref_batch)


def psnr(img_batch, ref_batch, dataset_size, factor=1.0):
    """
    For each image in img_batch, find the reference image in ref_batch with the highest PSNR.
    Returns: list of (best_psnr, best_index) tuples, one per image in img_batch.
    """
    img_batch, ref_batch = apply_resizing(img_batch, ref_batch, dataset_size)
    def get_psnr(img_in, img_ref, factor):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_in.new_tensor(float('nan'))
        else:
            return img_in.new_tensor(float('inf'))
        
    

    img_batch = img_batch.detach()
    ref_batch = ref_batch.detach()
    results = []

    for i, img in enumerate(img_batch):  # Iterate over each image in the reconstructed batch
        best_psnr = float('-inf')
        best_idx = -1
        for j, ref_img in enumerate(ref_batch):  # Compare with all reference images
            psnr_val = get_psnr(img, ref_img, factor)
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_idx = j
        results.append((best_psnr.item(), best_idx))

    return results 


def mse_image_space(img_batch, ref_batch, dataset_size):
    """
    Compute Mean Squared Error in image space for batched input.
    
    Inputs:
        img_batch: Tensor of shape [B, 3, H, W] (normalized)
        ref_batch: Tensor of shape [B, 3, H, W] (normalized)
    
    Returns:
        mse: Scalar float (mean MSE across batch)
    """
    img_batch, ref_batch = apply_resizing(img_batch, ref_batch, dataset_size)

    # Compute mean squared error per image, then average over batch
    mse = F.mse_loss(img_batch, ref_batch, reduction='mean')

    return mse.item()


def lpips_batch(img_batch, ref_batch, dataset_name='cifar10'):
    """
    Compute LPIPS distance between batches of images.

    Inputs:
        gen_imgs: Tensor [B, 3, H, W], normalized with ImageNet stats
        ref_imgs: Tensor [B, 3, H, W], normalized with ImageNet stats

    Returns:
        lpips_value: float, average LPIPS over batch
    """
    img_batch, ref_batch = apply_resizing(img_batch, ref_batch, dataset_name)

    # Load LPIPS model 
    lpips_fn = lpips.LPIPS(net='alex')  
    lpips_fn = lpips_fn.cuda() if torch.cuda.is_available() else lpips_fn

    # Compute LPIPS per image in batch
    with torch.no_grad():
        distances = lpips_fn(img_batch, ref_batch)

    return distances.mean().item()


def MSE_R(img_batch, ref_batch, dataset_name='imagenet'):
    """
    Compute MSE in Representation Space (MSE-R) between batches of images.

    Inputs:
        gen_imgs: Tensor [B, 3, H, W], normalized with ImageNet stats
        ref_imgs: Tensor [B, 3, H, W], normalized with ImageNet stats

    Returns:
        MSE-R: float
    """
    img_batch, ref_batch = apply_resizing(img_batch, ref_batch, dataset_name)
    #img_batch, ref_batch = apply_normalization(img_batch, ref_batch, dataset_name)
    # Load pretrained model
    resnet = models.resnet18(pretrained=True)
    resnet.eval()

    # Get feature extractor (before final classification layer)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
    feature_extractor = feature_extractor.to('cuda')

    # Extract feature vectors
    with torch.no_grad():
        target_feat = feature_extractor(ref_batch).squeeze()        
        recon_feat = feature_extractor(img_batch).squeeze()

    # Compute MSE in representation space
    mse_loss = nn.MSELoss()
    mse_r = mse_loss(target_feat, recon_feat)

    return mse_r.item()
