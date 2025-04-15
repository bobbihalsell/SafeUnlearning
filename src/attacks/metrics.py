"""Parts of this code are based on https://sudomake.ai/inception-score-explained/."""
import torch
import torch.nn.functional as F
import lpips

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()


def mse_image_space(img_batch, ref_batch):
    """
    Compute Mean Squared Error in image space for batched input.
    
    Inputs:
        img_batch: Tensor of shape [B, 3, H, W] (normalized)
        ref_batch: Tensor of shape [B, 3, H, W] (normalized)
    
    Returns:
        mse: Scalar float (mean MSE across batch)
    """

    # Compute mean squared error per image, then average over batch
    mse = F.mse_loss(img_batch, ref_batch, reduction='mean')

    return mse.item()


def lpips_batch(img_batch, ref_batch):
    """
    Compute LPIPS distance between batches of images.

    Inputs:
        gen_imgs: Tensor [B, 3, H, W], normalized with ImageNet stats
        ref_imgs: Tensor [B, 3, H, W], normalized with ImageNet stats

    Returns:
        lpips_value: float, average LPIPS over batch
    """

    # Load LPIPS model 
    lpips_fn = lpips.LPIPS(net='alex')  
    lpips_fn = lpips_fn.cuda() if torch.cuda.is_available() else lpips_fn

    # Compute LPIPS per image in batch
    with torch.no_grad():
        distances = lpips_fn(img_batch, ref_batch)

    return distances.mean().item()


def MSE_R(img_batch, ref_batch):
    """
    Compute MSE in Representation Space (MSE-R) between batches of images.

    Inputs:
        gen_imgs: Tensor [B, 3, H, W], normalized with ImageNet stats
        ref_imgs: Tensor [B, 3, H, W], normalized with ImageNet stats

    Returns:
        MSE-R: float
    """
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
