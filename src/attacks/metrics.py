"""Parts of this code are based on https://sudomake.ai/inception-score-explained/."""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.transforms.functional import resize
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage import data, img_as_float
import matplotlib.pyplot as plt
import lpips


def apply_resizing(img_batch, ref_batch, dataset_size):
    """
    Resize both image and reference batches to a common target size based on the dataset.

    Args:
        img_batch (torch.Tensor): Batch of images to be resized.
        ref_batch (torch.Tensor): Batch of reference images to be resized.
        dataset_size (tuple): Target size for resizing (height, width).

    Returns:
        torch.Tensor, torch.Tensor: Resized image and reference batches.
    """
    # Define default size based on dataset
    target_size = (dataset_size[1], dataset_size[2])

    # Resize both image batches
    resized_img_batch = [resize(img, target_size) for img in img_batch]
    resized_ref_batch = [resize(ref_img, target_size) for ref_img in ref_batch]

    # Return the resized batches
    return torch.stack(resized_img_batch), torch.stack(resized_ref_batch)


def apply_normalization(img_batch, ref_batch, dataset_mean, dataset_std):
    """
    Normalize image and reference batches based on dataset-specific mean and standard deviation.

    Args:
        img_batch (torch.Tensor): Batch of images to be normalized.
        ref_batch (torch.Tensor): Batch of reference images to be normalized.
        dataset_mean (list or tuple): Mean values for normalization.
        dataset_std (list or tuple): Standard deviation values for normalization.

    Returns:
        torch.Tensor, torch.Tensor: Normalized image and reference batches.
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
    Compute the Peak Signal-to-Noise Ratio (PSNR) for each image in img_batch against the most similar
    reference image in ref_batch based on the highest PSNR value.

    Args:
        img_batch (torch.Tensor): Batch of images to evaluate.
        ref_batch (torch.Tensor): Batch of reference images to compare against.
        dataset_size (tuple): Size to which the images should be resized.
        factor (float, optional): Scaling factor for PSNR calculation. Defaults to 1.0.

    Returns:
        list of tuples: Each tuple contains the best PSNR and the index of the reference image.
    """
    img_batch, ref_batch = apply_resizing(img_batch, ref_batch, dataset_size)

    def get_psnr(img_in, img_ref, factor):
        mse = ((img_in - img_ref) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor**2 / mse)
        elif not torch.isfinite(mse):
            return img_in.new_tensor(float("nan"))
        else:
            return img_in.new_tensor(float("inf"))

    img_batch = img_batch.detach()
    ref_batch = ref_batch.detach()
    results = []

    for i, img in enumerate(
        img_batch
    ):  # Iterate over each image in the reconstructed batch
        best_psnr = float("-inf")
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
    Compute the Mean Squared Error (MSE) for each image in img_batch against the most similar
    reference image in ref_batch based on the lowest MSE value.

    Args:
        img_batch (torch.Tensor): Batch of images to evaluate.
        ref_batch (torch.Tensor): Batch of reference images to compare against.
        dataset_size (tuple): Size to which the images should be resized.

    Returns:
        list of tuples: Each tuple contains the best MSE and the index of the reference image.
    """
    img_batch, ref_batch = apply_resizing(img_batch, ref_batch, dataset_size)

    img_batch = img_batch.detach()
    ref_batch = ref_batch.detach()
    results = []

    for i, img in enumerate(img_batch):
        best_mse = float("inf")
        best_idx = -1
        for j, ref_img in enumerate(ref_batch):
            mse_val = ((img - ref_img) ** 2).mean().item()
            if mse_val < best_mse:
                best_mse = mse_val
                best_idx = j
        results.append((best_mse, best_idx))

    return results

def ssim_image_space(img_batch, ref_batch, dataset_size):
    """
    For each image in img_batch, find the reference image in ref_batch with the highest SSIM.
    Returns: list of (best_ssim, best_index) tuples, one per image in img_batch.
    """
    img_batch, ref_batch = apply_resizing(img_batch, ref_batch, dataset_size)

    # detach and move to CPU, then convert to NumPy
    img_batch = img_batch.detach().cpu().numpy().copy()
    ref_batch = ref_batch.detach().cpu().numpy().copy()

    results = []

    for i, img in enumerate(img_batch):
        best_ssim = float('-inf')
        best_idx = -1
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        for j, ref_img in enumerate(ref_batch):
            if ref_img.ndim == 3 and ref_img.shape[0] == 3:
                ref_img = np.transpose(ref_img, (1, 2, 0))
            data_range = img.max() - img.min()
            if data_range == 0:
                data_range = 1.0

            # If your images are multichannel, enable multichannel
            multichannel = img.ndim == 3 and img.shape[2] == 3


            ssim_val = ssim(img, ref_img, multichannel=multichannel, data_range=data_range, channel_axis=-1)
            if ssim_val > best_ssim:
                best_ssim = ssim_val
                best_idx = j
                plt.imsave(f'img_{i}_ref_{j}_ssim_{ssim_val:.4f}.png', ref_img, cmap='gray')
        results.append((best_ssim, best_idx))

    return results

def lpips_image_space(img_batch, ref_batch, dataset_size):
    loss_fn_alex = lpips.LPIPS(net='alex')

    img_batch, ref_batch = apply_resizing(img_batch, ref_batch, dataset_size)

    img_batch = img_batch.detach().cpu().float()
    ref_batch = ref_batch.detach().cpu().float()

    # LPIPS expects inputs in range [-1, 1]
    img_batch = (img_batch * 2) - 1
    ref_batch = (ref_batch * 2) - 1

    results = []

    for i, img in enumerate(img_batch):
        best_lpips = float('inf')
        best_idx = -1

        for j, ref_img in enumerate(ref_batch):
            
            lpips_val = loss_fn_alex(img, ref_img).item()

            if lpips_val < best_lpips:
                best_lpips = lpips_val
                best_idx = j
                plt.imsave(f'img_{i}_ref_{j}_lpips_{lpips_val:.4f}.png', ref_img, cmap='gray')
        results.append((best_lpips, best_idx))
    
    return results

def test():
    img = img_as_float(data.camera())

    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    rng = np.random.default_rng()
    noise[rng.random(size=noise.shape) > 0.5] *= -1

    img_noise = img + noise
    img_const = img + abs(noise)
    img_flipped = np.fliplr(img).copy()

    random_img = img_as_float(data.astronaut())
    
    # Save the images
    plt.imsave('img_flipped.png', img_flipped, cmap='gray')
    plt.imsave('img_noise.png', img_noise, cmap='gray')
    plt.imsave('img_const.png', img_const, cmap='gray')
    plt.imsave('random_img.png', random_img, cmap='gray')

    img_batch = [torch.tensor(img_flipped), torch.tensor(img_noise), torch.tensor(img_const)]
    # print(f"Image batch shapes: {[img.shape for img in img_batch]}")
    ref_batch = [torch.tensor(img_noise), torch.tensor(img_const)]

    dataset_size = (1,512, 512)  # Example size for the images

    ssim_results = ssim_image_space(img_batch, ref_batch, dataset_size)
    llpips_results = lpips_image_space(img_batch, ref_batch, dataset_size)
    for i, (ssim_val, idx) in enumerate(ssim_results):
        print(f"Image {i}: Best SSIM = {ssim_val:.4f}, Best Index = {idx}")
    for i, (lpips_val, idx) in enumerate(llpips_results):
        print(f"Image {i}: Best LPIPS = {lpips_val:.4f}, Best Index = {idx}")

if __name__ == "__main__":
    test()