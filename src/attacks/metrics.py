"""Parts of this code are based on https://sudomake.ai/inception-score-explained/."""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.transforms.functional import resize


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
