import glob
import os
import random
from dataclasses import fields
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attacks.metrics import mse_image_space, psnr


class SaveImage:
    def __init__(
        self,
        attack_name: str,
        seed: str,
        experiment_name: str,
        image_mean: list = None,
        image_std: list = None,
        output_dir: str = "artifacts/reconstructed",
    ):
        """Class to save images and tensors.
        Args:
            attack_name (str): Name of the attack.
            seed (str): Seed for reproducibility.
            experiment_name (str): Name of the experiment.
            image_mean (list, optional): Mean for normalization. Defaults to None.
            image_std (list, optional): Std for normalization. Defaults to None.
            output_dir (str, optional): Directory to save images. Defaults to "artifacts/reconstructed".
        """
        self.attack_name = attack_name
        self.seed = seed
        self.experiment_name = experiment_name
        self.device = "cpu"

        # Default if not provided
        if image_mean is None:
            image_mean = [0.0]
        if image_std is None:
            image_std = [1.0]

        self.image_mean = torch.as_tensor(image_mean, device=self.device)[:, None, None]
        self.image_std = torch.as_tensor(image_std, device=self.device)[:, None, None]

        self.base_save_path = Path(output_dir)
        self.base_save_path.mkdir(parents=True, exist_ok=True)

        self.directory = f"{output_dir}/reconstruction/{attack_name}"
        os.makedirs(self.directory, exist_ok=True)  # Ensure the directory exists

    def save_png(
        self,
        images: list,
        filename: str = None,
        fig_size: tuple = None,
        n_cols: int = None,
        normalize: bool = True,
    ):
        """
        Save images as PNG files with a specified layout.
        Args:
            images (list): List of images to save.
            filename (str): Name of the file to save. Defaults to None.
            fig_size (tuple): Figure size. Defaults to None.
            n_cols (int): Number of columns. Defaults to None.
            normalize (bool): Whether to normalize the images. Defaults to True.

        Returns:
            None
        """

        if not isinstance(images, torch.Tensor):
            raise TypeError("Images should be a tensor.")

        self.num_images = len(images)

        images = images.clone().detach().to(self.device)

        if self.attack_name == 'ggl':
            x = images.detach().cpu()
            x = (x.squeeze(0) + 1) / 2.0  # Normalize from [-1, 1] to [0, 1]

            # Convert tensor to PIL image
            transform = transforms.ToPILImage()
            img_pil = transform(x.clamp(0, 1))  # Clamp to [0, 1] range
            plt.imshow(img_pil)
            plt.axis("off")

        
        else:
            clipped_images = [torch.clamp(img, 0.0, 1.0) for img in images]

            if normalize:
                clipped_images = [
                    img.mul_(self.image_std).add_(self.image_mean).clamp_(0, 1)
                    for img in clipped_images
                ]

            if self.num_images == 1:
                plt.imshow(clipped_images[0].permute(1, 2, 0).cpu())
                plt.axis("off")
            else:
                if n_cols is None:
                    max_cols = 8
                    n_cols = min(max_cols, self.num_images)
                n_rows = int(np.ceil(self.num_images / n_cols))

                _, h, w = clipped_images[0].shape
                scale = 2.5
                fig_width = (w * n_cols * scale) / 100
                fig_height = (h * n_rows * scale) / 100
                fig_size = (fig_width, fig_height)

                fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, dpi=300)
                axes = np.array(axes).flatten()

                for i, im in enumerate(clipped_images):
                    axes[i].imshow(im.permute(1, 2, 0).cpu())
                    axes[i].axis("off")

                # Hide any unused subplots
                for j in range(len(images), len(axes)):
                    axes[j].axis("off")

        if filename is None:
            filepath = os.path.join(
                self.directory,
                f"{self.attack_name}_{self.seed}_{self.experiment_name}.png",
            )
        else:
            filepath = os.path.join(self.directory, filename)



        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        print(f"Image saved at {filepath}")
        plt.show()
        

    def save_tensor(self, images: list, filepath: str = None):
        """Save images as a tensor.
        Args:
            images (list): List of images to save.
            filepath (str): Path to save the tensor.

        Returns:
            None
        """

        if filepath is None:
            filepath = os.path.join(
                self.directory,
                f"{self.attack_name}_{self.seed}_{self.experiment_name}.pt",
            )

        torch.save(images, filepath)

        print(f"Image saved as a tensor at {filepath}")

    def load_tensor(self, filepath: str):
        """Load images from a tensor file.
        Args:
            filepath (str): Path to the tensor file.
        Returns:
            list: Loaded images.
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")
        if not filepath.endswith(".pt"):
            raise ValueError(f"File {filepath} is not a .pt file.")
        return torch.load(filepath)


class ConfigError(Exception):
    def __init__(self, message="Configuration .YAML specification error."):
        super().__init__(message)


def safe_dataclass_load(dataclass_type, config_dict):
    field_names = {f.name for f in fields(dataclass_type)}
    filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
    return dataclass_type(**filtered_dict)


def calculate_metrics(img_batch, ref_batch, dataset_size, verbose=True):
    # Compute metrics
    ref_batch = ref_batch.to("cuda")
    img_batch = img_batch.to("cuda")

    # Make sure hat the range is the same for both batches
    img_batch = torch.clamp(img_batch, 0.0, 1.0)
    ref_batch = torch.clamp(ref_batch, 0.0, 1.0)

    psnr_value = psnr(img_batch, ref_batch, dataset_size, factor=1)
    mse_value = mse_image_space(img_batch, ref_batch, dataset_size)

    if verbose:
        print("\n*** Evaluation Metrics ***")

        for i, (val, idx) in enumerate(psnr_value):
            print(f"Recon image {i}: Best match is ref {idx} with PSNR {val:.6f}")
        for i, (mse, idx) in enumerate(mse_value):
            print(f"Recon image {i}: Best match is ref {idx} with MSE {mse:.6f}")

    return psnr_value, mse_value


def load_from_directory(dir_path):
    transform = transforms.ToTensor()
    images = []

    # Recursively find all image files in dir_path
    image_paths = sorted(glob.glob(os.path.join(dir_path, "**", "*.*"), recursive=True))
    image_paths = [
        p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        tensor_img = transform(img)
        images.append(tensor_img)

    if not images:
        raise RuntimeError(f"No image files found in {dir_path} or its subdirectories.")

    image_batch = torch.stack(images)
    return image_batch


def setup_device():
    """Setup a torch device.

    Returns:
        str
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior


if __name__ == "__main__":
    # Example usage
    save_image = SaveImage(
        attack_name="example_attack", seed="1234", experiment_name="example_experiment"
    )

    # Create a dummy tensor of images
    images = torch.randn(32, 3, 64, 64)

    # Save the images with custom parameters

    save_image.save_tensor(images, filepath="example_tensor.pt")

    loaded_images = save_image.load_tensor(filepath="example_tensor.pt")
    print(f"Loaded images shape: {loaded_images.shape}")

    save_image.save_png(loaded_images, normalize=True)
