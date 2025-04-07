import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from typing import Optional
import os


class SaveImage:
    def __init__(self, 
                 attack_name: str,
                 seed: str,
                 experiment_name: str,
                 image_mean: Optional[list] = None, 
                 image_std: Optional[list] = None, 
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
        self.device = 'cpu'
    
        # Default if not provided
        if image_mean is None:
            image_mean = [0.0]  
        if image_std is None:
            image_std = [1.0] 

        self.image_mean = torch.as_tensor(image_mean, device = self.device)[:, None, None]
        self.image_std = torch.as_tensor(image_std, device = self.device)[:, None, None]

        self.base_save_path = Path(output_dir)
        self.base_save_path.mkdir(parents=True, exist_ok=True)

        self.directory = f'{output_dir}/reconstruction/{attack_name}'
        os.makedirs(self.directory, exist_ok=True)  # Ensure the directory exists





    def save_png(self, 
                 images: list, 
                 filepath: Optional[str] = None, 
                 fig_size: Optional[tuple] = None,
                 n_cols: Optional[int] = None,
                 normalize: Optional[bool] = True
                 ): 
        """
        Save images as PNG files with a specified layout.
        Args:
            images (list): List of images to save.
            filepath (str, optional): Path to save the image. Defaults to None.
            fig_size (tuple, optional): Figure size. Defaults to None.
            n_cols (int, optional): Number of columns. Defaults to None.
            normalize (bool, optional): Whether to normalize the images. Defaults to True.
        
        Returns:
            None
        """
        self.num_images = len(images)

        images = images.clone().detach()

        if normalize:
            images.mul_(self.image_std).add_(self.image_mean).clamp_(0, 1) 


        if self.num_images == 1:
            plt.imshow(images[0].permute(1, 2, 0).cpu())
            plt.axis('off')
        else:
            if n_cols is None:
                n_cols = int(np.ceil(self.num_images / 2))
            n_rows = int(np.ceil(self.num_images / n_cols))

            if fig_size is None:
                _, h, w = images[0].shape
                scale = 1.5
                fig_width = (w * n_cols * scale) / 100
                fig_height = (h * n_rows * scale) / 100
                fig_size = (fig_width, fig_height)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size,
                                    dpi=300)
            for i, im in enumerate(images):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].imshow(im.permute(1, 2, 0).cpu())
                axes[row, col].axis('off')
                if i >= n_rows * n_cols:
                    break
            # Hide any unused subplots
            for j in range(i + 1, n_rows * n_cols):
                row = j // n_cols
                col = j % n_cols
                axes[row, col].axis('off')

        if filepath is None:
            filepath = os.path.join(self.directory, f'{self.attack_name}_{self.seed}_{self.experiment_name}.png')

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Image saved at {filepath}")
        plt.show()



    def save_tensor(self, images: list, 
                    filepath: str):
        """Save images as a tensor.
        Args:
            images (list): List of images to save.
            filepath (str): Path to save the tensor.

        Returns:
            None
        """
        
        if filepath is None:
            filepath = os.path.join(self.directory, f'{self.attack_name}_{self.seed}_{self.experiment_name}.pt')

        torch.save(images, filepath)

        print(f"Image saved as a tensor at {filepath}")


    def load_tensor(self, 
                    filepath: str):
        """Load images from a tensor file.
        Args:
            filepath (str): Path to the tensor file.
        Returns:
            list: Loaded images.
        """
        
        return torch.load(filepath)
    

def setup_device():
    """ Setup a torch device.

    Returns:
        str
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    
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
    save_image = SaveImage(attack_name="example_attack", 
                           seed="1234", 
                           experiment_name="example_experiment")
    
    # Create a dummy tensor of images
    images = torch.randn(50, 3, 64, 64)  

    # Save the images with custom parameters

    save_image.save_tensor(images,
                           filepath="example_tensor.pt")
    
    loaded_images = save_image.load_tensor(filepath="example_tensor.pt")    
    print(f"Loaded images shape: {loaded_images.shape}")

    save_image.save_png(loaded_images, 
                         n_cols=10, 
                         normalise=True)