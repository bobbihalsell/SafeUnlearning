import matplotlib.pyplot as plt
import torch
from pathlib import Path
import torch
import random 
import numpy as np

class ImageSaver:
    def __init__(self, base_save_path: str = "saved_images", num_images: int = 1):
        self.base_save_path = Path(base_save_path)
        self.num_images = num_images
        self.base_save_path.mkdir(parents=True, exist_ok=True)
        self.device = setup_device() 

    def save(self, images: list, filename: str, loss: float, image_mean: list, image_std: list) -> None:
        images = images.clone().detach()
        image_mean = torch.as_tensor(image_mean, device = self.device)[:, None, None]
        image_std = torch.as_tensor(image_std, device = self.device)[:, None, None]

        images.mul_(image_std).add_(image_mean).clamp_(0, 1)

        if self.num_images == 1:
            plt.title(f"Loss: {loss:.2f}") 
            plt.imshow(images[0].permute(1, 2, 0).cpu())
            plt.axis('off')
        else:
            fig, axes = plt.subplots(1, self.num_images == 1, figsize=(40, images.shape[0]*12))
            for i, im in enumerate(images):
                axes[i].imshow(im.permute(1, 2, 0).cpu())
                axes[i].set_title(f"Loss: {loss:.2f}", fontsize=40) 

        filename = self._generate_filename(filename)
        plt.savefig(self.base_save_path / filename, bbox_inches='tight')
        print(f"Image saved at {self.base_save_path / filename}")
    
        plt.show()

    def _generate_filename(self, filename:str) -> str:
        return f"{filename}.pdf"


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
