import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from typing import Optional


class SaveImage:
    def __init__(self, 
                 image_mean: Optional[list] = None, 
                 image_std: Optional[list] = None, 
                 base_save_path: str = "saved_images"
                 ):
        
        # Default if not provided
        if image_mean is None:
            image_mean = [0.0]  
        if image_std is None:
            image_std = [1.0] 

        self.image_mean = torch.as_tensor(image_mean, device = self.device)[:, None, None]
        self.image_std = torch.as_tensor(image_std, device = self.device)[:, None, None]

        self.base_save_path = Path(base_save_path)
        self.base_save_path.mkdir(parents=True, exist_ok=True)

        self.device = setup_device() 


    def save_png(self, 
                 images: list, 
                 filename: Optional[str] = None, 
                 fig_size: Optional[tuple] = None,
                 n_cols: Optional[int] = None,
                 n_rows: Optional[int] = None, 
                 normalise: Optional[bool] = True
                 ): 
        self.num_images = len(images)

        images = images.clone().detach()

        if normalise:
            images.mul_(self.image_std).add_(self.image_mean).clamp_(0, 1) 

        if fig_size is None:
            fig_size = (40, self.num_images*12)

        if self.num_images == 1:
            # plt.title(f"Loss: {loss:.2f}") 
            plt.imshow(images[0].permute(1, 2, 0).cpu())
            plt.axis('off')
        else:
            fig, axes = plt.subplots(1, self.num_images, figsize=(40, self.num_images*12))
            for i, im in enumerate(images):
                axes[i].imshow(im.permute(1, 2, 0).cpu())
                # axes[i].set_title(f"Loss: {loss:.2f}", fontsize=40) #TODO: optional annotations

        if filename is None:
            filename = f'{}'#TODO: follow naming convention with attack, seed, exp num

        plt.savefig(self.base_save_path / filename, bbox_inches='tight')

        print(f"Image saved at {self.base_save_path / filename}")
    
        plt.show()



    def save_tensor(self, images: list, 
                    filename: str):
        
        if filename is None:
            filename = f'{}'
        
        torch.save(images, self.base_save_path / filename)

        print(f"Image saved at {self.base_save_path / filename}")


    def load_tensor(self, 
                    filepath: str):
        
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


#test with 1, 5, 10, 64 imgs
