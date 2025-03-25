import math
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List


class ImageSaver:
    def __init__(self, base_save_path: str = "saved_images", num_cols: int = 5, max_images: int = 25):
        self.base_save_path = Path(base_save_path)
        self.num_cols = num_cols
        self.max_images = max_images
        self.base_save_path.mkdir(parents=True, exist_ok=True)

    def save(self, images: List) -> None:
        max_images = min(len(images), self.max_images)
        num_rows = math.ceil(max_images / self.num_cols)

        if max_images == 1:
            plt.imshow(images[0])
            plt.axis('off')
        else:
            plt.figure(figsize=(self.num_cols * 2, num_rows * 2))
            for i in range(max_images):
                plt.subplot(num_rows, self.num_cols, i + 1)
                plt.imshow(images[i])
                plt.axis('off')

        filename = self._generate_filename()
        plt.savefig(self.base_save_path / filename, bbox_inches='tight')
        print(f"Image saved at {self.base_save_path / filename}")
    
        plt.show()

    def _generate_filename(self) -> str:
        return f"dgl_.pdf"
    

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
