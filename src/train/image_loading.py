from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
from typing import Union, Tuple, List, Dict
from pathlib import Path
import os


class RobustImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]

        try:
            # Open the image file
            sample = Image.open(path)
            sample = sample.convert("RGB")

            if self.transform is not None:
                sample = self.transform(sample)

        except UnidentifiedImageError:
            # Skip the corrupted image and continue to the next one
            print(f"Skipping corrupted image: {path}")
            return self.__getitem__((index + 1) % len(self.samples))
        return sample, target

    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        Override of the default ImageFolder find_classes method to literally
        transcribe folder names into labels.
        """
        classes = sorted(entry.name for entry in
                         os.scandir(directory) if
                         entry.is_dir())
        if not classes:
            raise FileNotFoundError(
                f"Couldn't find any class folder in {directory}.")
        try:
            class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
        except ValueError:
            raise RuntimeError(
                'Dataset folder names must be integers corresponding exactly '
                'to image labels. Use the dataset splitting app to prepare '
                'your dataset, or manually adjust your folder names.')
        return classes, class_to_idx
