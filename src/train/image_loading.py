from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError


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
