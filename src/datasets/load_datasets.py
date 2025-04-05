from torch.utils.data import Subset
import torchvision.datasets as datasets
from datasets.cifar10 import (get_cifar10_train_transform,
                              get_cifar10_test_transform)
from datasets.cifar100 import (get_cifar100_train_transform,
                               get_cifar100_test_transform)
from datasets.imagenet import (get_imagenet_train_transform,
                               get_imagenet_test_transform)
import numpy as np
import os
import torch
from torchvision.transforms import ToPILImage
import requests
import tarfile


def _get_stratified_subset(dataset,
                           proportion: float):
    """ Get a stratified subset of the dataset, retaining class distributions."""
    targets = np.array(dataset.targets)
    num_classes = len(set(targets))
    indices = []

    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        num_samples = int(len(cls_indices) * proportion)
        indices.extend(np.random.choice(cls_indices,
                                        num_samples,
                                        replace=False))

    return Subset(dataset, indices)


def load_dataset(dataset_name: str,
                 proportion: float,
                 dataset_load_dir: str,
                 dataset_save_dir: str):
    """ Load, optionally filter, and save dataset in ImageFolder format. """

    # Load raw dataset without transforms to save clean images
    if dataset_name == 'cifar10':
        raw_train = datasets.CIFAR10(root=dataset_load_dir, train=True, download=True, transform=None)
        raw_test = datasets.CIFAR10(root=dataset_load_dir, train=False, download=True, transform=None)

    elif dataset_name == 'cifar100':
        raw_train = datasets.CIFAR100(root=dataset_load_dir, train=True, download=True, transform=None)
        raw_test = datasets.CIFAR100(root=dataset_load_dir, train=False, download=True, transform=None)

    elif dataset_name == 'cifar5':
        raw_train = datasets.CIFAR10(root=dataset_load_dir, train=True, download=True, transform=None)
        raw_test = datasets.CIFAR10(root=dataset_load_dir, train=False, download=True, transform=None)

        # Filter for classes 0–4
        train_indices = [i for i, (_, label) in enumerate(raw_train) if label < 5]
        test_indices = [i for i, (_, label) in enumerate(raw_test) if label < 5]
        raw_train = Subset(raw_train, train_indices)
        raw_test = Subset(raw_test, test_indices)

    elif dataset_name == 'imagenet':
        # Just load and return directly for ImageNet
        train_transform = get_imagenet_test_transform()
        test_transform = get_imagenet_test_transform()

        train_dataset = datasets.ImageFolder(
            root=os.path.join(dataset_load_dir, "train"),
            transform=train_transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(dataset_load_dir, "val"),
            transform=test_transform
        )
        return train_dataset, test_dataset

    else:
        raise Exception(f'{dataset_name} is an unsupported dataset.')

    if proportion < 1:
        raw_train = _get_stratified_subset(raw_train, proportion)
        raw_test = _get_stratified_subset(raw_test, proportion=1)  # keep full test set

    # Save to ImageFolder format
    save_as_imagefolder(raw_train, root_path=dataset_save_dir, name='train')
    save_as_imagefolder(raw_test, root_path=dataset_save_dir, name='test')

    return None


def save_as_imagefolder(dataset, root_path, name="train"):
    """
    Converts a dataset into ImageFolder-style format.
    Saves images under: root_path/name/class_x/*.png
    """
    to_pil = ToPILImage()
    for idx, (image, label) in enumerate(dataset):
        class_dir = os.path.join(root_path, name, str(label))
        os.makedirs(class_dir, exist_ok=True)

        # If image is a tensor, convert to PIL
        if isinstance(image, torch.Tensor):
            image = to_pil(image)

        image.save(os.path.join(class_dir, f"{idx}.png"))
