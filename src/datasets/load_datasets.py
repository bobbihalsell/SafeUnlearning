from torch.utils.data import Subset
import torchvision.datasets as datasets
from torchvision.transforms import ToPILImage
from train.image_loading import RobustImageFolder
import numpy as np
import os
import torch
import torchvision


def _get_stratified_split(dataset: torch.utils.data.Dataset,
                          proportion: float):
    """ Get a stratified split of a dataset into 2 subsets.

    This will retain class distributions.
    """
    targets = np.array(dataset.targets)
    unique_classes = np.unique(targets)
    indices = []
    remainder_indices = []

    for cls in unique_classes:
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        num_samples = int(len(cls_indices) * proportion)

        indices.extend(cls_indices[:num_samples])
        remainder_indices.extend(cls_indices[num_samples:])

    return Subset(dataset, indices), Subset(dataset, remainder_indices)


def load_train_val_test_datasets(dataset_name: str,
                                 proportion: float,
                                 val_ratio: float,
                                 dataset_load_dir: str,
                                 dataset_save_dir: str = "",
                                 transform: torchvision.transforms = None):
    """ Load, optionally filter, and save dataset in ImageFolder format. """

    # For CIFAR datasets, download if necessary and filter
    if dataset_name == 'cifar10':
        raw_train = datasets.CIFAR10(root=dataset_load_dir,
                                     train=True,
                                     download=True,
                                     transform=transform)
        raw_test = datasets.CIFAR10(root=dataset_load_dir,
                                    train=False,
                                    download=True,
                                    transform=transform)

    elif dataset_name == 'cifar100':
        raw_train = datasets.CIFAR100(root=dataset_load_dir,
                                      train=True,
                                      download=True,
                                      transform=transform)
        raw_test = datasets.CIFAR100(root=dataset_load_dir,
                                     train=False,
                                     download=True,
                                     transform=transform)

    elif dataset_name == 'cifar5':
        raw_train = datasets.CIFAR10(root=dataset_load_dir,
                                     train=True,
                                     download=True,
                                     transform=transform)
        raw_test = datasets.CIFAR10(root=dataset_load_dir,
                                    train=False,
                                    download=True,
                                    transform=transform)

        # Filter for classes 0–4
        train_indices = [i for i, (_, label) in
                         enumerate(raw_train) if label < 5]
        test_indices = [i for i, (_, label) in
                        enumerate(raw_test) if label < 5]
        raw_train = Subset(raw_train, train_indices)
        raw_test = Subset(raw_test, test_indices)

    elif dataset_name == 'imagenet':
        # Load in the dataset for filtering by proportion
        raw_train = RobustImageFolder(
            root=os.path.join(dataset_load_dir, "train"),
            transform=transform
        )
        # If val path does not exist, then do not save a test set
        if os.path.exists(os.path.join(dataset_load_dir, "val")):
            raw_test = RobustImageFolder(
                root=os.path.join(dataset_load_dir, "val"),
                transform=transform
            )
        else:
            raw_test = None

    else:
        raise Exception(f'{dataset_name} is an unsupported dataset.')

    if proportion < 1:
        raw_train, _ = _get_stratified_split(raw_train, proportion)
        if raw_test is not None:
            raw_test, _ = _get_stratified_split(raw_test, proportion=1)

    # Perform train/val split
    raw_train_subset, raw_val_subset = _get_stratified_split(raw_train,
                                                             1 - val_ratio)

    # Save datasets to ImageFolder format
    if dataset_save_dir:
        _save_as_imagefolder(raw_train_subset,
                            root_path=dataset_save_dir,
                            name='train')
        _save_as_imagefolder(raw_val_subset,
                            root_path=dataset_save_dir,
                            name='val')
        if raw_test is not None:
            _save_as_imagefolder(raw_test,
                                root_path=dataset_save_dir,
                                name='test')

    return raw_train_subset, raw_val_subset, raw_test


def _save_as_imagefolder(dataset, root_path, name="train"):
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
