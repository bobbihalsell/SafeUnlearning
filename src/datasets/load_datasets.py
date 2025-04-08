from torch.utils.data import Subset
import torchvision.datasets as datasets
import numpy as np
import os
import torch
from torchvision.transforms import ToPILImage


def _get_stratified_split(dataset: torch.utils.data.Dataset,
                          proportion: float):
    """ Get a stratified split of a dataset into 2 subsets. 

    This will retain class distributions.
    """
    targets = np.array(dataset.targets)
    num_classes = len(set(targets))
    indices = []
    remainder_indices = []

    for cls in range(num_classes):
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
                                 dataset_save_dir: str):
    """ Load, optionally filter, and save dataset in ImageFolder format. """

    # For CIFAR datasets, download if necessary and filter
    if dataset_name == 'cifar10':
        raw_train = datasets.CIFAR10(root=dataset_load_dir,
                                     train=True,
                                     download=True,
                                     transform=None)
        raw_test = datasets.CIFAR10(root=dataset_load_dir,
                                    train=False,
                                    download=True,
                                    transform=None)

    elif dataset_name == 'cifar100':
        raw_train = datasets.CIFAR100(root=dataset_load_dir,
                                      train=True,
                                      download=True,
                                      transform=None)
        raw_test = datasets.CIFAR100(root=dataset_load_dir,
                                     train=False,
                                     download=True,
                                     transform=None)

    elif dataset_name == 'cifar5':
        raw_train = datasets.CIFAR10(root=dataset_load_dir,
                                     train=True,
                                     download=True,
                                     transform=None)
        raw_test = datasets.CIFAR10(root=dataset_load_dir,
                                    train=False,
                                    download=True,
                                    transform=None)

        # Filter for classes 0–4
        train_indices = [i for i, (_, label) in
                         enumerate(raw_train) if label < 5]
        test_indices = [i for i, (_, label) in
                        enumerate(raw_test) if label < 5]
        raw_train = Subset(raw_train, train_indices)
        raw_test = Subset(raw_test, test_indices)

    elif dataset_name == 'imagenet':
        # Load in the dataset for filtering by proportion
        raw_train = datasets.ImageFolder(
            root=os.path.join(dataset_load_dir, "train"),
            transform=None
        )
        # TODO custom dataset handling. val should be optional, if path does not exist, then do not save a test set
        if os.path.exists(os.path.join(dataset_load_dir, "val")):
            raw_test = datasets.ImageFolder(
                root=os.path.join(dataset_load_dir, "val"),
                transform=None
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

    return None


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
