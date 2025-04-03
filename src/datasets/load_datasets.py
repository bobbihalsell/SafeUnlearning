from torch.utils.data import Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets.cifar10 import (get_cifar10_train_transform,
                              get_cifar10_test_transform)
from datasets.cifar100 import (get_cifar100_train_transform,
                               get_cifar100_test_transform)
from datasets.imagenet import (get_imagenet_train_transform,
                               get_imagenet_test_transform)
import numpy as np
import os
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
                 dataset_save_path: str):
    """ Load and transform a train and test dataset.

    Transforms for the dataset must be pre-provided in the code somewhere.

    Args:
        dataset_name (str): The name of the dataset-of-origin of the images
        proportion (float): The proportion of the dataset that will be used
        dataset_save_path (str): The path at which the dataset is saved.
            If the dataset is from ImageNet, use
            the parent of the train and val directories.
    Returns:
        train_dataset, test_dataset (Tuple[Dataset])
    """
    # Select the correct transform
    if dataset_name == 'cifar10':
        train_transform = get_cifar10_train_transform()
        test_transform = get_cifar10_test_transform()
        train_dataset = datasets.CIFAR10(root=dataset_save_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dataset_save_path, train=False, download=True, transform=test_transform)

    elif dataset_name == 'cifar100':
        train_transform = get_cifar100_train_transform()
        test_transform = get_cifar100_test_transform()
        train_dataset = datasets.CIFAR100(root=dataset_save_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=dataset_save_path, train=False, download=True, transform=test_transform)

    elif dataset_name == 'cifar5':
        train_transform = get_cifar10_train_transform()
        test_transform = get_cifar10_test_transform()
        full_train_dataset = datasets.CIFAR10(root=dataset_save_path, train=True, download=True, transform=train_transform)
        full_test_dataset = datasets.CIFAR10(root=dataset_save_path, train=False, download=True, transform=test_transform)

        # Filter classes to keep only 0-4
        train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label < 5]
        test_indices = [i for i, (_, label) in enumerate(full_test_dataset) if label < 5]

        train_dataset = Subset(full_train_dataset, train_indices)
        test_dataset = Subset(full_test_dataset, test_indices)

    elif dataset_name == 'imagenet':
        train_transform = get_imagenet_train_transform()
        test_transform = get_imagenet_test_transform()
        # Dataset directory structure must follow the structure required by ImageFolder
        train_dataset_save_path = os.path.join(dataset_save_path, "train")
        test_dataset_save_path = os.path.join(dataset_save_path, "val")
        train_dataset = datasets.ImageFolder(root=train_dataset_save_path,
                                             transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_dataset_save_path,
                                            transform=test_transform)

    else:
        # Note: All subsequent custom datasets must follow ImageNet structure
        raise Exception(f'{dataset_name} is an unsupported dataset.')

    # Apply proportional subsampling if needed
    if proportion < 1:
        train_dataset = _get_stratified_subset(train_dataset,
                                               proportion)
        test_dataset = _get_stratified_subset(test_dataset,
                                              proportion)

    return train_dataset, test_dataset


def download_dataset_from_web(url, save_dir):
    """
    Download a dataset from the web.

    Args:
        url: The URL to download the dataset.
        save_dir: Directory to save the extracted dataset.

    Returns:
        None
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the filename
    filename = url.split("/")[-1]
    file_path = os.path.join(save_dir, filename)

    # Download the dataset
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Download complete: {file_path}")
    else:
        print(f"{filename} already downloaded.")

    # Extract the tar file
    if tarfile.is_tarfile(file_path):
        print(f"Extracting {filename}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=save_dir)
        print("Extraction complete.")
    else:
        raise Exception("The downloaded file is not a valid tar file.")

    # Verify the dataset has correct structure
    train_dir = os.path.join(save_dir, 'train')
    test_dir = os.path.join(save_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise Exception(f"Train or test directory missing in {save_dir}."
                        " Please verify the structure.")
