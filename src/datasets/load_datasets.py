import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
import numpy as np
import os
import requests
import tarfile

def stratified_subset(dataset, proportion):
        targets = np.array(dataset.targets)
        num_classes = len(set(targets))
        indices = []

        for cls in range(num_classes):
            cls_indices = np.where(targets == cls)[0]
            num_samples = int(len(cls_indices) * proportion)
            indices.extend(np.random.choice(cls_indices, num_samples, replace=False))

        return Subset(dataset, indices)


def load_cifar10_datasets(proportion=1.0):
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize CIFAR images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True,
                                     download=True, transform=transform)
    # train_dataset = Subset(train_dataset, list(range(500)))
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                    download=True, transform=transform)
    # test_dataset = Subset(test_dataset, list(range(500)))

    if proportion<1:
        train_dataset = stratified_subset(train_dataset, proportion)
        test_dataset = stratified_subset(test_dataset, proportion)
    return train_dataset, test_dataset


def load_cifar5_datasets(proportion=1.0):
    """
    Load a subset of CIFAR-10 containing only the first 5 classes (0-4).
    """
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    
    # Load full CIFAR-10 datasets
    full_train_dataset = datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=transform)
    full_test_dataset = datasets.CIFAR10(root="./data", train=False,
                                        download=True, transform=transform)
    
    # Filter to keep only classes 0-4 (first 5 classes)
    train_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label < 5]
    test_indices = [i for i, (_, label) in enumerate(full_test_dataset) if label < 5]
    
    # Limit to 500 samples
    train_indices = train_indices[:500]
    test_indices = test_indices[:500]
    
    # Create subsets
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)

    if proportion<1:
        train_dataset = stratified_subset(train_dataset, proportion)
        test_dataset = stratified_subset(test_dataset, proportion)
    return train_dataset, test_dataset


def load_cifar100_datasets(proportion=1.0):
    """
    Load CIFAR-100 dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    
    # Load CIFAR-100 datasets
    train_dataset = datasets.CIFAR100(root="./data", train=True,
                                      download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root="./data", train=False,
                                     download=True, transform=transform)
    
    # Limit to 500 samples
    train_dataset = Subset(train_dataset, list(range(500)))
    test_dataset = Subset(test_dataset, list(range(500)))

    if proportion<1:
        train_dataset = stratified_subset(train_dataset, proportion)
        test_dataset = stratified_subset(test_dataset, proportion)
    return train_dataset, test_dataset


imagenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

def download_dataset_from_web(url, save_dir, proportion=1.0, transform=imagenet_transform):
    """
    Download, extract and load the dataset (train and test).

    Args:
    - url: The URL to download the dataset.
    - save_dir: Directory to save the extracted dataset.
    - dataset_name: Name of the dataset (default 'imagenet', modify if using others).
    
    Returns:
    - train_loader, test_loader: DataLoader objects for training and testing datasets.
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
        print(f"Extraction complete.")
    else:
        print("The downloaded file is not a valid tar file.")
        return None, None
    
    # Load the dataset from extracted files (assuming ImageNet structure)
    train_dir = os.path.join(save_dir, 'train')
    test_dir = os.path.join(save_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Train or test directory missing in {save_dir}. Please verify the structure.")
        return None, None
        
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    if proportion<1:
        train_dataset = stratified_subset(train_dataset, proportion)
        test_dataset = stratified_subset(test_dataset, proportion)
    return train_dataset, test_dataset

