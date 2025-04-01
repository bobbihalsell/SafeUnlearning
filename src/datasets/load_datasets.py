import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
import numpy as np


def load_cifar10_datasets():
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

    return train_dataset, test_dataset


def load_cifar5_datasets():
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

    return train_dataset, test_dataset


def load_cifar100_datasets():
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

    return train_dataset, test_dataset


def load_cifar10_datasets(proportion=1.0):
    assert 0 < proportion <= 1, "Proportion must be between 0 and 1."

    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    def stratified_subset(dataset, proportion):
        targets = np.array(dataset.targets)
        num_classes = len(set(targets))
        indices = []

        for cls in range(num_classes):
            cls_indices = np.where(targets == cls)[0]
            num_samples = int(len(cls_indices) * proportion)
            indices.extend(np.random.choice(cls_indices, num_samples, replace=False))

        return Subset(dataset, indices)
    
    if proportion<1:
        train_subset = stratified_subset(train_dataset, proportion)
        test_subset = stratified_subset(test_dataset, proportion)
        return train_subset, test_subset
    else:
        return train_dataset, test_dataset
