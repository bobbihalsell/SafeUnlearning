import torch
import numpy as np
import random
from datasets import DATASETS_TO_TRANSFORM
import os
from train.image_loading import RobustImageFolder
from torch.utils.data import DataLoader


def setup_device():
    """ 
    Setup a torch device.
    Returns:
          str: The selected device type ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int, optional): The seed value to set. Defaults to 4
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior


class ConfigError(Exception):
    """
    Custom exception raised for configuration errors in YAML specifications.
    
    """
    def __init__(self, message='Configuration .YAML specification error.'):
        super().__init__(message)


def initialize_dataloaders(splits, 
                           batch_sizes, 
                           num_workers, 
                           dataset_name, 
                           dataset_save_dir):
    """ 
    Initialize dataloaders from the dataset folder in ImageFolder format.
    Args:
        splits (list): List of splits to initialize (e.g., 'retain', 'forget', 'val').
        batch_sizes (dict): Dictionary mapping splits to their respective batch sizes.
        num_workers (int): Number of workers to load data in parallel.
        dataset_name (str): Name of the dataset.
        dataset_save_dir (str): Directory where the dataset is stored.

    Returns:
        dict: A dictionary of DataLoader objects indexed by split names.
    """
    dataloaders = {}
    datasets = initialize_datasets(splits, dataset_name, dataset_save_dir)
    for dataset in datasets:
        dataloaders[dataset] = DataLoader(
            datasets[dataset],
            batch_size=batch_sizes[dataset],
            shuffle=(dataset in ['retain', 'forget']),
            num_workers=num_workers,
            pin_memory=True
        )
    return dataloaders


def initialize_datasets(splits, dataset_name, dataset_save_dir):
    """ 
    Initialize dataloaders from the dataset folder.
     Args:
        splits (list): List of dataset splits (e.g., 'retain', 'forget', 'val').
        dataset_name (str): Name of the dataset.
        dataset_save_dir (str): Directory where the dataset is stored.

    Returns:
        dict: A dictionary of datasets indexed by split names.
    
    Raises:
        Exception: If any dataset directory is missing or incorrectly formatted.
    """
    transform = DATASETS_TO_TRANSFORM[dataset_name]()

    # Load datasets for each split
    datasets = {}

    for split in splits:
        split_dir = os.path.join(dataset_save_dir, split)
        if not os.path.exists(split_dir):
            raise Exception(f'{split_dir} does not exist. '
                            f'Is the dataset in {dataset_save_dir} format?')

        dataset = RobustImageFolder(root=split_dir, transform=transform)
        datasets[split] = dataset
    return datasets
