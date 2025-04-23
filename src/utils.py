import torch
import numpy as np
import random
from datasets import DATASETS_TO_TRANSFORM
import os
from train.image_loading import RobustImageFolder
from torch.utils.data import DataLoader


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


def set_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior


class ConfigError(Exception):
    def __init__(self, message='Configuration .YAML specification error.'):
        super().__init__(message)


def initialize_dataloaders(splits, 
                           batch_sizes, 
                           num_workers, 
                           dataset_name, 
                           dataset_save_dir):
    """ Initialize dataloaders from the dataset folder in ImageFolder format.
    Returns a dictionary of dataloaders for retain, forget, and val splits.
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
    Returns a dictionary of dataloaders for retain, forget, and val splits.
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
