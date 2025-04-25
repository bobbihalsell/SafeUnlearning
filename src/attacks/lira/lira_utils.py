from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray as Array
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision
import timm

from datasets.load_datasets import load_train_val_test_datasets
from datasets import DATASETS_TO_TRAIN_TRANSFORM, DATASETS_TO_TRANSFORM


def get_retain_forget_val_indices(
    lira_path: Path,
    split_ndx: int,
    forget_ndx: int
) -> Tuple[Array, Array, Array]:
    """Retrieves indices for retain, forget, and validation sets from stored splits.

    Args:
        lira_path: Path to the directory containing splits.
        split_ndx: Index of the split to use.
        forget_ndx: Index of the forget set to use.

    Returns:
        Tuple:
            - retain_indices: Array of indices for the retain set
            - forget_indices: Array of indices for the forget set
            - val_indices: Array of indices for the validation set

    Raises:
        AssertionError: If there is any overlap between retain, forget, and validation sets.
    """
    retains = np.load(lira_path / str(split_ndx) / "retains.npy")
    forgets = np.load(lira_path / str(split_ndx) / "forgets.npy")
    vals = np.load(lira_path / "val_matrices.npy")

    retain_indices = retains[forget_ndx]
    forget_indices = forgets[forget_ndx]
    val_indices = vals

    assert set(retain_indices) & set(forget_indices) == set()
    assert set(retain_indices) & set(val_indices) == set()
    assert set(forget_indices) & set(val_indices) == set()
    return retain_indices, forget_indices, val_indices


def get_loaders_from_indices(
    dataset_name: str,
    dataset_save_dir: str,
    indices: List[Array],
    batch_sizes: Dict[str, int],
    num_workers: int = 1,
) -> Dict[str, DataLoader]:
    """Creates DataLoaders for retain, forget, and validation sets based on provided indices.

    Args:
        dataset_name: Name of the dataset to load.
        dataset_save_dir: Directory where the dataset is saved.
        indices: List containing arrays of indices for [retain, forget, val] sets.
        batch_sizes: Dictionary mapping split names to batch sizes.
            Must contain keys: 'retain', 'forget', 'val'.
        num_workers: Number of workers for data loading. Defaults to 1.

    Returns:
        Dict: Dictionary containing DataLoaders for each split:
            - 'retain': DataLoader for retain set (with augmentation)
            - 'forget': DataLoader for forget set (with augmentation)
            - 'val': DataLoader for validation set (without augmentation)
    """
    train_transform = DATASETS_TO_TRAIN_TRANSFORM[dataset_name]()
    test_transform = DATASETS_TO_TRANSFORM[dataset_name]()

    train, _, test = load_train_val_test_datasets(
        dataset_name, 1, 0, dataset_save_dir, "", train_transform
    )
    dataset_aug = ConcatDataset([train, test])

    train, _, test = load_train_val_test_datasets(
        dataset_name, 1, 0, dataset_save_dir, "", test_transform
    )
    dataset_non_aug = ConcatDataset([train, test])

    retain_indices, forget_indices, val_indices = indices
    loaders = {}
    for split_name, indices in zip(
        ["retain", "forget", "val"], 
        [retain_indices, forget_indices, val_indices]
    ):
        if split_name == "val":
            dataset = dataset_non_aug
            shuffle = False
        else:
            dataset = dataset_aug
            shuffle = True

        data_set = Subset(dataset, indices)
        loader = DataLoader(
            data_set,
            batch_size=batch_sizes[split_name],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )
        loaders[split_name] = loader

    return loaders
