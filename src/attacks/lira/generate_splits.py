import os
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray as Array

from lira_utils import load_train_test_datasets


def generate_lira_train_tests(
    lira_dev_indices: Array, 
    num_attempts: int, 
    ratio: float = 0.5
) -> Tuple[Array, Array]:
    """Generates train and test splits for LiRA.

    Args:
        lira_dev_indices: Array of indices.
        num_attempts: Number of different train-test splits to generate.
        ratio: Ratio of test set size to total size. Defaults to 0.5.

    Returns:
        Tuple:
            - train_indices: Array of shape (num_attempts, train_size) with training indices
            - test_indices: Array of shape (num_attempts, test_size) with test indices
    """
    test_size = int(len(lira_dev_indices) * ratio)
    train_size = len(lira_dev_indices) - test_size
    indices = np.zeros(shape=(num_attempts, len(lira_dev_indices)), dtype=int)

    for attempt_ndx in range(num_attempts):
        indices[attempt_ndx] = np.random.default_rng(attempt_ndx).permutation(
            deepcopy(lira_dev_indices)
        )
    train_indices = indices[:, :train_size]
    test_indices = indices[:, train_size:]

    # Verify that indices are disjoint between train and test sets
    for train_row, test_row in zip(train_indices, test_indices):
        assert len(set(train_row) & set(test_row)) == 0

    return train_indices, test_indices


def generate_all_forgets(
    train_matrices: Array, 
    num_attempts: int, 
    ratio: float
) -> Tuple[Array, Array]:
    """Generates retain and forget splits from training matrices.

    Args:
        train_matrices: Array of training indices to generate splits from.
        num_attempts: Number of different retain-forget splits to generate.
        ratio: Ratio of forget set size to total size.

    Returns:
        Tuple:
            - retains: Array of retain set indices
            - forgets: Array of forget set indices
    """
    retains, forgets = [], []
    for row in train_matrices:
        retain, forget = generate_lira_train_tests(row, num_attempts, ratio)
        retains.append(retain)
        forgets.append(forget)
    retains_array = np.stack(retains)
    forgets_array = np.stack(forgets)
    return retains_array, forgets_array


def generate_splits(
    dataset_name: str,
    forget_ratio: float,
    val_ratio: float,
    num_splits: int,
    num_forgets: int,
    save_path: str,
    output_dir: Path,
    seed: int,
) -> None:
    """Generates and saves retain, forget and validation splits for LiRA.

    Args:
        dataset_name: Name of the dataset to use.
        forget_ratio: Ratio of samples to forget in each split.
        val_ratio: Ratio of validation set size to total training size.
        num_splits: Number of different train-test splits to generate.
        num_forgets: Number of different retain-forget splits to generate.
        save_path: Path to load the dataset.
        output_dir: Directory to save the generated splits.
        seed: Random seed for reproducibility.

    Returns:
        None. Splits are saved to disk at {output_dir}/{unlearner}/splits/.
    """
    path = output_dir / "splits"
    path.mkdir(parents=True, exist_ok=True)

    if os.listdir(path):
        print("Splits are already generated. Skipping.")
        return

    train, test = load_train_test_datasets(dataset_name, save_path)
    train_len, test_len = len(train), len(test)

    indices = np.arange(train_len + test_len)
    dev_indices = indices[:train_len]
    val_len = int(train_len * val_ratio)

    lira_indices = np.random.default_rng(seed).permutation(dev_indices)
    lira_dev_indices = lira_indices[:-val_len]
    lira_val_indices = lira_indices[-val_len:]

    train_matrices, test_matrices = generate_lira_train_tests(
        lira_dev_indices=lira_dev_indices, num_attempts=num_splits
    )

    np.save(path / "train_matrices.npy", train_matrices)
    np.save(path / "val_matrices.npy", lira_val_indices)
    np.save(path / "test_matrices.npy", test_matrices)

    retains, forgets = generate_all_forgets(
        train_matrices, num_attempts=num_forgets, ratio=forget_ratio
    )
    assert retains.shape[0] == num_splits

    for split_ndx in range(num_splits):
        split_path = path / str(split_ndx)
        if not split_path.exists():
            split_path.mkdir(parents=True)
        np.save(split_path / "retains.npy", retains[split_ndx])
        np.save(split_path / "forgets.npy", forgets[split_ndx])
