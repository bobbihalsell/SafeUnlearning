from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray as Array
from omegaconf import DictConfig

from datasets.load_datasets import load_train_val_test_datasets


def generate_lira_train_tests(
    lira_dev_indices: Array, num_attempts: int, ratio: float = 0.5
) -> Tuple[Array, Array]:
    test_size = int(len(lira_dev_indices) * ratio)
    train_size = len(lira_dev_indices) - test_size
    indices = np.zeros(shape=(num_attempts, len(lira_dev_indices)), dtype=int)

    for attempt_ndx in range(num_attempts):
        indices[attempt_ndx] = np.random.default_rng(attempt_ndx).permutation(
            deepcopy(lira_dev_indices)
        )
    train_indices, test_indices = indices[:, :train_size], indices[:, train_size:]

    # We verify that the indices row by row are different
    for train_row, test_row in zip(train_indices, test_indices):
        assert len(set(train_row) & set(test_row)) == 0

    return train_indices, test_indices


def generate_all_forgets(
    train_matrices: Array, num_attempts: int, ratio: int
) -> Tuple[Array, Array]:
    retains, forgets = [], []
    for row in train_matrices:
        retain, forget = generate_lira_train_tests(row, num_attempts, ratio)
        retains.append(retain)
        forgets.append(forget)
    retains = np.stack(retains)
    forgets = np.stack(forgets)
    return retains, forgets


def run(config: DictConfig, root: Path) -> None:
    dataset = config.dataset.name
    data_seed = config.seed
    val_ratio = config.dataset.val_ratio
    forget_ratio = config.dataset.forget_ratio
    num_splits = config.attack.cfg.num_splits
    num_forgets = config.attack.cfg.num_forgets
    save_dir = config.dataset.save_path

    train, _, test = load_train_val_test_datasets(dataset, 1, val_ratio, save_dir, save_dir)
    train_len, test_len = len(train), len(test)

    indices = np.arange(train_len + test_len)
    dev_indices = indices[:train_len]
    val_len = int(train_len * val_ratio)

    lira_indices = np.random.default_rng(data_seed).permutation(dev_indices)
    lira_dev_indices = lira_indices[:-val_len]
    lira_val_indices = lira_indices[-val_len:]

    train_matrices, test_matrices = generate_lira_train_tests(
        lira_dev_indices=lira_dev_indices, num_attempts=num_splits
    )
    path = Path(root / "splits")
    if not path.exists():
        path.mkdir(parents=True)
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
