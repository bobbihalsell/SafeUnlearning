from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np

from munl.datasets import get_dataset_and_lengths
from munl.datasets.cifar10 import (
    get_cifar10_train_transform,
)


def generate_lira_train_tests(lira_dev_indices, num_attempts, ratio=0.5):
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


def generate_all_forgets(train_matrices, num_attempts, ratio):
    retains, forgets = [], []
    for row in train_matrices:
        retain, forget = generate_lira_train_tests(row, num_attempts, ratio)
        retains.append(retain)
        forgets.append(forget)
    retains = np.stack(retains)
    forgets = np.stack(forgets)
    return retains, forgets


def main(args):
    data_seed = args.data_seed
    val_ratio = args.val_ratio
    forget_ratio = args.forget_ratio
    num_splits = args.num_splits
    num_forgets = args.num_forgets

    # TODO: fix hardcoded dataset
    cifar10 = get_dataset_and_lengths(
        Path("datasets"),
        dataset_name="cifar10",
        transform=get_cifar10_train_transform(),
    )
    dataset, (train_len, test_len) = cifar10

    indices = np.arange((len(dataset)))
    dev_indices = indices[:train_len]
    val_len = int(train_len * val_ratio)

    lira_indices = np.random.default_rng(data_seed).permutation(dev_indices)
    lira_dev_indices = lira_indices[:-val_len]
    lira_val_indices = lira_indices[-val_len:]

    train_matrices, test_matrices = generate_lira_train_tests(
        lira_dev_indices=lira_dev_indices, num_attempts=num_splits
    )
    path = Path("artifacts/lira/splits")
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_seed", type=int, default=123)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--forget_ratio", type=float, default=0.1)
    parser.add_argument("--num_splits", type=int, default=64)
    parser.add_argument("--num_forgets", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
