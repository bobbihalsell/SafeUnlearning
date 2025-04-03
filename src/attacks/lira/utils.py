from pathlib import Path
import numpy as np


def get_retain_forget_val_test_indices(
    lira_path: Path, split_ndx: int, forget_ndx: int
):
    retains = np.load(lira_path / str(split_ndx) / "retains.npy")
    forgets = np.load(lira_path / str(split_ndx) / "forgets.npy")
    vals = np.load(lira_path / "val_matrices.npy")
    tests = np.load(lira_path / "test_matrices.npy")

    retain_indices = retains[forget_ndx]
    forget_indices = forgets[forget_ndx]
    val_indices = vals
    test_indices = tests[split_ndx]

    assert set(retain_indices) & set(forget_indices) == set()
    assert set(retain_indices) & set(test_indices) == set()
    assert set(forget_indices) & set(test_indices) == set()
    assert set(val_indices) & set(test_indices) == set()
    assert set(retain_indices) & set(val_indices) == set()
    assert set(forget_indices) & set(val_indices) == set()
    assert set(test_indices) & set(val_indices) == set()
    return retain_indices, forget_indices, val_indices, test_indices
