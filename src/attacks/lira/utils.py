from pathlib import Path
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader


def get_retain_forget_val_indices(
    lira_path: Path, split_ndx: int, forget_ndx: int
):
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
    root: Path,
    indices: List,
    config: Dict,
) -> List[DataLoader]:
    dataset_root = root / "datasets"
    dataset_name = dataset_cfg.name
    non_augmented_dataset, _ = get_dataset_and_lengths(
        dataset_root, dataset_name, get_train_transform(dataset_name)
    )
    augmented_dataset, _ = get_dataset_and_lengths(
        dataset_root, dataset_name, get_test_transform(dataset_name)
    )

    retain_indices, forget_indices, val_indices = indices
    loaders = []
    for split_name, indices in zip(["retain", "forget", "val"], [retain_indices, forget_indices, val_indices]):
        data_set = get_dataset_based_on_split_state(
            non_augmented_dataset,
            augmented_dataset,
            getattr(unlearner_cfg.loaders, split_name).state,
            indices,
        )
        loader = DataLoader(
            data_set,
            batch_size=unlearner_cfg.batch_size,
            shuffle=getattr(unlearner_cfg.loaders, split_name).shuffle,
            num_workers=get_num_workers_from_shuffle(
                getattr(unlearner_cfg.loaders, split_name).shuffle
            ),
            pin_memory=False,
        )
        loaders.append(loader)

    return loaders
