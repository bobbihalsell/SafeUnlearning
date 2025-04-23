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
    lira_path: Path, split_ndx: int, forget_ndx: int
) -> Tuple[Array, Array, Array]:
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
    # dataset_name = config.dataset.name
    # save_dir = config.dataset.save_path
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

def get_loaders_from_indices2(
    config: DictConfig,
    indices: List[Array],
) -> Dict[str, DataLoader]:
    dataset_name = config.dataset.name
    save_dir = config.dataset.save_path
    train_transform = DATASETS_TO_TRAIN_TRANSFORM[dataset_name]()
    test_transform = DATASETS_TO_TRANSFORM[dataset_name]()

    train, _, test = load_train_val_test_datasets(
        dataset_name, 1, 0, save_dir, "", train_transform
    )
    dataset_aug = ConcatDataset([train, test])

    train, _, test = load_train_val_test_datasets(
        dataset_name, 1, 0, save_dir, "", test_transform
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
            batch_size=config.dataset.cfg.batch_sizes[split_name],
            shuffle=shuffle,
            num_workers=config.dataset.cfg.num_workers,
            pin_memory=False,
        )
        loaders[split_name] = loader

    return loaders


# def load_model(model_name: str, num_classes: int, checkpoint_path: Path
#                ) -> nn.Module:
#     if hasattr(torchvision.models, model_name):
#         model = torchvision.models.get_model(model_name, weights=None)

#         # Adjust the last layer based on model type
#         if hasattr(model, "fc"):  # ResNet-style
#             model.fc = nn.Linear(model.fc.in_features, num_classes)
#         elif hasattr(model, "classifier"):
#             # MobileNet, EfficientNet, VGG, DenseNet
#             if isinstance(model.classifier, nn.Sequential):
#                 # Handle cases where classifier is Sequential
#                 last_layer_idx = len(model.classifier) - 1
#                 model.classifier[last_layer_idx] = nn.Linear(
#                     model.classifier[last_layer_idx].in_features, num_classes
#                 )
#             else:
#                 model.classifier = nn.Linear(model.classifier.in_features, 
#                                              num_classes)
#         else:
#             raise AttributeError(
#                 f"Unknown classification layer for {model_name}"
#                 )

#     else:
#         print(
#             f"Could not find {model_name} in torchvision. Looking in timm."
#             )
#         try:
#             model = timm.create_model(
#                 model_name,
#                 pretrained=False,
#                 num_classes=num_classes,
#             )
#         except Exception:
#             raise AttributeError(f"{model_name} not found.")

#     checkpoint = torch.load(checkpoint_path)["model_state_dict"]
#     model.load_state_dict(checkpoint)
#     return model