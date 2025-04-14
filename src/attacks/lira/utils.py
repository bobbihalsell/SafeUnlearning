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
from datasets.cifar10 import get_cifar10_test_transform


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
    config: DictConfig,
    indices: List[Array],
) -> Dict[str, DataLoader]:
    dataset = config.dataset.name
    save_dir = config.dataset.save_path
    transform = get_cifar10_test_transform()

    train, val, test = load_train_val_test_datasets(dataset, 1, config.dataset.val_ratio, save_dir, save_dir, transform)
    dataset = ConcatDataset([train, val, test])

    retain_indices, forget_indices, val_indices = indices
    loaders = {}
    for split_name, indices in zip(
        ["retain", "forget", "val"], [retain_indices, forget_indices, val_indices]
    ):
        data_set = Subset(dataset, indices)
        loader = DataLoader(
            data_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=False,
        )
        loaders[split_name] = loader

    return loaders


def load_model(model_name: str, num_classes: int, checkpoint_path: Path) -> nn.Module:
    if hasattr(torchvision.models, model_name):
        model = torchvision.models.get_model(model_name, weights=None)

        # Adjust the last layer based on model type
        if hasattr(model, "fc"):  # ResNet-style
            model.fc = nn.Linear(
                model.fc.in_features,
                num_classes
            )
        elif hasattr(model, "classifier"):
            # MobileNet, EfficientNet, VGG, DenseNet
            if isinstance(model.classifier, nn.Sequential):
                # Handle cases where classifier is Sequential
                last_layer_idx = len(model.classifier) - 1
                model.classifier[last_layer_idx] = nn.Linear(
                    model.classifier[last_layer_idx].in_features,
                    num_classes
                )
            else:
                model.classifier = nn.Linear(
                    model.classifier.in_features,
                    num_classes
                )
        else:
            raise AttributeError("Unknown classification layer "
                                 f'for {model_name}')

    else:
        print(f'Could not find {model_name} in torchvision.'
              ' Looking in timm.')
        try:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
            )
        except Exception:
            raise AttributeError(f"{model_name} not found.")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model
